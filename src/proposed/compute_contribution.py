import argparse
import ast
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.data_loader import load_client_datasets
from src.data.dataset import RegressionDataset
from src.models import SalesSequenceTransformer
from src.proposed.gtg_shapley import GTGShapley
from src.proposed.server import FederatedServer
from src.proposed.trainer import ClientTrainer
from src.utils import parse_args, set_seed

logger = logging.getLogger(__name__)


def _client_weight(labels: torch.Tensor, target_scaler) -> float:
    labels_np = labels.detach().cpu().numpy().reshape(len(labels), -1)
    if target_scaler is not None:
        labels_np = target_scaler.inverse_transform(labels_np)
    flat_labels = labels_np.reshape(-1)
    return float(np.log1p(np.maximum(flat_labels, 0.0).sum()) + 1e-6)


def prepare_clients(args, datasets, active_clients) -> tuple[dict, int]:
    clients = {}
    input_dim = None

    for client_id in active_clients:
        if client_id not in datasets:
            continue

        data = datasets[client_id]
        train_features = data["train_features"].float()
        train_labels = data["train_labels"].float()
        test_features = data["test_features"].float()
        test_labels = data["test_labels"].float()
        target_scaler = data.get("target_scaler")

        train_loader = DataLoader(
            RegressionDataset(train_features, train_labels),
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            RegressionDataset(test_features, test_labels),
            batch_size=args.batch_size,
            shuffle=False,
        )

        clients[int(client_id)] = {
            "train_loader": train_loader,
            "test_loader": test_loader,
            "agg_weight": _client_weight(train_labels, target_scaler),
            "num_features": int(train_features.shape[-1]),
            "target_scaler": target_scaler,
        }

        if input_dim is None:
            input_dim = int(train_features.shape[-1])

    if input_dim is None:
        raise ValueError("No client datasets available to determine input_dim")
    return clients, input_dim


def build_runtime(args):
    set_seed(args.seed)
    args.device = torch.device("cpu")

    print("args.data_path:", args.data_path)
    print("args.target_item:", args.target_item)
    print("args.clients:", args.clients)
    print("args.train_size:", args.train_size)
    print("args.window_size:", args.window_size)

    datasets, _, active_clients = load_client_datasets(
        args.data_path,
        target_item=args.target_item,
        window_size=args.window_size,
        train_size=args.train_size,
        clients=args.clients,
    )

    print("num datasets:", len(datasets))
    print("datasets keys sample:", list(datasets.keys())[:10])
    print("num active_clients:", len(active_clients))
    print("active_clients sample:", active_clients[:10])

    clients, input_dim = prepare_clients(args, datasets, active_clients)

    print("num prepared clients:", len(clients))
    print("input_dim:", input_dim)

    def model_factory(dim: int) -> SalesSequenceTransformer:
        return SalesSequenceTransformer(
            dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        )

    trainer = ClientTrainer(args, input_dim, model_factory=model_factory)

    all_local_models = {}

    print("Start local training", flush=True)

    for idx, (cid, client) in enumerate(clients.items(), start=1):
        print(f"[local train] {idx}/{len(clients)} client={cid} start", flush=True)

        all_local_models[cid] = trainer.train(
            client["train_loader"],
            args.local_epoch,
            args.lr,
            None,
        )

        print(f"[local train] {idx}/{len(clients)} client={cid} done", flush=True)

    print("Finished local training", flush=True)

    server = FederatedServer(args, clients, trainer)

    return trainer, server, clients, all_local_models


def load_final_bubbles(bubble_csv: str, target_name: str) -> pd.DataFrame:
    df = pd.read_csv(bubble_csv)
    df["bubble"] = df["bubble"].apply(ast.literal_eval)

    df = df[df["name"] == target_name].copy()

    if df.empty:
        raise ValueError(f"name='{target_name}' is not found in {bubble_csv}")

    df["step"] = df.groupby(["client", "name"]).cumcount()

    final_bubbles = df.sort_values(["client", "name", "step"]).groupby(["client", "name"]).tail(1).copy()

    return final_bubbles


def compute_gtg_for_bubble(
    target_id: int,
    members: list[int],
    trainer,
    server,
    clients,
    all_local_models,
    fed_rounds: int,
    phase: str,
    eps: float,
    seed: int,
) -> dict[int, float]:

    players = [m for m in members if m != target_id]

    if len(players) == 0:
        return {}

    baseline_model = all_local_models[target_id].to("cpu")

    baseline_rmse = trainer.evaluate(
        baseline_model,
        clients[target_id],
        phase=phase,
    )[1]

    initial_utility = -float(baseline_rmse)

    def metric_fn(subset):
        fedavg_members = [target_id] + list(subset)

        result = server.run_fedavg(
            fedavg_members,
            fed_rounds,
        )

        if isinstance(result, tuple):
            model = result[0]
        else:
            model = result

        model = model.to("cpu")

        rmse = trainer.evaluate(
            model,
            clients[target_id],
            phase=phase,
        )[1]

        return -float(rmse)

    gtg = GTGShapley(
        players=players,
        metric_fn=metric_fn,
        initial_metric=initial_utility,
        eps=eps,
        seed=seed,
    )

    scores = gtg.compute()

    # negative contribution は0にする
    scores = {int(k): max(0.0, float(v)) for k, v in scores.items()}

    # 正規化
    total = sum(scores.values())

    if total > 0:
        scores = {k: v / total for k, v in scores.items()}
    else:
        n = len(scores)
        scores = {k: 1.0 / n for k in scores}

    return scores


def main():
    # 追加引数を先に読む
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--bubble_csv", required=True)
    extra_parser.add_argument("--contribution_name", default=None)
    extra_parser.add_argument("--contribution_out_path", default=None)
    extra_parser.add_argument("--contribution_phase", default="test")
    extra_parser.add_argument("--contribution_eps", type=float, default=1e-3)

    extra_args, remaining_argv = extra_parser.parse_known_args()

    # parse_args("proposed") に追加引数を渡さないようにする
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining_argv

    args = parse_args("proposed")

    if extra_args.contribution_name is None:
        extra_args.contribution_name = args.target_item

    sys.argv = original_argv

    trainer, server, clients, all_local_models = build_runtime(args)

    final_bubbles = load_final_bubbles(
        bubble_csv=extra_args.bubble_csv,
        target_name=extra_args.contribution_name,
    )

    rows = []

    for _, row in final_bubbles.iterrows():
        target_id = int(row["client"])
        members = [int(m) for m in row["bubble"]]

        logger.info(
            "Computing contribution: name=%s target=%s members=%s",
            extra_args.contribution_name,
            target_id,
            members,
        )

        scores = compute_gtg_for_bubble(
            target_id=target_id,
            members=members,
            trainer=trainer,
            server=server,
            clients=clients,
            all_local_models=all_local_models,
            fed_rounds=args.fed_rounds,
            phase=extra_args.contribution_phase,
            eps=extra_args.contribution_eps,
            seed=args.seed + target_id,
        )

        for member_id, contribution in scores.items():
            rows.append(
                {
                    "name": extra_args.contribution_name,
                    "target_client": target_id,
                    "member_client": member_id,
                    "final_bubble": members,
                    "contribution": contribution,
                }
            )

    result_df = pd.DataFrame(rows)

    if extra_args.contribution_out_path is None:
        safe_name = extra_args.contribution_name.replace("/", "_")
        out_path = Path("results") / f"gtg_contribution_{safe_name}.csv"
    else:
        out_path = Path(extra_args.contribution_out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path, index=False)

    print(f"Saved contribution results to: {out_path}")


if __name__ == "__main__":
    main()
