import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.data_loader import load_client_datasets
from src.data.dataset import RegressionDataset
from src.models import SalesSequenceTransformer
from src.proposed.server import FederatedServer
from src.proposed.strategies import StrategyProvider
from src.proposed.trainer import ClientTrainer
from src.utils import parse_args, set_seed, write_csv_rows

logger = logging.getLogger(__name__)


def _client_weight(labels: torch.Tensor, target_scaler) -> float:
    labels_np = labels.detach().cpu().numpy().reshape(len(labels), -1)
    if target_scaler is not None:
        labels_np = target_scaler.inverse_transform(labels_np)
    flat_labels = labels_np.reshape(-1)
    return float(np.log1p(np.maximum(flat_labels, 0.0).sum()) + 1e-6)


def prepare_clients(args, datasets, active_clients) -> tuple[dict, int]:
    clients: dict = {}
    input_dim: int | None = None
    skipped_missing = 0

    for client_id in active_clients:
        if client_id not in datasets:
            skipped_missing += 1
            continue

        data = datasets[client_id]
        train_features = data["train_features"].float()
        train_labels = data["train_labels"].float()
        test_features = data["test_features"].float()
        test_labels = data["test_labels"].float()
        target_scaler = data.get("target_scaler")

        train_loader = DataLoader(
            RegressionDataset(train_features, train_labels), batch_size=args.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            RegressionDataset(test_features, test_labels), batch_size=args.batch_size, shuffle=False
        )

        clients[int(client_id)] = {
            "train_loader": train_loader,
            "val_loader": test_loader,
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


def model_factory(dim: int) -> SalesSequenceTransformer:
    return SalesSequenceTransformer(
        dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )


def main(args):
    set_seed(args.seed)

    datasets, _, active_clients = load_client_datasets(
        args.data_path,
        target_item=args.target_item,
        window_size=args.window_size,
        train_size=args.train_size,
        clients=args.clients,
    )
    clients, input_dim = prepare_clients(args, datasets, active_clients)

    trainer = ClientTrainer(args, input_dim, model_factory=model_factory)

    all_local_models = {}
    for cid, client in clients.items():
        all_local_models[cid] = trainer.train(client["train_loader"], args.local_epoch, args.lr, None)

    server = FederatedServer(args, clients, trainer)
    provider = StrategyProvider(
        args,
        trainer=trainer,
        server=server,
        clients=clients,
        all_local_models=all_local_models,
    )

    rows = []
    for cid in sorted(clients):
        set_seed(args.seed + cid)
        stage_results = provider.execute(cid)
        for stage_name, stage_result in stage_results.items():
            members = stage_result["members"]
            model = stage_result["model"]
            train_loss = trainer.evaluate(model, clients[cid], phase="train")[3]
            mae, test_rmse, r2, test_loss = trainer.evaluate(model, clients[cid], phase="test")
            rows.append(
                {
                    "client_id": cid,
                    "method": stage_name,
                    "target_item": args.target_item,
                    "members": members,
                    "n_members": len(members),
                    "mae": mae,
                    "rmse": test_rmse,
                    "r_squared": r2,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                }
            )

    if rows:
        write_csv_rows(args.out_path, list(rows[0].keys()), rows)
        logger.info(f"Saved results to {args.out_path} ({len(rows)} rows)")
    if provider.bubble_rows:
        write_csv_rows(args.out_path_bubble, list(provider.bubble_rows[0].keys()), provider.bubble_rows)
        logger.info(f"Saved bubble results to {args.out_path_bubble} ({len(provider.bubble_rows)} rows)")


if __name__ == "__main__":
    args = parse_args("proposed")
    main(args)
