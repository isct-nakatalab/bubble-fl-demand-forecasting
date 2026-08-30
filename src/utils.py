import argparse
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)
formatter = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=formatter)


def load_item(data_dir: str | Path, dataset: str, item_number: int):
    with (Path(data_dir) / f"{dataset}.json").open() as f:
        items = json.load(f)
    items = items["client_table_meta"]["0"]["item_columns"]
    return items[item_number]


def set_flwr_log():
    class SuppressFlowerDeprecatedFilter(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            return "DEPRECATED FEATURE:" not in message

    flower_logger = logging.getLogger("flwr")
    flower_logger.setLevel(logging.INFO)
    flower_logger.addFilter(SuppressFlowerDeprecatedFilter())
    for handler in flower_logger.handlers:
        handler.setLevel(logging.INFO)
        handler.addFilter(SuppressFlowerDeprecatedFilter())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def write_csv_rows(out_path: str | Path, fieldnames: list[str], rows):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_final_result_rows(rows, *, method: str, target_item: str, members_by_client=None):
    latest_by_client: dict[int, dict[str, Any]] = {}
    for row in rows:
        client_id = int(row["client"])
        prev = latest_by_client.get(client_id)
        if prev is None or int(row["epoch"]) > int(prev["epoch"]):
            latest_by_client[client_id] = row

    final_rows = []
    for client_id in sorted(latest_by_client):
        row = latest_by_client[client_id]
        members = []
        if members_by_client is not None:
            members = list(members_by_client.get(client_id, []))
        final_rows.append(
            {
                "client_id": client_id,
                "method": method,
                "target_item": target_item,
                "members": members,
                "n_members": len(members),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "r_squared": float(row["r_squared"]),
                "train_loss": float(row["train_loss"]),
                "test_loss": float(row["test_loss"]),
            }
        )
    return final_rows


def parse_args(kind: str, argv=None):
    parser = argparse.ArgumentParser(description=f"{kind} run")
    # dir
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed dataset CSV/JSON files.",
    )
    parser.add_argument("--result_dir", type=Path, default=Path("results"), help="Directory to save results.")
    parser.add_argument("--exp_name", default=None, help="Output directory.")
    # target
    parser.add_argument("--dataset", choices=["dataco", "m5", "book"], default="m5", help="Dataset name.")
    parser.add_argument(
        "--target_item", default=None, help="Target item/series name used to build per-client datasets."
    )
    parser.add_argument("--item_number", type=int, default=0, help="Item in the list")
    parser.add_argument("--clients", default=None, help="Optional comma-separated client ID list.")
    # train param
    parser.add_argument(
        "--window_size", type=int, default=8, help="Number of historical steps used in each input sequence."
    )
    parser.add_argument(
        "--train_size", type=float, default=0.8, help="Train split ratio applied before rolling-window conversion."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for NumPy and PyTorch.")
    parser.add_argument("--device", default=None, help="Torch device such as cpu, cuda.")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size for training and evaluation.")
    parser.add_argument("--lr", type=float, default=0.00005, help="Learning rate for the optimizer.")
    parser.add_argument(
        "--local_epoch", type=int, default=50, help="Number of epochs for standalone/local pretraining."
    )
    # fedavg
    parser.add_argument(
        "--fed_epoch", type=int, default=10, help="Number of local epochs per client update in federated training."
    )
    parser.add_argument("--fed_rounds", type=int, default=1)
    # model param
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay coefficient for the optimizer.")
    parser.add_argument("--d_model", type=int, default=64, help="Transformer hidden size.")
    parser.add_argument("--nhead", type=int, default=2, help="Number of attention heads in the sequence transformer.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer encoder layers.")
    parser.add_argument(
        "--dim_feedforward", type=int, default=128, help="Hidden size of the transformer's feedforward block."
    )
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate used in the sequence transformer.")
    parser.set_defaults(model="seqtransformer")
    # Local
    if kind == "local":
        pass
    elif kind == "centralized":
        parser.add_argument("--early_stopping", action="store_true")
        parser.add_argument("--patience", type=int, default=5)
    # FedAvg
    elif kind == "fedavg":
        set_flwr_log()
        parser.add_argument("--address", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8080)
    # FedProx
    elif kind == "fedprox":
        set_flwr_log()
        parser.add_argument("--address", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8081)
        parser.add_argument("--mu", type=float, default=0.01)
    # IFCA
    elif kind == "ifca":
        set_flwr_log()
        parser.add_argument("--address", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8082)
        parser.add_argument("--n_clusters", type=int, default=3)
    # PACFL
    elif kind == "pacfl":
        set_flwr_log()
        parser.add_argument("--address", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8090)
        parser.add_argument("--epsilon", type=float, default=10.0)
        parser.add_argument("--n_clusters", type=int, default=None)
    # Proposed
    elif kind == "proposed":
        parser.description = "Direct FL adaptive/proposal runner"
        parser.add_argument("--adaptive_iter", type=int, default=1)
        parser.add_argument("--regen_iter", type=int, default=1)
        parser.add_argument("--bubble_eval_dataset", choices=["train", "valid", "test"], default="valid")
        parser.add_argument(
            "--adaptive_mode", choices=["target_on_other", "other_on_target"], default="other_on_target"
        )
        parser.add_argument("--restructure_mode", choices=["residual", "random"], default="residual")
        parser.add_argument("--min_improvement", type=float, default=0.0)
    else:
        raise ValueError(f"Unknown parse_args kind: {kind}")

    args = parser.parse_args(argv)
    args.data_path = args.data_dir / f"{args.dataset}.csv"
    args.clients = [int(x.strip()) for x in args.clients.split(",") if x.strip()] if args.clients else None

    if args.target_item is None:
        args.target_item = load_item(args.data_dir, args.dataset, args.item_number)
    logger.info(f"Target: {args.dataset}-{args.target_item}")

    if args.exp_name is None:
        if kind == "proposed":
            args.exp_name = f"{args.dataset}-{kind}-{args.adaptive_mode}-{args.restructure_mode}"
        else:
            args.exp_name = f"{args.dataset}-{kind}"

    args.out_dir = args.result_dir / args.exp_name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out_path = args.out_dir / f"{args.target_item}-{args.seed}.csv"
    args.out_path_bubble = args.out_dir / f"bubble-{args.target_item}-{args.seed}.csv"
    logger.info(f"Output to {args.out_dir}")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-selected device: {args.device}")

    if hasattr(args, "address"):
        logger.info(f"Flower Server: {args.address}:{args.port}")
    return args
