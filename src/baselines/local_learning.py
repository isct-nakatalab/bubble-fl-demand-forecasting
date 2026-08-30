import logging

import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.data_loader import load_client_datasets
from src.data.dataset import RegressionDataset
from src.metrics import compute_metrics
from src.models import SalesSequenceTransformer
from src.utils import build_final_result_rows, parse_args, set_seed, write_csv_rows

logger = logging.getLogger(__name__)
formatter = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=formatter)


def train_local(args):
    set_seed(args.seed)
    data_path = args.data_path
    batch_size = args.batch_size
    num_epochs = args.local_epoch
    device = args.device
    datasets, _, active_clients = load_client_datasets(
        data_path,
        target_item=args.target_item,
        window_size=args.window_size,
        train_size=args.train_size,
        clients=args.clients,
    )
    active_clients = set(active_clients)
    rows = []

    logger.info("Epoch  |  Loss  |  R2")

    for client_id, data in datasets.items():
        if client_id not in active_clients:
            continue
        logger.info(f"Client {client_id}")
        train_features = data["train_features"]
        train_labels = data["train_labels"]
        test_features = data["test_features"]
        test_labels = data["test_labels"]
        target_scaler = data.get("target_scaler")
        input_dim = train_features.shape[-1]

        train_dataset = RegressionDataset(train_features, train_labels)
        test_dataset = RegressionDataset(test_features, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = SalesSequenceTransformer(
            input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(num_epochs):
            model.train()
            batch_losses = []
            for batch_inputs, batch_targets in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_inputs.to(device))
                outputs = outputs.float()
                batch_targets = batch_targets.float().to(device)
                if outputs.ndim == 1:
                    outputs = outputs.unsqueeze(-1)
                if batch_targets.ndim == 1:
                    batch_targets = batch_targets.unsqueeze(-1)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_loss = np.mean(batch_losses)

            metrics = compute_metrics(
                model,
                test_loader,
                target_scaler=target_scaler,
            )
            rows.append(
                {
                    "client": client_id,
                    "epoch": epoch + 1,
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "r_squared": float(metrics["r_squared"]),
                    "train_loss": float(epoch_loss),
                    "test_loss": float(metrics["loss"]),
                }
            )

            if (epoch + 1) % 2 == 0:
                logger.info(
                    f"{epoch + 1}/{num_epochs}  |  {epoch_loss:.4f} / {metrics['loss']:.4f}  |  {metrics['r_squared']:.4f}"
                )

    if rows:
        final_rows = build_final_result_rows(
            rows,
            method="local",
            target_item=args.target_item,
            members_by_client={int(row["client"]): [int(row["client"])] for row in rows},
        )
        write_csv_rows(args.out_path, list(final_rows[0].keys()), final_rows)


if __name__ == "__main__":
    args = parse_args("local")
    train_local(args)
