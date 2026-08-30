import argparse
import copy
import logging
from typing import TypedDict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.data_loader import load_client_datasets
from src.models import CentralTransformer
from src.utils import parse_args, set_seed, write_csv_rows

logger = logging.getLogger(__name__)
formatter = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=formatter)


class CentralizedTestSet(TypedDict):
    client_id: int
    client_index: int
    test_features: torch.Tensor
    test_labels: torch.Tensor
    target_scaler: StandardScaler | None


class CentralizedDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, client_ids: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.client_ids = client_ids
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.client_ids[idx]), self.targets[idx]


def prepare_centralized_data(args: argparse.Namespace):
    datasets, _, active_clients = load_client_datasets(
        args.data_path,
        target_item=args.target_item,
        window_size=args.window_size,
        train_size=args.train_size,
        clients=args.clients,
    )
    client_ids = [cid for cid in active_clients if cid in datasets]
    if not client_ids:
        raise ValueError(
            f"No client datasets available for target_item={args.target_item}. "
            f"Check whether the item exists in {args.data_path} and whether window_size/train_size are valid."
        )
    client_to_index = {client_id: idx for idx, client_id in enumerate(client_ids)}
    train_features_list = []
    train_client_ids_list = []
    train_labels_list = []
    test_sets: list[CentralizedTestSet] = []
    for client_id in client_ids:
        data = datasets[client_id]
        client_index = client_to_index[client_id]
        train_features_list.append(data["train_features"])
        train_client_ids_list.append(torch.full((len(data["train_features"]),), client_index, dtype=torch.long))
        train_labels_list.append(data["train_labels"])
        test_sets.append(
            {
                "client_id": client_id,
                "client_index": client_index,
                "test_features": data["test_features"],
                "test_labels": data["test_labels"],
                "target_scaler": data.get("target_scaler"),
            }
        )

    if not train_features_list:
        raise ValueError(f"No training sequences were built for target_item={args.target_item}")
    train_features = torch.cat(train_features_list, dim=0)
    train_client_ids = torch.cat(train_client_ids_list, dim=0)
    train_labels = torch.cat(train_labels_list, dim=0)
    train_loader = DataLoader(
        CentralizedDataset(train_features, train_client_ids, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
    )
    return client_ids, train_loader, test_sets, train_features.shape[-1]


def build_model(input_dim: int, num_clients: int, args: argparse.Namespace) -> CentralTransformer:
    return CentralTransformer(
        input_dim,
        num_clients=num_clients,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(args.device)


def evaluate_loader(model: nn.Module, loader: DataLoader, *, target_scaler=None, device="cpu") -> dict[str, float]:
    criterion = nn.MSELoss()
    model.eval()
    predictions = []
    targets = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for (batch_inputs, batch_client_ids), batch_targets in loader:
            batch_inputs = batch_inputs.to(device)
            batch_client_ids = batch_client_ids.to(device)
            batch_targets = batch_targets.float().to(device)
            outputs = model((batch_inputs, batch_client_ids)).float()
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(-1)
            if batch_targets.ndim == 1:
                batch_targets = batch_targets.unsqueeze(-1)
            loss = criterion(outputs, batch_targets)
            batch_size = batch_inputs.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())

    prediction_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    if target_scaler is not None:
        prediction_array = target_scaler.inverse_transform(prediction_array)
        target_array = target_scaler.inverse_transform(target_array)
    flat_predictions = prediction_array.reshape(-1)
    flat_targets = target_array.reshape(-1)
    return {
        "mae": float(mean_absolute_error(flat_targets, flat_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(flat_targets, flat_predictions))),
        "r_squared": float(r2_score(flat_targets, flat_predictions)),
        "loss": float(total_loss / num_samples) if num_samples > 0 else 0.0,
    }


def evaluate_test_sets(
    model: nn.Module, test_sets: list[CentralizedTestSet], batch_size: int, device: str
) -> tuple[float, float]:
    losses = []
    r2_scores = []
    for test_set in test_sets:
        test_client_ids = torch.full((len(test_set["test_features"]),), test_set["client_index"], dtype=torch.long)
        test_loader = DataLoader(
            CentralizedDataset(test_set["test_features"], test_client_ids, test_set["test_labels"]),
            batch_size=batch_size,
            shuffle=False,
        )
        metrics = evaluate_loader(model, test_loader, target_scaler=test_set["target_scaler"], device=device)
        losses.append(float(metrics["loss"]))
        r2_scores.append(float(metrics["r_squared"]))
    avg_loss = float(sum(losses) / len(losses)) if losses else 0.0
    avg_r2 = float(sum(r2_scores) / len(r2_scores)) if r2_scores else 0.0
    return avg_loss, avg_r2


def fit_centralized_model(
    args: argparse.Namespace,
    train_loader: DataLoader,
    test_sets: list[CentralizedTestSet],
    input_dim: int,
) -> tuple[nn.Module, float]:
    model = build_model(input_dim, len(test_sets), args)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    final_train_loss = 0.0
    best_train_loss = float("inf")
    best_test_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    logger.info("Centralized training over %s clients", len(test_sets))
    logger.info("Epoch  |  Loss  |  R2")

    for epoch in range(args.local_epoch):
        model.train()
        batch_losses = []
        for (batch_inputs, batch_client_ids), batch_targets in train_loader:
            batch_inputs = batch_inputs.to(args.device)
            batch_client_ids = batch_client_ids.to(args.device)
            batch_targets = batch_targets.float().to(args.device)
            optimizer.zero_grad()
            outputs = model((batch_inputs, batch_client_ids)).float()
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(-1)
            if batch_targets.ndim == 1:
                batch_targets = batch_targets.unsqueeze(-1)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        final_train_loss = float(sum(batch_losses) / len(batch_losses)) if batch_losses else 0.0
        test_loss, test_r2 = evaluate_test_sets(model, test_sets, args.batch_size, args.device)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_train_loss = final_train_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        logger.info("%s/%s  |  %.4f / %.4f  |  %.4f", epoch + 1, args.local_epoch, final_train_loss, test_loss, test_r2)
        if stale_epochs >= args.patience:
            logger.info("early_stop epoch=%s best_test_loss=%.4f", epoch + 1, best_test_loss)
            break
    model.load_state_dict(best_state)
    return model, best_train_loss


def train(argv=None):
    args = parse_args("centralized", argv)
    set_seed(args.seed)
    client_ids, train_loader, test_sets, input_dim = prepare_centralized_data(args)
    model, final_train_loss = fit_centralized_model(args, train_loader, test_sets, input_dim)
    members = list(client_ids)
    rows = []
    for test_set in test_sets:
        test_client_ids = torch.full((len(test_set["test_features"]),), test_set["client_index"], dtype=torch.long)
        test_loader = DataLoader(
            CentralizedDataset(test_set["test_features"], test_client_ids, test_set["test_labels"]),
            batch_size=args.batch_size,
            shuffle=False,
        )
        metrics = evaluate_loader(model, test_loader, target_scaler=test_set["target_scaler"], device=args.device)
        rows.append(
            {
                "client_id": test_set["client_id"],
                "method": "centralized",
                "target_item": args.target_item,
                "members": members,
                "n_members": len(members),
                "mae": float(metrics["mae"]),
                "rmse": float(metrics["rmse"]),
                "r_squared": float(metrics["r_squared"]),
                "train_loss": final_train_loss,
                "test_loss": float(metrics["loss"]),
            }
        )
    write_csv_rows(args.out_path, list(rows[0].keys()), rows)


if __name__ == "__main__":
    train()
