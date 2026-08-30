import flwr as fl
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data.data_loader import load_client_datasets
from src.data.dataset import RegressionDataset
from src.metrics import compute_metrics
from src.models import SalesSequenceTransformer
from src.utils import set_seed


def run_client(args, result_rows=None):
    set_seed(args.seed)

    datasets, _, _ = load_client_datasets(
        args.data_path,
        target_item=args.target_item,
        window_size=args.window_size,
        train_size=args.train_size,
        clients=args.clients,
    )
    client_id = args.clients[0]
    data = datasets[client_id]

    train_features = data["train_features"]
    train_labels = data["train_labels"]
    test_features = data["test_features"]
    test_labels = data["test_labels"]
    target_scaler = data.get("target_scaler")

    train_dataset = RegressionDataset(train_features, train_labels)
    test_dataset = RegressionDataset(test_features, test_labels)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = train_features.shape[-1]
    device = args.device
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
    round_state = {"round": 0, "train_loss": None}

    class PACFLClient(fl.client.NumPyClient):
        def get_parameters(self, config=None):
            return [p.detach().numpy().astype("float32") for p in model.parameters()]

        def fit(self, parameters, config):
            round_state["round"] += 1
            for param, new_param in zip(model.parameters(), parameters):
                param.data.copy_(torch.from_numpy(new_param).float())

            batch_losses = []
            for _ in range(args.fed_epoch):
                permutation = torch.randperm(train_features.size(0))
                for i in range(0, train_features.size(0), args.batch_size):
                    idx = permutation[i : i + args.batch_size]
                    optimizer.zero_grad()
                    outputs = model(train_features[idx].to(device)).float()
                    batch_targets = train_labels[idx].float().to(device)
                    if outputs.ndim == 1:
                        outputs = outputs.unsqueeze(-1)
                    if batch_targets.ndim == 1:
                        batch_targets = batch_targets.unsqueeze(-1)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())

            round_state["train_loss"] = float(np.mean(batch_losses)) if batch_losses else 0.0

            return self.get_parameters(), len(train_dataset), {}

        def evaluate(self, parameters, config=None):
            for param, new_param in zip(model.parameters(), parameters):
                param.data.copy_(torch.from_numpy(new_param).float())

            metrics = compute_metrics(model, testloader, target_scaler=target_scaler)
            if result_rows is not None:
                result_rows.append(
                    {
                        "client": client_id,
                        "epoch": round_state["round"],
                        "train_loss": float(round_state["train_loss"])
                        if round_state["train_loss"] is not None
                        else 0.0,
                        "test_loss": float(metrics["loss"]),
                        "mae": float(metrics["mae"]),
                        "rmse": float(metrics["rmse"]),
                        "r_squared": float(metrics["r_squared"]),
                    }
                )
            return metrics["loss"], len(test_dataset), metrics

    fl.client.start_client(server_address=args.server_address, client=PACFLClient().to_client())
