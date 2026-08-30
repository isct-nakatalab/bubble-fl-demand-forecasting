from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import compute_metrics


class ClientTrainer:
    def __init__(
        self,
        args,
        input_dim: int,
        model_factory: Callable[[int], nn.Module],
    ):
        self.input_dim = int(input_dim)
        self.model_factory = model_factory
        self.device = args.device
        self.weight_decay = float(args.weight_decay)
        self.criterion = nn.MSELoss()

    def create_model(self) -> nn.Module:
        return self.model_factory(self.input_dim).to(self.device)

    @staticmethod
    def _align(outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = outputs.float()
        targets = targets.float()
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(-1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        return outputs, targets

    def train(
        self,
        loader: DataLoader,
        epochs: int,
        lr: float,
        initial_state: dict[str, torch.Tensor] | None,
        return_epoch_losses: bool = False,
    ) -> nn.Module | tuple[nn.Module, list[float]]:
        model = self.create_model()
        if initial_state is not None:
            model.load_state_dict(initial_state)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=self.weight_decay,
        )

        model.train()
        epoch_losses = []
        total_epochs = int(epochs)
        for _ in range(total_epochs):
            total_loss = 0.0
            num_samples = 0
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.float().to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs, targets = self._align(outputs, targets)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size

            avg_epoch_loss = total_loss / num_samples if num_samples > 0 else 0.0
            epoch_losses.append(avg_epoch_loss)

        if return_epoch_losses:
            return model, epoch_losses
        return model

    def evaluate(
        self,
        model: nn.Module,
        client_data: dict[str, object],
        phase: str,
    ) -> tuple[float, float, float, float]:
        if phase == "train":
            loader_key = "train_loader"
        elif phase in ("valid", "val"):
            loader_key = "val_loader"
        else:
            loader_key = "test_loader"

        metrics = compute_metrics(
            model.to(self.device),
            client_data[loader_key],
            target_scaler=client_data.get("target_scaler"),
        )
        return metrics["mae"], metrics["rmse"], metrics["r_squared"], metrics["loss"]

    def predict_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
        target_scaler,
    ) -> tuple[np.ndarray, np.ndarray]:
        model = model.to(self.device)
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs.to(self.device))
                outputs, targets = self._align(outputs.cpu(), targets)
                preds.append(outputs.numpy())
                trues.append(targets.numpy())
        predictions = np.concatenate(preds, axis=0)
        targets = np.concatenate(trues, axis=0)
        if target_scaler is not None:
            predictions = target_scaler.inverse_transform(predictions)
            targets = target_scaler.inverse_transform(targets)
        return predictions, targets
