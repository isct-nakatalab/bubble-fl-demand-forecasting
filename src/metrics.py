import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(model, dataloader, target_scaler=None, return_arrays=False):
    criterion = torch.nn.MSELoss()
    model.eval()
    predictions, targets = [], []
    total_loss, num_samples = 0.0, 0

    with torch.no_grad():
        for inputs, batch_targets in dataloader:
            batch_size = inputs.size(0)
            num_samples += batch_size
            outputs = model(inputs)
            outputs = outputs.float()
            batch_targets = batch_targets.float()
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(-1)
            if batch_targets.ndim == 1:
                batch_targets = batch_targets.unsqueeze(-1)
            loss = criterion(outputs, batch_targets)
            total_loss += loss.item() * batch_size
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_targets.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    if target_scaler is not None:
        predictions = target_scaler.inverse_transform(predictions)
        targets = target_scaler.inverse_transform(targets)

    flat_predictions = predictions.reshape(-1)
    flat_targets = targets.reshape(-1)
    result = {
        "rmse": float(np.sqrt(mean_squared_error(flat_targets, flat_predictions))),
        "mae": float(mean_absolute_error(flat_targets, flat_predictions)),
        "mape": float(np.mean(np.abs((flat_targets - flat_predictions) / (flat_targets + 1e-8))) * 100),
        "mse": float(mean_squared_error(flat_targets, flat_predictions)),
        "r_squared": float(r2_score(flat_targets, flat_predictions)),
        "loss": float(total_loss / num_samples) if num_samples > 0 else 0.0,
    }
    if return_arrays:
        result["predictions"] = predictions
        result["targets"] = targets
    return result
