import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.utils import make_sequence_tensors


def load_processed_payload(input_path: str | Path) -> dict[str, Any]:
    metadata_path = Path(input_path).with_suffix(".json")
    table = pd.read_csv(input_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    client_column = metadata["client_column"]
    client_table_meta = metadata["client_table_meta"]
    datasets = {}
    for client_id, raw_data in client_table_meta.items():
        client_table = (
            table.loc[pd.to_numeric(table[client_column], errors="coerce") == int(client_id)]
            .drop(columns=[client_column])
            .reset_index(drop=True)
        )
        datasets[int(client_id)] = {
            "table": client_table,
            "item_columns": list(raw_data["item_columns"]),
            "time_feature_columns": list(raw_data["time_feature_columns"]),
            "date_column": raw_data.get("date_column", "date"),
        }

    return {
        "datasets": datasets,
        "client_map": {int(k): v for k, v in metadata.get("client_map", {}).items()},
        "active_clients": [int(x) for x in metadata.get("active_clients", [])],
        "dataset_meta": metadata.get("dataset_meta", {}),
    }


def build_client_sequence_dataset(
    table: pd.DataFrame,
    *,
    target_item: str,
    item_columns: list[str],
    time_feature_columns: list[str],
    window_size: int,
    train_size: float,
    date_column: str = "date",
) -> dict[str, Any] | None:
    if not 0.0 < float(train_size) < 1.0:
        raise ValueError(f"train_size must be between 0 and 1, got: {train_size}")
    if target_item not in item_columns:
        return None

    weekly = table.copy()
    weekly[date_column] = pd.to_datetime(weekly[date_column], errors="coerce")
    weekly = weekly.sort_values(date_column, kind="stable").reset_index(drop=True)
    weekly[item_columns] = weekly[item_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weekly[time_feature_columns] = weekly[time_feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    n_total = len(weekly)
    if n_total == 0:
        return None

    n_train = int(np.floor(n_total * train_size))
    n_train = min(max(1, n_train), max(1, n_total - 1))
    if n_train <= window_size:
        return None

    train_raw = weekly.iloc[:n_train].copy().reset_index(drop=True)
    test_raw = weekly.iloc[max(0, n_train - window_size) :].copy().reset_index(drop=True)

    scaler = StandardScaler()
    train_sales = train_raw[[target_item]].to_numpy(dtype="float32")
    test_sales = test_raw[[target_item]].to_numpy(dtype="float32")
    scaler.fit(train_sales)

    train_features = pd.concat(
        [
            pd.DataFrame(scaler.transform(train_sales), columns=[target_item]),
            train_raw[time_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    test_features = pd.concat(
        [
            pd.DataFrame(scaler.transform(test_sales), columns=[target_item]),
            test_raw[time_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    train_labels = pd.DataFrame(scaler.transform(train_sales), columns=[target_item])
    test_labels = pd.DataFrame(scaler.transform(test_sales), columns=[target_item])

    train_x, train_y, train_ts = make_sequence_tensors(
        train_features,
        train_labels,
        train_raw[date_column],
        window_size,
    )
    test_x, test_y, test_ts = make_sequence_tensors(
        test_features,
        test_labels,
        test_raw[date_column],
        window_size,
    )
    if train_x.numel() == 0 or test_x.numel() == 0:
        return None

    return {
        "train_features": train_x,
        "train_labels": train_y,
        "test_features": test_x,
        "test_labels": test_y,
        "train_timestamps": train_ts,
        "test_timestamps": test_ts,
        "feature_scaler": scaler,
        "target_scaler": scaler,
        "num_features": int(train_x.shape[-1]),
        "item": target_item,
    }


def load_client_datasets(
    input_path: str | Path,
    *,
    target_item: str | None = None,
    window_size: int | None = None,
    train_size: float | None = None,
    clients: list[int] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, str], list[int]]:
    payload = load_processed_payload(input_path)
    datasets = payload["datasets"]
    client_map = payload["client_map"]
    active_clients = payload["active_clients"]
    if clients is not None:
        allowed = {int(x) for x in clients}
        client_map = {cid: name for cid, name in client_map.items() if cid in active_clients and cid in allowed}
        active_clients = [cid for cid in active_clients if cid in client_map]
        datasets = {cid: data for cid, data in datasets.items() if cid in allowed}

    if not datasets:
        return datasets, client_map, active_clients

    if target_item is None or window_size is None or train_size is None:
        raise ValueError("target_item, window_size, and train_size are required for client_table data")

    built = {}
    for client_id, raw_data in datasets.items():
        data = build_client_sequence_dataset(
            raw_data["table"],
            target_item=target_item,
            item_columns=list(raw_data["item_columns"]),
            time_feature_columns=list(raw_data["time_feature_columns"]),
            window_size=window_size,
            train_size=train_size,
            date_column=raw_data.get("date_column", "date"),
        )
        if data is not None:
            built[int(client_id)] = data

    active_clients = [cid for cid in active_clients if cid in built]
    client_map = {cid: client_map[cid] for cid in active_clients}
    return built, client_map, active_clients
