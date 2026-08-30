import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def save_processed_datasets(
    datasets: dict[int, dict[str, Any]],
    output_path: str | Path,
    *,
    client_map: dict[int, str] | None = None,
    active_clients: list[int] | None = None,
    dataset_meta: dict[str, Any] | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    client_tables = []
    client_table_meta = {}

    for client_id, raw_data in datasets.items():
        table = raw_data["table"].copy()
        table.insert(0, "client", int(client_id))
        client_tables.append(table)
        client_table_meta[str(int(client_id))] = {
            "item_columns": list(raw_data["item_columns"]),
            "time_feature_columns": list(raw_data["time_feature_columns"]),
            "date_column": raw_data.get("date_column", "date"),
        }

    combined = pd.concat(client_tables, axis=0, ignore_index=True) if client_tables else pd.DataFrame()
    combined.to_csv(path, index=False)

    metadata = {
        "storage_format": "client_table_csv",
        "client_column": "client",
        "client_map": {str(k): str(v) for k, v in (client_map or {}).items()},
        "active_clients": [int(x) for x in (active_clients or [])],
        "dataset_meta": dataset_meta or {},
        "client_table_meta": client_table_meta,
    }
    metadata_path = path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def encode_time_position_features(
    dates,
    *,
    include_week_of_month: bool = False,
    include_week_of_year: bool = False,
    include_year: bool = False,
) -> pd.DataFrame:
    ts = pd.to_datetime(pd.Series(dates), errors="coerce")
    features = []

    if include_week_of_month:
        week_of_month = pd.Categorical(ts.dt.day // 7, categories=[0, 1, 2, 3, 4])
        features.append(
            pd.get_dummies(
                week_of_month,
                prefix="week",
                dtype="float32",
                drop_first=True,
            ).reset_index(drop=True)
        )

    if include_week_of_year:
        week_of_year = ts.dt.isocalendar().week.astype("float32")
        features.append(
            pd.DataFrame(
                {
                    "week_of_year_sin": np.sin(2 * np.pi * week_of_year / 52.0),
                    "week_of_year_cos": np.cos(2 * np.pi * week_of_year / 52.0),
                }
            ).reset_index(drop=True)
        )

    if include_year:
        year_index = (ts.dt.year - ts.dt.year.min()).astype("float32")
        features.append(pd.DataFrame({"year_index": year_index}).reset_index(drop=True))

    if not features:
        return pd.DataFrame(index=ts.index)
    return pd.concat(features, axis=1)


def make_sequence_tensors(features, labels, timestamps, window_size):
    """Convert tabular samples into chronological rolling windows."""

    ts = pd.to_datetime(pd.Series(timestamps), errors="coerce")
    order = np.argsort(ts.to_numpy(dtype="datetime64[ns]"))

    if isinstance(features, torch.Tensor):
        feat = features.detach().cpu().numpy()
    else:
        feat = np.asarray(features)

    if isinstance(labels, torch.Tensor):
        target = labels.detach().cpu().numpy()
    else:
        target = np.asarray(labels)

    if target.ndim == 1:
        target = target.reshape(-1)
        target_dim = None
    else:
        target = target.reshape(target.shape[0], -1)
        target_dim = target.shape[1]

    feat = feat[order]
    target = target[order]
    ts = ts.iloc[order].reset_index(drop=True)

    n_samples = len(feat)
    n_features = feat.shape[1] if n_samples > 0 else 0

    if n_samples <= window_size:
        empty_x = torch.empty((0, window_size, n_features), dtype=torch.float32)
        empty_y_shape = (0,) if target_dim is None else (0, target_dim)
        empty_y = torch.empty(empty_y_shape, dtype=torch.float32)
        return empty_x, empty_y, ts.iloc[0:0]

    seq_x = []
    seq_y = []
    seq_ts = []
    for idx in range(window_size, n_samples):
        seq_x.append(feat[idx - window_size : idx])
        seq_y.append(target[idx])
        seq_ts.append(ts.iloc[idx])

    return (
        torch.tensor(np.asarray(seq_x), dtype=torch.float32),
        torch.tensor(np.asarray(seq_y), dtype=torch.float32),
        pd.Series(seq_ts).reset_index(drop=True),
    )
