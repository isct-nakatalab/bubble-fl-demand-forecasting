import pandas as pd

from src.data.utils import encode_time_position_features, save_processed_datasets


def preprocess_dataco(
    input_path="data/SuperstoreSalesDataset.csv",
    output_path="data/processed/dataco.csv",
    client_col="Order City",
    item_col="Category Name",
    topn=50,
):
    value_col = "Sales"
    date_col = "order date (DateOrders)"
    dataset = pd.read_csv(input_path, encoding="latin-1")
    dataset[date_col] = pd.to_datetime(dataset[date_col], errors="coerce")
    dataset[value_col] = pd.to_numeric(dataset[value_col], errors="coerce").fillna(0.0)
    dataset = dataset.dropna(subset=[client_col, item_col, date_col])
    if dataset.empty:
        raise ValueError("No records found")
    weekly = (
        dataset.groupby([client_col, item_col, pd.Grouper(key=date_col, freq="W-SUN")])[value_col]
        .sum()
        .reset_index()
        .rename(columns={date_col: "date"})
    )
    top_items = (
        weekly.groupby(item_col)[value_col].sum().sort_values(ascending=False).head(topn).index.astype(str).tolist()
    )
    weekly[item_col] = weekly[item_col].astype(str)
    weekly = weekly[weekly[item_col].isin(top_items)].copy()
    all_dates = pd.date_range(weekly["date"].min(), weekly["date"].max(), freq="W-SUN")
    all_clients = sorted(weekly[client_col].astype(str).dropna().unique().tolist())
    full_index = pd.MultiIndex.from_product([all_clients, top_items, all_dates], names=[client_col, item_col, "date"])
    weekly = (
        weekly.assign(**{client_col: weekly[client_col].astype(str)})
        .set_index([client_col, item_col, "date"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
    )
    client_map = {i: client for i, client in enumerate(all_clients)}
    active_clients = list(range(len(all_clients)))
    datasets = {}
    for client_id, client_name in client_map.items():
        client_weekly = weekly[weekly[client_col] == client_name].copy()
        sales_wide = (
            client_weekly.pivot(index="date", columns=item_col, values=value_col)
            .reindex(columns=top_items, fill_value=0.0)
            .fillna(0.0)
            .sort_index()
        )
        time_features = encode_time_position_features(
            sales_wide.index,
            include_week_of_month=True,
            include_week_of_year=True,
            include_year=True,
        ).set_index(sales_wide.index)
        table = pd.concat([sales_wide, time_features], axis=1).reset_index()
        datasets[client_id] = {
            "table": table,
            "item_columns": top_items,
            "time_feature_columns": time_features.columns.tolist(),
            "date_column": "date",
        }
    if output_path:
        save_processed_datasets(
            datasets,
            output_path,
            client_map=client_map,
            active_clients=active_clients,
            dataset_meta={
                "storage_format": "client_table",
                "client_col": client_col,
                "item_col": item_col,
                "topn": topn,
            },
        )
    return datasets, client_map, active_clients


if __name__ == "__main__":
    preprocess_dataco()
