import pandas as pd

from src.data.utils import encode_time_position_features, save_processed_datasets


def preprocess_m5(
    sales_path="data/m5-forecasting-accuracy/sales_train_validation.csv",
    calendar_path="data/m5-forecasting-accuracy/calendar.csv",
    output_path="data/processed/m5.csv",
    client_col="store_id",
    item_col="item_id",
    topn=50,
):
    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    daily_dates = pd.to_datetime(calendar.set_index("d").loc[day_cols, "date"]).to_numpy()
    sales[client_col] = sales[client_col].astype(str)
    sales[item_col] = sales[item_col].astype(str)
    if sales.empty:
        raise ValueError("No records found")
    item_totals = sales.groupby(item_col)[day_cols].sum().sum(axis=1)
    top_items = item_totals.sort_values(ascending=False).head(topn).index.astype(str).tolist()
    client_names = sorted(sales[client_col].dropna().unique().tolist())
    client_map = {i: client for i, client in enumerate(client_names)}
    active_clients = list(range(len(client_names)))
    datasets = {}
    for client_id, client_name in client_map.items():
        sales_client = sales[sales[client_col] == client_name].copy()
        sales_items = sales_client[sales_client[item_col].isin(top_items)].copy()
        if sales_items.empty:
            continue
        sales_daily = sales_items.groupby(item_col)[day_cols].sum().T
        sales_daily.index = daily_dates
        sales_daily = sales_daily.reindex(columns=top_items, fill_value=0.0)
        weekly_sales = sales_daily.resample("W-SUN").sum().sort_index()
        weekly_sales.columns.name = None
        state = str(sales_client["state_id"].iloc[0])
        snap_col = f"snap_{state}"
        calendar_weekly = (
            calendar.set_index("date")
            .assign(
                event_1_flag=calendar["event_name_1"].notna().astype("float32"),
                event_2_flag=calendar["event_name_2"].notna().astype("float32"),
            )
            .resample("W-SUN")
            .agg(
                {
                    snap_col: "max",
                    "event_1_flag": "max",
                    "event_2_flag": "max",
                }
            )
            .sort_index()
        )
        calendar_weekly = calendar_weekly.reindex(weekly_sales.index)
        time_features = pd.concat(
            [
                calendar_weekly,
                encode_time_position_features(
                    weekly_sales.index,
                    include_week_of_month=True,
                    include_week_of_year=True,
                    include_year=True,
                ).set_index(weekly_sales.index),
            ],
            axis=1,
        ).fillna(0.0)
        table = (
            pd.concat([weekly_sales, time_features], axis=1, join="inner")
            .reset_index()
            .rename(columns={"index": "date"})
        )
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
    preprocess_m5()
