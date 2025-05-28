import argparse
import logging
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from feat_engine.item_features.constants import (
    EventTypes,
)
from data_utils.data_dir import DataDir

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def load_events(data_dir: DataDir, event_type: EventTypes) -> pd.DataFrame:
    file_dir = data_dir.data_dir
    event_df = pd.read_parquet(file_dir / f"{event_type.value}.parquet")
    event_df["event_type"] = event_type.value
    return event_df[["sku", "timestamp", "event_type"]]


def create_features(data_dir: DataDir, num_days: List[int],) -> pd.DataFrame:
    logger.info("Creating features for item")
    # Load data from each event file
    events = [
        load_events(data_dir, EventTypes.PRODUCT_BUY),
        load_events(data_dir, EventTypes.ADD_TO_CART),
        load_events(data_dir, EventTypes.REMOVE_FROM_CART)
    ]
    all_data = pd.concat(events)
    all_data["date"] = pd.to_datetime(all_data["timestamp"]).dt.floor('D')

    # Aggregate daily counts for each event type
    daily_counts = (
        all_data.groupby(["sku", "date", "event_type"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={
            "product_buy": "daily_buys", 
            "add_to_cart": "daily_adds", 
            "remove_from_cart": "daily_removes"
        })
        .reset_index()
    )

    # Ensure all event columns exist
    for col in ["daily_buys", "daily_adds", "daily_removes"]:
        if col not in daily_counts:
            daily_counts[col] = 0

    # Generate a complete date range for each SKU
    sku_date_range = (
        daily_counts.groupby("sku")["date"]
        .apply(lambda x: pd.date_range(start=x.min(), end=x.max()))
        .reset_index(level=0)
        .explode("date")
        .reset_index(drop=True)
    )
    sku_date_range["date"] = pd.to_datetime(sku_date_range["date"])

    # Merge with daily counts to fill missing dates
    full_data = pd.merge(
        sku_date_range,
        daily_counts,
        on=["sku", "date"],
        how="left"
    ).fillna({"daily_buys": 0, "daily_adds": 0, "daily_removes": 0})

    # Sort by date for rolling window calculations
    full_data = full_data.sort_values(["sku", "date"])

    # Calculate cumulative sums
    cumulative_cols = ["cumulative_buys", "cumulative_adds", "cumulative_removes"]
    for col, daily_col in zip(cumulative_cols, ["daily_buys", "daily_adds", "daily_removes"]):
        full_data[col] = full_data.groupby("sku")[daily_col].cumsum()

    # Calculate rolling window features
    rolling_features = []
    for n in num_days:
        for event in ["buys", "adds", "removes"]:
            col_name = f"{n}d_{event}"
            rolling_col = f"daily_{event}"
            full_data[col_name] = (
                full_data.groupby("sku")[rolling_col]
                .rolling(window=n, min_periods=1)
                .sum()
                .astype(int)
                .reset_index(drop=True)
            )
            rolling_features.append(col_name)

    # Generate feature vectors
    feature_columns = rolling_features + cumulative_cols
    full_data["features"] = full_data[feature_columns].values.tolist()
    full_data["features"] = full_data["features"].apply(lambda x: np.array(x, dtype=np.int32))

    return full_data[["sku", "date", "features"]]


def save_features(features_dir: Path, features: pd.DataFrame) -> None:
    logger.info("Saving features")
    features_dir.mkdir(parents=True, exist_ok=True)
    output_file = features_dir / "item_features.parquet"
    features.to_parquet(output_file, index=False)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with input and target data – produced by data_utils.split_data",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory where to store generated features",
    )
    parser.add_argument(
        "--num-days",
        nargs="*",
        type=int,
        default=[1, 7, 30],
        help="Numer of days to compute features",
    )
    return parser


def main(params):
    data_dir = DataDir(Path(params.data_dir))
    features_dir = Path(params.features_dir)

    features = create_features(
        data_dir=data_dir,
        num_days=params.num_days,
    )
    save_features(
        features_dir=features_dir, 
        features=features
    )


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params=params)

