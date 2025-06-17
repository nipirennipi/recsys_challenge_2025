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
    logger.info("Aggregating daily counts for each event type")
    daily_counts = (
        all_data.groupby(["sku", "date", "event_type"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={
            "product_buy": "daily_product_buy", 
            "add_to_cart": "daily_add_to_cart", 
            "remove_from_cart": "daily_remove_from_cart"
        })
        .reset_index()
    )

    # Ensure all event columns exist
    for col in [
        "daily_product_buy", 
        "daily_add_to_cart", 
        "daily_remove_from_cart"
    ]:
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
    ).fillna({
        "daily_product_buy": 0, 
        "daily_add_to_cart": 0, 
        "daily_remove_from_cart": 0
    })

    # Sort by date for rolling window calculations
    full_data = full_data.sort_values(["sku", "date"])

    # Calculate cumulative sums        
    logger.info("Calculating cumulative sums for daily events")
    for col, daily_col in zip(
        [
            "product_buy_popularity_all", 
            "add_to_cart_popularity_all", 
            "remove_from_cart_popularity_all"
        ], 
        [
            "daily_product_buy", 
            "daily_add_to_cart", 
            "daily_remove_from_cart"
        ]
    ):
        full_data[col] = full_data.groupby("sku")[daily_col].cumsum()

    # Calculate rolling window features
    logger.info("Calculating rolling window features")
    product_buy_popularity_cols = ["product_buy_popularity_all"]
    add_to_cart_popularity_cols = ["add_to_cart_popularity_all"]
    remove_from_cart_popularity_cols = ["remove_from_cart_popularity_all"]
    
    for n in num_days:
        for event, cols in [
            ("product_buy", product_buy_popularity_cols),
            ("add_to_cart", add_to_cart_popularity_cols),
            ("remove_from_cart", remove_from_cart_popularity_cols),
        ]:
            col_name = f"{event}_popularity_{n}d"
            rolling_col = f"daily_{event}"
            full_data[col_name] = (
                full_data.groupby("sku")[rolling_col]
                .rolling(window=n, min_periods=1)
                .sum()
                .astype(int)
                .reset_index(drop=True)
            )
            cols.append(col_name)

    # Generate feature vectors for product_buy_popularity
    logger.info("Generating feature vectors for product_buy_popularity")
    full_data["product_buy_popularity"] = full_data[product_buy_popularity_cols].values.tolist()
    full_data["product_buy_popularity"] = full_data["product_buy_popularity"].apply(
        lambda x: np.array(x, dtype=np.int32)
    )
    # Generate feature vectors for add_to_cart_popularity
    logger.info("Generating feature vectors for add_to_cart_popularity")
    full_data["add_to_cart_popularity"] = full_data[add_to_cart_popularity_cols].values.tolist()
    full_data["add_to_cart_popularity"] = full_data["add_to_cart_popularity"].apply(
        lambda x: np.array(x, dtype=np.int32)
    )
    # Generate feature vectors for remove_from_cart_popularity
    logger.info("Generating feature vectors for remove_from_cart_popularity")
    full_data["remove_from_cart_popularity"] = full_data[remove_from_cart_popularity_cols].values.tolist()
    full_data["remove_from_cart_popularity"] = full_data["remove_from_cart_popularity"].apply(
        lambda x: np.array(x, dtype=np.int32)
    )

    return full_data[[
        "sku", 
        "date", 
        "product_buy_popularity", 
        "add_to_cart_popularity", 
        "remove_from_cart_popularity"
    ]]


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

