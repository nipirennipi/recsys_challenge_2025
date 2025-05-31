import argparse
import logging
from typing import List, Tuple, Dict
from pathlib import Path
from functools import reduce
from scipy.stats import entropy
import pandas as pd
import numpy as np

from feat_engine.user_features.constants import (
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
    event_df["timestamp"] = pd.to_datetime(event_df["timestamp"])
    return event_df


def create_features(data_dir: DataDir, num_days: List[int],) -> pd.DataFrame:
    logger.info("Creating features for user")
    
    # Load data from each event file
    events = {et.value: load_events(data_dir, et) for et in EventTypes}
    all_events = pd.concat([event_df[["client_id", "timestamp"]] for event_df in events.values()])
    START_TIME = all_events["timestamp"].min()
    END_TIME = all_events["timestamp"].max()
    MAX_TIME_SPAN = (END_TIME - START_TIME).days
    client_ids = pd.DataFrame({"client_id": all_events["client_id"].unique()})
    
    # Time features: days since first/last interaction for each event type
    logger.info("Calculating days since first/last interaction for each event type")
    user_first_last = []
    for event_name, event_df in events.items():
        temp = event_df.groupby("client_id")["timestamp"].agg(["min", "max"]).reset_index()
        temp[f"{event_name}_days_since_first"] = (END_TIME - temp["min"]).dt.days
        temp[f"{event_name}_days_since_last"] = (END_TIME - temp["max"]).dt.days
        temp = temp[["client_id", f"{event_name}_days_since_first", f"{event_name}_days_since_last"]]
        temp = client_ids.merge(temp, on="client_id", how="left").fillna(MAX_TIME_SPAN)
        user_first_last.append(temp)
    
    # Time features: Purchase interval statistics
    logger.info("Calculating purchase interval statistics")
    buy_df = events["product_buy"].copy()
    buy_df["timestamp"] = pd.to_datetime(buy_df["timestamp"])
    buy_df = buy_df.sort_values(["client_id", "timestamp"])
    buy_df["time_diff"] = buy_df.groupby("client_id")["timestamp"].diff().dt.days
    time_diff_stats = buy_df.groupby("client_id")["time_diff"].agg(
        ["mean", "median", "max", "min"]
    ).reset_index().fillna(MAX_TIME_SPAN)
    time_diff_stats.columns = ["client_id", "buy_time_diff_mean", "buy_time_diff_median", 
                               "buy_time_diff_max", "buy_time_diff_min"]
    time_diff_stats = client_ids.merge(time_diff_stats, on="client_id", how="left").fillna(MAX_TIME_SPAN)

    # Statistical features: event counts in time windows    
    logger.info("Calculating event counts in time windows")
    events_window_count = []
    for event_name, event_df in events.items():
        temp = event_df[["client_id", "timestamp"]].copy()
        temp["timestamp"] = pd.to_datetime(temp["timestamp"])
        masks = {w: (END_TIME - temp["timestamp"]).dt.days <= w for w in num_days}
        masks[0] = temp["timestamp"] <= END_TIME
        
        for window, mask in masks.items():
            col = f"{event_name}_count_{window}d" if window != 0 else f"{event_name}_count_all"
            count = temp[mask].groupby("client_id").size().reset_index(name=col)
            count = client_ids.merge(count, on="client_id", how="left").fillna(0)
            events_window_count.append(count)

    properties_dict = load_properties_dict(data_dir)
    # Statistical features: price propensity
    logger.info("Calculating price propensity features")
    price_stats = create_price_propensity_features(events, client_ids, END_TIME, properties_dict)

    dfs = [client_ids] + user_first_last + [time_diff_stats] + events_window_count + price_stats
    features = reduce(lambda left, right: left.merge(right, on="client_id", how="left"), dfs)

    return features


def load_properties_dict(data_dir: DataDir) -> Dict[str, Dict[str, int]]:
    """
    Load properties from the properties file and construct a dictionary
    with sku as the key and its attributes as the value.
    """
    logger.info("Loading properties")
    properties = pd.read_parquet(data_dir.properties_file)
    
    properties_dict = (
        properties[["sku", "category", "price"]]
        .set_index("sku")
        .to_dict(orient='index')
    )
    return properties_dict


def create_price_propensity_features(
    events: Dict[str, pd.DataFrame],
    client_ids: pd.DataFrame,
    end_time: pd.Timestamp,
    properties_dict: Dict[str, Dict[str, int]]
) -> List:
    price_stats  = []
    bins = [-1, 19, 39, 59, 79, 100]  # Adjust 100 if max price bucket is 99
    labels = [0, 1, 2, 3, 4] # 5 tiers

    for event_type in ["product_buy", "add_to_cart"]:
        df = events[event_type].copy()
        df["price"] = df["sku"].map(lambda sku: properties_dict[sku]["price"])
        df["price_tier"] = pd.cut(df["price"], bins=bins, labels=labels, right=True).astype(int)

        # 1. Price Tier Statistics
        for time_window in ["30d", "all"]:
            if time_window == "30d":
                window_df = df[df["timestamp"] >= (end_time - pd.Timedelta(days=30))]
            else:
                window_df = df
            
            tier_counts = window_df.groupby(
                ["client_id", "price_tier"]
            ).size().unstack(fill_value=0)
            tier_counts.columns = [
                f"{event_type}_price_tier_{col}_{time_window}" for col in tier_counts.columns
            ]
            tier_counts = tier_counts.reset_index()
            tier_counts = client_ids.merge(tier_counts, on="client_id", how="left").fillna(0)
            price_stats.append(tier_counts)

        # 2. User Price Bucket Summary Statistics
        price_stats_df = df.groupby("client_id")["price"].agg(
            ["mean", "median", "std", "max", "min"]
        ).reset_index()
        price_stats_df.columns = ["client_id", f"{event_type}_price_mean",
            f"{event_type}_price_median", f"{event_type}_price_std",
            f"{event_type}_price_max", f"{event_type}_price_min"
        ]
        price_stats_df = client_ids.merge(price_stats_df, on="client_id", how="left")
        price_stats.append(price_stats_df)
        
        # 3. Price Diversity
        tier_distribution = df.groupby(["client_id", "price_tier"]).size().groupby("client_id").apply(
            lambda x: entropy(x / x.sum(), base=2) if x.sum() > 0 else 0
        ).reset_index(name=f"{event_type}_price_entropy")
        unique_tiers = df.groupby(
            "client_id"
        )["price_tier"].nunique().reset_index(name=f"{event_type}_unique_price_tiers")
        diversity_df = pd.merge(tier_distribution, unique_tiers, on="client_id", how="outer").fillna(0)
        diversity_df = client_ids.merge(diversity_df, on="client_id", how="left").fillna(0)
        price_stats.append(diversity_df)

    return price_stats


def save_features(features_dir: Path, features: pd.DataFrame) -> None:
    logger.info("Saving features")
    features_dir.mkdir(parents=True, exist_ok=True)
    output_file = features_dir / "user_features.parquet"
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

