import argparse
import logging
from typing import List, Tuple
from pathlib import Path
from functools import reduce
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
    return event_df[["client_id", "timestamp", "event_type"]]


def create_features(data_dir: DataDir, num_days: List[int],) -> pd.DataFrame:
    logger.info("Creating features for user")
    
    # Load data from each event file
    events = {et.value: load_events(data_dir, et) for et in EventTypes}
    all_events = pd.concat(events.values())
    START_TIME = all_events["timestamp"].min()
    END_TIME = all_events["timestamp"].max()
    MAX_TIME_SPAN = (END_TIME - START_TIME).days
    client_ids = pd.DataFrame({"client_id": all_events["client_id"].unique()})
    
    # Time features: days since first/last interaction for each event type
    user_first_last = []
    for event_name, event_df in events.items():
        temp = event_df.groupby("client_id")["timestamp"].agg(["min", "max"]).reset_index()
        temp[f"{event_name}_days_since_first"] = (END_TIME - temp["min"]).dt.days
        temp[f"{event_name}_days_since_last"] = (END_TIME - temp["max"]).dt.days
        temp = temp[["client_id", f"{event_name}_days_since_first", f"{event_name}_days_since_last"]]
        temp = client_ids.merge(temp, on="client_id", how="left").fillna(MAX_TIME_SPAN)
        user_first_last.append(temp)
    
    # Purchase interval statistics
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
    dfs = [client_ids] + user_first_last + [time_diff_stats]
    features = reduce(lambda left, right: left.merge(right, on="client_id", how="left"), dfs)
    
    for event_name, event_df in events.items():
        temp = event_df[["client_id", "timestamp"]].copy()
        temp["timestamp"] = pd.to_datetime(temp["timestamp"])
        masks = {w: (END_TIME - temp["timestamp"]).dt.days <= w for w in num_days}
        masks[0] = temp["timestamp"] <= END_TIME
        
        for window, mask in masks.items():
            col = f"{event_name}_count_{window}d" if window != 0 else f"{event_name}_count_all"
            count = temp[mask].groupby("client_id").size().reset_index(name=col)
            features = features.merge(count, on="client_id", how="left").fillna(0)

    return features


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

