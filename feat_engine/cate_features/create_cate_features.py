import argparse
import logging
from typing import List, Tuple, Dict
from pathlib import Path
import pandas as pd
import numpy as np

from feat_engine.cate_features.constants import (
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


def load_properties(data_dir: DataDir) -> pd.DataFrame:
    """
    Load properties from the properties file and construct a dictionary
    with sku as the key and its attributes as the value.
    """
    logger.info("Loading properties")
    properties = pd.read_parquet(data_dir.properties_file)
    
    # # Normalize price, name
    # properties["name"] = properties["name"].apply(parse_to_array)
    # properties["name"] = (properties["name"] - NAME_MIN_VALUE) / (NAME_MAX_VALUE - NAME_MIN_VALUE)
    # properties["name"] = properties["name"].apply(lambda x: np.clip(x, 0, 1).astype(np.float32))
    
    # properties["price"] = (properties["price"] - PRICE_MIN_VALUE) / (PRICE_MAX_VALUE - PRICE_MIN_VALUE)
    # properties["price"] = properties["price"].apply(lambda x: np.clip(x, 0, 1).astype(np.float32))
    properties["price"] = properties["price"] + 1 # 0 for padding
    
    return properties


def create_features(
    data_dir: DataDir, 
    # num_days: List[int],
) -> pd.DataFrame:
    logger.info("Creating features for category")
       
    # Load properties
    properties = load_properties(data_dir)
    
    # Calculate average price per category
    logger.info("Calculating average price per category")
    category_avg_price = properties.groupby('category')['price'].mean().reset_index()
    category_avg_price.rename(columns={'price': 'avg_price_per_category'}, inplace=True)
    
    return category_avg_price[[
        "category", 
        "avg_price_per_category", 
    ]]


def save_features(features_dir: Path, features: pd.DataFrame) -> None:
    logger.info("Saving features")
    features_dir.mkdir(parents=True, exist_ok=True)
    output_file = features_dir / "cate_features.parquet"
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
    # parser.add_argument(
    #     "--num-days",
    #     nargs="*",
    #     type=int,
    #     default=[1, 7, 30],
    #     help="Numer of days to compute features",
    # )
    return parser


def main(params):
    data_dir = DataDir(Path(params.data_dir))
    features_dir = Path(params.features_dir)

    features = create_features(
        data_dir=data_dir,
        # num_days=params.num_days,
    )
    save_features(
        features_dir=features_dir, 
        features=features
    )


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params=params)

