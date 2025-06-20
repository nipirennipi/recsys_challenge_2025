import pandas as pd
import logging
from pathlib import Path
from dataclasses import dataclass


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

@dataclass(frozen=True)
class TargetData:
    """
    Dataclass for storing data for training and validation.
    """

    train_df: pd.DataFrame
    relevant_df: pd.DataFrame

    @classmethod
    def read_from_dir(cls, target_dir: Path, is_online: bool = False):
        if is_online:
            train_df = pd.read_parquet(target_dir / "train_target_online.parquet")
            logger.info("Reading online target data")
        else:
            train_df = pd.read_parquet(target_dir / "train_target.parquet")
            logger.info("Reading offline target data")
        relevant_df = pd.read_parquet(target_dir / "relevant_target.parquet")
        return cls(train_df, relevant_df)
