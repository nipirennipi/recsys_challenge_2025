import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetData:
    """
    Dataclass for storing data for training and validation.
    """

    train_df: pd.DataFrame
    relevant_df: pd.DataFrame

    @classmethod
    def read_from_dir(cls, target_dir: Path):
        train_df = pd.read_parquet(target_dir / "train_target.parquet")
        relevant_df = pd.read_parquet(target_dir / "relevant_target.parquet")
        return cls(train_df, relevant_df)
