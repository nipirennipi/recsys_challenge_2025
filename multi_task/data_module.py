import numpy as np
import pytorch_lightning as pl
import logging

from typing import List
from torch.utils.data import DataLoader

from data_utils.data_dir import DataDir
from multi_task.dataset import (
    BehavioralDataset,
)
from multi_task.target_data import (
    TargetData,
)
from multi_task.target_calculators import (
    TargetCalculator,
)
from multi_task.preprocess_data import (
    IdMapper,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class BehavioralDataModule(pl.LightningDataModule):
    """
    DataModule containing two BehavioralDatasets, one for
    training and one for validation.
    """

    def __init__(
        self,
        data_dir: DataDir,
        id_mapper: IdMapper,
        target_data: TargetData,
        target_calculators: List[TargetCalculator],
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.id_mapper = id_mapper
        self.target_data = target_data
        self.target_calculators = target_calculators

    def setup(self, stage) -> None:
        if stage == "fit":

            logger.info("Constructing datasets")

            self.train_data = BehavioralDataset(
                data_dir=self.data_dir,
                id_mapper=self.id_mapper,
                target_df=self.target_data.train_df,
                target_calculators=self.target_calculators,
                mode="train",
            )

            self.validation_data = BehavioralDataset(
                data_dir=self.data_dir,
                id_mapper=self.id_mapper,
                target_df=self.target_data.relevant_df,
                target_calculators=self.target_calculators,
                mode="validation",
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
