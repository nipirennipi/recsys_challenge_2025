import numpy as np
import pandas as pd
import pytorch_lightning as pl
import logging

from typing import Dict, Tuple, List, Set
from datetime import datetime
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
from multi_task.constants import (
    NAME_MIN_VALUE,
    NAME_MAX_VALUE,
    PRICE_MIN_VALUE,
    PRICE_MAX_VALUE,
)
from multi_task.utils import (
    parse_to_array,
)
from multi_task.gpu_allocator import (
    GPUAllocator,
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
        gpu_allocator: GPUAllocator,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.id_mapper = id_mapper
        self.target_data = target_data
        self.target_calculators = target_calculators
        self.gpu_allocator = gpu_allocator
        
        self.properties_dict: Dict[int, Dict[str, object]] = {}
        self.item_features_dict: Dict[int, Dict[datetime, np.ndarray]] = {}
        self.item_features_dim: int
        self._load_properties_dict()
        self._load_item_features_dict()

    def setup(self, stage) -> None:
        if stage == "fit":

            logger.info("Constructing training dataset")
            self.train_data = BehavioralDataset(
                data_dir=self.data_dir,
                id_mapper=self.id_mapper,
                target_df=self.target_data.train_df,
                target_calculators=self.target_calculators,
                properties_dict=self.properties_dict,
                item_stat_feat_dict=self.item_features_dict,
                item_stat_feat_dim=self.item_features_dim,
                mode="train",
            )

            logger.info("Constructing validation dataset")
            self.validation_data = BehavioralDataset(
                data_dir=self.data_dir,
                id_mapper=self.id_mapper,
                target_df=self.target_data.relevant_df,
                target_calculators=self.target_calculators,
                properties_dict=self.properties_dict,
                item_stat_feat_dict=self.item_features_dict,
                item_stat_feat_dim=self.item_features_dim,
                mode="validation",
            )
            
            self.gpu_allocator.release_gpu_memory()

    def _load_properties_dict(self) -> None:
        """
        Load properties from the properties file and construct a dictionary
        with sku as the key and its attributes as the value.
        """
        logger.info("Loading properties")
        properties = pd.read_parquet(self.data_dir.properties_file)
        
        # Normalize price, name
        properties["name"] = properties["name"].apply(parse_to_array)
        properties["name"] = (properties["name"] - NAME_MIN_VALUE) / (NAME_MAX_VALUE - NAME_MIN_VALUE)
        properties["name"] = properties["name"].apply(lambda x: np.clip(x, 0, 1).astype(np.float32))
        
        properties["price"] = (properties["price"] - PRICE_MIN_VALUE) / (PRICE_MAX_VALUE - PRICE_MIN_VALUE)
        properties["price"] = properties["price"].apply(lambda x: np.clip(x, 0, 1).astype(np.float32))

        self.properties_dict = (
            properties[["sku", "category", "price", "name"]]
            .set_index("sku")
            .to_dict(orient='index')
        )

    def _load_item_features_dict(self) -> None:
        """
        Load item features from the item_features.parquet file.
        Returns a dictionary with sku as the key and a dictionary of datetime to features as the value.
        """
        logger.info("Loading item statistic features")
        item_features = pd.read_parquet(self.data_dir.item_features_file)
        feature_array = np.stack(item_features["features"].values)
        feature_array = np.log1p(feature_array, dtype=np.float32)
        item_features["features"] = list(feature_array)
        
        self.item_features_dict = (
            item_features
            .groupby("sku")[["date", "features"]]
            .apply(lambda g: dict(zip(g["date"], g["features"])))
            .to_dict()
        )
            
        self.item_features_dim = len(next(iter(next(iter(self.item_features_dict.values())).values())))
        logger.info(f"Item statistic features dimension: {self.item_features_dim}")

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
