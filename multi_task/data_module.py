import numpy as np
import pandas as pd
import pytorch_lightning as pl
import logging
import torch

from typing import Dict, Tuple, List, Set, Iterator, Sized
from datetime import datetime
from torch.utils.data import DataLoader, Sampler

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
    BATCH_SIZE,
    GROUP_SIZE
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
        self.item_features_dim: int = 0
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
            self.train_sampler = ChunkedShuffleSampler(
                dataset_size=len(self.train_data),
                chunk_size=BATCH_SIZE * GROUP_SIZE,
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
            self.validation_sampler = ChunkedShuffleSampler(
                dataset_size=len(self.validation_data),
                chunk_size=BATCH_SIZE * GROUP_SIZE,
            )
            
            # Release memory
            self.properties_dict.clear()
            logger.info("Released memory for properties_dict")
            self.item_features_dict.clear()
            logger.info("Released memory for item_features_dim")
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
        
        # properties["price"] = (properties["price"] - PRICE_MIN_VALUE) / (PRICE_MAX_VALUE - PRICE_MIN_VALUE)
        # properties["price"] = properties["price"].apply(lambda x: np.clip(x, 0, 1).astype(np.float32))
        properties["price"] = properties["price"] + 1 # 0 for padding

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
            self.train_data, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            sampler=self.train_sampler,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=self.validation_sampler,
        )

class ChunkedShuffleSampler(Sampler[int]):
    """
    Chunked random sampler.
    It first randomly shuffles the order of chunks, and then randomly shuffles 
    the order of samples within each chunk. This can significantly improve IO 
    efficiency when processing large datasets that need to be read from disk in 
    chunks, while maintaining good randomness.

    Args:
        dataset_size (int): Number of sample in Dataset object.
        chunk_size (int): The number of samples contained in each chunk.
        seed (int, optional): Seed for reproducible randomization.
    """

    def __init__(self, dataset_size: int, chunk_size: int):
        super().__init__()
        self.dataset_size = dataset_size
        self.chunk_size = chunk_size
        self.num_chunks = (self.dataset_size + self.chunk_size - 1) // self.chunk_size

    def __len__(self) -> int:
        return self.dataset_size

    def __iter__(self) -> Iterator[int]:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        g = torch.Generator()
        g.manual_seed(seed)

        chunk_indices = torch.randperm(self.num_chunks, generator=g).tolist()

        for chunk_idx in chunk_indices:
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.dataset_size)
            samples_in_chunk = torch.arange(start_idx, end_idx, dtype=torch.int64)
            shuffled_indices_in_chunk = samples_in_chunk[torch.randperm(len(samples_in_chunk), generator=g)]

            yield from shuffled_indices_in_chunk.tolist()
