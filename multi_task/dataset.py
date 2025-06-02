import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import logging
import pickle
from typing import Dict, Tuple, List, Set
from torch.utils.data import Dataset
from datetime import datetime

from data_utils.data_dir import DataDir
from tqdm import tqdm
from multi_task.target_calculators import (
    TargetCalculator,
)
from multi_task.preprocess_data import (
    IdMapper,
)
from multi_task.constants import (
    EventTypes,
    EVENT_TYPE_TO_COLUMNS,
    ENTITY_COLUMN_NAME,
    PAD_VALUE_SKU,
    PAD_VALUE_CATEGORY,
    PAD_VALUE_PRICE,
    PAD_VALUE_NAME,
    PAD_VALUE_EVENT_TYPE,
    PAD_VALUE_TIME_FEAT,
    PAD_VALUE_URL,
    PAD_VALUE_QUERY,
    MAX_SEQUENCE_LENGTH,
    QUERY_MIN_VALUE,
    QUERY_MAX_VALUE,
    BATCH_SIZE,
    GROUP_SIZE,
)
from multi_task.utils import (
    parse_to_array,
)


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class BehavioralDataset(Dataset):
    """
    Dataset containing client embeddings, and target
    calculator that computes targets for each client.
    """

    def __init__(
        self,
        data_dir: DataDir,
        id_mapper: IdMapper,
        target_df: pd.DataFrame,
        target_calculators: List[TargetCalculator],
        properties_dict: Dict[int, Dict[str, object]],
        item_stat_feat_dict: Dict[int, Dict[datetime, np.ndarray]],
        item_stat_feat_dim: int,
        mode: str = "train",
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.id_mapper = id_mapper
        self.target_df = target_df
        self.target_calculators = target_calculators
        self.properties_dict = properties_dict
        self.item_stat_feat_dict = item_stat_feat_dict
        self.item_stat_feat_dim = item_stat_feat_dim
        self.PAD_VALUE_STAT_FEAT = np.zeros(
            self.item_stat_feat_dim, 
            dtype=np.float32
        )
        self.mode = mode
        
        self.user_features_dict: Dict[int, np.ndarray] = {}
        self.user_features_dim: int
        
        save_dir = self.data_dir._target_dir
        if self.mode == "train":
            self.ids_file = save_dir / "train_ids.txt"
            self.sequence_file = save_dir / "train_sequence.parquet"
        else:
            self.ids_file = save_dir / "relevant_ids.txt"
            self.sequence_file = save_dir / "relevant_sequence.parquet"

        self.chunk_idx: int = -1
        self.chunk_size: int = BATCH_SIZE * GROUP_SIZE
        self.behavior_sequence_chunk: List = []
        self.client_ids: Set[int] = set()
        self._behavior_sequence()
        self._load_user_features_dict()
        
        # Release memory
        self.properties_dict.clear()
        logger.info("Released memory for properties_dict")
        self.item_stat_feat_dict.clear()
        logger.info("Released memory for item_stat_feat_dict")

    def _load_user_features_dict(self) -> None:
        """
        Load user features from the user_features.parquet file.
        Returns a dictionary with client as the key to features as the value.
        """
        if self.mode == "train":
            user_features = pd.read_parquet(self.data_dir.input_dir / "user_features.parquet")
        else:
            user_features = pd.read_parquet(self.data_dir.data_dir / "user_features.parquet")
            
        # Normalize user features
        for col in user_features.columns:
            # Apply normalization for columns
            if any(key in col for key in ["_days_since_", "_time_diff_"]):
                max_value = user_features[col].max()
                user_features[col] = (max_value - user_features[col]) / max_value 
            
            # Apply np.log1p and min-max scaling to columns
            elif any(key in col for key in [
                "_count_", "_price_tier_", "_unique_price_tiers", "_unique_categories"
            ]):
                user_features[col] = np.log1p(user_features[col], dtype=np.float32)
                min_value = user_features[col].min()
                max_value = user_features[col].max()
                user_features[col] = (user_features[col] - min_value) / (max_value - min_value)
            
            # Apply min-max scaling and fillna with -1 for columns
            elif any(key in col for key in [
                "_price_mean", "_price_median", "_price_std", "_price_max", "_price_min"
            ]):
                min_value = user_features[col].min()
                max_value = user_features[col].max()
                user_features[col] = (user_features[col] - min_value) / (max_value - min_value)
                user_features[col] = user_features[col].fillna(-1).astype(np.float32)  
        
        # Defragmentation
        user_features = user_features.copy()
        
        # Concatenate all columns except "client_id" into a single np.array
        feature_cols = [col for col in user_features.columns if col != "client_id"]
        feature_array = user_features[feature_cols].to_numpy(dtype=np.float32)
        user_features["features"] = list(feature_array)
        
        self.user_features_dict = user_features.set_index("client_id")["features"].to_dict()    
        self.user_features_dim = len(next(iter(self.user_features_dict.values())))
        logger.info(f"Loaded statistic features for {len(self.user_features_dict)} users")
        logger.info(f"User statistic features dimension: {self.user_features_dim}")

    def _load_client_ids(self) -> bool:
        if not self.ids_file.exists():
            return False
        
        with open(self.ids_file, "r") as f:
            for line in f:
                self.client_ids.add(int(line.strip()))
        return True

    def _stream_behavior_sequence(self, idx: int) -> None:
        required_chunk_idx = idx // self.chunk_size
        
        if self.chunk_idx != required_chunk_idx or len(self.behavior_sequence_chunk) == 0:            
            parquet_file = pq.ParquetFile(self.sequence_file)
            table = parquet_file.read_row_group(required_chunk_idx)
            self.behavior_sequence_chunk = table.to_pandas().to_dict("records")
            self.chunk_idx = required_chunk_idx

    def _save_client_ids(self) -> None:
        save_dir = self.data_dir._target_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(self.ids_file, "w") as f:
            for uid in self.client_ids:
                f.write(f"{uid}\n")
        
    def _save_behavior_sequence(self, behavior_sequence: List) -> None:
        save_dir = self.data_dir._target_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(behavior_sequence).to_parquet(
            self.sequence_file, 
            row_group_size=self.chunk_size,
        )
    
    def _behavior_sequence(self) -> None:
        """
        Construct the user's historical behavior sequence, including two types 
        of events: ADD_TO_CART and PRODUCT_BUY.
        """
        if self._load_client_ids():
            logger.info("Behavior sequence already loaded, skipping construction.")
            logger.info(f"Stream load behavior sequence for {len(self.client_ids)} clients")
            return
        
        logger.info(f"Constructing {self.mode} user behavior sequence")
        if self.mode != "train":
            self.client_ids = set(self.target_df["client_id"])
        all_events = []

        for event_type in EventTypes:
            events = self._load_events(event_type=event_type)
            if event_type == EventTypes.SEARCH_QUERY:
                events["query"] = events["query"].apply(parse_to_array)
                events["query"] = (events["query"] - QUERY_MIN_VALUE) / (QUERY_MAX_VALUE - QUERY_MIN_VALUE)
                events["query"] = events["query"].apply(lambda x: np.clip(x, 0, 1).astype(np.float32))
                
            if self.mode != "train":
                events = events[events["client_id"].isin(self.client_ids)]
            events = events.rename(
                columns={EVENT_TYPE_TO_COLUMNS[event_type]: ENTITY_COLUMN_NAME}
            )
            events["event_type"] = event_type.get_index()
            all_events.append(events)

        all_events_df = pd.concat(all_events)
        # all_events_df = all_events_df.sample(frac=1).head(300000)
        logger.info(f"Sampled all_events_df length: {len(all_events_df)}")
        all_events_df["timestamp"] = pd.to_datetime(all_events_df["timestamp"])
        all_events_df = all_events_df.sort_values(by=["client_id", "timestamp"])

        behavior_sequence_entity = [
            {
                "client_id": client_id,
                "sequence": list(zip(
                    group['timestamp'], 
                    group[ENTITY_COLUMN_NAME], 
                    group['event_type'],
                ))
            }
            for client_id, group in tqdm(
                all_events_df.groupby("client_id"), desc="Processing clients"
            )
        ]
        
        self.client_ids = {client_behavior["client_id"] for client_behavior in behavior_sequence_entity}
        logger.info(f"Behavior sequence constructed for {len(self.client_ids)} clients")
        behavior_sequence = []
        
        for client_behavior in tqdm(behavior_sequence_entity, desc="Processing behavior sequences"):
            client_id = client_behavior["client_id"]
            sequence = client_behavior["sequence"]
            sequence_sku_info = []
            sequence_url_info = []
            sequence_query_info = []
            for timestamp, entity, event_type in sequence:
                if event_type in [
                    EventTypes.ADD_TO_CART.get_index(), 
                    EventTypes.PRODUCT_BUY.get_index(),
                    EventTypes.REMOVE_FROM_CART.get_index(),
                ]:
                    sku = int(entity)
                    properties = self.properties_dict[sku]
                    datetime = timestamp.floor('D')
                    stat_feat = self.item_stat_feat_dict[sku][datetime]
                    sequence_sku_info.append({
                        "sku": sku,
                        "category": properties["category"],
                        "price": properties["price"],
                        "name": properties["name"],
                        "event_type": event_type,
                        "timestamp": timestamp,
                        "features": stat_feat,
                    })
                if event_type == EventTypes.PAGE_VISIT.get_index():
                    url = int(entity)
                    sequence_url_info.append({
                        "url": url,
                        "event_type": event_type,
                        "timestamp": timestamp,
                    })
                if event_type == EventTypes.SEARCH_QUERY.get_index():
                    query = entity
                    sequence_query_info.append({
                        "query": query,
                        "event_type": event_type,
                        "timestamp": timestamp,
                    })
            
            behavior_sequence.append({
                "client_id": client_id,
                "sequence_sku": sequence_sku_info,
                "sequence_url": sequence_url_info,
                "sequence_query": sequence_query_info,
            })
        
        self._save_client_ids()
        self._save_behavior_sequence(behavior_sequence)
        logger.info(f"Behavior sequence saved to {self.data_dir._target_dir}")
    
    def _pad_sequence(self, sequence: np.ndarray, pad_value: object) -> np.ndarray:
        """
        Pads a sequence to the specified maximum length with a given pad value.
        Args:
            sequence (np.ndarray): The input sequence to be padded.
            pad_value (object): The value used for padding.
        """
        if len(sequence) >= MAX_SEQUENCE_LENGTH:
            return sequence[-MAX_SEQUENCE_LENGTH:]
        
        full_shape = (MAX_SEQUENCE_LENGTH,)  + sequence.shape[1:]
        padded_sequence = np.full(full_shape, pad_value, dtype=sequence.dtype)
        length = min(len(sequence), MAX_SEQUENCE_LENGTH)
        padded_sequence[:length] = sequence[:length]
        return padded_sequence
    
    def _load_events(self, event_type: EventTypes) -> pd.DataFrame:
        file_dir = self.data_dir.input_dir if self.mode == "train" else self.data_dir.data_dir
        return pd.read_parquet(
            file_dir / f"{event_type.value}.parquet"
        )

    def _augment_sequence(self, sequence: List[dict], pad_value: dict, max_length: int) -> List[dict]:
        """
        Augment a sequence by masking or cropping.

        Args:
            sequence (List[dict]): Input sequence.
            pad_value (dict): Padding value for the sequence.
            max_length (int): Maximum sequence length.

        Returns:
            List[dict]: Augmented sequence.
        """
        if len(sequence) <= 1:
            return sequence

        if np.random.rand() < 0.5:  # Masking
            num_mask = max(1, int(len(sequence) * 0.3))
            mask_indices = np.random.choice(len(sequence), num_mask, replace=False)
            for idx in mask_indices:
                sequence[idx] = pad_value
        else:  # Cropping
            crop_len = max(1, int(len(sequence) * 0.6))
            start_idx = np.random.randint(0, len(sequence) - crop_len + 1)
            sequence = sequence[start_idx:start_idx + crop_len]

        return sequence[:max_length]

    def _generate_timestamp_features(self, sequence_sku_info: List[dict]) -> None:
        total_events = len(sequence_sku_info)
        if total_events == 0:
            return
        
        for idx, item in enumerate(sequence_sku_info):
            timestamp = item["timestamp"]
            if timestamp is None:
                logger.info(f"sequence_sku_info: {sequence_sku_info}")
                continue
            hour_of_day = timestamp.hour  # Hour of the day (0-23)
            hour_sin = np.sin(2 * np.pi * hour_of_day / 24.)
            hour_cos = np.cos(2 * np.pi * hour_of_day / 24.)
            day_of_week = timestamp.weekday()  # Day of the week (0-6)
            day_sin = np.sin(2 * np.pi * day_of_week / 7.)
            day_cos = np.cos(2 * np.pi * day_of_week / 7.)
            is_weekend = 1 if day_of_week >= 5 else 0  # Is weekend (0/1)
            
            # Add event position in the sequence
            # event_position = (idx + 1) / total_events  # Position in the sequence (0-1)
            
            time_feat = [hour_sin, hour_cos, day_sin, day_cos, is_weekend]
            sequence_sku_info[idx]["time_feat"] = np.array(time_feat, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.client_ids)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        behavior_data = self._getitem(idx, is_augmentation=False)
        if self.mode == "train":
            return (
                behavior_data[0],
                self._getitem(idx, is_augmentation=True),
                self._getitem(idx, is_augmentation=True),
                behavior_data[1],  # target
            )
        return behavior_data

    def _getitem(self, idx, is_augmentation=False) -> tuple[np.ndarray, np.ndarray]:
        self._stream_behavior_sequence(idx)
        chunk_inner_idx = idx % self.chunk_size
        client_id = self.behavior_sequence_chunk[chunk_inner_idx]["client_id"]
        if not is_augmentation and self.mode == "train":
            target = [
                target_calculator.compute_target(
                    client_id=client_id, target_df=self.target_df
                )
                for target_calculator in self.target_calculators
            ]

        # Get the behavior sequence for the client
        sequence_sku_info = self.behavior_sequence_chunk[chunk_inner_idx]["sequence_sku"].copy()
        sequence_url_info = self.behavior_sequence_chunk[chunk_inner_idx]["sequence_url"].copy()
        sequence_query_info = self.behavior_sequence_chunk[chunk_inner_idx]["sequence_query"].copy()

        # Generate timestamp-related features for sequence_sku_info
        self._generate_timestamp_features(sequence_sku_info)

        if is_augmentation:
            pad_sku = {"sku": PAD_VALUE_SKU, "category": PAD_VALUE_CATEGORY, "price": PAD_VALUE_PRICE,
                       "name": PAD_VALUE_NAME, "event_type": PAD_VALUE_EVENT_TYPE, "timestamp": None, 
                       "features": self.PAD_VALUE_STAT_FEAT, "time_feat": PAD_VALUE_TIME_FEAT}
            pad_url = {"url": PAD_VALUE_URL, "event_type": PAD_VALUE_EVENT_TYPE, "timestamp": None}
            pad_query = {"query": PAD_VALUE_QUERY, "event_type": PAD_VALUE_EVENT_TYPE, "timestamp": None}

            sequence_sku_info = self._augment_sequence(sequence_sku_info, pad_sku, MAX_SEQUENCE_LENGTH)
            sequence_url_info = self._augment_sequence(sequence_url_info, pad_url, MAX_SEQUENCE_LENGTH)
            sequence_query_info = self._augment_sequence(sequence_query_info, pad_query, MAX_SEQUENCE_LENGTH)

        sequence_sku_length = max(min(len(sequence_sku_info), MAX_SEQUENCE_LENGTH), 1)
        sequence_url_length = max(min(len(sequence_url_info), MAX_SEQUENCE_LENGTH), 1)
        sequence_query_length = max(min(len(sequence_query_info), MAX_SEQUENCE_LENGTH), 1)

        # Convert sequence_info to a structured format (e.g., numpy arrays)
        sequence_sku = np.array([item["sku"] for item in sequence_sku_info], dtype=np.int64)
        sequence_category = np.array([item["category"] for item in sequence_sku_info], dtype=np.int64)
        sequence_price = np.array([item["price"] for item in sequence_sku_info], dtype=np.float32)
        sequence_event_type = np.array([item["event_type"] for item in sequence_sku_info], dtype=np.int64)
        sequence_url = np.array([item["url"] for item in sequence_url_info], dtype=np.int64)    
        
        if len(sequence_sku_info) > 0:
            sequence_name = np.stack(
                [item["name"] for item in sequence_sku_info],
                axis=0,
                dtype=np.float32,
            )
            sequence_stat_feat = np.stack(
                [item["features"] for item in sequence_sku_info],
                axis=0,
                dtype=np.float32,
            )
            sequence_time_feat = np.stack(
                [item["time_feat"] for item in sequence_sku_info],
                axis=0,
                dtype=np.float32,
            )
        else:
            sequence_name = np.expand_dims(PAD_VALUE_NAME, axis=0)
            sequence_stat_feat = np.expand_dims(self.PAD_VALUE_STAT_FEAT, axis=0)
            sequence_time_feat = np.expand_dims(PAD_VALUE_TIME_FEAT, axis=0)
        
        if len(sequence_query_info) > 0:
            sequence_query = np.stack(
                [query["query"] for query in sequence_query_info],
                axis=0,
                dtype=np.float32,
            )
        else:
            sequence_query = np.expand_dims(PAD_VALUE_QUERY, axis=0)
        
        # Mapping ids
        sequence_sku = np.array(
            [self.id_mapper.get_sku_id(sku) for sku in sequence_sku], 
            dtype=np.int64,
        )
        sequence_category = np.array(
            [self.id_mapper.get_category_id(category) for category in sequence_category], 
            dtype=np.int64,
        )
        sequence_url = np.array(
            [self.id_mapper.get_url_id(url) for url in sequence_url], 
            dtype=np.int64,
        )
        
        # # Normalize price, sequence_name and sequence_query
        # sequence_price = (sequence_price - PRICE_MIN_VALUE) / (PRICE_MAX_VALUE - PRICE_MIN_VALUE)
        # sequence_price = np.clip(sequence_price, -1, 1, dtype=np.float32)
        # sequence_name = (sequence_name - NAME_MIN_VALUE) / (NAME_MAX_VALUE - NAME_MIN_VALUE)
        # sequence_name = np.clip(sequence_name, -1, 1, dtype=np.float32)
        # sequence_query = (sequence_query - QUERY_MIN_VALUE) / (QUERY_MAX_VALUE - QUERY_MIN_VALUE)
        # sequence_query = np.clip(sequence_query, -1, 1, dtype=np.float32)
        
        # Padding sequences
        sequence_sku = self._pad_sequence(sequence_sku, PAD_VALUE_SKU)
        sequence_category = self._pad_sequence(sequence_category, PAD_VALUE_CATEGORY)
        sequence_price = self._pad_sequence(sequence_price, PAD_VALUE_PRICE)
        sequence_name = self._pad_sequence(sequence_name, PAD_VALUE_NAME)
        # logger.info(f"sequence_stat_feat.shape: {sequence_stat_feat.shape}")
        # logger.info(f"PAD_VALUE_ITEM_FEATURES.shape: {self.PAD_VALUE_ITEM_FEATURES.shape}")
        sequence_stat_feat = self._pad_sequence(sequence_stat_feat, self.PAD_VALUE_STAT_FEAT)
        sequence_event_type = self._pad_sequence(sequence_event_type, PAD_VALUE_EVENT_TYPE)
        sequence_time_feat = self._pad_sequence(sequence_time_feat, PAD_VALUE_TIME_FEAT)
        sequence_url = self._pad_sequence(sequence_url, PAD_VALUE_URL)
        sequence_query = self._pad_sequence(sequence_query, PAD_VALUE_QUERY)
        # logger.info(f"client_id: {client_id}")
        # logger.info(f"sequence_sku.shape: {sequence_sku.shape}")
        # logger.info(f"sequence_category.shape: {sequence_category.shape}")
        # logger.info(f"sequence_price.shape: {sequence_price.shape}")
        # logger.info(f"sequence_name.shape: {sequence_name.shape}")
        # logger.info(f"sequence_event_type.shape: {sequence_event_type.shape}")
        # logger.info(f"sequence_url.shape: {sequence_url.shape}")
        # logger.info(f"sequence_query.shape: {sequence_query.shape}")
        # logger.info(f"sequence_sku_length: {sequence_sku_length}")
        # logger.info(f"sequence_url_length: {sequence_url_length}")
        # logger.info(f"sequence_query_length: {sequence_query_length}")

        # logger.info(f"client_id: {client_id}")
        # logger.info(f"sequence_sku.dtype: {sequence_sku.dtype}")
        # logger.info(f"sequence_category.dtype: {sequence_category.dtype}")
        # logger.info(f"sequence_price.dtype: {sequence_price.dtype}")
        # logger.info(f"sequence_name.dtype: {sequence_name.dtype}")
        # logger.info(f"sequence_event_type.dtype: {sequence_event_type.dtype}")
        # logger.info(f"sequence_url.dtype: {sequence_url.dtype}")
        # logger.info(f"sequence_query.dtype: {sequence_query.dtype}")

        # Get user features
        user_features = self.user_features_dict[client_id]

        # logger.info(f"client_id: {client_id}")
        # logger.info(f"sequence_sku: {sequence_sku}")
        # logger.info(f"sequence_category: {sequence_category}")
        # logger.info(f"sequence_price: {sequence_price}")
        # logger.info(f"sequence_event_type: {sequence_event_type}")
        # sequence_timestamp = [item["timestamp"] for item in sequence_sku_info]
        # logger.info(f"sequence_timestamp: {sequence_timestamp}")
        # logger.info(f"sequence_time_feat: {sequence_time_feat}")

        # Combine the structured data into a single array or return as a tuple
        behavior_data = (
            client_id,
            sequence_sku, 
            sequence_category, 
            sequence_price, 
            sequence_name, 
            sequence_stat_feat,
            sequence_event_type, 
            sequence_time_feat,
            sequence_url,
            sequence_query,
            sequence_sku_length, 
            sequence_url_length,
            sequence_query_length,
            user_features,
        )
        
        # if np.any(sequence_name > 1) or np.any(sequence_name < -1):
        #     logger.info(f"Out of range values found in sequence_name, max: {np.max(sequence_name)}, min: {np.min(sequence_name)}")

        # if np.any(sequence_query > 1) or np.any(sequence_query < -1):
        #     logger.info(f"Out of range values found in sequence_query, max: {np.max(sequence_query)}, min: {np.min(sequence_query)}")

        return (behavior_data, target) if not is_augmentation and self.mode == "train" else behavior_data
    
