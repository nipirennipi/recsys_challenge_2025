import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import logging
import pickle
from typing import Dict, Tuple, List, Set
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

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
    PAD_VALUE_TARGET_FEAT,
    PAD_VALUE_URL,
    PAD_VALUE_QUERY,
    PAD_VALUE_TIMESTAMP,
    PAD_SKU,
    PAD_URL,
    PAD_QUERY,
    MAX_SEQUENCE_LENGTH,
    QUERY_MIN_VALUE,
    QUERY_MAX_VALUE,
    BATCH_SIZE,
    GROUP_SIZE,
)
from multi_task.utils import (
    parse_to_array,
)
from multi_task.tasks import (
    PropensityTasks,
)


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def load_propensity_targets(
    data_dir: DataDir, 
    task: PropensityTasks
) -> np.ndarray:
    propensity_targets = np.load(
        data_dir.target_dir / f"{task.value}.npy",
        allow_pickle=True,
    )
    return propensity_targets


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
            self.ids_file = save_dir / f"train_ids_{BATCH_SIZE}.txt"
            self.sequence_file = save_dir / f"train_sequence_{BATCH_SIZE}.parquet"
        else:
            self.ids_file = save_dir / f"relevant_ids_{BATCH_SIZE}.txt"
            self.sequence_file = save_dir / f"relevant_sequence_{BATCH_SIZE}.parquet"

        self.chunk_idx: int = -1
        self.chunk_size: int = BATCH_SIZE * GROUP_SIZE
        self.behavior_sequence_chunk: List = []
        self.client_ids: Set[int] = set()
        self._behavior_sequence()
        self._load_user_features_dict()
        
        # self.category_targets = load_propensity_targets(
        #     self.data_dir, PropensityTasks.PROPENSITY_CATEGORY
        # )
        # self.sku_targets = load_propensity_targets(
        #     self.data_dir, PropensityTasks.PROPENSITY_SKU
        # )
        # self.price_targets = load_propensity_targets(
        #     self.data_dir, PropensityTasks.PROPENSITY_PRICE
        # )

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
        min_max_scaler = MinMaxScaler()
        for col in user_features.columns:
            # Apply normalization for columns
            if any(key in col for key in ["_days_since_", "_time_diff_"]):
                max_value = user_features[col].max()
                user_features[col] = (max_value - user_features[col]) / max_value 
            
            # Apply np.log1p and min-max scaling to columns
            elif any(key in col for key in [
                "_count_", "_price_tier_", "_unique_price_tiers", "_unique_categories",
                "_target_sku_count_", "_unique_target_sku",
                "total_active_days", "_activity_trend_ratio",
            ]):
                user_features[col] = np.log1p(user_features[col], dtype=np.float32)
                user_features[col] = min_max_scaler.fit_transform(user_features[[col]].values)
            
            # Apply min-max scaling and fillna with -1 for columns
            elif any(key in col for key in [
                "_price_mean", "_price_median", "_price_std", "_price_max", "_price_min",
                "_category_avg_price_diff_", 
            ]):
                user_features[col] = min_max_scaler.fit_transform(user_features[[col]].values)
                user_features[col] = user_features[col].fillna(-1).astype(np.float32)  

            # Apply np.log1p, min-max scaling and fillna with -1 to columns
            elif any(key in col for key in [
                "_sku_popularity_mean_", "_sku_popularity_std_", "_sku_popularity_min_", 
                "_sku_popularity_max_", "_sku_popularity_median_",
            ]):
                user_features[col] = np.log1p(user_features[col], dtype=np.float32)
                user_features[col] = min_max_scaler.fit_transform(user_features[[col]].values)
                user_features[col] = user_features[col].fillna(-1).astype(np.float32)
        
            # Apply symmetric np.log1p and min-max scaling to columns
            elif any(key in col for key in [
                "_activity_trend_diff", 
            ]):
                user_features[col] = np.sign(user_features[col]) * np.log1p(
                    np.abs(user_features[col]), dtype=np.float32
                )
                user_features[col] = min_max_scaler.fit_transform(user_features[[col]].values)

            elif any(key in col for key in [
                "total_active_weeks", "total_active_months",
            ]):
                user_features[col] = min_max_scaler.fit_transform(user_features[[col]].values)
        
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
                timestamp = sequence[idx]["timestamp"]  # Preserve timestamp
                sequence[idx] = pad_value.copy()
                sequence[idx]['timestamp'] = timestamp 
        else:  # Cropping
            crop_len = max(1, int(len(sequence) * 0.6))
            start_idx = np.random.randint(0, len(sequence) - crop_len + 1)
            sequence = sequence[start_idx:start_idx + crop_len]

        return sequence[-max_length:]

    def _generate_timestamp_features(self, sequence_entity_info: List[dict]) -> None:
        total_events = len(sequence_entity_info)
        if total_events == 0:
            return
        
        for idx, item in enumerate(sequence_entity_info):
            timestamp = item["timestamp"]
            # if timestamp is None:
            #     logger.info(f"sequence_entity_info: {sequence_entity_info}")
            #     continue
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
            sequence_entity_info[idx]["time_feat"] = np.array(time_feat, dtype=np.float32)

    def _generate_target_features(self, sequence_sku_info: List[dict]) -> None:
        for idx, item in enumerate(sequence_sku_info):
            sku = item["sku"]
            category = item["category"]
            is_category_target = int(category in self.category_targets)
            is_sku_target = int(sku in self.sku_targets)
            target_feat = [is_category_target, is_sku_target]
            sequence_sku_info[idx]["target_feat"] = np.array(target_feat, dtype=np.float32)

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
        self._generate_timestamp_features(sequence_url_info)
        self._generate_timestamp_features(sequence_query_info)

        # # Generate target-related features for sequence_sku_info
        # self._generate_target_features(sequence_sku_info)

        if is_augmentation:
            sequence_sku_info = self._augment_sequence(sequence_sku_info, PAD_SKU, MAX_SEQUENCE_LENGTH)
            sequence_url_info = self._augment_sequence(sequence_url_info, PAD_URL, MAX_SEQUENCE_LENGTH)
            sequence_query_info = self._augment_sequence(sequence_query_info, PAD_QUERY, MAX_SEQUENCE_LENGTH)

        sequence_sku_length = min(len(sequence_sku_info), MAX_SEQUENCE_LENGTH)
        sequence_url_length = min(len(sequence_url_info), MAX_SEQUENCE_LENGTH)
        sequence_query_length = min(len(sequence_query_info), MAX_SEQUENCE_LENGTH)

        # Convert sequence_info to a structured format (e.g., numpy arrays)
        sequence_sku_id = np.array([item["sku"] for item in sequence_sku_info], dtype=np.int64)
        sequence_sku_category = np.array([item["category"] for item in sequence_sku_info], dtype=np.int64)
        sequence_sku_price = np.array([item["price"] for item in sequence_sku_info], dtype=np.int64)
        sequence_sku_event_type = np.array([item["event_type"] for item in sequence_sku_info], dtype=np.int64)
        sequence_sku_timestamp = np.array([item["timestamp"] for item in sequence_sku_info], dtype="datetime64[ns]")
        
        sequence_url_id = np.array([item["url"] for item in sequence_url_info], dtype=np.int64)    
        sequence_url_event_type = np.array([item["event_type"] for item in sequence_url_info], dtype=np.int64)
        sequence_url_timestamp = np.array([item["timestamp"] for item in sequence_url_info], dtype="datetime64[ns]")
        
        sequence_query_event_type = np.array([item["event_type"] for item in sequence_query_info], dtype=np.int64)
        sequence_query_timestamp = np.array([item["timestamp"] for item in sequence_query_info], dtype="datetime64[ns]") 
        
        if len(sequence_sku_info) > 0:
            sequence_sku_name = np.stack(
                [item["name"] for item in sequence_sku_info],
                axis=0,
                dtype=np.float32,
            )
            # sequence_stat_feat = np.stack(
            #     [item["features"] for item in sequence_sku_info],
            #     axis=0,
            #     dtype=np.float32,
            # )
            sequence_sku_time_feat = np.stack(
                [item["time_feat"] for item in sequence_sku_info],
                axis=0,
                dtype=np.float32,
            )
        else:
            sequence_sku_name = np.expand_dims(PAD_VALUE_NAME, axis=0)
            # sequence_stat_feat = np.expand_dims(self.PAD_VALUE_STAT_FEAT, axis=0)
            sequence_sku_time_feat = np.expand_dims(PAD_VALUE_TIME_FEAT, axis=0)
        
        if len(sequence_url_info) > 0:
            sequence_url_time_feat = np.stack(
                [item["time_feat"] for item in sequence_url_info],
                axis=0,
                dtype=np.float32,
            )
        else:
            sequence_url_time_feat = np.expand_dims(PAD_VALUE_TIME_FEAT, axis=0)
        
        if len(sequence_query_info) > 0:
            sequence_query_query = np.stack(
                [query["query"] for query in sequence_query_info],
                axis=0,
                dtype=np.float32,
            )
            sequence_query_time_feat = np.stack(
                [query["time_feat"] for query in sequence_query_info],
                axis=0,
                dtype=np.float32,
            )
        else:
            sequence_query_query = np.expand_dims(PAD_VALUE_QUERY, axis=0)
            sequence_query_time_feat = np.expand_dims(PAD_VALUE_TIME_FEAT, axis=0)
        
        # Mapping ids
        sequence_sku_id = np.array(
            [self.id_mapper.get_sku_id(sku) for sku in sequence_sku_id], 
            dtype=np.int64,
        )
        sequence_sku_category = np.array(
            [self.id_mapper.get_category_id(category) for category in sequence_sku_category], 
            dtype=np.int64,
        )
        sequence_url_id = np.array(
            [self.id_mapper.get_url_id(url) for url in sequence_url_id], 
            dtype=np.int64,
        )
        
        # Padding sequences
        sequence_sku_id = self._pad_sequence(sequence_sku_id, PAD_VALUE_SKU)
        sequence_sku_category = self._pad_sequence(sequence_sku_category, PAD_VALUE_CATEGORY)
        sequence_sku_price = self._pad_sequence(sequence_sku_price, PAD_VALUE_PRICE)
        sequence_sku_name = self._pad_sequence(sequence_sku_name, PAD_VALUE_NAME)
        sequence_sku_event_type = self._pad_sequence(sequence_sku_event_type, PAD_VALUE_EVENT_TYPE)
        sequence_sku_time_feat = self._pad_sequence(sequence_sku_time_feat, PAD_VALUE_TIME_FEAT)
        sequence_sku_timestamp = self._pad_sequence(sequence_sku_timestamp, PAD_VALUE_TIMESTAMP)
        
        sequence_url_id = self._pad_sequence(sequence_url_id, PAD_VALUE_URL)
        sequence_url_event_type = self._pad_sequence(sequence_url_event_type, PAD_VALUE_EVENT_TYPE)
        sequence_url_time_feat = self._pad_sequence(sequence_url_time_feat, PAD_VALUE_TIME_FEAT)
        sequence_url_timestamp = self._pad_sequence(sequence_url_timestamp, PAD_VALUE_TIMESTAMP)
        
        sequence_query_query = self._pad_sequence(sequence_query_query, PAD_VALUE_QUERY)
        sequence_query_event_type = self._pad_sequence(sequence_query_event_type, PAD_VALUE_EVENT_TYPE)
        sequence_query_time_feat = self._pad_sequence(sequence_query_time_feat, PAD_VALUE_TIME_FEAT)
        sequence_query_timestamp = self._pad_sequence(sequence_query_timestamp, PAD_VALUE_TIMESTAMP)
    
        # Get user features
        user_features = self.user_features_dict[client_id]

        # Cast the timestamp to int
        sequence_sku_timestamp = sequence_sku_timestamp.astype('int64')
        sequence_url_timestamp = sequence_url_timestamp.astype('int64')
        sequence_query_timestamp = sequence_query_timestamp.astype('int64')
        # 
        # cate_target_count, cate_diversity, cate_target_prop = create_target_propensity_features(
        #     sequence_sku_info=sequence_sku_info,
        #     propensity_targets=self.category_targets,
        #     target_col="category",
        # )
        
        # sku_target_count, sku_diversity, sku_target_prop = create_target_propensity_features(
        #     sequence_sku_info=sequence_sku_info,
        #     propensity_targets=self.sku_targets,
        #     target_col="sku",
        # )
        
        # price_target_count, price_diversity, price_target_prop = create_target_propensity_features(
        #     sequence_sku_info=sequence_sku_info,
        #     propensity_targets=self.price_targets,
        #     target_col="price",
        # )

        # Combine the structured data into a single array or return as a tuple
        behavior_data = (
            client_id,
            sequence_sku_id,
            sequence_sku_category,
            sequence_sku_price,
            sequence_sku_name,
            sequence_sku_event_type,
            sequence_sku_time_feat,
            sequence_sku_timestamp,
            sequence_url_id,
            sequence_url_event_type,
            sequence_url_time_feat,
            sequence_url_timestamp,
            sequence_query_query,
            sequence_query_event_type,
            sequence_query_time_feat,
            sequence_query_timestamp,
            sequence_sku_length, 
            sequence_url_length,
            sequence_query_length,
            user_features,
            # cate_target_count,
            # cate_diversity,
            # cate_target_prop,
            # sku_target_count, 
            # sku_diversity, 
            # sku_target_prop,
            # price_target_count,
            # price_diversity,
            # price_target_prop,
        )
        
        # if sequence_sku_length >= 1:
        #     logger.info(f"idx: \n {idx}")
        #     logger.info(f"is_augmentation: \n {is_augmentation}")
        #     logger.info(f"client_id: \n {client_id}")
        #     logger.info(f"sequence_sku_id: \n {sequence_sku_id}")
        #     logger.info(f"sequence_sku_category: \n {sequence_sku_category}")
        #     logger.info(f"sequence_sku_price: \n {sequence_sku_price}")
        #     logger.info(f"sequence_sku_event_type: \n {sequence_sku_event_type}")
        #     logger.info(f"cate_target_count: \n {cate_target_count}")
        #     logger.info(f"sku_target_count: \n {sku_target_count}")
        #     logger.info(f"price_target_count: \n {price_target_count}")
        #     # logger.info(f"sequence_sku_time_feat: \n {sequence_sku_time_feat}")
        #     logger.info(f"sequence_sku_timestamp: \n {sequence_sku_timestamp}")
        #     logger.info(f"sequence_url_id: \n {sequence_url_id}")
        #     logger.info(f"sequence_url_event_type: \n {sequence_url_event_type}")
        #     # logger.info(f"sequence_url_time_feat: \n {sequence_url_time_feat}")
        #     logger.info(f"sequence_url_timestamp: \n {sequence_url_timestamp}")
        #     # logger.info(f"sequence_query_query: \n {sequence_query_query}")
        #     logger.info(f"sequence_query_event_type: \n {sequence_query_event_type}")
        #     # logger.info(f"sequence_query_time_feat: \n {sequence_query_time_feat}")
        #     logger.info(f"sequence_query_timestamp: \n {sequence_query_timestamp}")
        #     logger.info(f"sequence_sku_length: \n {sequence_sku_length}")
        #     logger.info(f"sequence_url_length: \n {sequence_url_length}")
        #     logger.info(f"sequence_query_length: \n {sequence_query_length}")
        #     logger.info(f"-" * 50)
        
        return (behavior_data, target) if not is_augmentation and self.mode == "train" else behavior_data


def get_last_interaction_features(
    sequence_sku_info: List[Dict],
    target_col: str,
):
    if target_col == "category":
        default_pad_value = PAD_VALUE_CATEGORY
    elif target_col == "price":
        default_pad_value = PAD_VALUE_PRICE
    else:
        raise ValueError(f"target_col must 'price' or 'category', but given '{target_col}'")

    last_values = {
        EventTypes.PRODUCT_BUY.get_index(): default_pad_value,
        EventTypes.ADD_TO_CART.get_index(): default_pad_value,
        EventTypes.REMOVE_FROM_CART.get_index(): default_pad_value,
    }
    
    for item in sequence_sku_info:
        event_type = item.get("event_type")
        if event_type in last_values:
            target_value = item.get(target_col)
            if target_value is not None:
                last_values[event_type] = target_value

    last_interaction_features = np.array([
        last_values[EventTypes.PRODUCT_BUY.get_index()],
        last_values[EventTypes.ADD_TO_CART.get_index()],
        last_values[EventTypes.REMOVE_FROM_CART.get_index()],
    ], dtype=np.int64)
    
    return last_interaction_features


def create_target_propensity_features(
        sequence_sku_info: List[Dict],
        propensity_targets: np.ndarray,
        target_col: str,
):

    target_counts = {target: [0, 0, 0] for target in propensity_targets}
    diversity_sets = {
        EventTypes.PRODUCT_BUY.get_index(): set(),
        EventTypes.ADD_TO_CART.get_index(): set(),
        EventTypes.REMOVE_FROM_CART.get_index(): set()
    }
    
    total_events = 0
    target_events = 0

    for item in sequence_sku_info:
        event_type = item["event_type"]
        target = item[target_col]
        
        if event_type not in diversity_sets:
            continue
            
        total_events += 1
        diversity_sets[event_type].add(target)
        
        if target in propensity_targets:
            target_events += 1
            
            if event_type == EventTypes.PRODUCT_BUY.get_index():
                target_counts[target][0] += 1
            elif event_type == EventTypes.ADD_TO_CART.get_index():
                target_counts[target][1] += 1
            else:  # REMOVE_FROM_CART
                target_counts[target][2] += 1

    # 1. Target Statistics
    count_features = []
    for target in propensity_targets:
        count_features.extend(target_counts[target])
    
    # 2. Target Diversity
    diversity_features = np.array([
        len(diversity_sets[EventTypes.PRODUCT_BUY.get_index()]),
        len(diversity_sets[EventTypes.ADD_TO_CART.get_index()]),
        len(diversity_sets[EventTypes.REMOVE_FROM_CART.get_index()])
    ])
    
    # 3. Target Proportion
    proportion = target_events / total_events if total_events > 0 else 0.0
    
    # Normalization
    count_features = np.log1p(count_features, dtype=np.float32)
    diversity_features = np.log1p(diversity_features, dtype=np.float32)
    proportion = np.array([proportion], dtype=np.float32)
    
    return count_features, diversity_features, proportion
