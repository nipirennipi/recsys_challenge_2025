import numpy as np
import pandas as pd
import logging
import re
from typing import Dict, Tuple, List, Set
from torch.utils.data import Dataset

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
    PAD_VALUE_URL,
    PAD_VALUE_QUERY,
    MAX_SEQUENCE_LENGTH,
    NAME_MIN_VALUE,
    NAME_MAX_VALUE,
    QUERY_MIN_VALUE,
    QUERY_MAX_VALUE,
    PRICE_MIN_VALUE,
    PRICE_MAX_VALUE,
)


logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def raise_err_if_incorrect_form(string_representation_of_vector: str):
    """
    Checks if string_representation_of_vector has the correct form.

    Correct form is a string representing list of ints with arbitrary number of spaces in between.

    Args:
        string_representation_of_vector (str): potential string representation of vector
    """
    m = re.fullmatch(r"\[( *\d* *)*\]", string=string_representation_of_vector)
    if m is None:
        raise ValueError(
            f"{string_representation_of_vector} is incorrect form of string representation of vector – correct form is: '[( *\d* *)*]'"
        )


def parse_to_array(string_representation_of_vector: str) -> np.ndarray:
    """
    Parses string representing vector of integers into array of integers.

    Args:
        string_representation_of_vector (str): string representing vector of ints e.g. '[11 2 3]'
    Returns:
        np.ndarray: array of integers obtained from string representation
    """
    raise_err_if_incorrect_form(
        string_representation_of_vector=string_representation_of_vector
    )
    string_representation_of_vector = string_representation_of_vector.replace(
        "[", ""
    ).replace("]", "")
    return np.array(
        [int(s) for s in string_representation_of_vector.split(" ") if s != ""]
    ).astype(dtype=np.float32)


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
        mode: str = "train",
    ) -> None:
        super().__init__()

        self.data_dir = data_dir
        self.id_mapper = id_mapper
        self.target_df = target_df
        self.target_calculators = target_calculators
        self.mode = mode
        
        self.client_ids: Set[int] = set()
        self.behavior_sequence = []
        self.properties_dict: Dict[int, Dict[str, object]] = {}
        
        self._behavior_sequence()
        self._load_properties_dict()

    def _load_properties_dict(self) -> None:
        """
        Load properties from the properties file and construct a dictionary
        with sku as the key and its attributes as the value.
        """
        logger.info("Loading properties")
        properties = pd.read_parquet(self.data_dir.properties_file)

        self.properties_dict = {
            row["sku"]: {
                "category": row["category"],
                "price": row["price"],
                "name": row["name"],
            }
            for _, row in properties.iterrows()
        }

    def _behavior_sequence(self) -> None:
        """
        Construct the user's historical behavior sequence, including two types 
        of events: ADD_TO_CART and PRODUCT_BUY.
        """
        logger.info(f"Constructing {self.mode} user behavior sequence")
        if self.mode != "train":
            self.client_ids = set(self.target_df["client_id"])
        all_events = []

        for event_type in EventTypes:
            events = self._load_events(event_type=event_type)
            if self.mode != "train":
                events = events[events["client_id"].isin(self.client_ids)]
            events = events.rename(
                columns={EVENT_TYPE_TO_COLUMNS[event_type]: ENTITY_COLUMN_NAME}
            )
            events["event_type"] = event_type.get_index()
            all_events.append(events)

        all_events_df = pd.concat(all_events)
        # all_events_df = all_events_df.sample(frac=1).head(3000000)
        logger.info(f"Sampled all_events_df length: {len(all_events_df)}")
        all_events_df["timestamp"] = pd.to_datetime(all_events_df["timestamp"])
        all_events_df = all_events_df.sort_values(by=["client_id", "timestamp"])

        self.behavior_sequence = [
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
        
        self.client_ids = {entry["client_id"] for entry in self.behavior_sequence}
        # self.target_df = self.target_df[self.target_df["client_id"].isin(self.client_ids)]
        
        logger.info(f"Behavior sequence constructed for {len(self.client_ids)} clients")
        # [
        #     {
        #         "client_id": 123, 
        #         "sequence": [
        #           (Timestamp('2025-04-01 10:00:00'), '12345', 'ADD_TO_CART'),
        #           (Timestamp('2025-04-01 11:00:00'), '67890', 'PRODUCT_BUY'),
        #         ],
        #     }
        # ]
    
    def _pad_sequence(self, sequence: np.ndarray, pad_value: object) -> np.ndarray:
        """
        Pads a sequence to the specified maximum length with a given pad value.
        Args:
            sequence (np.ndarray): The input sequence to be padded.
            pad_value (object): The value used for padding.
        """
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

    def __len__(self) -> int:
        return len(self.client_ids)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        client_id = self.behavior_sequence[idx]["client_id"]
        target = [
            target_calculator.compute_target(
                client_id=client_id, target_df=self.target_df
            )
            for target_calculator in self.target_calculators
        ]

        # Get the behavior sequence for the client
        behavior_sequence = self.behavior_sequence[idx]["sequence"]
        sequence_sku_info = []
        sequence_url_info = []
        sequence_query_info = []

        for timestamp, entity, event_type in behavior_sequence:
            if event_type in [
                EventTypes.ADD_TO_CART.get_index(), 
                EventTypes.PRODUCT_BUY.get_index()
            ]:
                sku = int(entity)
                properties = self.properties_dict[sku]
                sequence_sku_info.append({
                    "sku": sku,
                    "category": properties["category"],
                    "price": properties["price"],
                    "name": properties["name"],
                    "event_type": event_type,
                    "timestamp": timestamp,
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

        sequence_sku_length = max(min(len(sequence_sku_info), MAX_SEQUENCE_LENGTH), 1)
        sequence_url_length = max(min(len(sequence_url_info), MAX_SEQUENCE_LENGTH), 1)
        sequence_query_length = max(min(len(sequence_query_info), MAX_SEQUENCE_LENGTH), 1)

        # Convert sequence_info to a structured format (e.g., numpy arrays)
        sequence_sku = np.array([item["sku"] for item in sequence_sku_info], dtype=np.int64)
        sequence_category = np.array([item["category"] for item in sequence_sku_info], dtype=np.int64)
        sequence_price = np.array([item["price"] for item in sequence_sku_info], dtype=np.int64)
        sequence_event_type = np.array([item["event_type"] for item in sequence_sku_info], dtype=np.int64)
        sequence_url = np.array([item["url"] for item in sequence_url_info], dtype=np.int64)    
        
        if len(sequence_sku_info) > 0:
            sequence_name = np.stack(
                [
                    parse_to_array(string_representation_of_vector=item["name"]) 
                    for item in sequence_sku_info
                ],
                axis=0,
            )
        else:
            sequence_name = np.expand_dims(PAD_VALUE_NAME, axis=0)
        
        if len(sequence_query_info) > 0:
            sequence_query = np.stack(
                [
                    parse_to_array(string_representation_of_vector=query["query"]) 
                    for query in sequence_query_info
                ],
                axis=0,
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
        
        # Normalize price, sequence_name and sequence_query
        sequence_price = (sequence_price - PRICE_MIN_VALUE) / (PRICE_MAX_VALUE - PRICE_MIN_VALUE)
        sequence_price = np.clip(sequence_price, -1, 1, dtype=np.float32)
        sequence_name = (sequence_name - NAME_MIN_VALUE) / (NAME_MAX_VALUE - NAME_MIN_VALUE)
        sequence_name = np.clip(sequence_name, -1, 1, dtype=np.float32)
        sequence_query = (sequence_query - QUERY_MIN_VALUE) / (QUERY_MAX_VALUE - QUERY_MIN_VALUE)
        sequence_query = np.clip(sequence_query, -1, 1, dtype=np.float32)
        
        # Padding sequences
        sequence_sku = self._pad_sequence(sequence_sku, PAD_VALUE_SKU)
        sequence_category = self._pad_sequence(sequence_category, PAD_VALUE_CATEGORY)
        sequence_price = self._pad_sequence(sequence_price, PAD_VALUE_PRICE)
        sequence_name = self._pad_sequence(sequence_name, PAD_VALUE_NAME)
        sequence_event_type = self._pad_sequence(sequence_event_type, PAD_VALUE_EVENT_TYPE)
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

        # Combine the structured data into a single array or return as a tuple
        behavior_data = (
            client_id,
            sequence_sku, 
            sequence_category, 
            sequence_price, 
            sequence_name, 
            sequence_event_type, 
            sequence_url,
            sequence_query,
            sequence_sku_length, 
            sequence_url_length,
            sequence_query_length,
        )
        
        if np.any(sequence_name > 1) or np.any(sequence_name < -1):
            logger.info(f"Out of range values found in sequence_name, max: {np.max(sequence_name)}, min: {np.min(sequence_name)}")

        if np.any(sequence_query > 1) or np.any(sequence_query < -1):
            logger.info(f"Out of range values found in sequence_query, max: {np.max(sequence_query)}, min: {np.min(sequence_query)}")
            
        return behavior_data, target
