import numpy as np
from enum import Enum
from typing import Dict, List


class EventTypes(str, Enum):
    PRODUCT_BUY = "product_buy"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PAGE_VISIT = "page_visit"
    SEARCH_QUERY = "search_query"

    def get_index(self):
        return list(EventTypes).index(self) + 1


EVENT_TYPE_TO_COLUMNS: Dict[EventTypes, List[str]] = {
    EventTypes.PRODUCT_BUY: "sku",
    EventTypes.ADD_TO_CART: "sku",
    EventTypes.REMOVE_FROM_CART: "sku",
    EventTypes.PAGE_VISIT: "url",
    EventTypes.SEARCH_QUERY: "query",
}


ENTITY_COLUMN_NAME = "entity"

# Gpu memory to allocate
GPU_MEMORY_TO_ALLOCATE = 15  # GB

# Frequency cutoff
URL_FREQUENCY_CUTOFF = 20   # 94.6% URLs appear less than 20 times

# In competition settings these hyperparameters are fixed
BATCH_SIZE = 128 * 16
GROUP_SIZE = 100    # DO NOT CHANGE THIS VALUE
HIDDEN_SIZE_THIN = 2048
HIDDEN_SIZE_WIDE = 4096
LEARNING_RATE = 0.001
MAX_EPOCH = 1
MAX_EMBEDDING_DIM = 2048
MAX_SEQUENCE_LENGTH = 50

EMBEDDINGS_DTYPE = np.float16
CLIENT_IDS_DTYPE = np.int64
NAME_EMBEDDING_DIM = 16
NAME_MIN_VALUE = 0
NAME_MAX_VALUE = 255
QUERY_MIN_VALUE = 0
QUERY_MAX_VALUE = 255
# PRICE_MIN_VALUE = 0
# PRICE_MAX_VALUE = 99
NUM_PRICE_BINS = 100
QUERY_EMBEDDING_DIM = 16
TIME_FEAT_DIM = 5
TARGET_FEAT_DIM = 2
USER_STAT_FEAT_DIM = (
    143
)
ITEM_STAT_FEAT_DIM = 15

PAD_VALUE_SKU = 0
PAD_VALUE_CATEGORY = 0
PAD_VALUE_PRICE = 0
PAD_VALUE_NAME = np.zeros(NAME_EMBEDDING_DIM, dtype=np.float32)
PAD_VALUE_EVENT_TYPE = 0
PAD_VALUE_TIME_FEAT = np.zeros(TIME_FEAT_DIM, dtype=np.float32)
PAD_VALUE_TARGET_FEAT = np.zeros(TARGET_FEAT_DIM, dtype=np.float32)
PAD_VALUE_URL = 0
PAD_VALUE_QUERY = np.zeros(QUERY_EMBEDDING_DIM, dtype=np.float32)
PAD_VALUE_TIMESTAMP = np.datetime64('2262-04-11T23:47:16.854775807', 'ns')
PAD_SKU = {
    "sku": PAD_VALUE_SKU, 
    "category": PAD_VALUE_CATEGORY, 
    "price": PAD_VALUE_PRICE,
    "name": PAD_VALUE_NAME, 
    "event_type": PAD_VALUE_EVENT_TYPE, 
    "timestamp": None, 
    "time_feat": PAD_VALUE_TIME_FEAT
}
PAD_URL = {
    "url": PAD_VALUE_URL, 
    "event_type": PAD_VALUE_EVENT_TYPE, 
    "timestamp": None,
    "time_feat": PAD_VALUE_TIME_FEAT
}
PAD_QUERY = {
    "query": PAD_VALUE_QUERY, 
    "event_type": PAD_VALUE_EVENT_TYPE, 
    "timestamp": None,
    "time_feat": PAD_VALUE_TIME_FEAT
}

SKU_ID_EMBEDDING_DIM = 32 * 4
SKU_CATEGORY_EMBEDDING_DIM = 16 * 2
SKU_PRICE_EMBEDDING_DIM = 16
URL_EMBEDDING_DIM = 32 * 2

# LSTM_HIDDEN_SIZE = 64 * 4
# LSTM_NUM_LAYERS = 2
# LSTM_DROPOUT = 0.1
# LSTM_BIDIRECTIONAL = False

SKU_EMBEDDING_DIM = (
    SKU_ID_EMBEDDING_DIM 
    + SKU_CATEGORY_EMBEDDING_DIM 
    + SKU_PRICE_EMBEDDING_DIM
    + NAME_EMBEDDING_DIM
)
SASREC_HIDDEN_DIM = 128
SASREC_NUM_HEADS = 1
SASREC_NUM_LAYERS = 2

EMBEDDING_DIM = (
    USER_STAT_FEAT_DIM
    + SASREC_HIDDEN_DIM
)

NUM_CROSS_LAYERS = 2
DEEP_HIDDEN_DIMS = [EMBEDDING_DIM * 2, EMBEDDING_DIM * 2, EMBEDDING_DIM]

# Contrastive Learning Hyperparameters
CONTRASTIVE_TEMP = 0.1  # Temperature for contrastive loss
CONTRASTIVE_LAMBDA = 0.5  # Weight for contrastive loss
MLP_PROJECTION_DIM = 128  # Dimension of the MLP projection head

# Augmentation Method Hyperparameters
AUGMENTATION_METHOD_1 = "mask"  # mask, crop and reorder
AUGMENTATION_METHOD_2 = "reorder"
MASK_PROPORTION = 0.3 # Proportion of items to mask in the sequence
CROP_PROPORTION = 0.8  # Proportion of items to crop in the sequence
REORDER_PROPORTION = 0.3  # Proportion of items to reorder in the sequence

# USER_FEATURE_AUGMENTATION_METHOD = "mask"
# USER_FEATURE_MASK_PROPORTION = 0.3  # Proportion of user features to mask
