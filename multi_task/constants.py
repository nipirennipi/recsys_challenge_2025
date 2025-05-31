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

# In competition settings these hyperparameters are fixed
BATCH_SIZE = 128 * 32
GROUP_SIZE = 100
HIDDEN_SIZE_THIN = 2048
HIDDEN_SIZE_WIDE = 4096
LEARNING_RATE = 0.001
MAX_EPOCH = 1
MAX_EMBEDDING_DIM = 2048

EMBEDDINGS_DTYPE = np.float16
CLIENT_IDS_DTYPE = np.int64
NAME_EMBEDDING_DIM = 16
NAME_MIN_VALUE = 0
NAME_MAX_VALUE = 255
QUERY_MIN_VALUE = 0
QUERY_MAX_VALUE = 255
PRICE_MIN_VALUE = 0
PRICE_MAX_VALUE = 99
QUERY_EMBEDDING_DIM = 16
TIME_FEAT_DIM = 5
USER_STAT_FEAT_DIM = 39 + 34 # other + price_prop

PAD_VALUE_SKU = 0
PAD_VALUE_CATEGORY = 0
PAD_VALUE_PRICE = 0
PAD_VALUE_NAME = np.zeros(NAME_EMBEDDING_DIM, dtype=np.float32)
PAD_VALUE_EVENT_TYPE = 0
PAD_VALUE_TIME_FEAT = np.zeros(TIME_FEAT_DIM, dtype=np.float32)
PAD_VALUE_URL = 0
PAD_VALUE_QUERY = np.zeros(QUERY_EMBEDDING_DIM, dtype=np.float32)

MAX_SEQUENCE_LENGTH = 100

SKU_EMBEDDING_DIM = 32 * 4
CATEGORY_EMBEDDING_DIM = 16 * 2
PRICE_EMBEDDING_DIM = 8 * 2
EVENT_TYPE_EMBEDDING_DIM = 8 * 2
URL_EMBEDDING_DIM = 32

LSTM_HIDDEN_SIZE = 64 * 4
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.1
LSTM_BIDIRECTIONAL = False

EMBEDDING_DIM = (
    LSTM_HIDDEN_SIZE
    + URL_EMBEDDING_DIM
    + QUERY_EMBEDDING_DIM
    + USER_STAT_FEAT_DIM
)

NUM_CROSS_LAYERS = 2
DEEP_HIDDEN_DIMS = [EMBEDDING_DIM * 2, EMBEDDING_DIM * 2, EMBEDDING_DIM]

# Contrastive Learning Hyperparameters
CONTRASTIVE_TEMP = 0.1  # Temperature for contrastive loss
CONTRASTIVE_LAMBDA = 0.5  # Weight for contrastive loss
MLP_PROJECTION_DIM = 128  # Dimension of the MLP projection head
