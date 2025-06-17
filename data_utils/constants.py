from enum import Enum


class EventTypes(str, Enum):
    PRODUCT_BUY = "product_buy"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    PAGE_VISIT = "page_visit"
    SEARCH_QUERY = "search_query"


PROPERTIES_FILE = "product_properties.parquet"
ITEM_FEATURES_FILE = "item_features.parquet"
CATE_FEATURES_FILE = "cate_features.parquet"

DAYS_IN_TARGET = 14
