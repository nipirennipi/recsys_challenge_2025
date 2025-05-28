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
