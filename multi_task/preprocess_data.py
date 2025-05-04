import pandas as pd
from typing import Dict
from pathlib import Path
import argparse
import logging

from data_utils.utils import join_properties
from data_utils.data_dir import DataDir
from multi_task.constants import (
    EventTypes,
    PAD_VALUE_SKU,
    PAD_VALUE_CATEGORY,
    PAD_VALUE_URL,
)
import json

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class IdMapper:
    def __init__(
            self, 
            challenge_data_dir: DataDir,
        ):
        """
        Args:
            challenge_data_dir (DataDir): The DataDir class where Paths to raw event data, input and targte folders are stored.
        """
        self.challenge_data_dir = challenge_data_dir
        self.client_id_map: Dict[int, int] = {}
        self.sku_id_map: Dict[int, int] = {}
        self.category_id_map: Dict[int, int] = {}
        self.url_id_map: Dict[int, int] = {}
    
    def load_events(self, event_type: EventTypes) -> pd.DataFrame:
        return pd.read_parquet(
            self.challenge_data_dir.data_dir / f"{event_type.value}.parquet"
        )
    
    def _map_client_ids(self) -> None:
        """
        Maps client ids across all event types to a new range of ids starting from 1.
        """
        all_client_ids = set()
        
        for event_type in EventTypes:
            msg = f"Loading client ids in {event_type.value} event type"
            logger.info(msg=msg)
            events = self.load_events(event_type=event_type)
            all_client_ids.update(events["client_id"].unique())
        
        self.client_id_map = {
            client_id: i + 1 for i, client_id in enumerate(all_client_ids)
        }
        
        logger.info("Client ID mapping completed.")
    
    def _map_sku_ids(self) -> None:
        """
        Maps sku ids across all event types to a new range of ids starting from 1.
        """
        properties = pd.read_parquet(self.challenge_data_dir.properties_file)
        self.sku_id_map = {
            sku_id: i + 1 for i, sku_id in enumerate(properties['sku'].unique())
        }
        
        logger.info("Sku ID mapping completed.")
    
    def _map_category_ids(self) -> None:
        """
        Maps category ids across all event types to a new range of ids starting from 1.
        """
        properties = pd.read_parquet(self.challenge_data_dir.properties_file)
        self.category_id_map = {
            category_id: i + 1 for i, category_id in enumerate(properties['category'].unique())
        }
        
        logger.info("Category ID mapping completed.")
    
    def _map_url_ids(self) -> None:
        """
        Maps URL ids across all event types to a new range of ids starting from 1.
        """
        events = self.load_events(event_type=EventTypes.PAGE_VISIT)
        self.url_id_map = {
            url: i + 1 for i, url in enumerate(events['url'].unique())
        }
        
        logger.info("URL ID mapping completed.")
    
    def get_client_id(self, client_id: int) -> int:
        """
        Returns the mapped client id.
        """
        return self.client_id_map[client_id]
    
    def get_sku_id(self, sku: int) -> int:
        """
        Returns the mapped sku id.
        """
        if sku == PAD_VALUE_SKU:
            return PAD_VALUE_SKU
        return self.sku_id_map[sku]
    
    def get_category_id(self, category: int) -> int:
        """
        Returns the mapped category id.
        """
        if category == PAD_VALUE_CATEGORY:
            return PAD_VALUE_CATEGORY
        return self.category_id_map[category]
    
    def get_url_id(self, url: int) -> int:
        """
        Returns the mapped url id.
        """
        if url == PAD_VALUE_URL:
            return PAD_VALUE_URL
        return self.url_id_map[url]
    
    def client_vocab_size(self) -> int:
        """
        Returns the size of the client id mapping.
        """
        return len(self.client_id_map)
    
    def sku_vocab_size(self) -> int:
        """
        Returns the size of the sku id mapping.
        """
        return len(self.sku_id_map)
    
    def category_vocab_size(self) -> int:
        """
        Returns the size of the category id mapping.
        """
        return len(self.category_id_map)
    
    def url_vocab_size(self) -> int:
        """
        Returns the size of the url id mapping.
        """
        return len(self.url_id_map)
    
    def id_mapping(self) -> None:
        """
        Maps client ids, sku ids and category ids across all event types.
        """
        self._map_client_ids()
        self._map_sku_ids()
        self._map_category_ids()
        self._map_url_ids()
        
        logger.info("All ID mapping completed.")
    
    def save_mapping(self) -> None:  
        """
        Saves the client_id_map, sku_id_map, and category_id_map to a single .json file.
        """
        mapping_data = {
            "client_id_map": {int(k): v for k, v in self.client_id_map.items()},
            "sku_id_map": {int(k): v for k, v in self.sku_id_map.items()},
            "category_id_map": {int(k): v for k, v in self.category_id_map.items()},
            "url_id_map": {int(k): v for k, v in self.url_id_map.items()},
        }
        
        save_path = self.challenge_data_dir.data_dir / "id_mapping.json"
        with open(save_path, "w") as f:
            json.dump(mapping_data, f)
        
        logger.info("Saved all ID mappings to id_mapping.json")

    def load_mapping(self) -> None:
        """
        Loads the client_id_map, sku_id_map, and category_id_map from a .json file.
        """
        mapping_file = self.challenge_data_dir.data_dir / "id_mapping.json"
        with open(mapping_file, "r") as f:
            mapping_data = json.load(f)
        
        self.client_id_map = {int(k): int(v) for k, v in mapping_data["client_id_map"].items()}
        self.sku_id_map = {int(k): int(v) for k, v in mapping_data["sku_id_map"].items()}
        self.category_id_map = {int(k): int(v) for k, v in mapping_data["category_id_map"].items()}
        self.url_id_map = {int(k): int(v) for k, v in mapping_data["url_id_map"].items()}
        
        logger.info("Loaded all ID mappings from id_mapping.json")
        logger.info(
            f"Loaded {len(self.client_id_map)} client ids, "
            f"{len(self.sku_id_map)} sku ids, and "
            f"{len(self.category_id_map)} category ids."
            f"{len(self.url_id_map)} url ids."
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--challenge-data-dir",
        type=str,
        required=True,
        help="Competition data directory which should consists of event files, product properties and two subdirectories — input and target",
    )
    return parser


def main():
    parser = get_parser()
    params = parser.parse_args()

    challenge_data_dir = DataDir(data_dir=Path(params.challenge_data_dir))

    # Map client ids across all event types
    id_mapper = IdMapper(challenge_data_dir=challenge_data_dir)
    id_mapper.id_mapping()
    id_mapper.save_mapping()


if __name__ == "__main__":
    main()
