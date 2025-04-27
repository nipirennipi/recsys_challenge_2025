import numpy as np
import logging
from typing import Dict
from pathlib import Path

from multi_task.constants import (
    EMBEDDINGS_DTYPE,
    CLIENT_IDS_DTYPE,
)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


EMBEDDINGS_DICT: Dict[int, np.ndarray] = {}
def record_embeddings(client_id: np.ndarray, embedding: np.ndarray) -> None:
    """
    Record embeddings for a given client_id and embedding.
    Args:
        client_id (str): The ID of the client.
        embedding (np.ndarray): The embedding to record.
    """
    global EMBEDDINGS_DICT
    for cid, emb in zip(client_id, embedding):
        EMBEDDINGS_DICT[cid] = emb


def save_embeddings(
    embeddings_dir: Path
):
    """
    Function creates embeddings directory and saves embeddings in competition entry format.

    Args:
    embeddings_dir (Path): The directory where to save embeddings and client_ids.
    """
    logger.info("Saving embeddings")
    global EMBEDDINGS_DICT
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    client_ids = np.array(list(EMBEDDINGS_DICT.keys()), dtype=CLIENT_IDS_DTYPE)
    embeddings = np.array(list(EMBEDDINGS_DICT.values()), dtype=EMBEDDINGS_DTYPE)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    np.save(embeddings_dir / "client_ids.npy", client_ids)
