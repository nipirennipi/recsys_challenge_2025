import numpy as np
import logging
import re
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
    global EMBEDDINGS_DICT
    if len(EMBEDDINGS_DICT) == 0:
        logger.warning("No embeddings to save.")
        return
    logger.info("Saving embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    client_ids = np.array(list(EMBEDDINGS_DICT.keys()), dtype=CLIENT_IDS_DTYPE)
    embeddings = np.array(list(EMBEDDINGS_DICT.values()), dtype=EMBEDDINGS_DTYPE)
    np.save(embeddings_dir / "embeddings.npy", embeddings)
    np.save(embeddings_dir / "client_ids.npy", client_ids)


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
