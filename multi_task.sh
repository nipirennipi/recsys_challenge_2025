VERSION=multi_task
EMBEDDINGS_DIR="/data/lyjiang/RecSys_Challenge_2025/submit/${VERSION}"

mkdir -p "${EMBEDDINGS_DIR}"

python -m multi_task.tarin \
    --data-dir /data/lyjiang/RecSys_Challenge_2025/input \
    --embeddings-dir "${EMBEDDINGS_DIR}"
    