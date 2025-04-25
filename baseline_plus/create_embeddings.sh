VERSION=baseline
STATE=online
EMBEDDINGS_DIR="/data/lyjiang/RecSys_Challenge_2025/submit/${VERSION}/${STATE}"

mkdir -p "${EMBEDDINGS_DIR}"

python -m baseline.aggregated_features_baseline.create_embeddings \
    --data-dir /data/lyjiang/RecSys_Challenge_2025 \
    --embeddings-dir "${EMBEDDINGS_DIR}"
    