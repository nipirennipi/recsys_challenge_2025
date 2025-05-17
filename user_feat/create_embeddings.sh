VERSION=user_feat
STATE=online

if [ "${STATE}" = "offline" ]; then
    DATA_DIR="/data/lyjiang/RecSys_Challenge_2025/input"
elif [ "${STATE}" = "online" ]; then
    DATA_DIR="/data/lyjiang/RecSys_Challenge_2025"
else
    echo "Error: Invalid STATE value '${STATE}'" >&2
    exit 1
fi

EMBEDDINGS_DIR="${DATA_DIR}/target"

python -m user_feat.aggregated_features_baseline.create_embeddings \
    --data-dir "${DATA_DIR}" \
    --embeddings-dir "${EMBEDDINGS_DIR}" \
    --num-days 1 7 30 60 90 \
