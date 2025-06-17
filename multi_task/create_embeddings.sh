VERSION=multi_task
STATE=online
DEVICE_ID=7
EMBEDDINGS_DIR="/data/lyjiang/RecSys_Challenge_2025/submit/${VERSION}/${STATE}"

mkdir -p "${EMBEDDINGS_DIR}"

if [ "${STATE}" = "offline" ]; then
    DATA_DIR="/data/lyjiang/RecSys_Challenge_2025/input"
elif [ "${STATE}" = "online" ]; then
    DATA_DIR="/data/lyjiang/RecSys_Challenge_2025"
else
    echo "Error: Invalid STATE value '${STATE}'" >&2
    exit 1
fi

python -m multi_task.train \
    --data-dir "${DATA_DIR}" \
    --embeddings-dir "${EMBEDDINGS_DIR}" \
    --tasks churn propensity_category propensity_sku propensity_price \
    --log-name "${VERSION}" \
    --accelerator gpu \
    --devices ${DEVICE_ID} \
    --disable-relevant-clients-check

if [ "${STATE}" = "offline" ]; then
    ./training.sh ${DEVICE_ID}
fi
