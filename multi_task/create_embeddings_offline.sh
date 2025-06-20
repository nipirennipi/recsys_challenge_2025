VERSION=multi_task
STATE=offline
DEVICE_ID=5
DATA_DIR="/data/lyjiang/RecSys_Challenge_2025/input"

FNAME=194
SCORES_DIR="./${VERSION}/score/scores_${FNAME}.txt"
EMBEDDINGS_DIR="/data/lyjiang/RecSys_Challenge_2025/submit/${VERSION}/${STATE}/${FNAME}"

TRIALS=(
    "mask mask 0.1 0.9 0.4"
    "mask crop 0.1 0.9 0.4"
    "mask reorder 0.1 0.9 0.4"
    # "crop crop 0.1 0.9 0.4"
    "crop reorder 0.1 0.9 0.4"
    "reorder reorder 0.1 0.9 0.4"
)

mkdir -p "${EMBEDDINGS_DIR}"
mkdir -p "./${VERSION}/score"

for trial in "${TRIALS[@]}"; do
    read -r METHOD_1 METHOD_2 MASK_PROP CROP_PROP REORDER_PROP <<< "$trial"

    echo "======================================================================"
    echo "Running Trial with Hyperparameters:"
    echo "  Augmentation Method 1: ${METHOD_1}"
    echo "  Augmentation Method 2: ${METHOD_2}"
    echo "  Mask Proportion:       ${MASK_PROP}"
    echo "  Crop Proportion:       ${CROP_PROP}"
    echo "  Reorder Proportion:    ${REORDER_PROP}"
    echo "======================================================================"

    python -m multi_task.train \
        --data-dir "${DATA_DIR}" \
        --embeddings-dir "${EMBEDDINGS_DIR}" \
        --tasks churn propensity_category propensity_sku propensity_price \
        --log-name "${VERSION}" \
        --accelerator gpu \
        --devices ${DEVICE_ID} \
        --score-dir "${SCORES_DIR}" \
        --augmentation-method-hyperparameters "${METHOD_1}" "${METHOD_2}" "${MASK_PROP}" "${CROP_PROP}" "${REORDER_PROP}" \
        --disable-relevant-clients-check
    
    python -m training_pipeline.train \
        --data-dir /data/lyjiang/RecSys_Challenge_2025 \
        --embeddings-dir "${EMBEDDINGS_DIR}" \
        --tasks churn propensity_category propensity_sku \
        --log-name "${VERSION}" \
        --accelerator gpu \
        --devices ${DEVICE_ID} \
        --score-dir "${SCORES_DIR}" \
        --augmentation-method-hyperparameters "${METHOD_1}" "${METHOD_2}" "${MASK_PROP}" "${CROP_PROP}" "${REORDER_PROP}" \
        --disable-relevant-clients-check

done

echo "All trials completed."
