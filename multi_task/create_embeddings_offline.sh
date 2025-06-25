VERSION=multi_task
STATE=offline
DEVICE_ID=6
DATA_DIR="/data/lyjiang/RecSys_Challenge_2025/input"

FNAME=std
SCORES_DIR="./${VERSION}/score/scores_${FNAME}.txt"
EMBEDDINGS_DIR="/data/lyjiang/RecSys_Challenge_2025/submit/${VERSION}/${STATE}/${FNAME}"

TRIALS=(
    "mask mask 0.1"
    # "mask mask 0.2"
    # "mask mask 0.3"
    # "mask mask 0.4"
    # "mask mask 0.5"
    # "mask mask 0.6"
)

mkdir -p "${EMBEDDINGS_DIR}"
mkdir -p "./${VERSION}/score"

for trial in "${TRIALS[@]}"; do
    read -r METHOD_1 METHOD_2 MASK_PROP <<< "$trial"

    # echo "======================================================================"
    # echo "Running Trial with Hyperparameters:"
    # echo "  Augmentation Method 1: ${METHOD_1}"
    # echo "  Augmentation Method 2: ${METHOD_2}"
    # echo "  Mask Proportion:       ${MASK_PROP}"
    # echo "  Crop Proportion:       ${CROP_PROP}"
    # echo "  Reorder Proportion:    ${REORDER_PROP}"
    # echo "======================================================================"

    python -m multi_task.train \
        --data-dir "${DATA_DIR}" \
        --embeddings-dir "${EMBEDDINGS_DIR}" \
        --tasks churn propensity_category propensity_sku propensity_price \
        --log-name "${VERSION}" \
        --accelerator gpu \
        --devices ${DEVICE_ID} \
        --score-dir "${SCORES_DIR}" \
        --disable-relevant-clients-check \
        # --augmentation-method-hyperparameters "${METHOD_1}" "${METHOD_2}" "${MASK_PROP}" \
    
    python -m training_pipeline.train \
        --data-dir /data/lyjiang/RecSys_Challenge_2025 \
        --embeddings-dir "${EMBEDDINGS_DIR}" \
        --tasks churn propensity_category propensity_sku \
        --log-name "${VERSION}" \
        --accelerator gpu \
        --devices ${DEVICE_ID} \
        --score-dir "${SCORES_DIR}" \
        --disable-relevant-clients-check \
        # --augmentation-method-hyperparameters "${METHOD_1}" "${METHOD_2}" "${MASK_PROP}" \

done

echo "All trials completed."
