VERSION=multi_task
STATE=online
DEVICE_ID=5
EMBEDDINGS_DIR="/data/lyjiang/RecSys_Challenge_2025/submit/${VERSION}/${STATE}"
DATA_DIR="/data/lyjiang/RecSys_Challenge_2025"

mkdir -p "${EMBEDDINGS_DIR}"

AUGMENTATION_METHOD_1="mask"  # mask, crop and reorder
AUGMENTATION_METHOD_2="reorder"
MASK_PROPORTION=0.3 # Proportion of items to mask in the sequence
CROP_PROPORTION=0.8  # Proportion of items to crop in the sequence
REORDER_PROPORTION=0.3  # Proportion of items to reorder in the sequence

python -m multi_task.train \
    --data-dir "${DATA_DIR}" \
    --embeddings-dir "${EMBEDDINGS_DIR}" \
    --tasks churn propensity_category propensity_sku propensity_price \
    --log-name "${VERSION}" \
    --accelerator gpu \
    --devices ${DEVICE_ID} \
    --augmentation-method-hyperparameters "${AUGMENTATION_METHOD_1}" "${AUGMENTATION_METHOD_2}" "${MASK_PROPORTION}" "${CROP_PROPORTION}" "${REORDER_PROPORTION}" \
    --disable-relevant-clients-check
