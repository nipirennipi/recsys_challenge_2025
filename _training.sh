DEVICE_ID=${1:-0}

VERSION=multi_task
mkdir -p "./${VERSION}/score"

python -m training_pipeline.train \
    --data-dir /data/lyjiang/RecSys_Challenge_2025 \
    --embeddings-dir "/data/lyjiang/RecSys_Challenge_2025/submit/${VERSION}/offline" \
    --tasks churn propensity_category propensity_sku \
    --log-name "${VERSION}" \
    --accelerator gpu \
    --devices ${DEVICE_ID} \
    --disable-relevant-clients-check