VERSION=baseline
mkdir -p "./${VERSION}/score"

python -m training_pipeline.train \
    --data-dir /data/lyjiang/RecSys_Challenge_2025 \
    --embeddings-dir "/data/lyjiang/RecSys_Challenge_2025/submit/${VERSION}" \
    --tasks churn propensity_category propensity_sku \
    --log-name "${VERSION}" \
    --accelerator gpu \
    --devices 0 \
    --score-dir "./${VERSION}/score" \
    --disable-relevant-clients-check