# # Create item features
# python -m feat_engine.item_features.create_item_features \
#     --data-dir "/data/lyjiang/RecSys_Challenge_2025" \
#     --features-dir "/data/lyjiang/RecSys_Challenge_2025" \
#     --num-days 3 7 30 60


# Create user features
DATA_DIR="/data/lyjiang/RecSys_Challenge_2025"

python -m feat_engine.user_features.create_user_features \
    --data-dir "${DATA_DIR}" \
    --features-dir "${DATA_DIR}" \
    --num-days 3 7 30 60
