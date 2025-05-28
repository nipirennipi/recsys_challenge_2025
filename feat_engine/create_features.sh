# Create item features
python -m feat_engine.item_features.create_item_features \
    --data-dir "/data/lyjiang/RecSys_Challenge_2025" \
    --features-dir "/data/lyjiang/RecSys_Challenge_2025" \
    --num-days 3 7 30 60


# STATE=offline
# FEATURES_DIR="/data/lyjiang/RecSys_Challenge_2025/${STATE}"

# if [ "${STATE}" = "offline" ]; then
#     DATA_DIR="/data/lyjiang/RecSys_Challenge_2025/input"
# elif [ "${STATE}" = "online" ]; then
#     DATA_DIR="/data/lyjiang/RecSys_Challenge_2025"
# else
#     echo "Error: Invalid STATE value '${STATE}'" >&2
#     exit 1
# fi
