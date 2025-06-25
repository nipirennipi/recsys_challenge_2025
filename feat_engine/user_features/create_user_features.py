import argparse
import logging
from typing import List, Tuple, Dict
from pathlib import Path
from functools import reduce
from scipy.stats import entropy
import pandas as pd
import numpy as np
import time

from feat_engine.user_features.constants import (
    EventTypes,
    PropensityTasks,
)
from data_utils.data_dir import DataDir

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def load_events(data_dir: DataDir, event_type: EventTypes) -> pd.DataFrame:
    file_dir = data_dir.data_dir
    event_df = pd.read_parquet(file_dir / f"{event_type.value}.parquet")
    event_df["event_type"] = event_type.value
    event_df["timestamp"] = pd.to_datetime(event_df["timestamp"])
    return event_df


def load_item_features(data_dir: DataDir) -> pd.DataFrame:
    logger.info("Loading item statistic features")
    item_features = pd.read_parquet(data_dir.item_features_file)
    item_features = item_features.set_index(['sku', 'date'])

    return item_features


def load_cate_features(data_dir: DataDir) -> pd.DataFrame:
    logger.info("Loading category statistic features")
    cate_features = pd.read_parquet(data_dir.cate_features_file)

    return cate_features


def create_features(
    data_dir: DataDir, 
    target_dir: DataDir, 
    num_days: List[int]
) -> pd.DataFrame:
    logger.info("Creating features for user")
    
    # Load data from each event file
    events = {et.value: load_events(data_dir, et) for et in EventTypes}
    all_events = pd.concat([event_df[["client_id", "timestamp"]] for event_df in events.values()])
    START_TIME = all_events["timestamp"].min()
    END_TIME = all_events["timestamp"].max()
    MAX_TIME_SPAN = (END_TIME - START_TIME).days
    client_ids = pd.DataFrame({"client_id": all_events["client_id"].unique()})
    
    # Statistical features: sku propensity
    logger.info("Calculating sku propensity features")
    sku_targets = load_propensity_targets(
        target_dir, PropensityTasks.PROPENSITY_SKU
    )
    sku_stats = create_sku_propensity_features(
        events, client_ids, END_TIME, sku_targets
    )

    # Statistical features: category propensity
    logger.info("Calculating category propensity features")
    properties_dict = load_properties_dict(data_dir)
    category_targets = load_propensity_targets(
        target_dir, PropensityTasks.PROPENSITY_CATEGORY
    )
    category_stats = create_category_propensity_features(
        events, client_ids, END_TIME, properties_dict, category_targets
    )
    
    # Time features: days since first/last interaction for each event type
    logger.info("Calculating days since first/last interaction for each event type")
    user_first_last = []
    for event_name, event_df in events.items():
        temp = event_df.groupby("client_id")["timestamp"].agg(["min", "max"]).reset_index()
        temp[f"{event_name}_days_since_first"] = (END_TIME - temp["min"]).dt.days
        temp[f"{event_name}_days_since_last"] = (END_TIME - temp["max"]).dt.days
        temp = temp[["client_id", f"{event_name}_days_since_first", f"{event_name}_days_since_last"]]
        temp = client_ids.merge(temp, on="client_id", how="left").fillna(MAX_TIME_SPAN)
        user_first_last.append(temp)
    
    # Time features: Interaction interval statistics
    logger.info("Calculating interaction interval statistics")
    time_diff_stats = []
    for event_type in ["product_buy", "add_to_cart", "remove_from_cart"]:
        event_df = events[event_type].copy()
        event_df["timestamp"] = pd.to_datetime(event_df["timestamp"])
        event_df = event_df.sort_values(["client_id", "timestamp"])
        event_df["time_diff"] = event_df.groupby("client_id")["timestamp"].diff().dt.days
        stats = event_df.groupby("client_id")["time_diff"].agg(
            ["mean", "median", "max", "min"]
        ).reset_index().fillna(MAX_TIME_SPAN)
        stats.columns = ["client_id", f"{event_type}_time_diff_mean", f"{event_type}_time_diff_median", 
                         f"{event_type}_time_diff_max", f"{event_type}_time_diff_min"]
        stats = client_ids.merge(stats, on="client_id", how="left").fillna(MAX_TIME_SPAN)
        time_diff_stats.append(stats)

    # # Time features: total active days, weeks, months
    # logger.info("Calculating total active days, weeks, months")
    # temp_df = all_events[["client_id", "timestamp"]].copy()
    # temp_df['day'] = temp_df['timestamp'].dt.date
    # temp_df['week'] = temp_df['timestamp'].dt.to_period('W')
    # temp_df['month'] = temp_df['timestamp'].dt.to_period('M')
    # active_time = temp_df.groupby('client_id').agg(
    #     total_active_days=('day', 'nunique'),
    #     total_active_weeks=('week', 'nunique'),
    #     total_active_months=('month', 'nunique')
    # ).reset_index()
    # user_active_time = client_ids.merge(active_time, on="client_id", how="left").fillna(0)

    # # Trend features: ratio of recent activity to previous activity
    # logger.info("Calculating activity trend features")
    # activity_trends = []
    # for event_name, event_df in events.items():
    #     temp = event_df[["client_id", "timestamp"]].copy()
    #     temp["timestamp"] = pd.to_datetime(temp["timestamp"])
        
    #     # Define masks for recent and previous periods
    #     recent_mask = (END_TIME - temp["timestamp"]).dt.days <= 30
    #     previous_mask = ((END_TIME - temp["timestamp"]).dt.days > 30) & ((END_TIME - temp["timestamp"]).dt.days <= 60)
        
    #     # Calculate counts for recent and previous periods
    #     recent_counts = temp[recent_mask].groupby("client_id").size().reset_index(name=f"{event_name}_recent_count")
    #     previous_counts = temp[previous_mask].groupby("client_id").size().reset_index(name=f"{event_name}_previous_count")
        
    #     # Merge counts and calculate ratio
    #     trend_df = pd.merge(recent_counts, previous_counts, on="client_id", how="outer").fillna(0)
    #     trend_df[f"{event_name}_activity_trend_ratio"] = (
    #         trend_df[f"{event_name}_recent_count"] / trend_df[f"{event_name}_previous_count"]
    #     ).replace([np.nan], 0).replace([np.inf], 999)
    #     trend_df[f"{event_name}_activity_trend_diff"] = (
    #         trend_df[f"{event_name}_recent_count"] - trend_df[f"{event_name}_previous_count"]
    #     )
    #     trend_df.drop(
    #         columns=[f"{event_name}_recent_count", f"{event_name}_previous_count"], 
    #         inplace=True
    #     )
    #     trend_df = client_ids.merge(trend_df, on="client_id", how="left").fillna(0)
    #     activity_trends.append(trend_df)

    # Statistical features: event counts in time windows    
    logger.info("Calculating event counts in time windows")
    events_window_count = []
    for event_name, event_df in events.items():
        temp = event_df[["client_id", "timestamp"]].copy()
        temp["timestamp"] = pd.to_datetime(temp["timestamp"])
        masks = {w: (END_TIME - temp["timestamp"]).dt.days <= w for w in num_days}
        masks[0] = temp["timestamp"] <= END_TIME
        
        for window, mask in masks.items():
            col = f"{event_name}_count_{window}d" if window != 0 else f"{event_name}_count_all"
            count = temp[mask].groupby("client_id").size().reset_index(name=col)
            count = client_ids.merge(count, on="client_id", how="left").fillna(0)
            events_window_count.append(count)

    properties_dict = load_properties_dict(data_dir)
    # Statistical features: price propensity
    logger.info("Calculating price propensity features")
    price_stats = create_price_propensity_features(
        events, client_ids, END_TIME, properties_dict
    )
    
    # # Statistical features: user's preference for item popularity
    # logger.info("Calculating summary stats of interacted item popularities")
    # item_features = load_item_features(data_dir)
    # sku_pop_stats = create_sku_popularity_propensity_features(
    #     events, client_ids, item_features
    # )
    
    # Statistical features: difference with category average price
    logger.info("Calculating category average price difference features")
    cate_features = load_cate_features(data_dir)
    category_avg_price_diff_stats = create_category_avg_price_diff_features(
        events, client_ids, properties_dict, cate_features
    )

    logger.info("Combining all features into a single DataFrame")
    dfs = (
        [client_ids] + sku_stats + category_stats + user_first_last + time_diff_stats + 
        events_window_count + price_stats + [category_avg_price_diff_stats]
    )
    logger.info(f"Number of feature DataFrames: {len(dfs)}")
    features = reduce(lambda left, right: left.merge(right, on="client_id", how="left"), dfs)

    return features


def load_properties_dict(data_dir: DataDir) -> Dict[str, Dict[str, int]]:
    """
    Load properties from the properties file and construct a dictionary
    with sku as the key and its attributes as the value.
    """
    logger.info("Loading properties")
    properties = pd.read_parquet(data_dir.properties_file)
    
    properties_dict = (
        properties[["sku", "category", "price"]]
        .set_index("sku")
        .to_dict(orient='index')
    )
    return properties_dict


def load_propensity_targets(
    target_dir: DataDir, 
    task: PropensityTasks
) -> np.ndarray:
    propensity_targets = np.load(
        target_dir.target_dir / f"{task.value}.npy",
        allow_pickle=True,
    )
    return propensity_targets


def create_category_avg_price_diff_features(
    events: Dict[str, pd.DataFrame],
    client_ids: pd.DataFrame,
    properties_dict: Dict[int, Dict[str, any]],
    cate_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates the user's "premium/bargain" preference relative to the average price of categories.

    For each user interaction, it computes the difference between the item's price and the global
    average price of its category. Then, it aggregates these differences (mean, std, etc.) for each user.
    """
    feature_dfs_to_merge = [client_ids.copy()]

    if 'category' in cate_features.columns:
        cate_features = cate_features.set_index('category')

    props_df = pd.DataFrame.from_dict(properties_dict, orient='index')[['category', 'price']]

    for event_type in ["product_buy", "add_to_cart", "remove_from_cart"]:
        if event_type not in events or events[event_type].empty:
            continue

        event_df = events[event_type][['client_id', 'sku']].copy()
        events_with_props = pd.merge(event_df, props_df, left_on='sku', right_index=True, how='left')

        events_with_global_price = pd.merge(
            events_with_props,
            cate_features[['avg_price_per_category']],
            left_on='category',
            right_index=True,
            how='left'
        )

        events_with_global_price.dropna(subset=['price', 'avg_price_per_category'], inplace=True)
        if events_with_global_price.empty:
            continue
        events_with_global_price['price_diff'] = (
            events_with_global_price['price'] - events_with_global_price['avg_price_per_category']
        )

        user_stats = events_with_global_price.groupby('client_id')['price_diff'].agg(
            ['mean', 'std', 'min', 'max', 'median']
        ).reset_index()

        stat_cols = {
            'mean': f'{event_type}_category_avg_price_diff_mean',
            'std': f'{event_type}_category_avg_price_diff_std',
            'min': f'{event_type}_category_avg_price_diff_min',
            'max': f'{event_type}_category_avg_price_diff_max',
            'median': f'{event_type}_category_avg_price_diff_median'
        }
        user_stats.rename(columns=stat_cols, inplace=True)
        user_stats[f'{event_type}_category_avg_price_diff_std'] = \
            user_stats[f'{event_type}_category_avg_price_diff_std'].fillna(0)
        feature_dfs_to_merge.append(user_stats)

    if len(feature_dfs_to_merge) == 1:
        return client_ids.copy()
        
    final_features = reduce(
        lambda left, right: pd.merge(left, right, on='client_id', how='left'), 
        feature_dfs_to_merge
    )
    return final_features


def create_sku_popularity_propensity_features(
    events: Dict[str, pd.DataFrame],
    client_ids: pd.DataFrame,
    item_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculates summary statistics (mean, std, min, max, median) of item popularity vectors 
    for items a user has interacted with.
    """
    # This list will hold the feature DataFrames for each event type.
    sku_pop_stats = []
    POPULARITY_METRIC_COL = 'product_buy_popularity'

    # Loop through each interaction type to generate features.
    for event_type in ["product_buy", "add_to_cart", "remove_from_cart"]:
        if event_type not in events or events[event_type].empty:
            sku_pop_stats.append(client_ids.copy())
            continue

        # 1. Get user interactions for the current event type.
        event_df = events[event_type][['client_id', 'timestamp', 'sku']].copy()
        event_df['date'] = pd.to_datetime(event_df['timestamp']).dt.floor('D')
        
        # 2. Point-in-Time Join to get item popularity at the time of interaction.
        merged_df = pd.merge(
            event_df,
            item_features, 
            on=['sku', 'date'],
            how='left'
        )
        
        # Drop rows where item features couldn't be found.
        merged_df.dropna(subset=[POPULARITY_METRIC_COL], inplace=True)

        if merged_df.empty:
            sku_pop_stats.append(client_ids.copy())
            continue

        # 3. Aggregate by client_id to calculate vector statistics.
        def agg_vector_stats(vectors: pd.Series):
            matrix = np.stack(vectors.values)
            # For std, if only one item, variance is 0.
            std_dev = np.std(matrix, axis=0) if matrix.shape[0] > 1 else np.zeros_like(matrix[0])
            
            return pd.Series({
                'mean': np.mean(matrix, axis=0),
                'std': std_dev,
                'min': np.min(matrix, axis=0),
                'max': np.max(matrix, axis=0),
                'median': np.median(matrix, axis=0)
            })

        # Apply the aggregation. .unstack() pivots the stats into columns.
        user_pop_stats = merged_df.groupby('client_id')[POPULARITY_METRIC_COL].apply(agg_vector_stats).unstack()
        user_pop_stats.columns = [f"{event_type}_sku_popularity_{col}" for col in user_pop_stats.columns]
        
        # Split the array columns into separate columns.
        expanded_dfs = []
        for col_name in user_pop_stats.columns:
            expanded_df = pd.DataFrame(
                user_pop_stats[col_name].tolist(),
                index=user_pop_stats.index
            )
            expanded_df.columns = [f"{col_name}_{i}" for i in range(expanded_df.shape[1])]
            expanded_dfs.append(expanded_df)

        user_pop_stats = pd.concat(expanded_dfs, axis=1)
        
        # Merge with all client_ids to ensure every user is included.
        final_stats_df = client_ids.merge(user_pop_stats, on="client_id", how="left")
        sku_pop_stats.append(final_stats_df)
        
    sku_pop_stats = reduce(lambda left, right: left.merge(right, on='client_id', how='left'), sku_pop_stats)
    return sku_pop_stats


def create_sku_propensity_features(
    events: Dict[str, pd.DataFrame],
    client_ids: pd.DataFrame,
    end_time: pd.Timestamp,
    sku_targets: np.ndarray,
) -> List:
    sku_stats = []

    for event_type in ["product_buy", "add_to_cart", "remove_from_cart"]:
        df = events[event_type].copy()
        target_df = df[df["sku"].isin(sku_targets)]
        
        # 1. SKU Statistics
        for time_window in ["7d", "30d", "all"]:
            if time_window == "7d":
                target_window_df = target_df[target_df["timestamp"] >= (end_time - pd.Timedelta(days=7))]
            elif time_window == "30d":
                target_window_df = target_df[target_df["timestamp"] >= (end_time - pd.Timedelta(days=30))]
            else:
                target_window_df = target_df

            sku_counts = target_window_df.groupby("client_id").size().reset_index(
                name=f"{event_type}_target_sku_count_{time_window}"
            )
            sku_counts = client_ids.merge(sku_counts, on="client_id", how="left").fillna(0)
            sku_stats.append(sku_counts)
        
        # 2. SKU Diversity
        unique_sku = (
            target_df.groupby("client_id")["sku"].nunique().
            reset_index(name=f"{event_type}_unique_target_sku")
        )
        
        diversity_df = client_ids.merge(unique_sku, on="client_id", how="left").fillna(0)
        sku_stats.append(diversity_df)

        # 3. Target SKU Proportion
        target_counts = target_df.groupby("client_id").size().reset_index(name="target_count")
        total_counts = df.groupby("client_id").size().reset_index(name="total_count")
        
        proportion_df = pd.merge(target_counts, total_counts, on="client_id", how="right").fillna(0)
        proportion_df[f"{event_type}_target_sku_proportion"] = (
            proportion_df["target_count"] / proportion_df["total_count"]
        ).fillna(0)
        
        proportion_df = proportion_df[["client_id", f"{event_type}_target_sku_proportion"]]
        proportion_df = client_ids.merge(proportion_df, on="client_id", how="left").fillna(0)
        sku_stats.append(proportion_df)
        
    return sku_stats


def create_category_propensity_features(
    events: Dict[str, pd.DataFrame],
    client_ids: pd.DataFrame,
    end_time: pd.Timestamp,
    properties_dict: Dict[str, Dict[str, int]],
    category_targets: np.ndarray,
) -> List:
    category_stats = []

    for event_type in ["product_buy", "add_to_cart", "remove_from_cart"]:
        df = events[event_type].copy()
        df["category"] = df["sku"].map(lambda sku: properties_dict[sku]["category"])
        target_df = df[df["category"].isin(category_targets)]
        
        # 1. Category Statistics
        for time_window in ["7d", "30d", "all"]:
            if time_window == "7d":
                target_window_df = target_df[target_df["timestamp"] >= (end_time - pd.Timedelta(days=7))]
            elif time_window == "30d":
                target_window_df = target_df[target_df["timestamp"] >= (end_time - pd.Timedelta(days=30))]
            else:
                target_window_df = target_df
            
            category_counts = target_window_df.groupby("client_id").size().reset_index(
                name=f"{event_type}_target_category_count_{time_window}"
            )
            category_counts = client_ids.merge(category_counts, on="client_id", how="left").fillna(0)
            category_stats.append(category_counts)
        
        # 2. Category Diversity    
        unique_category = (
            target_df.groupby("client_id")["category"].nunique().
            reset_index(name=f"{event_type}_unique_target_categories")
        )
        
        diversity_df = client_ids.merge(unique_category, on="client_id", how="left").fillna(0)
        category_stats.append(diversity_df)

        # 3. Target Category Proportion
        target_counts = target_df.groupby("client_id").size().reset_index(name="target_count")
        total_counts = df.groupby("client_id").size().reset_index(name="total_count")
        
        proportion_df = pd.merge(target_counts, total_counts, on="client_id", how="right").fillna(0)
        proportion_df[f"{event_type}_target_category_proportion"] = (
            proportion_df["target_count"] / proportion_df["total_count"]
        ).fillna(0)
        
        proportion_df = proportion_df[["client_id", f"{event_type}_target_category_proportion"]]
        proportion_df = client_ids.merge(proportion_df, on="client_id", how="left").fillna(0)
        category_stats.append(proportion_df)
        
    return category_stats


def create_price_propensity_features(
    events: Dict[str, pd.DataFrame],
    client_ids: pd.DataFrame,
    end_time: pd.Timestamp,
    properties_dict: Dict[str, Dict[str, int]]
) -> List:
    price_stats  = []
    bins = [-1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 100]  # Adjust 100 if max price bucket is 99
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 10 tiers

    for event_type in ["product_buy", "add_to_cart", "remove_from_cart"]:
        df = events[event_type].copy()
        df["price"] = df["sku"].map(lambda sku: properties_dict[sku]["price"])
        df["price_tier"] = pd.cut(df["price"], bins=bins, labels=labels, right=True).astype(int)

        # 1. Price Tier Statistics
        for time_window in ["30d", "all"]:
            if time_window == "30d":
                window_df = df[df["timestamp"] >= (end_time - pd.Timedelta(days=30))]
            else:
                window_df = df
            
            tier_counts = window_df.groupby(
                ["client_id", "price_tier"]
            ).size().unstack(fill_value=0)
            tier_counts.columns = [
                f"{event_type}_price_tier_{col}_{time_window}" for col in tier_counts.columns
            ]
            tier_counts = tier_counts.reset_index()
            tier_counts = client_ids.merge(tier_counts, on="client_id", how="left").fillna(0)
            price_stats.append(tier_counts)

        # 2. User Price Bucket Summary Statistics
        price_stats_df = df.groupby("client_id")["price"].agg(
            ["mean", "median", "std", "max", "min"]
        ).reset_index()
        price_stats_df.columns = ["client_id", f"{event_type}_price_mean",
            f"{event_type}_price_median", f"{event_type}_price_std",
            f"{event_type}_price_max", f"{event_type}_price_min"
        ]
        price_stats_df = client_ids.merge(price_stats_df, on="client_id", how="left")
        price_stats.append(price_stats_df)
        
        # 3. Price Diversity
        tier_distribution = ( 
            df.groupby(["client_id", "price_tier"]).size()
            .groupby("client_id")
            .apply(lambda x: entropy(x / x.sum(), base=2) if x.sum() > 0 else 0)
            .reset_index(name=f"{event_type}_price_entropy")
        )
        unique_tiers = (
            df.groupby("client_id")["price_tier"].nunique()
            .reset_index(name=f"{event_type}_unique_price_tiers")
        )
        
        diversity_df = pd.merge(tier_distribution, unique_tiers, on="client_id", how="outer").fillna(0)
        diversity_df = client_ids.merge(diversity_df, on="client_id", how="left").fillna(0)
        price_stats.append(diversity_df)

    return price_stats


def save_features(features_dir: Path, features: pd.DataFrame) -> None:
    logger.info(f"Saving features for {len(features)} users")
    features_dir.mkdir(parents=True, exist_ok=True)
    output_file = features_dir / "user_features.parquet"
    features.to_parquet(output_file, index=False)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with input and target data – produced by data_utils.split_data",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Directory where to store target data (e.g., propensity_category)",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        required=True,
        help="Directory where to store generated features",
    )
    parser.add_argument(
        "--num-days",
        nargs="*",
        type=int,
        default=[1, 7, 30],
        help="Numer of days to compute features",
    )
    return parser


def main(params):
    data_dir = DataDir(Path(params.data_dir))
    target_dir = DataDir(Path(params.target_dir))
    features_dir = Path(params.features_dir)

    features = create_features(
        data_dir=data_dir,
        target_dir=target_dir,
        num_days=params.num_days,
    )
    save_features(
        features_dir=features_dir, 
        features=features
    )


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params=params)

