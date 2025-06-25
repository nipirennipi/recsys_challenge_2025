# Solution for Universal Behavioral Modeling Data Challenge

This document outlines a comprehensive solution for the Universal Behavioral Modeling Data Challenge. The core of this solution is a sophisticated two-part approach that combines deep feature engineering with an advanced modeling architecture to generate robust, generalizable Universal Behavioral Profiles.

## 1. Solution Philosophy

The primary goal is to create user representations that are not only predictive for the open tasks but are also highly generalizable for the hidden tasks. This is achieved by focusing on two key principles:

1. **Rich, Multi-faceted Feature Representation:** Instead of relying on a single source of information, we model user behavior from multiple angles: their interaction with item attributes (price, category), their behavioral dynamics (recency, frequency, trends), and their latent "taste" profiles.
2. **Advanced Sequence Modeling with Self-Supervision:** We use a powerful, unified Transformer model to learn deep contextual patterns from raw, heterogeneous event sequences. This is augmented with a contrastive learning framework (SimCLR-style) to ensure the learned representations are robust and capture the essential semantics of user behavior.

## 2. Feature Engineering

We engineer features at two levels: **Item Features** which enrich the input sequences, and **User Features** which provide a static, aggregated summary of user behavior.

### 2.1. Item & Event-Level Features (Input for Sequence Model)

For each event in a user's sequence, we create a rich representation based on the following:

- **Core Attribute Embeddings:**
    - **SKU & Category:** Learnable embeddings are created for `sku` and `category` IDs to capture their unique identities and relationships.
    - **Price:** The `price` bucket ID is treated as a categorical variable and embedded to learn non-linear price effects.
    - **Name:** The provided 16-dimensional quantized embedding for the item `name` is used directly.
- **Point-in-Time Popularity Features:**
    - A feature vector named `features` is pre-calculated for each item for each day.
    - This vector includes the item's interaction counts (purchases, add-to-carts, removals) over various sliding windows (3, 7, 30, 60 days) and in total (cumulative), all `log1p` transformed. This provides the sequence model with dynamic information about an item's trend and popularity at the exact moment of user interaction, avoiding any data leakage from the future.
- **Temporal Features:**
    - **Cyclical Time:** The hour of the day and day of the week are encoded using sine/cosine transformations. A binary `is_weekend` flag is also included.
    - **Time Decay:** A recency weight, w_i=f(textEND_TIME−t_i), is calculated for each event and included as an input feature to explicitly inform the model about the event's freshness.
- **Target-Awareness Features:**
    - A binary flag indicating whether the item in the sequence belongs to the set of 100 target SKUs/categories is included.

### 2.2. User-Level Features (Static User Profile)

This is a set of pre-calculated, aggregated features that are concatenated with the sequence model's output.

- **Time-based Activity & Recency:**
    - `days_since_first_event` / `days_since_last_event` for each of the 5 event types.
    - `mean/median/max/min` time interval between consecutive interactions (purchase, add-to-cart, etc.).
    - Total number of unique active days, weeks, and months.
    - **Activity Trend:** The ratio and difference of activity counts in the last 30 days versus the prior 30-day period (days 31-60 ago), calculated for each event type. This is a strong predictor for the `churn` task.
- **Statistical Summaries:**
    - **Event Counts:** `log1p` transformed counts for each event type over various windows (3, 7, 30, 60 days, and cumulative).
    - Price Preference:
        - Interaction counts within 10 coarse-grained price tiers (cumulative and last 30 days).
        - `mean/median/std/max/min` of the price bucket IDs for items the user has purchased/added-to-cart.
        - Price diversity metrics like the entropy of interacted price tiers and the number of unique price tiers interacted with.
        - **Premium/Bargain Preference:** A feature calculated as the average difference between an item's price and the global average price of its category, over all user purchases. This captures if a user tends to buy higher-end or lower-end products within categories.
    - Category & SKU Propensity Summaries:
        - Total interaction counts with the 100 target categories/SKUs (cumulative and recent windows).
        - Diversity (number of unique target entities interacted with).
        - Proportion of user's total activity that is focused on the target entities.

## 3. Model Architecture

The modeling pipeline is designed to create a powerful, unified user representation by combining sequence dynamics and static features.

1. **Input & Embedding:**
    - All 5 event types (`product_buy`, `add_to_cart`, `remove_from_cart`, `page_visit`, `search_query`) are organized into a single chronological sequence for each user.
    - Each event is encoded based on its type (e.g., an item event is represented by its rich feature vector; a search event by its query vector).
    - A `Linear` layer projects these heterogeneous event representations into a unified dimension.
    - An `event_type` embedding and a positional embedding are added to the projected vector.
2. **Behavioral Sequence Modeling:**
    - A **Transformer Encoder** is used as the unified sequence model. It processes the entire sequence of mixed event types to learn deep contextual relationships and output a powerful, holistic sequence representation.
3. **Feature Fusion & Interaction:**
    - The output from the Transformer (the sequence representation) is **concatenated** with the static **User Feature** vector (from section 2.2).
    - This combined vector is then fed into a **Deep & Cross Network (DCN V2)** to explicitly and automatically model high-order feature interactions between the dynamic sequence representation and the static user profile.
4. **Output & Multi-Task Training:**
    - The final output of the DCN V2 serves as the **Universal Behavioral Profile**.
    - This single profile is fed into four separate, simple MLP "towers," one for each of the open tasks (`Churn`, `Categories Propensity`, `Product Propensity`, `Price Propensity`).
    - The entire network is trained end-to-end using a summed loss from all four tasks.

## 4. Training Strategy

To enhance the generalization of the learned profiles, we employ a **contrastive learning** objective alongside the multi-task objective.

- **Augmentation:** For each user sequence, we generate two different augmented "views" using a random choice of `mask`, `crop`, or `reorder`.
- **Contrastive Loss (SimCLR-style):** The model is trained to maximize the similarity (e.g., cosine similarity) between the user profiles generated from the two augmented views of the same sequence, while minimizing the similarity to profiles from other users in the same batch.
- **Final Loss:** The final loss is a weighted sum of the multi-task loss and the contrastive loss, where the contrastive loss acts as a powerful regularizer, forcing the model to learn what is essential about a user's behavior.
