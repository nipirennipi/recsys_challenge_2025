# Baseline Solution for the Universal Behavioral Modeling Data Challenge

Here, we present a baseline solution for the data challenge. For more information about the challenge and the dataset, see the README in the root directory.  
In order to run training with generated representations, the data needs to be split into input and target chunks, and embeddings should be generated based on the input event files.  

**IMPORTANT! To split the data, use the `data_utils.split_data` script. Instructions on how to run the script are provided in the README in the root directory, under the [Data Splitting](https://github.com/Synerise/recsys2025#data-splitting) section.**

## Baseline embedding approach: feature aggregation

This baseline follows feature engineering practices present in real-life solutions in the behavioral modeling domain. It constructs simple aggregated features of two types: statistical and query features.

### Statistical features

Statistical features represent categories, prices, and products users were interested in. They correspond to past interaction counts for each user, for example, *how many times the user bought a certain product X during the last 30 days*.

For each user, we consider a set of time windows, e.g., 1 day, 1 week, and 1 month and aggregate the number of events grouped by column values.  
**For example:**  
Last month the user added the following products to their cart: 3 of category X, 2 of category Y; 4 products were in the 3rd price bucket, and 1 in the 4th price bucket.  
Last month the user bought 1 product of category X in the 3rd price bucket.  
The same statistics are computed for other time windows and event types.

**Note**  
Given the significant number of categories, we may only use a subset of the values to compute features as the full set of values for all event types and time windows will result in very large vectors. For feature computation, we limit categories, price buckets, or products to 10 most popular values. 

### Query features

Since `search_query` event type contains integer vectors obtained by quantizing text embeddings of users' search queries, we apply a different method to extract valuable information for this event type. For each user, we construct new features by taking the average of integer vectors corresponding to user's queries.

### Baseline pipeline overview

User features are extracted from the recorded raw event data. First, features are calculated separately for each type of event. Then they are merged, leading to information-rich user representations (Universal Behavioral Profiles). In the end, in the `embedding_dir` defined by a user the `np.ndarray` with `client_ids` and the corresponding `np.ndarray` with `embeddings` are saved in `.npy` files, which corresponds to the submission format of the competition.

## Create competition entry embeddings

`create_embeddings.py` creates an exemplar competition submission, based on selected event types. The script generates features for each event type and merges them to create user representations, which serve as input embeddings for model training and validation. The script uses a default set of columns for statistical features:  
 - `category` and `price` for the event types: `product_buy`, `add_to_cart`, `remove_from_cart` 
 - `url` for `page_visit`
 - `query` for `search_query`

### Arguments

- `--data-dir`: Directory with train and target data – produced by `data_utils.split_data`
- `--embeddings-dir`: Directory to save the generated embeddings in the competition-compliant format
- `--num-days`:  A list of time windows (in days) for generating features. Each time window will produce a different set of features aggregating events from the defined period

  **For example:**
  Providing the following list [1, 7, 30] in the `--num-days` parameter will result in three different features created for a single column value, e.g., the number of products of a given category bought in the last 1, 7, and 30 days.
- `--top-n`: Number of top column values to consider in feature generation

### Output

Client ids and corresponding embeddings are saved in `--embeddings-dir` in two files: `client_ids.npy` and `embeddings.npy`.

### Running the script

```bash
python -m baseline.aggregated_features_baseline.create_embeddings --data-dir <splitted-challenge-data-dir> --embeddings-dir <your-embeddings-dir>
```
