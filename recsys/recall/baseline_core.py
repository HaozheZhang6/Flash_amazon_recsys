import torch
import pandas as pd
import numpy as np
import os
from typing import List
from datetime import datetime
import time

from recsys.recall.two_towers.eval import TwoTowerEvaluator
from recsys.recall.ann.ann import AnnoyRecall
from recsys.utils.training_utils import select_device
from recsys.data.embeding import convert_query
from recsys.utils.log import save_recall_log


PRODUCT_EMBEDDING_CSV = 'data/product_150k.csv'
PRODUCT_META_PARQUET = 'data/shopping_queries_dataset_products.parquet'
MODEL_PATH = "models/two_towers/final_model.pt"
INPUT_DIM_Q = 32
INPUT_DIM_P = 160
HIDDEN_Q = 64
HIDDEN_P = 64
EMBED = 32
ANN_QUERY_DIM = 32
ANN_PRODUCT_DIM = 160
RESULTS_DIR = "results"


def main(querys: List[str], k: int) -> pd.DataFrame:
    '''
    Given a list of queries, find the top k products using a hybrid approach.
    Set 1: Use two-tower model to calculate similarity and select top k/2.
    Set 2: Use ANN (Annoy) to find top k nearest products for the query vector.
    Combine, deduplicate, and return the top k unique products with titles.

    Args:
        querys: A list of query strings.
        k: The target number of products to recall for each query.

    Returns:
        A Pandas DataFrame with columns ['query', 'pid', 'product_title'].
        Returns an empty DataFrame if errors occur during data loading or model execution.
    '''
    start_time = time.time()
    data_loading_start = time.time()

    print(f"Starting recall for {len(querys)} queries with k={k}")

    # --- 1. Load Product Data ---
    print("Loading product data...")
    try:
        # Check if files exist before reading
        if not os.path.exists(PRODUCT_EMBEDDING_CSV) or not os.path.exists(PRODUCT_META_PARQUET):
            raise FileNotFoundError(f"Product data file(s) not found. Checked: {PRODUCT_EMBEDDING_CSV}, {PRODUCT_META_PARQUET}")

        df_product_emb_vectors = pd.read_csv(PRODUCT_EMBEDDING_CSV)
        df_product_meta = pd.read_parquet(PRODUCT_META_PARQUET)[['product_id', 'product_title']].drop_duplicates()
        df_product_embedding = pd.merge(df_product_emb_vectors, df_product_meta, on='product_id', how='inner').reset_index(drop=True)

        # Ensure 'pid' exists - create if necessary (using index)
        if 'pid' not in df_product_embedding.columns:
                df_product_embedding['pid'] = df_product_embedding.index
        else:
                # Ensure pid is integer type for consistency
                df_product_embedding['pid'] = df_product_embedding['pid'].astype(int)

        print(f"Loaded {len(df_product_embedding)} products.")
        # Create pid -> title mapping
        mp_product_dict = pd.Series(df_product_embedding.product_title.values, index=df_product_embedding.pid).to_dict()

        # Check if product embedding columns exist
        product_cols = [f'p{i}' for i in range(INPUT_DIM_P)]
        if not all(col in df_product_embedding.columns for col in product_cols):
            missing_cols = [col for col in product_cols if col not in df_product_embedding.columns]
            raise ValueError(f"Product embedding columns missing in product data: {missing_cols}")

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading or processing product data: {e}")
        return pd.DataFrame() # Return empty DataFrame on error

    data_loading_time = time.time() - data_loading_start
    computation_start = time.time()

    # --- 2. Convert Queries to Embeddings ---
    print("Converting queries to embeddings...")
    try:
        query_embedding = convert_query(querys) # Shape: (num_queries, query_dim)
        if query_embedding.shape[1] != INPUT_DIM_Q:
             raise ValueError(f"Query embedding dimension mismatch. Expected {INPUT_DIM_Q}, Got {query_embedding.shape[1]}")
        query_cols = [f'q{i}' for i in range(query_embedding.shape[1])]
        df_query_embedding = pd.DataFrame(query_embedding, columns=query_cols)
        df_query_embedding['query'] = querys
        # qid might be useful if ANN needed it, but not for search by vector
        # df_query_embedding['qid'] = df_query_embedding.index
        print("Query embeddings prepared.")
    except Exception as e:
        print(f"Error converting queries to embeddings: {e}")
        return pd.DataFrame()


    # --- 3. Two-Tower Model Recall (Top k/2) ---
    tt_start = time.time()
    print("Performing Two-Tower recall...")
    all_tt_pids = [[] for _ in querys] # Initialize with empty lists
    try:
        device = select_device()
        evaluator = TwoTowerEvaluator(
            model_path=MODEL_PATH,
            input_dim_q=INPUT_DIM_Q,
            input_dim_p=INPUT_DIM_P,
            hidden_dim_q=HIDDEN_Q,
            hidden_dim_p=HIDDEN_P,
            embed_dim=EMBED,
            device=device,
        )

        q_batch = torch.tensor(df_query_embedding[query_cols].values, dtype=torch.float32)
        p_batch = torch.tensor(df_product_embedding[product_cols].values, dtype=torch.float32)

        scores = evaluator.evaluate(q_batch, p_batch) # Shape: (num_queries, num_products)
        print("Two-Tower scores calculated.")

        num_products_tt = k // 2
        if num_products_tt > 0:
            k_safe_tt = min(num_products_tt, scores.shape[1])
            if k_safe_tt < num_products_tt:
                print(f"Warning: Requested top {num_products_tt} from Two-Tower, but only {scores.shape[1]} products available. Using {k_safe_tt}.")

            top_k_half_indices_tt = torch.topk(scores, k_safe_tt, dim=1).indices.cpu().numpy() # Shape: (num_queries, k_safe_tt)

            # Map indices back to pids
            product_pids_array = df_product_embedding['pid'].values # Get pids as numpy array for indexing
            for i in range(len(querys)):
                query_top_indices = top_k_half_indices_tt[i]
                # Ensure indices are valid before accessing product_pids_array
                valid_indices = query_top_indices[query_top_indices < len(product_pids_array)]
                all_tt_pids[i] = product_pids_array[valid_indices].tolist()

            print(f"Top {k_safe_tt} products from Two-Tower identified for each query.")
        else:
            print("k/2 is 0, skipping Two-Tower recall.")

    except FileNotFoundError:
         print(f"Error: Model file not found at {MODEL_PATH}. Skipping Two-Tower recall.")
    except Exception as e:
        print(f"Error during Two-Tower recall: {e}. Skipping.")

    tt_time = time.time() - tt_start


    # --- 4. ANN Recall (Based on Two-Tower Results) ---
    ann_start = time.time()
    print("Performing ANN recall based on Two-Tower results...")
    all_ann_pids = [[] for _ in querys] # Initialize with empty lists
    try:
        # Use dimensions inferred or defined earlier
        ann_recall = AnnoyRecall(
            query_tower_input_dim=ANN_QUERY_DIM,
            product_tower_input_dim=ANN_PRODUCT_DIM
        )
        ann_recall._ensure_product_index_loaded()
        print("Annoy product index loaded/ensured.")

        # Set k_safe_ann to k//2 to match the two-tower results
        k_safe_ann = k // 2
        print(f"Will retrieve up to {k_safe_ann} neighbors for each seed product.")

        if k_safe_ann > 0:
            product_vectors_map = df_product_embedding.set_index('pid')[product_cols].T.to_dict('list')

            for i in range(len(querys)):
                seed_pids = all_tt_pids[i] # Get product IDs from Two-Tower results
                query_ann_neighbors = set()

                if not seed_pids:
                    print(f"No seed products from Two-Tower for query '{querys[i]}', skipping ANN step for this query.")
                    continue

                for seed_pid in seed_pids:
                    # Ensure pid is int for lookup
                    seed_pid_int = int(seed_pid)
                    if seed_pid_int in product_vectors_map:
                        product_vec = product_vectors_map[seed_pid_int]
                        try:
                            # Search product index using the product vector
                            neighbor_pids = ann_recall.get_product_neighbors_by_vector(product_vec, k=k_safe_ann)
                            query_ann_neighbors.update(neighbor_pids)
                        except (RuntimeError, Exception) as e:
                            print(f"Error during ANN search for seed product PID {seed_pid_int} (query '{querys[i]}'): {e}")
                    else:
                        print(f"Warning: Seed product PID {seed_pid_int} from Two-Tower not found in product embedding map. Skipping ANN search for this seed.")

                # Remove the original seed PIDs from the neighbors if present
                query_ann_neighbors.difference_update(set(seed_pids))
                # Take only k//2 products from ANN results
                all_ann_pids[i] = list(query_ann_neighbors)[:k_safe_ann]

            print(f"ANN search completed, finding up to {k_safe_ann} neighbors for each seed product.")
        else:
            print("k is too small, skipping ANN recall.")

    except FileNotFoundError as e:
         print(f"Error related to Annoy index files: {e}. Skipping ANN recall.")
         all_ann_pids = [[] for _ in querys]
    except (RuntimeError, Exception) as e:
        print(f"Error initializing or using AnnoyRecall: {e}. Skipping ANN recall.")
        all_ann_pids = [[] for _ in querys]

    ann_time = time.time() - ann_start


    # --- 5. Combine Results ---
    print("Combining and finalizing results...")
    final_results = []
    for i, query in enumerate(querys):
        # Use sets for efficient union and deduplication
        tt_pids = set(all_tt_pids[i])
        ann_pids = set(all_ann_pids[i])

        print(f"\nQuery '{query}':")
        print(f"Two-tower products: {len(tt_pids)}")
        print(f"ANN products: {len(ann_pids)}")

        # Track which products came from which source
        tt_products = {pid: 'two_tower.vector' for pid in tt_pids}
        ann_products = {pid: 'ann.product.tree' for pid in ann_pids}
        
        # Combine products, with two_tower products taking precedence
        combined_products = {**ann_products, **tt_products}
        
        # Take first k products
        final_pids = list(combined_products.keys())[:k]

        for pid in final_pids:
            # Ensure pid is a standard Python int for dictionary lookup
            pid_int = int(pid)
            final_results.append({
                'query': query,
                'pid': pid_int,
                'product_title': mp_product_dict.get(pid_int, "Title Not Found"),
                'recall_source': combined_products[pid]  # Add source information
            })

    # --- 6. Create DataFrame ---
    df_final = pd.DataFrame(final_results)
    
    computation_time = time.time() - computation_start
    total_time = time.time() - start_time
    
    # Create timing info dictionary
    timing_info = {
        'total_time': total_time,
        'data_loading_time': data_loading_time,
        'computation_time': computation_time,
        'two_tower_time': tt_time,
        'ann_search_time': ann_time
    }
    
    # --- 7. Save Log and Return Results ---
    if not df_final.empty:
        print(f"Recall process completed. Returning {len(df_final)} recommendations.")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_recall_log(querys, df_final, k, timestamp, log_prefix="baseline_recall", timing_info=timing_info)
    else:
        print("Recall process completed. No recommendations generated (check logs for errors).")
    
    return df_final


if __name__ == "__main__":
    # Example usage:
    test_querys = ['smart phone, large screen', 'summer dress for women', 'smartphone charger', 'large iron pan']
    top_k = 30
    print(f"\n--- Running Example Recall for {len(test_querys)} queries ---")
    df_recommendations = main(test_querys, top_k)

    if not df_recommendations.empty:
        print("\n--- Example Recommendations ---")
        print(df_recommendations)
        # Display recommendations per query
        for query in test_querys:
             print(f"\nRecommendations for '{query}':")
             print(df_recommendations[df_recommendations['query'] == query][['pid', 'product_title']].to_string(index=False))
    else:
        print("\nExample run finished with no recommendations.")