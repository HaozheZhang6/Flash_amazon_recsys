import torch
import pandas as pd
import numpy as np
import os
from typing import List
from datetime import datetime
import time

from recsys.recall.two_towers.model import TwoTowerModel
from recsys.recall.ann.ann import AnnoyRecall
from recsys.recall.two_towers.utils import (
    select_device,
    generate_final_product_embeddings,
    generate_query_embeddings
)
from recsys.recall.vector_search import VectorSearch, prepare_vector_search_data
from recsys.utils.constants import *
from recsys.utils.log import save_recall_log

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
    # Initialize all timing variables at the start
    start_time = time.time()
    data_loading_start = time.time()
    vector_search_start = None
    vector_search_time = 0
    ann_search_start = None
    ann_search_time = 0
    
    print(f"Starting recall for {len(querys)} queries with k={k}")

    # --- 1. Load or Generate Final Product Embeddings ---
    print("Loading or generating final product embeddings...")
    try:
        if os.path.exists(FINAL_EMBEDDINGS_CSV):
            print("Loading existing final embeddings...")
            df_product_embedding = pd.read_csv(FINAL_EMBEDDINGS_CSV)
        else:
            print("Generating new final embeddings...")
            # Load original product data
            if not os.path.exists(PRODUCT_EMBEDDING_CSV) or not os.path.exists(PRODUCT_META_PARQUET):
                raise FileNotFoundError(f"Product data file(s) not found.")
            
            df_product_emb_vectors = pd.read_csv(PRODUCT_EMBEDDING_CSV)
            df_product_meta = pd.read_parquet(PRODUCT_META_PARQUET)[['product_id', 'product_title']].drop_duplicates()
            df_product_embedding = pd.merge(df_product_emb_vectors, df_product_meta, on='product_id', how='inner')
            
            # Ensure we have pid column
            if 'pid' not in df_product_embedding.columns:
                df_product_embedding['pid'] = df_product_embedding.index
            
            # Generate final embeddings using product tower
            df_product_embedding = generate_final_product_embeddings(
                df_product_embedding, 
                MODEL_PATH_DICT[MODEL_METHOD]
            )
            
            # Prepare vector search data
            prepare_vector_search_data(df_product_embedding)
            
            print(f"Saved final embeddings to {FINAL_EMBEDDINGS_CSV}")

        print(f"Loaded {len(df_product_embedding)} products.")
        
        # Ensure we have the required columns
        if 'pid' not in df_product_embedding.columns:
            df_product_embedding['pid'] = df_product_embedding.index
            
        mp_product_dict = pd.Series(df_product_embedding.product_title.values, index=df_product_embedding.pid).to_dict()

        # Check product embedding columns
        product_cols = [f'p{i}' for i in range(INPUT_DIM_P)]
        if not all(col in df_product_embedding.columns for col in product_cols):
            missing_cols = [col for col in product_cols if col not in df_product_embedding.columns]
            raise ValueError(f"Product embedding columns missing in product data: {missing_cols}")

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading or processing product data: {e}")
        return pd.DataFrame()

    data_loading_time = time.time() - data_loading_start
    computation_start = time.time()

    # --- 2. Convert Queries to Embeddings ---
    print("Converting queries to embeddings...")
    try:
        # Generate final query embeddings using query tower
        query_embeddings, df_query_embedding = generate_query_embeddings(
            querys,
            MODEL_PATH_DICT[MODEL_METHOD]
        )
        print("Query embeddings prepared.")
        df_query_embedding.to_csv('data/query_embeddings.csv', index=False)
    except Exception as e:
        print(f"Error converting queries to embeddings: {e}")
        return pd.DataFrame()
    vector_search_start = time.time()

    # --- 3. Vector Search Recall (Top k/2) ---
    print("Performing vector search recall...")
    all_tt_pids = [[] for _ in querys]
    try:
        # Initialize vector search
        vs = VectorSearch()
        vs.region_centers = np.load('data/region_centers.npy')
        vs.region_hashes = df_product_embedding['region_hash'].values
        final_emb_cols = [col for col in df_product_embedding.columns if col.startswith('final_emb_')]
        vs.embeddings = df_product_embedding[final_emb_cols].values
        
        # Get query embeddings from df_query_embedding
        query_emb_cols = [col for col in df_query_embedding.columns if col.startswith('final_emb_')]
        query_embeddings = df_query_embedding[query_emb_cols].values
        
        # Get top k/2 products for each query
        num_products_tt = k // 2
        for i, query in enumerate(querys):
            query_vec = query_embeddings[i]
            indices, distances = vs.search(query_vec, num_products_tt)
            all_tt_pids[i] = df_product_embedding.iloc[indices]['pid'].tolist()
            print(f"Query '{query}': Found {len(all_tt_pids[i])} products from vector search")
            
        print(f"Top {num_products_tt} products from vector search identified for each query.")
        
    except Exception as e:
        print(f"Error during vector search: {e}")
        all_tt_pids = [[] for _ in querys]

    vector_search_time = time.time() - vector_search_start

    # --- 4. ANN Recall (Based on Two-Tower Results) ---
    print("Performing ANN recall based on Two-Tower results...")
    ann_search_start = time.time()
    all_ann_pids = [[] for _ in querys]
    try:
        # Use dimensions inferred or defined earlier
        ann_recall = AnnoyRecall(
            query_tower_input_dim=ANN_QUERY_DIM,
            product_tower_input_dim=ANN_PRODUCT_DIM
        )
        
        if not os.path.exists(ann_recall.p_tree_path):
            print("Building ANN indices...")
            # Add qid to query embeddings if not present
            if 'qid' not in df_query_embedding.columns:
                df_query_embedding['qid'] = df_query_embedding.index
            
            # Build and save indices using existing method
            ann_recall.build_and_save_indices(
                df_query_embedding=df_query_embedding,
                df_product_embedding=df_product_embedding,
                num_trees=NUM_PRODUCT_TREES
            )
        
        # Load indices
        ann_recall._ensure_product_index_loaded()
        print("Annoy indices loaded/ensured.")

        # Ensure k is not greater than the number of items in the index
        num_items_in_index = ann_recall.p_index.get_n_items() if ann_recall.p_index else 0
        k_safe_ann = min(k//NUM_PRODUCT_TREES, num_items_in_index)
        if k_safe_ann < k//NUM_PRODUCT_TREES:
            print(f"Warning: Requested top {k//NUM_PRODUCT_TREES} neighbors from ANN per seed, but only {num_items_in_index} items in index. Using {k_safe_ann}.")

        if k_safe_ann > 0:
            # Prepare product embeddings lookup
            product_vectors_map = df_product_embedding.set_index('pid')[product_cols].T.to_dict('list')

            for i in range(len(querys)):
                seed_pids = all_tt_pids[i]
                query_ann_neighbors = set()

                if not seed_pids:
                    print(f"No seed products from Two-Tower for query '{querys[i]}', skipping ANN step for this query.")
                    continue

                for seed_pid in seed_pids:
                    seed_pid_int = int(seed_pid)
                    if seed_pid_int in product_vectors_map:
                        product_vec = product_vectors_map[seed_pid_int]
                        try:
                            neighbor_pids = ann_recall.get_product_neighbors_by_vector(product_vec, k=k_safe_ann)
                            query_ann_neighbors.update(neighbor_pids)
                        except (RuntimeError, Exception) as e:
                            print(f"Error during ANN search for seed product PID {seed_pid_int} (query '{querys[i]}'): {e}")
                    else:
                        print(f"Warning: Seed product PID {seed_pid_int} from Two-Tower not found in product embedding map. Skipping ANN search for this seed.")

                # Remove the original seed PIDs from the neighbors if present
                query_ann_neighbors.difference_update(set(seed_pids))
                all_ann_pids[i] = list(query_ann_neighbors)

            print(f"ANN search completed, finding up to {k_safe_ann} neighbors for each seed product.")
        else:
            print("k or number of items in ANN index is 0, skipping ANN recall.")

    except FileNotFoundError as e:
        print(f"Error related to Annoy index files: {e}. Skipping ANN recall.")
        all_ann_pids = [[] for _ in querys]
    except (RuntimeError, Exception) as e:
        print(f"Error initializing or using AnnoyRecall: {e}. Skipping ANN recall.")
        all_ann_pids = [[] for _ in querys]

    ann_search_time = time.time() - ann_search_start

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
        
        # Limit ANN products to k//2 and exclude any that are in tt_pids
        ann_pids = ann_pids - tt_pids  # Remove duplicates
        ann_pids = list(ann_pids)[:k//2]  # Take only k//2 products
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
                'recall_source': combined_products[pid]
            })

    # --- 6. Create DataFrame ---
    df_final = pd.DataFrame(final_results)

    # Calculate final timing metrics
    computation_time = time.time() - computation_start
    total_time = time.time() - start_time

    # Create timing info dictionary
    timing_info = {
        'total_time': total_time,
        'data_loading_time': data_loading_time,
        'computation_time': computation_time,
        'vector_search_time': vector_search_time,
        'ann_search_time': ann_search_time
    }
    
    # --- 7. Save Log and Return Results ---
    if not df_final.empty:
        print(f"Recall process completed. Returning {len(df_final)} recommendations.")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_recall_log(querys, df_final, k, timestamp, log_prefix="recall", timing_info=timing_info)
    else:
        print("Recall process completed. No recommendations generated (check logs for errors).")
    
    return df_final


def generate_final_product_embeddings(df_product_embedding: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """
    Generate final product embeddings using the product tower and normalize them.
    
    Args:
        df_product_embedding: DataFrame containing product embeddings
        model_path: Path to the trained model
    
    Returns:
        DataFrame with final normalized embeddings
    """
    device = select_device()
    
    # First create the model with correct architecture
    model = TwoTowerModel(
        q_in=INPUT_DIM_Q,
        p_in=INPUT_DIM_P,
        q_hidden=HIDDEN_Q,
        p_hidden=HIDDEN_P,
        emb=EMBED
    ).to(device)
    
    # Then load the state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Get product embeddings
    product_cols = [f'p{i}' for i in range(INPUT_DIM_P)]
    p_batch = torch.tensor(df_product_embedding[product_cols].values, dtype=torch.float32).to(device)
    
    # Generate final embeddings
    with torch.no_grad():
        prod_emb = model.product_tower(p_batch)
        # Normalize embeddings
        prod_norm = prod_emb.norm(p=2, dim=1, keepdim=True)
        final_embeddings = prod_emb / prod_norm
    
    # Move back to CPU for DataFrame creation
    final_embeddings = final_embeddings.cpu().numpy()
    
    # Create new DataFrame with final embeddings
    final_emb_cols = [f'final_emb_{i}' for i in range(final_embeddings.shape[1])]
    df_final = df_product_embedding.copy()
    
    # Ensure we have the required columns
    if 'pid' not in df_final.columns:
        df_final['pid'] = df_final.index
    if 'product_title' not in df_final.columns:
        df_final['product_title'] = df_final['product_id']  # Use product_id as title if title is missing
    
    # Add final embeddings
    df_final[final_emb_cols] = final_embeddings
    
    return df_final


def prepare_vector_search_data(df_final: pd.DataFrame) -> None:
    """
    Prepare and save vector search data.
    
    Args:
        df_final: DataFrame containing final embeddings
    """
    # Extract final embeddings
    final_emb_cols = [col for col in df_final.columns if col.startswith('final_emb_')]
    embeddings = df_final[final_emb_cols].values
    
    # Initialize and build vector search
    vs = VectorSearch()
    vs.build_regions(embeddings)
    
    # Visualize regions
    vs.visualize_regions()
    
    # Save region hashes
    df_final['region_hash'] = vs.region_hashes
    df_final.to_csv('data/final_product_embeddings.csv', index=False)
    print("Saved vector search data")


if __name__ == "__main__":
    # Example usage:
    test_querys = [
        'smart phone, large screen', 
        'summer dress for women', 
        'smartphone charger', 
        'large iron pan'
        ]
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
