import torch
import pandas as pd
import numpy as np
import os
from typing import List
from datetime import datetime

from recsys.ranking.dcn.eval import DCNEvaluator
from recsys.ranking.dcn.train import DCNConfig, MODEL_DIR
from recsys.recall.two_towers.utils import select_device
from recsys.data.embeding import convert_query
from recsys.recall.core import main as recall_main
from recsys.utils.log import save_ranking_results

# Define constants (consider moving these to a config file)
PRODUCT_EMBEDDING_CSV = 'data/product_150k.csv'
PRODUCT_META_PARQUET = 'data/shopping_queries_dataset_products.parquet'
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pt")

# Dimensions should match the saved model
INPUT_DIM_Q = 32
INPUT_DIM_P = 160
HIDDEN_DIM = 32
EMBED_DIM = 32

def main(querys: List[str], k: int = 30) -> pd.DataFrame:
    '''
    Given a list of queries, first get candidates from recall core (10*k),
    then rank the top k products using the DCN model.
    
    Args:
        querys: A list of query strings.
        k: The target number of products to rank for each query (default: 30).
        
    Returns:
        A Pandas DataFrame with columns ['query', 'pid', 'product_title', 'score', 'rank'].
        Returns an empty DataFrame if errors occur during data loading or model execution.
    '''
    print(f"Starting ranking for {len(querys)} queries with k={k}")
    
    # --- 1. Get candidates from recall core (10*k) ---
    print(f"Getting candidates from recall core (top {10*k})...")
    recall_results = recall_main(querys, k=10*k)  # Get 10 times more candidates
    if recall_results.empty:
        print("No candidates found from recall core.")
        return pd.DataFrame()
    print(f"Found {len(recall_results)} candidates from recall core.")
    
    # --- 2. Load Product Data for ranking ---
    print("Loading product data for ranking...")
    try:
        if not os.path.exists(PRODUCT_EMBEDDING_CSV) or not os.path.exists(PRODUCT_META_PARQUET):
            raise FileNotFoundError(f"Product data file(s) not found. Checked: {PRODUCT_EMBEDDING_CSV}, {PRODUCT_META_PARQUET}")
            
        df_product_emb_vectors = pd.read_csv(PRODUCT_EMBEDDING_CSV)
        df_product_meta = pd.read_parquet(PRODUCT_META_PARQUET)[['product_id', 'product_title']].drop_duplicates()
        df_product_embedding = pd.merge(df_product_emb_vectors, df_product_meta, on='product_id', how='inner').reset_index(drop=True)
        
        if 'pid' not in df_product_embedding.columns:
            df_product_embedding['pid'] = df_product_embedding.index
        else:
            df_product_embedding['pid'] = df_product_embedding['pid'].astype(int)
            
        print(f"Loaded {len(df_product_embedding)} products.")
        
        # Filter product embeddings to only include recalled products
        recalled_pids = set(recall_results['pid'].unique())
        df_product_embedding = df_product_embedding[df_product_embedding['pid'].isin(recalled_pids)].reset_index(drop=True)
        print(f"Filtered to {len(df_product_embedding)} recalled products.")
        
        product_cols = [f'p{i}' for i in range(INPUT_DIM_P)]
        if not all(col in df_product_embedding.columns for col in product_cols):
            missing_cols = [col for col in product_cols if col not in df_product_embedding.columns]
            raise ValueError(f"Product embedding columns missing in product data: {missing_cols}")
            
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading or processing product data: {e}")
        return pd.DataFrame()
        
    # --- 3. Convert Queries to Embeddings ---
    print("Converting queries to embeddings...")
    try:
        query_embedding = convert_query(querys)
        if query_embedding.shape[1] != INPUT_DIM_Q:
            raise ValueError(f"Query embedding dimension mismatch. Expected {INPUT_DIM_Q}, Got {query_embedding.shape[1]}")
        query_cols = [f'q{i}' for i in range(query_embedding.shape[1])]
        df_query_embedding = pd.DataFrame(query_embedding, columns=query_cols)
        df_query_embedding['query'] = querys
        print("Query embeddings prepared.")
    except Exception as e:
        print(f"Error converting queries to embeddings: {e}")
        return pd.DataFrame()
        
    # --- 4. Initialize DCN Model and Rank Products ---
    print("Initializing DCN model and ranking products...")
    try:
        device = select_device()
        config = DCNConfig(
            q_in=INPUT_DIM_Q,
            p_in=INPUT_DIM_P,
            hidden_dim=HIDDEN_DIM,
            embed_dim=EMBED_DIM
        )
        
        evaluator = DCNEvaluator(
            model_path=MODEL_PATH,
            config=config,
            device=device
        )
        
        # Convert data to tensors
        q_batch = torch.tensor(df_query_embedding[query_cols].values, dtype=torch.float32)
        p_batch = torch.tensor(df_product_embedding[product_cols].values, dtype=torch.float32)
        
        # Get scores for each query-product pair
        results = []
        for i, query in enumerate(querys):
            # Get query embedding
            q_emb = q_batch[i:i+1]  # Keep batch dimension [1, query_dim]
            
            # Get products for this query from recall results
            query_recall = recall_results[recall_results['query'] == query]
            query_pids = set(query_recall['pid'])
            query_products = df_product_embedding[df_product_embedding['pid'].isin(query_pids)]
            
            if query_products.empty:
                print(f"No products found for query: {query}")
                continue
            
            # Get product embeddings for this query
            p_emb = torch.tensor(query_products[product_cols].values, dtype=torch.float32)
            
            # Get scores for this query-product pairs
            scores = evaluator.evaluate(q_emb, p_emb)  # [1, num_products]
            scores_np = scores[0].cpu().numpy()  # Remove batch dimension
            
            # Get top k products
            top_k_indices = np.argsort(scores_np)[-k:][::-1]
            
            # Add results for this query
            for rank, prod_idx in enumerate(top_k_indices):
                pid = query_products.iloc[prod_idx]['pid']
                results.append({
                    'query': query,
                    'pid': int(pid),
                    'product_title': query_products.iloc[prod_idx]['product_title'],
                    'score': float(scores_np[prod_idx]),
                    'rank': rank + 1
                })
                
        print(f"Ranking completed. Returning {len(results)} recommendations.")
        results_df = pd.DataFrame(results)
        
        # Save results
        if not results_df.empty:
            print(f"Ranking process completed. Returning {len(results_df)} recommendations.")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_ranking_results(results_df, recall_results, querys, timestamp)
        else:
            print("Ranking process completed. No recommendations generated (check logs for errors).")
        
        return results_df
        
    except Exception as e:
        print(f"Error during ranking: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage:
    test_querys = [
        'bluetooth earphones', 
        'summer dress for women', 
        'smartphone charger', 
        'large iron pan',
        'large screen smart phone',
        'smart phone with large screen'
        ]
    top_k = 30
    results = main(test_querys, top_k)
    print("\nRanked Results:")
    print(results)
