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

# Define constants (consider moving these to a config file)
PRODUCT_EMBEDDING_CSV = 'versions/1/product_150k.csv'
PRODUCT_META_PARQUET = 'versions/1/shopping_queries_dataset_products.parquet'
MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pt")

# Dimensions should match the saved model
INPUT_DIM_Q = 32
INPUT_DIM_P = 160
HIDDEN_DIM = 32
EMBED_DIM = 32

def save_results_to_file(results: pd.DataFrame, recall_results: pd.DataFrame, queries: List[str], output_dir: str = 'results'):
    """
    Save the ranking results to a text file, including rank changes from recall.
    
    Args:
        results: DataFrame containing the ranking results
        recall_results: DataFrame containing the recall results
        queries: List of original queries
        output_dir: Directory to save the results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(output_dir, f'ranking_results_{timestamp}.txt')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Ranking Results with Recall Comparison\n")
        f.write("=" * 100 + "\n\n")
        
        # Write results for each query
        for query in queries:
            f.write(f"Query: {query}\n")
            f.write("-" * 100 + "\n")
            
            # Get results for this query
            query_results = results[results['query'] == query]
            query_recall = recall_results[recall_results['query'] == query]
            
            if query_results.empty:
                f.write("No results found for this query.\n\n")
                continue
                
            # Create a mapping of pid to recall rank
            recall_rank_map = {row['pid']: idx + 1 for idx, row in query_recall.iterrows()}
            
            # Write each product result
            for _, row in query_results.iterrows():
                recall_rank = recall_rank_map.get(row['pid'], 'N/A')
                rank_change = f"{recall_rank - row['rank']:+d}" if isinstance(recall_rank, int) else "N/A"
                
                f.write(f"Rank {row['rank']} (Recall Rank: {recall_rank}, Change: {rank_change})\n")
                f.write(f"Product ID: {row['pid']}\n")
                f.write(f"Title: {row['product_title']}\n")
                f.write(f"Score: {row['score']:.4f}\n")
                f.write("-" * 50 + "\n")
            
            f.write("\n")
        
        # Write summary
        f.write("\nSummary\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total queries: {len(queries)}\n")
        f.write(f"Total results: {len(results)}\n")
        f.write(f"Results per query: {len(results) // len(queries)}\n")
        
        # Calculate average rank changes
        rank_changes = []
        for query in queries:
            query_results = results[results['query'] == query]
            query_recall = recall_results[recall_results['query'] == query]
            recall_rank_map = {row['pid']: idx + 1 for idx, row in query_recall.iterrows()}
            
            for _, row in query_results.iterrows():
                recall_rank = recall_rank_map.get(row['pid'])
                if isinstance(recall_rank, int):
                    rank_changes.append(recall_rank - row['rank'])
        
        if rank_changes:
            avg_change = sum(rank_changes) / len(rank_changes)
            f.write(f"\nAverage rank change: {avg_change:+.2f}\n")
            f.write(f"Positive changes (improved): {sum(1 for x in rank_changes if x > 0)}\n")
            f.write(f"Negative changes (worsened): {sum(1 for x in rank_changes if x < 0)}\n")
            f.write(f"No change: {sum(1 for x in rank_changes if x == 0)}\n")
    
    print(f"\nResults saved to: {filename}")
    return filename

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
        
        # Get scores from DCN model
        scores = evaluator.evaluate(q_batch, p_batch)
        print("DCN scores calculated.")
        
        # Get top k products for each query
        results = []
        for i, query in enumerate(querys):
            # Convert scores to numpy and get top k indices
            scores_np = scores[i].cpu().numpy()
            top_k_indices = np.argsort(scores_np)[-k:][::-1]  # Get indices of top k scores in descending order
            
            # Add results for this query
            for rank, prod_idx in enumerate(top_k_indices):
                pid = df_product_embedding.iloc[prod_idx]['pid']
                results.append({
                    'query': query,
                    'pid': int(pid),
                    'product_title': df_product_embedding.iloc[prod_idx]['product_title'],
                    'score': float(scores_np[prod_idx]),
                    'rank': rank + 1
                })
                
        print(f"Ranking completed. Returning {len(results)} recommendations.")
        results_df = pd.DataFrame(results)
        
        # Save results to file with recall comparison
        save_results_to_file(results_df, recall_results, querys)
        
        return results_df
        
    except Exception as e:
        print(f"Error during ranking: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage:
    test_querys = ['bluetooth earphones', 'summer dress for women', 'smartphone charger', 'large iron pan']
    top_k = 30
    results = main(test_querys, top_k)
    print("\nRanked Results:")
    print(results)
