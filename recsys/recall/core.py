
from typing import List, Tuple, Any, Optional
from recsys.recall.two_towers.eval import *
from recsys.recall.ann import *
from recsys.recall.two_towers.utils import select_device
from recsys.data.load_data import load_data
from recsys.data.utils import *
from recsys.data.embeding import convert_query
import pandas as pd


def main(querys: List[str], k: int, m: int):
    '''
    Given an array of query, find the top k products, 
    set 1: using two tower model to calculate the similarity with all products and select the top k/2,
    set 2: using ann to find the top m products of each of the top k/2 products
    select the top k products from the two sets
    '''
    
    df_product_embedding = pd.merge(
        # product_id,p0-p159,
        pd.read_csv('versions/1/product_150k.csv'), 
        # product_id,product_title,product_description,product_bullet_point,product_brand,product_color,product_locale
        pd.read_parquet('versions/1/shopping_queries_dataset_products.parquet')[['product_id','product_title']].drop_duplicates(),
        on = ['product_id']
        ).reset_index(drop=True)
    
    query_embedding = convert_query(querys)
    # Define the column names for the embeddings
    query_cols = [f'q{i}' for i in range(query_embedding.shape[1])] # Use shape[1] for robustness
    # Create DataFrame with correct column names
    df_query_embedding = pd.DataFrame(query_embedding, columns=query_cols)
    df_query_embedding['query'] = querys
    df_query_embedding['qid'] = range(0, df_query_embedding.shape[0])
    # Reorder columns (now the 'q' columns exist)
    df_query_embedding = df_query_embedding[['query','qid'] + query_cols]
    print("Query Embedding DataFrame Head:")
    print(df_query_embedding.head())

    # find the top k/2 products using two tower model for each query
    # TODO
    model_path = (
        "/Users/haozhezhang/Documents/Python Project/two_tower_search/models/two_towers/final_model.pt"
    )

    # These must match your training config
    input_dim_q = 32
    input_dim_p = 160
    hidden_q = 64
    hidden_p = 64
    embed = 32
    device = select_device() 
    evaluator = TwoTowerEvaluator(
        model_path=model_path,
        input_dim_q=input_dim_q,
        input_dim_p=input_dim_p,
        hidden_dim_q=hidden_q,
        hidden_dim_p=hidden_p,
        embed_dim=embed,
        device=device,
    )
    # Dummy batches for demonstration
    print("Product Embedding DataFrame Head:")
    print(df_product_embedding.head())
    # print(df_query_embedding.head()) # Already printed above

    # Prepare query batch for evaluator (use the correct columns and specify dtype)
    q_batch = torch.tensor(df_query_embedding[query_cols].values, dtype=torch.float32)

    # Prepare product batch for evaluator (use the correct columns and specify dtype)
    product_cols = [f'p{i}' for i in range(input_dim_p)] # Assuming product columns are p0 to p159
    p_batch = torch.tensor(df_product_embedding[product_cols].values, dtype=torch.float32)

    scores = evaluator.evaluate(q_batch, p_batch)
    print("Score matrix shape:", scores.shape)
    print("Scores:")
    print(scores)


if __name__ == "__main__":
    querys = ['samsung galaxy s21', 'samsung galaxy s22']
    k = 10
    m = 5
    main(querys, k, m)
