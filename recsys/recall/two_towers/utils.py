import torch
import pandas as pd
import numpy as np
from typing import List, Tuple

from recsys.utils.constants import *
from recsys.utils.training_utils import select_device
from recsys.data.embeding import convert_query
from recsys.recall.two_towers.model import TwoTowerModel

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
    
    # Initialize model with correct dimensions
    model = TwoTowerModel(
        q_in=INPUT_DIM_Q,
        p_in=INPUT_DIM_P,
        q_hidden=HIDDEN_Q,
        p_hidden=HIDDEN_P,
        emb=EMBED
    ).to(device)
    
    # Load state dict
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
    df_final[final_emb_cols] = final_embeddings
    
    return df_final


def generate_query_embeddings(querys: List[str], model_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Generate query embeddings using the query tower and normalize them.
    
    Args:
        querys: List of query strings
        model_path: Path to the trained model
    
    Returns:
        Tuple of (normalized embeddings array, DataFrame with embeddings)
    """
    device = select_device()
    
    # Initialize model with correct dimensions
    model = TwoTowerModel(
        q_in=INPUT_DIM_Q,
        p_in=INPUT_DIM_P,
        q_hidden=HIDDEN_Q,
        p_hidden=HIDDEN_P,
        emb=EMBED
    ).to(device)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Convert queries to initial embeddings
    initial_embeddings = convert_query(querys)

    # Convert to tensor and move to device
    q_batch = torch.tensor(initial_embeddings, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        query_emb = model.query_tower(q_batch)
        # Normalize embeddings
        query_norm = query_emb.norm(p=2, dim=1, keepdim=True)
        final_embeddings = query_emb / query_norm
    
    # Move back to CPU for DataFrame creation
    final_embeddings = final_embeddings.cpu().numpy()
    
    # Create DataFrame with final embeddings
    final_emb_cols = [f'final_emb_{i}' for i in range(final_embeddings.shape[1])]
    df_query_embedding = pd.DataFrame(final_embeddings, columns=final_emb_cols)
    df_query_embedding['query'] = querys
    
    return final_embeddings, df_query_embedding


