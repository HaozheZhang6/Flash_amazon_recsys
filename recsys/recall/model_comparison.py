import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path

from recsys.data.load_data import load_test_data
from recsys.utils.constants import *
from recsys.recall.two_towers.model import TwoTowerModel
from recsys.utils.training_utils import select_device
from recsys.recall.metric import evaluate_recall_model, format_metrics

def evaluate_two_tower_model(
    model_path: str,
    test_data: List[np.ndarray],
    clicked_products: Dict[int, List[int]],
    k_values: List[int] = [5, 10, 20, 50],
    device: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a two-tower model on test data.
    
    Args:
        model_path: Path to the trained model
        test_data: List of numpy arrays [query_ids, product_ids, query_embeddings, product_embeddings]
        clicked_products: Dictionary mapping query IDs to clicked product IDs
        k_values: List of k values to evaluate
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if device is None:
        device = select_device()
    
    # Unpack test data
    query_ids, product_ids, query_embeddings, product_embeddings = test_data
    
    # Load model
    model = TwoTowerModel(
        q_in=INPUT_DIM_Q,
        p_in=INPUT_DIM_P,
        q_hidden=HIDDEN_Q,
        p_hidden=HIDDEN_P,
        emb=EMBED
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Get unique query IDs
    unique_query_ids = np.unique(query_ids)
    
    # Generate final query embeddings
    try:
        with torch.no_grad():
            # Get unique query embeddings
            unique_query_embeddings = {}
            for query_id in unique_query_ids:
                mask = query_ids == query_id
                query_emb = query_embeddings[mask][0]  # Take first occurrence
                q_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0).to(device)
                final_emb = model.query_tower(q_tensor)
                # Normalize
                final_emb = final_emb / final_emb.norm(p=2, dim=1, keepdim=True)
                unique_query_embeddings[query_id] = final_emb.cpu().numpy()[0]
    except Exception as e:
        print(f"Error generating query embeddings: {e}")
        return None
    
    # Generate final product embeddings
    try:
        with torch.no_grad():
            # Get unique product embeddings
            unique_product_embeddings = {}
            unique_products = np.unique(product_ids)
            for product_id in unique_products:
                mask = product_ids == product_id
                prod_emb = product_embeddings[mask][0]  # Take first occurrence
                p_tensor = torch.tensor(prod_emb, dtype=torch.float32).unsqueeze(0).to(device)
                final_emb = model.product_tower(p_tensor)
                # Normalize
                final_emb = final_emb / final_emb.norm(p=2, dim=1, keepdim=True)
                unique_product_embeddings[product_id] = final_emb.cpu().numpy()[0]
    except Exception as e:
        print(f"Error generating product embeddings: {e}")
        return None
    
    # Get recall results for each query
    recall_results = {}
    for query_id in unique_query_ids:
        try:
            query_emb = unique_query_embeddings[query_id]
            
            # Calculate similarity scores with all products
            similarities = []
            for product_id, prod_emb in unique_product_embeddings.items():
                similarity = np.dot(query_emb, prod_emb)
                similarities.append((product_id, similarity))
            
            # Sort by similarity and get top k products
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_k_pids = [pid for pid, _ in similarities[:max(k_values)]]
            recall_results[query_id] = top_k_pids
        except Exception as e:
            print(f"Error processing query ID {query_id}: {e}")
            continue
    
    if not recall_results:
        print("No valid recall results generated")
        return None
    
    # Evaluate using metrics
    metrics = evaluate_recall_model(recall_results, clicked_products, k_values)
    return metrics

def compare_two_tower_models(
    model_paths: Dict[str, str],
    test_data_path: str = "data/dataset_150k_processed.csv",
    k_values: List[int] = [5, 10, 20, 50],
    output_dir: str = "results"
) -> None:
    """
    Compare different two-tower models on test data.
    
    Args:
        model_paths: Dictionary mapping model names to their paths
        test_data_path: Path to test data file
        k_values: List of k values to evaluate
        output_dir: Directory to save comparison results
    """
    # Load test data
    print("Loading test data...")
    test_data, clicked_products = load_test_data(test_data_path)
    
    if not test_data or not clicked_products:
        print("Error: No test data loaded or no clicked products found")
        return
    
    print(f"\nTest data summary:")
    print(f"Number of test samples: {len(test_data[0])}")  # Length of query_ids
    print(f"Number of unique queries: {len(clicked_products)}")
    print(f"Number of clicked products: {sum(len(pids) for pids in clicked_products.values())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Evaluate each model
    results = {}
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            continue
            
        print(f"\nEvaluating {model_name}...")
        metrics = evaluate_two_tower_model(
            model_path,
            test_data,
            clicked_products,
            k_values
        )
        if metrics is not None:
            results[model_name] = metrics
        else:
            print(f"Failed to evaluate {model_name}")
    
    if not results:
        print("No models were successfully evaluated")
        return
    
    # Save comparison results
    output_file = os.path.join(output_dir, f'two_tower_comparison_{timestamp}.txt')
    with open(output_file, 'w') as f:
        f.write("Two-Tower Model Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Write results for each model
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write("-" * 30 + "\n")
            f.write(format_metrics(metrics))
            f.write("\n")
        
        # Add comparison summary
        f.write("\nComparison Summary:\n")
        f.write("-" * 30 + "\n")
        for metric_type in ['precision', 'recall']:
            f.write(f"\n{metric_type.title()}:\n")
            for k in k_values:
                f.write(f"\n@{k}:\n")
                for model_name, metrics in results.items():
                    score = metrics[metric_type][f'@{k}']
                    f.write(f"{model_name}: {score:.4f}\n")
    
    print(f"\nComparison results saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    model_paths = {
        "base_model": "models/two_towers/final_model.pt",
        "point_wise_model": "models/two_towers/point_wise/final_model.pt",
        "pair_wise_model": "models/two_towers/pair_wise/final_model.pt",
        "list_wise_model": "models/two_towers/list_wise/final_model.pt"
    }
    k_values = [5, 10, 20, 50]
    
    compare_two_tower_models(model_paths, k_values=k_values) 