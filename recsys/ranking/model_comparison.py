import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from recsys.data.load_data import load_test_data
from recsys.utils.constants import *
from recsys.utils.training_utils import select_device
from recsys.ranking.metric import evaluate_ranking_model, format_ranking_metrics

# Import FM model and config
try:
    from recsys.ranking.fm.model import FM
    from recsys.ranking.fm.train import FMConfig, MODEL_DIR as FM_MODEL_DIR
except ImportError:
    print("Warning: Could not import FM model components")
    FM = None
    @dataclass
    class FMConfig:
        q_in: int = 32
        p_in: int = 160
        hidden_dim: int = 32
        embed_dim: int = 32
    FM_MODEL_DIR = 'models/ranking-fm'

# Import DCN model and config
try:
    from recsys.ranking.dcn.model import DCN
    from recsys.ranking.dcn.train import DCNConfig, MODEL_DIR as DCN_MODEL_DIR
except ImportError:
    print("Warning: Could not import DCN model components")
    DCN = None
    @dataclass
    class DCNConfig:
        dcn_layers: int = 4
        dnn_layers: int = 4
        q_in: int = 32
        p_in: int = 160
        hidden_dim: int = 32
        embed_dim: int = 32
    DCN_MODEL_DIR = 'models/ranking-dcn'

class RankingModelEvaluator:
    """Base class for evaluating ranking models (FM, DCN)."""
    
    def __init__(self, model_path: str, config, model_class, device: str = None):
        if device is None:
            device = select_device()
        self.device = device
        self.config = config
        
        # Load model
        self.model = model_class(config=config).to(self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def score_query_product_pairs(
        self, 
        query_embeddings: torch.Tensor, 
        product_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Score all query-product pairs.
        
        Args:
            query_embeddings: (N_queries, query_dim)
            product_embeddings: (N_products, product_dim)
            
        Returns:
            scores: (N_queries, N_products) - scores for each query-product pair
        """
        N = query_embeddings.size(0)
        M = product_embeddings.size(0)
        
        query_embeddings = query_embeddings.to(self.device)
        product_embeddings = product_embeddings.to(self.device)
        
        # Expand for pairwise scoring
        q_expanded = query_embeddings.unsqueeze(1).expand(-1, M, -1)
        p_expanded = product_embeddings.unsqueeze(0).expand(N, -1, -1)
        
        # Flatten for model input
        q_flat = q_expanded.reshape(-1, self.config.q_in)
        p_flat = p_expanded.reshape(-1, self.config.p_in)
        
        with torch.no_grad():
            scores_flat = self.model(q_flat, p_flat)
            if scores_flat.dim() > 1:
                scores_flat = scores_flat.squeeze(-1)  # Remove last dim if present
        
        return scores_flat.view(N, M)

def evaluate_ranking_model_on_test_data(
    model_path: str,
    model_class,
    config,
    test_data: List[np.ndarray],
    clicked_products: Dict[int, List[int]],
    k_values: List[int] = [5, 10, 25, 50],
    device: str = None
) -> Dict[str, float]:
    """
    Evaluate a ranking model (FM or DCN) on test data.
    
    Args:
        model_path: Path to the trained model
        model_class: Model class (FM or DCN)
        config: Model configuration
        test_data: List of numpy arrays [query_ids, product_ids, query_embeddings, product_embeddings]
        clicked_products: Dictionary mapping query IDs to clicked product IDs
        k_values: List of k values to evaluate
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        evaluator = RankingModelEvaluator(model_path, config, model_class, device)
    except Exception as e:
        print(f"Error initializing model evaluator: {e}")
        return None
    
    # Unpack test data
    query_ids, product_ids, query_embeddings, product_embeddings = test_data
    
    # Get unique query and product IDs
    unique_query_ids = np.unique(query_ids)
    unique_product_ids = np.unique(product_ids)
    
    # Create mappings for unique embeddings
    query_embedding_map = {}
    product_embedding_map = {}
    
    # Build query embedding mapping
    for query_id in unique_query_ids:
        mask = query_ids == query_id
        query_embedding_map[query_id] = query_embeddings[mask][0]  # Take first occurrence
    
    # Build product embedding mapping  
    for product_id in unique_product_ids:
        mask = product_ids == product_id
        product_embedding_map[product_id] = product_embeddings[mask][0]  # Take first occurrence
    
    # Convert to tensors
    query_tensor = torch.tensor(
        [query_embedding_map[qid] for qid in unique_query_ids], 
        dtype=torch.float32
    )
    product_tensor = torch.tensor(
        [product_embedding_map[pid] for pid in unique_product_ids],
        dtype=torch.float32
    )
    
    try:
        # Get scores from ranking model
        scores = evaluator.score_query_product_pairs(query_tensor, product_tensor)
        scores_np = scores.cpu().numpy()
    except Exception as e:
        print(f"Error during model inference: {e}")
        return None
    
    # Generate ranking results for each query
    ranking_results = {}
    for i, query_id in enumerate(unique_query_ids):
        if query_id not in clicked_products:
            continue
            
        # Get scores for this query
        query_scores = scores_np[i]
        
        # Create list of (product_id, score) pairs
        product_score_pairs = list(zip(unique_product_ids, query_scores))
        
        # Sort by score (descending)
        product_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract ranked product IDs
        ranked_products = [pid for pid, _ in product_score_pairs[:max(k_values)]]
        ranking_results[query_id] = ranked_products
    
    if not ranking_results:
        print("No valid ranking results generated")
        return None
    
    # Evaluate using HR@K and NDCG@1 metrics
    metrics = evaluate_ranking_model(ranking_results, clicked_products, k_values)
    return metrics

def compare_ranking_models(
    model_configs: Dict[str, Dict],
    test_data_path: str = "data/dataset_150k_processed.csv",
    k_values: List[int] = [5, 10, 25, 50],
    output_dir: str = "results"
) -> None:
    """
    Compare different ranking models (FM, DCN) on test data.
    
    Args:
        model_configs: Dictionary with model configurations
            Format: {
                'model_name': {
                    'path': 'path/to/model.pt',
                    'type': 'FM' or 'DCN',
                    'config': model_config_object (optional)
                }
            }
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
    print(f"Number of test samples: {len(test_data[0])}")
    print(f"Number of unique queries: {len(clicked_products)}")
    print(f"Number of clicked products: {sum(len(pids) for pids in clicked_products.values())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Evaluate each model
    results = {}
    for model_name, model_info in model_configs.items():
        model_path = model_info['path']
        model_type = model_info['type'].upper()
        
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            continue
        
        print(f"\nEvaluating {model_name} ({model_type})...")
        
        # Get model class and config
        if model_type == 'FM':
            if FM is None:
                print(f"FM model class not available, skipping {model_name}")
                continue
            model_class = FM
            config = model_info.get('config', FMConfig())
        elif model_type == 'DCN':
            if DCN is None:
                print(f"DCN model class not available, skipping {model_name}")
                continue
            model_class = DCN
            config = model_info.get('config', DCNConfig())
        else:
            print(f"Unknown model type: {model_type}")
            continue
        
        # Evaluate model
        metrics = evaluate_ranking_model_on_test_data(
            model_path,
            model_class,
            config,
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
    output_file = os.path.join(output_dir, f'ranking_model_comparison_{timestamp}.txt')
    with open(output_file, 'w') as f:
        f.write("Ranking Model Comparison Results (HR@K and NDCG@1)\n")
        f.write("=" * 60 + "\n\n")
        
        # Write results for each model
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            f.write("-" * 30 + "\n")
            f.write(format_ranking_metrics(metrics))
            f.write("\n")
        
        # Add comparison summary
        f.write("\nComparison Summary:\n")
        f.write("-" * 30 + "\n")
        
        # HR@K comparison
        f.write("\nHit Rate@K Comparison:\n")
        for k in k_values:
            f.write(f"\nHR@{k}:\n")
            for model_name, metrics in results.items():
                score = metrics.get(f'HR@{k}', 0.0)
                f.write(f"{model_name}: {score:.4f}\n")
        
        # NDCG@1 comparison
        f.write(f"\nNDCG@1 Comparison:\n")
        for model_name, metrics in results.items():
            score = metrics.get('NDCG@1', 0.0)
            f.write(f"{model_name}: {score:.4f}\n")
    
    print(f"\nComparison results saved to: {output_file}")

if __name__ == "__main__":
    # Example usage comparing FM and DCN models
    model_configs = {
        "FM_base": {
            "path": os.path.join(FM_MODEL_DIR, "final_model.pt"),
            "type": "FM",
            "config": FMConfig()
        },
        "DCN_base": {
            "path": os.path.join(DCN_MODEL_DIR, "final_model.pt"), 
            "type": "DCN",
            "config": DCNConfig()
        },
        # Add more model configurations as needed
        # "FM_variant": {
        #     "path": "models/ranking-fm/variant_model.pt",
        #     "type": "FM",
        #     "config": FMConfig(hidden_dim=64)  # Example variant
        # },
    }
    
    k_values = [5, 10, 25, 50]
    
    compare_ranking_models(model_configs, k_values=k_values) 