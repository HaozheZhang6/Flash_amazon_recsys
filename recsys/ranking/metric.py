import numpy as np
from typing import List, Dict, Union, Tuple
import pandas as pd
from collections import defaultdict

def calculate_hit_rate_at_k(
    predicted_items: List[int],
    relevant_items: List[int],
    k: int
) -> float:
    """
    Calculate Hit Rate@k for a single query.
    HR@k = 1 if any relevant item is in top k predictions, 0 otherwise.
    
    Args:
        predicted_items: List of predicted item IDs (ranked by relevance)
        relevant_items: List of relevant/clicked item IDs
        k: Number of items to consider
        
    Returns:
        Hit Rate@k score (0 or 1)
    """
    if not predicted_items or not relevant_items:
        return 0.0
    
    # Take top k items
    top_k_items = set(predicted_items[:k])
    relevant_set = set(relevant_items)
    
    # HR@k = 1 if any relevant item is in top k, 0 otherwise
    return 1.0 if len(top_k_items & relevant_set) > 0 else 0.0

def calculate_ndcg_at_1(
    predicted_items: List[int],
    relevant_items: List[int]
) -> float:
    """
    Calculate NDCG@1 for a single query.
    NDCG@1 = 1 if the top predicted item is relevant, 0 otherwise.
    
    Args:
        predicted_items: List of predicted item IDs (ranked by relevance)
        relevant_items: List of relevant/clicked item IDs
        
    Returns:
        NDCG@1 score (0 or 1)
    """
    if not predicted_items or not relevant_items:
        return 0.0
    
    # Check if the top item is relevant
    top_item = predicted_items[0]
    return 1.0 if top_item in relevant_items else 0.0

def evaluate_ranking_model(
    ranking_results: Dict[str, List[int]],
    test_data: Dict[str, List[int]],
    k_values: List[int] = [5, 10, 25, 50]
) -> Dict[str, float]:
    """
    Evaluate a ranking model using HR@K and NDCG@1 metrics.
    
    Args:
        ranking_results: Dictionary mapping queries to lists of ranked item IDs
        test_data: Dictionary mapping queries to lists of relevant/clicked item IDs
        k_values: List of k values to evaluate for HR@K
        
    Returns:
        Dictionary containing HR@K and NDCG@1 scores
    """
    metrics = defaultdict(list)
    
    # Calculate metrics for each query
    for query in ranking_results.keys():
        if query not in test_data:
            continue
            
        predicted_items = ranking_results[query]
        relevant_items = test_data[query]
        
        # Calculate HR@K for each k value
        for k in k_values:
            hr_score = calculate_hit_rate_at_k(predicted_items, relevant_items, k)
            metrics[f'HR@{k}'].append(hr_score)
        
        # Calculate NDCG@1
        ndcg_score = calculate_ndcg_at_1(predicted_items, relevant_items)
        metrics['NDCG@1'].append(ndcg_score)
    
    # Calculate average metrics
    results = {}
    for metric_name, scores in metrics.items():
        if scores:
            results[metric_name] = np.mean(scores)
        else:
            results[metric_name] = 0.0
    
    return results

def format_ranking_metrics(metrics: Dict[str, float]) -> str:
    """
    Format ranking metrics into a readable string.
    
    Args:
        metrics: Dictionary of metrics from evaluate_ranking_model
        
    Returns:
        Formatted string with metrics
    """
    output = "Ranking Model Evaluation Metrics:\n"
    output += "=" * 50 + "\n"
    
    # Display HR@K metrics
    output += "\nHit Rate@K:\n"
    output += "-" * 30 + "\n"
    hr_metrics = {k: v for k, v in metrics.items() if k.startswith('HR@')}
    for metric_name in sorted(hr_metrics.keys(), key=lambda x: int(x.split('@')[1])):
        output += f"{metric_name}: {hr_metrics[metric_name]:.4f}\n"
    
    # Display NDCG@1
    output += "\nNDCG@1:\n"
    output += "-" * 30 + "\n"
    output += f"NDCG@1: {metrics.get('NDCG@1', 0.0):.4f}\n"
    
    return output

# Legacy function aliases for backward compatibility
def evaluate_recall_model(
    recall_results: Dict[str, List[int]],
    test_data: Dict[str, List[int]],
    k_values: List[int] = [5, 10, 25, 50]
) -> Dict[str, float]:
    """
    Legacy function for backward compatibility.
    Now evaluates using HR@K and NDCG@1 instead of precision/recall.
    """
    return evaluate_ranking_model(recall_results, test_data, k_values)

def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Legacy function for backward compatibility.
    """
    return format_ranking_metrics(metrics) 