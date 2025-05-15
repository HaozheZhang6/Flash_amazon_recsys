import numpy as np
from typing import List, Dict, Union, Tuple
import pandas as pd
from collections import defaultdict

def calculate_precision_at_k(
    recalled_items: List[int],
    clicked_items: List[int],
    k: int
) -> float:
    """
    Calculate precision@k for a single query.
    
    Args:
        recalled_items: List of recalled item IDs
        clicked_items: List of clicked item IDs
        k: Number of items to consider
        
    Returns:
        Precision@k score
    """
    if not recalled_items or not clicked_items:
        return 0.0
    
    # Take top k items
    top_k_items = recalled_items[:k]
    
    # Count how many of the top k items were clicked
    clicked_in_top_k = sum(1 for item in top_k_items if item in clicked_items)
    
    # Precision@k = (clicked items in top k) / k
    return clicked_in_top_k / k

def calculate_recall_at_k(
    recalled_items: List[int],
    clicked_items: List[int],
    k: int
) -> float:
    """
    Calculate recall@k for a single query.
    
    Args:
        recalled_items: List of recalled item IDs
        clicked_items: List of clicked item IDs
        k: Number of items to consider
        
    Returns:
        Recall@k score
    """
    if not recalled_items or not clicked_items:
        return 0.0
    
    # Take top k items
    top_k_items = recalled_items[:k]
    
    # Count how many of the clicked items are in top k
    clicked_in_top_k = sum(1 for item in clicked_items if item in top_k_items)
    
    # Recall@k = (clicked items in top k) / (total clicked items)
    return clicked_in_top_k / len(clicked_items)

def evaluate_recall_model(
    recall_results: Dict[str, List[int]],
    test_data: Dict[str, List[int]],
    k_values: List[int] = [5, 10, 20, 50]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a recall model using precision@k and recall@k metrics.
    
    Args:
        recall_results: Dictionary mapping queries to lists of recalled item IDs
        test_data: Dictionary mapping queries to lists of clicked item IDs
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary containing precision@k and recall@k scores for each k
    """
    metrics = defaultdict(lambda: defaultdict(list))
    
    # Calculate metrics for each query
    for query in recall_results.keys():
        if query not in test_data:
            continue
            
        recalled_items = recall_results[query]
        clicked_items = test_data[query]
        
        for k in k_values:
            # Calculate precision@k
            precision = calculate_precision_at_k(recalled_items, clicked_items, k)
            metrics['precision'][k].append(precision)
            
            # Calculate recall@k
            recall = calculate_recall_at_k(recalled_items, clicked_items, k)
            metrics['recall'][k].append(recall)
    
    # Calculate average metrics
    results = {}
    for metric_type in ['precision', 'recall']:
        results[metric_type] = {}
        for k in k_values:
            scores = metrics[metric_type][k]
            if scores:
                results[metric_type][f'@{k}'] = np.mean(scores)
            else:
                results[metric_type][f'@{k}'] = 0.0
    
    return results

def format_metrics(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Format metrics into a readable string.
    
    Args:
        metrics: Dictionary of metrics from evaluate_recall_model
        
    Returns:
        Formatted string with metrics
    """
    output = "Recall Model Evaluation Metrics:\n"
    output += "=" * 50 + "\n"
    
    for metric_type in ['precision', 'recall']:
        output += f"\n{metric_type.title()}:\n"
        output += "-" * 30 + "\n"
        for k, score in metrics[metric_type].items():
            output += f"{k}: {score:.4f}\n"
    
    return output 