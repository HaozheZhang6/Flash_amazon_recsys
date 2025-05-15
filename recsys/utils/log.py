import os
from typing import List, Dict
import pandas as pd
from datetime import datetime
import time

RESULTS_DIR = "results"

def save_recall_log(
    querys: List[str], 
    results_df: pd.DataFrame, 
    k: int, 
    timestamp: str, 
    log_prefix: str = "recall",
    timing_info: Dict[str, float] = None
):
    """
    Save recall results and statistics to a log file.
    
    Args:
        querys: List of input queries
        results_df: DataFrame containing recall results
        k: Number of items requested per query
        timestamp: Timestamp for the log file
        log_prefix: Prefix for the log file name (default: "recall")
        timing_info: Dictionary containing timing information for different stages
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    log_file = os.path.join(RESULTS_DIR, f'{log_prefix}_log_{timestamp}.txt')
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"{log_prefix.title()} Process Log\n")
        f.write("=" * 80 + "\n\n")
        
        # Write timing information if available
        if timing_info:
            f.write("Timing Information:\n")
            f.write("-" * 40 + "\n")
            if 'total_time' in timing_info:
                f.write(f"Total execution time: {timing_info['total_time']:.2f} seconds\n")
            if 'data_loading_time' in timing_info:
                f.write(f"Data loading time: {timing_info['data_loading_time']:.2f} seconds\n")
            if 'computation_time' in timing_info:
                f.write(f"Pure computation/search time: {timing_info['computation_time']:.2f} seconds\n")
            if 'vector_search_time' in timing_info:
                f.write(f"Vector search time: {timing_info['vector_search_time']:.2f} seconds\n")
            if 'ann_search_time' in timing_info:
                f.write(f"ANN search time: {timing_info['ann_search_time']:.2f} seconds\n")
            f.write("\n")
        
        # Write input parameters
        f.write("Input Parameters:\n")
        f.write(f"Number of queries: {len(querys)}\n")
        f.write(f"Items per query (k): {k}\n")
        f.write("\nQueries:\n")
        for i, query in enumerate(querys, 1):
            f.write(f"{i}. {query}\n")
        
        # Write results summary
        f.write("\nResults Summary:\n")
        f.write(f"Total recommendations: {len(results_df)}\n")
        f.write(f"Average recommendations per query: {len(results_df)/len(querys):.2f}\n")
        
        # Write source distribution if recall_source column exists
        if 'recall_source' in results_df.columns:
            source_counts = results_df['recall_source'].value_counts()
            f.write("\nSource Distribution:\n")
            for source, count in source_counts.items():
                f.write(f"{source}: {count} products ({count/len(results_df)*100:.1f}%)\n")
        
        # Write detailed results per query
        f.write("\nDetailed Results:\n")
        for query in querys:
            query_results = results_df[results_df['query'] == query]
            f.write(f"\nQuery: {query}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Number of recommendations: {len(query_results)}\n")
            
            # Write source distribution for this query if recall_source exists
            if 'recall_source' in query_results.columns:
                source_counts = query_results['recall_source'].value_counts()
                f.write("\nSource Distribution for this query:\n")
                for source, count in source_counts.items():
                    f.write(f"{source}: {count} products ({count/len(query_results)*100:.1f}%)\n")
            
            if not query_results.empty:
                f.write("\nTop recommendations:\n")
                for _, row in query_results.iterrows():
                    log_line = f"PID: {row['pid']}, Title: {row['product_title']}"
                    if 'recall_source' in row:
                        log_line += f", Source: {row['recall_source']}"
                    if 'score' in row:
                        log_line += f", Score: {row['score']:.4f}"
                    f.write(log_line + "\n")
        
        print(f"\n{log_prefix.title()} log saved to: {log_file}")

def save_ranking_results(results: pd.DataFrame, recall_results: pd.DataFrame, queries: List[str], timestamp: str):
    """
    Save the ranking results to a text file, including rank changes from recall.
    
    Args:
        results: DataFrame containing the ranking results
        recall_results: DataFrame containing the recall results
        queries: List of original queries
        timestamp: Timestamp for the log file
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    filename = os.path.join(RESULTS_DIR, f'ranking_results_{timestamp}.txt')
    
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