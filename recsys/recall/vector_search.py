import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm

from recsys.utils.constants import VECTOR_SEARCH_N_REGIONS, VECTOR_SEARCH_N_NEAREST_ZONES

class VectorSearch:
    def __init__(self, n_regions: int = VECTOR_SEARCH_N_REGIONS):
        self.n_regions = n_regions
        self.region_centers = None
        self.region_hashes = None
        self.embeddings = None
    
    def _get_nearest_zones(self, vector: np.ndarray, n_zones: int = VECTOR_SEARCH_N_NEAREST_ZONES) -> List[int]:
        """Get the n nearest zone indices for a vector."""
        distances = np.linalg.norm(self.region_centers - vector, axis=1)
        return np.argsort(distances)[:n_zones].tolist()
    
    def build_regions(self, embeddings: np.ndarray):
        """Build regions using K-means clustering."""
        print("Building regions using K-means...")
        kmeans = KMeans(n_clusters=self.n_regions, random_state=42)
        self.region_hashes = kmeans.fit_predict(embeddings)
        self.region_centers = kmeans.cluster_centers_
        self.embeddings = embeddings
        
        # Save region centers
        np.save('data/region_centers.npy', self.region_centers)
        print(f"Built {self.n_regions} regions")
    
    def visualize_regions(self, output_path: str = 'results/region_visualization.png'):
        """
        Visualize the regions in 2D space using PCA.
        
        Args:
            output_path: Path to save the visualization
        """
        if self.embeddings is None or self.region_hashes is None:
            raise ValueError("Must build regions before visualization")
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Reduce dimensions to 2D using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(self.embeddings)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot points for each region with different colors
        for region in range(self.n_regions):
            mask = self.region_hashes == region
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=f'Region {region}',
                alpha=0.6
            )
        
        # Plot region centers
        centers_2d = pca.transform(self.region_centers)
        plt.scatter(
            centers_2d[:, 0],
            centers_2d[:, 1],
            c='black',
            marker='x',
            s=100,
            label='Region Centers'
        )
        
        plt.title('Vector Regions Visualization (PCA)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {output_path}")
    
    def search(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors using region-based approach.
        
        Args:
            query_vector: Query vector to search for
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (indices, distances) of nearest neighbors
        """
        # Get query's nearest zones
        nearest_zones = self._get_nearest_zones(query_vector)
        
        # Get vectors in the nearest zones
        zone_mask = np.isin(self.region_hashes, nearest_zones)
        zone_vectors = self.embeddings[zone_mask]
        zone_indices = np.where(zone_mask)[0]
        
        # Calculate distances
        distances = np.linalg.norm(zone_vectors - query_vector, axis=1)
        
        # Get top k
        if len(distances) < k:
            # If not enough vectors in zones, search in all vectors
            all_distances = np.linalg.norm(self.embeddings - query_vector, axis=1)
            top_k_indices = np.argsort(all_distances)[:k]
            return top_k_indices, all_distances[top_k_indices]
        
        top_k_indices = np.argsort(distances)[:k]
        return zone_indices[top_k_indices], distances[top_k_indices]

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
    
    # Save region hashes
    df_final['region_hash'] = vs.region_hashes
    df_final.to_csv('data/final_product_embeddings.csv', index=False)
    print("Saved vector search data") 

