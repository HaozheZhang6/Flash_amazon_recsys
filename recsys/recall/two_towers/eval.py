import torch
from torch.utils.data import TensorDataset, DataLoader
from recsys.recall.two_towers.model import TwoTowerModel
from recsys.recall.two_towers.utils import select_device
from recsys.data.load_data import load_data
import numpy as np
from typing import Dict, Tuple

class TwoTowerEvaluator:
    """Loads a TwoTowerModel and computes scores between query and product batches."""

    def __init__(
        self,
        model_path: str,
        input_dim_q: int,
        input_dim_p: int,
        hidden_dim_q: int,
        hidden_dim_p: int,
        embed_dim: int,
        device: torch.device = None,
    ):
        """
        Initialize evaluator by loading a pretrained TwoTowerModel.

        Args:
            model_path: Path to the .pt file with saved state_dict.
            input_dim_q: Dim of query-tower input features.
            input_dim_p: Dim of product-tower input features.
            hidden_dim_q: Hidden size for query tower.
            hidden_dim_p: Hidden size for product tower.
            embed_dim: Output embedding dimension.
            device: Torch device; defaults to CPU.
        """
        if device is None:
            device = select_device()
        self.device = device

        # Instantiate and load weights
        self.model = TwoTowerModel(
            q_in=input_dim_q,
            p_in=input_dim_p,
            q_hidden=hidden_dim_q,
            p_hidden=hidden_dim_p,
            emb=embed_dim,
        ).to(self.device)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def evaluate(self, q_batch: torch.Tensor, p_batch: torch.Tensor) -> torch.Tensor:
        """
        Evaluate query-product pairs and return similarity scores.
        
        Args:
            q_batch: Query embeddings [num_queries, query_dim]
            p_batch: Product embeddings [num_products, product_dim]
        """
        with torch.no_grad():
            q_batch = q_batch.to(self.device)  # [num_queries, query_dim]
            p_batch = p_batch.to(self.device)  # [num_products, product_dim]
            
            # Get embeddings from the model
            query_emb = self.model.query_tower(q_batch)  # [num_queries, embed_dim]
            prod_emb = self.model.product_tower(p_batch)  # [num_products, embed_dim]
            
            # Reshape for broadcasting
            query_emb = query_emb.unsqueeze(1)  # [num_queries, 1, embed_dim]
            prod_emb = prod_emb.unsqueeze(0)    # [1, num_products, embed_dim]
            
            # Calculate similarity scores
            sim = torch.sum(query_emb * prod_emb, dim=2)  # [num_queries, num_products]
            
            return sim

def compare_models() -> Dict[str, float]:
    """
    Compare the cross-entropy loss of all three training methods (point-wise, pair-wise, list-wise)
    on the test data, considering only E->1 cases.
    
    Returns:
        Dictionary containing the test loss for each training method
    """
    device = select_device()
    results = {}
    
    # Load test data
    test_inputs, test_labels, _, _ = load_data(usage="recall", training_method="point_wise")
    
    # Convert test data to tensors
    test_q = torch.tensor(test_inputs[0], dtype=torch.float32)
    test_p = torch.tensor(test_inputs[1], dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.float32)
    
    # Filter for E->1 cases only
    e_mask = (test_y == 1.0)
    test_q = test_q[e_mask]
    test_p = test_p[e_mask]
    test_y = test_y[e_mask]
    
    # Create test loader
    test_loader = DataLoader(
        TensorDataset(test_q, test_p, test_y),
        batch_size=512,
        shuffle=True
    )
    
    # Load and evaluate each model
    training_methods = ["point_wise", "pair_wise", "list_wise"]
    
    for method in training_methods:
        # Load model
        model = TwoTowerModel(
            q_in=32,
            p_in=160,
            q_hidden=64,
            p_hidden=64,
            emb=32
        ).to(device)
        
        # Load the best model for this method
        model_path = f'models/two_towers/{method}/final_model.pt'
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Calculate loss
            total_loss = 0.0
            num_samples = 0
            
            with torch.no_grad():
                for q_batch, p_batch, y_batch in test_loader:
                    q_batch, p_batch, y_batch = (
                        q_batch.to(device),
                        p_batch.to(device),
                        y_batch.to(device)
                    )
                    
                    # Get model predictions
                    pred = model(q_batch, p_batch)
                    
                    # Calculate cross-entropy loss
                    loss = torch.nn.BCEWithLogitsLoss()(pred, y_batch.unsqueeze(1))
                    
                    total_loss += loss.item() * q_batch.size(0)
                    num_samples += q_batch.size(0)
            
            # Calculate average loss
            avg_loss = total_loss / num_samples
            results[method] = avg_loss
            
            print(f"{method} test loss: {avg_loss:.4f}")
            
        except Exception as e:
            print(f"Error loading {method} model: {str(e)}")
            results[method] = float('inf')
    
    return results

if __name__ == "__main__":
    results = compare_models()
    print("\nFinal Results:")
    for method, loss in results.items():
        print(f"{method}: {loss:.4f}")
