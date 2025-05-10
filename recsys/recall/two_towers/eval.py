import torch
from recsys.recall.two_towers.model import TwoTowerModel
from recsys.recall.two_towers.train import select_device
import numpy as np

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
        training_method: str = "point_wise",
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
            training_method: The training method used ("point_wise", "pair_wise", or "list_wise")
            device: Torch device; defaults to CPU.
        """
        if device is None:
            device = select_device()
        self.device = device
        self.training_method = training_method

        # Instantiate and load weights
        self.model = TwoTowerModel(
            q_in=input_dim_q,
            p_in=input_dim_p,
            q_hidden=hidden_dim_q,
            p_hidden=hidden_dim_p,
            emb=embed_dim,
        ).to(self.device)

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def evaluate(
        self,
        query_batch: torch.Tensor,
        product_batch: torch.Tensor,
        log_probs: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the score matrix for all query-product pairs.

        Args:
            query_batch: Tensor of shape (n_queries, input_dim_q).
            product_batch: Tensor of shape (n_products, input_dim_p).
            log_probs: Optional tensor of log probabilities for list-wise evaluation.

        Returns:
            scores: Tensor of shape (n_queries, n_products), where
                scores[i, j] = dot(query_emb[i], product_emb[j]).
        """
        with torch.no_grad():
            q_emb = self.model.query_tower(query_batch.to(self.device))
            p_emb = self.model.product_tower(product_batch.to(self.device))
            
            # Normalize embeddings
            q_norm = q_emb.norm(p=2, dim=1, keepdim=True)
            p_norm = p_emb.norm(p=2, dim=1, keepdim=True)
            eps = 1e-8
            q_emb = q_emb / (q_norm + eps)
            p_emb = p_emb / (p_norm + eps)
            
            # Calculate cosine similarity
            scores = q_emb @ p_emb.t()
            
            # For list-wise evaluation, subtract log probabilities if provided
            if self.training_method == "list_wise" and log_probs is not None:
                scores = scores - log_probs.unsqueeze(0)
            
        return scores

def main():
    """Example usage of TwoTowerEvaluator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_method = "point_wise"  # or "pair_wise" or "list_wise"
    model_path = f"/Users/haozhezhang/Documents/Python Project/two_tower_search/models/two_towers/{training_method}/final_model.pt"

    # These must match your training config
    input_dim_q = 32
    input_dim_p = 160
    hidden_q = 64
    hidden_p = 64
    embed = 32

    evaluator = TwoTowerEvaluator(
        model_path=model_path,
        input_dim_q=input_dim_q,
        input_dim_p=input_dim_p,
        hidden_dim_q=hidden_q,
        hidden_dim_p=hidden_p,
        embed_dim=embed,
        training_method=training_method,
        device=device,
    )

    # Dummy batches for demonstration
    q_batch = torch.randn(5, input_dim_q)
    p_batch = torch.randn(8, input_dim_p)
    
    # For list-wise evaluation, you might want to provide log probabilities
    log_probs = None
    if training_method == "list_wise":
        log_probs = torch.tensor(np.log(np.ones(8) / 8), device=device)  # Example uniform distribution
    
    scores = evaluator.evaluate(q_batch, p_batch, log_probs)
    print("Score matrix shape:", scores.shape)
    print(scores)

if __name__ == "__main__":
    main()
