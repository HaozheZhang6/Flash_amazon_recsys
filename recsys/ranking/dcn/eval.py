import torch
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# Attempt to import DCN model and necessary components
try:
    from recsys.ranking.dcn.model import DCN
except ImportError:
    print("Error: Could not import DCN from recsys.ranking.dcn.model. Please ensure it's correctly defined.")
    # Define a dummy DCN if not available, for the script to be syntactically valid
    class DCN(torch.nn.Module):
        def __init__(self, config): 
            super().__init__()
            self.fc = torch.nn.Linear(config.q_in + config.p_in, 1)
        def forward(self, q, p): 
            return self.fc(torch.cat([q, p], dim=1))

try:
    from recsys.ranking.dcn.train import DCNConfig, MODEL_DIR
except ImportError:
    print("Warning: Could not import DCNConfig or MODEL_DIR from recsys.ranking.dcn.train.")
    print("Using fallback definitions for DCNConfig and MODEL_DIR.")
    # Fallback DCNConfig definition
    @dataclass
    class DCNConfig:
        dcn_layers: int = 4
        dnn_layers: int = 4
        q_in: int = 32  # Example value
        p_in: int = 160 # Example value
        hidden_dim: int = 32 # Example value
        embed_dim: int = 32  # Example value
    MODEL_DIR = 'models/ranking-dcn' # Fallback MODEL_DIR

try:
    from recsys.recall.two_towers.utils import select_device
except ImportError:
    print("Warning: Could not import select_device. Defaulting to CPU.")
    def select_device(): return torch.device("cpu")


class DCNEvaluator:
    """Loads a DCN model and computes scores between query and product batches."""

    def __init__(
        self,
        model_path: str,
        config: DCNConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize evaluator by loading a pretrained DCN model.

        Args:
            model_path: Path to the .pt file with saved state_dict.
            config: DCNConfig object for model instantiation.
            device: Torch device; defaults to CPU or auto-selected.
        """
        if device is None:
            device = select_device()
        self.device = device
        self.config = config

        self.model = DCN(config=self.config).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Please ensure the model is trained and saved correctly."
            )

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def evaluate(
        self,
        query_batch: torch.Tensor,
        product_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores between queries and products using the DCN model.
        
        Args:
            query_batch: Tensor of shape (N, config.q_in)
            product_batch: Tensor of shape (M, config.p_in)
            
        Returns:
            scores: Tensor of shape (N, M) containing scores for each query-product pair
        """
        N = query_batch.size(0)
        M = product_batch.size(0)

        query_batch = query_batch.to(self.device)
        product_batch = product_batch.to(self.device)

        # Expand batches for pairwise scoring
        q_expanded = query_batch.unsqueeze(1).expand(-1, M, -1)
        p_expanded = product_batch.unsqueeze(0).expand(N, -1, -1)

        # Flatten for model input
        q_flat = q_expanded.reshape(-1, self.config.q_in)
        p_flat = p_expanded.reshape(-1, self.config.p_in)

        with torch.no_grad():
            scores_flat = self.model(q_flat, p_flat)

        return scores_flat.view(N, M)


def main():
    """Example usage of DCNEvaluator with the recall core integration."""
    device = select_device()
    
    # Use DCNConfig imported or the fallback
    config = DCNConfig()

    # Use MODEL_DIR imported or the fallback
    model_path = os.path.join(MODEL_DIR, "final_model.pt")
    
    print(f"Loading DCN model for evaluation")
    print(f"Config: q_in={config.q_in}, p_in={config.p_in}, "
          f"hidden_dim={config.hidden_dim}, embed_dim={config.embed_dim}")
    print(f"Model path: {model_path}")


    # Check if the model file exists. If not, create a dummy for demonstration.
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found.")
        if not os.path.exists(MODEL_DIR):
            try:
                os.makedirs(MODEL_DIR)
                print(f"Created directory: {MODEL_DIR}")
            except OSError as e:
                print(f"Error creating directory {MODEL_DIR}: {e}. Cannot save dummy model.")
                return
        
        print(f"Creating a dummy DCN model and saving to {model_path} for demonstration purposes.")
        try:
            dummy_model_instance = DCN(config=config)
            torch.save(dummy_model_instance.state_dict(), model_path)
            print("Dummy model saved.")
        except Exception as e:
            print(f"Error creating or saving dummy model: {e}")
            print("Please ensure the DCN model has been trained and saved to the expected location,")
            print("or update the 'model_path' and 'config' accordingly.")
            return


    try:
        evaluator = DCNEvaluator(
            model_path=model_path,
            config=config,
            device=device,
        )
    except FileNotFoundError as e:
        print(e)
        print("Evaluation cannot proceed without the model file.")
        return
    except Exception as e:
        print(f"Error initializing DCNEvaluator: {e}")
        return

    # Example usage with recall core output
    from recsys.recall.core import main as recall_main
    
    # Get recommendations from recall core
    test_queries = ['wireless headphones', 'running shoes for men']
    recall_results = recall_main(test_queries, k=30)
    
    # Convert recall results to tensors for ranking
    # This is a simplified example - you'll need to implement the actual conversion
    # based on your embedding generation process
    query_embeddings = torch.randn(len(test_queries), config.q_in)
    product_embeddings = torch.randn(len(recall_results), config.p_in)
    
    # Get scores from DCN
    scores = evaluator.evaluate(query_embeddings, product_embeddings)
    print("\nScore matrix shape:", scores.shape)
    

if __name__ == "__main__":
    main()