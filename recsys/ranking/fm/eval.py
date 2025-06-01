import torch
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# Attempt to import FM model and necessary components
try:
    from recsys.ranking.fm.model import FM
except ImportError:
    print("Error: Could not import FM from recsys.ranking.fm.model. Please ensure it's correctly defined.")

try:
    from recsys.ranking.fm.train import FMConfig, MODEL_DIR
except ImportError:
    print("Warning: Could not import FMConfig or MODEL_DIR from recsys.ranking.fm.train.")
    print("Using fallback definitions for FMConfig and MODEL_DIR.")
    # Fallback FMConfig definition
    @dataclass
    class FMConfig:
        q_in: int = 32  # Example value
        p_in: int = 160 # Example value
        hidden_dim: int = 32 # Example value
        embed_dim: int = 32  # Example value
    MODEL_DIR = 'models/ranking-fm' # Fallback MODEL_DIR

try:
    from recsys.recall.two_towers.utils import select_device
except ImportError:
    print("Warning: Could not import select_device. Defaulting to CPU.")
    def select_device(): return torch.device("cpu")


class FMEvaluator:
    """Loads a FM model and computes scores between query and product batches."""

    def __init__(
        self,
        model_path: str,
        config: FMConfig,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize evaluator by loading a pretrained FM model.

        Args:
            model_path: Path to the .pt file with saved state_dict.
            config: FMConfig object for model instantiation.
            device: Torch device; defaults to CPU or auto-selected.
        """
        if device is None:
            device = select_device()
        self.device = device
        self.config = config

        self.model = FM(config=self.config).to(self.device)

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
        Compute scores between queries and products using the FM model.
        
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
    """Example usage of FMEvaluator with the recall core integration."""
    device = select_device()
    
    # Use FMConfig imported or the fallback
    config = FMConfig()

    # Use MODEL_DIR imported or the fallback
    model_path = os.path.join(MODEL_DIR, "final_model.pt")
    
    print(f"Loading FM model for evaluation")
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
            dummy_model_instance = FM(config=config)
            torch.save(dummy_model_instance.state_dict(), model_path)
            print("Dummy model saved.")
        except Exception as e:
            print(f"Error creating or saving dummy model: {e}")
            print("Please ensure the FM model has been trained and saved to the expected location,")
            print("or update the 'model_path' and 'config' accordingly.")
            return


    try:
        evaluator = FMEvaluator(
            model_path=model_path,
            config=config,
            device=device,
        )
    except FileNotFoundError as e:
        print(e)
        print("Evaluation cannot proceed without the model file.")
        return
    except Exception as e:
        print(f"Error initializing FMEvaluator: {e}")
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
    
    # Get scores from FM
    scores = evaluator.evaluate(query_embeddings, product_embeddings)
    print("\nScore matrix shape:", scores.shape)
    

if __name__ == "__main__":
    main()