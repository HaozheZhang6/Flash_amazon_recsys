import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

class FactorizationMachineBlock(nn.Module):
    """
    Single Factorization Machine Layer for FM.
    Output dimension is the same as input dimension.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # Linear layer for the interaction: W*x + b
        self.l1 = nn.Linear(input_dim, input_dim)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x0: Original input tensor (batch_size, input_dim)
            x: Input tensor from previous layer (batch_size, input_dim)
        Returns:
            Output tensor (batch_size, input_dim)
        """
        assert x0.shape == x.shape
        input_x = x * x0
        # Calculate interaction: x0 * (W*x + b)
        return input_x * F.relu(self.l1(x))

class NNBlock(nn.Module):
    """
    Single Block for NN.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # NN part
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            Output tensor (batch_size, input_dim)
        """
        return F.relu(self.fc(x))


class FM(nn.Module):
    """
    Factorization Machine.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fm = nn.ModuleDict(dict(
            linear = nn.Linear(config.hidden_dim, config.hidden_dim),
            fm = FactorizationMachineBlock(config.hidden_dim),
            ln_f = nn.LayerNorm(config.embed_dim)
        ))

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (batch_size, q_in)
            p: (batch_size, p_in)
        Returns:
            (batch_size, emb) - Logits for BCEWithLogitsLoss
        """
        # FM part
        out = self.fm(q, p)
        out = self.linear(out)
        out = self.ln_f(out)
        return out
