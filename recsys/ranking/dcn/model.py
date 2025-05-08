import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

class CrossNetworkBlock(nn.Module):
    """
    Single Cross Layer for DCN.
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
        # Calculate interaction: x0 * (W*x + b)
        interaction = x0 * F.relu(self.l1(x))
        # Add residual connection
        output = interaction + x
        return output

class NNBlock(nn.Module):
    """
    Single Block for dnn.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        # DNN part
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            Output tensor (batch_size, input_dim)
        """
        return F.relu(self.fc(x))


class DCN(nn.Module):
    """
    Deep & Cross Network.
    """
    def __init__(self, config):
        super().__init__()
        concat_dim = config.q_in + config.p_in
        self.config = config
        self.dcn = nn.ModuleDict(dict(
            fc_reduce = nn.Linear(concat_dim, config.hidden_dim),
            cn = nn.ModuleList([CrossNetworkBlock(config.hidden_dim) for _ in range(config.dcn_layers)]),
            nn = nn.ModuleList([NNBlock(config.hidden_dim) for _ in range(config.dnn_layers)]),
            fc2 = nn.Linear(config.hidden_dim*2, config.embed_dim),
            ln_f = nn.LayerNorm(config.embed_dim)
        ))
        self.lmhead = nn.Linear(config.embed_dim, 1)



    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q: (batch_size, q_in)
            p: (batch_size, p_in)
        Returns:
            (batch_size, emb) - Logits for BCEWithLogitsLoss
        """
        # Concatenate query and product features
        x_initial_concat = torch.cat([q, p], dim=1) # shape: (batch_size, q_in + p_in)

        x_reduced_input = F.relu(self.dcn.fc_reduce(x_initial_concat)) # shape: (batch_size, hidden_dim)

        # Deep Network part
        x_deep = x_reduced_input
        for block in self.dcn.nn:
            x_deep = block(x_deep) # shape: (batch_size, hidden_dim)
        
        # Cross Network part
        x0_cross = x_reduced_input # This is the x0 for all cross layers
        x_current_cross = x_reduced_input # This is the x_l that evolves
        for block in self.dcn.cn:
            x_current_cross = block(x0_cross, x_current_cross) # Pass both x0 and x_l

        # Concatenate outputs from deep and cross parts
        x_concat = torch.cat([x_deep, x_current_cross], dim=1) # shape: (batch_size, hidden_dim*2)

        x = self.dcn.fc2(x_concat) # shape: (batch_size, embed_dim)
        x = self.dcn.ln_f(x)
        out = self.lmhead(x)
        return out
