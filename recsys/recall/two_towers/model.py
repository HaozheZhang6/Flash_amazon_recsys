import torch
from torch import nn
from torch.nn import functional as F


class QueryTowerLinear(nn.Module):
    """
    Query tower with two linear layers and layer normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, embedding_dim)
        """
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class ProductTowerLinear(nn.Module):
    """
    Product tower with two linear layers and layer normalization.
    """
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim)
        Returns:
            (batch_size, embedding_dim)
        """
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class TwoTowerModel(nn.Module):
    """
    Combines query and product towers; outputs similarity logits.
    """
    def __init__(self, q_in: int, p_in: int, q_hidden: int, p_hidden: int, emb: int):
        super().__init__()
        self.query_tower = QueryTowerLinear(q_in, q_hidden, emb)
        self.product_tower = ProductTowerLinear(p_in, p_hidden, emb)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        query_emb = self.query_tower(q)
        prod_emb = self.product_tower(p)
        # L2 norms
        query_norm = query_emb.norm(p=2, dim=1, keepdim=True)
        prod_norm = prod_emb.norm(p=2, dim=1, keepdim=True)

        # normalize and avoid division by zero
        eps = 1e-8
        query_emb = query_emb / (query_norm + eps)
        prod_emb = prod_emb / (prod_norm + eps)

        return torch.sum(query_emb * prod_emb, dim=1, keepdim=True)

