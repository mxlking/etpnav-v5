from __future__ import annotations

from torch import Tensor, nn


class StateHead(nn.Module):
    """Read out physical relation state from unified rollout representation."""

    def __init__(self, relation_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, relation_dim),
        )

    def forward(self, unified_state: Tensor) -> Tensor:
        return self.net(unified_state)
