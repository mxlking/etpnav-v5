from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NodeSurpriseHead(nn.Module):
    """Predicts the expected next topological-node feature."""

    def __init__(
        self,
        topo_state_dim: int,
        feat_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.topo_state_dim = int(topo_state_dim)
        self.feat_dim = int(feat_dim)
        self.hidden_dim = int(hidden_dim)
        self.predictor = nn.Sequential(
            nn.Linear(self.topo_state_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.hidden_dim, self.feat_dim),
        )

    def forward(self, topo_state: Tensor) -> Tensor:
        return self.predictor(topo_state)

    @staticmethod
    def mse_surprise(
        predicted_feat: Tensor,
        target_feat: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        surprise = F.mse_loss(predicted_feat, target_feat, reduction="none").mean(dim=-1)
        if valid_mask is None:
            return surprise
        masked = torch.zeros_like(surprise)
        masked[valid_mask] = surprise[valid_mask]
        return masked

    @staticmethod
    def cosine_novelty(
        current_feat: Tensor,
        reference_feat: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        novelty = 1.0 - F.cosine_similarity(current_feat, reference_feat, dim=-1, eps=1e-6)
        novelty = novelty.clamp_min(0.0)
        if valid_mask is None:
            return novelty
        masked = torch.zeros_like(novelty)
        masked[valid_mask] = novelty[valid_mask]
        return masked
