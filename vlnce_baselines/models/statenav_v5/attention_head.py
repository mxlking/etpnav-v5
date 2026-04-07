from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class AttentionHead(nn.Module):
    """Predict next-step language attention from unified rollout state."""

    def __init__(
        self,
        relation_dim: int,
        lang_dim: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.query_proj = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lang_key = nn.Linear(lang_dim, hidden_dim)
        self.prior_gate = nn.Linear(relation_dim, 1)

    @staticmethod
    def _masked_softmax(scores: Tensor, valid_mask: Tensor) -> Tensor:
        has_valid = valid_mask.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        safe_scores = torch.where(has_valid, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(safe_scores, dim=-1)
        weights = weights * valid_mask.to(weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return torch.where(has_valid, weights / denom, torch.zeros_like(weights))

    def forward(
        self,
        *,
        attn_post: Tensor,
        unified_states: Tensor,
        lang_tokens: Tensor,
        lang_mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, num_candidates, _ = unified_states.shape
        valid_mask = lang_mask.bool() if lang_mask is not None else torch.ones_like(attn_post, dtype=torch.bool)
        query = self.query_proj(unified_states)
        keys = self.lang_key(lang_tokens)
        scores = torch.einsum("bkd,bld->bkl", query, keys)
        scores = scores / max(float(self.hidden_dim) ** 0.5, 1.0)
        prior_gate = torch.sigmoid(self.prior_gate(unified_states))
        prior_bias = torch.log(attn_post.clamp_min(1e-6)).unsqueeze(1)
        scores = scores + prior_gate * prior_bias
        return self._masked_softmax(scores, valid_mask.unsqueeze(1).expand(-1, num_candidates, -1))
