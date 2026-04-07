from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class AttnHealthScorer(nn.Module):
    """Score whether a predicted attention shift is semantically healthy."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    @staticmethod
    def _masked_entropy(weights: Tensor, valid_mask: Tensor) -> Tensor:
        eps = torch.finfo(weights.dtype).eps
        masked = weights * valid_mask.to(weights.dtype)
        masked = masked / masked.sum(dim=-1, keepdim=True).clamp_min(eps)
        return -(masked.clamp_min(eps) * torch.log(masked.clamp_min(eps))).sum(dim=-1)

    def forward(
        self,
        *,
        predicted_attn: Tensor,
        attn_post: Tensor,
        lang_tokens: Optional[Tensor] = None,
        lang_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        del lang_tokens
        batch_size, num_candidates, lang_len = predicted_attn.shape
        valid_mask = (
            lang_mask.bool()
            if lang_mask is not None
            else torch.ones(batch_size, lang_len, device=predicted_attn.device, dtype=torch.bool)
        )
        positions = torch.linspace(
            0.0,
            1.0,
            steps=lang_len,
            device=predicted_attn.device,
            dtype=predicted_attn.dtype,
        )
        current_pos = (attn_post * positions.unsqueeze(0)).sum(dim=-1)
        predicted_pos = (predicted_attn * positions.view(1, 1, -1)).sum(dim=-1)
        current_entropy = self._masked_entropy(attn_post, valid_mask)
        predicted_entropy = self._masked_entropy(
            predicted_attn,
            valid_mask.unsqueeze(1).expand(-1, num_candidates, -1),
        )
        current_peak = attn_post.max(dim=-1).values
        predicted_peak = predicted_attn.max(dim=-1).values
        delta = predicted_pos - current_pos.unsqueeze(1)
        feature = torch.stack(
            [
                predicted_pos,
                delta,
                predicted_peak,
                -predicted_entropy,
                current_pos.unsqueeze(1).expand(-1, num_candidates),
                current_peak.unsqueeze(1).expand(-1, num_candidates),
                -current_entropy.unsqueeze(1).expand(-1, num_candidates),
            ],
            dim=-1,
        )
        attn_health = self.mlp(feature).squeeze(-1)
        return {
            "attn_health_scores": attn_health,
            "current_attn_position": current_pos,
            "predicted_attn_position": predicted_pos,
        }
