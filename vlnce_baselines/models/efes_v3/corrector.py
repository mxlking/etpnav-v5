from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class Corrector(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        cand_dim: int = 768,
        hidden_dim: int = 256,
        max_delta: float = 1.0,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.ctx_proj = nn.Sequential(
            nn.Linear(int(d_model) + 1, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.cand_proj = nn.Linear(int(cand_dim), int(hidden_dim))
        self.delta_head = nn.Linear(int(hidden_dim), 1)
        self.gate_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, float(gate_bias))

    def forward(
        self,
        s_t: Tensor,
        surprise_t: Tensor,
        etp_logits: Tensor,
        cand_embeds: Tensor,
        cand_mask: Tensor,
    ) -> Dict[str, Tensor]:
        ctx = self.ctx_proj(torch.cat([s_t, surprise_t.unsqueeze(-1)], dim=-1))
        cand = self.cand_proj(cand_embeds)
        hidden = torch.tanh(cand + ctx.unsqueeze(1))
        delta = torch.tanh(self.delta_head(hidden).squeeze(-1)) * self.max_delta
        gate = torch.sigmoid(self.gate_net(surprise_t.unsqueeze(-1)).squeeze(-1))
        corrected = etp_logits + gate.unsqueeze(-1) * delta
        corrected = corrected.masked_fill(cand_mask, -1.0e4)
        return {
            "corrected_logits": corrected,
            "gate": gate,
            "delta": delta,
        }
