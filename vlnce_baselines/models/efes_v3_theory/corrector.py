from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class TheoryCorrector(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        cand_dim: int = 768,
        hidden_dim: int = 256,
        delta_bound: float = 1.0,
        gate_bias: float = -2.0,
        lambda_scale: float = 1.0,
        use_dual_gate: bool = True,
        use_shifted_surprise: bool = True,
    ) -> None:
        super().__init__()
        self.delta_bound = float(delta_bound)
        self.lambda_scale = max(float(lambda_scale), 1e-6)
        self.use_dual_gate = bool(use_dual_gate)
        self.use_shifted_surprise = bool(use_shifted_surprise)

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
        u_signal: Tensor,
        etp_logits: Tensor,
        cand_embeds: Tensor,
        cand_mask: Tensor,
    ) -> Dict[str, Tensor]:
        ctx = self.ctx_proj(torch.cat([s_t, u_signal.unsqueeze(-1)], dim=-1))
        cand = self.cand_proj(cand_embeds)
        hidden = torch.tanh(cand + ctx.unsqueeze(1))
        delta = torch.tanh(self.delta_head(hidden).squeeze(-1)) * self.delta_bound

        gate_raw = self.gate_net(u_signal.unsqueeze(-1)).squeeze(-1)
        if self.use_dual_gate:
            lambda_hat = torch.nn.functional.softplus(gate_raw)
            gate = torch.clamp(lambda_hat / self.lambda_scale, min=0.0, max=1.0)
        else:
            lambda_hat = torch.nn.functional.softplus(gate_raw)
            gate = torch.sigmoid(gate_raw)

        corrected = etp_logits + gate.unsqueeze(-1) * delta
        corrected = corrected.masked_fill(cand_mask, -1.0e4)
        return {
            "corrected_logits": corrected,
            "gate": gate,
            "lambda_hat": lambda_hat,
            "delta": delta,
            "delta_abs_max": delta.detach().abs().amax(dim=-1),
        }
