from __future__ import annotations

import torch
from torch import Tensor, nn


class DualRecovery(nn.Module):
    PROCEED = 0
    EXPLORE = 1
    BACKTRACK = 2
    BELIEF_UPDATE = 3

    def __init__(
        self,
        d_model: int = 768,
        tau_low: float = 1.0,
        tau_high: float = 2.5,
        pi_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.pi_threshold = float(pi_threshold)
        self.belief_mod = nn.Sequential(
            nn.Linear(int(d_model) + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(self, s_t: Tensor, a_t: Tensor, pi_t: Tensor) -> dict[str, Tensor]:
        batch = s_t.size(0)
        device = s_t.device
        mode = torch.full((batch,), self.PROCEED, dtype=torch.long, device=device)
        alpha_prior = torch.ones(batch, device=device, dtype=s_t.dtype)
        alpha_progress = torch.ones(batch, device=device, dtype=s_t.dtype)

        high_mask = a_t >= self.tau_high
        mid_mask = (a_t >= self.tau_low) & (~high_mask)
        explore_mask = mid_mask & (pi_t > self.pi_threshold)
        belief_mask = mid_mask & (~explore_mask)

        mode[explore_mask] = self.EXPLORE
        mode[belief_mask] = self.BELIEF_UPDATE
        mode[high_mask] = self.BACKTRACK

        mod_mask = belief_mask | high_mask
        if bool(mod_mask.any().item()):
            alphas = self.belief_mod(torch.cat([s_t, a_t.unsqueeze(-1), pi_t.unsqueeze(-1)], dim=-1))
            alphas = alphas.to(dtype=alpha_prior.dtype)
            alpha_prior[mod_mask] = alphas[mod_mask, 0]
            alpha_progress[mod_mask] = alphas[mod_mask, 1]
            alpha_progress[high_mask] = 0.0

        return {
            "recovery_mode": mode,
            "alpha_prior": alpha_prior,
            "alpha_progress": alpha_progress,
        }
