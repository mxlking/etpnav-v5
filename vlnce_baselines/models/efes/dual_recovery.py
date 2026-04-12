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
        # Always evaluate the modulation head so its parameters stay in the
        # autograd/DDP graph on every step, even when no sample enters a
        # recovery branch on a given rank.
        alphas = self.belief_mod(torch.cat([s_t, a_t.unsqueeze(-1), pi_t.unsqueeze(-1)], dim=-1))
        alphas = alphas.to(dtype=alpha_prior.dtype)

        high_mask = a_t >= self.tau_high
        mid_mask = (a_t >= self.tau_low) & (~high_mask)
        explore_mask = mid_mask & (pi_t > self.pi_threshold)
        belief_mask = mid_mask & (~explore_mask)

        mode[explore_mask] = self.EXPLORE
        mode[belief_mask] = self.BELIEF_UPDATE
        mode[high_mask] = self.BACKTRACK

        mod_mask = belief_mask | high_mask
        alpha_prior = torch.where(mod_mask, alphas[:, 0], alpha_prior)
        alpha_progress = torch.where(mod_mask, alphas[:, 1], alpha_progress)
        alpha_progress = torch.where(high_mask, torch.zeros_like(alpha_progress), alpha_progress)
        alpha_prior = alpha_prior + (alphas[:, 0] * 0.0)
        alpha_progress = alpha_progress + (alphas[:, 1] * 0.0)

        return {
            "recovery_mode": mode,
            "alpha_prior": alpha_prior,
            "alpha_progress": alpha_progress,
        }
