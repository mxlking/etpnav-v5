from __future__ import annotations

import torch
from torch import Tensor, nn


class RecoveryRouterV2(nn.Module):
    PROCEED = 0
    EXPLORE = 1
    BACKTRACK = 2
    BELIEF_UPDATE = 3
    NUM_MODES = 4

    def __init__(
        self,
        d_model: int = 768,
        temperature: float = 1.0,
        tau_low: float = 1.0,
        tau_high: float = 2.5,
        pi_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.pi_threshold = float(pi_threshold)
        in_dim = int(d_model) + 9
        self.mode_head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.NUM_MODES),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

    def forward(
        self,
        s_t: Tensor,
        a_t: Tensor,
        pi_t: Tensor,
        c_micro_diag: Tensor,
        c_macro_diag: Tensor,
        u_macro: Tensor,
        progress_gap: Tensor,
        frontier_size: Tensor,
        revisit_count: Tensor,
        loop_evidence: Tensor,
    ) -> dict[str, Tensor]:
        features = torch.cat(
            [
                s_t,
                a_t.unsqueeze(-1),
                pi_t.unsqueeze(-1),
                c_micro_diag.unsqueeze(-1),
                c_macro_diag.unsqueeze(-1),
                u_macro.unsqueeze(-1),
                progress_gap.unsqueeze(-1),
                frontier_size.unsqueeze(-1),
                revisit_count.unsqueeze(-1),
                loop_evidence.unsqueeze(-1),
            ],
            dim=-1,
        )
        mode_logits = self.mode_head(features)
        mode_probs = torch.softmax(mode_logits / max(self.temperature, 1e-6), dim=-1)
        alpha_prior, alpha_progress = self.alpha_head(features).chunk(2, dim=-1)
        return {
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
            "alpha_prior": alpha_prior.squeeze(-1),
            "alpha_progress": alpha_progress.squeeze(-1),
        }

    def argmax_mode(self, mode_logits: Tensor) -> Tensor:
        return mode_logits.argmax(dim=-1)

    def bootstrap_targets(self, a_t: Tensor, pi_t: Tensor) -> Tensor:
        batch = a_t.size(0)
        target = torch.full((batch,), self.PROCEED, dtype=torch.long, device=a_t.device)
        high_mask = a_t >= self.tau_high
        mid_mask = (a_t >= self.tau_low) & (~high_mask)
        explore_mask = mid_mask & (pi_t > self.pi_threshold)
        belief_mask = mid_mask & (~explore_mask)
        target[explore_mask] = self.EXPLORE
        target[belief_mask] = self.BELIEF_UPDATE
        target[high_mask] = self.BACKTRACK
        return target
