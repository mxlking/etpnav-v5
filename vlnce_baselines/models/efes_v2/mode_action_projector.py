from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from vlnce_baselines.models.efes_v2.recovery_router import RecoveryRouterV2


class ModeActionProjectorV2(nn.Module):
    def __init__(self, stop_idx: int = 0, belief_relax_penalty: float = 1.0) -> None:
        super().__init__()
        self.stop_idx = int(stop_idx)
        self.belief_relax_penalty = float(belief_relax_penalty)

    @staticmethod
    def build_candidate_masks(
        candidate_mask: Tensor,
        frontier_mask: Optional[Tensor],
        local_frontier_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        strict_mask = candidate_mask.clone()
        expanded_mask = candidate_mask.clone()
        if local_frontier_mask is not None:
            local_keep = local_frontier_mask.clone()
            if 0 <= 0 < local_keep.size(1):
                local_keep[:, 0] = True
            strict_mask = candidate_mask | (~local_keep)
        if frontier_mask is not None:
            frontier_keep = frontier_mask.clone()
            if 0 <= 0 < frontier_keep.size(1):
                frontier_keep[:, 0] = True
            expanded_mask = candidate_mask | (~frontier_keep)
        return strict_mask, expanded_mask

    def forward(
        self,
        etp_scores: Tensor,
        candidate_mask: Tensor,
        frontier_mask: Optional[Tensor],
        local_frontier_mask: Optional[Tensor],
        history_backtrack_values: Optional[Tensor],
        history_valid_mask: Optional[Tensor],
    ) -> dict[str, Tensor]:
        batch_size, num_candidates = etp_scores.shape
        neg_inf = torch.full_like(etp_scores, -1.0e9)
        strict_mask, expanded_mask = self.build_candidate_masks(candidate_mask, frontier_mask, local_frontier_mask)

        proceed_logits = etp_scores.masked_fill(strict_mask, -1.0e9)
        explore_logits = etp_scores.masked_fill(expanded_mask, -1.0e9)

        belief_logits = etp_scores.masked_fill(expanded_mask, -1.0e9)
        relaxed_only_mask = strict_mask & (~expanded_mask)
        belief_logits = torch.where(relaxed_only_mask, belief_logits - self.belief_relax_penalty, belief_logits)

        backtrack_logits = neg_inf.clone()
        for batch_idx in range(batch_size):
            chosen_idx = self.stop_idx if 0 <= self.stop_idx < num_candidates else 0
            if history_backtrack_values is not None and history_valid_mask is not None:
                backtrack_scores = history_backtrack_values[batch_idx].masked_fill(
                    ~history_valid_mask[batch_idx], -float("inf")
                )
                if torch.isfinite(backtrack_scores).any():
                    chosen_idx = int(backtrack_scores.argmax().item())
            backtrack_logits[batch_idx, chosen_idx] = 10.0

        per_mode_logits = torch.stack(
            [proceed_logits, explore_logits, backtrack_logits, belief_logits],
            dim=1,
        )
        return {
            "per_mode_logits": per_mode_logits,
            "strict_mask": strict_mask,
            "expanded_mask": expanded_mask,
        }

    @staticmethod
    def fuse_logits(per_mode_logits: Tensor, mode_probs: Tensor) -> Tensor:
        log_mode_probs = mode_probs.clamp_min(1e-8).log().unsqueeze(-1)
        return torch.logsumexp(per_mode_logits + log_mode_probs, dim=1)

    @staticmethod
    def select_hard_logits(per_mode_logits: Tensor, argmax_mode: Tensor) -> Tensor:
        batch_idx = torch.arange(per_mode_logits.size(0), device=per_mode_logits.device)
        return per_mode_logits[batch_idx, argmax_mode]
