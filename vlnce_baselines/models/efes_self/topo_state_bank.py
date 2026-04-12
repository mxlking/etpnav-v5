from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass
class TopoStateBankV2:
    bank_feat: Tensor
    bank_mask: Tensor
    bank_vp_ids: list[list[Any]]
    bank_len: Tensor
    last_trigger_step: Tensor

    @classmethod
    def create(
        cls,
        num_envs: int,
        max_nodes: int,
        feat_dim: int,
        device: torch.device | str,
    ) -> "TopoStateBankV2":
        return cls(
            bank_feat=torch.zeros(num_envs, max_nodes, feat_dim, device=device),
            bank_mask=torch.zeros(num_envs, max_nodes, dtype=torch.bool, device=device),
            bank_vp_ids=[[] for _ in range(num_envs)],
            bank_len=torch.zeros(num_envs, dtype=torch.long, device=device),
            last_trigger_step=torch.zeros(num_envs, dtype=torch.long, device=device),
        )

    def commit(
        self,
        update_mask: Tensor,
        current_vp_ids: Sequence[Any],
        compressed_feat: Tensor,
        current_step: int,
    ) -> None:
        if not bool(update_mask.any().item()):
            return
        max_nodes = self.bank_feat.size(1)
        for batch_idx in torch.nonzero(update_mask, as_tuple=False).flatten().tolist():
            cur_len = int(self.bank_len[batch_idx].item())
            if cur_len < max_nodes:
                write_idx = cur_len
                self.bank_feat[batch_idx, write_idx] = compressed_feat[batch_idx].detach()
                self.bank_mask[batch_idx, write_idx] = True
                self.bank_len[batch_idx] = cur_len + 1
                self.bank_vp_ids[batch_idx].append(current_vp_ids[batch_idx])
            else:
                self.bank_feat[batch_idx, :-1] = self.bank_feat[batch_idx, 1:].clone()
                self.bank_feat[batch_idx, -1] = compressed_feat[batch_idx].detach()
                self.bank_mask[batch_idx].fill_(True)
                self.bank_vp_ids[batch_idx] = self.bank_vp_ids[batch_idx][1:] + [current_vp_ids[batch_idx]]
            self.last_trigger_step[batch_idx] = int(current_step)

    def build_update_mask(
        self,
        current_vp_ids: Sequence[Any],
        candidate_counts: Tensor,
        current_step: int,
        topo_novelty: Tensor,
        *,
        decision_candidate_min: int,
        timeout_steps: int,
        novelty_threshold: float,
        cooldown_steps: int,
    ) -> Tensor:
        update_mask = self.bank_len.eq(0)
        current_step_i = int(current_step)
        cooldown_ready = (current_step_i - self.last_trigger_step) >= int(cooldown_steps)
        for batch_idx, current_vp in enumerate(current_vp_ids):
            if not bool(cooldown_ready[batch_idx].item()):
                continue
            seen_vps = self.bank_vp_ids[batch_idx]
            is_new_vp = current_vp is not None and current_vp not in seen_vps
            enough_candidates = int(candidate_counts[batch_idx].item()) >= int(decision_candidate_min)
            timeout_ready = (
                current_step_i - int(self.last_trigger_step[batch_idx].item())
            ) >= int(timeout_steps)
            novelty_ready = float(topo_novelty[batch_idx].item()) >= float(novelty_threshold)
            if (is_new_vp and enough_candidates) or novelty_ready or timeout_ready:
                update_mask[batch_idx] = True
        return update_mask

    @staticmethod
    def unique_vp_delta(recent_vps: Sequence[Any], current_vp: Any) -> int:
        all_vps = [vp for vp in recent_vps if vp is not None] + ([current_vp] if current_vp is not None else [])
        return max(len(set(all_vps)) - 1, 0)

    def index_select(self, keep_indices: Sequence[int]) -> "TopoStateBankV2":
        keep_tensor = torch.as_tensor(keep_indices, device=self.bank_feat.device, dtype=torch.long)
        return TopoStateBankV2(
            bank_feat=self.bank_feat.index_select(0, keep_tensor),
            bank_mask=self.bank_mask.index_select(0, keep_tensor),
            bank_vp_ids=[self.bank_vp_ids[int(idx)] for idx in keep_indices],
            bank_len=self.bank_len.index_select(0, keep_tensor),
            last_trigger_step=self.last_trigger_step.index_select(0, keep_tensor),
        )
