from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from vlnce_baselines.models.statenav_v6.node_surprise_head import NodeSurpriseHead


@dataclass
class TopoStateBank:
    topo_states: Tensor
    topo_feats: Tensor
    has_state: Tensor
    micro_steps: Tensor
    last_vp_ids: list[Any]

    @classmethod
    def create(
        cls,
        num_envs: int,
        topo_state_dim: int,
        feat_dim: int,
        device: torch.device | str,
    ) -> "TopoStateBank":
        return cls(
            topo_states=torch.zeros(num_envs, topo_state_dim, device=device),
            topo_feats=torch.zeros(num_envs, feat_dim, device=device),
            has_state=torch.zeros(num_envs, dtype=torch.bool, device=device),
            micro_steps=torch.zeros(num_envs, dtype=torch.long, device=device),
            last_vp_ids=[None for _ in range(num_envs)],
        )

    def compute_novelty(self, current_feat: Tensor) -> Tensor:
        return NodeSurpriseHead.cosine_novelty(
            current_feat=current_feat,
            reference_feat=self.topo_feats,
            valid_mask=self.has_state,
        )

    def build_update_mask(
        self,
        current_vp_ids: Sequence[Any],
        candidate_counts: Tensor,
        novelty_scores: Tensor,
        novelty_tau: float,
        max_micro_steps: int,
        decision_candidate_min: int = 2,
    ) -> Tensor:
        update_mask = self.has_state.logical_not()
        for idx, current_vp in enumerate(current_vp_ids):
            if self.last_vp_ids[idx] is None or current_vp != self.last_vp_ids[idx]:
                update_mask[idx] = True

        update_mask |= candidate_counts >= int(decision_candidate_min)
        update_mask |= novelty_scores > float(novelty_tau)
        update_mask |= self.micro_steps >= int(max_micro_steps)
        return update_mask

    def commit(
        self,
        update_mask: Tensor,
        current_vp_ids: Sequence[Any],
        topo_state: Tensor,
        topo_feat: Tensor,
    ) -> None:
        self.micro_steps = self.micro_steps + 1
        if not bool(update_mask.any().item()):
            return
        self.topo_states[update_mask] = topo_state[update_mask].detach()
        self.topo_feats[update_mask] = topo_feat[update_mask].detach()
        self.has_state[update_mask] = True
        self.micro_steps[update_mask] = 0
        update_indices = torch.nonzero(update_mask, as_tuple=False).flatten().tolist()
        for idx in update_indices:
            self.last_vp_ids[idx] = current_vp_ids[idx]

    def index_select(self, keep_indices: Sequence[int]) -> "TopoStateBank":
        keep_tensor = torch.as_tensor(keep_indices, device=self.topo_states.device, dtype=torch.long)
        return TopoStateBank(
            topo_states=self.topo_states.index_select(0, keep_tensor),
            topo_feats=self.topo_feats.index_select(0, keep_tensor),
            has_state=self.has_state.index_select(0, keep_tensor),
            micro_steps=self.micro_steps.index_select(0, keep_tensor),
            last_vp_ids=[self.last_vp_ids[int(idx)] for idx in keep_indices],
        )
