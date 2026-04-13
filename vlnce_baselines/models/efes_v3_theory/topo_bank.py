from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


@dataclass
class TheoryTopoBank:
    feat: Tensor
    mask: Tensor
    length: Tensor
    vp_ids: list[list[Any]]

    @classmethod
    def create(
        cls,
        num_envs: int,
        max_nodes: int,
        feat_dim: int,
        device: torch.device,
    ) -> "TheoryTopoBank":
        return cls(
            feat=torch.zeros(num_envs, max_nodes, feat_dim, device=device),
            mask=torch.zeros(num_envs, max_nodes, dtype=torch.bool, device=device),
            length=torch.zeros(num_envs, dtype=torch.long, device=device),
            vp_ids=[[] for _ in range(num_envs)],
        )

    def should_update(self, current_vp_ids: Sequence[Any]) -> Tensor:
        update = torch.zeros(self.length.size(0), dtype=torch.bool, device=self.feat.device)
        for idx, vp_id in enumerate(current_vp_ids):
            last_vp = self.vp_ids[idx][-1] if self.vp_ids[idx] else None
            if vp_id != last_vp:
                update[idx] = True
        return update

    def commit(self, update_mask: Tensor, current_vp_ids: Sequence[Any], node_feat: Tensor) -> None:
        if not bool(update_mask.any().item()):
            return
        max_nodes = self.feat.size(1)
        for idx in update_mask.nonzero(as_tuple=False).flatten().tolist():
            cur_len = int(self.length[idx].item())
            if cur_len < max_nodes:
                self.feat[idx, cur_len] = node_feat[idx].detach()
                self.mask[idx, cur_len] = True
                self.length[idx] = cur_len + 1
                self.vp_ids[idx].append(current_vp_ids[idx])
            else:
                self.feat[idx, :-1] = self.feat[idx, 1:].clone()
                self.feat[idx, -1] = node_feat[idx].detach()
                self.mask[idx].fill_(True)
                self.vp_ids[idx] = self.vp_ids[idx][1:] + [current_vp_ids[idx]]

    def index_select(self, keep_indices: Sequence[int]) -> "TheoryTopoBank":
        index = torch.as_tensor(keep_indices, device=self.feat.device, dtype=torch.long)
        return TheoryTopoBank(
            feat=self.feat.index_select(0, index),
            mask=self.mask.index_select(0, index),
            length=self.length.index_select(0, index),
            vp_ids=[self.vp_ids[i] for i in keep_indices],
        )
