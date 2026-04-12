from __future__ import annotations

import torch
from torch import Tensor, nn


class SelfClarityHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        kappa_self: Tensor,
        micro_recent_mean: Tensor,
        macro_recent_mean: Tensor,
        self_mismatch_recent: Tensor,
        progress_gap: Tensor,
    ) -> Tensor:
        features = torch.stack(
            [
                kappa_self,
                micro_recent_mean,
                macro_recent_mean,
                self_mismatch_recent,
                progress_gap,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.mlp(features)).squeeze(-1)
