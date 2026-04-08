from __future__ import annotations

import torch
from torch import Tensor, nn


class SelfConfidence(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, prior_var_mean: Tensor, attn_entropy: Tensor, c_micro_recent_mean: Tensor) -> Tensor:
        features = torch.stack(
            [
                torch.log1p(prior_var_mean.clamp_min(0.0)),
                attn_entropy,
                c_micro_recent_mean,
            ],
            dim=-1,
        )
        return torch.sigmoid(self.mlp(features)).squeeze(-1)
