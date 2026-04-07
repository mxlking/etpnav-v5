from __future__ import annotations

import torch
from torch import Tensor, nn


class ProgressCurvePredictor(nn.Module):
    """Predict future H-step progress curve from the posterior self-state.

    V4 predicts the agent's own short-horizon progress trajectory instead of a
    world reconstruction target. The same module also exposes per-candidate
    health scores for anticipatory reranking.
    """

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        cand_dim: int,
        horizon: int = 5,
        mlp_hidden_dim: int = 256,
        relation_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        raw_state_dim = int(hidden_dim + z_dim)
        state_dim = int(raw_state_dim if relation_dim is None else relation_dim)
        self.state_dim = state_dim
        self.state_proj = None
        if state_dim != raw_state_dim:
            self.state_proj = nn.Linear(raw_state_dim, state_dim)

        self.curve_head = nn.Sequential(
            nn.Linear(state_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, self.horizon),
            nn.Tanh(),
        )
        self.cand_curve_head = nn.Sequential(
            nn.Linear(state_dim + cand_dim, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, self.horizon),
            nn.Tanh(),
        )
        self.health_weights = nn.Linear(3, 3)

    @staticmethod
    def _state_input(h_state: Tensor, z_state_flat: Tensor) -> Tensor:
        return torch.cat([h_state, z_state_flat], dim=-1)

    def _curve_to_health(self, curve: Tensor) -> dict[str, Tensor]:
        if curve.size(-1) <= 1:
            early = curve.mean(dim=-1)
            late = curve.mean(dim=-1)
        else:
            split = min(2, curve.size(-1) - 1)
            early = curve[..., :split].mean(dim=-1)
            late = curve[..., split:].mean(dim=-1)
        trend = curve[..., -1] - curve[..., 0]
        stats = torch.stack([early, late, trend], dim=-1)
        weights = torch.softmax(self.health_weights(stats), dim=-1)
        health = torch.sum(weights * stats, dim=-1)
        return {
            "health_scalar": health,
            "health_weights": weights,
            "health_components": stats,
        }

    def forward_from_state(self, h_state: Tensor, z_state_flat: Tensor) -> dict[str, Tensor]:
        state_input = self._state_input(h_state, z_state_flat)
        if self.state_proj is not None:
            state_input = self.state_proj(state_input)
        curve = self.curve_head(state_input)
        health_out = self._curve_to_health(curve)
        return {
            "progress_curve": curve,
            **health_out,
        }

    def forward_from_relation(self, relation_state: Tensor) -> dict[str, Tensor]:
        curve = self.curve_head(relation_state)
        health_out = self._curve_to_health(curve)
        return {
            "progress_curve": curve,
            **health_out,
        }

    def forward(self, h_post: Tensor, z_post_flat: Tensor) -> dict[str, Tensor]:
        return self.forward_from_state(h_post, z_post_flat)

    def forward_per_candidate(
        self,
        h_post: Tensor,
        z_post_flat: Tensor,
        cand_feats: Tensor,
    ) -> dict[str, Tensor]:
        batch_size, num_candidates, _ = cand_feats.shape
        state_input = self._state_input(h_post, z_post_flat)
        if self.state_proj is not None:
            state_input = self.state_proj(state_input)
        state_expanded = state_input.unsqueeze(1).expand(-1, num_candidates, -1)
        cand_input = torch.cat([state_expanded, cand_feats], dim=-1)
        cand_curves = self.cand_curve_head(cand_input)
        health_out = self._curve_to_health(cand_curves)
        return {
            "cand_progress_curves": cand_curves,
            "cand_health_scores": health_out["health_scalar"],
            "cand_health_weights": health_out["health_weights"],
            "cand_health_components": health_out["health_components"],
        }

    def forward_per_relation(
        self,
        relation_states: Tensor,
    ) -> dict[str, Tensor]:
        cand_curves = self.curve_head(relation_states)
        health_out = self._curve_to_health(cand_curves)
        return {
            "cand_progress_curves": cand_curves,
            "cand_health_scores": health_out["health_scalar"],
            "cand_health_weights": health_out["health_weights"],
            "cand_health_components": health_out["health_components"],
        }
