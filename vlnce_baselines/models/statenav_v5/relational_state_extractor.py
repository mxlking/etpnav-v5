from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class RelationalStateExtractor(nn.Module):
    """Extract relation state by querying external world tokens with internal state."""

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        token_dim: int,
        relation_dim: int = 512,
        max_visual_tokens: int = 64,
        max_graph_tokens: int = 512,
        max_extra_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.relation_dim = int(relation_dim)

        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, relation_dim),
            nn.Tanh(),
        )
        self.visual_proj = nn.Linear(token_dim, relation_dim)
        self.graph_proj = nn.Linear(token_dim, relation_dim)
        self.extra_proj = nn.Linear(token_dim, relation_dim)

        self.visual_type = nn.Parameter(torch.zeros(1, 1, relation_dim))
        self.graph_type = nn.Parameter(torch.zeros(1, 1, relation_dim))
        self.extra_type = nn.Parameter(torch.zeros(1, 1, relation_dim))

        self.visual_pos = nn.Embedding(max_visual_tokens, relation_dim)
        self.graph_pos = nn.Embedding(max_graph_tokens, relation_dim)
        self.extra_pos = nn.Embedding(max_extra_tokens, relation_dim)

        self.output_mlp = nn.Sequential(
            nn.Linear(relation_dim * 2, relation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(relation_dim, relation_dim),
        )

    @staticmethod
    def _make_positions(length: int, device: torch.device, max_len: int) -> Tensor:
        if length <= 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        pos = torch.arange(length, device=device)
        if length > max_len:
            pos = pos.clamp(max=max_len - 1)
        return pos

    @staticmethod
    def _masked_softmax(scores: Tensor, valid_mask: Tensor) -> Tensor:
        has_valid = valid_mask.any(dim=-1, keepdim=True)
        masked_scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        safe_scores = torch.where(has_valid, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(safe_scores, dim=-1)
        weights = weights * valid_mask.to(weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        return torch.where(has_valid, weights / denom, torch.zeros_like(weights))

    def _project_visual_tokens(self, x_tokens: Tensor) -> Tensor:
        bsz, n, _ = x_tokens.shape
        pos = self._make_positions(n, x_tokens.device, self.visual_pos.num_embeddings)
        pos_emb = self.visual_pos(pos).unsqueeze(0)
        return self.visual_proj(x_tokens) + self.visual_type + pos_emb

    def _project_graph_tokens(self, g_tokens: Tensor) -> Tensor:
        bsz, n, _ = g_tokens.shape
        pos = self._make_positions(n, g_tokens.device, self.graph_pos.num_embeddings)
        pos_emb = self.graph_pos(pos).unsqueeze(0)
        return self.graph_proj(g_tokens) + self.graph_type + pos_emb

    def _project_extra_tokens(self, extra_tokens: Tensor) -> Tensor:
        bsz, n, _ = extra_tokens.shape
        pos = self._make_positions(n, extra_tokens.device, self.extra_pos.num_embeddings)
        pos_emb = self.extra_pos(pos).unsqueeze(0)
        return self.extra_proj(extra_tokens) + self.extra_type + pos_emb

    def forward(
        self,
        *,
        h_state: Tensor,
        z_state_flat: Tensor,
        x_tokens: Tensor,
        x_mask: Optional[Tensor],
        g_tokens: Tensor,
        g_mask: Optional[Tensor],
        extra_tokens: Optional[Tensor] = None,
        extra_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        query = self.query_proj(torch.cat([h_state, z_state_flat], dim=-1)).unsqueeze(1)

        world_tokens = [self._project_visual_tokens(x_tokens), self._project_graph_tokens(g_tokens)]
        world_masks = []
        if x_mask is None:
            world_masks.append(torch.ones(x_tokens.shape[:2], device=x_tokens.device, dtype=torch.bool))
        else:
            world_masks.append(x_mask.bool())
        if g_mask is None:
            world_masks.append(torch.ones(g_tokens.shape[:2], device=g_tokens.device, dtype=torch.bool))
        else:
            world_masks.append(g_mask.bool())

        if extra_tokens is not None:
            if extra_tokens.ndim == 2:
                extra_tokens = extra_tokens.unsqueeze(1)
            world_tokens.append(self._project_extra_tokens(extra_tokens))
            if extra_mask is None:
                world_masks.append(
                    torch.ones(extra_tokens.shape[:2], device=extra_tokens.device, dtype=torch.bool)
                )
            else:
                if extra_mask.ndim == 1:
                    extra_mask = extra_mask.unsqueeze(1)
                world_masks.append(extra_mask.bool())

        kv = torch.cat(world_tokens, dim=1)
        valid_mask = torch.cat(world_masks, dim=1)

        scores = torch.einsum("bid,bjd->bij", query, kv).squeeze(1)
        scores = scores / max(float(self.relation_dim) ** 0.5, 1.0)
        attn = self._masked_softmax(scores, valid_mask)
        context = torch.einsum("bn,bnd->bd", attn, kv)
        relation = self.output_mlp(torch.cat([query.squeeze(1), context], dim=-1))

        return {
            "relation_state": relation,
            "relation_attn": attn,
            "relation_context": context,
        }
