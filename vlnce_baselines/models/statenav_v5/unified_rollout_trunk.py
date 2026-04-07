from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class UnifiedRolloutTrunk(nn.Module):
    """Candidate-conditioned unified rollout trunk for V5.

    This module implements the Phase-2 core described in docs/v5/v5.md:
    evolve the internal state one step forward under a candidate action, then
    query the current world tokens plus the candidate token to produce a single
    unified relation representation ``u_next_k``.
    """

    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        token_dim: int,
        cand_dim: int,
        relation_dim: int = 512,
        ffn_hidden_dim: int = 1024,
        max_visual_tokens: int = 64,
        max_graph_tokens: int = 512,
    ) -> None:
        super().__init__()
        self.relation_dim = int(relation_dim)

        self.query_proj = nn.Linear(hidden_dim + z_dim, relation_dim)
        self.visual_proj = nn.Linear(token_dim, relation_dim)
        self.graph_proj = nn.Linear(token_dim, relation_dim)
        self.cand_proj = nn.Linear(cand_dim, relation_dim)

        self.visual_type = nn.Parameter(torch.zeros(1, 1, relation_dim))
        self.graph_type = nn.Parameter(torch.zeros(1, 1, relation_dim))
        self.cand_type = nn.Parameter(torch.zeros(1, 1, relation_dim))

        self.visual_pos = nn.Embedding(max_visual_tokens, relation_dim)
        self.graph_pos = nn.Embedding(max_graph_tokens, relation_dim)

        self.norm1 = nn.LayerNorm(relation_dim)
        self.norm2 = nn.LayerNorm(relation_dim)
        self.ffn = nn.Sequential(
            nn.Linear(relation_dim, ffn_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ffn_hidden_dim, relation_dim),
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

    @staticmethod
    def _prior_sample_flat(
        latent_model: nn.Module,
        h_prior: Tensor,
        temp: float = 1.0,
        hard: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Sample a future prior latent while preserving gradients during training.

        The previous implementation used ``argmax -> onehot`` for every rollout
        step, which made the future prior branch effectively non-trainable from
        rollout losses. We keep deterministic argmax behaviour in eval mode, but
        use straight-through Gumbel-Softmax during training so gradients can flow
        back into ``latent_model.prior_net``.
        """
        batch_size = h_prior.size(0)
        prior_logits = latent_model.prior_net(h_prior).view(
            batch_size,
            latent_model.latent_groups,
            latent_model.latent_classes,
        )
        if latent_model.training:
            tau = max(float(temp), 1e-5)
            z_onehot = F.gumbel_softmax(
                prior_logits.view(-1, latent_model.latent_classes),
                tau=tau,
                hard=bool(hard),
                dim=-1,
            ).view(batch_size, latent_model.latent_groups, latent_model.latent_classes)
        else:
            z_indices = prior_logits.argmax(dim=-1)
            z_onehot = F.one_hot(
                z_indices,
                num_classes=latent_model.latent_classes,
            ).to(prior_logits.dtype)
        return prior_logits, z_onehot.flatten(start_dim=1)

    def _project_visual_tokens(self, x_tokens: Tensor) -> Tensor:
        length = x_tokens.size(1)
        pos = self._make_positions(length, x_tokens.device, self.visual_pos.num_embeddings)
        return self.visual_proj(x_tokens) + self.visual_type + self.visual_pos(pos).unsqueeze(0)

    def _project_graph_tokens(self, g_tokens: Tensor) -> Tensor:
        length = g_tokens.size(1)
        pos = self._make_positions(length, g_tokens.device, self.graph_pos.num_embeddings)
        return self.graph_proj(g_tokens) + self.graph_type + self.graph_pos(pos).unsqueeze(0)

    def forward(
        self,
        *,
        transition_model: nn.Module,
        latent_model: nn.Module,
        action_encoder: nn.Module,
        h_post: Tensor,
        z_post_flat: Tensor,
        relation_state: Optional[Tensor],
        cand_feats: Tensor,
        lang_tokens: Tensor,
        x_tokens: Tensor,
        g_tokens: Tensor,
        lang_mask: Optional[Tensor] = None,
        x_mask: Optional[Tensor] = None,
        g_mask: Optional[Tensor] = None,
        candidate_mask: Optional[Tensor] = None,
        temp: float = 1.0,
        hard: bool = True,
    ) -> dict[str, Tensor]:
        batch_size, num_candidates, cand_dim = cand_feats.shape
        if num_candidates == 0:
            zero_state = h_post.new_zeros(batch_size, 0, self.relation_dim)
            zero_h = h_post.new_zeros(batch_size, 0, h_post.size(-1))
            zero_z = z_post_flat.new_zeros(batch_size, 0, z_post_flat.size(-1))
            zero_attn = h_post.new_zeros(batch_size, 0, x_tokens.size(1) + g_tokens.size(1) + 1)
            return {
                "unified_state": zero_state,
                "unified_attn": zero_attn,
                "candidate_h_next": zero_h,
                "candidate_z_next": zero_z,
            }

        action_emb = action_encoder(cand_feats)
        h_repeated = h_post.unsqueeze(1).expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, -1)
        z_repeated = z_post_flat.unsqueeze(1).expand(-1, num_candidates, -1).reshape(
            batch_size * num_candidates, -1
        )
        action_repeated = action_emb.reshape(batch_size * num_candidates, -1)
        lang_tokens_repeated = (
            lang_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1, -1)
            .reshape(batch_size * num_candidates, lang_tokens.size(1), lang_tokens.size(2))
        )
        lang_mask_repeated = None
        if lang_mask is not None:
            lang_mask_repeated = (
                lang_mask.unsqueeze(1)
                .expand(-1, num_candidates, -1)
                .reshape(batch_size * num_candidates, lang_mask.size(1))
            )

        h_next, _, _ = transition_model(
            prev_h=h_repeated,
            prev_z_post=z_repeated,
            prev_action_emb=action_repeated,
            lang_tokens=lang_tokens_repeated,
            lang_mask=lang_mask_repeated,
        )
        _, z_next = self._prior_sample_flat(latent_model, h_next, temp=temp, hard=hard)

        x_tokens_repeated = (
            x_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1, -1)
            .reshape(batch_size * num_candidates, x_tokens.size(1), x_tokens.size(2))
        )
        g_tokens_repeated = (
            g_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1, -1)
            .reshape(batch_size * num_candidates, g_tokens.size(1), g_tokens.size(2))
        )
        cand_tokens = cand_feats.reshape(batch_size * num_candidates, 1, cand_dim)

        query = self.query_proj(torch.cat([h_next, z_next], dim=-1)).unsqueeze(1)
        if relation_state is not None:
            relation_repeated = (
                relation_state.unsqueeze(1)
                .expand(-1, num_candidates, -1)
                .reshape(batch_size * num_candidates, 1, relation_state.size(-1))
            )
            query = query + relation_repeated
        visual_tokens = self._project_visual_tokens(x_tokens_repeated)
        graph_tokens = self._project_graph_tokens(g_tokens_repeated)
        candidate_tokens = self.cand_proj(cand_tokens) + self.cand_type

        memory = torch.cat([visual_tokens, graph_tokens, candidate_tokens], dim=1)
        valid_masks = []
        if x_mask is None:
            valid_masks.append(torch.ones(x_tokens_repeated.shape[:2], device=x_tokens.device, dtype=torch.bool))
        else:
            valid_masks.append(
                x_mask.unsqueeze(1).expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, x_mask.size(1))
            )
        if g_mask is None:
            valid_masks.append(torch.ones(g_tokens_repeated.shape[:2], device=g_tokens.device, dtype=torch.bool))
        else:
            valid_masks.append(
                g_mask.unsqueeze(1).expand(-1, num_candidates, -1).reshape(batch_size * num_candidates, g_mask.size(1))
            )
        if candidate_mask is None:
            valid_masks.append(torch.ones(batch_size * num_candidates, 1, device=cand_feats.device, dtype=torch.bool))
        else:
            valid_masks.append(candidate_mask.logical_not().reshape(batch_size * num_candidates, 1))
        valid_mask = torch.cat(valid_masks, dim=1)

        scores = torch.einsum("bid,bjd->bij", query, memory).squeeze(1)
        scores = scores / max(float(self.relation_dim) ** 0.5, 1.0)
        attn = self._masked_softmax(scores, valid_mask)
        context = torch.einsum("bn,bnd->bd", attn, memory)

        r_next = self.norm1(query.squeeze(1) + context)
        u_next = self.norm2(r_next + self.ffn(r_next))

        unified_state = u_next.view(batch_size, num_candidates, -1)
        unified_attn = attn.view(batch_size, num_candidates, -1)
        h_next = h_next.view(batch_size, num_candidates, -1)
        z_next = z_next.view(batch_size, num_candidates, -1)

        if candidate_mask is not None:
            valid = candidate_mask.logical_not().unsqueeze(-1)
            unified_state = torch.where(valid, unified_state, torch.zeros_like(unified_state))
            h_next = torch.where(valid, h_next, torch.zeros_like(h_next))
            z_next = torch.where(valid, z_next, torch.zeros_like(z_next))
            unified_attn = torch.where(
                valid.expand(-1, -1, unified_attn.size(-1)),
                unified_attn,
                torch.zeros_like(unified_attn),
            )

        return {
            "unified_state": unified_state,
            "unified_attn": unified_attn,
            "candidate_h_next": h_next,
            "candidate_z_next": z_next,
        }
