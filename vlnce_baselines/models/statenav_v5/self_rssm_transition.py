from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class SelfRSSMTransition(nn.Module):
    """RSSM prior transition: predict h_t^prior before seeing current reality."""

    def __init__(
        self,
        hidden_dim: int,
        lang_dim: int,
        action_dim: int,
        z_dim: int,
        mlp_hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lang_dim = lang_dim
        self.action_dim = action_dim
        self.z_dim = z_dim

        self.query_proj = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + z_dim, lang_dim),
            nn.Tanh(),
        )
        self.lang_key_proj = nn.Linear(lang_dim, lang_dim)
        self.lang_value_proj = nn.Linear(lang_dim, lang_dim)

        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)

        self.fuse_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def _safe_language_attention(
        self,
        query_input: Tensor,
        lang_tokens: Tensor,
        lang_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = self.query_proj(query_input)
        lang_keys = self.lang_key_proj(lang_tokens)
        lang_values = self.lang_value_proj(lang_tokens)

        attn_scores = torch.einsum("bd,btd->bt", query, lang_keys)
        attn_scores = attn_scores / max(self.lang_dim ** 0.5, 1.0)

        if lang_mask is None:
            valid_mask = torch.ones_like(attn_scores, dtype=torch.bool)
        else:
            valid_mask = lang_mask.bool()

        has_valid = valid_mask.any(dim=-1, keepdim=True)
        masked_scores = attn_scores.masked_fill(
            ~valid_mask,
            torch.finfo(attn_scores.dtype).min,
        )
        safe_scores = torch.where(
            has_valid,
            masked_scores,
            torch.zeros_like(masked_scores),
        )
        attn_weights = torch.softmax(safe_scores, dim=-1)
        attn_weights = attn_weights * valid_mask.to(attn_weights.dtype)
        denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(attn_weights.dtype).eps
        )
        attn_weights = torch.where(
            has_valid,
            attn_weights / denom,
            torch.zeros_like(attn_weights),
        )
        lang_glimpse = torch.einsum("bt,btd->bd", attn_weights, lang_values)
        lang_glimpse = lang_glimpse * has_valid.to(lang_glimpse.dtype)
        return lang_glimpse, attn_weights

    def forward(
        self,
        prev_h: Tensor,
        prev_z_post: Tensor,
        prev_action_emb: Tensor,
        lang_tokens: Tensor,
        lang_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query_input = torch.cat([prev_h, prev_action_emb, prev_z_post], dim=-1)
        lang_glimpse_ctx, attn_weights = self._safe_language_attention(
            query_input=query_input,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask,
        )
        fused_input = torch.cat(
            [
                self.self_proj(prev_h),
                self.action_proj(prev_action_emb),
                self.z_proj(prev_z_post),
                self.lang_proj(lang_glimpse_ctx),
            ],
            dim=-1,
        )
        gru_input = self.fuse_mlp(fused_input)
        h_prior = self.gru(gru_input, prev_h)
        return h_prior, lang_glimpse_ctx, attn_weights

