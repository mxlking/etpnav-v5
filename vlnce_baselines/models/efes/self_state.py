from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SelfState(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_z: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.z_proj = nn.Linear(int(d_z), self.d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(float(dropout)),
        )
        self.norm2 = nn.LayerNorm(self.d_model)
        self.progress_head = nn.Linear(self.d_model, 1)

    def forward(
        self,
        h_prev: Tensor,
        instr_tokens: Tensor,
        z_post_prev: Tensor,
        instr_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        post_token = self.z_proj(z_post_prev).unsqueeze(1)
        kv = torch.cat([instr_tokens, post_token], dim=1)
        key_padding_mask = None
        if instr_mask is not None:
            post_mask = torch.ones(
                instr_mask.size(0),
                1,
                device=instr_mask.device,
                dtype=instr_mask.dtype,
            )
            key_padding_mask = torch.cat([instr_mask, post_mask], dim=1).logical_not()

        query = h_prev.unsqueeze(1)
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        s_t = self.norm1(h_prev + attn_out.squeeze(1))
        s_t = self.norm2(s_t + self.ffn(s_t))
        progress = torch.sigmoid(self.progress_head(s_t))

        mean_attn = attn_weights.mean(dim=1).squeeze(1).clamp_min(1e-8)
        attn_entropy = -(mean_attn * mean_attn.log()).sum(dim=-1) / math.log(mean_attn.size(-1) + 1e-8)
        return s_t, progress, attn_entropy
