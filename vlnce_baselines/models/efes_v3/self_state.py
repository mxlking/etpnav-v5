from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class SelfState(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_obs: int = 768,
        d_z: int = 256,
        d_action: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_somatic_loop: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_obs = int(d_obs)
        self.use_somatic_loop = bool(use_somatic_loop)

        self.z_proj = nn.Linear(int(d_z), self.d_model)
        self.action_proj = nn.Linear(int(d_action), self.d_model)
        self.obs_proj = nn.Linear(self.d_obs, self.d_model)
        self.body_proj = nn.Linear(2, self.d_model)

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

    @staticmethod
    def _as_column(x: Tensor) -> Tensor:
        return x.unsqueeze(-1) if x.dim() == 1 else x

    def forward(
        self,
        prev_self: Tensor,
        instr_tokens: Tensor,
        prev_z: Tensor,
        prev_action_emb: Tensor,
        prev_surprise: Tensor,
        prev_progress: Tensor,
        obs_feat: Tensor,
        instr_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if not self.use_somatic_loop:
            prev_surprise = torch.zeros_like(prev_surprise)

        body_token = self.body_proj(
            torch.cat(
                [
                    self._as_column(prev_surprise),
                    self._as_column(prev_progress),
                ],
                dim=-1,
            )
        ).unsqueeze(1)
        kv = torch.cat(
            [
                instr_tokens,
                self.z_proj(prev_z).unsqueeze(1),
                self.action_proj(prev_action_emb).unsqueeze(1),
                body_token,
                self.obs_proj(obs_feat).unsqueeze(1),
            ],
            dim=1,
        )

        key_padding_mask = None
        if instr_mask is not None:
            if instr_mask.dtype != torch.bool:
                instr_mask = instr_mask.to(dtype=torch.bool)
            extra_valid = torch.ones(
                instr_mask.size(0),
                4,
                device=instr_mask.device,
                dtype=torch.bool,
            )
            key_padding_mask = torch.cat([instr_mask, extra_valid], dim=1).logical_not()

        attn_out, _ = self.cross_attn(
            query=prev_self.unsqueeze(1),
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        s_t = self.norm1(prev_self + attn_out.squeeze(1))
        s_t = self.norm2(s_t + self.ffn(s_t))
        progress = torch.sigmoid(self.progress_head(s_t))
        return s_t, progress
