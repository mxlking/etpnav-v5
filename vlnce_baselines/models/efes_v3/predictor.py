from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Predictor(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        d_obs: int = 768,
        d_action: int = 128,
        d_h: int = 512,
        d_z: int = 256,
        feat_dim: int = 768,
        num_heads: int = 8,
        sigma_min: float = 0.01,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_obs = int(d_obs)
        self.d_h = int(d_h)
        self.d_z = int(d_z)
        self.sigma_min = float(sigma_min)

        self.micro_input_proj = nn.Linear(self.d_model + int(d_action), self.d_h)
        self.micro_obs_proj = nn.Linear(self.d_obs, self.d_model)
        self.gru = nn.GRUCell(self.d_h, self.d_h)
        self.prior_net = nn.Sequential(
            nn.Linear(self.d_h, self.d_h),
            nn.ELU(),
            nn.Linear(self.d_h, self.d_z * 2),
        )
        self.post_net = nn.Sequential(
            nn.Linear(self.d_h + self.d_model, self.d_h),
            nn.ELU(),
            nn.Linear(self.d_h, self.d_z * 2),
        )

        self.bank_proj = nn.Linear(int(feat_dim), self.d_model)
        self.node_proj = nn.Linear(int(feat_dim), self.d_model)
        self.retrieve_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.retrieve_norm = nn.LayerNorm(self.d_model)
        self.node_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model * 2),
        )

        self.surprise_weight = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))

    @staticmethod
    def _diag_kl(mu_q: Tensor, std_q: Tensor, mu_p: Tensor, std_p: Tensor) -> Tensor:
        var_q = std_q.square()
        var_p = std_p.square().clamp_min(1e-8)
        return (
            std_p.clamp_min(1e-8).log()
            - std_q.clamp_min(1e-8).log()
            + (var_q + (mu_q - mu_p).square()) / (2.0 * var_p)
            - 0.5
        ).sum(dim=-1)

    @staticmethod
    def _gaussian_nll(target: Tensor, mu: Tensor, sigma2: Tensor, valid_mask: Tensor) -> Tensor:
        sq = (target - mu).square()
        nll = 0.5 * (sq / sigma2.clamp_min(1e-8) + sigma2.clamp_min(1e-8).log()).mean(dim=-1)
        return nll * valid_mask.to(dtype=nll.dtype)

    def forward(
        self,
        s_t: Tensor,
        obs_feat: Tensor,
        h_prev: Tensor,
        a_prev_emb: Tensor,
        node_feat: Tensor,
        bank_feat: Tensor,
        bank_mask: Tensor,
    ) -> Dict[str, Tensor]:
        obs_emb = self.micro_obs_proj(obs_feat)
        micro_in = F.elu(self.micro_input_proj(torch.cat([s_t, a_prev_emb], dim=-1)))
        h_t = self.gru(micro_in, h_prev)

        mu_pri, raw_pri = self.prior_net(h_t).chunk(2, dim=-1)
        std_pri = F.softplus(raw_pri) + 1e-4
        mu_post, raw_post = self.post_net(torch.cat([h_t, obs_emb], dim=-1)).chunk(2, dim=-1)
        std_post = F.softplus(raw_post) + 1e-4
        z_post = mu_post + std_post * torch.randn_like(std_post)
        kl = self._diag_kl(mu_post, std_post, mu_pri, std_pri)

        has_bank = bank_mask.any(dim=-1)
        if bank_feat.size(1) > 0 and bool(has_bank.any().item()):
            bank_tokens = self.bank_proj(bank_feat)
            ctx, _ = self.retrieve_attn(
                query=s_t.unsqueeze(1),
                key=bank_tokens,
                value=bank_tokens,
                key_padding_mask=bank_mask.logical_not(),
                need_weights=False,
            )
            topo_ctx = self.retrieve_norm(s_t + ctx.squeeze(1))
        else:
            topo_ctx = s_t

        node_target = self.node_proj(node_feat)
        mu_node, raw_sigma = self.node_predictor(topo_ctx).chunk(2, dim=-1)
        sigma2 = F.softplus(raw_sigma) + self.sigma_min
        nll = self._gaussian_nll(node_target.detach(), mu_node, sigma2, has_bank)

        weight = torch.softmax(self.surprise_weight, dim=0)
        surprise_t = weight[0] * kl.detach() + weight[1] * nll.detach()

        return {
            "surprise_t": surprise_t,
            "h_t": h_t,
            "z_post": z_post,
            "loss_micro": kl,
            "loss_macro": nll,
            "kl": kl.detach(),
            "nll": nll.detach(),
            "mu_node": mu_node,
            "sigma2": sigma2,
            "compressed_node_feat": node_target.detach(),
        }
