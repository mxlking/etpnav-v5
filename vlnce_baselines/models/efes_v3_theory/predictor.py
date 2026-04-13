from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TheoryPredictor(nn.Module):
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
        sigma_max: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.1,
        mi_dim: int = 128,
        dropout: float = 0.1,
        use_sigma_max_clamp: bool = True,
        use_shifted_surprise: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_obs = int(d_obs)
        self.d_h = int(d_h)
        self.d_z = int(d_z)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.use_sigma_max_clamp = bool(use_sigma_max_clamp)
        self.use_shifted_surprise = bool(use_shifted_surprise)

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
        self.null_memory = nn.Parameter(torch.zeros(self.d_model))
        self.mi_z_proj = nn.Linear(self.d_z, int(mi_dim))
        self.mi_x_proj = nn.Linear(self.d_obs, int(mi_dim))

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

    def _macro_sigma(self, raw_sigma: Tensor) -> Tensor:
        sigma = F.softplus(raw_sigma) + 1e-4
        sigma = sigma.clamp_min(self.sigma_min)
        if self.use_sigma_max_clamp:
            sigma = sigma.clamp_max(self.sigma_max)
        return sigma

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
            topo_ctx = self.retrieve_norm(s_t + self.null_memory.unsqueeze(0))

        node_target = self.node_proj(node_feat)
        mu_node, raw_sigma = self.node_predictor(topo_ctx).chunk(2, dim=-1)
        sigma = self._macro_sigma(raw_sigma)
        sigma2 = sigma.square()
        d_eff = int(node_target.size(-1))
        recon_sq_error = (node_target.detach() - mu_node).square().sum(dim=-1)
        log_term = torch.log(2.0 * math.pi * sigma2.clamp_min(1e-8))
        nll = 0.5 * (
            (node_target.detach() - mu_node).square() / sigma2.clamp_min(1e-8) + log_term
        ).sum(dim=-1)

        u_t_raw = self.alpha * kl.detach() + self.beta * nll.detach()
        shift_const = self.beta * (0.5 * float(d_eff)) * math.log(2.0 * math.pi * (self.sigma_max ** 2))
        if self.use_shifted_surprise:
            u_tilde = u_t_raw - shift_const
        else:
            u_tilde = u_t_raw

        sigma_hit_low_rate = sigma.le(self.sigma_min + 1e-6).to(dtype=sigma.dtype).mean(dim=-1)
        sigma_hit_high_rate = sigma.ge(self.sigma_max - 1e-6).to(dtype=sigma.dtype).mean(dim=-1) if self.use_sigma_max_clamp else torch.zeros_like(nll)

        mi_query = F.normalize(self.mi_z_proj(z_post), dim=-1)
        mi_key = F.normalize(self.mi_x_proj(obs_feat), dim=-1)

        return {
            "u_t_raw": u_t_raw,
            "u_tilde": u_tilde,
            "h_t": h_t,
            "z_post": z_post,
            "loss_micro": kl,
            "loss_macro": nll,
            "kl_micro": kl.detach(),
            "nll_macro": nll.detach(),
            "mu_node": mu_node,
            "sigma": sigma,
            "recon_sq_error": recon_sq_error.detach(),
            "sigma_mean": sigma.detach().mean(dim=-1),
            "sigma_hit_low_rate": sigma_hit_low_rate.detach(),
            "sigma_hit_high_rate": sigma_hit_high_rate.detach(),
            "compressed_node_feat": node_target.detach(),
            "mi_query": mi_query,
            "mi_key": mi_key,
        }
