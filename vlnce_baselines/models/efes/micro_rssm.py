from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MicroRSSM(nn.Module):
    def __init__(
        self,
        d_self: int = 768,
        d_action: int = 128,
        d_obs: int = 768,
        d_h: int = 512,
        d_z: int = 256,
    ) -> None:
        super().__init__()
        self.d_h = int(d_h)
        self.d_z = int(d_z)
        self.input_proj = nn.Sequential(
            nn.Linear(int(d_self) + int(d_action), self.d_h),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(self.d_h, self.d_h)
        self.prior_net = nn.Sequential(
            nn.Linear(self.d_h, self.d_h),
            nn.ELU(),
            nn.Linear(self.d_h, self.d_z * 2),
        )
        self.post_net = nn.Sequential(
            nn.Linear(self.d_h + int(d_obs), self.d_h),
            nn.ELU(),
            nn.Linear(self.d_h, self.d_z * 2),
        )

    @staticmethod
    def _diag_gaussian_kl(
        mu_post: Tensor,
        std_post: Tensor,
        mu_pri: Tensor,
        std_pri: Tensor,
    ) -> Tensor:
        var_post = std_post.square()
        var_pri = std_pri.square()
        kl = (
            torch.log(std_pri.clamp_min(1e-8))
            - torch.log(std_post.clamp_min(1e-8))
            + (var_post + (mu_post - mu_pri).square()) / (2.0 * var_pri.clamp_min(1e-8))
            - 0.5
        )
        return kl.sum(dim=-1)

    def forward(
        self,
        s_t: Tensor,
        a_prev_emb: Tensor,
        obs_feat: Tensor,
        h_prev: Tensor,
    ) -> dict[str, Tensor]:
        gru_input = self.input_proj(torch.cat([s_t, a_prev_emb], dim=-1))
        h_t = self.gru(gru_input, h_prev)

        mu_pri, raw_pri = self.prior_net(h_t).chunk(2, dim=-1)
        std_pri = F.softplus(raw_pri) + 1e-4

        mu_post, raw_post = self.post_net(torch.cat([h_t, obs_feat], dim=-1)).chunk(2, dim=-1)
        std_post = F.softplus(raw_post) + 1e-4

        eps = torch.randn_like(std_post)
        z_post = mu_post + std_post * eps
        c_micro = self._diag_gaussian_kl(mu_post, std_post, mu_pri, std_pri)
        return {
            "h_t": h_t,
            "mu_pri": mu_pri,
            "std_pri": std_pri,
            "mu_post": mu_post,
            "std_post": std_post,
            "z_post": z_post,
            "c_micro": c_micro,
        }
