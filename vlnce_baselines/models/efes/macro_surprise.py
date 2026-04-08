from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MacroSurprise(nn.Module):
    def __init__(
        self,
        feat_dim: int = 768,
        num_heads: int = 8,
        max_nodes: int = 50,
        sigma_min: float = 0.01,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.max_nodes = int(max_nodes)
        self.sigma_min = float(sigma_min)
        self.compressor = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.LayerNorm(self.feat_dim),
            nn.GELU(),
        )
        self.retrieve_attn = nn.MultiheadAttention(
            embed_dim=self.feat_dim,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.retrieve_norm = nn.LayerNorm(self.feat_dim)
        self.predictor = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.GELU(),
            nn.Linear(self.feat_dim, self.feat_dim * 2),
        )

    def compress(self, node_feat: Tensor) -> Tensor:
        return self.compressor(node_feat)

    def retrieve(self, s_t: Tensor, bank_feat: Tensor, bank_mask: Tensor) -> Tensor:
        if bank_feat.size(1) == 0:
            return s_t
        query = s_t.unsqueeze(1)
        key_padding_mask = bank_mask.logical_not()
        attn_out, _ = self.retrieve_attn(
            query=query,
            key=bank_feat,
            value=bank_feat,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.retrieve_norm(s_t + attn_out.squeeze(1))

    def predict(self, topo_context: Tensor) -> tuple[Tensor, Tensor]:
        mu_pred, raw_sigma = self.predictor(topo_context).chunk(2, dim=-1)
        sigma2 = F.softplus(raw_sigma) + self.sigma_min
        return mu_pred, sigma2

    @staticmethod
    def gaussian_nll(
        real_feat: Tensor,
        mu_pred: Tensor,
        sigma2: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        value = (((real_feat - mu_pred).square() / sigma2) + sigma2.log()).sum(dim=-1)
        if valid_mask is None:
            return value
        masked = torch.zeros_like(value)
        masked[valid_mask] = value[valid_mask]
        return masked

    @staticmethod
    def cosine_novelty(current_feat: Tensor, reference_feat: Tensor, valid_mask: Tensor) -> Tensor:
        if reference_feat.size(1) == 0:
            return torch.zeros(current_feat.size(0), device=current_feat.device, dtype=current_feat.dtype)
        current = F.normalize(current_feat, dim=-1, eps=1e-6)
        reference = F.normalize(reference_feat, dim=-1, eps=1e-6)
        sims = torch.einsum("bd,bnd->bn", current, reference)
        sims = sims.masked_fill(valid_mask.logical_not(), -1.0)
        best_sim, _ = sims.max(dim=-1)
        novelty = (1.0 - best_sim).clamp_min(0.0)
        no_ref = valid_mask.sum(dim=-1).eq(0)
        novelty = torch.where(no_ref, torch.zeros_like(novelty), novelty)
        return novelty
