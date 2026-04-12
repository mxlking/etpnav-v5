from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ViabilityPredictor(nn.Module):
    def __init__(self, d_model: int = 768, d_action: int = 128, local_dim: int = 768) -> None:
        super().__init__()
        self.local_dim = int(local_dim)
        hidden_dim = 256
        self.backbone = nn.Sequential(
            nn.Linear(int(d_model) + int(d_action), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.local_mu = nn.Linear(hidden_dim, self.local_dim)
        self.local_logvar = nn.Linear(hidden_dim, self.local_dim)
        self.vp_change_head = nn.Linear(hidden_dim, 1)
        self.novelty_head = nn.Linear(hidden_dim, 1)
        self.frontier_delta_head = nn.Linear(hidden_dim, 1)
        self.progress_delta_head = nn.Linear(hidden_dim, 1)
        self.advance_conf_head = nn.Linear(hidden_dim, 1)

    def forward(self, z_self: Tensor, prev_action_emb: Tensor) -> dict[str, Tensor]:
        hidden = self.backbone(torch.cat([z_self, prev_action_emb], dim=-1))
        local_logvar = self.local_logvar(hidden).clamp(min=-4.0, max=4.0)
        return {
            "local_mu": self.local_mu(hidden),
            "local_var": F.softplus(local_logvar) + 1e-4,
            "vp_change_logit": self.vp_change_head(hidden).squeeze(-1),
            "novelty_pred": torch.sigmoid(self.novelty_head(hidden).squeeze(-1)),
            "frontier_delta_pred": torch.tanh(self.frontier_delta_head(hidden).squeeze(-1)),
            "delta_progress_pred": torch.sigmoid(self.progress_delta_head(hidden).squeeze(-1)),
            "advance_confidence_logit": self.advance_conf_head(hidden).squeeze(-1),
        }


class MacroDiagnosisHead(nn.Module):
    def __init__(
        self,
        feat_dim: int = 768,
        num_heads: int = 8,
        sigma_min: float = 0.01,
        diag_sigma_min: float = 0.05,
        diag_sigma_max: float = 5.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.sigma_min = float(sigma_min)
        self.diag_sigma_min = float(diag_sigma_min)
        self.diag_sigma_max = float(diag_sigma_max)
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

    def retrieve(self, z_self: Tensor, bank_feat: Tensor, bank_mask: Tensor) -> Tensor:
        if bank_feat.size(1) == 0:
            return z_self
        query = z_self.unsqueeze(1)
        attn_out, _ = self.retrieve_attn(
            query=query,
            key=bank_feat,
            value=bank_feat,
            key_padding_mask=bank_mask.logical_not(),
            need_weights=False,
        )
        return self.retrieve_norm(z_self + attn_out.squeeze(1))

    def predict(self, topo_context: Tensor) -> tuple[Tensor, Tensor]:
        mu_pred, raw_sigma = self.predictor(topo_context).chunk(2, dim=-1)
        sigma2 = F.softplus(raw_sigma) + self.sigma_min
        return mu_pred, sigma2

    def forward(
        self,
        node_feat: Tensor,
        topo_context: Tensor,
        bank_feat: Tensor,
        bank_mask: Tensor,
        macro_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        x_comp = self.compress(node_feat)
        mu_pred, sigma2 = self.predict(topo_context)
        sq_error = (x_comp.detach() - mu_pred).square()
        node_nll_loss = 0.5 * ((sq_error / sigma2) + sigma2.log().clamp_min(0.0)).mean(dim=-1)
        sigma2_diag = sigma2.detach().clamp(min=self.diag_sigma_min, max=self.diag_sigma_max)
        macro_diag_score = (sq_error / sigma2_diag).mean(dim=-1)
        macro_diag_score = macro_diag_score * macro_valid_mask.to(dtype=macro_diag_score.dtype)
        macro_uncertainty = sigma2.detach().log().mean(dim=-1)
        topo_novelty = self.cosine_novelty(x_comp.detach(), bank_feat, bank_mask)
        return {
            "x_comp": x_comp,
            "mu_pred": mu_pred,
            "sigma2": sigma2,
            "node_nll_loss": node_nll_loss * macro_valid_mask.to(dtype=node_nll_loss.dtype),
            "macro_diag_score": macro_diag_score,
            "macro_uncertainty": macro_uncertainty,
            "topo_novelty": topo_novelty,
            "macro_valid_mask": macro_valid_mask,
        }

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
        return torch.where(no_ref, torch.zeros_like(novelty), novelty)


class RealityEncoder(nn.Module):
    def forward(
        self,
        path_delta: Tensor,
        topo_advanced: Tensor,
        frontier_delta: Tensor,
        eps_path: float,
    ) -> dict[str, Tensor]:
        eps = torch.as_tensor(float(eps_path), device=path_delta.device, dtype=path_delta.dtype).clamp_min(1e-6)
        motion_stall = (1.0 - (path_delta / eps).clamp(max=1.0)).clamp_min(0.0)
        topo_stall = 1.0 - topo_advanced.to(dtype=path_delta.dtype).clamp(0.0, 1.0)
        frontier_stall = torch.sigmoid(frontier_delta.neg())
        ground_diag = (motion_stall + topo_stall + frontier_stall) / 3.0
        return {
            "ground_diag": ground_diag,
            "path_delta": path_delta,
            "topo_advanced": topo_advanced.to(dtype=path_delta.dtype),
            "frontier_delta": frontier_delta,
        }


class ActionConditioner(nn.Module):
    """Lightweight EFES residual adapter over ETPNav candidate logits.

    Shapes:
      nav_logits: [B, N]
      candidate_embeds: [B, N, D_cand]
      self_post_z: [B, D_model]
      scalar rupture/self inputs: [B] or [B, 1]
    """

    def __init__(
        self,
        d_model: int = 768,
        cand_dim: int = 768,
        hidden_dim: int = 256,
        max_delta: float = 1.0,
        gate_bias: float = -2.0,
        mode: str = "prob_mixture",
        beta_max: float = 0.1,
        prob_eps: float = 1e-8,
        rupture_threshold: float = 0.75,
        eligibility_temperature: float = 0.1,
        stop_delta_scale: float = 1.0,
        stop_idx: int = 0,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.mode = str(mode).strip().lower()
        if self.mode not in {"prob_mixture", "logit_residual"}:
            raise ValueError("Unknown ActionConditioner mode: {}".format(self.mode))
        self.beta_max = max(0.0, min(float(beta_max), 1.0))
        self.prob_eps = max(float(prob_eps), 1e-12)
        self.rupture_threshold = float(rupture_threshold)
        self.eligibility_temperature = max(float(eligibility_temperature), 1e-4)
        self.stop_delta_scale = float(stop_delta_scale)
        self.stop_idx = int(stop_idx)
        ctx_dim = int(d_model) + 8
        self.context_proj = nn.Sequential(
            nn.Linear(ctx_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
        )
        self.candidate_proj = nn.Linear(int(cand_dim), int(hidden_dim))
        self.residual_head = nn.Linear(int(hidden_dim), 1)
        self.gate_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, float(gate_bias))

    @staticmethod
    def _as_column(x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x.unsqueeze(-1)
        return x

    def forward(
        self,
        nav_logits: Tensor,
        candidate_embeds: Tensor,
        self_post_z: Tensor,
        self_continuity: Tensor,
        r_local: Tensor,
        r_topo: Tensor,
        r_ground: Tensor,
        rupture_confidence: Tensor,
        type_probs: Tensor,
        invalid_candidate_mask: Tensor,
    ) -> dict[str, Tensor]:
        self_continuity = self._as_column(self_continuity)
        r_local = self._as_column(r_local)
        r_topo = self._as_column(r_topo)
        r_ground = self._as_column(r_ground)
        rupture_confidence = self._as_column(rupture_confidence)
        context = torch.cat(
            [
                self_post_z,
                self_continuity,
                r_local,
                r_topo,
                r_ground,
                rupture_confidence,
                type_probs,
            ],
            dim=-1,
        )
        ctx = self.context_proj(context)
        cand = self.candidate_proj(candidate_embeds)
        hidden = torch.tanh(cand + ctx.unsqueeze(1))
        residual = torch.tanh(self.residual_head(hidden).squeeze(-1)) * self.max_delta
        raw_gate = torch.sigmoid(self.gate_head(ctx).squeeze(-1))
        eligibility = torch.sigmoid(
            (rupture_confidence.squeeze(-1) - self.rupture_threshold) / self.eligibility_temperature
        )
        gate = raw_gate * eligibility
        if 0 <= self.stop_idx < residual.size(-1):
            stop_scale = residual.new_ones(residual.shape)
            stop_scale[:, self.stop_idx] = self.stop_delta_scale
            residual = residual * stop_scale
        valid = invalid_candidate_mask.logical_not()
        masked_nav_logits = nav_logits.masked_fill(invalid_candidate_mask, -1.0e4)
        residual_logits = (nav_logits + residual).masked_fill(invalid_candidate_mask, -1.0e4)
        if self.mode == "prob_mixture":
            etp_log_probs = F.log_softmax(masked_nav_logits, dim=-1)
            efes_log_probs = F.log_softmax(residual_logits, dim=-1)
            pi_etp = etp_log_probs.exp()
            pi_efes = efes_log_probs.exp()
            beta = (gate * self.beta_max).clamp(0.0, 1.0)
            pi_final = (1.0 - beta.unsqueeze(-1)) * pi_etp + beta.unsqueeze(-1) * pi_efes
            pi_final = pi_final.masked_fill(invalid_candidate_mask, 0.0)
            pi_final = pi_final / pi_final.sum(dim=-1, keepdim=True).clamp_min(self.prob_eps)
            safe_logits = pi_final.clamp_min(self.prob_eps).log().masked_fill(invalid_candidate_mask, -1.0e4)
            condition_delta_tensor = (pi_final - pi_etp).abs()
            kl_terms = pi_final * (
                pi_final.clamp_min(self.prob_eps).log()
                - pi_etp.clamp_min(self.prob_eps).log()
            )
            kl_terms = kl_terms.masked_fill(invalid_candidate_mask, 0.0)
            condition_policy_kl = torch.nan_to_num(
                kl_terms.sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(0.0)
        else:
            beta = gate
            delta = gate.unsqueeze(-1) * residual
            safe_logits = (nav_logits + delta).masked_fill(invalid_candidate_mask, -1.0e4)
            pi_final = F.softmax(safe_logits, dim=-1)
            pi_etp = F.softmax(masked_nav_logits, dim=-1)
            condition_delta_tensor = delta.abs()
            kl_terms = pi_final * (
                pi_final.clamp_min(self.prob_eps).log()
                - pi_etp.clamp_min(self.prob_eps).log()
            )
            kl_terms = kl_terms.masked_fill(invalid_candidate_mask, 0.0)
            condition_policy_kl = torch.nan_to_num(
                kl_terms.sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
            ).clamp_min(0.0)
        valid_count = valid.sum(dim=-1).clamp_min(1).to(safe_logits.dtype)
        condition_delta = torch.where(
            valid,
            condition_delta_tensor,
            torch.zeros_like(condition_delta_tensor),
        ).sum(dim=-1) / valid_count
        return {
            "safe_logits": safe_logits,
            "condition_gate": beta,
            "condition_beta": beta,
            "condition_raw_gate": raw_gate,
            "condition_eligibility": eligibility,
            "condition_delta": condition_delta,
            "condition_residual": residual,
            "condition_policy_kl": condition_policy_kl,
        }
