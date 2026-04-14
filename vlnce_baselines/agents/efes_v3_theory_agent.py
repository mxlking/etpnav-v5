from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vlnce_baselines.models.efes_v3_theory import (
    TheoryCorrector,
    TheoryPredictor,
    TheorySelfState,
)


class EFESV3TheoryAgent(nn.Module):
    def __init__(
        self,
        x_dim: int,
        lang_dim: int,
        cand_dim: int,
        d_model: int = 768,
        d_z: int = 256,
        d_h: int = 512,
        d_action: int = 128,
        max_nodes: int = 50,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.1,
        gate_bias: float = -2.0,
        delta_bound: float = 1.0,
        lambda_scale: float = 1.0,
        gate_logit_init_std: float = 0.02,
        use_body_loop: bool = True,
        use_predictor: bool = True,
        use_corrector: bool = True,
        use_shifted_surprise: bool = True,
        use_sigma_max_clamp: bool = True,
        use_dual_gate: bool = True,
    ) -> None:
        super().__init__()
        self.x_dim = int(x_dim)
        self.lang_dim = int(lang_dim)
        self.cand_dim = int(cand_dim)
        self.d_model = int(d_model)
        self.d_z = int(d_z)
        self.d_h = int(d_h)
        self.d_action = int(d_action)
        self.max_nodes = int(max_nodes)
        self.use_predictor = bool(use_predictor)
        self.use_corrector = bool(use_corrector)

        self.self_state = TheorySelfState(
            d_model=self.d_model,
            d_obs=self.x_dim,
            d_instr=self.lang_dim,
            d_z=self.d_z,
            d_action=self.d_action,
            use_body_loop=bool(use_body_loop),
        )
        self.predictor = TheoryPredictor(
            d_model=self.d_model,
            d_obs=self.x_dim,
            d_action=self.d_action,
            d_h=self.d_h,
            d_z=self.d_z,
            feat_dim=self.x_dim,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            alpha=float(alpha),
            beta=float(beta),
            use_sigma_max_clamp=bool(use_sigma_max_clamp),
            use_shifted_surprise=bool(use_shifted_surprise),
        )
        self.corrector = TheoryCorrector(
            d_model=self.d_model,
            cand_dim=self.cand_dim,
            delta_bound=float(delta_bound),
            gate_bias=float(gate_bias),
            gate_logit_init_std=float(gate_logit_init_std),
            lambda_scale=float(lambda_scale),
            use_dual_gate=bool(use_dual_gate),
            use_shifted_surprise=bool(use_shifted_surprise),
        )
        self.action_encoder = nn.Linear(self.cand_dim, self.d_action)

    @classmethod
    def from_config(
        cls,
        config: Any,
        x_dim: int,
        lang_dim: int,
        cand_dim: int,
    ) -> "EFESV3TheoryAgent":
        efes_cfg = config.EFES_V3_THEORY
        return cls(
            x_dim=x_dim,
            lang_dim=lang_dim,
            cand_dim=cand_dim,
            d_model=int(efes_cfg.d_model),
            d_z=int(efes_cfg.d_z),
            d_h=int(efes_cfg.d_h),
            d_action=int(efes_cfg.d_action),
            max_nodes=int(efes_cfg.max_nodes),
            sigma_min=float(efes_cfg.sigma_min),
            sigma_max=float(efes_cfg.sigma_max),
            alpha=float(efes_cfg.alpha),
            beta=float(efes_cfg.beta),
            gate_bias=float(efes_cfg.gate_bias),
            delta_bound=float(efes_cfg.delta_bound),
            lambda_scale=float(efes_cfg.lambda_scale),
            gate_logit_init_std=float(getattr(efes_cfg, "gate_logit_init_std", 0.02)),
            use_body_loop=bool(getattr(efes_cfg, "use_body_loop", True)),
            use_predictor=bool(getattr(efes_cfg, "use_predictor", True)),
            use_corrector=bool(getattr(efes_cfg, "use_corrector", True)),
            use_shifted_surprise=bool(getattr(efes_cfg, "use_shifted_surprise", True)),
            use_sigma_max_clamp=bool(getattr(efes_cfg, "use_sigma_max_clamp", True)),
            use_dual_gate=bool(getattr(efes_cfg, "use_dual_gate", True)),
        )

    def init_recurrent_state(self, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        return {
            "prev_self": torch.zeros(batch_size, self.d_model, device=device),
            "prev_rssm_h": torch.zeros(batch_size, self.d_h, device=device),
            "prev_z": torch.zeros(batch_size, self.d_z, device=device),
            "prev_action_emb": torch.zeros(batch_size, self.d_action, device=device),
            "prev_u_tilde": torch.zeros(batch_size, device=device),
            "prev_progress": torch.zeros(batch_size, 1, device=device),
        }

    def project_action_features(self, action_feats: Tensor) -> Tensor:
        return self.action_encoder(action_feats)

    def apply_static_freeze(self) -> None:
        return None

    @staticmethod
    def _masked_mean(scores: Tensor, valid_mask: Tensor) -> Tensor:
        finite_valid = valid_mask & torch.isfinite(scores)
        count = finite_valid.sum(dim=-1).clamp_min(1)
        safe_scores = torch.where(finite_valid, scores, torch.zeros_like(scores))
        return safe_scores.sum(dim=-1) / count.to(scores.dtype)

    @staticmethod
    def _masked_policy_kl(corrected_logits: Tensor, base_logits: Tensor, valid_mask: Tensor) -> Tensor:
        masked_corrected = corrected_logits.masked_fill(valid_mask.logical_not(), -1.0e4)
        masked_base = base_logits.masked_fill(valid_mask.logical_not(), -1.0e4)
        log_p = F.log_softmax(masked_corrected, dim=-1)
        log_q = F.log_softmax(masked_base, dim=-1)
        p = log_p.exp()
        pointwise = p * (log_p - log_q)
        pointwise = torch.where(valid_mask, pointwise, torch.zeros_like(pointwise))
        pointwise = torch.nan_to_num(pointwise, nan=0.0, posinf=0.0, neginf=0.0)
        return pointwise.sum(dim=-1)

    def forward_step(
        self,
        x_t: Tensor,
        node_feat: Tensor,
        lang_tokens: Tensor,
        cand_feats: Tensor,
        etp_candidate_scores: Tensor,
        prev_action_emb: Tensor,
        prev_z: Tensor,
        prev_self: Tensor,
        prev_rssm_h: Tensor,
        prev_progress: Tensor,
        prev_u_tilde: Tensor,
        topo_bank_feat: Tensor,
        topo_bank_mask: Tensor,
        candidate_mask: Tensor,
        lang_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size = x_t.size(0)
        if cand_feats.size(0) != batch_size or etp_candidate_scores.size(0) != batch_size:
            raise ValueError("EFESV3TheoryAgent batch mismatch between observation and candidates.")
        if candidate_mask.shape != etp_candidate_scores.shape:
            raise ValueError("candidate_mask shape {} does not match logits shape {}.".format(candidate_mask.shape, etp_candidate_scores.shape))

        self_t, progress_t = self.self_state(
            prev_self=prev_self,
            instr_tokens=lang_tokens,
            prev_z=prev_z,
            prev_action_emb=prev_action_emb,
            prev_u_tilde=prev_u_tilde,
            prev_progress=prev_progress,
            obs_feat=x_t,
            instr_mask=lang_mask,
        )

        if self.use_predictor:
            pred = self.predictor(
                s_t=self_t,
                obs_feat=x_t,
                h_prev=prev_rssm_h,
                a_prev_emb=prev_action_emb,
                node_feat=node_feat,
                bank_feat=topo_bank_feat,
                bank_mask=topo_bank_mask,
            )
            u_t_raw = pred["u_t_raw"]
            u_tilde = pred["u_tilde"]
            h_t = pred["h_t"]
            z_post = pred["z_post"]
            loss_micro = pred["loss_micro"]
            loss_macro = pred["loss_macro"]
            compressed_node_feat = pred["compressed_node_feat"]
        else:
            zero = torch.zeros(self_t.size(0), device=self_t.device, dtype=self_t.dtype)
            u_t_raw = zero
            u_tilde = zero
            h_t = prev_rssm_h
            z_post = prev_z
            loss_micro = zero
            loss_macro = zero
            compressed_node_feat = node_feat.detach()
            pred = {
                "kl_micro": zero,
                "nll_macro": zero,
                "recon_sq_error": zero,
                "sigma_mean": zero,
                "sigma_hit_low_rate": zero,
                "sigma_hit_high_rate": zero,
                "mi_query": F.normalize(torch.zeros(batch_size, 128, device=self_t.device), dim=-1),
                "mi_key": F.normalize(torch.zeros(batch_size, 128, device=self_t.device), dim=-1),
            }

        u_signal = u_tilde
        if self.use_corrector:
            corr = self.corrector(
                s_t=self_t,
                u_signal=u_signal,
                etp_logits=etp_candidate_scores,
                cand_embeds=cand_feats,
                cand_mask=candidate_mask,
            )
            corrected_logits = corr["corrected_logits"]
            gate = corr["gate"]
            gate_raw = corr["gate_raw"]
            alarm_prob = corr["alarm_prob"]
            delta = corr["delta"]
            lambda_hat = corr["lambda_hat"]
            delta_abs_max = corr["delta_abs_max"]
        else:
            corrected_logits = etp_candidate_scores.masked_fill(candidate_mask, -1.0e4)
            gate = torch.zeros(batch_size, device=self_t.device, dtype=self_t.dtype)
            gate_raw = torch.zeros(batch_size, device=self_t.device, dtype=self_t.dtype)
            alarm_prob = torch.zeros(batch_size, device=self_t.device, dtype=self_t.dtype)
            delta = torch.zeros_like(etp_candidate_scores)
            lambda_hat = torch.zeros(batch_size, device=self_t.device, dtype=self_t.dtype)
            delta_abs_max = torch.zeros(batch_size, device=self_t.device, dtype=self_t.dtype)

        valid_mask = candidate_mask.logical_not()
        masked_base_logits = etp_candidate_scores.masked_fill(candidate_mask, -1.0e4)
        kl_corrected_vs_base = self._masked_policy_kl(corrected_logits, masked_base_logits, valid_mask)

        return {
            "corrected_logits": corrected_logits,
            "gate": gate,
            "gate_raw": gate_raw,
            "alarm_prob": alarm_prob,
            "lambda_hat": lambda_hat,
            "delta": delta,
            "delta_abs_max": delta_abs_max,
            "u_t_raw": u_t_raw,
            "u_tilde": u_tilde,
            "progress_t": progress_t.squeeze(-1),
            "loss_micro": loss_micro,
            "loss_macro": loss_macro,
            "kl_micro": pred["kl_micro"],
            "nll_macro": pred["nll_macro"],
            "self_t": self_t,
            "rssm_h_t": h_t,
            "z_flat": z_post,
            "compressed_node_feat": compressed_node_feat,
            "recon_sq_error": pred["recon_sq_error"],
            "sigma_mean": pred["sigma_mean"],
            "sigma_hit_low_rate": pred["sigma_hit_low_rate"],
            "sigma_hit_high_rate": pred["sigma_hit_high_rate"],
            "mi_query": pred["mi_query"],
            "mi_key": pred["mi_key"],
            "score_before_mean": self._masked_mean(masked_base_logits, valid_mask),
            "score_after_mean": self._masked_mean(corrected_logits, valid_mask),
            "kl_corrected_vs_base": kl_corrected_vs_base,
        }

    def forward(self, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward_step(**kwargs)
