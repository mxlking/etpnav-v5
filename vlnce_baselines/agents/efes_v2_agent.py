from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from vlnce_baselines.models.efes_v2 import (
    EmbodiedSelfStateV2,
    FreeEnergyMonitorV2,
    MacroSurpriseV2,
    MicroRSSMV2,
    ModeActionProjectorV2,
    RecoveryRouterV2,
    SelfConfidenceV2,
)


class EFESV2Agent(nn.Module):
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
        timeout_steps: int = 10,
        sigma_min: float = 0.01,
        diag_sigma_min: float = 0.05,
        diag_sigma_max: float = 5.0,
        router_temperature: float = 1.0,
        tau_low: float = 1.0,
        tau_high: float = 2.5,
        pi_threshold: float = 0.5,
        stop_idx: int = 0,
        belief_relax_penalty: float = 1.0,
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
        self.timeout_steps = int(timeout_steps)

        self.self_state = EmbodiedSelfStateV2(
            d_model=self.d_model,
            d_z=self.d_z,
            d_action=self.d_action,
        )
        self.micro_rssm = MicroRSSMV2(
            d_self=self.d_model,
            d_action=self.d_action,
            d_obs=self.x_dim,
            d_h=self.d_h,
            d_z=self.d_z,
        )
        self.macro_surprise = MacroSurpriseV2(
            feat_dim=self.x_dim,
            sigma_min=float(sigma_min),
            diag_sigma_min=float(diag_sigma_min),
            diag_sigma_max=float(diag_sigma_max),
        )
        self.free_energy = FreeEnergyMonitorV2()
        self.self_confidence = SelfConfidenceV2()
        self.recovery_router = RecoveryRouterV2(
            d_model=self.d_model,
            temperature=float(router_temperature),
            tau_low=float(tau_low),
            tau_high=float(tau_high),
            pi_threshold=float(pi_threshold),
        )
        self.mode_projector = ModeActionProjectorV2(
            stop_idx=int(stop_idx),
            belief_relax_penalty=float(belief_relax_penalty),
        )
        self.action_encoder = nn.Linear(self.cand_dim, self.d_action)

    @classmethod
    def from_config(
        cls,
        config: Any,
        x_dim: int,
        lang_dim: int,
        cand_dim: int,
    ) -> "EFESV2Agent":
        efes_cfg = config.EFES_V2
        return cls(
            x_dim=x_dim,
            lang_dim=lang_dim,
            cand_dim=cand_dim,
            d_model=int(efes_cfg.d_model),
            d_z=int(efes_cfg.d_z),
            d_h=int(efes_cfg.d_h),
            d_action=int(efes_cfg.d_action),
            max_nodes=int(efes_cfg.max_nodes),
            timeout_steps=int(efes_cfg.timeout_steps),
            sigma_min=float(efes_cfg.sigma_min),
            diag_sigma_min=float(efes_cfg.diag_sigma_min),
            diag_sigma_max=float(efes_cfg.diag_sigma_max),
            router_temperature=float(efes_cfg.router_temperature),
            tau_low=float(efes_cfg.tau_low),
            tau_high=float(efes_cfg.tau_high),
            pi_threshold=float(efes_cfg.pi_threshold),
            stop_idx=int(efes_cfg.stop_idx),
            belief_relax_penalty=float(efes_cfg.belief_relax_penalty),
        )

    def init_recurrent_state(self, batch_size: int, device: torch.device | str) -> Dict[str, Tensor]:
        return {
            "prev_self": torch.zeros(batch_size, self.d_model, device=device),
            "prev_rssm_h": torch.zeros(batch_size, self.d_h, device=device),
            "prev_z": torch.zeros(batch_size, self.d_z, device=device),
            "prev_action_emb": torch.zeros(batch_size, self.d_action, device=device),
            "prev_progress": torch.zeros(batch_size, 1, device=device),
            "prev_prior_alpha": torch.ones(batch_size, device=device),
            "prev_topo_novelty": torch.zeros(batch_size, device=device),
            "prev_progress_gap": torch.zeros(batch_size, device=device),
        }

    def project_action_features(self, action_feats: Tensor) -> Tensor:
        return self.action_encoder(action_feats)

    @staticmethod
    def _masked_mean(scores: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        if valid_mask is None:
            valid_mask = torch.isfinite(scores)
        else:
            valid_mask = valid_mask & torch.isfinite(scores)
        count = valid_mask.sum(dim=-1).clamp_min(1)
        safe_scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
        return safe_scores.sum(dim=-1) / count.to(scores.dtype)

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
        prev_prior_alpha: Tensor,
        prev_topo_novelty: Tensor,
        prev_progress_gap: Tensor,
        c_micro_recent_mean: Tensor,
        c_macro_recent_mean: Tensor,
        ground_progress_ref: Tensor,
        ground_is_static: Tensor,
        topo_bank_feat: Tensor,
        topo_bank_mask: Tensor,
        macro_valid_mask: Tensor,
        candidate_mask: Tensor,
        frontier_size: Tensor,
        revisit_count: Tensor,
        loop_evidence: Tensor,
        history_backtrack_values: Optional[Tensor] = None,
        history_valid_mask: Optional[Tensor] = None,
        frontier_mask: Optional[Tensor] = None,
        local_frontier_mask: Optional[Tensor] = None,
        lang_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        s_t, progress_t, attn_entropy = self.self_state(
            prev_self=prev_self,
            instr_tokens=lang_tokens,
            prev_z=prev_z,
            prev_action_emb=prev_action_emb,
            prev_prior_alpha=prev_prior_alpha,
            revisit_count=revisit_count,
            loop_evidence=loop_evidence,
            obs_feat=x_t,
            frontier_size=frontier_size,
            topo_novelty_prev=prev_topo_novelty,
            progress_gap_prev=prev_progress_gap,
            instr_mask=lang_mask,
        )

        micro_out = self.micro_rssm(
            s_t=s_t,
            a_prev_emb=prev_action_emb,
            obs_feat=x_t,
            h_prev=prev_rssm_h,
            prior_alpha=prev_prior_alpha,
        )
        c_micro_diag = micro_out["c_micro_diag"]

        has_bank = topo_bank_mask.any(dim=-1)
        topo_context = s_t
        if topo_bank_feat.size(1) > 0:
            topo_context = self.macro_surprise.retrieve(s_t, topo_bank_feat, topo_bank_mask)
        macro_out = self.macro_surprise(
            node_feat=node_feat,
            topo_context=topo_context,
            bank_feat=topo_bank_feat,
            bank_mask=topo_bank_mask,
            macro_valid_mask=(has_bank & macro_valid_mask),
        )

        progress_gap = (progress_t.squeeze(-1) - ground_progress_ref.squeeze(-1)).abs() * ground_is_static.to(
            progress_t.dtype
        )
        g_t = progress_gap

        free_energy = self.free_energy(
            c_micro_diag=c_micro_diag,
            c_macro_diag=macro_out["c_macro_diag"],
            g_t=g_t,
            macro_valid_mask=(has_bank & macro_valid_mask),
        )
        a_t = free_energy["A_t"]

        prior_var_mean = micro_out["std_pri"].square().mean(dim=-1)
        pi_t = self.self_confidence(
            prior_var_mean=prior_var_mean,
            attn_entropy=attn_entropy,
            c_micro_recent_mean=c_micro_recent_mean,
            c_macro_recent_mean=c_macro_recent_mean,
            u_macro=macro_out["u_macro"],
            progress_gap=progress_gap,
        )

        router_out = self.recovery_router(
            s_t=s_t,
            a_t=a_t,
            pi_t=pi_t,
            c_micro_diag=c_micro_diag,
            c_macro_diag=macro_out["c_macro_diag"],
            u_macro=macro_out["u_macro"],
            progress_gap=progress_gap,
            frontier_size=frontier_size,
            revisit_count=revisit_count,
            loop_evidence=loop_evidence,
        )
        mode_action = self.mode_projector(
            etp_scores=etp_candidate_scores,
            candidate_mask=candidate_mask,
            frontier_mask=frontier_mask,
            local_frontier_mask=local_frontier_mask,
            history_backtrack_values=history_backtrack_values,
            history_valid_mask=history_valid_mask,
        )
        per_mode_logits = mode_action["per_mode_logits"]
        fused_logits = self.mode_projector.fuse_logits(per_mode_logits, router_out["mode_probs"])
        argmax_mode = self.recovery_router.argmax_mode(router_out["mode_logits"])
        hard_logits = self.mode_projector.select_hard_logits(per_mode_logits, argmax_mode)

        alpha_prior = torch.ones_like(a_t)
        alpha_progress = torch.ones_like(a_t)
        mod_mask = argmax_mode.eq(RecoveryRouterV2.BELIEF_UPDATE) | argmax_mode.eq(RecoveryRouterV2.BACKTRACK)
        alpha_prior = torch.where(mod_mask, router_out["alpha_prior"], alpha_prior)
        alpha_progress = torch.where(mod_mask, router_out["alpha_progress"], alpha_progress)
        alpha_progress = torch.where(
            argmax_mode.eq(RecoveryRouterV2.BACKTRACK),
            torch.zeros_like(alpha_progress),
            alpha_progress,
        )
        adjusted_progress = progress_t * alpha_progress.unsqueeze(-1)

        valid_before = ~mode_action["strict_mask"]
        valid_after = torch.isfinite(hard_logits)
        return {
            "s_t": s_t,
            "h_t": s_t,
            "rssm_h_t": micro_out["h_t"],
            "z_post": micro_out["z_post"],
            "z_flat": micro_out["z_post"],
            "progress_t": adjusted_progress,
            "attn_entropy": attn_entropy,
            "prior_mu": micro_out["mu_pri"],
            "prior_std": micro_out["std_pri"],
            "post_mu": micro_out["mu_post"],
            "post_std": micro_out["std_post"],
            "C_micro": c_micro_diag,
            "C_macro_diag": macro_out["c_macro_diag"],
            "U_macro": macro_out["u_macro"],
            "g_t": g_t,
            "A_t": a_t,
            "pi_t": pi_t,
            "z_micro": free_energy["z_micro"],
            "z_macro": free_energy["z_macro"],
            "z_ground": free_energy["z_ground"],
            "mode_logits": router_out["mode_logits"],
            "mode_probs": router_out["mode_probs"],
            "mode_entropy": -(router_out["mode_probs"].clamp_min(1e-8) * router_out["mode_probs"].clamp_min(1e-8).log()).sum(dim=-1),
            "argmax_mode": argmax_mode,
            "alpha_prior": alpha_prior,
            "alpha_progress": alpha_progress,
            "mu_pred": macro_out["mu_pred"],
            "sigma2_pred": macro_out["sigma2"],
            "compressed_node_feat": macro_out["x_comp"],
            "topo_novelty": macro_out["topo_novelty"],
            "macro_valid_mask": has_bank & macro_valid_mask,
            "fused_logits": fused_logits,
            "hard_logits": hard_logits,
            "per_mode_logits": per_mode_logits,
            "strict_candidate_mask": mode_action["strict_mask"],
            "expanded_candidate_mask": mode_action["expanded_mask"],
            "score_before_mean": self._masked_mean(etp_candidate_scores, valid_before),
            "score_after_mean": self._masked_mean(hard_logits, valid_after),
            "D_t": a_t,
            "D_t_fused": a_t,
            "D_t_latent": c_micro_diag,
            "D_t_node": macro_out["c_macro_diag"],
            "node_nll_loss": macro_out["node_nll_loss"],
            "router_boot_targets": self.recovery_router.bootstrap_targets(a_t, pi_t),
        }

    def forward(self, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward_step(**kwargs)
