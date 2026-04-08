from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from vlnce_baselines.models.efes import (
    DualRecovery,
    FreeEnergyMonitor,
    MacroSurprise,
    MicroRSSM,
    SelfConfidence,
    SelfState,
)


class EFESAgent(nn.Module):
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
        decision_candidate_min: int = 2,
        timeout_steps: int = 10,
        eps_path: float = 0.3,
        tau_low: float = 1.0,
        tau_high: float = 2.5,
        pi_threshold: float = 0.5,
        sigma_min: float = 0.01,
        stop_idx: int = 0,
        use_macro: bool = True,
        use_grounding: bool = True,
        use_confidence: bool = True,
        use_recovery: bool = True,
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
        self.decision_candidate_min = int(decision_candidate_min)
        self.timeout_steps = int(timeout_steps)
        self.eps_path = float(eps_path)
        self.stop_idx = int(stop_idx)
        self.use_macro = bool(use_macro)
        self.use_grounding = bool(use_grounding)
        self.use_confidence = bool(use_confidence)
        self.use_recovery = bool(use_recovery)

        self.self_state = SelfState(d_model=self.d_model, d_z=self.d_z)
        self.micro_rssm = MicroRSSM(
            d_self=self.d_model,
            d_action=self.d_action,
            d_obs=self.x_dim,
            d_h=self.d_h,
            d_z=self.d_z,
        )
        self.macro_surprise = MacroSurprise(
            feat_dim=self.x_dim,
            max_nodes=self.max_nodes,
            sigma_min=float(sigma_min),
        )
        self.free_energy = FreeEnergyMonitor()
        self.self_confidence = SelfConfidence()
        self.dual_recovery = DualRecovery(
            d_model=self.d_model,
            tau_low=float(tau_low),
            tau_high=float(tau_high),
            pi_threshold=float(pi_threshold),
        )
        self.action_encoder = nn.Linear(self.cand_dim, self.d_action)

    @classmethod
    def from_config(
        cls,
        config: Any,
        x_dim: int,
        lang_dim: int,
        cand_dim: int,
    ) -> "EFESAgent":
        efes_cfg = config.EFES
        return cls(
            x_dim=x_dim,
            lang_dim=lang_dim,
            cand_dim=cand_dim,
            d_model=int(efes_cfg.d_model),
            d_z=int(efes_cfg.d_z),
            d_h=int(efes_cfg.d_h),
            d_action=int(efes_cfg.d_action),
            max_nodes=int(efes_cfg.max_nodes),
            decision_candidate_min=int(efes_cfg.node_decision_candidate_min),
            timeout_steps=int(efes_cfg.timeout_steps),
            eps_path=float(efes_cfg.eps_path),
            tau_low=float(efes_cfg.tau_low),
            tau_high=float(efes_cfg.tau_high),
            pi_threshold=float(efes_cfg.pi_threshold),
            sigma_min=float(efes_cfg.sigma_min),
            stop_idx=int(efes_cfg.stop_idx),
            use_macro=bool(efes_cfg.use_macro),
            use_grounding=bool(efes_cfg.use_grounding),
            use_confidence=bool(efes_cfg.use_confidence),
            use_recovery=bool(efes_cfg.use_recovery),
        )

    def init_recurrent_state(self, batch_size: int, device: torch.device | str) -> Dict[str, Tensor]:
        return {
            "prev_self": torch.zeros(batch_size, self.d_model, device=device),
            "prev_rssm_h": torch.zeros(batch_size, self.d_h, device=device),
            "prev_z": torch.zeros(batch_size, self.d_z, device=device),
            "prev_action_emb": torch.zeros(batch_size, self.d_action, device=device),
            "prev_progress": torch.zeros(batch_size, 1, device=device),
            "prev_pi": torch.full((batch_size,), 0.5, device=device),
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

    def _build_candidate_masks(
        self,
        candidate_mask: Tensor,
        frontier_mask: Optional[Tensor],
        local_frontier_mask: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        strict_mask = candidate_mask.clone()
        expanded_mask = candidate_mask.clone()
        if local_frontier_mask is not None:
            local_keep = local_frontier_mask.clone()
            if 0 <= self.stop_idx < local_keep.size(1):
                local_keep[:, self.stop_idx] = True
            strict_mask = candidate_mask | (~local_keep)
        if frontier_mask is not None:
            frontier_keep = frontier_mask.clone()
            if 0 <= self.stop_idx < frontier_keep.size(1):
                frontier_keep[:, self.stop_idx] = True
            expanded_mask = candidate_mask | (~frontier_keep)
        return strict_mask, expanded_mask

    def _apply_recovery(
        self,
        etp_scores: Tensor,
        candidate_mask: Tensor,
        frontier_mask: Optional[Tensor],
        local_frontier_mask: Optional[Tensor],
        history_backtrack_values: Optional[Tensor],
        history_valid_mask: Optional[Tensor],
        recovery_mode: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_candidates = etp_scores.shape
        strict_mask, expanded_mask = self._build_candidate_masks(
            candidate_mask=candidate_mask,
            frontier_mask=frontier_mask,
            local_frontier_mask=local_frontier_mask,
        )
        rescored = etp_scores.clone()
        level3_selected_idx = torch.full((batch_size,), -1, dtype=torch.long, device=etp_scores.device)
        level3_decision_code = torch.zeros_like(level3_selected_idx)

        for batch_idx in range(batch_size):
            mode = int(recovery_mode[batch_idx].item())
            if mode == DualRecovery.PROCEED or mode == DualRecovery.BELIEF_UPDATE:
                rescored[batch_idx] = rescored[batch_idx].masked_fill(strict_mask[batch_idx], -float("inf"))
                continue
            if mode == DualRecovery.EXPLORE:
                rescored[batch_idx] = rescored[batch_idx].masked_fill(expanded_mask[batch_idx], -float("inf"))
                continue

            backtrack_scores = None
            if history_backtrack_values is not None and history_valid_mask is not None:
                backtrack_scores = history_backtrack_values[batch_idx].masked_fill(
                    ~history_valid_mask[batch_idx], -float("inf")
                )
            chosen_idx = self.stop_idx if 0 <= self.stop_idx < num_candidates else 0
            decision_code = 2
            if backtrack_scores is not None and torch.isfinite(backtrack_scores).any():
                chosen_idx = int(backtrack_scores.argmax().item())
                decision_code = 1
            override = torch.full_like(rescored[batch_idx], -float("inf"))
            override[chosen_idx] = 10.0
            rescored[batch_idx] = override
            level3_selected_idx[batch_idx] = chosen_idx
            level3_decision_code[batch_idx] = decision_code

        valid_before = (~strict_mask) & torch.isfinite(etp_scores)
        valid_after = torch.isfinite(rescored)
        return rescored, strict_mask, expanded_mask, level3_selected_idx, level3_decision_code

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
        c_micro_recent_mean: Tensor,
        ground_progress_ref: Tensor,
        ground_is_static: Tensor,
        topo_bank_feat: Tensor,
        topo_bank_mask: Tensor,
        topo_update_mask: Tensor,
        candidate_mask: Tensor,
        history_backtrack_values: Optional[Tensor] = None,
        history_valid_mask: Optional[Tensor] = None,
        frontier_mask: Optional[Tensor] = None,
        local_frontier_mask: Optional[Tensor] = None,
        lang_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        s_t, progress_t, attn_entropy = self.self_state(
            h_prev=prev_self,
            instr_tokens=lang_tokens,
            z_post_prev=prev_z,
            instr_mask=lang_mask,
        )
        micro_out = self.micro_rssm(
            s_t=s_t,
            a_prev_emb=prev_action_emb,
            obs_feat=x_t,
            h_prev=prev_rssm_h,
        )
        c_micro = micro_out["c_micro"]

        compressed_node_feat = self.macro_surprise.compress(node_feat)
        topo_context = s_t
        mu_pred = torch.zeros_like(node_feat)
        sigma2_pred = torch.ones_like(node_feat)
        has_bank = topo_bank_mask.any(dim=-1)
        if self.use_macro and topo_bank_feat.size(1) > 0:
            topo_context = self.macro_surprise.retrieve(s_t, topo_bank_feat, topo_bank_mask)
            mu_pred, sigma2_pred = self.macro_surprise.predict(topo_context)
            c_macro = self.macro_surprise.gaussian_nll(
                real_feat=node_feat,
                mu_pred=mu_pred,
                sigma2=sigma2_pred,
                valid_mask=has_bank & topo_update_mask,
            )
            topo_novelty = self.macro_surprise.cosine_novelty(node_feat, topo_bank_feat, topo_bank_mask)
        else:
            c_macro = torch.zeros_like(c_micro)
            topo_novelty = torch.zeros_like(c_micro)

        if self.use_grounding:
            g_t = (progress_t.squeeze(-1) - ground_progress_ref.squeeze(-1)).abs() * ground_is_static.to(
                progress_t.dtype
            )
        else:
            g_t = torch.zeros_like(c_micro)

        free_energy = self.free_energy(
            c_micro=c_micro,
            c_macro=c_macro,
            g_t=g_t,
            macro_valid_mask=(has_bank & topo_update_mask),
        )
        a_t = free_energy["A_t"]

        prior_var_mean = micro_out["std_pri"].square().mean(dim=-1)
        if self.use_confidence:
            pi_t = self.self_confidence(
                prior_var_mean=prior_var_mean,
                attn_entropy=attn_entropy,
                c_micro_recent_mean=c_micro_recent_mean,
            )
        else:
            pi_t = torch.full_like(c_micro, 0.5)

        recovery = self.dual_recovery(s_t=s_t, a_t=a_t, pi_t=pi_t) if self.use_recovery else {
            "recovery_mode": torch.zeros_like(a_t, dtype=torch.long),
            "alpha_prior": torch.ones_like(a_t),
            "alpha_progress": torch.ones_like(a_t),
        }

        adjusted_progress = progress_t * recovery["alpha_progress"].unsqueeze(-1)
        adjusted_self = s_t * recovery["alpha_progress"].unsqueeze(-1)
        adjusted_z = micro_out["z_post"] * recovery["alpha_prior"].unsqueeze(-1)

        rescored_scores, strict_mask, expanded_mask, level3_selected_idx, level3_decision_code = self._apply_recovery(
            etp_scores=etp_candidate_scores,
            candidate_mask=candidate_mask,
            frontier_mask=frontier_mask,
            local_frontier_mask=local_frontier_mask,
            history_backtrack_values=history_backtrack_values,
            history_valid_mask=history_valid_mask,
            recovery_mode=recovery["recovery_mode"],
        )
        before_valid = ~strict_mask
        after_valid = torch.isfinite(rescored_scores)

        return {
            "s_t": s_t,
            "h_t": adjusted_self,
            "rssm_h_t": micro_out["h_t"],
            "z_post": micro_out["z_post"],
            "z_flat": adjusted_z,
            "progress_t": adjusted_progress,
            "attn_entropy": attn_entropy,
            "prior_mu": micro_out["mu_pri"],
            "prior_std": micro_out["std_pri"],
            "post_mu": micro_out["mu_post"],
            "post_std": micro_out["std_post"],
            "C_micro": c_micro,
            "C_macro": c_macro,
            "g_t": g_t,
            "A_t": a_t,
            "pi_t": pi_t,
            "z_micro": free_energy["z_micro"],
            "z_macro": free_energy["z_macro"],
            "z_ground": free_energy["z_ground"],
            "recovery_mode": recovery["recovery_mode"],
            "alpha_prior": recovery["alpha_prior"],
            "alpha_progress": recovery["alpha_progress"],
            "mu_pred": mu_pred,
            "sigma2_pred": sigma2_pred,
            "compressed_node_feat": compressed_node_feat,
            "topo_novelty": topo_novelty,
            "macro_valid_mask": has_bank & topo_update_mask,
            "rescored_candidate_scores": rescored_scores,
            "strict_candidate_mask": strict_mask,
            "expanded_candidate_mask": expanded_mask,
            "frontier_candidate_mask": frontier_mask if frontier_mask is not None else (~candidate_mask),
            "local_frontier_candidate_mask": local_frontier_mask if local_frontier_mask is not None else (~candidate_mask),
            "level3_selected_idx": level3_selected_idx,
            "level3_decision_code": level3_decision_code,
            "intervention_level": recovery["recovery_mode"].to(a_t.dtype) + 1.0,
            "score_before_mean": self._masked_mean(etp_candidate_scores, before_valid),
            "score_after_mean": self._masked_mean(rescored_scores, after_valid),
            "d_t_norm": a_t,
            "dt_threshold_low": torch.full_like(a_t, self.dual_recovery.tau_low),
            "dt_threshold_high": torch.full_like(a_t, self.dual_recovery.tau_high),
            "dt_temperature": torch.ones_like(a_t),
            "cand_health_scores": torch.zeros_like(etp_candidate_scores),
            "cand_curve_health_scores": torch.zeros_like(etp_candidate_scores),
            "attn_health_scores": torch.zeros_like(etp_candidate_scores),
            "health_scalar": torch.zeros_like(a_t),
            "curve_health_mean": torch.zeros_like(a_t),
            "cand_health_mean": torch.zeros_like(a_t),
            "attn_health_mean": torch.zeros_like(a_t),
            "health_bonus_mean": torch.zeros_like(a_t),
            "progress_curve": torch.zeros_like(etp_candidate_scores[..., :1]),
            "current_attn_position": torch.zeros_like(a_t),
            "predicted_attn_position": torch.zeros_like(etp_candidate_scores),
            "D_t": a_t,
            "D_t_fused": a_t,
            "D_t_latent": c_micro,
            "D_t_node": c_macro,
        }

    def forward(self, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward_step(**kwargs)
