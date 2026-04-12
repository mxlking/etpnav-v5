from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vlnce_baselines.models.efes_self import (
    ActionConditioner,
    MacroDiagnosisHead,
    MicroRSSMV2,
    RealityEncoder,
    SelfBinder,
    SelfClarityHead,
    SelfMismatchAggregator,
    SelfState,
    ViabilityPredictor,
)


class EFESSelfAgent(nn.Module):
    PROCEED = 0
    BELIEF_UPDATE = 1
    RECOVER = 2
    NUM_MODES = 3

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
        diag_sigma_min: float = 0.05,
        diag_sigma_max: float = 5.0,
        eps_path: float = 0.3,
        tau_low: float = 1.0,
        tau_high: float = 2.5,
        stop_idx: int = 0,
        conditioner_max_delta: float = 1.0,
        conditioner_gate_bias: float = -2.0,
        conditioner_mode: str = "prob_mixture",
        conditioner_beta_max: float = 0.1,
        conditioner_prob_eps: float = 1e-8,
        conditioner_rupture_threshold: float = 0.75,
        conditioner_eligibility_temperature: float = 0.1,
        conditioner_stop_delta_scale: float = 1.0,
        use_macro_rupture: bool = True,
        use_grounding_rupture: bool = True,
        use_typed_rupture: bool = True,
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
        self.eps_path = float(eps_path)
        self.tau_low = float(tau_low)
        self.tau_high = float(tau_high)
        self.stop_idx = int(stop_idx)
        self.conditioner_max_delta = float(conditioner_max_delta)
        self.conditioner_gate_bias = float(conditioner_gate_bias)
        self.conditioner_mode = str(conditioner_mode).strip().lower()
        self.conditioner_beta_max = float(conditioner_beta_max)
        self.conditioner_prob_eps = float(conditioner_prob_eps)
        self.conditioner_rupture_threshold = float(conditioner_rupture_threshold)
        self.conditioner_eligibility_temperature = float(conditioner_eligibility_temperature)
        self.conditioner_stop_delta_scale = float(conditioner_stop_delta_scale)
        self.use_macro_rupture = bool(use_macro_rupture)
        self.use_grounding_rupture = bool(use_grounding_rupture)
        self.use_typed_rupture = bool(use_typed_rupture)

        self.self_binder = SelfBinder(
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
        self.macro_head = MacroDiagnosisHead(
            feat_dim=self.x_dim,
            sigma_min=float(sigma_min),
            diag_sigma_min=float(diag_sigma_min),
            diag_sigma_max=float(diag_sigma_max),
        )
        self.viability = ViabilityPredictor(
            d_model=self.d_model,
            d_action=self.d_action,
            local_dim=self.x_dim,
        )
        self.reality = RealityEncoder()
        self.self_mismatch = SelfMismatchAggregator()
        self.self_clarity = SelfClarityHead()
        self.action_encoder = nn.Linear(self.cand_dim, self.d_action)
        self.action_conditioner = ActionConditioner(
            d_model=self.d_model,
            cand_dim=self.cand_dim,
            max_delta=self.conditioner_max_delta,
            gate_bias=self.conditioner_gate_bias,
            mode=self.conditioner_mode,
            beta_max=self.conditioner_beta_max,
            prob_eps=self.conditioner_prob_eps,
            rupture_threshold=self.conditioner_rupture_threshold,
            eligibility_temperature=self.conditioner_eligibility_temperature,
            stop_delta_scale=self.conditioner_stop_delta_scale,
            stop_idx=self.stop_idx,
        )
        self.reality_proj = nn.Sequential(
            nn.Linear(self.x_dim + 8, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.revision_gate = nn.Sequential(
            nn.Linear(self.d_model * 2 + 5, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )
        self.agency_head = nn.Linear(self.d_model, 1)
        self.phase_head = nn.Linear(self.d_model, 1)
        self.continuity_head = nn.Linear(self.d_model, 1)

    @classmethod
    def from_config(
        cls,
        config: Any,
        x_dim: int,
        lang_dim: int,
        cand_dim: int,
    ) -> "EFESSelfAgent":
        efes_cfg = config.EFES_SELF
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
            diag_sigma_min=float(efes_cfg.diag_sigma_min),
            diag_sigma_max=float(efes_cfg.diag_sigma_max),
            eps_path=float(efes_cfg.eps_path),
            tau_low=float(efes_cfg.tau_low),
            tau_high=float(efes_cfg.tau_high),
            stop_idx=int(efes_cfg.stop_idx),
            conditioner_max_delta=float(getattr(efes_cfg, "conditioner_max_delta", 1.0)),
            conditioner_gate_bias=float(getattr(efes_cfg, "conditioner_gate_bias", -2.0)),
            conditioner_mode=str(getattr(efes_cfg, "conditioner_mode", "prob_mixture")),
            conditioner_beta_max=float(getattr(efes_cfg, "conditioner_beta_max", 0.1)),
            conditioner_prob_eps=float(getattr(efes_cfg, "conditioner_prob_eps", 1e-8)),
            conditioner_rupture_threshold=float(getattr(efes_cfg, "conditioner_rupture_threshold", 0.75)),
            conditioner_eligibility_temperature=float(
                getattr(efes_cfg, "conditioner_eligibility_temperature", 0.1)
            ),
            conditioner_stop_delta_scale=float(getattr(efes_cfg, "conditioner_stop_delta_scale", 1.0)),
            use_macro_rupture=bool(getattr(efes_cfg, "use_macro_rupture", True)),
            use_grounding_rupture=bool(getattr(efes_cfg, "use_grounding_rupture", True)),
            use_typed_rupture=bool(getattr(efes_cfg, "use_typed_rupture", True)),
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

    def apply_static_freeze(self) -> None:
        return None

    @staticmethod
    def _masked_mean(scores: Tensor, valid_mask: Optional[Tensor]) -> Tensor:
        if valid_mask is None:
            valid_mask = torch.isfinite(scores)
        else:
            valid_mask = valid_mask & torch.isfinite(scores)
        count = valid_mask.sum(dim=-1).clamp_min(1)
        safe_scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
        return safe_scores.sum(dim=-1) / count.to(scores.dtype)

    def _minimal_behavior(
        self,
        etp_scores: Tensor,
        candidate_mask: Tensor,
        self_mismatch: Tensor,
        delta_control: Tensor,
        delta_boundary: Tensor,
        delta_progress: Tensor,
        history_backtrack_values: Optional[Tensor],
        history_valid_mask: Optional[Tensor],
    ) -> dict[str, Tensor]:
        neg_large = torch.tensor(-1.0e4, device=etp_scores.device, dtype=etp_scores.dtype)
        proceed_logits = etp_scores.masked_fill(candidate_mask, neg_large)
        belief_logits = proceed_logits
        recover_logits = torch.full_like(etp_scores, neg_large)
        num_candidates = etp_scores.size(1)
        default_idx = self.stop_idx if 0 <= self.stop_idx < num_candidates else 0
        for batch_idx in range(etp_scores.size(0)):
            chosen_idx = default_idx
            if history_backtrack_values is not None and history_valid_mask is not None:
                backtrack_scores = history_backtrack_values[batch_idx].masked_fill(
                    ~history_valid_mask[batch_idx], -float("inf")
                )
                if torch.isfinite(backtrack_scores).any():
                    chosen_idx = int(backtrack_scores.argmax().item())
                recover_logits[batch_idx, chosen_idx] = 10.0 + self_mismatch[batch_idx]

        dominant_delta = torch.stack([delta_control, delta_boundary, delta_progress], dim=-1).argmax(dim=-1)
        recover_mask = self_mismatch >= self.tau_high
        recover_mask = recover_mask | dominant_delta.eq(1)
        belief_mask = (self_mismatch >= self.tau_low) & (~recover_mask) & dominant_delta.ne(1)

        argmax_mode = torch.full_like(dominant_delta, self.PROCEED)
        argmax_mode = torch.where(belief_mask, torch.full_like(argmax_mode, self.BELIEF_UPDATE), argmax_mode)
        argmax_mode = torch.where(recover_mask, torch.full_like(argmax_mode, self.RECOVER), argmax_mode)
        mode_probs = F.one_hot(argmax_mode, num_classes=self.NUM_MODES).to(dtype=etp_scores.dtype)
        mode_logits = mode_probs
        mode_entropy = torch.zeros_like(self_mismatch)
        per_mode_logits = torch.stack([proceed_logits, belief_logits, recover_logits], dim=1)
        batch_idx = torch.arange(per_mode_logits.size(0), device=per_mode_logits.device)
        fused_logits = per_mode_logits[batch_idx, argmax_mode]
        hard_logits = per_mode_logits[batch_idx, argmax_mode]
        update_strength = torch.maximum(delta_control, delta_progress).clamp(min=0.0, max=1.0)
        alpha_prior = torch.where(argmax_mode.eq(self.BELIEF_UPDATE), 1.0 + update_strength, torch.ones_like(self_mismatch))
        alpha_progress = torch.where(
            argmax_mode.eq(self.RECOVER),
            torch.zeros_like(self_mismatch),
            torch.ones_like(self_mismatch),
        )
        return {
            "mode_logits": mode_logits,
            "mode_probs": mode_probs,
            "mode_entropy": mode_entropy,
            "argmax_mode": argmax_mode,
            "fused_logits": fused_logits,
            "hard_logits": hard_logits,
            "alpha_prior": alpha_prior,
            "alpha_progress": alpha_progress,
        }

    def _build_reality_summary(
        self,
        macro_out: Dict[str, Tensor],
        reality: Dict[str, Tensor],
        frontier_size: Tensor,
        revisit_count: Tensor,
        loop_evidence: Tensor,
    ) -> Dict[str, Tensor]:
        progress_proxy = (
            reality["path_delta"].clamp_min(0.0) / max(self.eps_path, 1e-6)
            + reality["topo_advanced"]
            + (1.0 - reality["ground_diag"])
        ) / 3.0
        return {
            "cur_node_feat_compressed": macro_out["x_comp"],
            "cur_vp_changed": reality["topo_advanced"],
            "topo_novelty_actual": macro_out["topo_novelty"],
            "frontier_delta_actual": reality["frontier_delta"],
            "frontier_size_actual": frontier_size,
            "path_delta": reality["path_delta"],
            "revisit_count": revisit_count,
            "loop_indicator": loop_evidence,
            "ground_diag": reality["ground_diag"],
            "progress_proxy_actual": progress_proxy.clamp(0.0, 1.0),
        }

    @staticmethod
    def _mean_feature_error(pred: Tensor, target: Tensor, var: Tensor, var_floor: float = 1e-4) -> Tensor:
        safe_var = var.clamp_min(float(var_floor))
        return ((pred - target).square() / safe_var).mean(dim=-1)

    def _attribute_rupture(
        self,
        viability_out: Dict[str, Tensor],
        micro_diag: Tensor,
        macro_out: Dict[str, Tensor],
        reality_summary: Dict[str, Tensor],
        kappa_self: Tensor,
    ) -> Dict[str, Tensor]:
        local_train_error = self._mean_feature_error(
            viability_out["local_mu"],
            reality_summary["cur_node_feat_compressed"],
            viability_out["local_var"],
        )
        local_diag_error = self._mean_feature_error(
            viability_out["local_mu"],
            reality_summary["cur_node_feat_compressed"],
            viability_out["local_var"],
            var_floor=5e-2,
        )
        topo_vp = F.binary_cross_entropy_with_logits(
            viability_out["vp_change_logit"],
            reality_summary["cur_vp_changed"],
            reduction="none",
        )
        topo_novelty = (viability_out["novelty_pred"] - reality_summary["topo_novelty_actual"]).square()
        topo_frontier = F.smooth_l1_loss(
            viability_out["frontier_delta_pred"],
            reality_summary["frontier_delta_actual"],
            reduction="none",
        )
        progress_gap = F.relu(
            viability_out["delta_progress_pred"] - reality_summary["progress_proxy_actual"]
        )
        advance_target = reality_summary["progress_proxy_actual"].gt(0.34).to(dtype=kappa_self.dtype)
        advance_loss = F.binary_cross_entropy_with_logits(
            viability_out["advance_confidence_logit"],
            advance_target,
            reduction="none",
        )

        r_local_raw = torch.log1p(local_diag_error) + 0.1 * torch.relu(micro_diag - 1.0)
        r_topo_raw = topo_vp + topo_novelty + 0.5 * topo_frontier + 0.25 * macro_out["macro_diag_score"]
        r_ground_raw = (
            0.6 * progress_gap
            + 0.4 * advance_loss
            + 0.25 * reality_summary["ground_diag"]
            + 0.1 * reality_summary["revisit_count"].clamp_min(0.0)
            + 0.1 * reality_summary["loop_indicator"]
        )
        macro_valid_mask = macro_out["macro_valid_mask"]
        if not self.use_macro_rupture:
            r_topo_raw = torch.zeros_like(r_topo_raw)
            macro_valid_mask = torch.zeros_like(macro_valid_mask, dtype=torch.bool)
        if not self.use_grounding_rupture:
            r_ground_raw = torch.zeros_like(r_ground_raw)

        mismatch = self.self_mismatch(
            micro_diag=r_local_raw,
            macro_diag_score=r_topo_raw,
            ground_diag=r_ground_raw,
            macro_valid_mask=macro_valid_mask,
            kappa_self=kappa_self,
        )
        type_logits = torch.stack(
            [
                mismatch["delta_control"],
                mismatch["delta_boundary"],
                mismatch["delta_progress"],
            ],
            dim=-1,
        )
        if self.use_typed_rupture:
            r_local = mismatch["delta_control"]
            r_topo = mismatch["delta_boundary"]
            r_ground = mismatch["delta_progress"]
            z_micro = mismatch["z_micro"]
            z_macro = mismatch["z_macro"]
            z_ground = mismatch["z_ground"]
            type_probs = torch.softmax(type_logits, dim=-1)
            rupture_confidence = torch.sigmoid(type_logits.max(dim=-1)[0])
        else:
            scalar = mismatch["self_mismatch"]
            r_local = scalar
            r_topo = scalar
            r_ground = scalar
            z_micro = scalar
            z_macro = scalar
            z_ground = scalar
            type_probs = torch.full(
                (scalar.size(0), 3),
                1.0 / 3.0,
                device=scalar.device,
                dtype=scalar.dtype,
            )
            rupture_confidence = torch.sigmoid(scalar)

        local_pred_loss = 0.5 * (
            local_train_error + viability_out["local_var"].log().clamp_min(0.0).mean(dim=-1)
        )
        topo_pred_loss = topo_vp + topo_novelty + topo_frontier
        prog_pred_loss = F.smooth_l1_loss(
            viability_out["delta_progress_pred"],
            reality_summary["progress_proxy_actual"],
            reduction="none",
        ) + advance_loss

        return {
            "r_local": r_local,
            "r_topo": r_topo,
            "r_ground": r_ground,
            "rupture_confidence": rupture_confidence,
            "type_probs": type_probs,
            "self_mismatch": mismatch["self_mismatch"],
            "z_micro": z_micro,
            "z_macro": z_macro,
            "z_ground": z_ground,
            "local_pred_loss": local_pred_loss,
            "topo_pred_loss": topo_pred_loss,
            "prog_pred_loss": prog_pred_loss,
            "progress_target": reality_summary["progress_proxy_actual"],
        }

    def _revise_self(
        self,
        self_pred: SelfState,
        reality_summary: Dict[str, Tensor],
        rupture: Dict[str, Tensor],
    ) -> SelfState:
        scalar_summary = torch.stack(
            [
                reality_summary["cur_vp_changed"],
                reality_summary["topo_novelty_actual"],
                reality_summary["frontier_delta_actual"],
                reality_summary["path_delta"],
                reality_summary["revisit_count"],
                reality_summary["loop_indicator"],
                reality_summary["ground_diag"],
                reality_summary["progress_proxy_actual"],
            ],
            dim=-1,
        )
        reality_feat = self.reality_proj(
            torch.cat([reality_summary["cur_node_feat_compressed"], scalar_summary], dim=-1)
        )
        revision_gate = torch.sigmoid(
            self.revision_gate(
                torch.cat(
                    [
                        self_pred.z_self,
                        reality_feat,
                        rupture["r_local"].unsqueeze(-1),
                        rupture["r_topo"].unsqueeze(-1),
                        rupture["r_ground"].unsqueeze(-1),
                        rupture["rupture_confidence"].unsqueeze(-1),
                        self_pred.continuity,
                    ],
                    dim=-1,
                )
            )
        )
        z_post = revision_gate * self_pred.z_self + (1.0 - revision_gate) * reality_feat
        agency = torch.sigmoid(self.agency_head(z_post) - rupture["rupture_confidence"].unsqueeze(-1))
        phase = torch.sigmoid(self.phase_head(z_post) + reality_summary["progress_proxy_actual"].unsqueeze(-1))
        continuity = torch.sigmoid(
            self.continuity_head(z_post) - rupture["rupture_confidence"].unsqueeze(-1)
        )
        rupture_memory = torch.cat(
            [
                rupture["type_probs"],
                rupture["rupture_confidence"].unsqueeze(-1),
            ],
            dim=-1,
        )
        valid_mask = torch.ones_like(continuity)
        return SelfState(
            z_self=z_post,
            agency=agency,
            phase=phase,
            continuity=continuity,
            rupture_memory=rupture_memory,
            valid_mask=valid_mask,
        )

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
        path_delta: Tensor,
        topo_advanced: Tensor,
        frontier_delta: Tensor,
        topo_bank_feat: Tensor,
        topo_bank_mask: Tensor,
        macro_valid_mask: Tensor,
        candidate_mask: Tensor,
        invalid_candidate_mask: Tensor,
        frontier_size: Tensor,
        revisit_count: Tensor,
        loop_evidence: Tensor,
        history_backtrack_values: Optional[Tensor] = None,
        history_valid_mask: Optional[Tensor] = None,
        lang_mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        z_self, kappa_self, progress_t, binding_stats = self.self_binder(
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
            s_t=z_self,
            a_prev_emb=prev_action_emb,
            obs_feat=x_t,
            h_prev=prev_rssm_h,
            prior_alpha=prev_prior_alpha,
        )
        micro_diag = micro_out["c_micro_diag"]
        self_pred = SelfState(
            z_self=z_self,
            agency=kappa_self.unsqueeze(-1),
            phase=progress_t,
            continuity=kappa_self.unsqueeze(-1),
            rupture_memory=torch.zeros(z_self.size(0), 4, device=z_self.device, dtype=z_self.dtype),
            valid_mask=torch.ones(z_self.size(0), 1, device=z_self.device, dtype=z_self.dtype),
        )

        has_bank = topo_bank_mask.any(dim=-1)
        topo_context = z_self
        if topo_bank_feat.size(1) > 0:
            topo_context = self.macro_head.retrieve(z_self, topo_bank_feat, topo_bank_mask)
        macro_out = self.macro_head(
            node_feat=node_feat,
            topo_context=topo_context,
            bank_feat=topo_bank_feat,
            bank_mask=topo_bank_mask,
            macro_valid_mask=(has_bank & macro_valid_mask),
        )
        reality = self.reality(
            path_delta=path_delta,
            topo_advanced=topo_advanced,
            frontier_delta=frontier_delta,
            eps_path=self.eps_path,
        )
        reality_summary = self._build_reality_summary(
            macro_out=macro_out,
            reality=reality,
            frontier_size=frontier_size,
            revisit_count=revisit_count,
            loop_evidence=loop_evidence,
        )
        viability_out = self.viability(self_pred.z_self, prev_action_emb)
        rupture_out = self._attribute_rupture(
            viability_out=viability_out,
            micro_diag=micro_diag,
            macro_out=macro_out,
            reality_summary=reality_summary,
            kappa_self=kappa_self,
        )
        self_clarity = self.self_clarity(
            kappa_self=kappa_self,
            micro_recent_mean=rupture_out["r_local"],
            macro_recent_mean=rupture_out["r_topo"],
            self_mismatch_recent=rupture_out["self_mismatch"],
            progress_gap=rupture_out["r_ground"],
        )
        self_post = self._revise_self(
            self_pred=self_pred,
            reality_summary=reality_summary,
            rupture=rupture_out,
        )
        behavior = self._minimal_behavior(
            etp_scores=etp_candidate_scores,
            candidate_mask=candidate_mask,
            self_mismatch=rupture_out["self_mismatch"],
            delta_control=rupture_out["r_local"],
            delta_boundary=rupture_out["r_topo"],
            delta_progress=rupture_out["r_ground"],
            history_backtrack_values=history_backtrack_values,
            history_valid_mask=history_valid_mask,
        )
        if invalid_candidate_mask is None:
            invalid_candidate_mask = candidate_mask
        conditioned = self.action_conditioner(
            nav_logits=etp_candidate_scores,
            candidate_embeds=cand_feats,
            self_post_z=self_post.z_self,
            self_continuity=self_post.continuity,
            r_local=rupture_out["r_local"],
            r_topo=rupture_out["r_topo"],
            r_ground=rupture_out["r_ground"],
            rupture_confidence=rupture_out["rupture_confidence"],
            type_probs=rupture_out["type_probs"],
            invalid_candidate_mask=invalid_candidate_mask,
        )
        adjusted_progress = self_post.phase * behavior["alpha_progress"].unsqueeze(-1)
        valid_after = torch.isfinite(behavior["hard_logits"])
        return {
            "s_t": self_pred.z_self,
            "h_t": self_post.z_self,
            "rssm_h_t": micro_out["h_t"],
            "z_post": micro_out["z_post"],
            "z_flat": micro_out["z_post"],
            "progress_t": adjusted_progress,
            "attn_entropy": binding_stats["attn_entropy"],
            "prior_mu": micro_out["mu_pri"],
            "prior_std": micro_out["std_pri"],
            "post_mu": micro_out["mu_post"],
            "post_std": micro_out["std_post"],
            "C_micro_kl_raw": micro_out["c_micro_kl_raw"],
            "C_micro": micro_diag,
            "C_macro_diag": macro_out["macro_diag_score"],
            "U_macro": macro_out["macro_uncertainty"],
            "g_t": reality["ground_diag"],
            "A_t": rupture_out["self_mismatch"],
            "z_micro": rupture_out["z_micro"],
            "z_macro": rupture_out["z_macro"],
            "z_ground": rupture_out["z_ground"],
            "delta_control": rupture_out["r_local"],
            "delta_boundary": rupture_out["r_topo"],
            "delta_progress": rupture_out["r_ground"],
            "self_mismatch": rupture_out["self_mismatch"],
            "kappa_self": kappa_self,
            "self_clarity": self_clarity,
            "self_pred_z": self_pred.z_self,
            "self_post_z": self_post.z_self,
            "self_agency": self_post.agency.squeeze(-1),
            "self_phase": self_post.phase.squeeze(-1),
            "self_continuity": self_post.continuity.squeeze(-1),
            "self_pred_continuity": self_pred.continuity.squeeze(-1),
            "rupture_confidence": rupture_out["rupture_confidence"],
            "type_probs": rupture_out["type_probs"],
            "mode_logits": behavior["mode_logits"],
            "mode_probs": behavior["mode_probs"],
            "mode_entropy": behavior["mode_entropy"],
            "argmax_mode": behavior["argmax_mode"],
            "alpha_prior": behavior["alpha_prior"],
            "alpha_progress": behavior["alpha_progress"],
            "local_mu": viability_out["local_mu"],
            "local_var": viability_out["local_var"],
            "vp_change_logit": viability_out["vp_change_logit"],
            "novelty_pred": viability_out["novelty_pred"],
            "frontier_delta_pred": viability_out["frontier_delta_pred"],
            "delta_progress_pred": viability_out["delta_progress_pred"],
            "advance_confidence_logit": viability_out["advance_confidence_logit"],
            "mu_pred": macro_out["mu_pred"],
            "sigma2_pred": macro_out["sigma2"],
            "compressed_node_feat": macro_out["x_comp"],
            "topo_novelty": macro_out["topo_novelty"],
            "macro_valid_mask": has_bank & macro_valid_mask,
            "fused_logits": behavior["fused_logits"],
            "hard_logits": behavior["hard_logits"],
            "safe_logits": conditioned["safe_logits"],
            "condition_gate": conditioned["condition_gate"],
            "condition_beta": conditioned["condition_beta"],
            "condition_raw_gate": conditioned["condition_raw_gate"],
            "condition_eligibility": conditioned["condition_eligibility"],
            "condition_delta": conditioned["condition_delta"],
            "condition_residual": conditioned["condition_residual"],
            "condition_policy_kl": conditioned["condition_policy_kl"],
            "score_before_mean": self._masked_mean(etp_candidate_scores, ~invalid_candidate_mask),
            "score_after_mean": self._masked_mean(conditioned["safe_logits"], ~invalid_candidate_mask),
            "score_hard_mean": self._masked_mean(behavior["hard_logits"], valid_after),
            "D_t": rupture_out["self_mismatch"],
            "D_t_fused": rupture_out["self_mismatch"],
            "D_t_latent": micro_diag,
            "D_t_node": macro_out["macro_diag_score"],
            "node_nll_loss": macro_out["node_nll_loss"],
            "local_pred_loss": rupture_out["local_pred_loss"],
            "topo_pred_loss": rupture_out["topo_pred_loss"],
            "prog_pred_loss": rupture_out["prog_pred_loss"],
            "progress_target": rupture_out["progress_target"],
        }

    def forward(self, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward_step(**kwargs)
