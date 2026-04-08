from __future__ import annotations

"""StateNav V6 main agent.

V6 keeps the micro temporal RSSM from V5 and adds a topo-level surprise head.
The fused self-diagnosis signal is:

    D_t_fused = alpha_latent * D_t_latent + beta_node * D_t_node

Macro intervention still consumes the diagnosis signal and produces the final
rescored candidate logits. V5-only curve/attention branches are kept as
zero-output compatibility placeholders so the V6 trainer can reuse the
existing action/debug plumbing without training those heads.
"""

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor, nn

from vlnce_baselines.models.statenav_v5.attention_head import AttentionHead
from vlnce_baselines.models.statenav_v5.attn_health_scorer import AttnHealthScorer
from vlnce_baselines.models.statenav_v5.losses import latent_kl_loss
from vlnce_baselines.models.statenav_v5.macro_intervention_manager import MacroInterventionManager
from vlnce_baselines.models.statenav_v5.progress_curve_predictor import (
    ProgressCurvePredictor,
)
from vlnce_baselines.models.statenav_v5.relational_state_extractor import RelationalStateExtractor
from vlnce_baselines.models.statenav_v5.self_rssm_correction import SelfRSSMCorrection
from vlnce_baselines.models.statenav_v5.self_rssm_latent import SelfRSSMLatent
from vlnce_baselines.models.statenav_v5.self_rssm_transition import SelfRSSMTransition
from vlnce_baselines.models.statenav_v5.state_head import StateHead
from vlnce_baselines.models.statenav_v5.unified_rollout_trunk import UnifiedRolloutTrunk
from vlnce_baselines.models.statenav_v6.node_surprise_head import NodeSurpriseHead


class StateNavAgentV6(nn.Module):
    """ASM-VLN V6 agent with topo-gated node surprise."""

    def __init__(
        self,
        x_dim: int,
        g_dim: int,
        lang_dim: int,
        cand_dim: int,
        hidden_dim: int = 512,
        action_dim: int = 512,
        latent_groups: int = 16,
        latent_classes: int = 16,
        use_reality_calibration: bool = False,
        decision_mode: str = "hierarchical_self_model",
        decision_beta_d: float = 1.0,
        lambda_health: float = 0.1,
        progress_curve_horizon: int = 5,
        progress_curve_hidden_dim: int = 256,
        relation_dim: int = 512,
        lambda_attn_health: float = 0.25,
        lambda_curve_health: float = 1.0,
        dt_low_percentile: float = 0.75,
        dt_high_percentile: float = 0.95,
        dt_threshold_ema: float = 0.95,
        high_level_backtrack_bias: float = 5.0,
        enable_level2_frontier: bool = True,
        enable_level3_macro: bool = True,
        level3_policy: str = "energy_recovery",
        level3_min_step: int = 3,
        level3_min_dnorm: float = 1.0,
        level3_switch_margin: float = 0.0,
        level3_allow_stop: bool = False,
        level3_stop_margin: float = 0.0,
        backtrack_history_size: int = 8,
        backtrack_recency_bias: float = 0.05,
        use_relation_state_extractor: bool = False,
        use_attention_transition: bool = False,
        use_open_eye_rollout: bool = False,
        alpha_latent: float = 1.0,
        beta_node: float = 1.0,
        use_topo_bank: bool = True,
        use_topo_gate: bool = True,
        use_node_surprise: bool = True,
        lambda_node_surprise: float = 0.5,
        node_surprise_hidden_dim: int = 256,
        stop_idx: int = 0,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.z_dim = latent_groups * latent_classes
        self.topo_state_dim = hidden_dim + self.z_dim
        self.use_reality_calibration = bool(use_reality_calibration)
        self.decision_mode = str(decision_mode)
        self.decision_beta_d = float(decision_beta_d)
        self.lambda_health = float(lambda_health)
        self.lambda_attn_health = float(lambda_attn_health)
        self.lambda_curve_health = float(lambda_curve_health)
        self.alpha_latent = float(alpha_latent)
        self.beta_node = float(beta_node)
        self.lambda_node_surprise = float(lambda_node_surprise)
        self.use_topo_bank = bool(use_topo_bank)
        self.use_topo_gate = bool(use_topo_gate)
        self.use_node_surprise = bool(use_node_surprise)
        self.use_relation_state_extractor = bool(use_relation_state_extractor)
        self.use_attention_transition = bool(use_attention_transition)
        self.use_open_eye_rollout = bool(use_open_eye_rollout)

        self.action_encoder = nn.Linear(cand_dim, action_dim)
        self.self_rssm_transition = SelfRSSMTransition(
            hidden_dim=hidden_dim,
            lang_dim=lang_dim,
            action_dim=action_dim,
            z_dim=self.z_dim,
        )
        self.self_rssm_correction = SelfRSSMCorrection(
            hidden_dim=hidden_dim,
            x_dim=x_dim,
            g_dim=g_dim,
            lang_dim=lang_dim,
        )
        self.self_rssm_latent = SelfRSSMLatent(
            hidden_dim=hidden_dim,
            x_dim=x_dim,
            g_dim=g_dim,
            lang_dim=lang_dim,
            latent_groups=latent_groups,
            latent_classes=latent_classes,
        )
        self.relational_state_extractor = RelationalStateExtractor(
            hidden_dim=hidden_dim,
            z_dim=self.z_dim,
            token_dim=x_dim,
            relation_dim=relation_dim,
        )
        self.progress_curve_predictor = ProgressCurvePredictor(
            hidden_dim=hidden_dim,
            z_dim=self.z_dim,
            cand_dim=cand_dim,
            horizon=progress_curve_horizon,
            mlp_hidden_dim=progress_curve_hidden_dim,
            relation_dim=relation_dim,
        )
        self.unified_rollout_trunk = UnifiedRolloutTrunk(
            hidden_dim=hidden_dim,
            z_dim=self.z_dim,
            token_dim=x_dim,
            cand_dim=cand_dim,
            relation_dim=relation_dim,
        )
        self.state_head = StateHead(relation_dim=relation_dim, hidden_dim=progress_curve_hidden_dim)
        self.attention_head = AttentionHead(
            relation_dim=relation_dim,
            lang_dim=lang_dim,
        )
        self.attn_health_scorer = AttnHealthScorer()
        self.node_surprise_head = NodeSurpriseHead(
            topo_state_dim=self.topo_state_dim,
            feat_dim=x_dim,
            hidden_dim=node_surprise_hidden_dim,
        )
        self.macro_intervention_manager = MacroInterventionManager(
            lambda_health=lambda_health,
            low_percentile=dt_low_percentile,
            high_percentile=dt_high_percentile,
            threshold_ema=dt_threshold_ema,
            high_level_backtrack_bias=high_level_backtrack_bias,
            stop_idx=stop_idx,
            enable_level2_frontier=enable_level2_frontier,
            enable_level3_macro=enable_level3_macro,
            level3_policy=level3_policy,
            level3_min_step=level3_min_step,
            level3_min_dnorm=level3_min_dnorm,
            level3_switch_margin=level3_switch_margin,
            level3_allow_stop=level3_allow_stop,
            level3_stop_margin=level3_stop_margin,
        )
        self.backtrack_history_size = int(backtrack_history_size)
        self.backtrack_recency_bias = float(backtrack_recency_bias)
        self.enable_level2_frontier = bool(enable_level2_frontier)
        self.enable_level3_macro = bool(enable_level3_macro)
        self.level3_policy = str(level3_policy)

    @classmethod
    def from_config(
        cls,
        config: Any,
        x_dim: int,
        g_dim: int,
        lang_dim: int,
        cand_dim: int,
    ) -> "StateNavAgentV6":
        return cls(
            x_dim=x_dim,
            g_dim=g_dim,
            lang_dim=lang_dim,
            cand_dim=cand_dim,
            hidden_dim=int(config.STATENAV.hidden_dim),
            action_dim=int(config.STATENAV.action_dim),
            latent_groups=int(config.STATENAV.latent_groups),
            latent_classes=int(config.STATENAV.latent_classes),
            use_reality_calibration=bool(config.STATENAV.use_reality_calibration),
            decision_mode=str(config.STATENAV.decision_mode),
            decision_beta_d=float(config.STATENAV.decision_beta_d),
            lambda_health=float(config.STATENAV.lambda_health),
            progress_curve_horizon=int(config.STATENAV.progress_curve_horizon),
            progress_curve_hidden_dim=int(config.STATENAV.progress_curve_hidden_dim),
            relation_dim=int(config.STATENAV.relation_dim),
            lambda_attn_health=float(config.STATENAV.lambda_attn_health),
            lambda_curve_health=float(getattr(config.STATENAV, "lambda_curve_health", 1.0)),
            dt_low_percentile=float(config.STATENAV.dt_low_percentile),
            dt_high_percentile=float(config.STATENAV.dt_high_percentile),
            dt_threshold_ema=float(config.STATENAV.dt_threshold_ema),
            high_level_backtrack_bias=float(config.STATENAV.high_level_backtrack_bias),
            enable_level2_frontier=bool(config.STATENAV.enable_level2_frontier),
            enable_level3_macro=bool(config.STATENAV.enable_level3_macro),
            level3_policy=str(config.STATENAV.level3_policy),
            level3_min_step=int(config.STATENAV.level3_min_step),
            level3_min_dnorm=float(config.STATENAV.level3_min_dnorm),
            level3_switch_margin=float(config.STATENAV.level3_switch_margin),
            level3_allow_stop=bool(config.STATENAV.level3_allow_stop),
            level3_stop_margin=float(config.STATENAV.level3_stop_margin),
            backtrack_history_size=int(config.STATENAV.backtrack_history_size),
            backtrack_recency_bias=float(config.STATENAV.backtrack_recency_bias),
            use_relation_state_extractor=bool(config.STATENAV.use_relation_state_extractor),
            use_attention_transition=bool(config.STATENAV.use_attention_transition),
            use_open_eye_rollout=bool(config.STATENAV.use_open_eye_rollout),
            alpha_latent=float(getattr(config.STATENAV, "alpha_latent", 1.0)),
            beta_node=float(getattr(config.STATENAV, "beta_node", 1.0)),
            use_topo_bank=bool(getattr(config.STATENAV, "use_topo_bank", True)),
            use_topo_gate=bool(getattr(config.STATENAV, "use_topo_gate", True)),
            use_node_surprise=bool(getattr(config.STATENAV, "use_node_surprise", True)),
            lambda_node_surprise=float(getattr(config.STATENAV, "lambda_node_surprise", 0.5)),
            node_surprise_hidden_dim=int(
                getattr(config.STATENAV, "node_surprise_hidden_dim", 256)
            ),
            stop_idx=int(config.STATENAV.adapter_stop_idx),
        )

    def init_recurrent_state(self, batch_size: int, device: torch.device | str) -> Dict[str, Tensor]:
        return {
            "prev_h": torch.zeros(batch_size, self.hidden_dim, device=device),
            "prev_z": torch.zeros(batch_size, self.z_dim, device=device),
            "prev_action_emb": torch.zeros(batch_size, self.action_dim, device=device),
        }

    def project_action_features(self, action_feats: Tensor) -> Tensor:
        return self.action_encoder(action_feats)

    @staticmethod
    def _validate_offline_seq_shape(
        seq: Tensor,
        batch_size: int,
        seq_len: int,
        name: str,
    ) -> None:
        if seq.size(0) != batch_size or seq.size(1) != seq_len:
            raise ValueError(
                f"{name} must have leading shape [batch, seq_len] = "
                f"[{batch_size}, {seq_len}], got {tuple(seq.shape)}"
            )

    def _select_offline_executed_action_feats(
        self,
        cand_feats_t: Tensor,
        teacher_action_idx_t: Optional[Tensor] = None,
        executed_action_feat_t: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        if teacher_action_idx_t is not None:
            batch_size, num_candidates, _ = cand_feats_t.shape
            teacher_action_idx_t = teacher_action_idx_t.long()
            safe_idx = teacher_action_idx_t.clamp(min=0, max=max(num_candidates - 1, 0))
            gathered = cand_feats_t[
                torch.arange(batch_size, device=cand_feats_t.device),
                safe_idx,
            ]
            valid = (teacher_action_idx_t >= 0) & (teacher_action_idx_t < num_candidates)
            return torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))
        if executed_action_feat_t is not None:
            return executed_action_feat_t
        return None

    def forward_step(
        self,
        x_t: Tensor,
        g_t: Tensor,
        lang_tokens: Tensor,
        cand_feats: Tensor,
        etp_candidate_scores: Tensor,
        prev_action_emb: Tensor,
        prev_z: Tensor,
        prev_h: Tensor,
        lang_mask: Optional[Tensor] = None,
        candidate_mask: Optional[Tensor] = None,
        visited_mask: Optional[Tensor] = None,
        frontier_mask: Optional[Tensor] = None,
        local_frontier_mask: Optional[Tensor] = None,
        x_tokens: Optional[Tensor] = None,
        x_token_mask: Optional[Tensor] = None,
        g_tokens: Optional[Tensor] = None,
        g_token_mask: Optional[Tensor] = None,
        history_backtrack_values: Optional[Tensor] = None,
        history_valid_mask: Optional[Tensor] = None,
        topo_prev_state: Optional[Tensor] = None,
        topo_valid_mask: Optional[Tensor] = None,
        topo_update_mask: Optional[Tensor] = None,
        topo_novelty: Optional[Tensor] = None,
        current_step: Union[int, Tensor] = 0,
        adapter_warmup: float = 1.0,
        temp: float = 1.0,
        hard: bool = True,
    ) -> Dict[str, Tensor]:
        h_prior, lang_glimpse_prior, attn_weights_prior = self.self_rssm_transition(
            prev_h=prev_h,
            prev_z_post=prev_z,
            prev_action_emb=prev_action_emb,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask,
        )
        h_post, lang_glimpse_post, attn_weights_post = self.self_rssm_correction(
            h_prior=h_prior,
            x_t=x_t,
            g_t=g_t,
            lang_tokens=lang_tokens,
            lang_mask=lang_mask,
        )
        latent_outs = self.self_rssm_latent(
            h_prior=h_prior,
            h_post=h_post,
            x_t=x_t,
            g_t=g_t,
            lang_glimpse_obs=lang_glimpse_post,
            temp=temp,
            hard=hard,
        )
        d_t, _ = latent_kl_loss(
            post_logits=latent_outs["post_logits"],
            prior_logits=latent_outs["prior_logits"],
            free_bits=0.0,
            reduction="none",
        )
        d_t_latent = d_t
        predicted_node_feat = x_t.new_zeros(x_t.shape)
        d_t_node = torch.zeros_like(d_t_latent)
        if self.use_topo_bank and self.use_node_surprise and topo_prev_state is not None and topo_valid_mask is not None:
            predicted_node_feat = self.node_surprise_head(topo_prev_state)
            if topo_update_mask is None:
                topo_update_mask = topo_valid_mask
            valid_node_mask = topo_valid_mask.bool() & topo_update_mask.bool()
            d_t_node = self.node_surprise_head.mse_surprise(
                predicted_feat=predicted_node_feat,
                target_feat=x_t,
                valid_mask=valid_node_mask,
            )
        d_t = self.alpha_latent * d_t_latent + self.beta_node * d_t_node
        decision_d_t = d_t if self.use_reality_calibration else torch.zeros_like(d_t)

        batch_size, num_candidates, _ = cand_feats.shape
        lang_len = lang_tokens.size(1)
        relation_state = h_post.new_zeros(batch_size, self.relational_state_extractor.relation_dim)
        relation_attn = h_post.new_zeros(batch_size, 1)
        candidate_states = cand_feats.new_zeros(batch_size, num_candidates, self.relational_state_extractor.relation_dim)
        rollout_out = {
            "unified_state": candidate_states,
            "unified_attn": cand_feats.new_zeros(batch_size, num_candidates, 1),
            "candidate_h_next": h_post.unsqueeze(1).expand(batch_size, num_candidates, h_post.size(-1)),
            "candidate_z_next": latent_outs["z_post_flat"].unsqueeze(1).expand(
                batch_size, num_candidates, latent_outs["z_post_flat"].size(-1)
            ),
        }
        curve_out = {
            "progress_curve": cand_feats.new_zeros(batch_size, 1),
            "health_scalar": cand_feats.new_zeros(batch_size),
            "health_weights": cand_feats.new_zeros(batch_size, 3),
            "health_components": cand_feats.new_zeros(batch_size, 3),
        }
        cand_curve_out = {
            "cand_progress_curves": cand_feats.new_zeros(batch_size, num_candidates, 1),
            "cand_health_scores": cand_feats.new_zeros(batch_size, num_candidates),
            "cand_health_weights": cand_feats.new_zeros(batch_size, num_candidates, 3),
            "cand_health_components": cand_feats.new_zeros(batch_size, num_candidates, 3),
        }
        attn_out = {
            "predicted_attn": cand_feats.new_zeros(batch_size, num_candidates, lang_len),
            "attn_health_scores": cand_feats.new_zeros(batch_size, num_candidates),
            "current_attn_position": cand_feats.new_zeros(batch_size),
            "predicted_attn_position": cand_feats.new_zeros(batch_size, num_candidates),
        }
        combined_cand_health_scores = cand_feats.new_zeros(batch_size, num_candidates)

        invalid_mask = None
        if candidate_mask is not None:
            if visited_mask is None:
                invalid_mask = candidate_mask
            else:
                invalid_mask = candidate_mask & visited_mask.logical_not()
        adapter_outputs = self.macro_intervention_manager(
            etp_candidate_scores=etp_candidate_scores,
            cand_health_scores=combined_cand_health_scores,
            curve_health_scores=cand_curve_out["cand_health_scores"],
            attn_health_scores=attn_out["attn_health_scores"],
            d_t=decision_d_t,
            invalid_mask=invalid_mask,
            visited_mask=visited_mask,
            frontier_mask=frontier_mask,
            local_frontier_mask=local_frontier_mask,
            history_backtrack_values=history_backtrack_values,
            history_valid_mask=history_valid_mask,
            decision_mode=self.decision_mode,
            beta_d=self.decision_beta_d,
            lambda_health=self.lambda_health,
            adapter_warmup=adapter_warmup,
            enable_level2_frontier=self.enable_level2_frontier,
            enable_level3_macro=self.enable_level3_macro,
            level3_policy=self.level3_policy,
            current_step=current_step,
        )

        return {
            "h_t": h_post,
            "h_prior": h_prior,
            "h_post": h_post,
            "lang_glimpse_prior": lang_glimpse_prior,
            "lang_glimpse_post": lang_glimpse_post,
            "prior_logits": latent_outs["prior_logits"],
            "post_logits": latent_outs["post_logits"],
            "z_prior_onehot": latent_outs["z_prior_onehot"],
            "z_post_onehot": latent_outs["z_post_onehot"],
            "z_prior_flat": latent_outs["z_prior_flat"],
            "z_post_flat": latent_outs["z_post_flat"],
            "z_onehot": latent_outs["z_onehot"],
            "z_flat": latent_outs["z_flat"],
            "D_t": d_t,
            "D_t_fused": d_t,
            "D_t_latent": d_t_latent,
            "D_t_node": d_t_node,
            "predicted_node_feat": predicted_node_feat,
            "topo_novelty": topo_novelty if topo_novelty is not None else torch.zeros_like(d_t),
            "relation_state": relation_state,
            "relation_attn": relation_attn,
            "s_t": relation_state,
            "progress_curve": curve_out["progress_curve"],
            "health_scalar": curve_out["health_scalar"],
            "health_weights": curve_out["health_weights"],
            "health_components": curve_out["health_components"],
            "cand_progress_curves": cand_curve_out["cand_progress_curves"],
            "cand_curve_health_scores": cand_curve_out["cand_health_scores"],
            "cand_curve_health_weights": cand_curve_out["cand_health_weights"],
            "cand_curve_health_components": cand_curve_out["cand_health_components"],
            "candidate_relation_states": candidate_states,
            "candidate_relation_attn": rollout_out["unified_attn"],
            "candidate_h_next": rollout_out["candidate_h_next"],
            "candidate_z_next": rollout_out["candidate_z_next"],
            "unified_state": rollout_out["unified_state"],
            "predicted_attn": attn_out["predicted_attn"],
            "attn_health_scores": attn_out["attn_health_scores"],
            "current_attn_position": attn_out["current_attn_position"],
            "predicted_attn_position": attn_out["predicted_attn_position"],
            "cand_health_scores": combined_cand_health_scores,
            "health_bonus": adapter_outputs["health_bonus"],
            "health_bonus_mean": adapter_outputs["health_bonus_mean"],
            "curve_health_mean": adapter_outputs["curve_health_mean"],
            "attn_health_mean": adapter_outputs["attn_health_mean"],
            "cand_health_mean": adapter_outputs["cand_health_mean"],
            "dt_temperature": adapter_outputs["dt_temperature"],
            "d_t_norm": adapter_outputs["d_t_norm"],
            "intervention_level": adapter_outputs["intervention_level"],
            "dt_threshold_low": adapter_outputs["dt_threshold_low"],
            "dt_threshold_high": adapter_outputs["dt_threshold_high"],
            "strict_candidate_mask": adapter_outputs["strict_candidate_mask"],
            "expanded_candidate_mask": adapter_outputs["expanded_candidate_mask"],
            "frontier_candidate_mask": adapter_outputs["frontier_candidate_mask"],
            "local_frontier_candidate_mask": adapter_outputs["local_frontier_candidate_mask"],
            "level3_selected_idx": adapter_outputs["level3_selected_idx"],
            "level3_decision_code": adapter_outputs["level3_decision_code"],
            "level3_policy_code": adapter_outputs["level3_policy_code"],
            "score_before_after": adapter_outputs["score_before_after"],
            "score_before_mean": adapter_outputs["score_before_mean"],
            "score_after_mean": adapter_outputs["score_after_mean"],
            "rescored_candidate_scores": adapter_outputs["rescored_candidate_scores"],
            "attn_weights_prior": attn_weights_prior,
            "attn_weights_post": attn_weights_post,
        }

    def forward(self, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward_step(**kwargs)

    def online_inference_step(self, **kwargs: Any) -> Dict[str, Tensor]:
        return self.forward_step(**kwargs)

    def offline_training_forward(
        self,
        x_seq: Tensor,
        g_seq: Tensor,
        lang_tokens_seq: Tensor,
        cand_feats_seq: Tensor,
        etp_scores_seq: Tensor,
        prev_action_emb_init: Tensor,
        teacher_action_idx_seq: Optional[Tensor] = None,
        executed_action_feat_seq: Optional[Tensor] = None,
        candidate_mask_seq: Optional[Tensor] = None,
        lang_mask_seq: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        batch_size, seq_len = x_seq.shape[:2]
        if teacher_action_idx_seq is not None:
            self._validate_offline_seq_shape(
                teacher_action_idx_seq,
                batch_size=batch_size,
                seq_len=seq_len,
                name="teacher_action_idx_seq",
            )
        if executed_action_feat_seq is not None:
            self._validate_offline_seq_shape(
                executed_action_feat_seq,
                batch_size=batch_size,
                seq_len=seq_len,
                name="executed_action_feat_seq",
            )
            if executed_action_feat_seq.size(-1) != cand_feats_seq.size(-1):
                raise ValueError(
                    "executed_action_feat_seq last dimension must match candidate feature dim, "
                    f"got {executed_action_feat_seq.size(-1)} vs {cand_feats_seq.size(-1)}"
                )
        prev_h = x_seq.new_zeros(batch_size, self.hidden_dim)
        prev_z = x_seq.new_zeros(batch_size, self.z_dim)
        prev_action_emb = prev_action_emb_init

        outputs = []
        prev_action_emb_inputs = []
        prev_action_emb_outputs = []
        for t in range(seq_len):
            prev_action_emb_inputs.append(prev_action_emb)
            step_out = self.forward_step(
                x_t=x_seq[:, t],
                g_t=g_seq[:, t],
                lang_tokens=lang_tokens_seq[:, t],
                cand_feats=cand_feats_seq[:, t],
                etp_candidate_scores=etp_scores_seq[:, t],
                prev_action_emb=prev_action_emb,
                prev_z=prev_z,
                prev_h=prev_h,
                candidate_mask=None if candidate_mask_seq is None else candidate_mask_seq[:, t],
                lang_mask=None if lang_mask_seq is None else lang_mask_seq[:, t],
            )
            outputs.append(step_out)
            prev_h = step_out["h_post"]
            prev_z = step_out["z_flat"]
            executed_action_feats_t = self._select_offline_executed_action_feats(
                cand_feats_t=cand_feats_seq[:, t],
                teacher_action_idx_t=None if teacher_action_idx_seq is None else teacher_action_idx_seq[:, t],
                executed_action_feat_t=None if executed_action_feat_seq is None else executed_action_feat_seq[:, t],
            )
            if executed_action_feats_t is not None:
                # offline recurrent propagation now includes previous executed action
                # embedding to keep prior self-state prediction action-conditioned
                # and consistent with online rollout.
                prev_action_emb = self.project_action_features(executed_action_feats_t)
            prev_action_emb_outputs.append(prev_action_emb)

        stacked: Dict[str, Tensor] = {}
        for key in outputs[0].keys():
            stacked[key] = torch.stack([step_out[key] for step_out in outputs], dim=1)
        stacked["offline_prev_action_emb_input_seq"] = torch.stack(prev_action_emb_inputs, dim=1)
        stacked["offline_prev_action_emb_output_seq"] = torch.stack(prev_action_emb_outputs, dim=1)
        return stacked
