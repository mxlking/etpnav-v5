from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor, nn


class MacroInterventionManager(nn.Module):
    """Three-level D_t intervention controller for V5."""

    def __init__(
        self,
        lambda_health: float = 0.1,
        low_percentile: float = 0.75,
        high_percentile: float = 0.95,
        threshold_ema: float = 0.95,
        high_level_backtrack_bias: float = 5.0,
        stop_idx: int = 0,
        enable_level2_frontier: bool = True,
        enable_level3_macro: bool = True,
        level3_policy: str = "energy_recovery",
        level3_min_step: int = 3,
        level3_min_dnorm: float = 1.0,
        level3_switch_margin: float = 0.0,
        level3_allow_stop: bool = False,
        level3_stop_margin: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_health = float(lambda_health)
        self.low_percentile = float(low_percentile)
        self.high_percentile = float(high_percentile)
        self.threshold_ema = float(threshold_ema)
        self.high_level_backtrack_bias = float(high_level_backtrack_bias)
        self.stop_idx = int(stop_idx)
        self.enable_level2_frontier = bool(enable_level2_frontier)
        self.enable_level3_macro = bool(enable_level3_macro)
        self.level3_policy = str(level3_policy)
        self.level3_min_step = int(level3_min_step)
        self.level3_min_dnorm = float(level3_min_dnorm)
        self.level3_switch_margin = float(level3_switch_margin)
        self.level3_allow_stop = bool(level3_allow_stop)
        self.level3_stop_margin = float(level3_stop_margin)
        self.register_buffer("dt_low_threshold", torch.tensor(0.0))
        self.register_buffer("dt_high_threshold", torch.tensor(0.0))
        self.register_buffer("thresholds_initialized", torch.tensor(False))

    @staticmethod
    def _masked_candidate_mean(scores: Tensor, candidate_mask: Optional[Tensor]) -> Tensor:
        if candidate_mask is None:
            valid_mask = torch.isfinite(scores)
        else:
            valid_mask = candidate_mask.logical_not() & torch.isfinite(scores)
        valid_count = valid_mask.sum(dim=-1)
        safe_scores = torch.where(valid_mask, scores, torch.zeros_like(scores))
        mean_scores = safe_scores.sum(dim=-1) / valid_count.clamp_min(1).to(scores.dtype)
        return torch.where(valid_count > 0, mean_scores, torch.zeros_like(mean_scores))

    def _update_thresholds(self, d_t: Tensor) -> None:
        flat = d_t.detach().float().flatten()
        if flat.numel() == 0:
            return
        low = torch.quantile(flat, q=self.low_percentile).to(self.dt_low_threshold.device)
        high = torch.quantile(flat, q=self.high_percentile).to(self.dt_high_threshold.device)
        if bool(self.thresholds_initialized.item()):
            ema = float(self.threshold_ema)
            self.dt_low_threshold.mul_(ema).add_(low * (1.0 - ema))
            self.dt_high_threshold.mul_(ema).add_(high * (1.0 - ema))
        else:
            self.dt_low_threshold.copy_(low)
            self.dt_high_threshold.copy_(high)
            self.thresholds_initialized.fill_(True)

    def _resolve_thresholds(self, d_t: Tensor) -> tuple[Tensor, Tensor]:
        low_th = self.dt_low_threshold
        high_th = torch.maximum(self.dt_high_threshold, low_th + 1e-6)
        if not bool(self.thresholds_initialized.item()):
            flat = d_t.detach().float().flatten()
            low_th = torch.quantile(flat, q=self.low_percentile)
            high_th = torch.quantile(flat, q=self.high_percentile)
            high_th = torch.maximum(high_th, low_th + 1e-6)
        return low_th.to(d_t.dtype), high_th.to(d_t.dtype)

    @staticmethod
    def _masked_best_score(scores: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        masked_scores = scores.masked_fill(~valid_mask, -float("inf"))
        best_scores, best_indices = masked_scores.max(dim=-1)
        best_indices = torch.where(
            torch.isfinite(best_scores),
            best_indices,
            best_indices.new_full(best_indices.shape, -1),
        )
        return best_scores, best_indices

    def _build_level3_switch_mask(
        self,
        *,
        candidate_scores: Tensor,
        invalid_mask: Tensor,
        visited_mask: Tensor,
        local_frontier_mask: Tensor,
        history_backtrack_values: Optional[Tensor],
        history_valid_mask: Optional[Tensor],
    ) -> Tensor:
        switch_mask = torch.zeros(candidate_scores.size(0), device=candidate_scores.device, dtype=torch.bool)

        for batch_idx in range(candidate_scores.size(0)):
            valid_nodes = ~invalid_mask[batch_idx]
            non_stop_valid = valid_nodes.clone()
            if 0 <= self.stop_idx < non_stop_valid.numel():
                non_stop_valid[self.stop_idx] = False

            local_forward_mask = non_stop_valid & local_frontier_mask[batch_idx] & (~visited_mask[batch_idx])
            local_best = candidate_scores[batch_idx].masked_fill(~local_forward_mask, -float("inf")).max()

            recovery_mask = non_stop_valid & visited_mask[batch_idx]
            if history_backtrack_values is not None:
                recovery_scores = history_backtrack_values[batch_idx].clone()
            else:
                recovery_scores = candidate_scores[batch_idx].clone()
            if history_valid_mask is not None:
                recovery_scores = recovery_scores.masked_fill(~history_valid_mask[batch_idx], -float("inf"))
            recovery_scores = recovery_scores.masked_fill(~recovery_mask, -float("inf"))
            recovery_best = recovery_scores.max()

            has_recovery = bool(torch.isfinite(recovery_best).item())
            has_local_forward = bool(torch.isfinite(local_best).item())
            if not has_recovery:
                switch_mask[batch_idx] = False
                continue
            if not has_local_forward:
                switch_mask[batch_idx] = True
                continue
            switch_mask[batch_idx] = bool(
                (recovery_best > (local_best + self.level3_switch_margin)).item()
            )

        return switch_mask

    def _select_level3_actions(
        self,
        *,
        candidate_scores: Tensor,
        invalid_mask: Tensor,
        visited_mask: Tensor,
        local_frontier_mask: Tensor,
        history_backtrack_values: Optional[Tensor],
        history_valid_mask: Optional[Tensor],
        level3_mask: Tensor,
        level3_policy: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        high_scores = torch.full_like(candidate_scores, -float("inf"))
        selected_idx = torch.full(
            (candidate_scores.size(0),),
            -1,
            device=candidate_scores.device,
            dtype=torch.long,
        )
        decision_code = torch.zeros_like(selected_idx)

        for batch_idx in range(candidate_scores.size(0)):
            if not bool(level3_mask[batch_idx].item()):
                continue

            valid_nodes = ~invalid_mask[batch_idx]
            valid_visited = valid_nodes & visited_mask[batch_idx]
            stop_valid = self.stop_idx < valid_nodes.numel() and bool(valid_nodes[self.stop_idx].item())
            non_stop_valid = valid_nodes.clone()
            if 0 <= self.stop_idx < non_stop_valid.numel():
                non_stop_valid[self.stop_idx] = False

            backtrack_scores = None
            if valid_visited.any():
                if history_backtrack_values is not None:
                    backtrack_scores = history_backtrack_values[batch_idx].clone()
                else:
                    backtrack_scores = candidate_scores[batch_idx].clone()
                if history_valid_mask is not None:
                    backtrack_scores = backtrack_scores.masked_fill(
                        ~history_valid_mask[batch_idx], -float("inf")
                    )
                backtrack_scores = backtrack_scores.masked_fill(~valid_visited, -float("inf"))
            has_backtrack_candidate = bool(
                backtrack_scores is not None and torch.isfinite(backtrack_scores).any().item()
            )

            stop_allowed = False
            if self.level3_allow_stop and stop_valid:
                best_non_stop = candidate_scores[batch_idx].masked_fill(~non_stop_valid, -float("inf")).max()
                stop_score = candidate_scores[batch_idx, self.stop_idx]
                stop_allowed = bool(
                    torch.isfinite(stop_score).item()
                    and (
                        (not torch.isfinite(best_non_stop).item())
                        or (stop_score >= (best_non_stop + self.level3_stop_margin)).item()
                    )
                )

            if level3_policy == "energy_recovery":
                if has_backtrack_candidate:
                    chosen_idx = int(backtrack_scores.argmax().item())
                    decision = 1
                elif stop_allowed:
                    chosen_idx = self.stop_idx
                    decision = 2
                else:
                    chosen_idx = -1
                    decision = 0
            elif level3_policy == "stop_first":
                if stop_allowed:
                    chosen_idx = self.stop_idx
                    decision = 2
                elif has_backtrack_candidate:
                    chosen_idx = int(backtrack_scores.argmax().item())
                    decision = 1
                else:
                    chosen_idx = self.stop_idx if self.stop_idx < candidate_scores.size(1) else 0
                    decision = 4
            else:
                if has_backtrack_candidate:
                    chosen_idx = int(backtrack_scores.argmax().item())
                    decision = 1
                elif stop_allowed:
                    chosen_idx = self.stop_idx
                    decision = 2
                else:
                    chosen_idx = self.stop_idx if self.stop_idx < candidate_scores.size(1) else 0
                    decision = 4

            selected_idx[batch_idx] = int(chosen_idx)
            decision_code[batch_idx] = int(decision)
            if chosen_idx >= 0:
                boosted_score = candidate_scores[batch_idx, chosen_idx]
                if decision == 1:
                    boosted_score = boosted_score + self.high_level_backtrack_bias
                high_scores[batch_idx, chosen_idx] = boosted_score
            else:
                # No explicit recovery override was justified; fall back to
                # the best frontier-expanded candidate instead of inventing a stop.
                forward_mask = non_stop_valid & local_frontier_mask[batch_idx] & (~visited_mask[batch_idx])
                forward_scores = candidate_scores[batch_idx].masked_fill(~forward_mask, -float("inf"))
                if torch.isfinite(forward_scores).any():
                    best_forward_idx = int(forward_scores.argmax().item())
                    high_scores[batch_idx, best_forward_idx] = candidate_scores[batch_idx, best_forward_idx]
                    selected_idx[batch_idx] = best_forward_idx
                    decision_code[batch_idx] = 5

        return high_scores, selected_idx, decision_code

    def forward(
        self,
        *,
        etp_candidate_scores: Tensor,
        cand_health_scores: Tensor,
        curve_health_scores: Tensor,
        attn_health_scores: Tensor,
        d_t: Tensor,
        invalid_mask: Optional[Tensor],
        visited_mask: Optional[Tensor],
        frontier_mask: Optional[Tensor] = None,
        local_frontier_mask: Optional[Tensor] = None,
        history_backtrack_values: Optional[Tensor] = None,
        history_valid_mask: Optional[Tensor] = None,
        decision_mode: str = "relational_self_model",
        beta_d: float = 1.0,
        lambda_health: Optional[float] = None,
        adapter_warmup: float = 1.0,
        enable_level2_frontier: Optional[bool] = None,
        enable_level3_macro: Optional[bool] = None,
        level3_policy: Optional[str] = None,
        current_step: Optional[Union[int, Tensor]] = None,
    ) -> dict[str, Tensor]:
        if self.training:
            self._update_thresholds(d_t)

        low_th, high_th = self._resolve_thresholds(d_t)

        score_before = etp_candidate_scores
        weight_health = float(self.lambda_health if lambda_health is None else lambda_health)
        health_bonus = float(adapter_warmup) * weight_health * cand_health_scores

        d_t_norm = d_t / high_th.clamp_min(1e-6)
        dt_temperature = torch.exp(-float(beta_d) * d_t_norm.clamp_min(0.0)).view(-1, 1)
        base_rescored = score_before + dt_temperature * health_bonus

        if invalid_mask is None:
            invalid_mask = torch.zeros_like(score_before, dtype=torch.bool)
        if visited_mask is None:
            visited_mask = torch.zeros_like(score_before, dtype=torch.bool)
        if frontier_mask is None:
            frontier_mask = torch.zeros_like(score_before, dtype=torch.bool)
        else:
            frontier_mask = frontier_mask.bool() & invalid_mask.logical_not()
        if local_frontier_mask is None:
            local_frontier_mask = frontier_mask.clone()
        else:
            local_frontier_mask = local_frontier_mask.bool() & frontier_mask
        has_any_frontier = frontier_mask.any(dim=-1, keepdim=True)
        has_local_frontier = local_frontier_mask.any(dim=-1, keepdim=True)
        local_frontier_mask = torch.where(
            has_any_frontier & has_local_frontier,
            local_frontier_mask,
            frontier_mask,
        )

        remote_frontier_mask = frontier_mask & local_frontier_mask.logical_not()
        strict_mask = invalid_mask | visited_mask | remote_frontier_mask
        allow_level2 = self.enable_level2_frontier if enable_level2_frontier is None else bool(enable_level2_frontier)
        allow_level3 = self.enable_level3_macro if enable_level3_macro is None else bool(enable_level3_macro)
        resolved_level3_policy = str(self.level3_policy if level3_policy is None else level3_policy).strip().lower()
        if resolved_level3_policy not in {"energy_recovery", "backtrack_first", "stop_first"}:
            resolved_level3_policy = "energy_recovery"

        expanded_mask = (invalid_mask | visited_mask) if allow_level2 else strict_mask

        strict_scores = base_rescored.masked_fill(strict_mask, -float("inf"))
        expanded_scores = base_rescored.masked_fill(expanded_mask, -float("inf"))

        intervention_level = torch.zeros(score_before.size(0), device=score_before.device, dtype=torch.long)
        use_level2 = d_t >= low_th
        use_level3 = d_t >= high_th
        if current_step is None:
            current_step_tensor = d_t.new_zeros(d_t.shape)
        elif torch.is_tensor(current_step):
            current_step_tensor = current_step.to(device=d_t.device, dtype=d_t.dtype)
        else:
            current_step_tensor = d_t.new_full(d_t.shape, float(current_step))
        use_level3 = use_level3 & (current_step_tensor >= float(self.level3_min_step))
        use_level3 = use_level3 & (d_t_norm >= float(self.level3_min_dnorm))
        if not allow_level2:
            use_level2 = torch.zeros_like(use_level2)
        if not allow_level3:
            use_level3 = torch.zeros_like(use_level3)
        if use_level3.any():
            switch_mask = self._build_level3_switch_mask(
                candidate_scores=base_rescored,
                invalid_mask=invalid_mask,
                visited_mask=visited_mask,
                local_frontier_mask=local_frontier_mask,
                history_backtrack_values=history_backtrack_values,
                history_valid_mask=history_valid_mask,
            )
            use_level3 = use_level3 & switch_mask
        intervention_level = torch.where(use_level2, torch.ones_like(intervention_level), intervention_level)
        intervention_level = torch.where(use_level3, torch.full_like(intervention_level, 2), intervention_level)

        rescored = strict_scores.clone()
        rescored = torch.where(use_level2.unsqueeze(-1), expanded_scores, rescored)

        level3_selected_idx = torch.full_like(intervention_level, -1)
        level3_decision_code = torch.zeros_like(intervention_level)
        if use_level3.any():
            high_scores, level3_selected_idx, level3_decision_code = self._select_level3_actions(
                candidate_scores=base_rescored,
                invalid_mask=invalid_mask,
                visited_mask=visited_mask,
                local_frontier_mask=local_frontier_mask,
                history_backtrack_values=history_backtrack_values,
                history_valid_mask=history_valid_mask,
                level3_mask=use_level3,
                level3_policy=resolved_level3_policy,
            )
            rescored = torch.where(use_level3.unsqueeze(-1), high_scores, rescored)

        if str(decision_mode).strip().lower() == "baseline":
            rescored = score_before.masked_fill(strict_mask, -float("inf"))
            intervention_level = torch.zeros_like(intervention_level)
            level3_selected_idx = torch.full_like(intervention_level, -1)
            level3_decision_code = torch.zeros_like(intervention_level)
            dt_temperature = torch.ones_like(dt_temperature)
            d_t_norm = torch.zeros_like(d_t_norm)
            health_bonus = torch.zeros_like(health_bonus)

        # Keep acting logits numerically sane even if a masking policy removes
        # every candidate in a row. Preference order:
        #   1) original frontier-safe fallback (invalid masked only)
        #   2) raw planner scores as a last resort
        fallback_scores = score_before.masked_fill(invalid_mask, -float("inf"))
        no_finite = ~torch.isfinite(rescored).any(dim=-1)
        if no_finite.any():
            rescored = torch.where(no_finite.unsqueeze(-1), fallback_scores, rescored)
        no_finite = ~torch.isfinite(rescored).any(dim=-1)
        if no_finite.any():
            rescored = torch.where(no_finite.unsqueeze(-1), score_before, rescored)

        score_before_mean = self._masked_candidate_mean(score_before, strict_mask)
        score_after_mean = self._masked_candidate_mean(rescored, expanded_mask)
        health_bonus_mean = self._masked_candidate_mean(health_bonus, expanded_mask)
        cand_health_mean = self._masked_candidate_mean(cand_health_scores, expanded_mask)
        curve_health_mean = self._masked_candidate_mean(curve_health_scores, expanded_mask)
        attn_health_mean = self._masked_candidate_mean(attn_health_scores, expanded_mask)

        return {
            "rescored_candidate_scores": rescored,
            "health_bonus": health_bonus,
            "health_bonus_mean": health_bonus_mean,
            "curve_health_mean": curve_health_mean,
            "attn_health_mean": attn_health_mean,
            "cand_health_mean": cand_health_mean,
            "score_before_mean": score_before_mean,
            "score_after_mean": score_after_mean,
            "score_before_after": torch.stack([score_before_mean, score_after_mean], dim=-1),
            "dt_temperature": dt_temperature.squeeze(-1),
            "d_t_norm": d_t_norm,
            "intervention_level": intervention_level,
            "dt_threshold_low": d_t.new_full(d_t.shape, float(low_th.item())),
            "dt_threshold_high": d_t.new_full(d_t.shape, float(high_th.item())),
            "strict_candidate_mask": strict_mask,
            "expanded_candidate_mask": expanded_mask,
            "frontier_candidate_mask": frontier_mask,
            "local_frontier_candidate_mask": local_frontier_mask,
            "level3_selected_idx": level3_selected_idx,
            "level3_decision_code": level3_decision_code,
            "level3_policy_code": intervention_level.new_full(
                intervention_level.shape,
                0 if resolved_level3_policy == "energy_recovery" else (1 if resolved_level3_policy == "backtrack_first" else 2),
            ),
        }
