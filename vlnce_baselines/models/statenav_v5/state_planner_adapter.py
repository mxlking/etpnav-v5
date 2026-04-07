import torch
from torch import Tensor, nn
from typing import Optional


class StatePlannerAdapter(nn.Module):
    """V4 anticipatory planner adapter.

    V4 keeps the external planner backbone untouched and only injects
    anticipatory self-model signals:
    - per-candidate future health scores
    - D_t-driven trust modulation
    """

    def __init__(
        self,
        lambda_health: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_health = float(lambda_health)

    @staticmethod
    def _masked_candidate_mean(scores: Tensor, candidate_mask: Optional[Tensor]) -> Tensor:
        if candidate_mask is None:
            return scores.mean(dim=-1)

        valid_mask = candidate_mask.logical_not()
        valid_count = valid_mask.sum(dim=-1).clamp_min(1)
        valid_scores = scores.masked_fill(candidate_mask.bool(), 0.0)
        return valid_scores.sum(dim=-1) / valid_count.to(scores.dtype)

    def forward(
        self,
        etp_candidate_scores: Tensor,
        cand_health_scores: Optional[Tensor] = None,
        d_t: Optional[Tensor] = None,
        candidate_mask: Optional[Tensor] = None,
        decision_mode: str = "anticipatory_self_model",
        beta_d: float = 1.0,
        lambda_health: Optional[float] = None,
        adapter_warmup: float = 1.0,
    ) -> dict[str, Tensor]:
        mode = str(decision_mode).strip().lower()
        if mode not in {"baseline", "anticipatory_self_model"}:
            raise ValueError(f"Unsupported decision_mode: {decision_mode}")

        score_before = etp_candidate_scores
        weight_health = float(self.lambda_health if lambda_health is None else lambda_health)

        if cand_health_scores is None:
            cand_health_scores = torch.zeros_like(score_before)
        if d_t is None:
            d_t = torch.zeros(score_before.size(0), device=score_before.device, dtype=score_before.dtype)

        health_bonus = float(adapter_warmup) * weight_health * cand_health_scores
        dt_temperature = torch.exp(-float(beta_d) * d_t.clamp(min=0.0)).view(-1, 1)

        if mode == "baseline":
            health_bonus = torch.zeros_like(score_before)
            dt_temperature = torch.ones_like(dt_temperature)
            rescored_candidate_scores = score_before
        else:
            # D_t is a step-level mismatch signal. It should modulate how strongly
            # health-based anticipatory reranking can deviate from the base planner,
            # rather than be subtracted as a candidate-invariant constant.
            rescored_candidate_scores = score_before + dt_temperature * health_bonus

        if candidate_mask is not None:
            rescored_candidate_scores = rescored_candidate_scores.masked_fill(
                candidate_mask.bool(), -float("inf")
            )

        score_before_mean = self._masked_candidate_mean(score_before, candidate_mask)
        score_after_mean = self._masked_candidate_mean(rescored_candidate_scores, candidate_mask)
        health_bonus_mean = self._masked_candidate_mean(health_bonus, candidate_mask)
        cand_health_mean = self._masked_candidate_mean(cand_health_scores, candidate_mask)

        return {
            "rescored_candidate_scores": rescored_candidate_scores,
            "cand_health_scores": cand_health_scores,
            "health_bonus": health_bonus,
            "dt_temperature": dt_temperature.squeeze(-1),
            "health_bonus_mean": health_bonus_mean,
            "cand_health_mean": cand_health_mean,
            "score_before_mean": score_before_mean,
            "score_after_mean": score_after_mean,
            "score_before_after": torch.stack([score_before_mean, score_after_mean], dim=-1),
        }
