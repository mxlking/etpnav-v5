from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_PATH = ROOT / "vlnce_baselines" / "models" / "statenav_v5" / "macro_intervention_manager.py"
SPEC = importlib.util.spec_from_file_location("statenav_v5_macro_intervention_manager", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
MacroInterventionManager = MODULE.MacroInterventionManager


def _build_manager() -> MacroInterventionManager:
    manager = MacroInterventionManager(
        stop_idx=0,
        enable_level2_frontier=True,
        enable_level3_macro=True,
        level3_policy="backtrack_first",
        level3_allow_stop=True,
    )
    manager.eval()
    manager.dt_low_threshold.fill_(0.1)
    manager.dt_high_threshold.fill_(0.2)
    manager.thresholds_initialized.fill_(True)
    return manager


def test_level3_backtrack_requires_finite_history_scores():
    manager = _build_manager()

    out = manager(
        etp_candidate_scores=torch.tensor([[-8.0, 4.5, 1.0, 0.5]], dtype=torch.float32),
        cand_health_scores=torch.zeros(1, 4),
        curve_health_scores=torch.zeros(1, 4),
        attn_health_scores=torch.zeros(1, 4),
        d_t=torch.tensor([0.3], dtype=torch.float32),
        invalid_mask=torch.tensor([[False, False, False, False]]),
        visited_mask=torch.tensor([[False, True, False, False]]),
        frontier_mask=torch.tensor([[False, False, True, True]]),
        local_frontier_mask=torch.tensor([[False, False, True, True]]),
        history_backtrack_values=torch.full((1, 4), -float("inf")),
        history_valid_mask=torch.zeros(1, 4, dtype=torch.bool),
        current_step=3,
        decision_mode="relational_self_model",
        beta_d=1.0,
        lambda_health=0.0,
        adapter_warmup=1.0,
        enable_level2_frontier=True,
        enable_level3_macro=True,
        level3_policy="backtrack_first",
    )

    assert int(out["intervention_level"][0].item()) == 1
    assert int(out["level3_selected_idx"][0].item()) == -1
    assert int(out["level3_decision_code"][0].item()) == 0


def test_level3_backtrack_uses_history_when_available():
    manager = _build_manager()

    out = manager(
        etp_candidate_scores=torch.tensor([[-8.0, 0.5, 1.0, 0.2]], dtype=torch.float32),
        cand_health_scores=torch.zeros(1, 4),
        curve_health_scores=torch.zeros(1, 4),
        attn_health_scores=torch.zeros(1, 4),
        d_t=torch.tensor([0.3], dtype=torch.float32),
        invalid_mask=torch.tensor([[False, False, False, False]]),
        visited_mask=torch.tensor([[False, True, False, False]]),
        frontier_mask=torch.tensor([[False, False, True, True]]),
        local_frontier_mask=torch.tensor([[False, False, True, True]]),
        history_backtrack_values=torch.tensor([[-float("inf"), 3.0, -float("inf"), -float("inf")]]),
        history_valid_mask=torch.tensor([[False, True, False, False]]),
        current_step=3,
        decision_mode="relational_self_model",
        beta_d=1.0,
        lambda_health=0.0,
        adapter_warmup=1.0,
        enable_level2_frontier=True,
        enable_level3_macro=True,
        level3_policy="backtrack_first",
    )

    assert int(out["intervention_level"][0].item()) == 2
    assert int(out["level3_selected_idx"][0].item()) == 1
    assert int(out["level3_decision_code"][0].item()) == 1


def test_energy_recovery_waits_until_recovery_is_better_than_forward():
    manager = MacroInterventionManager(
        stop_idx=0,
        enable_level2_frontier=True,
        enable_level3_macro=True,
        level3_policy="energy_recovery",
        level3_min_step=3,
        level3_min_dnorm=1.0,
    )
    manager.eval()
    manager.dt_low_threshold.fill_(0.1)
    manager.dt_high_threshold.fill_(0.2)
    manager.thresholds_initialized.fill_(True)

    out = manager(
        etp_candidate_scores=torch.tensor([[-8.0, 1.0, 4.0, 3.5]], dtype=torch.float32),
        cand_health_scores=torch.zeros(1, 4),
        curve_health_scores=torch.zeros(1, 4),
        attn_health_scores=torch.zeros(1, 4),
        d_t=torch.tensor([0.3], dtype=torch.float32),
        invalid_mask=torch.tensor([[False, False, False, False]]),
        visited_mask=torch.tensor([[False, True, False, False]]),
        frontier_mask=torch.tensor([[False, False, True, True]]),
        local_frontier_mask=torch.tensor([[False, False, True, True]]),
        history_backtrack_values=torch.tensor([[-float("inf"), 0.5, -float("inf"), -float("inf")]]),
        history_valid_mask=torch.tensor([[False, True, False, False]]),
        current_step=3,
        decision_mode="relational_self_model",
        beta_d=1.0,
        lambda_health=0.0,
        adapter_warmup=1.0,
        enable_level2_frontier=True,
        enable_level3_macro=True,
        level3_policy="energy_recovery",
    )

    assert int(out["intervention_level"][0].item()) == 1
    assert int(out["level3_selected_idx"][0].item()) == -1
    assert int(out["level3_decision_code"][0].item()) == 0


def test_energy_recovery_backtracks_when_recovery_dominates_forward():
    manager = MacroInterventionManager(
        stop_idx=0,
        enable_level2_frontier=True,
        enable_level3_macro=True,
        level3_policy="energy_recovery",
        level3_min_step=3,
        level3_min_dnorm=1.0,
    )
    manager.eval()
    manager.dt_low_threshold.fill_(0.1)
    manager.dt_high_threshold.fill_(0.2)
    manager.thresholds_initialized.fill_(True)

    out = manager(
        etp_candidate_scores=torch.tensor([[-8.0, 4.5, 1.0, 0.2]], dtype=torch.float32),
        cand_health_scores=torch.zeros(1, 4),
        curve_health_scores=torch.zeros(1, 4),
        attn_health_scores=torch.zeros(1, 4),
        d_t=torch.tensor([0.3], dtype=torch.float32),
        invalid_mask=torch.tensor([[False, False, False, False]]),
        visited_mask=torch.tensor([[False, True, False, False]]),
        frontier_mask=torch.tensor([[False, False, True, True]]),
        local_frontier_mask=torch.tensor([[False, False, True, True]]),
        history_backtrack_values=torch.tensor([[-float("inf"), 5.0, -float("inf"), -float("inf")]]),
        history_valid_mask=torch.tensor([[False, True, False, False]]),
        current_step=3,
        decision_mode="relational_self_model",
        beta_d=1.0,
        lambda_health=0.0,
        adapter_warmup=1.0,
        enable_level2_frontier=True,
        enable_level3_macro=True,
        level3_policy="energy_recovery",
    )

    assert int(out["intervention_level"][0].item()) == 2
    assert int(out["level3_selected_idx"][0].item()) == 1
    assert int(out["level3_decision_code"][0].item()) == 1


def test_energy_recovery_respects_min_step_gate():
    manager = MacroInterventionManager(
        stop_idx=0,
        enable_level2_frontier=True,
        enable_level3_macro=True,
        level3_policy="energy_recovery",
        level3_min_step=3,
        level3_min_dnorm=1.0,
    )
    manager.eval()
    manager.dt_low_threshold.fill_(0.1)
    manager.dt_high_threshold.fill_(0.2)
    manager.thresholds_initialized.fill_(True)

    out = manager(
        etp_candidate_scores=torch.tensor([[-8.0, 4.5, 1.0, 0.2]], dtype=torch.float32),
        cand_health_scores=torch.zeros(1, 4),
        curve_health_scores=torch.zeros(1, 4),
        attn_health_scores=torch.zeros(1, 4),
        d_t=torch.tensor([0.3], dtype=torch.float32),
        invalid_mask=torch.tensor([[False, False, False, False]]),
        visited_mask=torch.tensor([[False, True, False, False]]),
        frontier_mask=torch.tensor([[False, False, True, True]]),
        local_frontier_mask=torch.tensor([[False, False, True, True]]),
        history_backtrack_values=torch.tensor([[-float("inf"), 5.0, -float("inf"), -float("inf")]]),
        history_valid_mask=torch.tensor([[False, True, False, False]]),
        current_step=1,
        decision_mode="relational_self_model",
        beta_d=1.0,
        lambda_health=0.0,
        adapter_warmup=1.0,
        enable_level2_frontier=True,
        enable_level3_macro=True,
        level3_policy="energy_recovery",
    )

    assert int(out["intervention_level"][0].item()) == 1
    assert int(out["level3_selected_idx"][0].item()) == -1
    assert int(out["level3_decision_code"][0].item()) == 0
