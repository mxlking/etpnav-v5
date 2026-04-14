# EFES Version Map

This document is the canonical version map for the EFES-related code paths in this repository.

## Current Mainline

Use this line for new training, evaluation, inference, and the current paper draft.

- Branch: `efes-v3-theory`
- Trainer: `EFESV3Theory`
- Script: `run_r2r/efes_v3_theory/efes_v3_theory.bash`
- Config: `run_r2r/efes_v3_theory/efes_v3_theory_main.yaml`
- Agent: `vlnce_baselines/agents/efes_v3_theory_agent.py`
- Models: `vlnce_baselines/models/efes_v3_theory/`
- Trainer implementation: `vlnce_baselines/trainers/train_efes_v3_theory.py`
- Checkpoint contract: `v3-theory`
- Paper docs:
  - `docs/last/EFES_FINAL_PAPER.md`
  - `docs/last/EFES_final_paper_blueprint.md`
  - `docs/last/EFES_code_upgrade_theorem_mapping_zh.md`

Status:

- This is the only EFES line that should be treated as the current paper-aligned implementation.
- New experiments should be named with the `efes_v3_theory_*` prefix.

## Baseline Line

Use this only for the frozen ETPNav baseline.

- Script: `run_r2r/main.bash`
- Main config: `run_r2r/r2r_vlnce.yaml`
- Purpose: baseline training / baseline evaluation / baseline inference

Status:

- Keep for fair comparison.
- Do not mix it with EFES checkpoint interpretation.

## Historical EFES Lines

These lines are preserved for reference, regression comparison, or reading old experiments. They are not the current mainline.

### EFES (legacy)

- Trainer: `EFES`
- Script: `run_r2r/efes/efes.bash`
- Config: `run_r2r/efes/efes_main.yaml`
- Agent: `vlnce_baselines/agents/efes_agent.py`
- Models: `vlnce_baselines/models/efes/`
- Trainer: `vlnce_baselines/trainers/train_efes.py`

Status:

- Legacy line.
- Do not use for new paper experiments.

### EFESSelf

- Trainer: `EFESSelf`
- Script: `run_r2r/efes_self/efes_self.bash`
- Config: `run_r2r/efes_self/efes_self_main.yaml`
- Agent: `vlnce_baselines/agents/efes_self_agent.py`
- Models: `vlnce_baselines/models/efes_self/`
- Trainer: `vlnce_baselines/trainers/train_efes_self.py`

Status:

- Self-only / diagnosis-oriented line.
- Historical, not current paper mainline.

### EFESV2

- Trainer: `EFESV2`
- Script: `run_r2r/efes_v2/efes_v2.bash`
- Config: `run_r2r/efes_v2/efes_v2_main.yaml`
- Agent: `vlnce_baselines/agents/efes_v2_agent.py`
- Models: `vlnce_baselines/models/efes_v2/`
- Trainer: `vlnce_baselines/trainers/train_efes_v2.py`

Status:

- Historical embodied-self / recovery-routing line.
- Not current mainline.

### EFESV3

- Trainer: `EFESV3`
- Script: `run_r2r/efes_v3/efes_v3.bash`
- Config: `run_r2r/efes_v3/efes_v3_main.yaml`
- Agent: `vlnce_baselines/agents/efes_v3_agent.py`
- Models: `vlnce_baselines/models/efes_v3/`
- Trainer: `vlnce_baselines/trainers/train_efes_v3.py`
- Checkpoint contract: `v3-clean`

Status:

- Transitional pre-theory V3 line.
- Keep for comparison only.
- Do not treat `v3-clean` checkpoints as the final paper implementation.

## Non-EFES StateNav Research Lines

These are separate exploration lines and should not be mixed with EFES paper claims.

- `run_r2r/v5/*`
- `run_r2r/v6/*`
- `vlnce_baselines/models/statenav_v5/`
- `vlnce_baselines/models/statenav_v6/`

Status:

- Separate StateNav research tracks.
- Not part of the current EFES paper mainline.

## Documentation Map

Use only the `docs/last/` set for the current mainline.

### Current docs

- `docs/last/EFES_FINAL_PAPER.md`
- `docs/last/EFES_final_paper_blueprint.md`
- `docs/last/EFES_code_upgrade_theorem_mapping_zh.md`
- `docs/last/EFES_VERSION_MAP.md`

### Historical docs

- `docs/efes/*`
- `docs/efes_v2/*`
- `docs/last/1.md`
- `docs/last/2.md`

Status:

- `docs/last/1.md` and `docs/last/2.md` are historical auxiliary drafts.
- They are not the source of truth for the current implementation.

## Naming Rules

To avoid mixing lines, use these prefixes consistently:

- Current mainline:
  - `efes_v3_theory_*`
- Transitional V3:
  - `efes_v3_*`
- Historical active-safe / self / v2:
  - `efes_active_*`
  - `efes_self_*`
  - `efes_v2_*`
- Baseline:
  - `baseline_etpnav_*`

## Practical Rule

If you are unsure which path to use, the answer should be:

1. Baseline comparison: `run_r2r/main.bash`
2. Current EFES paper train/eval/infer: `run_r2r/efes_v3_theory/efes_v3_theory.bash`
3. Old EFES folders: read-only unless you are explicitly reproducing an old result
