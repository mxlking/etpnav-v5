# V5 代码索引

这份索引只做一件事：把当前仓库里和 `StateNav V5` 直接相关的代码按职责整理出来，方便单独阅读、迁移或打包。

说明：
- 这份文件偏 `V5-only` 主线。
- 如果你要的是“`V5` 自己的代码 + 它真正依赖的 `ETPNav` 骨架”，请直接看：
  - `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/V5_ETP_CODE_INDEX_CN.md`

## 一、主入口

- Agent:
  - `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/agents/statenav_agent_v5.py`
- Trainer:
  - `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/trainers/train_statenav_v5_stage1.py`
  - `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/trainers/train_statenav_v5_stage2.py`

## 二、V5 核心模型

目录：
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/models/statenav_v5`

主线文件：
- `relational_state_extractor.py`
  - Phase 1：`s_t` 提取
- `unified_rollout_trunk.py`
  - Phase 2：候选条件统一推演主干
- `state_head.py`
  - 从统一表示读出 `s_next_k`
- `attention_head.py`
  - 从统一表示读出 `predicted_attn_k`
- `progress_curve_predictor.py`
  - `curve_k` 与 `curve_health_k`
- `attn_health_scorer.py`
  - `attn_health_k`
- `macro_intervention_manager.py`
  - `D_t` 分级干预：Level 1 / 2 / 3
- `losses.py`
  - `L_planner / L_kl / L_curve / L_attn`

## 三、V5 复用的 RSSM 子模块

- `self_rssm_transition.py`
- `self_rssm_correction.py`
- `self_rssm_latent.py`

这些文件负责：
- `h_prior`
- `h_post`
- `z_post`
- `D_t = KL(post || prior)`

## 四、当前代码树状态

当前仍在目录里的兼容保留文件：
- `state_planner_adapter.py`

已经从 `V5` 代码树移除的早期独立模块：
- `attention_transition.py`
- `open_eye_rollout.py`
- `imagination_rollout.py`

说明：
- `attention transition` 已收进 `AttentionHead + attention_transition_loss`
- `open-eye rollout` 已并入 `UnifiedRolloutTrunk`
- `imagination` 属于 `V4.5` 历史分支，不再属于 `v5.md` 主线

## 五、运行配置

R2R：
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_r2r/v5/statenav_v5.bash`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_r2r/v5/statenav_v5_stage1.yaml`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_r2r/v5/statenav_v5_stage2.yaml`

常见消融：
- `statenav_v5_stage1_no_attn.yaml`
- `statenav_v5_stage1_no_level3.yaml`
- `statenav_v5_stage1_no_frontier.yaml`
- `statenav_v5_stage1_no_open_eye.yaml`
- `statenav_v5_stage1_hpost_only.yaml`

RxR：
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_rxr/v5/statenav_v5.bash`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_rxr/v5/statenav_v5_stage1.yaml`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_rxr/v5/statenav_v5_stage2.yaml`

## 六、说明文档

- 方法规范稿：
  - `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/v5.md`
- 当前代码对齐说明：
  - `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/V5_CODE_ALIGNED_IMPL_CN.md`
- 当前剩余缺口：
  - `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/V5_GAP_LIST_CN.md`
- 流程与公式：
  - `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/V5_FLOW_AND_FORMULAS_CN.md`

## 七、最小必读集合

如果你只想最快看懂当前 V5 主线，按这个顺序读：

1. `docs/v5/v5.md`
2. `vlnce_baselines/agents/statenav_agent_v5.py`
3. `vlnce_baselines/models/statenav_v5/relational_state_extractor.py`
4. `vlnce_baselines/models/statenav_v5/unified_rollout_trunk.py`
5. `vlnce_baselines/models/statenav_v5/macro_intervention_manager.py`
6. `vlnce_baselines/trainers/train_statenav_v5_stage1.py`

## 八、最小迁移集合

如果你只是想把当前 V5 主线单独拎出去，最小需要：

- `vlnce_baselines/agents/statenav_agent_v5.py`
- `vlnce_baselines/models/statenav_v5/relational_state_extractor.py`
- `vlnce_baselines/models/statenav_v5/unified_rollout_trunk.py`
- `vlnce_baselines/models/statenav_v5/state_head.py`
- `vlnce_baselines/models/statenav_v5/attention_head.py`
- `vlnce_baselines/models/statenav_v5/progress_curve_predictor.py`
- `vlnce_baselines/models/statenav_v5/attn_health_scorer.py`
- `vlnce_baselines/models/statenav_v5/macro_intervention_manager.py`
- `vlnce_baselines/models/statenav_v5/losses.py`
- `vlnce_baselines/models/statenav_v5/self_rssm_transition.py`
- `vlnce_baselines/models/statenav_v5/self_rssm_correction.py`
- `vlnce_baselines/models/statenav_v5/self_rssm_latent.py`
- `vlnce_baselines/trainers/train_statenav_v5_stage1.py`
- `vlnce_baselines/trainers/train_statenav_v5_stage2.py`
