# V5 + ETP 代码索引

这份索引是给“我要拿到一份真正能解释 `V5` 为什么能跑起来的源码集合”准备的。

它不是只列 `StateNav V5` 自己的文件，而是把当前 `V5` 训练/推理真正依赖的两层代码都列出来：

- `V5 mainline`
- `ETPNav backbone`

这样你看到的就不只是外壳，而是：

`V5 决策层 + ETP 感知/图规划底盘 + 训练/环境公共支撑`

## 一、V5 Mainline

### 1. Agent / Trainer

- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/agents/statenav_agent_v5.py`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/trainers/train_statenav_v5_stage1.py`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/trainers/train_statenav_v5_stage2.py`

### 2. V5 核心模型目录

目录：
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/models/statenav_v5`

主线文件：
- `relational_state_extractor.py`
- `unified_rollout_trunk.py`
- `state_head.py`
- `attention_head.py`
- `progress_curve_predictor.py`
- `attn_health_scorer.py`
- `macro_intervention_manager.py`
- `losses.py`
- `self_rssm_transition.py`
- `self_rssm_correction.py`
- `self_rssm_latent.py`

兼容/过渡保留：
- `state_planner_adapter.py`

说明：
- 早期独立文件 `attention_transition.py / open_eye_rollout.py / imagination_rollout.py`
  已从当前 `V5` 代码树移除。
- 它们的语义现在分别并入：
  - `AttentionHead + attention_transition_loss`
  - `UnifiedRolloutTrunk`
  - `V4.5` 历史分支

## 二、ETPNav Backbone

这部分是 `V5` 真正站在上面的底盘。没有它们，`V5` 只有“关系状态 + rerank 逻辑”，但没有：
- waypoint proposal
- panorama encoding
- global map embedding
- ETP base planner

### 1. ETP 训练器与主入口

- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/ss_trainer_ETP.py`

### 2. ETP 核心策略 / 图规划

- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/models/Policy_ViewSelection_ETP.py`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/models/policy.py`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/models/graph_utils.py`

### 3. Waypoint Predictor

目录：
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/waypoint_pred`

重点文件：
- `TRM_net.py`
- `utils.py`
- `transformer/waypoint_bert.py`
- `transformer/pytorch_transformer/modeling_bert.py`
- `transformer/pytorch_transformer/modeling_utils.py`
- `transformer/pytorch_transformer/file_utils.py`

## 三、公共训练 / 环境支撑

这些文件不是 `V5` 方法本身，但 `V5` 训练运行时会直接依赖它们。

目录：
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/common`

重点文件：
- `base_il_trainer.py`
- `env_utils.py`
- `environments.py`
- `ops.py`
- `utils.py`
- `transformer.py`
- `aux_losses.py`
- `torch_compat.py`

说明：
- 这里我把整层 `common/` 都算进完整源码集合了。
- 原因是 `V5` 实际运行时，环境构造、tensor ops、公共 transformer 均会碰到这些文件。

## 四、配置 / 数据 / 初始化

### 1. 初始化与工具

- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/__init__.py`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/models/__init__.py`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/utils.py`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/navmorph_compat.py`

### 2. 配置

- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/config/default.py`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/config/shim_config.py`

### 3. 数据标签辅助

- `/home/D/liumeng/vln/260316P/ETPNav_selected/vlnce_baselines/datasets/state_label_builder.py`

## 五、运行入口

### R2R

- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_r2r/v5/statenav_v5.bash`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_r2r/v5/statenav_v5_stage1.yaml`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_r2r/v5/statenav_v5_stage2.yaml`

常见消融：
- `statenav_v5_stage1_no_attn.yaml`
- `statenav_v5_stage1_no_level3.yaml`
- `statenav_v5_stage1_no_frontier.yaml`
- `statenav_v5_stage1_no_open_eye.yaml`
- `statenav_v5_stage1_hpost_only.yaml`

### RxR

- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_rxr/v5/statenav_v5.bash`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_rxr/v5/statenav_v5_stage1.yaml`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/run_rxr/v5/statenav_v5_stage2.yaml`

## 六、文档

- `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/v5.md`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/V5_CODE_ALIGNED_IMPL_CN.md`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/V5_GAP_LIST_CN.md`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/V5_FLOW_AND_FORMULAS_CN.md`
- `/home/D/liumeng/vln/260316P/ETPNav_selected/docs/v5/V5_CODE_INDEX_CN.md`

## 七、推荐阅读顺序

如果你是想从“方法 -> 代码 -> 底盘”一路看清楚，建议按这个顺序读：

1. `docs/v5/v5.md`
2. `vlnce_baselines/agents/statenav_agent_v5.py`
3. `vlnce_baselines/models/statenav_v5/`
4. `vlnce_baselines/trainers/train_statenav_v5_stage1.py`
5. `vlnce_baselines/ss_trainer_ETP.py`
6. `vlnce_baselines/models/Policy_ViewSelection_ETP.py`
7. `vlnce_baselines/models/graph_utils.py`
8. `vlnce_baselines/waypoint_pred/`
9. `vlnce_baselines/common/`

## 八、导出包说明

当前我保留两种源码包：

- `V5-only`
  - 只包含 `V5` 主线自身
- `V5 + ETP`
  - 包含 `V5` 主线 + `ETPNav backbone` + 公共训练/环境支撑

如果你的目的是：
- 单独看 `V5` 方法结构
  - 用 `V5-only`
- 真正把当前系统源码带走、复盘、迁移
  - 用 `V5 + ETP`
