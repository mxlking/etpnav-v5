# V5 Gap And Roadmap

`version`: `V5 规范 v1`
`updated_at`: `2026-04-06`

文档角色：唯一 gap / roadmap 文档

本文件只记录当前仍未关闭的问题。每个问题都必须属于以下三类之一：

- `research gap`
- `architecture gap`
- `engineering blocker`

每项都必须包含：

- `priority`
- `why it matters`
- `evidence gap`
- `evidence artifact`
- `owner module`
- `close condition`

本文件不重复主方法定义，不记录纯愿景叙事。

## 1. Research Gap

### RG-01 语义推进仍停留在 attention-shape 层

- `priority`: `P0`
- `why it matters`: `V5` 的上位蓝图要求语义分支刻画 instruction progression，而不仅是 attention 分布形状；否则“semantic health”不足以支撑更强的方法主张。
- `evidence gap`: 当前 `AttentionHead + AttnHealthScorer` 主要依赖 `attn_post` 偏置和 7 维统计量，没有 instruction dependency graph、phase transition target、或证明其超越统计 attention 特征的独立证据。
- `evidence artifact`: `docs/v5/V5_CODE_CURRENT_CN.md` 中 `semantic progression` 仍标记为 `partial`，且当前实现映射只对应 `AttentionHead + AttnHealthScorer`
- `owner module`: `vlnce_baselines/models/statenav_v5/attention_head.py`, `vlnce_baselines/models/statenav_v5/attn_health_scorer.py`, `vlnce_baselines/trainers/train_statenav_v5_stage1.py`
- `close condition`: 语义分支显式消费 instruction 结构或阶段目标，并通过 ablation 证明其优于 stats-only 版本；届时 `semantic progression` 才能从 `partial` 升为 `implemented`。

### RG-02 宏观干预仍偏 heuristic control

- `priority`: `P0`
- `why it matters`: `intervention_policy` 是 `V5` 闭环主线的一部分；若 Level 2 / 3 主要依赖固定阈值、backtrack bias 和手工 fallback，它更像强工程控制器，而不是可辩护的 self-correction policy。
- `evidence gap`: 当前缺少明确的 risk / regret / hazard 对象，也缺少“为何在该时刻接管”的学习式或理论式依据。
- `evidence artifact`: `docs/v5/V5_CODE_CURRENT_CN.md` 中 `intervention policy` 仍标记为 `partial`，说明当前实现有效但更偏 heuristic controller
- `owner module`: `vlnce_baselines/models/statenav_v5/macro_intervention_manager.py`
- `close condition`: Level 2 / 3 决策由显式风险输入和可解释决策目标驱动，并有独立 ablation 证明优于现有 heuristic policy。

### RG-03 `V5` 的通用性还未超出 ETP 专属挂接

- `priority`: `P1`
- `why it matters`: 如果目标是更高阶的方法贡献，`V5` 需要从 ETP 专属外挂升级为可挂接不同 embodied planner 的 self-model layer。
- `evidence gap`: 当前只有 ETP 主干上的 R2R / RxR 路线，没有第二个 planner 或统一后端接口的证据。
- `evidence artifact`: `docs/v5/V5_CODE_CURRENT_CN.md` 当前入口和主线模块仍全部绑定在 `ETP` 主干与 `StateNavAgentV5` 路径上
- `owner module`: `vlnce_baselines/agents/statenav_agent_v5.py`, `vlnce_baselines/trainers/train_statenav_v5_stage1.py`, `vlnce_baselines/trainers/train_statenav_v5_stage2.py`
- `close condition`: 抽出 planner-facing 接口，并在至少一个非 ETP 后端或明确的接口模拟层上完成接入和 smoke evaluation。

## 2. Architecture Gap

### AG-01 trainer 主逻辑仍过度集中在长 `rollout()`

- `priority`: `P0`
- `why it matters`: 当前 `rollout()` 混合了图更新、label 构造、loss、intervention、trace 和 logging，架构逻辑被实现细节掩盖，调试和重构成本都偏高。
- `evidence gap`: 难以从 trainer 结构直接对应 `belief update -> relation state -> rollout -> readout -> intervention` 这条 blueprint 主线。
- `evidence artifact`: `vlnce_baselines/trainers/train_statenav_v5_stage1.py` 与 `vlnce_baselines/trainers/train_statenav_v5_stage2.py` 中的长 `rollout()` 主流程
- `owner module`: `vlnce_baselines/trainers/train_statenav_v5_stage1.py`, `vlnce_baselines/trainers/train_statenav_v5_stage2.py`
- `close condition`: trainer 被拆成与 blueprint 对齐的命名阶段函数，并保留等价行为验证。

### AG-02 核心对象还不是 first-class interface

- `priority`: `P1`
- `why it matters`: `m_t`, `c_t`, `s_t`, `D_t` 目前散落在多个张量和局部变量中，难以做跨模块检查、trace 和 planner-agnostic 封装。
- `evidence gap`: 当前没有显式 step-level container 或统一接口承载这些对象。
- `evidence artifact`: `docs/v5/V5_CODE_CURRENT_CN.md` 的 blueprint 映射中，`material_state m_t` 与 `conscious_state c_t` 仍标记为 `partial` 且说明为分散表达
- `owner module`: `vlnce_baselines/agents/statenav_agent_v5.py`, `vlnce_baselines/trainers/train_statenav_v5_stage1.py`
- `close condition`: 主线中间对象被收口到显式结构或稳定接口，并能被 trace / logging / intervention 复用。

### AG-03 主线、消融、诊断路径仍有混淆风险

- `priority`: `P1`
- `why it matters`: 若 `extra_tokens`、`h_post-only`、`eval_action_source`、`no_level3` 等路径不被清晰标注，外部读者很容易把它们误读成主方法组成部分。
- `evidence gap`: 当前代码与配置同时承载主线、ablation、diagnostic；没有统一的命名和文档制度来区分。
- `evidence artifact`: `docs/v5/V5_CODE_CURRENT_CN.md` 的“当前非主线项”节已列出这些路径，但代码与配置层尚未全面加标签
- `owner module`: `docs/v5/README_CN.md`, `vlnce_baselines/config/default.py`, `run_r2r/v5/*.yaml`, `run_rxr/v5/*.yaml`
- `close condition`: 所有配置和接口都有 `mainline / ablation / diagnostic / archive` 标签，主线定义不再引用这些辅助路径。

## 3. Engineering Blocker

### EB-01 `statenav` 评测仍可能出现一步停

- `priority`: `P0`
- `why it matters`: 这会直接污染对 checkpoint 和方法本身的判断，导致无法区分“推理异常”与“训练失效”。
- `evidence gap`: 现有三路矩阵已经证明问题集中在旧的 `statenav + Level-3` 路径。当前代码已将 Level-3 升级为 `energy_recovery` 形式，但这条新策略还没有经过 fresh 训练与复评闭环验证，因此仍不能宣称该 gap 已关闭。
- `evidence artifact`: `2026-04-06 diag_eval_r2r_s1` 三路结果显示旧 `statenav` 仅 `success=0.1506`, `one_step_stop_ratio=0.9500`, `l3_step_rate=0.8696`，而 `etp` 为 `0.5824`、`no_level3` 为 `0.5500`；当前代码已把 `level3_policy` 主线默认升级为 `energy_recovery`
- `owner module`: `vlnce_baselines/models/statenav_v5/macro_intervention_manager.py`, `vlnce_baselines/trainers/train_statenav_v5_stage1.py`
- `close condition`: 在同一 checkpoint 上完成标准诊断矩阵，`statenav + Level-3` 路径不再系统性地一步停，且关键指标不再显著差于 `etp` 与 `no_level3`。

### EB-02 诊断矩阵还没有成为默认复评纪律

- `priority`: `P0`
- `why it matters`: 当前要定位问题时，仍需要临时组合 `eval_action_source`、`Level 3` 开关和 checkpoint；没有标准矩阵就很难快速归因。
- `evidence gap`: 当前已经补上 `statenav / etp / no_level3` 三路诊断入口、统一汇总脚本，并已在 `diag_eval_r2r_s1` 上完成一轮 fresh 验证；但它还没有沉淀成所有 checkpoint 的默认复评纪律。
- `evidence artifact`: `run_r2r/v5/statenav_v5_eval_matrix.bash`、`run_rxr/v5/statenav_v5_eval_matrix.bash`、`scripts/v5_eval_summary.py`、`data/logs/eval_results/diag_eval_r2r_s1_{statenav,etp,no_level3}`
- `owner module`: `vlnce_baselines/config/default.py`, `run_r2r/v5/statenav_v5_stage1.yaml`, `vlnce_baselines/trainers/train_statenav_v5_stage1.py`
- `close condition`: 固化一个最小评测矩阵和结果模板，任何可疑 checkpoint 都能按统一流程复评。

### EB-03 环境侧开销压过模型侧开销

- `priority`: `P1`
- `why it matters`: 当前训练和调试的主要时间消耗来自 `env / crpc`，会显著拉长故障定位和实验迭代周期。
- `evidence gap`: 现有 `step_metrics.tsv` 中多次出现 `crpc` 与 `env` 远高于 `nav / pano / st / opt` 的情况，但缺少统一 profiling 结论和固定运行建议。
- `evidence artifact`: `data/logs/checkpoints/exp_r2r_v5_s1_4k/step_metrics.tsv` 中多步记录显示 `crpc` 与 `env` 明显高于模型侧阶段
- `owner module`: 训练 launcher、环境配置和 Habitat 运行参数
- `close condition`: 给出标准 profiling 结论与默认运行配置，或将主要实验路径的 `env / crpc` 开销压到可接受阈值。

## 4. Close Discipline

任何 gap 关闭前都必须满足以下条件：

1. 有明确证据，而不只是“代码看起来已经改了”。
2. 有对应模块 owner。
3. 有可复现的验证路径。
4. 能在 `V5_FINAL_BLUEPRINT_CN.md` 和 `V5_CODE_CURRENT_CN.md` 中同步更新状态。
