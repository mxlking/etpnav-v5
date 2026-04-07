# V5 Code Current

`version`: `V5 规范 v1`
`updated_at`: `2026-04-06`

文档角色：当前代码唯一事实来源

本文件只记录当前仓库中 `V5` 主线实际在做什么。它不承担愿景表达，不复述历史白皮书，也不写未来承诺。

## 1. 当前入口

当前 `V5` 的主要入口在以下文件：

- [`../../vlnce_baselines/agents/statenav_agent_v5.py`](../../vlnce_baselines/agents/statenav_agent_v5.py)
- [`../../vlnce_baselines/trainers/train_statenav_v5_stage1.py`](../../vlnce_baselines/trainers/train_statenav_v5_stage1.py)
- [`../../vlnce_baselines/trainers/train_statenav_v5_stage2.py`](../../vlnce_baselines/trainers/train_statenav_v5_stage2.py)
- [`../../vlnce_baselines/models/statenav_v5/relational_state_extractor.py`](../../vlnce_baselines/models/statenav_v5/relational_state_extractor.py)
- [`../../vlnce_baselines/models/statenav_v5/unified_rollout_trunk.py`](../../vlnce_baselines/models/statenav_v5/unified_rollout_trunk.py)
- [`../../vlnce_baselines/models/statenav_v5/macro_intervention_manager.py`](../../vlnce_baselines/models/statenav_v5/macro_intervention_manager.py)
- [`../../vlnce_baselines/config/default.py`](../../vlnce_baselines/config/default.py)
- [`../../run_r2r/v5/statenav_v5_stage1.yaml`](../../run_r2r/v5/statenav_v5_stage1.yaml)

## 2. 当前主线模块

| 能力 | 当前模块 | 当前事实 |
| --- | --- | --- |
| Base Planner | `Policy_ViewSelection_ETP` 与 ETP trainer 主干 | `ETP` 负责环境编码、图候选构造、基础候选分数 |
| Belief Update | `self_rssm_transition.py`, `self_rssm_correction.py`, `self_rssm_latent.py` | 当前 belief 通过 prior / posterior 和 latent KL 建模 |
| Relation Extraction | `relational_state_extractor.py` | `s_t` 由内部状态查询视觉 / 图 token 得到 |
| Future Rollout | `unified_rollout_trunk.py` | 候选条件化未来关系推演共用一条主干 |
| Physical Readout | `state_head.py`, `progress_curve_predictor.py` | 从统一未来表示读出 progress curve 和 curve health |
| Semantic Readout | `attention_head.py`, `attn_health_scorer.py` | 通过未来 attention 预测和统计量 MLP 读出 `attn_health` |
| Health Fusion | `progress_curve_predictor.py`, trainer 融合逻辑 | `curve_health` 与 `attn_health` 线性融合为 `health_k` |
| Intervention | `macro_intervention_manager.py` | Level 1 / 2 / 3 依赖 `D_t` 阈值和当前候选状态执行宏观控制 |

## 3. 当前唯一主数据流

当前仓库中的 `V5` 主数据流固定为：

`ETP backbone -> belief update -> relation_state -> rollout -> physical/semantic readout -> health fusion -> intervention`

更具体地说：

1. `ETP` 提供当前步基础输入：
   `x_t`, `g_t`, `x_tokens`, `g_tokens`, `lang_tokens`, `cand_feats`, `score_base_k`
2. RSSM 更新内部状态：
   `h_prior -> h_post`
   `z_post`
   `D_t = KL(post || prior)`
3. `RelationalStateExtractor` 构造：
   `s_t = RelationalStateExtractor([h_post, z_post], [x_tokens, g_tokens])`
4. 对每个 candidate `k`：
   - 动作条件下的内部状态先演化到 `h_next_k`, `z_next_k`
   - `UnifiedRolloutTrunk` 生成共享未来关系表示 `u_next_k`
5. 两路读出：
   - 物理路：`StateHead -> ProgressCurvePredictor -> curve_k -> curve_health_k`
   - 语义路：`AttentionHead -> predicted_attn_k -> AttnHealthScorer -> attn_health_k`
6. 健康融合：
   `health_k = lambda_curve_health * curve_health_k + lambda_attn_health * attn_health_k`
7. 校准与干预：
   - 常规分数融合：`score_final_k = score_base_k + alpha_t * health_bonus_k`
   - `MacroInterventionManager` 在 Level 1 / 2 / 3 上调整候选空间或直接执行宏观决策

## 4. 当前训练与推理约束

以下约束是当前代码事实：

- `Stage1` 冻结 `ETP` 主干，只训练 `V5` 模块。
- `Stage2` 延续同一主线，仅按 `stage2_unfreeze_keywords` 局部解冻 `ETP` 顶层。
- `D_t` 阈值当前采用 `batch quantile + EMA` 更新，并在推理期使用累计阈值 buffer。
- `Level 3` 当前实现默认策略已升级为 `energy_recovery`：由 self-model rollout 产出的未来健康度参与比较“继续前进 vs 恢复节点 vs STOP”；但主线默认仍保持关闭，避免在 `EB-01` 关闭前污染训练与评测。
- `MacroInterventionManager` 当前包含空 history 防护，避免把全 `-inf` backtrack history 误解释为有效回退目标。
- `STATENAV.eval_action_source` 是当前评估链路中的诊断接口，用于让动作选择与 stop score 来源保持一致；它不是方法主线贡献。
- `RelationalStateExtractor` 中 `extra_tokens` 相关分支当前不进入 full V5 主线，会被冻结。
- `ProgressCurvePredictor` 中 `h_post-only` 相关分支当前保留为消融基础设施，不属于 full V5 主线。
- 当前 `attn_health` 来源于 attention 统计特征和 MLP，不是显式 instruction graph 或 semantic stage model。

## 5. Blueprint 映射

| Blueprint 能力 | 当前实现 | 状态 | 说明 |
| --- | --- | --- | --- |
| `material_state m_t` | 由 `x_t`, `g_t`, `x_tokens`, `g_tokens`, `cand_feats` 分散表达 | `partial` | 当前没有统一的 `m_t` 容器 |
| `conscious_state c_t` | 由 `h_prior`, `h_post`, `z_post`, `attn_post` 分散表达 | `partial` | 当前没有单一 `c_t` 接口 |
| `contradiction_signal D_t` | RSSM posterior vs prior KL | `implemented` | 当前主线稳定存在 |
| `relation_state s_t` | `RelationalStateExtractor` 输出 | `implemented` | 当前已接回 rollout 主干 |
| candidate-conditioned future rollout | `UnifiedRolloutTrunk` + action-conditioned transition | `implemented` | 当前物理路和语义路共享未来主干 |
| semantic progression | `AttentionHead` + `AttnHealthScorer` | `partial` | 目前仍以 attention 统计量为主 |
| health fusion | trainer 融合 `curve_health` 与 `attn_health` | `implemented` | 当前默认参与主决策 |
| intervention policy | `MacroInterventionManager` Level 1 / 2 / 3 | `partial` | Level 1 / 2 当前随主线默认启用；Level 3 因 `EB-01` 暂不作为主线默认。当前实现已升级为 `energy_recovery` 形式：用 planner 分数与 self-model health 的联合价值比较“继续前进 vs 恢复节点 vs STOP” |

## 6. 当前非主线项

以下内容存在于仓库中，但不应被当作当前 `V5` 主线：

- `extra_tokens` 相关路径：保留分支
- `h_post-only` curve 路径：消融基础设施
- `STATENAV.eval_action_source`：诊断接口
- 显式打开 `STATENAV.enable_level3_macro=True` 的运行覆盖：诊断 / experimental override
- `run_r2r/v5/statenav_v5_eval_matrix.bash` 与 `run_rxr/v5/statenav_v5_eval_matrix.bash`：标准诊断矩阵入口
- `scripts/v5_eval_summary.py`：诊断汇总工具
- `no_attn` / `no_level3` 等配置：消融配置

它们可以作为实验或调试工具存在，但不能在方法定义中冒充主线能力。
