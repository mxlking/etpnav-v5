# V5 Traceability Matrix

`version`: `V5 规范 v1`
`updated_at`: `2026-04-06`

文档角色：蓝图-代码-gap 对照总表

本表只做一件事：把 `V5_FINAL_BLUEPRINT_CN.md`、`V5_CODE_CURRENT_CN.md`、`V5_GAP_ROADMAP_CN.md` 三份规范文档串成可执行约束。

使用规则：

1. 每行只描述一个核心能力。
2. `status` 必须与 `V5_FINAL_BLUEPRINT_CN.md` 保持一致。
3. `current code module` 只写当前主映射，不写愿景性模块。
4. `mainline / ablation / diagnostic` 只允许填一种主 lane。
5. `gap id` 与 `evidence artifact` 必须能在 `V5_GAP_ROADMAP_CN.md` 中找到对应项。

| capability | blueprint definition | current code module | status | mainline / ablation / diagnostic | gap id | evidence artifact | owner |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `material_state m_t` | 当前步对 agent 独立存在的外部现实证据 | `x_t`, `g_t`, `x_tokens`, `g_tokens`, `cand_feats` 在 trainer / agent 中分散表达 | `partial` | `mainline` | `AG-02` | `V5_CODE_CURRENT_CN.md` 中 `material_state m_t` 标记为 `partial` 且说明为分散表达 | `statenav_agent_v5.py`, `train_statenav_v5_stage1.py` |
| `conscious_state c_t` | 经现实修正后的内部 belief / task focus / self-evaluation | `h_prior`, `h_post`, `z_post`, `attn_post` 在 RSSM 与 trainer 中分散表达 | `partial` | `mainline` | `AG-02` | `V5_CODE_CURRENT_CN.md` 中 `conscious_state c_t` 标记为 `partial` 且说明为分散表达 | `self_rssm_transition.py`, `self_rssm_correction.py`, `self_rssm_latent.py` |
| `contradiction_signal D_t` | 现实对内部状态的纠偏强度 | RSSM posterior vs prior KL | `implemented` | `mainline` | `AG-02` | `V5_CODE_CURRENT_CN.md` 中 `contradiction_signal D_t` 标记为 `implemented` | `self_rssm_latent.py`, `train_statenav_v5_stage1.py` |
| `relation_state s_t` | `conscious_state` 主动检索 `material_state` 后得到的关系压缩表示 | `relational_state_extractor.py` 输出并接入 rollout query | `implemented` | `mainline` | `AG-02` | `V5_CODE_CURRENT_CN.md` 中 `relation_state s_t` 标记为 `implemented` | `relational_state_extractor.py`, `unified_rollout_trunk.py` |
| `candidate-conditioned future rollout` | 在候选动作条件下对未来关系演化的统一推演 | `unified_rollout_trunk.py` 与 action-conditioned transition | `implemented` | `mainline` | `AG-01` | `V5_CODE_CURRENT_CN.md` 中 rollout 主线映射；trainer 仍通过长 `rollout()` 驱动 | `unified_rollout_trunk.py`, `train_statenav_v5_stage1.py` |
| `semantic progression` | 从统一未来表示读出语义阶段推进质量 | `attention_head.py` + `attn_health_scorer.py` | `partial` | `mainline` | `RG-01` | `V5_GAP_ROADMAP_CN.md` 中 `RG-01` 的 evidence artifact | `attention_head.py`, `attn_health_scorer.py` |
| `health fusion` | 统一融合物理健康度与语义健康度 | trainer 中 `curve_health` 与 `attn_health` 的线性融合 | `implemented` | `mainline` | `RG-01` | `V5_CODE_CURRENT_CN.md` 中 `health fusion` 标记为 `implemented`；语义路仍受 `RG-01` 约束 | `progress_curve_predictor.py`, `train_statenav_v5_stage1.py` |
| `intervention policy` | 基于 `D_t` 与未来健康度执行 Level 1 / 2 / 3 宏观控制 | `macro_intervention_manager.py`，当前 `Level 3` 已升级为 `energy_recovery`，用 planner 分数与 self-model health 比较 continue / recover / stop | `partial` | `mainline` | `RG-02, EB-01, EB-02` | `V5_CODE_CURRENT_CN.md` 已记录当前主线默认保留 Level 1 / 2，Level 3 仅作为显式 override；`V5_GAP_ROADMAP_CN.md` 中 `EB-01`、`EB-02` 记录了 `diag_eval_r2r_s1` 证据 | `macro_intervention_manager.py`, `train_statenav_v5_stage1.py` |
| `eval action-source parity` | 让动作选择与 stop score 来源保持一致的诊断能力 | `STATENAV.eval_action_source` 与 trainer 中 `behavior_logits` 路径 | `implemented` | `diagnostic` | `EB-02` | `V5_GAP_ROADMAP_CN.md` 中 `EB-02` 的 evidence artifact | `default.py`, `train_statenav_v5_stage1.py` |
| `standard eval matrix` | 固定 `statenav / etp / no_level3` 三路对照与统一汇总模板 | `run_r2r/v5/statenav_v5_eval_matrix.bash`, `run_rxr/v5/statenav_v5_eval_matrix.bash`, `scripts/v5_eval_summary.py` | `implemented` | `diagnostic` | `EB-02` | `diag_eval_r2r_s1` 已给出 fresh 三路结果；矩阵脚本现显式切换 `STATENAV.enable_level3_macro`，不再受主线默认值影响 | `run_r2r/v5/statenav_v5_eval_matrix.bash`, `run_rxr/v5/statenav_v5_eval_matrix.bash`, `scripts/v5_eval_summary.py` |
| `h_post-only curve path` | 仅依赖 `h_post` 的 curve 消融路径 | `progress_curve_predictor.py` 中保留分支与相关 YAML | `partial` | `ablation` | `AG-03` | `V5_CODE_CURRENT_CN.md` “当前非主线项”中已标记为消融基础设施 | `progress_curve_predictor.py`, `run_r2r/v5/*.yaml` |
| `extra_tokens path` | 额外 token 的保留分支 | `relational_state_extractor.py` 中 `extra_tokens` 相关路径 | `partial` | `ablation` | `AG-03` | `V5_CODE_CURRENT_CN.md` “当前非主线项”中已标记为保留分支 | `relational_state_extractor.py` |
