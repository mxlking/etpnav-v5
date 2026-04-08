# EFES：用于视觉语言导航自我诊断与双模式恢复的具身自由能框架

> 稿件状态：方法初稿
> 说明：本稿件是 EFES 主线的论文占位版本，用于替换早期 V6 叙事。

## 核心主线

EFES 不把 topo memory 或 world-model prediction 当作最终贡献，而是把它们统一组织成一条更明确的链路：

`Self State -> Dual-Scale Coherence -> Free Energy -> Self-Confidence -> Dual Recovery`

对应的四个论文重点是：

1. `Self State`
   用统一意识态表示当前指令阶段、后验信念和语言注意。
2. `Dual-Scale Coherence`
   用 `MicroRSSM` 计算微观矛盾度 `C_micro`，用 `TopoStateBank + NodeSurprise` 计算宏观惊讶度 `C_macro`。
3. `Free Energy + Self-Confidence`
   用统一异常量 `A_t` 做报警，再用 `pi_t` 校准“我是否相信自己当前判断”。
4. `Dual Recovery`
   区分“环境不对，需要探索”和“我自己可能错了，需要更新信念”。

## 当前实现范围

- 任务：R2R
- 训练：单 trainer，双阶段冻结/解冻
- 基线：V6 保留，EFES 单独实现
- 当前代码入口：
  - `vlnce_baselines/trainers/train_efes.py`
  - `vlnce_baselines/agents/efes_agent.py`
  - `vlnce_baselines/models/efes/`
  - `run_r2r/efes/efes.bash`

## 后续写作要求

- 正文不再以 V6 为主方法名
- 强调 EFES 的顶层逻辑是“自我诊断驱动恢复”
- V6 的 `D_t_latent + D_t_node + MacroIntervention` 叙事只作为早期基础子系统，不再作为顶层方法标题
