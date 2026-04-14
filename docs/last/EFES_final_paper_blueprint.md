# EFES 主稿写作蓝图

> Canonical source: `docs/last/EFES_FINAL_PAPER.md`
>
> 本文件不是第二篇论文，只是作者写作和实现对齐说明。所有公式、定理、术语最终都以主稿为准。

## 1. 当前统一口径

- 论文主线：`Self-State -> Predictor -> bounded gated Corrector`
- 公式主线：
  - `u_t = alpha * L_micro + beta * L_macro`
  - `u_tilde = u_t - beta * d/2 * log(2*pi*sigma_max^2)`
  - `g_t = sigmoid(eta * (u_tilde - tau))`
  - `delta_t = B * tanh(f_delta(...))`
  - `l_t = l_t^0 + g_t * delta_t`
- 训练主线：
  - `L_nav + lambda_fe * L_fe + lambda_cal * L_cal + lambda_safe * L_safe`
  - raw quantities 与 optimization proxies 分层记录

## 2. 每节应该怎么写

### Abstract

- 只写三模块和一个核心结论：
  - predictive inconsistency 可以作为内生诊断
  - bounded gated correction 可以保守干预
- 结果部分只写已经拿到的正式 full eval，不写 partial live 曲线。

### Introduction

- 把问题限定为 operational self-awareness，不谈 phenomenology。
- 强调 EFES 的对象是：
  - agent 是否还在自己的 predictive manifold 上
  - 何时应该保守修正 base policy

### Theory

- 以主稿 theorem 编号为准，不再在别的文档里换编号。
- 主文保留：
  - T1
  - T2
  - T3
  - T4
  - T6
- T5 和 T7 的经验表述收紧：
  - T5 写作 variational / diagnostic interpretation
  - T7 只报 `mi_proxy`，放 discussion 或 appendix

### Method

- 方法节只解释论文对象如何落到代码，不额外发明新术语。
- gate 统一写成：
  - `g_t = sigmoid(eta * (u_tilde - tau))`
- `lambda_hat` 只作为诊断量提及，不写成最终动作控制量。
- corrected path 共享 valid-action mask，要明确写出。

### Training Objective

- 论文正文写 raw loss：
  - `L_fe_raw`
  - `L_cal_raw`
  - `L_safe`
- 紧接着加一个实现说明段：
  - 代码为优化稳定使用 `L_fe_opt`、`L_cal_opt`
  - 但 raw quantities 继续输出到日志和分析

### Experiments

- 主结果表只引用：
  - baseline
  - `EFESV3Theory`
- 诊断表只引用：
  - `surprise_auc_base_error`
  - `gate_mean`
  - `gate_p90`
  - `gate_tp_rate`
  - `gate_fp_rate`
  - `kl_corrected_vs_base`
  - `teacher_margin_delta`
  - `episode_u_tilde_bar`

## 3. 禁止再出现的混乱写法

- 不再把 `1.md / 2.md` 当成公式来源。
- 不再把 `lambda_hat` 写成和主 gate 并列的动作控制量。
- 不再出现 `sigma_min` 版 shifted surprise。
- 不再把 optimization proxy 写成理论定义本身。
- 不再把旧 `v3-clean` 结果写成当前理论主线结果。

## 4. 结果解释规则

- 先看 theory diagnostics 是否成立，再解释导航指标。
- 如果出现以下情况，不先改论文结论，先回到实现排查：
  - `surprise_auc_base_error` 接近随机
  - `gate_tp_rate / gate_fp_rate` 长期全零
  - `safe_loss` 和 `kl_corrected_vs_base` 长期不动
- 只有当 theory diagnostics 和 full eval 同时支持时，才把强理论叙事写进主文结果段。
