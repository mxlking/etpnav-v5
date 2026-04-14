# EFES 作者工作蓝图（论文固定版）

> Canonical source: `docs/last/EFES_FINAL_PAPER.md`
>
> 本文件服务作者工作流，不再主动削弱主稿。它的职责是：固定论文、组织实现补齐顺序、
> 规定后续实验如何回应主稿，而不是改写主稿强度。

## 1. 基本规则

- 主稿不可变，除非后续明确决定“论文与实现联动修订”。
- 蓝图、映射表、实验记录都必须服从主稿。
- `1.md / 2.md` 是历史稿，不再参与公式解释。

## 2. 当前主稿要求作者记住的对象

- `u_t = alpha * L_micro + beta * L_macro`
- `u_tilde = u_t - beta * d/2 * log(2*pi*sigma_max^2)`
- `g_t = sigmoid(eta * (u_tilde - tau))`
- `delta_t = B * tanh(f_delta(...))`
- `l_t = l_t^0 + g_t * delta_t`
- `L_nav + L_fe + L_cal + L_safe`

这些是下一轮实现和实验必须追的对象，不允许蓝图自行改写。

## 3. 主稿每节的写作纪律

### Abstract

- 保持主稿原有强理论主线。
- 不因为当前实现尚未闭环，就先把 theorem 降级。
- 结果段只写正式 full eval，不写 partial live 曲线。

### Introduction

- 固定 operational self-awareness 口径。
- 明确 EFES 的对象是：
  - agent 是否还在自己的 predictive manifold 上
  - 何时应该修正 base policy
- 不用历史草稿的旧叙事覆盖主稿。

### Theory

- theorem 编号与强度完全以主稿为准。
- 蓝图不得自行把 T5/T7 改成“只剩解释”。
- 如果当前实现不足，记到“代码缺口”，不是记到“论文降级”。

### Method

- 方法节解释“主稿对象如何落到当前代码”，而不是重新定义主稿对象。
- 可以如实写：
  - 当前 gate 已按主公式进入策略
  - 当前 dual quantity / MI 量还有实现缺口
- 不能写：
  - 主稿已经被当前弱实现替代

### Training Objective

- 论文主文仍然写 raw 定义：
  - `L_fe_raw`
  - `L_cal_raw`
  - `L_safe`
- 可以在实现说明中写 optimization proxies，但只能作为工程实现细节。
- 任何 proxy 都不能在蓝图中反向改写理论定义。

### Experiments

- 主实验服务于主稿检验，不服务于“证明当前代码没问题”。
- 结果表需要回答：
  - `u_tilde` 是否真的有判别力
  - gate 是否真的 selective
  - bounded residual 与 safe KL 是否真的形成保守纠正
- 如果结果弱，先判定为“实现未追上主稿”，不是“主稿应该自动降级”。

## 4. 下一轮代码追论文的优先级

1. **T5 对象闭环**
   - 当前 gate 已进入策略，但 dual quantity 仍未形成完整主稿闭环。
   - 下一轮代码要优先补这部分。

2. **T7 对象闭环**
   - 当前只有 `mi_proxy`。
   - 下一轮代码要决定如何把长期量做得更接近主稿 theorem。

3. **raw-vs-opt 训练路径**
   - 当前训练有 optimization proxy。
   - 要持续检查它们是否削弱了主稿对象的可学性。

4. **theory diagnostics 与导航指标联动**
   - `surprise_auc_base_error`
   - `gate_tp_rate / gate_fp_rate`
   - `kl_corrected_vs_base`
   - `teacher_margin_delta`
   - `episode_u_tilde_bar`

## 5. 结果解释规则

- 先问：当前代码有没有把主稿对象学起来。
- 再问：这些对象是否转化为导航收益。
- 如果出现以下情况，默认解释为“实现差距”，不是“主稿错误”：
  - `surprise_auc_base_error` 接近随机
  - `gate_tp_rate / gate_fp_rate` 长期全零
  - `safe_loss` 与 `kl_corrected_vs_base` 长期不动
  - full eval 与主稿主叙事相反

## 6. 本轮边界

- 本轮不继续改 theory 代码。
- 本轮不改 checkpoint / contract。
- 本轮只恢复“论文为准”的基准面，并产出后续代码追论文的输入。
