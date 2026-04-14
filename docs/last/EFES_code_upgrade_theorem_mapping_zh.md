# EFES 论文驱动代码差距审计

> Canonical source: `docs/last/EFES_FINAL_PAPER.md`
>
> 本文件不再决定论文能写多强，而是以主稿为固定基准，审计当前 `EFESV3Theory`
> 代码距离主稿还差哪些 theorem-critical 对象。

## 1. 使用原则

- 主稿是唯一强口径来源。
- 本文件只回答两件事：
  - 哪些论文对象已经落到代码。
  - 哪些论文对象还没有被当前实现真正支撑。
- 任何“先降级论文再解释代码”的做法都不再采用。

## 2. 已对齐的主稿对象

| 主稿对象 | 论文定义 | 当前代码对象 | 当前日志/诊断 |
| --- | --- | --- | --- |
| `u_t` | `alpha * L_micro + beta * L_macro` | `TheoryPredictor.forward()` 中 `u_t_raw` | `u_t_raw_mean` |
| `u_tilde` | `u_t - beta * d/2 * log(2*pi*sigma_max^2)` | `TheoryPredictor.forward()` 中 `u_tilde` | `u_tilde_mean`, `episode_u_tilde_bar`, `surprise_auc_base_error` |
| `sigma` clamp | `sigma_i in [sigma_min, sigma_max]` | `TheoryPredictor._macro_sigma()` | `sigma_mean`, `sigma_hit_low_rate`, `sigma_hit_high_rate` |
| bounded residual | `delta_t = B * tanh(f_delta(...))` | `TheoryCorrector.forward()` 中 `delta` | `delta_abs_max` |
| gate 主公式 | `g_t = sigmoid(eta * (u_tilde - tau))` | `TheoryCorrector.forward()` 中 `gate_raw`, `gate` | `gate_mean`, `gate_p90`, `gate_tp_rate`, `gate_fp_rate` |
| corrected policy | `l_t = l_t^0 + g_t * delta_t` | `corrected_logits` | `kl_corrected_vs_base`, `teacher_margin_delta` |
| shared valid mask | corrected path 与 base path 共享 `A_t^valid` | `cand_mask` / `invalid_candidate_mask` 同源 | `kl_corrected_vs_base`, eval 行为一致性 |
| `L_nav` | `CE(pi, y_t)` | trainer 中 `nav_loss` | `nav_loss` |
| `L_fe_raw` | `L_micro + lambda_macro * L_macro` | trainer 中 `fe_step_raw` | `fe_loss_raw` |
| `L_cal_raw` | `BCE(g_t, 1[argmax pi_0 != y_t])` | trainer 中 `calib_raw` | `calibrate_loss_raw` |
| `L_safe` | `KL(pi || pi_0)` | trainer 中 `safe_loss_sum` | `safe_loss`, `kl_corrected_vs_base` |
| train/eval/infer 同语义 | 同一 corrected policy | theory trainer + agent | contract/meta/action_source |

## 3. 仍未追上主稿的实现缺口

### 3.1 T5 对应缺口

主稿当前口径仍然是：

- gate 从 KL-constrained policy optimization 推出
- gate 与最优 Lagrange multiplier 同构

当前代码真实状态：

- `gate` 已按主稿主公式进入策略
- `lambda_hat` / `use_dual_gate` 仍主要是诊断量
- `lambda_hat` 不单独控制 corrected logits，也没有形成独立 dual objective 闭环

结论：

- **T5 的主公式对象已部分落地**
- **T5 的强实现闭环还没有完全落地**

下一轮代码工作需要补：

- dual quantity 是否要真正进入策略或训练闭环
- 如果不进入，如何给出比“诊断量”更强的 theorem-consistent realization

### 3.2 T7 对应缺口

主稿当前口径仍然保留 theorem-level 叙事：

- 长时平均 surprise 与 agent-environment mutual information 的关系

当前代码真实状态：

- 只有 batch-level `mi_proxy`
- 没有 trajectory-level、ergodic-style、stationary-style 的直接估计

结论：

- **T7 目前只具备经验 proxy**
- **距离主稿 theorem 级实现还差一整层**

下一轮代码工作需要补：

- 轨迹级 MI 估计或更贴近 theorem 的经验对象
- `episode_u_tilde_bar` 与长期量之间的更直接诊断闭环

### 3.3 raw 理论量与优化代理的分裂

主稿理论对象是：

- `L_fe_raw`
- `L_cal_raw`
- `u_tilde`

当前训练实现同时使用：

- `L_fe_opt`
- `L_cal_opt`
- stop-gradient / detached diagnostic path

这不是错误，但它意味着：

- 当前训练不是“主稿对象单一路径直训”
- 后续如果结果不理想，首先要审查 optimization proxies 是否削弱了主稿对象

### 3.4 theorem object 与实现对象的空间差异

主稿记号里 `x_t` 可被读成 observation。
当前实现里更接近：

- feature-space observation
- projected node feature target

这不是本轮要改主稿的理由，而是下一轮代码/实验说明要补的对象界定。

## 4. 当前不视为代码缺口、但必须持续监控的项

- `surprise_auc_base_error`
- `gate_tp_rate`
- `gate_fp_rate`
- `kl_corrected_vs_base`
- `teacher_margin_delta`
- `episode_u_tilde_bar`

这些量决定的是：

- 代码是否真的把主稿对象学起来
- 不是主稿本身是否成立

## 5. 下一轮代码追论文的直接输入

后续代码改造优先顺序固定为：

1. T5 闭环：让 dual quantity 与策略/训练目标的关系更接近主稿。
2. T7 闭环：从 `mi_proxy` 走向更贴近 theorem 的长期量实现。
3. raw-vs-opt 问题：检查 optimization proxies 是否在削弱 `u_tilde` 与 gate selectivity。
4. object-space 问题：明确实现里 `x_t` 是 feature-space object，并决定是否要进一步贴近主稿对象。

## 6. 历史文档地位

- `1.md / 2.md` 继续保留为历史稿。
- 它们不再参与当前公式解释，也不参与主线实验口径。
