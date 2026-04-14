# EFES 论文-代码对齐映射表

> Canonical source: `docs/last/EFES_FINAL_PAPER.md`
>
> 本文件只做“论文对象 -> 代码对象 -> 日志对象”的映射说明，不再自立一套更强或更弱的主公式。

## 1. 对齐原则

- 理论主公式以 `EFES_FINAL_PAPER.md` 为准。
- `EFESV3Theory` 是当前唯一正式实现主线。
- raw theoretical quantities 保持不变。
- optimization proxies 只作为训练稳定手段，不能替代理论对象。

## 2. 核心对象映射

| 论文对象 | 论文定义 | 当前代码对象 | 当前日志/诊断 |
| --- | --- | --- | --- |
| `u_t` | `alpha * L_micro + beta * L_macro` | `TheoryPredictor.forward()` 中 `u_t_raw` | `u_t_raw_mean` |
| `u_tilde` | `u_t - beta * d/2 * log(2*pi*sigma_max^2)` | `TheoryPredictor.forward()` 中 `u_tilde` | `u_tilde_mean`, `episode_u_tilde_bar`, `surprise_auc_base_error` |
| `sigma` clamp | `sigma_i in [sigma_min, sigma_max]` | `TheoryPredictor._macro_sigma()` | `sigma_mean`, `sigma_hit_low_rate`, `sigma_hit_high_rate` |
| bounded residual | `delta_t = B * tanh(f_delta(...))` | `TheoryCorrector.forward()` 中 `delta` | `delta_abs_max` |
| gate | `g_t = sigmoid(eta * (u_tilde - tau))` | `TheoryCorrector.forward()` 中 `gate_raw`, `gate` | `gate_mean`, `gate_p90`, `gate_tp_rate`, `gate_fp_rate` |
| corrected policy | `l_t = l_t^0 + g_t * delta_t` | `corrected_logits` | `kl_corrected_vs_base`, `teacher_margin_delta` |
| shared valid mask | corrected path 与 base path 共享 `A_t^valid` | `cand_mask` / `invalid_candidate_mask` 只做同源 mask | `kl_corrected_vs_base`, eval 行为一致性 |
| `L_nav` | `CE(pi, y_t)` | trainer 中 `nav_loss` | `nav_loss` |
| `L_fe_raw` | `L_micro + lambda_macro * L_macro` | trainer 中 `fe_step_raw` | `fe_loss_raw` |
| `L_fe_opt` | raw FE 的优化代理 | trainer 中 `fe_step_opt` | `fe_loss_opt` |
| `L_cal_raw` | `BCE(g_t, 1[argmax pi_0 != y_t])` | trainer 中 `calib_raw` | `calibrate_loss_raw` |
| `L_cal_opt` | `BCEWithLogits(gate_raw, target)` + class balance | trainer 中 `calib_opt` | `calibrate_loss_opt` |
| `L_safe` | `KL(pi || pi_0)` | trainer 中 `safe_loss_sum` | `safe_loss`, `kl_corrected_vs_base` |

## 3. theorem 对应实现强度

| Theorem / Claim | 当前实现状态 | 当前可写强度 |
| --- | --- | --- |
| T1 variational discrepancy | 已对齐 | 主文可写 |
| T2 shifted surprise lower-bounds manifold floor | 已对齐到 `u_tilde` 与 `sigma_max` | 主文可写 |
| T3 tight infimum / achieved | 论文给出主推导，代码只提供对象和诊断 | 主文保留理论，实验不额外夸大 |
| T4 bounded intervention | 已对齐到 bounded residual + shared mask + safe KL diagnostics | 主文可写 |
| T5 gate / dual interpretation | 当前以主 gate 为准，`lambda_hat` 仅保留诊断解释 | 只写 variational / diagnostic interpretation |
| T6 selective intervention | 已对齐到 `gate_tp_rate`, `gate_fp_rate`, `surprise_auc_base_error` | 主文可写 |
| T7 ergodic MI characterization | 当前只有 `mi_proxy` | 附录 / discussion |

## 4. 训练与日志约束

- 任何训练问题先看 raw quantities：
  - `fe_loss_raw`
  - `calibrate_loss_raw`
  - `u_tilde_mean`
- 再看 optimization proxies 是否把梯度压坏：
  - `fe_loss_opt`
  - `calibrate_loss_opt`
- 结果解释只允许引用与论文对象直接对应的诊断：
  - `surprise_auc_base_error`
  - `gate_mean`
  - `gate_p90`
  - `gate_tp_rate`
  - `gate_fp_rate`
  - `kl_corrected_vs_base`
  - `teacher_margin_delta`
  - `episode_u_tilde_bar`

## 5. 当前实现约束

- `lambda_hat`、`use_dual_gate` 只保留为诊断/解释对象，不作为主策略控制量。
- `train / eval / infer` 必须共用同一 corrected policy 语义。
- `1.md / 2.md` 是历史稿，不再参与公式解释或实验口径。
