
# EFESV3 代码改造清单 + Theorem 对应实现表（最终版）

**版本**：v1.0  
**日期**：2026-04-13  
**用途**：把 EFES 论文蓝图中的理论对象，逐条落到 EFESV3 的可执行实现、日志与回归测试上。  
**适用仓库范围**：`efes-main` 中 EFESV3 新链路。  
**理论来源**：本文档以你上传的强化理论稿（T1–T7 + Corollary 1）为基底，但对“能直接写进投稿稿的强度”和“必须落到代码的条件”做了重新分级。  

---

## 0. 一句话原则

**论文可以比当前代码更前瞻，但论文中的每个强 claim，都必须能在代码里找到对应对象、在实验里找到对应诊断、在回归测试里找到对应验收。**

本版再加一条实现约束：

**理论主公式保持 raw 形式不变；训练允许使用等价 optimization proxy，但 raw quantities 必须继续显式输出到日志与分析。**

因此，EFES 的理论与实现建议分成三层：

### A. 必须落代码、且建议写进主文
这些是现在最值得绑定到实现的：
- **T1** Free-energy / variational discrepancy 的诊断解释
- **T2** shifted surprise 的 manifold-distance floor
- **T4** bounded intervention
- **T6** selective intervention gain lower bound

### B. 可以保留，但建议放 Appendix 或降级为“在条件下成立”
这些可以写，但不宜当主打 headline：
- **T3** tight infimum = squared manifold distance
- **C1** on-manifold characterization
- **T5** gate = optimal Lagrange multiplier

### C. 暂不建议作为主文强 claim
这些更适合作为“理论方向”或附录讨论，除非你愿意继续加重实现：
- **T7** ergodic self-consistency / mutual information characterization

---

## 1. 先给最终判断：哪些 theorem 现在必须驱动代码改造

| 等级 | Theorem / Claim | 推荐对外表述 | 是否必须改代码 |
|---|---|---|---|
| P0 | T4 Bounded intervention | 主文可写 | 是 |
| P0 | T6 Selective intervention | 主文可写 | 是 |
| P0 | T2 Manifold-distance floor | 主文可写，但要明确条件 | 是 |
| P1 | T1 Variational surprisal upper bound | 主文可写 | 是（对象要齐） |
| P1 | T3 Tight infimum | Appendix theorem / proof sketch | 是（若要保留） |
| P1 | C1 On-manifold characterization | Appendix corollary | 是（若要保留） |
| P2 | T5 Gate = Lagrange multiplier | 更稳妥写成“variational interpretation”，除非做 dual optimization | 是（若要强写） |
| P3 | T7 Ergodic MI theorem | 暂放附录或 future work | 是（若要强写） |

---

## 2. 代码改造总清单（按优先级）

## P0：今天就要改，不然 train/eval/infer 与主文不一致

### P0-1. 把 surprise 统一成 `u_t` 与 `u_tilde`
**要做什么**
- 保留训练损失中的原始
  \[
  u_t = \alpha L_{micro} + \beta L_{macro}
  \]
- 新增
  \[
  \tilde u_t = u_t - \beta \frac{d}{2}\log(2\pi \sigma_{max}^2)
  \]
- gate、日志、ROC、分析图，统一优先使用 `u_tilde`

**为什么**
- T2 / T3 / C1 都要求 shifted surprise，而不是原始 NLL 常数项未去掉的 surprise

**实现位置**
- `vlnce_baselines/models/efes_v3/predictor.py`
- `vlnce_baselines/models/efes_v3/agent.py`
- `vlnce_baselines/trainers/train_efes_v3.py`

---

### P0-2. 增加 `sigma_max`，并把 decoder variance clamp 到 `[sigma_min, sigma_max]`
**要做什么**
- 配置新增：
  - `sigma_min`
  - `sigma_max`
- 代码里：
  - `sigma = clamp(raw_sigma, sigma_min, sigma_max)`

**为什么**
- T2 的下界要用到 `sigma_max`
- T3 的 tight-infimum 也依赖上界方差
- 没有 `sigma_max`，只能讲一个很弱的“经验异常分数”，讲不出几何下界

**实现位置**
- `run_r2r/efes_v3/efes_v3_main.yaml`
- `vlnce_baselines/config/default.py`
- `vlnce_baselines/models/efes_v3/predictor.py`

---

### P0-3. Corrector 的 residual 改成有界形式
**要做什么**
- 把 correction residual 改成
  \[
  \delta_t = B \tanh(f_\delta(\cdot))
  \]
- 配置新增：
  - `delta_bound`

**为什么**
- T4 的 KL / TV bound 依赖 residual 有界
- 不加界，bounded intervention 就只是口号，不是 theorem

**实现位置**
- `run_r2r/efes_v3/efes_v3_main.yaml`
- `vlnce_baselines/config/default.py`
- `vlnce_baselines/models/efes_v3/corrector.py`

---

### P0-4. corrected path 与 base path 必须共享同一 valid mask
**要做什么**
- `invalid action mask` 可以保留
- **不能** 在 corrected path 额外加入 base path 没有的 visited-node mask
- visited 信息如果有价值，只能当 feature，不要当 correction 专属 mask

**为什么**
- T4 的 bounded intervention 有硬条件：同一 valid action set
- 你之前静态审查已经发现：只要 corrected path 偷偷多屏蔽 visited node，就算 `gate=0`，也不是 transparent pass-through

**实现位置**
- `vlnce_baselines/trainers/train_efes_v3.py`
- `vlnce_baselines/agents/efes_v3_agent.py`
- 如有 mask 拼接逻辑，也检查 base trainer 复用路径

---

### P0-5. `L_cal` 的监督目标固定为 base-policy error
**要做什么**
- calibration target 统一写成：
  \[
  y^{cal}_t = 1[\arg\max \pi_0 \neq y_t]
  \]
- 不要用 corrected-policy error 作为 gate target

**为什么**
- T5 / T6 都默认 gate 估计的是 “base policy 现在是否不可靠”
- 这能把 gate 的语义、理论和分析全部对齐

**实现位置**
- `vlnce_baselines/trainers/train_efes_v3.py`

---

### P0-6. train / eval / infer 三条链必须完全同语义
**要做什么**
- 训练用什么 corrected logits，评估和 inference 就也要用什么 corrected logits
- infer 模式必须写全 `path_eps` / `inst_ids` / prediction 输出逻辑
- 不能 train 用 `u_tilde + bounded residual + shared mask`，eval/infer 又走旧路径

**为什么**
- 这是论文最容易被代码打脸的地方
- 你之前已经有 inference 链路未闭合的问题

**实现位置**
- `vlnce_baselines/trainers/train_efes_v3.py`
- 若有必要，显式覆写 `inference()`

---

### P0-7. 修掉 all-masked bank attention 风险
**要做什么**
- 对 `has_bank == False` 的 batch 样本，不要硬做 attention
- 直接走 zero / learnable null memory 分支

**为什么**
- mixed precision 下 all-masked attention 行容易出 NaN
- 会导致训练中后期偶发数值炸掉，排查非常痛苦

**实现位置**
- `vlnce_baselines/models/efes_v3/predictor.py`

---

### P0-8. shape 合约显式化
**要做什么**
- 在 agent / trainer 初始化时 assert：
  - `d_model == obs_dim == cand_dim == instr_dim`
- 或者补全统一投影层

**为什么**
- 现在配置看起来 `d_model` 可调，但实际上高度耦合到 ETP hidden dim
- 不显式声明，后面改超参时会悄悄坏

**实现位置**
- `vlnce_baselines/models/efes_v3/agent.py`
- `vlnce_baselines/trainers/train_efes_v3.py`

---

## P1：一周内补齐，让理论在实验上站得住

### P1-1. 新增 `L_safe = KL(pi || pi0)`（建议）
**要做什么**
- 轻权重加一个 safe loss
- 配置新增：`lambda_safe`

**为什么**
- 虽然 T4 理论上只要求有界 residual + shared mask
- 但训练上加一个小的 KL 约束，更符合“保守纠偏”叙事，也更稳

**实现位置**
- `train_efes_v3.py`

---

### P1-2. 记录 theorem 对应的 6 个核心诊断
必须加的日志：
- `surprise_auc_base_error`
- `gate_mean`
- `gate_p90`
- `gate_tp_rate`（alpha）
- `gate_fp_rate`（beta_fp）
- `teacher_margin_delta`
- `kl_corrected_vs_base`
- `episode_u_tilde_bar`

**为什么**
- 没有这些指标，T4 / T5 / T6 / T7 都只能停留在 prose 上

---

### P1-3. 加完整 ablation 开关
配置至少支持：
- `use_body_loop`
- `use_predictor`
- `use_corrector`
- `use_shifted_surprise`
- `use_shared_valid_mask_only`
- `use_safe_kl`
- `use_sigma_max_clamp`

**为什么**
- 每条 theorem 最好有一个对应消融或诊断，不然论文很难 defend

---

## P2：想冲更强理论版时再做

### P2-1. 若要强写 T5，就不要只用 sigmoid gate
当前：
\[
g_t = \sigma(\eta(\tilde u_t - \tau))
\]

这更像 **dual variable 的单调近似**。  
如果你要在主文里强写 “gate 就是 optimal Lagrange multiplier”，建议改成两种之一：

**方案 A：降级写法（推荐）**
- 论文写成：
  - “the gate admits a variational interpretation as a monotone approximation to the optimal dual variable”
- 代码不必大改

**方案 B：强实现写法**
- 显式定义 KL-constrained correction objective
- 每步求一个 dual variable \(\lambda_t\)
- gate 由 \(\lambda_t\) 映射得到，或直接令 gate = clipped dual variable
- 记录 KKT residual

如果不做 B，T5 不建议写成 headline theorem。

---

### P2-2. 若要强写 T7，就需要 MI 侧的经验对象
当前只靠 episode 平均 surprise，不足以支撑
\[
\bar u_T \leftrightarrow H(X) - I(X;Z|H,M)
\]

至少应做：
- `episode_u_tilde_bar`
- 一个 MI proxy（例如 InfoNCE / MINE 风格的 `I_hat(X;Z|H,M)`）
- 站稳的 stationarity / long-run averaging 评估方案

否则 T7 更适合写成：
- appendix interpretation
- future direction
- 或 theorem under idealized assumptions, without empirical claim

---

## 3. 文件级改造表

| 文件 | 必改项 | 对应 theorem | 优先级 |
|---|---|---|---|
| `run_r2r/efes_v3/efes_v3_main.yaml` | 加 `sigma_max`, `delta_bound`, `lambda_safe`, ablation 开关, logging 开关 | T2/T3/T4/T6 | P0 |
| `vlnce_baselines/config/default.py` | 注册所有新字段，保证 merge 后可读 | 全部 | P0 |
| `vlnce_baselines/models/efes_v3/self_state.py` | 明确 body loop 输入，建议用 `prev_u_tilde`；可加 normalize | T7 / self-reference | P1 |
| `vlnce_baselines/models/efes_v3/predictor.py` | `sigma` clamp；输出 `u_t` 和 `u_tilde`；修 all-mask attention；返回 micro/macro 分项 | T1/T2/T3/T7 | P0 |
| `vlnce_baselines/models/efes_v3/corrector.py` | `delta = B*tanh(...)`；gate 用 `u_tilde`；不引入额外 visited mask | T4/T5 | P0 |
| `vlnce_baselines/models/efes_v3/agent.py` | 把 `u_tilde`, `gate`, `delta_norm`, `kl_dev` 等打包返回；assert shape 合约 | T2/T4/T6 | P0 |
| `vlnce_baselines/agents/efes_v3_agent.py` | 确保 eval / infer 走同一 corrected policy；检查 mask 语义 | T4/T6 | P0 |
| `vlnce_baselines/trainers/train_efes_v3.py` | base-error calibration；`L_safe`；train/eval/infer 对齐；记录 theorem diagnostics；修 inference 闭环 | T4/T5/T6/T7 | P0 |
| `run.py` | 通常不用大改，只需确认 trainer/inference 调度正确 | 链路闭环 | P1 |
| `vlnce_baselines/trainers/__init__.py` | 确认 trainer 注册无误 | 链路闭环 | P1 |

---

## 4. Theorem 对应实现表（最终版）

### T1 · Free energy upper-bounds surprisal

| 项目 | 内容 |
|---|---|
| 论文对象 | `micro KL + macro NLL` 组成的 variational discrepancy |
| 代码必须具备 | 明确 prior / posterior / observation likelihood 三个对象 |
| 最小实现 | `predictor.py` 中返回 `kl_micro`, `nll_macro`, `u_t` |
| 必要日志 | `kl_micro_mean`, `nll_macro_mean`, `u_t_mean` |
| 最小实验 | 训练中这三项都非退化；hard step 上 `u_t` 上升 |
| 不做会怎样 | “free-energy-style”会沦为文案，无法自证是 variational quantity |

---

### T2 · Shifted surprise lower-bounds manifold distance

| 项目 | 内容 |
|---|---|
| 论文对象 | `u_tilde = u_t - beta * d/2 * log(2*pi*sigma_max^2)` |
| 代码必须具备 | `sigma_max`；decoder variance clamp；`u_tilde` 显式输出 |
| 最小实现 | `predictor.py` 增加 `sigma_max` 和 `u_tilde`；gate 用 `u_tilde` |
| 必要日志 | `u_tilde_mean`, `recon_sq_error`, `sigma_mean`, `sigma_max_hit_rate` |
| 最小实验 | `u_tilde` 与 reconstruction error / base-policy error 正相关 |
| 不做会怎样 | 只能说“surprise 是经验分数”，讲不出几何下界 |

---

### T3 · Tight infimum equals squared manifold distance

| 项目 | 内容 |
|---|---|
| 推荐状态 | Appendix theorem，主文只说“tight under anomaly-regime assumptions” |
| 代码必须具备 | T2 全部条件 + self-manifold 的 decoder closure 假设不被实现直接破坏 |
| 最小实现 | 保证 model class 足以表达当前 decoder family；记录 `tightness_proxy_gap = u_tilde - beta/(2*sigma_max^2)*recon_sq_error` |
| 必要日志 | `tightness_proxy_gap`, `hard_state_fraction` |
| 最小实验 | 在 hard states 上，gap 不应系统性发散 |
| 不做会怎样 | 仍可保留 theorem，但只能是理想化 model-class 结果，不能写成经验性“已验证 tight” |

---

### C1 · On-manifold characterization

| 项目 | 内容 |
|---|---|
| 推荐状态 | Appendix corollary |
| 代码必须具备 | `u_tilde` 与 decoder reconstruction 对齐；允许 near-zero regime 观测 |
| 最小实现 | 记录 low-surprise easy-step 统计；定义 practical threshold `u_tilde < eps` |
| 必要日志 | `near_zero_surprise_rate`, `easy_step_recon_error` |
| 最小实验 | easy states 上 `u_tilde` 与低 reconstruction error 同步 |
| 不做会怎样 | 这条只能当数学 corollary，不能映到实验语言 |

---

### T4 · Bounded intervention

| 项目 | 内容 |
|---|---|
| 论文对象 | `delta = B*tanh(...)` + shared valid mask |
| 代码必须具备 | bounded residual；base/corrected 共享 valid mask；train/eval/infer 同语义 |
| 最小实现 | `corrector.py` 改 bounded residual；trainer 删 correction-only visited mask |
| 必要日志 | `delta_abs_max`, `kl_corrected_vs_base`, `gate_mean`, `pass_through_error` |
| 最小实验 | `KL(pi||pi0)` 随 gate 上升；gate≈0 时 corrected≈base |
| 不做会怎样 | “non-invasive”与“transparent pass-through”都不成立 |

---

### T5 · Gate as Lagrange multiplier

| 项目 | 内容 |
|---|---|
| 推荐状态 | 更稳妥写成“variational interpretation”；若强写 theorem，则需 dual implementation |
| 代码必须具备（弱版） | gate 是 `u_tilde` 的单调函数；`L_cal` 对 base error 学习 |
| 代码必须具备（强版） | 显式 KL-constrained objective + dual variable 求解/近似 |
| 最小实现（推荐） | 保留 sigmoid gate，但论文写 “monotone approximation to the optimal dual variable” |
| 必要日志 | `gate_mean`, `gate_tp_rate`, `gate_fp_rate`, `teacher_margin_delta` |
| 最小实验 | gate 对错步有明显区分；高 gate 时 margin 改善更多 |
| 不做会怎样 | 若还强写 “equals optimal multiplier”，reviewer 很容易抓住实现不符 |

---

### T6 · Selective intervention gain lower bound

| 项目 | 内容 |
|---|---|
| 论文对象 | 对错步上 gate 的选择性决定净收益 |
| 代码必须具备 | base-error target；gate threshold；统计 TP/FP 和 margin 改变 |
| 最小实现 | trainer/eval 中统计 `alpha`, `beta_fp`, `margin_gain_wrong`, `margin_cost_right` |
| 必要日志 | `surprise_auc_base_error`, `gate_tp_rate`, `gate_fp_rate`, `margin_gain_wrong`, `margin_cost_right` |
| 最小实验 | AUC ≥ 0.70；`alpha/beta_fp` 显著 > 1 |
| 不做会怎样 | 无法证明 gate 是 selective，而不是随机扰动 |

---

### T7 · Ergodic self-consistency / MI characterization

| 项目 | 内容 |
|---|---|
| 推荐状态 | 暂放 Appendix / Future，除非你愿意继续加 MI 估计器 |
| 代码必须具备（最低） | `episode_u_tilde_bar`、长期平均统计、body-loop 表示不退化 |
| 代码必须具备（强版） | 条件 MI proxy：`I_hat(X;Z|H,M)` |
| 最小实现 | 先记录 episode 平均 shifted surprise；可选再加 InfoNCE/MINE |
| 必要日志 | `episode_u_tilde_bar`, `success_vs_u_bar`, 可选 `mi_proxy` |
| 最小实验 | 低 `u_bar` episode 更可能成功；去 body loop 后这个关系变弱 |
| 不做会怎样 | 这条最多作为理论讨论，不能作为主实验结论 |

---

## 5. 推荐的配置新增字段

```yaml
EFES_V3:
  d_model: 768
  d_z: 256
  d_h: 512
  d_action: 128
  max_nodes: 50

  lr: 1.0e-4
  alpha: 1.0
  beta: 1.0
  lambda_macro: 1.0
  lambda_safe: 0.01
  grad_clip_norm: 5.0

  sigma_min: 0.01
  sigma_max: 1.0
  delta_bound: 2.0

  use_shifted_surprise: true
  use_body_loop: true
  use_predictor: true
  use_corrector: true
  use_safe_kl: true
  use_shared_valid_mask_only: true
  use_sigma_max_clamp: true

  LOGGING:
    log_gate_stats: true
    log_surprise_auc: true
    log_margin_shift: true
    log_kl_dev: true
    log_episode_u_bar: true
```

---

## 6. 最小代码改造顺序（最推荐）

### 第 1 轮：让论文最核心的 3 条先成立
1. `sigma_max + u_tilde`
2. `bounded residual`
3. `shared valid mask`

做到这三项后，论文里至少能稳地写：
- T2
- T4
- T6 的主体

---

### 第 2 轮：让 gate 真正可分析
4. base-error calibration target
5. 完整 diagnostics logging
6. train/eval/infer 同语义闭环

做到这轮后，你的 analysis section 才真正能写。

---

### 第 3 轮：决定是否冲更强理论稿
7. T3 作为 Appendix theorem 保留，并加 tightness proxy
8. 决定 T5 用“variational interpretation”还是强 dual 实现
9. 决定 T7 只放讨论，还是做 MI proxy

---

## 7. 不同 theorem 的“建议写法级别”

| Theorem | 最稳写法 | 更强写法 | 建议 |
|---|---|---|---|
| T1 | proposition / theorem | theorem | 可主文 |
| T2 | theorem with assumptions | theorem | 可主文 |
| T3 | appendix theorem | main theorem | 建议附录 |
| C1 | corollary | corollary | 建议附录 |
| T4 | theorem | theorem | 可主文 |
| T5 | variational interpretation | theorem | 默认先用前者 |
| T6 | theorem / proposition | theorem | 可主文 |
| T7 | discussion / appendix | theorem | 暂不建议主打 |

---

## 8. 最小回归测试清单（改完必须过）

### 语义正确性
- [ ] gate=0 时，所有合法动作上 `corrected_logits == base_logits`
- [ ] train/eval/infer 使用同一 corrected policy 逻辑
- [ ] inference 能完整输出 predictions，不再卡在旧父类隐式结构上

### 数值稳定性
- [ ] `sigma` 始终在 `[sigma_min, sigma_max]`
- [ ] 混合精度下 bank attention 不出 NaN
- [ ] `delta_abs_max <= delta_bound + tol`

### 理论映射
- [ ] `u_tilde` 已显式记录并驱动 gate
- [ ] calibration target 为 base-policy error
- [ ] corrected/base 共享同一 valid mask
- [ ] `KL(pi||pi0)` 与 gate 正相关，不出现 gate 低但 KL 巨大的样本

### 实验可写性
- [ ] 能导出 `surprise_auc_base_error`
- [ ] 能导出 `gate_tp_rate / gate_fp_rate`
- [ ] 能导出 `teacher_margin_delta`
- [ ] 能导出 `episode_u_tilde_bar`

---

## 9. 最后建议：哪些不要现在就写太满

### 不建议现在就在摘要里强写的
- “free energy is the unique canonical diagnostic”
- “gate equals the optimal Lagrange multiplier”
- “self-awareness emerges from material coupling” 的强数学版

### 更稳的摘要写法
- “free-energy surprise provides a principled endogenous anomaly signal”
- “the corrector admits a variational interpretation as conservative KL-constrained adjustment”
- “the embodied body–environment loop yields a self-referential state that supports internal failure diagnosis”

---

## 10. 最终执行建议

如果你的目标是 **先把论文主线打稳，再让代码跟上**，我建议你现在按下面的路线：

### Submission-safe 路线
- 主文：T1 + T2 + T4 + T6
- 附录：T3 + C1
- T5 写成 variational interpretation
- T7 放 appendix discussion / future work

### Strong-theory 路线
- 在上面基础上，再额外实现：
  - dual gate
  - MI proxy
  - tightness proxy
- 然后再考虑把 T5 / T7 升级

**一句话结论**：  
先把 **`sigma_max`、`u_tilde`、`bounded residual`、`shared valid mask`、`base-error calibration target`、`train/eval/infer 同语义`** 六件事做干净。  
这六件事，是 EFES 从“概念上很强”变成“论文、代码、定理三者对齐”的分水岭。
