# EFES_SELF 损失与诊断冻结合同 V1

> 版本号：`EFES_SELF v1-loss-contract`
> 状态：冻结
> 作用范围：`EFES_SELF` 当前主实现路径
> 生效原则：先冻结定义，再迭代实现；后续只允许调权重和阈值，不允许随意改核心公式和职责

## 1. 目的

本合同用于解决当前 `EFES_SELF` 推进中的两个核心风险：

1. 训练项与诊断项语义混用
2. 代码与论文对同一指标的命名和职责不一致

因此，本合同明确规定：

- 每个核心量是什么
- 每个核心量做什么
- 哪些公式当前冻结
- 哪些量允许进入诊断聚合器
- 哪些改动必须视为版本升级

## 2. 核心量分层

所有核心量只允许属于以下三类之一。

### 2.1 训练项

只用于优化，不直接进入诊断解释。

- `node_nll_loss`
- `micro_train_loss`
- `bind_loss`
- `clarity_loss`
- `plan_loss`

### 2.2 诊断项

只用于解释、行为输入和案例分析，不承担训练目标语义。

- `macro_diag_score`
- `micro_diag_score`
- `ground_diag`
- `self_mismatch`
- `delta_control`
- `delta_boundary`
- `delta_progress`

### 2.3 辅助读出

只负责自我组织性或可解释性读出。

- `macro_uncertainty`
- `self_clarity`
- `kappa_self`

## 3. 核心量职责

### 3.1 `node_nll_loss`

- 类别：训练项
- 职责：训练宏观 supporting prediction mechanism
- 允许为负：是
- 进入 `self_mismatch`：否
- 进入论文主诊断图：否

### 3.2 `macro_diag_score`

- 类别：诊断项
- 职责：表示宏观层的非负裂缝证据
- 允许为负：否
- 进入 `self_mismatch`：是
- 进入论文主诊断图：是

### 3.3 `macro_uncertainty`

- 类别：辅助读出
- 职责：表示宏观预测不确定性
- 直接等同诊断强度：否
- 进入 `self_mismatch`：否

### 3.4 `micro_train_loss`

- 类别：训练项
- 职责：训练微观连续动态
- 允许经过 cap：是
- 进入 `self_mismatch`：否

### 3.5 `micro_diag_score`

- 类别：诊断项
- 职责：表示微观层的非负裂缝证据
- 与 `micro_train_loss` 同名同义：否
- 进入 `self_mismatch`：是

### 3.6 `self_mismatch`

- 类别：诊断项
- 职责：统一摘要当前 self 与 reality 的裂缝强度
- 允许读取训练项：否
- 允许读取 `kappa_self`：是

### 3.7 `self_clarity`

- 类别：辅助读出
- 职责：probe 当前 self 是否清晰、稳定、可持续
- 当前是否主控制信号：否

### 3.8 `kappa_self`

- 类别：辅助读出
- 职责：表示 self 组织内聚度
- 可否进入 `self_mismatch`：是

## 4. 当前冻结公式

## 4.1 宏观三分离

### `node_nll_loss`

\[
L_{node}^{nll}
=
\frac{1}{2}\operatorname{mean}_d
\left(
\frac{(x_{comp}^{sg}-\mu)^2}{\sigma^2}
+
\log \sigma^2
\right)
\]

说明：

- `sg` 表示 stop-gradient 目标分支
- 该量允许为负
- 只作训练项

### `macro_diag_score`

\[
C_{macro}^{diag}
=
\operatorname{mean}_d
\left(
\frac{(x_{comp}^{sg}-\mu^{sg})^2}
{\operatorname{clip}(\operatorname{stopgrad}(\sigma^2), s_{min}, s_{max})}
\right)
\]

说明：

- 必须非负
- 不含 `log sigma^2`
- 只作诊断项

### `macro_uncertainty`

\[
U_{macro}
=
\operatorname{mean}_d(\log \operatorname{stopgrad}(\sigma^2))
\]

说明：

- 只作不确定性读出
- 不直接等于诊断强度

## 4.2 微观训练与诊断双通道

### `micro_train_loss`

\[
L_{\mu}^{train}
=
\operatorname{clip}(\max(\mathrm{KL}_{raw}, \tau_{free}), \tau_{cap})
\]

当前实现对应：

- `free_nats = 3.0`
- `micro_kl_cap = 20.0`

### `micro_diag_score`

\[
C_{\mu}^{diag}
=
\log(1 + \max(\mathrm{KL}_{raw}, 0))
\]

说明：

- 这是当前 `v1` 的正式合同定义
- 后续若更改，必须升版本

## 4.3 统一诊断摘要

\[
z = \frac{raw - \mu}{\sqrt{var}}
\]

\[
z_{\mu} = \operatorname{relu}(z_{micro}),\;
z_{M} = \operatorname{relu}(z_{macro}),\;
z_{g} = \operatorname{relu}(z_{ground})
\]

\[
self\_penalty = 1 - \operatorname{clip}(kappa\_self, 0, 1)
\]

\[
self\_mismatch
=
\log\left(1 + z_{\mu} + z_{M} + z_{g} + self\_penalty\right)
\]

说明：

- `z_*` 使用 train-only EMA 统计量标准化
- `self_mismatch` 只读取诊断项和 `kappa_self`
- `macro_valid_mask = False` 时，宏观通道不得贡献诊断量

## 4.4 `bind_loss`

当前 `v1` 冻结为：

\[
y_{bind}
=
\operatorname{clip}\left(\exp(-\operatorname{stopgrad}(self\_mismatch)), 0.05, 0.95\right)
\]

\[
L_{bind}
=
\operatorname{BCE}(kappa\_self, y_{bind})
\]

说明：

- 该定义当前允许保留为 `v1`
- 但它是否进入最终论文主叙事，仍需后续评审

## 4.5 `clarity_loss`

当前 `v1` 冻结为：

- 基于近未来窗口内 `micro_diag_score` 的稳定性构造二元目标
- 用于 `self_clarity` 校准

说明：

- 该项当前只作辅助项
- 不得当作核心理论结果解释

## 5. 明确禁止事项

以下做法在 `v1` 中被明确禁止：

1. `self_mismatch` 读取 `node_nll_loss`
2. `self_mismatch` 读取 `micro_train_loss`
3. 任何 raw NLL / raw KL 训练项直接进入诊断聚合器
4. `RealityEncoder` 读取人工 `stuck / drift / proceed` 标签
5. 在论文或日志中重新使用含糊的 `node_loss` 指代 `node_nll_loss`
6. 把 `self_clarity` 重新上升为主控制信号
7. 把 `recovery` 写成方法主体

## 6. 当前实现分类

按照顶会代码纪律，当前 `EFES_SELF` 中的内容分为三类：

### 6.1 结构修正

这些内容可直接进入主线：

- 宏观三分离
- 诊断聚合器不读训练项
- `RealityEncoder` 只接收现实回波
- 最小三模式行为接口

### 6.2 稳定化护栏

这些内容当前保留在代码中，但默认不写成最终方法定义：

- `micro_kl_cap`
- 非有限梯度清零
- 各类只为防爆炸而设的 clamp / warmup / cap

### 6.3 待重做项

这些内容当前只能视为实验性基线：

- `self_clarity` 目标定义
- `macro_valid` 的稀疏性门控
- `SelfBinder` 的组织性约束是否足够强

## 7. 后续允许改什么

在 `v1-loss-contract` 冻结后，后续只允许直接改：

- `lambda_*`
- 各类阈值
- 各类窗口大小
- warmup、cooldown、cap 的数值

这些不算版本升级。

## 8. 什么算版本升级

以下动作必须升版本，不能混入当前结果：

- 修改任何核心公式
- 修改任何核心量职责
- 让训练项重新进入诊断聚合器
- 把 `self_clarity` 从 probe 改成主控制信号
- 把行为层从最小接口扩成复杂 router 并改变主训练目标

如果发生上述任一动作，应创建：

- `EFES_SELF_LOSS_DIAG_CONTRACT_V2_CN.md`

并在论文、方法说明、日志中同步版本号。

## 9. 与论文和代码的对应关系

本合同约束以下三处必须同名同义：

- [EFES_PAPER_DRAFT_CN.md](/home/D/liumeng/vln/260316P/etpnav-v5/docs/efes/EFES_PAPER_DRAFT_CN.md)
- [EFES_METHOD_CN.md](/home/D/liumeng/vln/260316P/etpnav-v5/docs/efes/EFES_METHOD_CN.md)
- `EFES_SELF` 代码与日志

不允许出现：

- 论文里叫诊断量，代码里其实是训练项
- 论文里叫 self mismatch，日志里却混入 raw NLL
- 论文里写 recovery 是下游，代码里却让行为层重新主导训练
