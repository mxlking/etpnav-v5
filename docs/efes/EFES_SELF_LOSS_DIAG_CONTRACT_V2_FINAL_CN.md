# EFES_SELF 损失与诊断冻结合同 V2-Final

> 版本号：`EFES_SELF v2-final`
> 状态：冻结
> 作用范围：`EFES_SELF` 当前最终版主实现路径
> 生效原则：先冻结定义，再迭代实现；后续只允许调权重和阈值，不允许随意改核心公式和职责

## 1. 目的

本合同用于把 `EFES_SELF` 收成顶会最终版最小骨架，并明确：

- 哪些量是训练项
- 哪些量是诊断项
- 哪些量只是辅助读出
- 哪些内容只算工程 guardrail

## 2. 核心量分层

### 2.1 训练项

- `node_nll_loss`
- `micro_train_loss`
- `bind_loss`
- `plan_loss`

### 2.2 诊断项

- `macro_diag_score`
- `micro_diag_score`
- `ground_diag`
- `self_mismatch`
- `delta_control`
- `delta_boundary`
- `delta_progress`

### 2.3 辅助读出

- `macro_uncertainty`
- `kappa_self`

`self_clarity` 不再属于 `v2-final` 主路径；若保留代码，仅作实验性分支。

## 3. 核心量职责

### 3.1 `node_nll_loss`

- 类别：训练项
- 职责：训练宏观 supporting prediction mechanism
- 允许为负：是
- 进入 `self_mismatch`：否

### 3.2 `macro_diag_score`

- 类别：诊断项
- 职责：表示宏观层的非负 rupture 证据
- 允许为负：否
- 进入 `self_mismatch`：是

### 3.3 `macro_uncertainty`

- 类别：辅助读出
- 职责：表示宏观预测不确定性
- 直接等同诊断强度：否

### 3.4 `micro_train_loss`

- 类别：训练项
- 职责：训练微观连续动态
- 允许经过 `free_nats` 与 `kl_cap`：是
- 进入 `self_mismatch`：否

### 3.5 `micro_diag_score`

- 类别：诊断项
- 职责：表示微观层的非负 rupture 证据
- 与 `micro_train_loss` 同名同义：否
- 进入 `self_mismatch`：是

### 3.6 `ground_diag`

- 类别：诊断项
- 职责：表示 grounding rupture 证据
- 数据来源：只允许来自 reality echo
- 进入 `self_mismatch`：是

### 3.7 `self_mismatch`

- 类别：诊断项
- 职责：统一摘要当前 self 与 reality 的裂缝强度
- 允许读取训练项：否
- 允许读取 `kappa_self`：是

### 3.8 `kappa_self`

- 类别：辅助读出
- 职责：表示 self coherence
- 可否进入 `self_mismatch`：是
- 是否允许被 `self_mismatch` 镜像监督：否

## 4. 当前冻结公式

### 4.1 宏观三分离

#### `node_nll_loss`

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

#### `macro_diag_score`

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
- 不含 `log \sigma^2`
- 只作诊断项

#### `macro_uncertainty`

\[
U_{macro}
=
\operatorname{mean}_d(\log \operatorname{stopgrad}(\sigma^2))
\]

说明：

- 只作不确定性读出
- 不直接等于诊断强度

### 4.2 微观训练与诊断双通道

#### `micro_train_loss`

\[
L_{\mu}^{train}
=
\operatorname{clip}(\max(\mathrm{KL}_{raw}, \tau_{free}), \tau_{cap})
\]

当前实现对应：

- `free_nats = 3.0`
- `micro_kl_cap = 20.0`

#### `micro_diag_score`

\[
C_{\mu}^{diag}
=
\log(1 + \max(\mathrm{KL}_{raw}, 0))
\]

说明：

- 这是 `v2-final` 中保留的工程 guardrail 定义
- 它不写成方法主创新
- 后续若更改，必须升 `v3`

### 4.3 统一诊断摘要

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

### 4.4 `bind_loss`

`v2-final` 中，`bind_loss` 不再直接使用 `exp(-self_mismatch)` 作为教师信号。

冻结为：

\[
y_{bind}
=
\operatorname{clip}
\left(
\exp\left(
- \frac{\delta_{control} + \delta_{boundary} + \delta_{progress}}{3}
\right),
0.05,
0.95
\right)
\]

\[
L_{bind}
=
\operatorname{BCE}(kappa\_self, y_{bind})
\]

说明：

- 它只依赖局部一致性目标
- 不再把 `kappa_self` 训练成 `self_mismatch` 的镜像量

## 5. 明确禁止事项

以下做法在 `v2-final` 中被明确禁止：

1. `self_mismatch` 读取 `node_nll_loss`
2. `self_mismatch` 读取 `micro_train_loss`
3. 任何 raw NLL / raw KL 训练项直接进入诊断聚合器
4. `RealityEncoder` 读取人工 `stuck / drift / proceed` 标签
5. 在论文或日志中重新使用含糊的 `node_loss`
6. 让 `self_clarity` 回到主路径和主结果表
7. 把 `recovery` 写成方法主体
8. 把 `phase2` 当作 `v2-final` 主结果组成部分

## 6. 当前实现分类

### 6.1 结构修正

这些内容可直接进入主线：

- 宏观三分离
- 诊断聚合器不读训练项
- `RealityEncoder` 只接收 reality echo
- 最小三模式行为接口

### 6.2 工程 guardrail

这些内容允许保留在代码中，但默认不写成方法创新：

- `micro_kl_cap`
- 非有限梯度清零
- grad clip
- 其他只为防爆炸而设的 clamp / warmup / cap

### 6.3 延后项

这些内容不属于 `v2-final` 主路径：

- `self_clarity`
- `ViabilityPredictor`
- `phase2`
- 复杂 router / mode projector

## 7. 后续允许改什么

在 `v2-final` 冻结后，后续只允许直接改：

- `lambda_*`
- 阈值
- 窗口大小
- warmup、cooldown、cap 的数值

这些不算版本升级。

## 8. 什么算版本升级

以下动作必须升 `v3`，不能混入 `v2-final` 结果：

- 修改任何核心公式
- 修改任何核心量职责
- 让训练项重新进入诊断聚合器
- 让 `self_clarity` 重新进入主路径
- 把最小行为接口扩成复杂 router 并改变主训练目标
- 让 `phase2` 进入主实验表

## 9. 与论文和代码的对应关系

本合同约束以下三处必须同名同义：

- [EFES_PAPER_DRAFT_CN.md](/home/D/liumeng/vln/260316P/etpnav-v5/docs/efes/EFES_PAPER_DRAFT_CN.md)
- [EFES_METHOD_CN.md](/home/D/liumeng/vln/260316P/etpnav-v5/docs/efes/EFES_METHOD_CN.md)
- `EFES_SELF` 代码与日志

不允许出现：

- 论文里叫诊断量，代码里其实是训练项
- 论文里叫 `self_mismatch`，日志里却混入 raw NLL
- 论文里写最小行为外化，代码里却让行为层重新主导训练
