# EFES 方法说明（EFES_SELF v2-final）

> 适用实现：`EFES_SELF`
> 方法状态：`v2-final`
> 对外方法名：`EFES`
> 内部实现名：`EFES_SELF`
> 约束来源：`EFES_SELF_LOSS_DIAG_CONTRACT_V2_FINAL_CN.md`

## 1. 方法边界

`EFES` 的方法边界固定为三层：

1. `Consciousness`
   只作为哲学起点，定义为内部在场，不是直接工程对象。
2. `Self`
   定义为内部在具身系统中的持续组织形式，是方法的语义中心。
3. `Rupture-aware Self`
   定义为当 self 与 reality 发生裂缝并被迫修正时的工作态，是 `EFES` 的直接研究对象。

因此，`EFES` 明确不主张：

- 已经解决 Hard Problem
- 已经构造完整 consciousness 模型
- `recovery` 是方法主体

`EFES` 的工程目标只有一个：构造一个具身 self，并研究它如何在微观、宏观和接地三条证据链上显露 rupture，以及这些 rupture 如何外化为最小行为后果。

## 2. 最终版方法骨架

`EFES_SELF v2-final` 只保留三块主结构：

1. **Embodied Self State**
2. **Multi-scale Diagnosis**
3. **Behavioral Consequence**

这三块之外的训练护栏、阈值、warmup、cap、版本治理，全部视为工程支撑层，不进入方法主体。

### 2.1 Embodied Self State

最终版只保留一个 Self Core 模块：`SelfBinder`。

`SelfBinder` 输入三类槽位：

- `body_slot`
  - 本体感受读数
  - 上一步动作传出副本
  - 底层碰撞/打滑信号
- `intent_slot`
  - 内化后的局部目标表示
  - 意图张力
- `trace_slot`
  - 近期 `z_self`
  - 近期 rupture residual

`SelfBinder` 不允许退化为 `concat -> MLP` 的普通状态拼接器。当前实现采用能量绑定，只输出：

- `z_self`
- `kappa_self`
- `progress_t`

其中：

- `z_self` 表示当前 self 的组织读出
- `kappa_self` 表示 self coherence
- `progress_t` 表示 self 在当前任务阶段上的内部推进读出

当前世界特征不再直接并入 `body_slot`。Self Core 负责组织 self，而不是直接重建或混合 world state。

### 2.2 Multi-scale Diagnosis

最终版诊断层只保留三条真实证据链：

#### Micro

`MicroRSSM` 保留，但训练项与诊断项强制分离：

- `micro_train_loss`
- `micro_diag_score`

前者只负责训练连续动态，后者只负责微观 rupture 证据。

#### Macro

宏观分支保留压缩拓扑空间，但也只保留三分离：

- `node_nll_loss`
- `macro_diag_score`
- `macro_uncertainty`

其中：

- `node_nll_loss` 只作训练项
- `macro_diag_score` 是唯一宏观诊断量
- `macro_uncertainty` 只作不确定性读出

#### Grounding

`RealityEncoder` 在最终版中只编码 reality echo，不再读取 self 输出。

grounding 只基于现实证据构造：

- 实际位移/转向结果
- 拓扑节点变化
- frontier 或路径推进证据

`ground_diag` 只来自现实与历史参考的比较，不得混入 self 侧读出。

#### Unified Diagnosis

`SelfMismatchAggregator` 只聚合：

- `micro_diag_score`
- `macro_diag_score`
- `ground_diag`
- 可选 `kappa_self` 的 self penalty

它输出统一诊断摘要：

- `self_mismatch`
- `delta_control`
- `delta_boundary`
- `delta_progress`

这三类 rupture 用于解释 self 在控制连续性、边界维持和 progress grounding 上分别在哪里裂开。

### 2.3 Behavioral Consequence

最终版行为层只保留最小三模式：

- `PROCEED`
- `BELIEF_UPDATE`
- `RECOVER`

这三模式只作为下游验证接口，不作为主创新点。

模式决策直接基于三类 rupture：

- `delta_control`
- `delta_boundary`
- `delta_progress`

而不再依赖复杂 router、mode projector 或 `self_clarity`。

行为外化规则保持最小化：

- `PROCEED`
  - mismatch 低，且无主导 rupture
- `BELIEF_UPDATE`
  - mismatch 中等，主导 rupture 为 `delta_control` 或 `delta_progress`
- `RECOVER`
  - mismatch 高，或 `delta_boundary` 主导且存在可用历史回退点

## 3. 不进入主方法的内容

以下内容允许存在于代码中，但默认不进入主方法和主实验表：

- `ViabilityPredictor`
- `self_clarity`
- 复杂 mode 逻辑
- `phase2`
- `router boot loss`
- 训练护栏本身

这些内容若保留，只能作为实验性分支或后续 ablation。

## 4. 训练与版本纪律

`v2-final` 的主实验只依赖 `phase1`：

- backbone 冻结
- 只训练 `Self Core + Diagnosis Core + Minimal Behavior`
- `phase2` 默认关闭，不进入主结果

主损失固定为：

- `L_plan`
- `L_bind`
- `L_micro_train`
- `L_node_nll`

以下内容只允许作为工程 guardrail 保留：

- `micro_kl_cap`
- 非有限梯度清零
- grad clip

它们必须在文档中被明确标记为 guardrail，不得写成方法贡献。

所有运行产物必须记录：

- `contract_version = v2-final`

## 5. 文档关系

当前 `EFES` 相关文档按以下方式使用：

- [EFES_PAPER_DRAFT_CN.md](/home/D/liumeng/vln/260316P/etpnav-v5/docs/efes/EFES_PAPER_DRAFT_CN.md)
  - 论文主稿
- [EFES_METHOD_CN.md](/home/D/liumeng/vln/260316P/etpnav-v5/docs/efes/EFES_METHOD_CN.md)
  - 方法边界与最终版实现口径说明
- [EFES_SELF_LOSS_DIAG_CONTRACT_V2_FINAL_CN.md](/home/D/liumeng/vln/260316P/etpnav-v5/docs/efes/EFES_SELF_LOSS_DIAG_CONTRACT_V2_FINAL_CN.md)
  - `v2-final` 冻结合同

后续任何实现变更，都应先对应到冻结合同，再进入代码实现。
