# EFES：用于视觉语言导航的具身自我模型与分层自诊断

> 稿件状态：中文方法初稿  
> 版本口径：论文对外统一使用 `EFES` 作为方法名；当前实现对应内部版本 `EFES-v2`。  
> 写作原则：主线固定为 `self-model -> diagnosis -> recovery`，其中 recovery 只是自我模型有效性的外化体现，不是方法标题中心。  
> 写作公理：意识是内部在场；自我是这种内部在具身系统中的持续组织形式；EFES 研究的是这种 self 如何形成预期、发生破裂并通过修正维持连续性。本文不直接解决意识本体论，只研究其中可工程化的一层。  

## 摘要

现有视觉语言导航系统已经能够在局部候选选择、语言 grounding 和图规划上取得较强表现，但在长程连续执行中仍缺少一个持续的内部 self，因此难以及时识别“自己已经偏离预期”。从更宽的哲学视角看，意识首先可被理解为一种内部在场；而在具身导航系统中，这种内部在场最先表现为一个持续的 self 及其对自身与环境关系的追踪。本文关注的正是这一可工程化层面：系统是否能够持续追踪“我当前所处的任务阶段、我与环境的耦合状态、以及现实是否仍与我的内部预期一致”。已有方法通常沿两条路线尝试处理这一问题：一类依赖时间连续的 supporting prediction mechanism 或预测误差近似内部自检，另一类直接学习一个风险分数并触发启发式恢复。然而，前者往往难以突出由关键拓扑节点决定的结构性偏航，后者又缺少清晰的内部语义。本文提出 **EFES**，一个面向视觉语言导航的具身自我模型与分层自诊断框架。EFES 将 self state 定义为身体状态与环境状态持续耦合后在系统内部形成的可操作变化状态，并在微观时间尺度、宏观拓扑尺度与 progress grounding 三条通道上共同建模 self 与 reality 的裂缝。在宏观分支中，本文进一步将训练目标、自我裂缝证据与不确定性显式分离，并统一到压缩拓扑特征空间中，从而避免将训练损失直接误当作诊断量。最终，EFES 将微观失配、宏观失配与 grounding 信号统一为一个 self mismatch summary，并仅将恢复视为这种自我诊断的行为外化。需要强调的是，本文的核心不在于设计一个更复杂的恢复控制器，而在于构造一个更可信、更可解释的 self-model，使系统能够知道“自己哪里错了、错得有多严重”。本文的贡献包括：其一，提出一种基于身体—环境耦合的具身 self state；其二，提出由微观时序、宏观拓扑与进度接地共同构成的分层自诊断结构；其三，提出宏观训练目标、宏观裂缝证据和宏观不确定性的三分离设计，从而为偏航识别与恢复提供稳定的内部依据。

**关键词：** 视觉语言导航，具身自我模型，自我诊断，拓扑记忆，进度接地，偏航恢复

## 1. 引言

### 1.1 问题背景

视觉语言导航（Vision-and-Language Navigation, VLN）要求智能体根据自然语言指令在三维环境中持续执行导航动作。近年来，随着大规模预训练视觉语言表征、graph planner 和 waypoint predictor 的引入，系统在局部动作选择上已取得明显进展。然而，在长程连续执行场景下，系统仍然频繁出现一种更根本的失败：它并不是不会做局部选择，而是在偏离预期路径之后无法意识到“自己已经错了”。

这类失败的核心不在于某一帧图像是否难，而在于系统是否具有稳定的内部 self-model。一个具备 self-model 的导航系统，不仅要能编码当前观测，还要持续追踪以下关系：当前任务进度如何、自己处于怎样的拓扑阶段、身体动作是否真的带来了预期推进、以及现实观测是否仍与内部 belief 保持一致。缺少这层 self-model，系统即使局部评分正确，也容易在长走廊、重复房间或复杂转角处累积偏差，最终表现为循环、漂移、过早停止或错误回退。

### 1.2 现有方法的不足

现有方法应对这一问题通常有两种思路。第一类方法使用 supporting prediction mechanism、latent dynamics 或预测误差近似内部自检，希望通过 prior/posterior mismatch 或未来预测误差来判断当前状态是否失配。这类方法的优点在于有清晰的“内部预期 vs 外部现实”比较机制，但在 VLN-CE 中往往会受到时间连续冗余的干扰：走廊中的大量相似观测会稀释关键节点信息，使诊断量在真正需要恢复的地方不够稳定。第二类方法则直接学习一个风险分数，在分数较高时触发重规划、回退或停止。这类方法的问题是：风险值本身常常缺少结构性语义，它能告诉系统“有风险”，却很难告诉系统“为什么风险在这里出现”。

从方法论上看，这两类思路都没有真正把“self”放在中心。前者更像是时间连续预测器，后者更像是风险分类器。它们可以为恢复提供某种触发条件，却不足以构成一个可解释的 self-model。

### 1.3 哲学边界与工程对象

从更宽的角度看，意识首先意味着一种内部在场；而在具身系统中，这种内部在场最先表现为一个持续的 self。本文不试图解决 consciousness 的完整本体，也不主张已经回答了 Hard Problem。本文只研究其中最可工程化的一层：**rupture-aware embodied self**。换言之，EFES 关心的不是“内部在场为何存在”的终极问题，而是“当一个系统已经以 self 的形式组织自身时，它如何形成预期、如何在与现实的碰撞中发生破裂，并如何通过修正重新缝合这种连续性”。

### 1.4 本文思路

本文提出 **EFES**，其出发点不是再叠加一个异常分数，而是把导航系统的 self-model 显式建出来。我们将 self 定义为：**身体状态与环境状态持续耦合后，在系统内部形成的可操作变化状态**。基于这一观点，EFES 的方法主线由三层组成：

1. **具身 self state**：用身体 token、环境 token、历史 self token 和 instruction token 共同构成内部 self representation；
2. **分层自诊断**：在微观时间尺度、宏观拓扑尺度和 progress grounding 三条通道上同时检测 self 与现实的失配；
3. **恢复外化**：把分层自诊断作为恢复行为的内部依据，用恢复成功与否来验证 self-model 是否可信。

这里 recovery 是必要的，但它在本文中不是主体。主体是 self-model 本身。恢复只是 self-model 是否有效的行为外化结果。基于这一设计，本文的贡献可概括为三点：

1. 提出一种基于身体—环境耦合的具身 self state，使自我状态不再只是单一 latent，而是同时吸纳身体状态、环境状态与历史 belief。
2. 提出一种分层自诊断框架，在微观时序、宏观拓扑和进度接地三条通道上共同度量 self 与现实之间的失配。
3. 提出一种宏观三分离设计，将宏观训练目标、宏观异常证据和宏观不确定性彻底拆开，从而避免训练目标与诊断语义混用，并为恢复提供稳定依据。

## 2. 相关工作

### 2.1 视觉语言导航与具身决策

视觉语言导航方法长期围绕语言 grounding、视觉编码、候选动作排序和图规划展开。近期系统尤其擅长利用全景特征、graph memory 和 waypoint candidates 构建强大的局部 planner。在此基础上，很多工作已经能实现较高的单步决策精度。

但本文关注的问题与单步候选排序不同。本文不试图替代 planner，而是关注：当 planner 的内部 belief 与现实逐渐分离时，系统是否拥有稳定的 self-model 来检测这一点。因此，EFES 的定位不是新的 graph planner，而是 planner 之上的 self-model layer。

### 2.2 Supporting Prediction Mechanisms 与自我监控

各类 supporting prediction mechanism、latent dynamics 与预测误差长期被用于强化学习和具身智能中的状态建模与不确定性估计。其核心思想是：如果内部模型能够预测未来，那么现实对内部预测的修正本身就能成为自我监控信号。KL 散度、重建误差和未来误差都属于这一范式。

本文继承了这一思想，但不直接把训练目标当作诊断量。特别是在宏观分支中，我们强调训练损失、自我裂缝证据和不确定性必须显式分离，因为训练 objective 的数值性质并不天然等价于可解释的 self mismatch summary。

### 2.3 恢复、回退与层级控制

在导航和机器人系统中，回退、候选扩展、重规划和停止都是常见恢复行为。部分工作会显式设计多级恢复逻辑，根据当前分数或启发式条件切换行为模式。这些工作说明：恢复能力本身是长程任务成功的重要组成部分。

本文与这些工作的差异在于：我们不把 recovery 当成方法主体，也不把它视为独立 heuristic 模块。EFES 中的 recovery 只是分层自诊断的行为外化结果。换言之，本文真正关心的是系统如何“知道自己错了”，而不是单独研究“恢复器怎么写”。

## 3. 方法

### 3.1 EFES 的研究层次

为了避免把方法写成恢复控制器、异常打分器或单纯的预测器，本文首先明确区分三个层次：

1. **Consciousness**：内部在场。这是本文的哲学起点，但不是直接工程对象。
2. **Self**：内部在具身系统中的持续组织形式，是本文方法的核心语义层。
3. **Rupture-aware Self**：当 self 与 reality 发生裂缝并被迫修正时的工作态，是 EFES 的直接研究对象。

因此，本文不主张构造 consciousness 的完整模型，也不把 recovery 视为主体。EFES 的任务是构造并分析一个具身 self 如何形成预期、如何在 micro / macro / grounding 三条通道上发生裂缝、以及如何在裂缝后维持连续性。

### 3.2 问题定义

给定语言指令 $I$、当前观测 $\boldsymbol{x}_t$、图结构候选集合以及环境拓扑状态，智能体需要持续输出下一步导航行为。本文假设底层导航 backbone 仍由 ETP 风格策略负责，它输出：

- instruction tokens
- 当前全景观测特征
- 当前节点特征
- candidate logits
- 图结构与 frontier 信息

EFES 作为上层 self-model，学习回答三个问题：

1. 当前 self 是否仍与现实保持一致；
2. 若不一致，这种失配来自微观时间尺度、宏观拓扑尺度还是进度接地失衡；
3. 这种失配是否足以支持恢复行为。

### 3.3 总体框架

EFES 的总体主线固定为：

`Embodied Self State -> Micro/Macro Diagnosis -> Grounding Check -> Unified Diagnosis -> Recovery Externalization`

图 1 应突出以下三个创新点：

1. **Self Core**：`EmbodiedSelfState`
2. **Diagnosis Core**：`Micro + Macro + Grounding` 的分层自诊断
3. **Behavioral Consequence**：恢复仅作为下游行为外化

图 1 中，`Recovery` 必须位于最右侧，作为 downstream behavior；任何 router / projector 都不得成为图中的视觉中心。

### 3.4 具身 Self State

EFES 的第一步是构造具身 self state。与只依赖上一时刻 hidden state 或 latent posterior 的做法不同，我们显式引入三类 token：

1. **body tokens**
   - 上一动作 embedding
   - prior 调制因子
   - revisit count
   - loop evidence

2. **environment tokens**
   - 当前观测 summary
   - frontier size
   - topo novelty
   - progress gap

3. **history/instruction tokens**
   - 上一 self state
   - 上一 latent posterior
   - instruction tokens

设 body token 表示为 $\boldsymbol{b}_t$，environment token 表示为 $\boldsymbol{e}_t$，历史 self state 为 $\boldsymbol{s}_{t-1}$，则当前 self state 定义为：

\[
\boldsymbol{s}_t = f_{\mathrm{self}}(\boldsymbol{b}_t, \boldsymbol{e}_t, \boldsymbol{s}_{t-1}, I)\,.
\]

在实现上，`EmbodiedSelfState` 使用以 $\boldsymbol{s}_{t-1}$ 为 query、以 body/environment/history token 为 key/value 的 cross-attention，并输出：

- 当前 self state `s_t`
- progress 估计 `progress_t`
- attention entropy `attn_entropy`

这一步的目的不是仅仅产生一个新的隐变量，而是让 self representation 显式携带身体—环境耦合证据。

### 3.5 微观自诊断：Micro RSSM

微观分支保留时间连续的 RSSM 结构，用于检测局部时间尺度上的 internal mismatch。设当前确定性隐状态为 $\boldsymbol{h}_t$，prior/posterior 分别为：

\[
p(\boldsymbol{z}_t \mid \boldsymbol{h}_t)=\mathcal{N}(\boldsymbol{\mu}_t^{pri},{\boldsymbol{\sigma}_t^{pri}}^2)\,,
\]

\[
q(\boldsymbol{z}_t \mid \boldsymbol{h}_t,\boldsymbol{x}_t)=\mathcal{N}(\boldsymbol{\mu}_t^{post},{\boldsymbol{\sigma}_t^{post}}^2)\,.
\]

微观诊断量定义为：

\[
C_{\mu,t}^{diag}
=
\mathrm{KL}\big(q(\boldsymbol{z}_t)\,\|\,p(\boldsymbol{z}_t)\big)\,.
\]

训练时仍使用 free-nats 约束：

\[
L_{\mu}^{kl} = \max(C_{\mu,t}^{diag}, \tau_{free})\,,
\]

其中 $\tau_{free}=3.0$。这一分支主要回答：现实观测在当前时间尺度上是否显著修正了系统对自身状态的预测。

### 3.6 宏观自诊断：拓扑层与三分离设计

宏观分支承担拓扑尺度上的 self-model 检查。首先，将当前节点特征压缩到统一空间：

\[
\tilde{\boldsymbol{x}}_t = f_{\mathrm{comp}}(\boldsymbol{x}^{node}_t)\,.
\]

`TopoStateBank` 存储的也是 $\tilde{\boldsymbol{x}}_t$。因此：

- bank 存 compressed feat
- retrieval 用 compressed feat
- novelty 也在 compressed feat 空间计算

给定从 `TopoStateBank` 检索得到的拓扑上下文 $\boldsymbol{c}_t^{topo}$，宏观预测头输出：

\[
\boldsymbol{\mu}_t^{pred},\;{\boldsymbol{\sigma}_t^{pred}}^2
=
f_{\mathrm{macro}}(\boldsymbol{c}_t^{topo})\,.
\]

随后，我们显式定义三类量。

**宏观训练损失：**

\[
L_{node}^{nll}
=
\frac{1}{2}\operatorname{mean}_d
\left(
\frac{(\tilde{\boldsymbol{x}}_t^{sg}-\boldsymbol{\mu}_t^{pred})^2}
{{\boldsymbol{\sigma}_t^{pred}}^2}

+ \log {\boldsymbol{\sigma}_t^{pred}}^2
\right)\,.
\]

**宏观诊断量：**

\[
C_{macro,t}^{diag}
=
\operatorname{mean}_d
\left(
\frac{(\tilde{\boldsymbol{x}}_t^{sg}-{\boldsymbol{\mu}_t^{pred}}^{sg})^2}
{\operatorname{clip}(\operatorname{stopgrad}({\boldsymbol{\sigma}_t^{pred}}^2), s_{min}^{diag}, s_{max}^{diag})}
\right)\,.
\]

**宏观不确定性：**

\[
U_{macro,t}
=
\operatorname{mean}_d
\left(\log {\boldsymbol{\sigma}_t^{pred}}^{2,sg}\right)\,.
\]

这里的关键不是公式本身，而是三者的语义分离：

- `node_nll_loss` 只用于训练
- `c_macro_diag` 只用于诊断
- `u_macro` 只用于表示模型对宏观预测的不确定程度

这一设计直接修正了旧版中把异方差 NLL 同时当作训练损失和宏观异常量的语义混用问题。

### 3.7 TopoStateBank 与稀疏宏观触发

为了避免宏观分支退化成逐步异常检测器，`TopoStateBank` 的更新必须是稀疏的。我们构造 `build_update_mask()`，同时考虑：

1. bank 是否为空；
2. 当前 `cur_vp` 是否变化；
3. 当前候选数是否达到决策阈值；
4. 是否超时未更新；
5. 当前 `topo_novelty` 是否超过阈值；
6. 是否满足 cooldown 约束。

其中：

- bank empty 和 `cur_vp changed` 强制触发；
- `candidate_count`、`timeout` 和 `topo_novelty` 仅在 cooldown 满足时触发。

这样，macro 分支只在关键节点、决策点和长时间未推进的兜底点激活，而不是几乎每步都激活。

### 3.8 Progress Grounding

仅有 micro 和 macro 两条分支还不足以完整表达自我失配。EFES 还引入 progress grounding，用于检测“我以为自己在推进，但身体实际上并没有真正前进”的耦合脱节。

设最近 $K$ 步的 progress 变化为 $\Delta p_t$，路径长度累计为 $\Delta l_t$，拓扑唯一节点增量为 $\Delta v_t$，则 grounding 信号可以写成：

\[
g_t
=
\Delta p_t \cdot \mathbb{1}\left[\Delta l_t < \epsilon \;\land\; \Delta v_t = 0\right]\,.
\]

这一步的意义在于：它不是一般性的几何约束，而是 self-model 的第三条诊断通道，用来检测 self 的内部进度感是否真正接地到身体推进。

### 3.9 统一自诊断量

EFES 使用一个冻结统计的异常监视器，将三条诊断通道统一起来：

- `c_micro_diag`
- `c_macro_diag`
- `g_t`

标准化后得到：

\[
z_{\mu,t},\quad z_{M,t},\quad z_{g,t}\,,
\]

最终统一诊断量定义为：

\[
A_t = \max(z_{\mu,t}, z_{M,t}, z_{g,t})\,.
\]

当 `macro_valid=False` 时，强制令 $z_{M,t}=-\infty$，从而确保宏观未触发步不会污染异常量。该监视器只在训练模式下更新 running stats，在评估与推理时完全冻结。

### 3.10 Recovery 的定位

EFES 中的 recovery 不是方法主体，而是自我模型有效性的行为外化。系统先通过 `A_t` 形成统一自诊断，再据此选择恢复行为。这里最重要的不是恢复器写得多复杂，而是恢复是否由 self-model 提供了可信依据。

因此，本文在叙事上只把恢复放在一个从属位置：

- 若 self-model 可信，则 recovery 更应成功；
- 若 self-model 不可信，则 recovery 只是无意义的模式切换。

换言之，恢复的作用是证明 self-model，而不是取代 self-model。

### 3.11 训练目标

EFES 的总训练目标为：

\[
L
=
L_{plan}
+
\lambda_{kl}L_{\mu}^{kl}
+
\lambda_{node}L_{node}^{nll}
+
\lambda_{\pi}L_{\pi}^{cal}\,.
\]

其中：

- $L_{plan}$：导航主目标
- $L_{\mu}^{kl}$：微观时序失配建模
- $L_{node}^{nll}$：宏观预测训练目标
- $L_{\pi}^{cal}$：self clarity 近未来校准目标

需要强调的是，`C_macro_diag` 不再直接等于 `L_{node}^{nll}`，二者语义分离是本文方法成立的基础。

## 4. 实验设计

### 4.1 目标

本文实验关注的不是某个单独 loss 是否下降，而是验证：

1. 具身 self state 是否提供了更稳定的 self representation；
2. 分层自诊断是否比单一时间连续异常量更可信；
3. 宏观三分离是否改善了宏观诊断语义；
4. 自我模型是否能更稳定地支持恢复行为。

### 4.2 数据与设置

首版实验固定在 R2R 上进行，不在主文叙事中同时展开 RxR。训练配置使用：

- `run_r2r/efes_v2/efes_v2_main.yaml`

### 4.3 主结果表

**表 1：R2R 主结果。**

| 方法 | Success | SPL | nDTW | sDTW | 备注 |
| --- | --- | --- | --- | --- | --- |
| ETP backbone | 待填 | 待填 | 待填 | 待填 | 无 self-model 层 |
| 旧 EFES | 待填 | 待填 | 待填 | 待填 | 早期原型 |
| EFES | 待填 | 待填 | 待填 | 待填 | 具身 self-model + 分层诊断 |

这里第三列的主语是 `EFES`，不是 router，也不是 recovery policy。

### 4.4 诊断相关指标

建议固定报告以下诊断量：

- `c_micro_mean`
- `c_macro_diag_mean`
- `u_macro_mean`
- `g_t_mean`
- `a_t_mean`
- `pi_t_mean`
- `macro_valid_ratio`

它们分别对应：

- 微观时序失配
- 宏观拓扑失配
- 宏观不确定性
- grounding 脱节
- self mismatch summary
- self clarity
- 宏观触发稀疏性

### 4.5 可视化分析

应固定三类图：

1. **Self-diagnosis 曲线图**  
   展示 `c_micro_diag`、`c_macro_diag` 与 `A_t` 随时间的变化。

2. **Progress grounding 图**  
   展示 progress 变化与真实空间推进不一致时 `g_t` 的响应。

3. **恢复案例图**  
   用于说明：当自我模型更可信时，系统能否更早识别偏航并更稳地回到正确路径。

### 4.6 成功与失败案例

成功案例应回答：

- 系统何时察觉自己偏航；
- 哪一条诊断通道先升高；
- 恢复后是否回到正确轨迹。

失败案例应回答：

- 是 micro 失配不敏感；
- 还是 macro 触发不准；
- 还是 progress grounding 未能反映真实脱节；
- 还是虽然诊断到了，但恢复仍不足。

这类分析的目的不是展示恢复器，而是展示 self-model 的边界。

## 5. 讨论

### 5.1 为什么本文不是恢复控制论文

虽然本文中 recovery 是必要模块，但 recovery 在本文中并不是标题主语。本文的核心不在于设计一种更复杂的 mode router，而在于构造一个可信的 self-model，使系统能够形成可解释的内部诊断证据。恢复只是这一内部证据的行为外化。

### 5.2 为什么本文不是单纯的 supporting prediction 论文

本文也不是单纯比较预测误差。EFES 的宏观分支并不以“预测得多准”为最终目的，而是以“诊断量是否有清晰语义”为核心。因此，我们强调宏观训练目标、宏观裂缝证据和宏观不确定性的三分离，而不是把 supporting prediction objective 直接当成自我裂缝证据。

### 5.3 局限性

当前版本仍有三个边界：

1. 首版实验集中在 R2R，尚未系统验证更复杂语言分布上的泛化；
2. 宏观触发仍依赖当前拓扑节点定义质量，因此 waypoint 或 graph 边界不稳定时会影响宏观自诊断；
3. 恢复行为仍主要用于验证 self-model，而尚未展开更完整的 failure attribution 学习。

这些局限不会改变本文的中心：本文首先是一篇关于 self-model 与 self-diagnosis 的工作，而不是一篇关于恢复器调参的工作。

### 5.4 本文不主张什么

为了避免概念过度包装，本文显式说明三点：

1. 本文不主张已经解决 Hard Problem，也不主张已经解释了 consciousness 为何存在。
2. 本文不主张 EFES 是完整的 consciousness 模型；本文只研究其中可工程化的一层，即 rupture-aware embodied self。
3. 本文不主张 recovery 是方法主体；恢复在本文中只是 self-diagnosis 是否有效的行为外化。

## 6. 结论

本文提出 EFES，一个面向视觉语言导航的具身自我模型与分层自诊断框架。本文的哲学起点是：意识首先意味着一种内部在场；而在具身导航系统中，这种内部在场最先表现为一个持续的 self。EFES 的工程对象并不是 consciousness 的完整本体，而是其中最可工程化的一层：一个具身 self 如何形成预期，并在微观时序、宏观拓扑与进度接地三个层次上持续检测自己与 reality 之间的裂缝。基于这一设计，本文进一步提出宏观训练目标、宏观裂缝证据和宏观不确定性的三分离结构，从而为偏航识别提供更稳定的内部依据。恢复行为在本文中不是中心，而是 self-model 是否有效的外化结果。后续工作将围绕大规模实验、失败归因以及更复杂任务中的自我模型泛化展开。

## 附录 A：图表与待填内容

### A.1 必备图表

- 图 1：EFES 方法总览图
- 图 2：micro / macro / grounding 三通道诊断曲线
- 图 3：偏航后的恢复案例图
- 表 1：主结果表
- 表 2：诊断与消融结果表

### A.2 待补证据

- R2R 主结果数值
- 分层自诊断消融结果
- `c_macro_diag` 与 `node_nll_loss` 分离后的统计图
- 成功恢复案例与失败案例各至少一组
