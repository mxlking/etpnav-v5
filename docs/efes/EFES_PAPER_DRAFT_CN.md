# EFES：面向连续视觉语言导航的破裂感知具身自我修正

> 稿件状态：中文主稿重写版
> 适用范围：R2R 主线，当前结果为 pilot 级别，完整消融与更长训练结果待补
> 方法命名：对外统一使用 `EFES`
> 写作原则：主线固定为 `self constitution -> consequence expectation -> typed rupture attribution -> self revision -> safe action conditioning`
> 写作公理：意识是内部在场；自我是这种内部在具身系统中的持续组织形式；EFES 研究的是这种 self 如何形成预期、发生破裂并通过修正维持连续性。本文不直接解决意识本体论，只研究其中可工程化的一层。
> 当前实现锚点：训练与评估入口为 [efes.bash](/home/D/liumeng/vln/260316P/etpnav-v5/run_r2r/efes/efes.bash)，配置文件为 [efes_main.yaml](/home/D/liumeng/vln/260316P/etpnav-v5/run_r2r/efes/efes_main.yaml)。

## 摘要

现有连续视觉语言导航系统已经具备较强的局部候选选择与拓扑规划能力，但在长程执行中仍缺少一个持续的内部 self，因此难以及时识别“自己已经偏离预期”。从哲学起点上说，意识可以被理解为一种内部在场；但本文不试图解决 consciousness 的本体论问题，而只研究其在具身导航中最可工程化的一层：一个行动者如何维持自身的动作后果、拓扑推进与任务连续性。本文提出 **EFES**，一个面向 VLN-CE 的破裂感知具身自我修正框架。EFES 不把失败简单归结为规划误差、异常分数或世界预测误差，而是将其建模为 action-conditioned self continuity 的可分型破裂。具体而言，EFES 首先形成 self prior，预测由当前 self 与动作应当带来的局部、拓扑与推进后果；随后将这些 consequence expectation 与现实回波比较，得到 local rupture、topological rupture 和 grounding rupture；最后根据 rupture evidence 形成 revised self，并通过安全 residual conditioner 约束 ETPNav planner 的动作 logits。当前 R2R pilot 结果显示，在冻结 ETPNav backbone 的设定下，EFES-active-safe 相比 ETPNav baseline 取得了可比的 Success，并在 SPL、nDTW 和 sDTW 上出现小幅改善；但 Success 尚未全面超过 baseline，完整结论仍需要更长训练、完整 full eval 与消融实验确认。本文的贡献在于：第一，提出 action-conditioned embodied self continuity 作为 VLN 中可研究的新对象；第二，提出 typed rupture attribution，将失败证据拆分为局部、拓扑和 grounding 三类破裂；第三，提出 self revision loop，使行为调制基于 revised self，而不是直接由异常分数触发恢复。

**关键词：** 视觉语言导航，具身自我模型，破裂归因，自我修正，安全动作调制

## 1. 引言

### 1.1 现有 VLN 系统缺少持续 self

视觉语言导航（Vision-and-Language Navigation, VLN）要求智能体根据自然语言指令，在三维环境中持续执行动作并最终到达目标位置。近年来，基于 transformer、拓扑图、候选 waypoint 和 cross-modal planner 的系统显著增强了 VLN-CE 中的局部感知和路径选择能力。以 ETPNav 为代表的系统已经能够在线构建拓扑结构、组织候选视点并进行 obstacle-aware control。

但是，长程导航的关键失败往往不只是“某一步选错了候选”。系统在重复走廊、相似房间、复杂转角或局部回环中，可能仍然保持较高的局部打分，却逐步失去任务连续性。它不知道自己是否仍处在正确的任务阶段，不知道上一动作是否真的带来了预期推进，也不知道当前拓扑处境是否仍与内部预期一致。换言之，现有系统缺少一个持续 self，因而缺少真正的 self-diagnosis。

### 1.2 从内部在场到具身 self

从更宽的角度看，consciousness 可以作为一个哲学起点：它意味着系统存在某种内部在场，即“从内部看是怎样的”。本文不把这一点当作可直接编程的对象，也不主张解决 Hard Problem。在工程上，我们只取其中最可约束的一层：在具身系统中，内部在场首先表现为 self 的持续组织形式。

这里的 self 不是一个普通 latent，也不是记忆池或 UI 标签。它表示 agent 对“我是否仍是有效行动者、我处在任务什么阶段、我与拓扑和身体推进是否仍一致、我的动作本应产生什么后果”的持续组织。这个定义使 self 既保留哲学上的主体性来源，又能落到可训练、可评估、可消融的工程对象上。

### 1.3 EFES 研究 rupture-aware embodied self

本文关注的是 `rupture-aware embodied self revision`。所谓 rupture，并不是一般意义上的 anomaly，也不是训练损失的别名。它表示现实迫使 self 承认自身连续性出现裂缝。VLN 中这种裂缝至少包含三类：局部动作后果与现实观测不一致，拓扑阶段推进与节点/新颖性变化不一致，以及系统以为自己在推进但身体/路径/拓扑没有真实推进。

因此，EFES 的目标不是构建一个更强 world model，也不是在 planner 后面加一个 recovery controller。EFES 的目标是先形成 self prior，再预测当前 self 与动作应当产生的后果，再对现实回波进行 typed rupture attribution，接着得到 revised self，最后用 revised self 安全约束后续动作。

### 1.4 本文贡献

本文贡献总结如下：

1. **最小具身 self 的工程定义。** 我们将 self 定义为 action-conditioned continuity state，用来组织 agency、phase、situated continuity 与 action consequence expectation。
2. **Typed rupture attribution。** 我们将 VLN 失败证据拆分为 local、topological 和 grounding 三类 rupture，使自我失配不再退化成单一风险分数。
3. **Self revision loop。** 我们提出 `self prior -> rupture -> self posterior -> action conditioning` 的闭环，使行为使用 revised self，而不是由异常分数直接触发启发式恢复。
4. **Safe action conditioning。** 在冻结 ETPNav backbone 的第一版实现中，EFES 通过安全 residual 方式影响 planner logits，从而避免把收益混入 backbone finetuning。

## 2. 相关工作

### 2.1 VLN-CE 与拓扑规划

连续视觉语言导航研究关注如何在真实感三维环境中，根据自然语言指令完成长程导航。已有方法围绕语言 grounding、候选 waypoint、图结构规划、视觉编码与局部控制展开。ETPNav 这类方法的优势在于：它们能够在线维护拓扑图、进行 cross-modal planner 打分，并通过底层控制器执行导航动作。

EFES 与这类工作的关系是插件式而非替代式。本文不重写 ETPNav planner，不改变其候选构造、拓扑规划和底层控制逻辑。ETPNav 回答的是“如何规划和行动”；EFES 回答的是“执行这些规划时，我这个行动者是否仍保持动作后果、拓扑推进与任务连续性”。因此，EFES 是 frozen planner 之上的 self revision layer，而不是新的 topology planner。

### 2.2 World Model 与 supporting prediction mechanism

World-model 方法通常学习环境动力学、未来观测、未来状态或可控视觉生成，用于 imagined planning、foresight reasoning 或 trajectory scoring。这类方法回答的问题是：给定当前观测与动作，世界接下来会怎样。

EFES 使用预测机制，但研究对象不同。EFES 不追求成为环境的充分模拟器，也不进行 imagined rollout planning。它预测的是与 self 相关的 action-conditioned consequence：如果当前是“我”执行了这一步动作，本应看到怎样的局部后果、拓扑变化和推进信号。预测误差在这里不是 world-model planning 的终点，而是 self rupture attribution 的材料。判断标准很直接：如果一个模块拿掉 self prior / self posterior 后仍能单独用于 imagined planning，它更像 world model；如果它离开 self revision loop 就失去意义，它才是 EFES 所需的 self-model substrate。

### 2.3 Self-monitoring、progress estimation 与 uncertainty trigger

Self-monitoring、progress estimation 和 uncertainty-triggered reasoning 为 VLN 提供了重要基础。它们说明 agent 需要评估任务推进、内部不确定性或风险状态。但这类方法通常输出 progress score、uncertainty scalar 或 trigger signal，难以解释“裂缝来自哪里”，也不一定形成 revised self。

EFES 的区别在于：它不把 progress 或 uncertainty 当作最终对象，而是把它们纳入 typed rupture evidence。local rupture、topological rupture 与 grounding rupture 分别对应不同层面的 self continuity 裂缝，最终进入 self revision。没有 self revision 的方法最多是 monitoring；EFES 的核心是 revised self 继续参与后续动作选择。

## 3. 方法

### 3.1 研究层次与总体流程

本文明确区分三个层次：

1. **Consciousness**：内部在场，是本文哲学起点，不是直接工程对象。
2. **Self**：内部在具身系统中的持续组织形式，是方法语义层。
3. **Rupture-aware Self**：self 与 reality 发生裂缝并被迫修正时的工作态，是 EFES 的直接工程对象。

在该边界下，EFES 的总体流程为：

\[
S_t^- \rightarrow \hat{C}_t \rightarrow R_t \rightarrow S_t^+ \rightarrow a_t
\]

其中 $S_t^-$ 表示 self prior，$\hat{C}_t$ 表示 consequence expectation，$R_t$ 表示 typed rupture evidence，$S_t^+$ 表示 revised self，$a_t$ 表示由 revised self 约束后的动作选择。展开为工程流程：

`self constitution -> consequence expectation -> typed rupture attribution -> self revision -> safe action conditioning`

### 3.2 Self Prior / Self Constitution

EFES 首先构造 self prior $S_t^-$。它不是普通 hidden state，而是包含以下信息的结构化状态：

- `z_self`：高维 self representation
- `agency`：当前是否仍是有效行动者
- `phase`：当前任务阶段或推进阶段
- `continuity`：self 与环境、拓扑、动作后果的连续性
- `rupture_memory`：近期 rupture 摘要

Self constitution 的输入来自上一时刻 revised self、上一动作嵌入、instruction summary、planner context、当前节点特征、拓扑上下文与轨迹统计量。形式上，

\[
S_t^- = f_{\mathrm{self}}(S_{t-1}^+, a_{t-1}, I, c_t^{plan}, x_t^{node}, c_t^{topo}, \tau_t)
\]

这里 $f_{\mathrm{self}}$ 可以由 gated fusion 或 cross-attention 实现。关键不是具体选择哪种融合器，而是 `Self` 必须先于 rupture 出现。也就是说，系统先形成“我当前是谁、处于什么阶段、预期自己应如何推进”的 prior，再让现实回波去挑战这个 prior。

### 3.3 Consequence Expectation

EFES 的预测对象不是完整世界图像，也不是所有未来状态，而是 self-relevant consequence。给定 $S_t^-$ 与上一动作 $a_{t-1}$，consequence model 输出三类预期：

\[
\hat{C}_t =
\{\hat{C}_t^{local}, \hat{C}_t^{topo}, \hat{C}_t^{ground}\}
\]

其中：

- $\hat{C}_t^{local}$ 预测当前压缩 node/observation feature 应该落在哪个范围内。
- $\hat{C}_t^{topo}$ 预测视点变化、拓扑新颖性与 frontier 变化。
- $\hat{C}_t^{ground}$ 预测这一步动作应当带来多少真实推进。

这个设计避免 EFES 滑向 world model。我们不是问“世界下一步完整长什么样”，而是问“如果当前 self 是正确的，并且上一动作确实由我执行，现实应当怎样回应我”。

### 3.4 Typed Rupture Attribution

现实回波 $G_t$ 不预先携带 `stuck`、`drift` 或 `wrong-way` 这样的诊断标签。它只包含当前可在线获得的现实摘要，例如当前 node feature、视点是否变化、拓扑新颖性、frontier size、路径长度增量、唯一视点增量、revisit/loop evidence 与 progress proxy。

EFES 将 $\hat{C}_t$ 与 $G_t$ 比较，得到三类 rupture：

**Local rupture.**

\[
r_t^{local}
= d(\hat{x}_t^{local}, x_t^{local})
\]

它表示局部动作后果与当前观测或节点特征不一致。若训练端使用异方差预测，诊断量也不能直接等于 NLL，而应使用单独定义的 normalized mismatch。

**Topological rupture.**

\[
r_t^{topo}
= d(\hat{v}_t^{change}, v_t^{change})
 + d(\hat{n}_t^{novelty}, n_t^{novelty})
 + d(\hat{f}_t^{frontier}, f_t^{frontier})
\]

它表示拓扑阶段推进与视点变化、新颖性或 frontier 变化不一致。

**Grounding rupture.**

\[
r_t^{ground}
= \mathbb{1}[\hat{\Delta p}_t > \theta_p]
  \cdot \mathbb{1}[\Delta \ell_t < \epsilon_\ell]
  \cdot \mathbb{1}[\Delta v_t = 0 \lor loop_t = 1]
\]

它表示系统以为自己在推进，但身体路径、唯一视点或回环证据显示并没有真实推进。

最终得到：

\[
R_t = [r_t^{local}, r_t^{topo}, r_t^{ground}]
\]

以及类型分布：

\[
p_t^{type} = \mathrm{softmax}(R_t)
\]

这里的 `type_probs` 不是为了制造一个更漂亮的分类头，而是为了让 self revision 知道裂缝来源。

### 3.5 Self Revision

EFES 的分水岭在于 self revision。若系统只是输出 rupture score，然后直接触发 recovery，它仍然只是 monitoring 或 risk detection。EFES 要求 rupture 先修正 self，再由 revised self 影响行为。

给定 self prior $S_t^-$、现实摘要 $G_t$ 和 rupture evidence $R_t$，self revision 输出：

\[
S_t^+ = f_{\mathrm{rev}}(S_t^-, G_t, R_t)
\]

第一版实现中可以写成 gated update：

\[
g_t = \sigma(\mathrm{MLP}([z_t^-, G_t, R_t]))
\]

\[
z_t^+ = g_t \odot z_t^- + (1-g_t) \odot \phi(G_t, R_t)
\]

\[
continuity_t^+ = f_c(continuity_t^-, R_t)
\]

其中 $z_t^-$ 是 self prior 的 representation，$z_t^+$ 是 revised self representation。该更新必须是真实更新，而不是恒等映射。`use_self_revision=False` 的消融应将 recurrent self 回写为 `self_pred_z`，而完整 EFES 使用 `self_post_z`。

### 3.6 Safe Action Conditioning

EFES-active-safe 不替代 ETPNav planner，而是在其 logits 上施加安全 residual 调制。设 ETPNav 原始候选动作 logits 为 $\ell_t^{nav}$，conditioner 根据 revised self、rupture evidence 与候选 embedding 生成 clipped residual：

\[
\Delta \ell_t = \operatorname{clip}(f_{\mathrm{cond}}(S_t^+, R_t, e_t^{cand}), -\delta, \delta)
\]

\[
\ell_t^{safe} = \ell_t^{nav} + g_t^{cond} \cdot \Delta \ell_t
\]

其中 $g_t^{cond}$ 是 conditioner gate。实现上只保留原始 invalid candidate mask，不额外禁止 visited/backtrack，以避免破坏 ETPNav 的动作空间。当前实现支持三种 action source：

- `etp`：直接使用 ETPNav `nav_logits`，作为 passive/no-action 消融。
- `efes_safe`：使用 `safe_logits`，是论文主方法。
- `efes_hard`：使用更激进的 `hard_logits`，只作为风险诊断和对照，不作为主结果。

在训练和评估中，`gmap.node_stop_scores`、teacher-safe CE、eval action 与 backtrack score 都必须使用选定的 `action_logits`，否则会出现训练路径与推理路径不一致的问题。

### 3.7 Grouped Training Objectives

为避免 loss soup，EFES 将训练目标组织成四组：

\[
\mathcal{L}
=
\mathcal{L}_{plan}
+ \lambda_{self}\mathcal{L}_{self}
+ \lambda_{cons}\mathcal{L}_{cons}
+ \lambda_{aux}\mathcal{L}_{aux}
\]

其中：

- $\mathcal{L}_{plan}$：导航主目标。当 `action_source=efes_safe` 时，它训练 conditioner 如何安全修正动作。
- $\mathcal{L}_{self}$：self prior、self state 与 revision 相关约束。
- $\mathcal{L}_{cons}$：local/topological/grounding consequence expectation 的 grouped objective。
- $\mathcal{L}_{aux}$：弱辅助项，默认关闭或极弱。

训练损失与诊断量必须分离。`L_local_pred`、`L_topo_pred` 或 `L_prog_pred` 可以参与训练，但不能直接作为论文中的 rupture score。论文报告的诊断量应是 `r_local`、`r_topo`、`r_ground`、`rupture_confidence`、`self continuity` 与 action conditioning 统计。

## 4. 实验

### 4.1 实验目标与设定

本文实验目标不是证明某个预测头更准，而是验证 EFES 是否真的形成了 `self prior -> rupture -> revised self -> action` 的闭环。当前实验范围固定为 R2R，正式比较统一使用 `IL.back_algo=control`，避免把 `teleport` 与 `control` 的差异误写成方法收益。

当前代码入口为：

- 训练与评估脚本：[run_r2r/efes/efes.bash](/home/D/liumeng/vln/260316P/etpnav-v5/run_r2r/efes/efes.bash)
- 主配置：[run_r2r/efes/efes_main.yaml](/home/D/liumeng/vln/260316P/etpnav-v5/run_r2r/efes/efes_main.yaml)

当前策略是：冻结 ETPNav backbone，训练 EFES self/revision/conditioner，并在 eval/inference 中严格要求 EFES-full checkpoint。baseline ETPNav 通过独立 baseline eval 路径评估，不混入 EFES eval。

### 4.2 比较方法

正式实验应至少包含：

1. **ETPNav baseline.** 原始 ETPNav release checkpoint，`back_algo=control`。
2. **EFES-passive / no_action.** `action_source=etp`，计算 self、rupture 与 revised self，但最终动作仍使用 ETPNav logits。
3. **EFES-active-safe.** `action_source=efes_safe`，使用 revised self 通过 safe residual conditioner 调制动作，是本文主方法。
4. **EFES-hard.** `action_source=efes_hard`，只作为风险诊断，不作为论文主结果。

消融实验应进一步包含：

- `w/o self revision`：recurrent self 使用 `self_pred_z`，不用 `self_post_z`。
- `w/o typed rupture`：三类 rupture 合并为 scalar mismatch。
- `w/o macro/topological rupture`：关闭宏观拓扑 rupture 通道和对应训练项。
- `w/o grounding rupture`：关闭 grounding rupture 通道和对应训练项。

### 4.3 当前 pilot 主结果

当前最可信的 full eval 是 EFES-active-safe iter1200 与 ETPNav baseline。iter1700 目前只有 live/partial 结果，因此不能进入最终主表。

**表 1：R2R val-unseen pilot 结果。**

| 方法 | Checkpoint | Episodes | Success | SPL | nDTW | sDTW | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ETPNav baseline | `ckpt_59` | 1839 | 0.5688 | 0.4896 | 0.5569 | 0.4292 | 原始 ETPNav 评估结果 |
| EFES-active-safe | `iter1200` | 1839 | 0.5633 | 0.4943 | 0.5716 | 0.4325 | full eval，`action_source=efes_safe` |
| EFES-active-safe | `iter1700` | 182 / 1839 | 0.4505 | 0.3965 | 0.4026 | 0.3373 | live partial，不作最终结论 |

当前结果只能支持一个克制结论：EFES-active-safe iter1200 在 Success 上略低于 ETPNav baseline，但在 SPL、nDTW 和 sDTW 上小幅改善，说明 safe action conditioning 可能改善路径效率与轨迹相似度；但这还不是“全面超过 ETPNav”。iter1700 的 partial 结果显示 action distribution 可能发生变化，需要等待 full eval，并结合消融确认 conditioner 是否过强。

### 4.4 诊断统计与实现元数据

EFES eval 输出必须同时报告导航指标与 self/rupture 统计。当前 iter1200 full eval 的附加统计为：

| 指标 | 数值 |
| --- | ---: |
| `mean_a_t` | 2.0611 |
| `mean_c_micro` | 4.6312 |
| `mean_c_macro` | 0.0001 |
| `mean_g_t` | 0.1578 |
| `mean_macro_valid` | 0.1196 |

对应 metadata 必须记录：

- `action_source=efes_safe`
- `back_algo=control`
- `checkpoint_type=efes-full`
- `efes_contract_version=v3-grouped`
- `has_statenav_state=True`
- `use_self_revision=True`
- `use_macro_rupture=True`
- `use_grounding_rupture=True`
- `use_typed_rupture=True`

这些元数据不是附属信息，而是防止误评的必要条件。EFES eval 不应接受 baseline-only checkpoint，也不应在缺少 `statenav_state_dict` 时 fresh init 后继续评估。

### 4.5 消融实验设计

正式投稿前必须补齐以下表格：

**表 2：EFES-active-safe 消融。**

| 设置 | Action source | Self revision | Typed rupture | Macro rupture | Grounding rupture | Success | SPL | nDTW | sDTW |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| ETPNav baseline | ETP route | N/A | N/A | N/A | N/A | 待填 | 待填 | 待填 | 待填 |
| EFES-passive / no_action | `etp` | 开 | 开 | 开 | 开 | 待填 | 待填 | 待填 | 待填 |
| EFES-active-safe | `efes_safe` | 开 | 开 | 开 | 开 | 待填 | 待填 | 待填 | 待填 |
| w/o self revision | `efes_safe` | 关 | 开 | 开 | 开 | 待填 | 待填 | 待填 | 待填 |
| w/o typed rupture | `efes_safe` | 开 | 关 | 开 | 开 | 待填 | 待填 | 待填 | 待填 |
| w/o macro rupture | `efes_safe` | 开 | 开 | 关 | 开 | 待填 | 待填 | 待填 | 待填 |
| w/o grounding rupture | `efes_safe` | 开 | 开 | 开 | 关 | 待填 | 待填 | 待填 | 待填 |
| EFES-hard | `efes_hard` | 开 | 开 | 开 | 开 | 待填 | 待填 | 待填 | 待填 |

其中 `no_action` 必须等价于 `action_source=etp`，不能通过把 `lambda_plan=0.0` 伪装成 action 消融。`no_macro` 与 `no_grounding` 必须真实关闭对应 rupture 通道和对应训练项，而不是只在日志中隐藏该项。

### 4.6 可视化与案例分析

本文至少需要三类可视化：

1. **Rupture 曲线图。** 在同一 episode 中展示 `r_local`、`r_topo`、`r_ground`、`rupture_confidence` 与成功/失败片段的对应关系。
2. **Self continuity 曲线图。** 展示 self prior 与 revised self 在关键转折点前后的变化，证明 self revision 不是恒等映射。
3. **Action conditioning 案例图。** 展示 `nav_logits` 与 `safe_logits` 的变化，说明 EFES 是通过 safe residual 调制动作，而不是替代 planner。

失败案例必须明确区分两类情况：第一，EFES 读出了 rupture，但 conditioner 调制不足或过强；第二，EFES 未能在关键拓扑点读出 rupture。前者是 action conditioning 问题，后者是 self/rupture modeling 问题，不能混成一个“恢复失败”。

## 5. 讨论

### 5.1 为什么 EFES 不是 recovery 论文

EFES 的行为变化来自 revised self 对动作 logits 的安全调制，而不是一个外部 recovery controller。若把论文写成“检测异常，然后恢复”，它会退化成 risk head + heuristic controller。本文的主线必须保持为：typed rupture 先修正 self，revised self 再约束动作。`efes_safe` 是 self revision 的行为接口，而不是方法主体。

### 5.2 为什么 EFES 不是 world-model 论文

EFES 的 prediction heads 是 supporting prediction mechanism，但不是最终对象。它们服务于 rupture attribution，而不是为了 rollout、想象规划或生成未来观测。世界模型回答“世界下一步会怎样”；EFES 回答“世界下一步发生时，我是否仍是那个按指令、按预期、按自身连续性推进的我”。

### 5.3 当前 pilot 风险

当前 iter1200 full eval 的结果比较克制：SPL、nDTW 与 sDTW 小幅高于 baseline，但 Success 略低。这个结果可以支持 active-safe 的可行性，但不能支持“全面优于 ETPNav”的强 claim。

iter1700 的 partial 结果还不能与 full baseline 或 full iter1200 直接比较。它显示 `mean_a_t` 低于 iter1200，说明 conditioner 可能改变了 action distribution。若完整评估后 Success 下降，需要优先检查 residual clipping、condition gate、policy loss 权重与 active 训练时长，而不是把 partial 结果写成有效提升。

### 5.4 本文不主张什么

为了避免概念过度包装，本文显式说明三点：

1. 本文不主张已经解决 Hard Problem，也不主张解释了 consciousness 为何存在。
2. 本文不主张 EFES 是完整的 consciousness 模型；本文只研究其中可工程化的一层，即 rupture-aware embodied self revision。
3. 本文不主张 recovery 或 action conditioning 是方法主体；它们只是 revised self 的行为验证接口。

## 6. 结论

本文提出 EFES，一个面向连续视觉语言导航的破裂感知具身自我修正框架。EFES 将 VLN 中的长程失败重新表述为 action-conditioned self continuity 的可分型 rupture，并通过 `self prior -> consequence expectation -> typed rupture attribution -> self revision -> safe action conditioning` 的闭环进行建模。当前 pilot 结果显示，在冻结 ETPNav backbone 的前提下，EFES-active-safe 能够以可控方式影响动作，并在部分轨迹质量指标上带来小幅改善；但正式投稿前仍需补齐 full eval、完整消融与更多案例分析。本文的核心价值不在于声称已经解决 consciousness，也不在于设计一个更复杂的恢复器，而在于把 rupture-aware embodied self revision 作为 VLN 中一个可定义、可实现、可评估的新研究对象。

## 附录 A：图表与素材清单

### A.1 必备图表

- 图 1：方法总览图，核心展示 `self constitution -> consequence expectation -> typed rupture attribution -> self revision -> safe action conditioning`
- 图 2：Rupture 曲线图，展示 local/topological/grounding rupture 与关键失败点
- 图 3：Self revision 前后对比图，展示 `self_pred_z` 与 `self_post_z`
- 图 4：Action conditioning 图，展示 `nav_logits -> safe_logits`
- 表 1：R2R val-unseen pilot 主结果表
- 表 2：EFES-active-safe 消融表

### A.2 当前已确认的实现事实

- 当前 active 训练目录：
  [efes_active_safe_r2r_pilot_1gpu8env_nohup](/home/D/liumeng/vln/260316P/etpnav-v5/data/logs/checkpoints/efes_active_safe_r2r_pilot_1gpu8env_nohup)
- 当前 full eval 结果：
  [efes_active_safe_iter1200_eval](/home/D/liumeng/vln/260316P/etpnav-v5/data/logs/eval_results/efes_active_safe_iter1200_eval)
- 当前 partial eval 结果：
  [efes_active_safe_iter1700_eval](/home/D/liumeng/vln/260316P/etpnav-v5/data/logs/eval_results/efes_active_safe_iter1700_eval)
- 当前 ETPNav baseline 结果：
  [release_r2r](/home/D/liumeng/vln/260316P/ETPNav/data/logs/eval_results/release_r2r)
- 当前 EFES checkpoint contract：
  `checkpoint_type=efes-full`，`efes_contract_version=v3-grouped`，`has_statenav_state=True`

## 附录 B：参考文献占位

最终版本建议至少覆盖以下文献组：

1. VLN / VLN-CE / R2R 基础工作；
2. ETPNav 与 graph-based / waypoint-based navigation；
3. DREAMWALKER、Navigation World Models、NavMorph 等 world-model navigation；
4. Self-Monitoring、progress estimation 与 uncertainty-triggered reasoning；
5. embodied self、predictive processing 与 rupture/self-revision 相关理论背景。
