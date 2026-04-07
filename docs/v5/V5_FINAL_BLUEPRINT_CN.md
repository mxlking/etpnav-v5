# V5 Final Blueprint

`version`: `V5 规范 v1`
`updated_at`: `2026-04-06`

文档角色：未来版唯一上位定义

本文件是 `V5` 的未来版唯一上位定义。它只描述 `V5` 应当由哪些可计算对象、方法不变量、单一主线和接口边界组成；它不会把 `roadmap` 叙述成当前代码事实。
本文件中的“当前状态”列只用于标记 blueprint 项的落地程度，不替代 `V5_CODE_CURRENT_CN.md` 作为当前代码唯一事实来源。

## 1. 角色与边界

`V5` 的角色固定为：

`ETP-style planner` 之上的 `self-model / self-calibration layer`

它承担的任务是：

- 维护内部 belief 与现实修正之间的关系
- 抽取当前关系状态
- 对候选动作进行条件化未来关系推演
- 基于未来健康度与失配强度执行校准和宏观干预

它不承担的任务是：

- 替代主导航器的环境编码和候选生成
- 建一个独立的世界模型去预测完整外部环境演化
- 把任意解释性哲学表述直接当作当前实现事实

## 2. 可计算对象

| 对象 | 规范定义 | 边界要求 | 当前状态 |
| --- | --- | --- | --- |
| `material_state m_t` | 当前步对 agent 独立存在的外部现实证据 | 只能来自观测、图结构、候选几何后果和环境反馈，不能退化为纯内部 hidden state | `partial` |
| `conscious_state c_t` | 经现实修正后的内部 belief / task focus / self-evaluation | 不能等同于任意单一张量；需要能区分 prior 与 posterior | `partial` |
| `contradiction_signal D_t` | 现实对内部状态的纠偏强度 | 必须由 `prior / posterior` 失配导出，而不是任意手工风险分数 | `implemented` |
| `relation_state s_t` | `conscious_state` 主动检索 `material_state` 后得到的当前关系压缩表示 | 不是 `h_post` 的换名，也不是单纯世界编码 | `implemented` |
| `candidate-conditioned future rollout` | 在候选动作条件下对未来关系演化的统一推演 | 所有未来读出都应共享同一未来关系主干 | `implemented` |
| `health fusion` | 对候选的物理健康度与语义健康度进行统一融合 | 不得把任一单路读出伪装成完整健康模型 | `implemented` |
| `intervention_policy` | 基于 `D_t` 与未来健康度执行 Level 1 / 2 / 3 宏观控制 | Level 3 必须体现“继续前进 vs 恢复节点 vs STOP”的恢复模式切换，而不是单阈值硬覆盖 | `partial` |

## 3. 方法不变量

以下 4 条是 `V5` 的硬不变量：

1. `ETP` 是主导航器，`V5` 不能把它完全替代。
2. `V5` 是上层 `self-model / self-calibration layer`，不是第二个独立 planner。
3. `relation_state s_t` 不是 `h_post` 的换名，而是内部状态主动检索外部现实后的关系表示。
4. `intervention_policy` 属于闭环主线的一部分，不是后处理 heuristic 彩蛋。

## 4. 单一主线

`V5` 的规范主线固定为：

`ETP backbone -> belief update -> contradiction signal -> relation state -> candidate-conditioned future rollout -> physical readout -> semantic readout -> health fusion -> intervention policy`

各阶段定义如下：

| 阶段 | 规范定义 | 主要输出 | 当前状态 |
| --- | --- | --- | --- |
| Base Planner | `ETP` 提供外部 token、候选特征和基础分数 | `score_base_k`, world tokens, candidate features | `implemented` |
| Belief Update | 从 `c_t^-` 到 `c_t^+` 的现实修正 | prior / posterior internal state | `partial` |
| Contradiction Signal | 从 `c_t^-` 与 `c_t^+` 计算现实纠偏强度 | `D_t` | `implemented` |
| Relation Extraction | `s_t = Φ(c_t^+, m_t)` | `relation_state s_t` | `implemented` |
| Future Rollout | 在候选动作条件下演化未来关系 | candidate-conditioned future representation | `implemented` |
| Physical Readout | 从统一未来表示读出物理推进质量 | progress / curve / physical health | `implemented` |
| Semantic Readout | 从统一未来表示读出语义阶段推进质量 | semantic transition / semantic health | `partial` |
| Health Fusion | 将物理与语义健康度合成为行为校准信号 | `health_k` | `implemented` |
| Intervention Policy | 基于 `D_t` 和健康度执行 Level 1 / 2 / 3 决策 | calibrated scores or macro action | `partial` |

## 5. 接口边界

`V5` 的最小接口边界如下：

- 输入边界：
  - `ETP` 风格 planner 输出的外部 token、候选特征、基础分数
  - 当前步与上一步的内部 belief 变量
  - 指令语义上下文
- 中间边界：
  - `D_t`
  - `s_t`
  - candidate-conditioned future representation
  - physical / semantic health
- 输出边界：
  - calibrated candidate scores
  - Level 1 / 2 / 3 intervention decision

以下能力明确不在 blueprint 当前 `implemented` 范围：

- planner-agnostic 的统一后端接口
- 完整的 semantic consciousness model
- 完全 principled 的 risk-aware intervention theory

它们只能在状态为 `roadmap` 或 `partial` 时被引用，不得写成当前既有事实。
