# V5 文档治理规范

`version`: `V5 规范 v1`
`updated_at`: `2026-04-06`

本页是 `docs/v5` 的总索引与宪法页。它定义文档职责、阅读顺序、术语优先级、状态标签、禁止性表述和维护规则。
若代码与文档冲突，以 `V5_CODE_CURRENT_CN.md` 判当前事实，以 `V5_FINAL_BLUEPRINT_CN.md` 判未来目标。

## 1. 目标

`V5` 的规范体系只服务一件事：让外部读者、协作者和后续重构都围绕同一个对象展开。

从本次治理开始，`V5` 不再允许“蓝图、现状、档案、路线图”混写在同一份文档里。

## 2. 规范文档集合

| 文档 | 角色 | 允许内容 | 明确不写什么 |
| --- | --- | --- | --- |
| `README_CN.md` | 总索引 / 宪法页 | 文档职责、阅读顺序、术语优先级、状态标签、禁止性表述、维护规则 | 方法细节、实验日志、路线图叙事 |
| `V5_FINAL_BLUEPRINT_CN.md` | 未来版唯一上位定义 | 可计算对象、方法不变量、单一主线、接口边界、能力项状态 | 训练日志、patch 历史、愿景宣言式叙事 |
| `V5_CODE_CURRENT_CN.md` | 当前代码唯一事实来源 | 当前模块、当前数据流、当前约束、当前实现状态、与 blueprint 的映射 | 未来承诺、理想化白皮书表述 |
| `V5_GAP_ROADMAP_CN.md` | 差距 / 原因 / 关闭条件 | `research gap`、`architecture gap`、`engineering blocker`、证据缺口、优先级、close condition | 已关闭但无证据的“预期优化” |
| `V5_TRACEABILITY_MATRIX_CN.md` | 蓝图-代码-gap 对照总表 | blueprint 定义、current code module、status、lane、gap id、evidence artifact、owner | 长叙事、额外方法说明 |
| `p21_V5.md` | 思想档案 | 原始哲学动机、原始问题意识、解释性叙事来源 | 当前规范、当前代码定义、当前实现状态 |

## 3. 阅读顺序

固定阅读顺序如下：

1. `README_CN.md`
2. `V5_FINAL_BLUEPRINT_CN.md`
3. `V5_CODE_CURRENT_CN.md`
4. `V5_GAP_ROADMAP_CN.md`
5. `V5_TRACEABILITY_MATRIX_CN.md`
6. `p21_V5.md`

辅助文档不进入主顺序：

- `v5.md`：历史白皮书档案
- `V5_FLOW_AND_FORMULAS_CN.md`：补充公式说明
- `V5_CODE_INDEX_CN.md`：V5 代码索引
- `V5_ETP_CODE_INDEX_CN.md`：ETP 相关索引
- `prompts.md`：外部评审提示词模板

## 4. 术语优先级

术语冲突时，按以下优先级覆盖：

1. 本页的术语表
2. `V5_FINAL_BLUEPRINT_CN.md`
3. `V5_CODE_CURRENT_CN.md`
4. `archive` 文档只作为背景，不得反向覆盖前 3 份规范文档

核心术语以本页为准：

| 术语 | 规范定义 |
| --- | --- |
| `V5` | 挂接在 `ETP` 风格 planner 之上的 `self-model / self-calibration layer` |
| `ETP` | 主导航器，负责环境编码、图候选构造和基础候选分数 |
| `material_state m_t` | 当前步对 agent 独立存在的外部现实证据，包括观测、图结构、候选几何后果和环境反馈 |
| `conscious_state c_t` | 经过现实修正后的内部 belief / task focus / self-evaluation 组织形式 |
| `contradiction_signal D_t` | `prior / posterior` 失配信号，用于刻画现实对内部状态的纠偏强度 |
| `relation_state s_t` | 由 `conscious_state` 主动检索 `material_state` 后得到的当前关系压缩表示 |
| `intervention_policy` | 基于 `D_t` 与未来健康度触发的 Level 1 / 2 / 3 宏观控制策略 |

## 5. 状态标签规则

状态标签只允许使用这 4 类：

| 标签 | 含义 |
| --- | --- |
| `implemented` | 当前代码主线已存在、默认启用、且有明确模块映射 |
| `partial` | 当前代码中有实现或保留分支，但不完整、非默认、或证据不足 |
| `roadmap` | 尚未进入当前代码主线，只能作为未来目标 |
| `archive` | 历史叙事或旧版本背景，不参与当前规范 |

状态制度有 4 条硬规则：

1. 任一能力点只能有一个主状态。
2. `implemented` 必须能在 `V5_CODE_CURRENT_CN.md` 找到模块映射。
3. `roadmap` 不能写成当前事实。
4. `archive` 不能在规范文档里被复述成现状。

## 6. 禁止性表述

以下表述在规范文档中禁止出现：

- 把 `roadmap` 写成“已经实现”
- 把 `p21_V5.md` 或 `v5.md` 的哲学叙事直接当作当前代码定义
- 把非默认分支、消融分支、诊断接口、保留分支写成主方法
- 使用“已经形成完整闭环”“已经实现真正的语义意识层”这类超出证据的结论
- 在 `V5_FINAL_BLUEPRINT_CN.md` 中写训练日志、预期收益、阶段性 patch 历史
- 在 `V5_CODE_CURRENT_CN.md` 中写“应该”“未来将”“理论上更好”等承诺性语言
- 在 `V5_GAP_ROADMAP_CN.md` 中只写“后续优化”，不写证据缺口和关闭条件

## 7. 维护规则

文档维护遵循以下规则：

1. `V5_FINAL_BLUEPRINT_CN.md` 只定义系统规范，不承担论文故事线。
2. `V5_CODE_CURRENT_CN.md` 只陈述当前仓库事实，不承担愿景表达。
3. `V5_GAP_ROADMAP_CN.md` 只跟踪未关闭问题，不记录已解决但未验证的设想。
4. 任何主线能力改动，都必须同步更新 `V5_FINAL_BLUEPRINT_CN.md` 的状态和 `V5_CODE_CURRENT_CN.md` 的映射。
5. 任何新 ablation 或 diagnostic 接口，都必须显式标记为 `ablation` 或 `diagnostic`，不得混入主线定义。
6. 任何 gap 只有在满足 `close condition` 且补齐证据后，才能从 `V5_GAP_ROADMAP_CN.md` 移除或改状态。
7. 同一 PR 只要改动 `V5` 主线定义、当前实现映射或 gap 状态，就必须同步更新 `V5_FINAL_BLUEPRINT_CN.md`、`V5_CODE_CURRENT_CN.md`、`V5_GAP_ROADMAP_CN.md` 这三份规范文档。

## 8. 补充文档分类

| 文档 | 分类 | 说明 |
| --- | --- | --- |
| `README.md` | `archive` / redirect | 历史入口，现仅保留跳转 |
| `V5_BLUEPRINT_CN.md` | `archive` / redirect | 旧蓝图入口，现跳转到新 blueprint |
| `V5_CODE_ALIGNED_IMPL_CN.md` | `archive` / redirect | 旧 code-aligned 入口，现跳转到新 current 文档 |
| `V5_GAP_LIST_CN.md` | `archive` / redirect | 旧 gap 清单入口，现跳转到新 roadmap 文档 |
| `V5_TRACEABILITY_MATRIX_CN.md` | normative support | 蓝图-代码-gap 对照总表 |
| `v5.md` | `archive` | 历史白皮书，不再作为唯一方法定义 |
| `p21_V5.md` | `archive` | 思想档案，不再承担当前规范职责 |
| `V5_FLOW_AND_FORMULAS_CN.md` | supplement | 公式补充说明 |
| `V5_CODE_INDEX_CN.md` | supplement | V5 主代码索引 |
| `V5_ETP_CODE_INDEX_CN.md` | supplement | ETP 相关代码索引 |
| `prompts.md` | external template | 外部评审模板，不进入规范体系 |
