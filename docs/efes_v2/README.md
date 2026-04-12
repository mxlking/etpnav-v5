# EFES-v2

EFES-v2 是 EFES 的语义重构版。

目标：
- 拆开 macro 训练损失、异常证据和不确定性
- 统一 macro 全链路到 compressed feature space
- 让 phase2 成为真实联合训练
- 用可训练 router 替代纯规则式恢复切换

路径：
- `vlnce_baselines/models/efes_v2/`
- `vlnce_baselines/agents/efes_v2_agent.py`
- `vlnce_baselines/trainers/train_efes_v2.py`
- `run_r2r/efes_v2/`

旧版关系：
- `efes-main`：早期原型，保留作对照
- `version/v5-base`：V5 固定基线
- `version/v6-hsm`：V6 固定基线

本目录只记录 EFES-v2 主线，不回写旧 EFES 的行为。
