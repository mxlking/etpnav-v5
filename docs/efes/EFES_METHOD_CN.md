# EFES 方法说明（实现对齐版）

EFES 是独立于 V6 的新主线，目标是把自我诊断拆成六个明确模块：

1. `SelfState`
   用上一时刻意识态、语言 token 和上一时刻后验 latent 构造当前意识态 `s_t`，并输出进度估计 `progress_t` 与 attention entropy。
2. `MicroRSSM`
   维护 step-wise prior / posterior，输出微观矛盾度 `C_micro = KL(post || prior)`。
3. `TopoStateBank + NodeSurprise`
   在 trainer 侧维护 batched topo bank，只在关键节点触发时写入；通过高斯预测头输出节点惊讶度 `C_macro`。
4. `FreeEnergyMonitor`
   对 `C_micro`、`C_macro` 和 `g_t` 做 EMA 标准化，输出统一异常量 `A_t`。
5. `SelfConfidence`
   用 `log(1 + prior_var)`、attention entropy 和近期 `C_micro` 均值估计自我置信度 `pi_t`。
6. `DualRecovery`
   根据 `A_t` 与 `pi_t` 决定四种模式：`PROCEED / EXPLORE / BELIEF_UPDATE / BACKTRACK`。

当前实现口径：

- EFES 作为独立 trainer：`EFESTrainer`
- 独立入口：`run_r2r/efes/efes.bash`
- 独立配置：`run_r2r/efes/efes_main.yaml`
- Phase 1：冻结 ETP backbone，只训练 EFES
- Phase 2：解冻 ETP 最后若干层，使用更小学习率联合微调

当前训练目标：

`L = L_plan + 1.0 * L_KL + 0.5 * L_node + 0.1 * L_cal`

其中：

- `L_plan`：行为 logits 的 imitation CE
- `L_KL`：`C_micro` 的 free-nats 版本
- `L_node`：宏观节点高斯 NLL
- `L_cal`：`pi_t` 的一拍延迟 calibration BCE
