# EFES-v2 论文主线备注

方法主线：

1. Embodied Coupling State
2. Dual-Scale Diagnosis
3. Macro Tri-Factor Separation
4. Frozen-Stat Anomaly Monitor
5. Trainable Recovery Routing

与旧 EFES 的区别：

- `node_nll_loss` 只用于训练
- `c_macro_diag` 只用于诊断
- `u_macro` 只表示宏观不确定性
- `BELIEF_UPDATE` 不再与 `PROCEED` 共用完全相同的候选语义
- phase2 需要真实出现 ETP 解冻参数的非零梯度

首版优先目标：

- `1 iter smoke` 可跑
- `grad_norm` 有限
- `node_nll_loss` 有限
- `c_macro_diag_mean >= 0`
- `mode_hist_argmax` 不塌到单一模式
