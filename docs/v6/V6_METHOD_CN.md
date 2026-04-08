# StateNav V6

## Summary

V6 的正式主线是 **Hierarchical Self-Model for Self-Diagnosis and Macro Intervention**。

核心目标不是单独做拓扑建图，也不是单独做世界模型预测，而是构造一个更稳定的自我诊断信号，并把这个信号交给宏观干预模块：

- 微观层：保留 step-wise RSSM，建模局部连续动态。
- 宏观层：引入 trainer-side `TopoStateBank`，只在拓扑节点或兜底触发点更新长期状态。
- 诊断层：融合 `D_t_latent` 和 `D_t_node`，得到 `D_t_fused`。
- 决策层：`MacroInterventionManager` 继续作为恢复、回退、停止的最高控制器。

## Method

### Micro temporal self-model

V6 保留 V5 的 micro RSSM：

- `transition(prev_h, prev_z, prev_action) -> h_prior`
- `correction(h_prior, x_t, g_t) -> h_post`
- `latent(h_prior, h_post, x_t, g_t) -> z_post`

微观自我失配定义为：

`D_t_latent = KL(post || prior)`

### Topological self-model

V6 新增 `TopoStateBank`，由 trainer 维护 runtime state：

- `topo_state`
- `topo_feat`
- `last_vp_id`
- `micro_step_count`
- `has_state`

拓扑更新满足任一条件即触发：

- 首次没有 topo state
- `cur_vp` 变化
- 候选数达到决策点阈值
- topo novelty 超阈值
- 连续 micro steps 超过 `max_k`

第一版 topo novelty 定义为：

`1 - cosine(current_avg_pano_embed, cached_topo_feat)`

### Node surprise

V6 新增独立模块 `NodeSurpriseHead`：

- 输入：上一 committed topo state
- 输出：预测当前 topo node feature

节点惊奇度第一版使用 feature-space MSE：

`D_t_node = MSE(pred_next_node_feat, current_avg_pano_embed)`

只在 `topo_valid_mask & topo_update_mask` 上生效。

### Fused self-diagnosis

最终自我诊断信号定义为：

`D_t_fused = alpha_latent * D_t_latent + beta_node * D_t_node`

该信号直接送入 `MacroInterventionManager`，用于控制：

- strict candidate set
- frontier expansion
- level-3 recovery / stop

## Training

V6 使用单阶段训练，主目标固定为：

`L_total = L_plan + lambda_micro_kl * L_micro_kl + lambda_node_surprise * L_node_surprise`

其中：

- `L_plan`：teacher-forcing cross entropy
- `L_micro_kl`：micro RSSM latent KL
- `L_node_surprise`：topo update step 上的 node feature MSE

V6 主线不再使用 V5 的：

- `curve_loss`
- `attn_loss`

## Ablations

`run_r2r/v6/statenav_v6.bash` 支持以下 V6 专用阶段：

- `main`
- `no_node_surprise`
- `no_topo_gate`
- `no_level3`
- `latent_only`
- `topo_only`

## Story Constraint

论文和实验口径统一如下：

- topo bank 不是主贡献，它只是支持 self-diagnosis 的记忆结构
- node surprise 不是单独的 world-model benchmark，它是宏观自检证据
- 核心贡献始终是：
  - 更可信的自我诊断
  - 更稳定的宏观恢复与回正
