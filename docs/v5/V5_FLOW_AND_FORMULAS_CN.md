# V5 流程与公式

本文档按代码执行顺序说明 `V5` 的输入、输出和核心公式。

## 输入

当前步输入包括：

- `x_t`：当前全景摘要
- `g_t`：当前全局图摘要
- `x_tokens`：当前视觉 token
- `g_tokens`：当前图 token
- `L`：语言 token
- `f_{t,k}`：第 `k` 个 candidate 特征
- `score_base_k`：ETP 给第 `k` 个 candidate 的原始分数
- `h_{t-1}, z_{t-1}, a_{t-1}`：上一时刻的 RSSM 递推状态

## Step 1：RSSM 当前步更新

### Transition

`h_t^{prior} = Transition(h_{t-1}, z_{t-1}, a_{t-1}, L)`

### Correction

`h_t^{post} = Correction(h_t^{prior}, x_t, g_t, L)`

### Latent

`p(z_t | h_t^{prior})`

`q(z_t | h_t^{post}, x_t, g_t)`

### Reality calibration

`D_t = KL(q(z_t) || p(z_t))`

## Step 2：当前关系状态提取

`s_t = RelationalStateExtractor([h_t^{post}, z_t^{post}], [x_tokens, g_tokens])`

语义：

- 内部状态主动检索外部环境
- 得到当前的三元关系状态

## Step 3：当前关系曲线预测

`progress_curve_t = CurvePredictor(s_t)`

`health_scalar_t = DynamicWeight(progress_curve_t)`

这里只回答：

“如果保持当前关系状态继续发展，未来会不会变坏？”

## Step 4：候选条件驱动统一关系推演

对每个 candidate `k`：

### 4.1 动作条件化内部演化

`a_k = ActionEncoder(f_{t,k})`

`h_{next,k}, z_{next,k} = RSSM_Transition(h_t^{post}, z_t^{post}, a_k, L)`

### 4.2 统一关系推演

`u_{next,k} = UnifiedRolloutTrunk([h_{next,k}, z_{next,k}], [x_tokens, g_tokens, f_{t,k}])`

### 4.3 两路读出

物理读出：

`s_{next,k} = StateHead(u_{next,k})`

`curve_k = CurvePredictor(s_{next,k})`

语义读出：

`predicted_attn_k = AttentionHead(attn_post, u_{next,k})`

## Step 5：健康度计算

### 物理健康度

把 `curve_k` 分解成：

- `early`
- `late`
- `trend`

再做动态加权：

`curve_health_k = w_e * early + w_l * late + w_t * trend`

### 语义健康度

`attn_health_k = AttnHealthScorer(predicted_attn_k, attn_post, L)`

### 最终候选健康度

`health_k = lambda_curve_health * curve_health_k + lambda_attn_health * attn_health_k`

## Step 6：D_t 三级干预

### Level 1

`D_t < theta_low`

只在 strict candidate set 上打分。

### Level 2

`theta_low <= D_t < theta_high`

把 frontier 节点加入 expanded candidate set。

### Level 3

`D_t >= theta_high`

默认：

`backtrack_first`

从历史 `health` 最优节点中选回退目标。

## Step 7：最终打分

`D_t_norm = D_t / theta_high`

`alpha_t = exp(-beta_d * D_t_norm)`

`score_final_k = score_base_k + alpha_t * lambda_health * health_k`

如果触发 Level 2 / 3，则在 expanded set 或 level3 宏观策略上执行同一套最终决策。

## 训练目标

### Planner

`L_planner = CrossEntropy(score_final, GT_action)`

### KL

`L_kl = KL(q(z_t | h_post, x_t, g_t) || p(z_t | h_prior))`

### Curve

`L_curve = MSE(curve_GT_action, GT_curve)`

其中：

- `Δp_0 = current_progress`
- `Δp_h` 基于未来距离变化构造

### Attention

`L_attn = KL(predicted_attn_k_GT || actual_attn_post_{t+1})`

### 总损失

`L_total = L_planner + lambda_kl * L_kl + lambda_curve * L_curve + lambda_attn * L_attn`

## 当前日志主字段

训练和评估主要记录：

- `D_t`
- `d_t_norm`
- `s_t_norm`
- `curve_health`
- `attn_health`
- `cand_health`
- `health_bonus`
- `intervention_level`
- `strict / expanded candidate count`
- `level3 decision`
