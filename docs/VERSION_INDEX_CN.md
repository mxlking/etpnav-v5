# 版本索引与隔离策略

本文档用于固定当前仓库内几个可运行版本的边界，避免 V5、V6 与 EFES 互相覆盖或污染。

## 版本锚点

| 版本 | Git 锚点 | 用途 |
| --- | --- | --- |
| V5 基线 | `version/v5-base` -> `cebf725` | StateNav V5 主线代码，用作早期稳定基线 |
| V6 分层自我模型 | `version/v6-hsm` -> `694c2ad` | 已推送的 V6 hierarchical self-model 版本 |
| EFES 主线 | `efes-main` / `version/efes-main` | 当前 EFES 独立主线与运行兼容层；最新版包含 prior 方差调制和 macro valid-only EMA |
| EFES-v2 | 当前工作树中的 `run_r2r/efes_v2` 与 `vlnce_baselines/models/efes_v2` | EFES 的语义重构版；与旧 EFES 并行维护，不回写旧行为 |

## 隔离原则

- V5 和 V6 不再继续混入 EFES 改动；它们通过 tag 固定到历史提交。
- EFES 在独立分支 `efes-main` 上维护，包含 EFES 模块、运行入口、论文草稿和从旧 export 恢复的 Habitat runtime compatibility。
- EFES-v2 是新的独立实现路径，重点修正 macro 三分离、compressed-space 一致性、phase2 真联合训练和可训练 recovery router。
- EFES 最新版中，`alpha_prior` 作为下一步 RSSM prior 方差放大因子使用；宏观 surprise 只在 `macro_valid_mask=True` 的步上更新 EMA 与统计，避免非触发步污染异常量。
- `data/logs`、checkpoint、运行输出、`__pycache__` 等运行产物不进入版本提交。
- 若服务器目录再次出现兼容问题，优先从 `efes-main` 分支恢复代码，再用本文件中的 smoke 命令检查环境。

## EFES 运行入口

EFES 主入口：

```bash
cd /home/D/liumeng/vln/260316P/etpnav-v5
PYTHON_BIN=/home/D/liumeng/miniconda3/envs/navmorph_try/bin/python \
CUDA_VISIBLE_DEVICES=6 \
TRAIN_ENVS_PER_RANK=1 \
USE_TQDM=False \
WRITE_STEP_METRICS=True \
LOG_FULL_MODEL_REPORT_TO_RUNTIME=False \
bash run_r2r/efes/efes.bash main train 30419 efes_smoke_gpu6 \
  data/logs/checkpoints/release_r2r/ckpt.iter12000.pth \
  IL.iters 1 IL.log_every 1 NUM_ENVIRONMENTS 1
```

## 已验证参考

在 `ETPNav_selected` 工作目录中，恢复旧 export runtime compatibility 后已完成：

- V5 1-iter smoke：`statenav_v5_smoke_gpu6_exportcompat_v3`
- EFES 1-iter smoke：`efes_smoke_gpu6_exportcompat_v4`

注意：EFES smoke 已能完成 1 iter，但初始 `node_loss` 较大，日志中可能出现 `grad_norm=inf`。这属于后续训练稳定性调参项，不是启动链路失败。
