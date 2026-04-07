# GitHub Upload Manifest

`version`: `V5 规范 v1`
`updated_at`: `2026-04-07`

文档角色：GitHub 上传白名单 / 黑名单

目标：把当前 `StateNav V5` 可运行代码上传到你自己的 GitHub，同时避免把日志、权重、缓存、临时文件和环境残留一起传上去。

## 1. 上传原则

- 不要直接 `git add .`
- 不要把 `origin` 当成你的 GitHub
- 推荐新建你自己的远端，例如 `mygithub`
- 推荐先上传一个“V5 可运行分支”，而不是一次性把所有脏工作区都推上去

## 2. 必须上传

这些文件和目录构成当前 `V5` 主线的最小可运行集合。

### 2.1 V5 主模型

- `vlnce_baselines/models/statenav_v5/`

说明：
- 这里是 `V5` 的核心实现，包括 RSSM、relation state、rollout、health、macro intervention。

### 2.2 V5 Agent

- `vlnce_baselines/agents/statenav_agent_v5.py`
- `vlnce_baselines/agents/__init__.py`

说明：
- 这里把 `V5` 模型和 trainer 主线接起来。

### 2.3 V5 Trainer

- `vlnce_baselines/trainers/train_statenav_v5_stage1.py`
- `vlnce_baselines/trainers/train_statenav_v5_stage2.py`
- `vlnce_baselines/trainers/__init__.py`

说明：
- 这是当前 `Stage1 / Stage2` 的主训练入口。

### 2.4 配置与注册

- `vlnce_baselines/config/default.py`
- `vlnce_baselines/__init__.py`

说明：
- 当前 `V5` 的默认配置、`Level-3` 策略、诊断接口都在这里注册。

### 2.5 运行脚本

- `run_r2r/v5/`
- `run_rxr/v5/`

说明：
- 这里包含当前主线 YAML、ablation YAML、launcher 和评测矩阵脚本。

### 2.6 诊断与测试

- `scripts/v5_eval_summary.py`
- `tests/test_statenav_v5_macro_intervention.py`
- `tests/test_statenav_v5_gradient_regressions.py`

说明：
- 这是当前 `V5` 的最低限度验证基础设施。

### 2.7 文档

- `docs/v5/`

说明：
- 当前方法定义、代码事实、gap、对照表、上传清单都在这里。

## 3. 建议一起上传

这些文件不一定都属于 `V5` 核心，但如果你想让别人拉下来更容易跑通，建议一起带上。

### 3.1 入口与基类集成层

- `run.py`
- `vlnce_baselines/ss_trainer_ETP.py`
- `vlnce_baselines/common/base_il_trainer.py`

说明：
- 当前训练和评测链路会经过这些文件。

### 3.2 ETP 侧适配文件

- `vlnce_baselines/models/Policy_ViewSelection_ETP.py`
- `vlnce_baselines/models/policy.py`
- `vlnce_baselines/models/__init__.py`

说明：
- 当前 `StateNav V5` 仍然挂接在 `ETP` 主导航器上，这几处改动如果漏掉，分支可能不完整。

### 3.3 与当前分支行为强相关的公共工具

- `vlnce_baselines/common/aux_losses.py`
- `vlnce_baselines/common/env_utils.py`
- `vlnce_baselines/common/environments.py`
- `vlnce_baselines/common/recollection_dataset.py`
- `vlnce_baselines/common/utils.py`
- `vlnce_baselines/utils.py`

说明：
- 如果这些文件在你的当前分支里也改过，并且训练依赖它们，建议一起提交。

## 4. 暂时不要上传

这些内容不是代码主线的一部分，要么是运行产物，要么是环境残留，要么是明显临时文件。

### 4.1 日志、评测结果、权重

- `data/logs/`
- `tmp_eval_ckpt_600/`
- `*.pth`
- `*.pt`
- `*.tsv`
- `*.json` 中属于运行产物的结果文件

### 4.2 Python 缓存和临时目录

- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`

### 4.3 安装/环境残留

- `install.log`
- `get-pip.py`
- `exports/`

### 4.4 备份和临时副本

- `*.bak`
- `* copy*`

例如：
- `habitat_extensions/nav.py.bak`
- `run_r2r/main copy.bash`
- `vlnce_baselines/models/Policy_ViewSelection_ETP copy.py`

## 5. 第一版推荐上传命令

如果你想先推一个“V5 主线分支”，建议先用这一版白名单：

```bash
cd /data5/MP3D/etpnav
git checkout -b statenav-v5

git add \
  docs/v5 \
  run.py \
  run_r2r/v5 \
  run_rxr/v5 \
  scripts/v5_eval_summary.py \
  tests/test_statenav_v5_macro_intervention.py \
  tests/test_statenav_v5_gradient_regressions.py \
  vlnce_baselines/__init__.py \
  vlnce_baselines/config/default.py \
  vlnce_baselines/agents/__init__.py \
  vlnce_baselines/agents/statenav_agent_v5.py \
  vlnce_baselines/trainers/__init__.py \
  vlnce_baselines/trainers/train_statenav_v5_stage1.py \
  vlnce_baselines/trainers/train_statenav_v5_stage2.py \
  vlnce_baselines/models/statenav_v5 \
  vlnce_baselines/models/Policy_ViewSelection_ETP.py \
  vlnce_baselines/models/policy.py \
  vlnce_baselines/models/__init__.py \
  vlnce_baselines/ss_trainer_ETP.py \
  vlnce_baselines/common/base_il_trainer.py
```

如果这些公共工具文件也确实是你当前分支依赖的，再补：

```bash
git add \
  vlnce_baselines/common/aux_losses.py \
  vlnce_baselines/common/env_utils.py \
  vlnce_baselines/common/environments.py \
  vlnce_baselines/common/recollection_dataset.py \
  vlnce_baselines/common/utils.py \
  vlnce_baselines/utils.py
```

## 6. 上传前最后检查

先看暂存区，不要盲推：

```bash
git diff --cached --name-only
```

再跑一次上传检查脚本：

```bash
python scripts/check_github_upload.py
```

默认规则：

- 命中运行产物/权重/日志/缓存黑名单：直接报错
- 单文件大于 `10 MB`：警告
- 单文件大于 `50 MB`：直接报错

理想状态：
- 看到的是 `docs/v5/`、`run_r2r/v5/`、`run_rxr/v5/`、`scripts/`、`tests/`、`vlnce_baselines/` 里的 V5 相关代码
- 看不到 `data/logs/`、`__pycache__/`、`*.pyc`、`*.bak`、`install.log`

## 7. 最稳的远端策略

不要改现在的 `origin`。当前仓库的 `origin` 仍是：

- `MarSaKi/ETPNav`

推荐新增你自己的远端：

```bash
git remote add mygithub git@github.com:<你的GitHub用户名>/etpnav-v5.git
```

然后推送：

```bash
git commit -m "StateNav V5 upgrade"
git push -u mygithub statenav-v5
```

## 8. 一句话建议

如果你的目标是：

- “先把我现在这版 `V5` 代码安全上传到我的 GitHub”

那就按本文件的白名单上传。

如果你的目标是：

- “完整镜像我这台机器上的所有改动”

那不该直接上传，必须先二次筛查 `habitat_extensions/`、`bert_config/`、`pretrain_src/` 等大范围改动。
