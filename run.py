#!/usr/bin/env python3

import argparse
import random
import os
import numpy as np
import torch
from navmorph_compat import patch_habitat_compat

patch_habitat_compat()
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry

import habitat_extensions  # noqa: F401
import vlnce_baselines  # noqa: F401
from vlnce_baselines.models.Policy_ViewSelection_ETP import PolicyViewSelectionETP  # noqa: F401
from vlnce_baselines.trainers.train_efes import EFESTrainer  # noqa: F401
from vlnce_baselines.trainers.train_efes_self import EFESSelfTrainer  # noqa: F401
from vlnce_baselines.trainers.train_efes_v3 import EFESV3Trainer  # noqa: F401
from vlnce_baselines.trainers.train_efes_v2 import EFESV2Trainer  # noqa: F401
from vlnce_baselines.trainers.train_statenav_v5_stage1 import StateNavV5Stage1Trainer  # noqa: F401
from vlnce_baselines.trainers.train_statenav_v5_stage2 import StateNavV5Stage2Trainer  # noqa: F401
from vlnce_baselines.trainers.train_statenav_v6_stage1 import StateNavV6Trainer  # noqa: F401
from vlnce_baselines.config.default import get_config


def _ensure_dir_writable(path: str, label: str, probe_tag: str = "") -> None:
    os.makedirs(path, exist_ok=True)
    probe_name = f".write_probe_{label}_{probe_tag or os.getpid()}"
    probe_path = os.path.join(path, probe_name)
    with open(probe_path, "w", encoding="utf-8") as f:
        f.write("")
    os.remove(probe_path)


def _ensure_parent_dir_writable(file_path: str, label: str, probe_tag: str = "") -> None:
    parent = os.path.dirname(file_path) or "."
    _ensure_dir_writable(parent, label, probe_tag=probe_tag)
# from vlnce_baselines.nonlearning_agents import (
#     evaluate_agent,
#     nonlearning_inference,
# )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--run-type",
        choices=["train", "eval", "inference"],
        required=True,
        help="run type of the experiment (train, eval, inference)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    # 新加: 同时兼容 torch.distributed.launch 旧参数 --local_rank 和新参数 --local-rank。
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        dest="local_rank",
        type=int,
        default=None,
        help="local gpu id",
    )
    args = parser.parse_args()
    # 新加: 当 launch 使用 --use-env / LOCAL_RANK 时，从环境变量兜底读取。
    if args.local_rank is None:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    run_exp(**vars(args))


def run_exp(exp_name: str, exp_config: str, 
            run_type: str, opts=None, local_rank=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    config.defrost()

    config.TENSORBOARD_DIR += exp_name
    config.CHECKPOINT_FOLDER += exp_name
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.VIDEO_DIR += exp_name
    # config.TASK_CONFIG.TASK.RXR_INSTRUCTION_SENSOR.max_text_len = config.IL.max_text_len
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE

    if 'CMA' in config.MODEL.policy_name and 'r2r' in config.BASE_TASK_CONFIG_PATH:
        config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}.json.gz'

    probe_tag = f"rank{local_rank if local_rank is not None else 0}_{os.getpid()}"
    _ensure_dir_writable(config.TENSORBOARD_DIR, "tensorboard", probe_tag=probe_tag)
    _ensure_dir_writable(config.CHECKPOINT_FOLDER, "checkpoint", probe_tag=probe_tag)
    _ensure_dir_writable(config.RESULTS_DIR, "results", probe_tag=probe_tag)
    _ensure_dir_writable(config.VIDEO_DIR, "video", probe_tag=probe_tag)
    if run_type == "inference":
        _ensure_parent_dir_writable(
            config.INFERENCE.PREDICTIONS_FILE,
            "inference_predictions",
            probe_tag=probe_tag,
        )

    config.local_rank = local_rank
    config.freeze()
    runtime_log_dir = os.environ.get("RUNTIME_LOG_DIR", "data/logs/running_log")
    _ensure_dir_writable(runtime_log_dir, "runtime_log", probe_tag=probe_tag)
    base_log_path = os.path.join(runtime_log_dir, config.LOG_FILE)
    if local_rank is None or local_rank == 0:
        log_path = base_log_path
    else:
        stem, ext = os.path.splitext(base_log_path)
        log_path = f"{stem}.rank{local_rank}{ext or '.log'}"
    logger.add_filehandler(log_path)

    if local_rank is None or local_rank == 0:
        logger.info("Runtime log file: %s", log_path)

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    # if run_type == "eval" and config.EVAL.EVAL_NONLEARNING:
    #     evaluate_agent(config)
    #     return

    # if run_type == "inference" and config.INFERENCE.INFERENCE_NONLEARNING:
    #     nonlearning_inference(config)
    #     return

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    # import pdb; pdb.set_trace()
    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()
    elif run_type == "inference":
        trainer.inference()

if __name__ == "__main__":
    main()
