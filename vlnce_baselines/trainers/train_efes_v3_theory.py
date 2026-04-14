from __future__ import annotations

import math
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from habitat import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

import tqdm
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from fastdtw import fastdtw
from habitat_extensions.measures import NDTW
from vlnce_baselines.agents.efes_v3_theory_agent import EFESV3TheoryAgent
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.datasets.state_label_builder import StateLabelBuilder
from vlnce_baselines.models.efes_v3_theory.topo_bank import TheoryTopoBank
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.trainers.train_efes_self import (
    _RELEASE_R2R_CKPT,
    _state_model,
    cuda_autocast,
    make_grad_scaler,
)
from vlnce_baselines.trainers.train_efes_v3 import EFESV3Trainer


@baseline_registry.register_trainer(name="EFESV3Theory")
class EFESV3TheoryTrainer(EFESV3Trainer):
    stage_name = "EFESV3Theory"
    contract_version = "v3-theory"
    accepted_contract_versions = ("v3-theory",)

    def __init__(self, config=None):
        super().__init__(config)
        self._phase_name = "main"

    def _efes_cfg(self):
        return self.config.EFES_V3_THEORY

    def _logging_cfg(self):
        return self.config.EFES_V3_THEORY.LOGGING

    def _efes_action_source(self) -> str:
        source = str(getattr(self._efes_cfg(), "action_source", "corrected")).strip().lower()
        if source not in {"corrected", "etp"}:
            logger.warning("Unknown EFES_V3_THEORY.action_source=%s. Falling back to corrected.", source)
            source = "corrected"
        return source

    def _get_eval_result_metadata(self) -> Dict[str, Any]:
        payload = dict(self._last_checkpoint_fingerprint) if self._last_checkpoint_fingerprint else {}
        cfg = self._efes_cfg()
        payload.update(
            {
                "action_source": self._efes_action_source(),
                "back_algo": str(self.config.IL.back_algo),
                "alpha": float(getattr(cfg, "alpha", 0.5)),
                "beta": float(getattr(cfg, "beta", 0.1)),
                "lambda_macro": float(getattr(cfg, "lambda_macro", 1.0)),
                "lambda_fe": float(getattr(cfg, "lambda_fe", 1.0)),
                "lambda_cal": float(getattr(cfg, "lambda_cal", 1.0)),
                "lambda_safe": float(getattr(cfg, "lambda_safe", 0.0)),
                "lambda_mi": float(getattr(cfg, "lambda_mi", 0.0)),
                "sigma_min": float(getattr(cfg, "sigma_min", 0.01)),
                "sigma_max": float(getattr(cfg, "sigma_max", 1.0)),
                "use_body_loop": bool(getattr(cfg, "use_body_loop", True)),
                "use_predictor": bool(getattr(cfg, "use_predictor", True)),
                "use_corrector": bool(getattr(cfg, "use_corrector", True)),
                "use_shifted_surprise": bool(getattr(cfg, "use_shifted_surprise", True)),
                "use_shared_valid_mask_only": bool(getattr(cfg, "use_shared_valid_mask_only", True)),
                "use_safe_kl": bool(getattr(cfg, "use_safe_kl", True)),
                "use_sigma_max_clamp": bool(getattr(cfg, "use_sigma_max_clamp", True)),
                "use_dual_gate": bool(getattr(cfg, "use_dual_gate", True)),
                "use_mi_proxy": bool(getattr(cfg, "use_mi_proxy", True)),
            }
        )
        return payload

    def _make_statenav_components(self) -> None:
        if self.world_size > 1:
            etp_dim = self.policy.net.module.output_size
        else:
            etp_dim = self.policy.net.output_size
        self.statenav_agent = EFESV3TheoryAgent.from_config(
            self.config,
            x_dim=etp_dim,
            lang_dim=etp_dim,
            cand_dim=etp_dim,
        ).to(self.device)
        self.label_builder = StateLabelBuilder.from_config(
            self.config.STATENAV.LABEL_BUILDER,
            state_bins=int(self.config.STATENAV.state_bins),
        )

    def _wrap_statenav_for_ddp(self) -> None:
        if self.world_size <= 1 or isinstance(self.statenav_agent, DDP):
            return
        self.statenav_agent = DDP(
            self.statenav_agent,
            device_ids=[self.device_id],
            output_device=self.device_id,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )

    def _load_stage_checkpoint(self, checkpoint_path: str) -> int:
        ckpt, checkpoint_realpath, fingerprint = self._inspect_checkpoint(checkpoint_path)
        mode = getattr(self, "_checkpoint_mode", "train")
        self._validate_efes_checkpoint_contract(
            fingerprint,
            mode=mode,
            is_requeue=bool(self.config.IL.is_requeue),
        )
        self._log_checkpoint_fingerprint(fingerprint, context=mode)

        has_statenav_state = bool(fingerprint["has_statenav_state"])
        if "policy_state_dict" in ckpt:
            policy_state_dict = self._align_state_dict_prefix(ckpt["policy_state_dict"], self.policy)
        elif "state_dict" in ckpt:
            policy_state_dict = self._align_state_dict_prefix(ckpt["state_dict"], self.policy)
        else:
            raise KeyError(
                "Checkpoint {} has neither state_dict nor policy_state_dict".format(checkpoint_path)
            )
        policy_load_msg = self.policy.load_state_dict(policy_state_dict, strict=False)
        if policy_load_msg.missing_keys or policy_load_msg.unexpected_keys:
            raise RuntimeError(
                "Invalid Policy checkpoint load for {}: missing_keys={}, unexpected_keys={}".format(
                    checkpoint_path,
                    len(policy_load_msg.missing_keys),
                    len(policy_load_msg.unexpected_keys),
                )
            )

        if has_statenav_state:
            statenav_state_dict = self._align_state_dict_prefix(ckpt["statenav_state_dict"], self.statenav_agent)
            load_msg = self.statenav_agent.load_state_dict(statenav_state_dict, strict=False)
            if load_msg.missing_keys or load_msg.unexpected_keys:
                raise RuntimeError(
                    "Invalid EFESV3Theory checkpoint load for {}: missing_keys={}, unexpected_keys={}".format(
                        checkpoint_path,
                        len(load_msg.missing_keys),
                        len(load_msg.unexpected_keys),
                    )
                )
            if self.config.IL.is_requeue and "optim_state" in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt["optim_state"])
                except Exception as exc:
                    logger.warning("Unable to restore optimizer state cleanly: %s", exc)

        self._resume_scaler_state = ckpt.get("scaler_state") if has_statenav_state else None
        self.best_metric = float(ckpt.get("best_metric", self.best_metric)) if has_statenav_state else float("inf")
        start_iter = int(ckpt.get("iteration", 0)) if has_statenav_state else 0
        self._last_checkpoint_fingerprint = fingerprint

        if has_statenav_state:
            logger.info(
                "Loaded EFESV3Theory checkpoint for %s: %s at iteration %d [contract=%s phase=%s]",
                mode,
                checkpoint_path,
                start_iter,
                fingerprint.get("efes_contract_version"),
                fingerprint.get("efes_phase"),
            )
        else:
            logger.info(
                "Loaded bootstrap-only ETPNav checkpoint for EFESV3Theory training: %s.",
                checkpoint_path,
            )
        return start_iter

    @staticmethod
    def _mean_float_list(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _binary_auc_from_scores(scores: Tensor, labels: Tensor) -> Tensor:
        scores = scores.detach().float()
        labels = labels.detach().float()
        pos = labels > 0.5
        neg = labels <= 0.5
        pos_count = int(pos.sum().item())
        neg_count = int(neg.sum().item())
        if pos_count == 0 or neg_count == 0:
            return scores.new_tensor(0.5)
        ranked = torch.argsort(scores)
        ranks = torch.empty_like(ranked, dtype=torch.float32)
        ranks[ranked] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
        sum_pos = ranks[pos].sum()
        auc = (sum_pos - pos_count * (pos_count + 1) / 2.0) / float(pos_count * neg_count)
        return auc.clamp(0.0, 1.0)

    def _info_nce_proxy(self, query: Tensor, key: Tensor) -> Tensor:
        if query.size(0) <= 1:
            return query.new_zeros(query.size(0))
        temp = max(float(getattr(self._efes_cfg(), "mi_temperature", 0.1)), 1e-6)
        logits = torch.matmul(query, key.transpose(0, 1)) / temp
        labels = torch.arange(query.size(0), device=query.device)
        ce = F.cross_entropy(logits, labels, reduction="none")
        lower_bound = math.log(max(query.size(0), 2)) - ce
        return lower_bound

    @staticmethod
    def _teacher_margin_delta(
        base_logits: Tensor,
        corrected_logits: Tensor,
        teacher_actions: Tensor,
        invalid_mask: Tensor,
    ) -> Tensor:
        valid = teacher_actions.ge(0)
        if not bool(valid.any().item()):
            return base_logits.new_zeros(base_logits.size(0))
        masked_base = base_logits.masked_fill(invalid_mask, -1.0e4)
        masked_corr = corrected_logits.masked_fill(invalid_mask, -1.0e4)
        safe_teacher = teacher_actions.clamp(min=0)
        batch_index = torch.arange(base_logits.size(0), device=base_logits.device)
        base_teacher = masked_base[batch_index, safe_teacher]
        corr_teacher = masked_corr[batch_index, safe_teacher]
        base_other = masked_base.clone()
        corr_other = masked_corr.clone()
        base_other[batch_index, safe_teacher] = -1.0e4
        corr_other[batch_index, safe_teacher] = -1.0e4
        base_margin = base_teacher - base_other.max(dim=-1).values
        corr_margin = corr_teacher - corr_other.max(dim=-1).values
        delta = corr_margin - base_margin
        return delta * valid.to(dtype=delta.dtype)

    @staticmethod
    def _step_metric_columns() -> List[str]:
        return [
            "contract_version",
            "action_source",
            "iteration",
            "phase",
            "interval_step",
            "interval_size",
            "world_size",
            "envs_per_rank",
            "sample_ratio",
            "lr",
            "grad_norm",
            "total_loss",
            "nav_loss",
            "fe_loss_opt",
            "fe_loss_raw",
            "calibrate_loss_opt",
            "calibrate_loss_raw",
            "safe_loss",
            "mi_loss",
            "kl_micro_mean",
            "nll_macro_mean",
            "u_t_raw_mean",
            "u_tilde_mean",
            "recon_sq_error_mean",
            "sigma_mean",
            "sigma_hit_low_rate",
            "sigma_hit_high_rate",
            "gate_mean",
            "gate_p90",
            "lambda_hat_mean",
            "delta_abs_max",
            "kl_corrected_vs_base",
            "teacher_margin_delta",
            "base_error_rate",
            "gate_tp_rate",
            "gate_fp_rate",
            "surprise_auc_base_error",
            "mi_proxy",
            "progress_mean",
            "score_before_mean",
            "score_after_mean",
            "total_actions",
            "total_topo_updates",
            "topo_update_ratio",
            "train_step_sec",
            "rollout_sec",
            "backward_sec",
            "optimizer_sec",
            "lang_sec",
            "waypoint_sec",
            "panorama_sec",
            "graph_identify_sec",
            "cand_real_pos_sec",
            "graph_update_sec",
            "navigation_sec",
            "statenav_sec",
            "env_step_sec",
            "batch_prep_sec",
        ]

    def _log_step_metric_record(self, record: Dict[str, Any]) -> None:
        if not self._is_main_process():
            return
        step_log_every = max(int(self._logging_cfg().step_log_every), 1)
        if int(record["iteration"]) % step_log_every != 0:
            return
        logger.info(
            "[EFESV3Theory %s action=%s %06d/%06d | interval %03d/%03d] total=%.4f nav=%.4f fe_opt=%.4f(raw=%.4f) cal_opt=%.4f(raw=%.4f) safe=%.4f mi=%.4f grad=%.4f"
            " | u=%.3f u~=%.3f kl=%.3f nll=%.3f recon=%.3f sigma=%.3f gate=%.3f lambda=%.3f auc=%.3f tp=%.3f fp=%.3f"
            " | sec(step=%.2f roll=%.2f bw=%.2f opt=%.2f nav=%.2f theory=%.2f env=%.2f)",
            str(record.get("contract_version", self.contract_version)),
            str(record.get("action_source", self._efes_action_source())),
            int(record["iteration"]),
            int(self.config.IL.iters),
            int(record.get("interval_step", 0)),
            int(record.get("interval_size", 0)),
            float(record.get("total_loss", 0.0)),
            float(record.get("nav_loss", 0.0)),
            float(record.get("fe_loss_opt", 0.0)),
            float(record.get("fe_loss_raw", 0.0)),
            float(record.get("calibrate_loss_opt", 0.0)),
            float(record.get("calibrate_loss_raw", 0.0)),
            float(record.get("safe_loss", 0.0)),
            float(record.get("mi_loss", 0.0)),
            float(record.get("grad_norm", 0.0)),
            float(record.get("u_t_raw_mean", 0.0)),
            float(record.get("u_tilde_mean", 0.0)),
            float(record.get("kl_micro_mean", 0.0)),
            float(record.get("nll_macro_mean", 0.0)),
            float(record.get("recon_sq_error_mean", 0.0)),
            float(record.get("sigma_mean", 0.0)),
            float(record.get("gate_mean", 0.0)),
            float(record.get("lambda_hat_mean", 0.0)),
            float(record.get("surprise_auc_base_error", 0.0)),
            float(record.get("gate_tp_rate", 0.0)),
            float(record.get("gate_fp_rate", 0.0)),
            float(record.get("train_step_sec", 0.0)),
            float(record.get("rollout_sec", 0.0)),
            float(record.get("backward_sec", 0.0)),
            float(record.get("optimizer_sec", 0.0)),
            float(record.get("navigation_sec", 0.0)),
            float(record.get("statenav_sec", 0.0)),
            float(record.get("env_step_sec", 0.0)),
        )

    def train(self):
        self._checkpoint_mode = "train"
        self._set_config()
        self._load_gt_data()
        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._train_iteration = start_iter
        total_iter = self.config.IL.iters
        log_every = self.config.IL.log_every
        writer = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = make_grad_scaler()
        if self._resume_scaler_state is not None:
            try:
                self.scaler.load_state_dict(self._resume_scaler_state)
            except Exception as exc:
                logger.warning("Unable to restore scaler state cleanly: %s", exc)

        self._prepare_step_metric_writer()
        logger.info("EFESV3Theory training starts...")
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter - idx, 0))
            cur_iter = idx + interval
            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                summary = {k: float(np.mean(v)) for k, v in logs.items()}
                total_metric = summary.get("total_loss", float("inf"))
                is_best = False
                if bool(self._logging_cfg().save_best_by_train_loss):
                    is_best = total_metric < self.best_metric
                    if is_best:
                        self.best_metric = total_metric
                logger.info(
                    "[EFESV3Theory summary %06d] %s",
                    cur_iter,
                    ", ".join("{}: {:.4f}".format(k, v) for k, v in summary.items()),
                )
                for k, v in summary.items():
                    writer.add_scalar("loss/{}".format(k), v, cur_iter)
                self.save_checkpoint(cur_iter, is_best=is_best)

    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.eval()
        self.statenav_agent.train()
        self.waypoint_predictor.eval()

        use_tqdm = bool(self._logging_cfg().use_tqdm) and self._is_main_process()
        pbar = (
            tqdm.trange(interval, leave=False, dynamic_ncols=True, desc="EFESV3Theory")
            if use_tqdm
            else range(interval)
        )
        self.logs = defaultdict(list)
        for idx in pbar:
            train_step_start = self._timing_now()
            self.optimizer.zero_grad(set_to_none=True)
            self.loss = torch.zeros((), device=self.device)

            rollout_start = self._timing_now()
            with cuda_autocast():
                self.rollout("train", ml_weight, sample_ratio)
            rollout_sec = self._timing_now() - rollout_start

            backward_start = self._timing_now()
            self.scaler.scale(self.loss).backward()
            self.scaler.unscale_(self.optimizer)
            trainable_params = [param for param in self.statenav_agent.parameters() if param.grad is not None]
            for param in trainable_params:
                if not torch.isfinite(param.grad).all():
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=float(self._efes_cfg().grad_clip_norm),
            )
            backward_sec = self._timing_now() - backward_start

            optimizer_start = self._timing_now()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_sec = self._timing_now() - optimizer_start
            self._train_iteration += 1

            train_step_sec = self._timing_now() - train_step_start
            self._last_step_metrics.update(
                {
                    "train_step_sec": float(train_step_sec),
                    "rollout_sec": float(rollout_sec),
                    "backward_sec": float(backward_sec),
                    "optimizer_sec": float(optimizer_sec),
                }
            )

            if self._is_main_process():
                step_record = self._build_step_metric_record(
                    interval_step=idx + 1,
                    interval_size=interval,
                    sample_ratio=float(sample_ratio),
                    grad_norm=float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm),
                )
                self._append_step_metric_record(step_record)
                self._log_step_metric_record(step_record)
                if use_tqdm:
                    pbar.set_postfix(
                        {
                            "step": "{}/{}".format(self._train_iteration, self.config.IL.iters),
                            "loss": "{:.4f}".format(step_record.get("total_loss", 0.0)),
                        }
                    )
        return deepcopy(self.logs)

    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        feedback = "sample" if mode == "train" else "argmax"
        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len
        instr_pad_id = 1 if self.config.MODEL.task_type == "rxr" else 0
        observations = extract_instruction_tokens(
            observations,
            self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            max_length=instr_max_len,
            pad_id=instr_pad_id,
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if mode == "eval":
            env_to_pause = [
                i
                for i, ep in enumerate(self.envs.current_episodes())
                if ep.episode_id in self.stat_eps
            ]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0:
                return
        if mode == "infer":
            env_to_pause = [
                i
                for i, ep in enumerate(self.envs.current_episodes())
                if ep.episode_id in self.path_eps
            ]
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0:
                return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == "rxr":
                    ep_id = curr_eps[i].episode_id
                    self.inst_ids[ep_id] = int(curr_eps[i].instruction.instruction_id)

        self.statenav_agent.train() if mode == "train" else self.statenav_agent.eval()

        lang_start = self._timing_now()
        with torch.no_grad():
            all_txt_ids = batch["instruction"]
            all_txt_masks = all_txt_ids != instr_pad_id
            all_txt_embeds = self.policy.net(
                mode="language",
                txt_ids=all_txt_ids,
                txt_masks=all_txt_masks,
            )
        timing_sums = {
            "lang_sec": self._timing_now() - lang_start,
            "waypoint_sec": 0.0,
            "panorama_sec": 0.0,
            "graph_identify_sec": 0.0,
            "cand_real_pos_sec": 0.0,
            "graph_update_sec": 0.0,
            "navigation_sec": 0.0,
            "statenav_sec": 0.0,
            "env_step_sec": 0.0,
            "batch_prep_sec": 0.0,
        }

        nav_loss_sum = torch.zeros((), device=self.device)
        fe_loss_sum = torch.zeros((), device=self.device)
        fe_loss_raw_sum = torch.zeros((), device=self.device)
        calibrate_loss_sum = torch.zeros((), device=self.device)
        calibrate_loss_raw_sum = torch.zeros((), device=self.device)
        safe_loss_sum = torch.zeros((), device=self.device)
        mi_loss_sum = torch.zeros((), device=self.device)

        gate_sum = torch.zeros((), device=self.device)
        gate_p90_sum = torch.zeros((), device=self.device)
        gate_p90_count = torch.zeros((), device=self.device)
        lambda_hat_sum = torch.zeros((), device=self.device)
        delta_abs_max_sum = torch.zeros((), device=self.device)
        u_raw_sum = torch.zeros((), device=self.device)
        u_tilde_sum = torch.zeros((), device=self.device)
        kl_micro_sum = torch.zeros((), device=self.device)
        nll_macro_sum = torch.zeros((), device=self.device)
        recon_sq_error_sum = torch.zeros((), device=self.device)
        sigma_mean_sum = torch.zeros((), device=self.device)
        sigma_low_sum = torch.zeros((), device=self.device)
        sigma_high_sum = torch.zeros((), device=self.device)
        progress_sum = torch.zeros((), device=self.device)
        score_before_sum = torch.zeros((), device=self.device)
        score_after_sum = torch.zeros((), device=self.device)
        teacher_margin_delta_sum = torch.zeros((), device=self.device)
        kl_corrected_vs_base_sum = torch.zeros((), device=self.device)
        gate_tp_sum = torch.zeros((), device=self.device)
        gate_fp_sum = torch.zeros((), device=self.device)
        auc_sum = torch.zeros((), device=self.device)
        auc_count_sum = torch.zeros((), device=self.device)
        base_error_sum = torch.zeros((), device=self.device)
        mi_proxy_sum = torch.zeros((), device=self.device)
        topo_update_sum = torch.zeros((), device=self.device)
        total_actions = 0

        not_done_index = list(range(self.envs.num_envs))
        have_real_pos = mode == "train" or self.config.VIDEO_OPTION
        ghost_aug = self.config.IL.ghost_aug if mode == "train" else 0
        self.gmaps = [
            GraphMap(
                have_real_pos,
                self.config.IL.loc_noise,
                self.config.MODEL.merge_ghost,
                ghost_aug,
            )
            for _ in range(self.envs.num_envs)
        ]
        prev_vp = [None] * self.envs.num_envs
        state_model = _state_model(self.statenav_agent)
        recurrent_state = state_model.init_recurrent_state(self.envs.num_envs, self.device)
        topo_bank = TheoryTopoBank.create(
            num_envs=self.envs.num_envs,
            max_nodes=int(self._efes_cfg().max_nodes),
            feat_dim=int(state_model.x_dim),
            device=self.device,
        )
        active_pos_histories: List[Deque[Tensor]] = [
            deque(maxlen=int(self._efes_cfg().history_window)) for _ in range(self.envs.num_envs)
        ]
        active_eval_diag = [
            {"u_raw": [], "u_tilde": [], "gate": [], "progress": []}
            for _ in range(self.envs.num_envs)
        ] if mode == "eval" else None

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]

            with torch.no_grad():
                waypoint_start = self._timing_now()
                wp_outputs = self.policy.net(
                    mode="waypoint",
                    waypoint_predictor=self.waypoint_predictor,
                    observations=batch,
                    in_train=(mode == "train" and self.config.IL.waypoint_aug),
                )
                timing_sums["waypoint_sec"] += self._timing_now() - waypoint_start

                panorama_start = self._timing_now()
                vp_inputs = self._vp_feature_variable(wp_outputs)
                vp_inputs.update({"mode": "panorama"})
                pano_embeds, pano_masks = self.policy.net(**vp_inputs)
                pano_mask_f = pano_masks.to(dtype=pano_embeds.dtype)
                pano_denom = torch.sum(pano_mask_f, 1, keepdim=True).clamp_min(1.0)
                avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / pano_denom
                timing_sums["panorama_sec"] += self._timing_now() - panorama_start

                identify_start = self._timing_now()
                cur_pos, cur_ori = self.get_pos_ori()
                cur_vp, cand_vp, cand_pos = [], [], []
                for i in range(self.envs.num_envs):
                    cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                        cur_pos[i],
                        cur_ori[i],
                        wp_outputs["cand_angles"][i],
                        wp_outputs["cand_distances"][i],
                    )
                    cur_vp.append(cur_vp_i)
                    cand_vp.append(cand_vp_i)
                    cand_pos.append(cand_pos_i)
                timing_sums["graph_identify_sec"] += self._timing_now() - identify_start

                if mode == "train" or self.config.VIDEO_OPTION:
                    cand_real_pos_start = self._timing_now()
                    cand_real_pos = self.envs.call(
                        ["get_cand_real_pos_batch"] * self.envs.num_envs,
                        [
                            {
                                "angles": [float(ang) for ang in wp_outputs["cand_angles"][i]],
                                "forwards": [float(dis) for dis in wp_outputs["cand_distances"][i]],
                            }
                            for i in range(self.envs.num_envs)
                        ],
                    )
                    timing_sums["cand_real_pos_sec"] += self._timing_now() - cand_real_pos_start
                else:
                    cand_real_pos = [None] * self.envs.num_envs

                graph_update_start = self._timing_now()
                for i in range(self.envs.num_envs):
                    cur_embeds = avg_pano_embeds[i]
                    cand_embeds = pano_embeds[i][vp_inputs["nav_types"][i] == 1]
                    self.gmaps[i].update_graph(
                        prev_vp[i],
                        stepk + 1,
                        cur_vp[i],
                        cur_pos[i],
                        cur_embeds,
                        cand_vp[i],
                        cand_pos[i],
                        cand_embeds,
                        cand_real_pos[i],
                    )
                timing_sums["graph_update_sec"] += self._timing_now() - graph_update_start

                nav_start = self._timing_now()
                nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
                nav_inputs.update(
                    {
                        "mode": "navigation",
                        "txt_embeds": txt_embeds,
                        "txt_masks": txt_masks,
                    }
                )
                no_vp_left = nav_inputs.pop("no_vp_left")
                nav_outs = self.policy.net(**nav_inputs)
                nav_logits = nav_outs["global_logits"]
                timing_sums["navigation_sec"] += self._timing_now() - nav_start

            invalid_candidate_mask = nav_inputs["gmap_masks"].logical_not()
            visited_candidate_mask = nav_inputs["gmap_visited_masks"]
            if bool(getattr(self._efes_cfg(), "use_shared_valid_mask_only", True)):
                candidate_mask = invalid_candidate_mask
            else:
                candidate_mask = invalid_candidate_mask | visited_candidate_mask

            statenav_start = self._timing_now()
            step_outs = self.statenav_agent(
                x_t=avg_pano_embeds,
                node_feat=avg_pano_embeds,
                lang_tokens=txt_embeds,
                cand_feats=nav_outs["gmap_embeds"],
                etp_candidate_scores=nav_logits,
                prev_action_emb=recurrent_state["prev_action_emb"],
                prev_z=recurrent_state["prev_z"],
                prev_self=recurrent_state["prev_self"],
                prev_rssm_h=recurrent_state["prev_rssm_h"],
                prev_progress=recurrent_state["prev_progress"],
                prev_u_tilde=recurrent_state["prev_u_tilde"],
                topo_bank_feat=topo_bank.feat,
                topo_bank_mask=topo_bank.mask,
                candidate_mask=candidate_mask,
                lang_mask=txt_masks,
            )
            timing_sums["statenav_sec"] += self._timing_now() - statenav_start

            topo_update_mask = topo_bank.should_update(cur_vp)
            topo_bank.commit(
                update_mask=topo_update_mask,
                current_vp_ids=cur_vp,
                node_feat=step_outs["compressed_node_feat"],
            )

            gate_det = step_outs["gate"].detach().float()
            lambda_hat_det = step_outs["lambda_hat"].detach().float()
            delta_abs_max_det = step_outs["delta_abs_max"].detach().float()
            u_raw_det = step_outs["u_t_raw"].detach().float()
            u_tilde_det = step_outs["u_tilde"].detach().float()
            kl_micro_det = step_outs["kl_micro"].detach().float()
            nll_macro_det = step_outs["nll_macro"].detach().float()
            recon_sq_error_det = step_outs["recon_sq_error"].detach().float()
            sigma_mean_det = step_outs["sigma_mean"].detach().float()
            sigma_low_det = step_outs["sigma_hit_low_rate"].detach().float()
            sigma_high_det = step_outs["sigma_hit_high_rate"].detach().float()
            progress_det = step_outs["progress_t"].detach().float()
            score_before_det = step_outs["score_before_mean"].detach().float()
            score_after_det = step_outs["score_after_mean"].detach().float()
            kl_corrected_vs_base_det = step_outs["kl_corrected_vs_base"].detach().float()

            gate_sum = gate_sum + gate_det.sum()
            gate_p90_sum = gate_p90_sum + torch.quantile(gate_det, 0.9)
            gate_p90_count = gate_p90_count + 1.0
            lambda_hat_sum = lambda_hat_sum + lambda_hat_det.sum()
            delta_abs_max_sum = delta_abs_max_sum + delta_abs_max_det.sum()
            u_raw_sum = u_raw_sum + u_raw_det.sum()
            u_tilde_sum = u_tilde_sum + u_tilde_det.sum()
            kl_micro_sum = kl_micro_sum + kl_micro_det.sum()
            nll_macro_sum = nll_macro_sum + nll_macro_det.sum()
            recon_sq_error_sum = recon_sq_error_sum + recon_sq_error_det.sum()
            sigma_mean_sum = sigma_mean_sum + sigma_mean_det.sum()
            sigma_low_sum = sigma_low_sum + sigma_low_det.sum()
            sigma_high_sum = sigma_high_sum + sigma_high_det.sum()
            progress_sum = progress_sum + progress_det.sum()
            score_before_sum = score_before_sum + score_before_det.sum()
            score_after_sum = score_after_sum + score_after_det.sum()
            kl_corrected_vs_base_sum = kl_corrected_vs_base_sum + kl_corrected_vs_base_det.sum()
            topo_update_sum = topo_update_sum + topo_update_mask.to(dtype=avg_pano_embeds.dtype).sum()

            if mode == "eval" and active_eval_diag is not None:
                for i in range(self.envs.num_envs):
                    active_eval_diag[i]["u_raw"].append(float(step_outs["u_t_raw"][i].detach().item()))
                    active_eval_diag[i]["u_tilde"].append(float(step_outs["u_tilde"][i].detach().item()))
                    active_eval_diag[i]["gate"].append(float(step_outs["gate"][i].detach().item()))
                    active_eval_diag[i]["progress"].append(float(step_outs["progress_t"][i].detach().item()))

            if mode == "train" or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs["gmap_vp_ids"], no_vp_left)
            else:
                teacher_actions = None

            action_logits = nav_logits if self._efes_action_source() == "etp" else step_outs["corrected_logits"]
            nav_probs = F.softmax(action_logits, dim=1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].detach().item()

            if mode == "train":
                loss_logits = self._make_teacher_safe_logits(
                    rescored_logits=action_logits,
                    fallback_logits=nav_logits,
                    teacher_actions=teacher_actions,
                    invalid_candidate_mask=invalid_candidate_mask,
                )
                nav_loss_sum = nav_loss_sum + F.cross_entropy(
                    loss_logits,
                    teacher_actions.long(),
                    ignore_index=-100,
                    reduction="sum",
                )

                fe_step_raw = step_outs["loss_micro"] + float(getattr(self._efes_cfg(), "lambda_macro", 1.0)) * step_outs["loss_macro"]
                fe_loss_raw_sum = fe_loss_raw_sum + fe_step_raw.sum()
                feat_dim = max(int(step_outs["compressed_node_feat"].size(-1)), 1)
                fe_step_opt = fe_step_raw / float(feat_dim)
                fe_loss_sum = fe_loss_sum + fe_step_opt.sum()

                teacher_valid = teacher_actions.ge(0)
                masked_base = nav_logits.masked_fill(invalid_candidate_mask, -1.0e4)
                base_choice = masked_base.argmax(dim=-1)
                error_target = (teacher_valid & base_choice.ne(teacher_actions)).to(dtype=avg_pano_embeds.dtype)
                base_error_sum = base_error_sum + error_target.sum()

                gate_or_alarm = step_outs["gate"].float().clamp(1e-4, 1.0 - 1.0e-4)
                with cuda_autocast(enabled=False):
                    calib_raw = F.binary_cross_entropy(
                        gate_or_alarm.float(),
                        error_target.float(),
                        reduction="none",
                    )
                calibrate_loss_raw_sum = calibrate_loss_raw_sum + (
                    calib_raw * teacher_valid.to(dtype=calib_raw.dtype)
                ).sum()

                valid_count = int(teacher_valid.sum().item())
                if valid_count > 0:
                    gate_logits = step_outs["gate_raw"][teacher_valid].float()
                    target_valid = error_target[teacher_valid].float()
                    pos_count = float(target_valid.sum().item())
                    neg_count = float(valid_count) - pos_count
                    if pos_count > 0.0:
                        pos_weight_value = min(max(neg_count / max(pos_count, 1.0), 1.0), 8.0)
                    else:
                        pos_weight_value = 1.0
                    pos_weight = gate_logits.new_tensor(pos_weight_value)
                    with cuda_autocast(enabled=False):
                        calib_opt = F.binary_cross_entropy_with_logits(
                            gate_logits,
                            target_valid,
                            pos_weight=pos_weight,
                            reduction="sum",
                        )
                    calibrate_loss_sum = calibrate_loss_sum + calib_opt

                if bool(getattr(self._efes_cfg(), "use_safe_kl", True)):
                    safe_loss_sum = safe_loss_sum + step_outs["kl_corrected_vs_base"].sum()

                if bool(getattr(self._efes_cfg(), "use_mi_proxy", True)):
                    mi_lb = self._info_nce_proxy(step_outs["mi_query"], step_outs["mi_key"])
                    mi_loss_sum = mi_loss_sum + (-mi_lb).sum()
                    mi_proxy_sum = mi_proxy_sum + mi_lb.sum()

                teacher_margin_delta = self._teacher_margin_delta(
                    base_logits=masked_base,
                    corrected_logits=step_outs["corrected_logits"],
                    teacher_actions=teacher_actions,
                    invalid_mask=invalid_candidate_mask,
                )
                teacher_margin_delta_sum = teacher_margin_delta_sum + teacher_margin_delta.sum()

                gate_binary = step_outs["gate"].ge(float(getattr(self._efes_cfg(), "gate_threshold", 0.5)))
                pos_count = teacher_valid & error_target.bool()
                neg_count = teacher_valid & error_target.logical_not().bool()
                gate_tp_sum = gate_tp_sum + (gate_binary & pos_count).to(dtype=avg_pano_embeds.dtype).sum()
                gate_fp_sum = gate_fp_sum + (gate_binary & neg_count).to(dtype=avg_pano_embeds.dtype).sum()
                auc_sum = auc_sum + self._binary_auc_from_scores(step_outs["u_tilde"], error_target)
                auc_count_sum = auc_count_sum + 1.0

            if feedback == "sample":
                c = torch.distributions.Categorical(probs=nav_probs)
                a_t = c.sample().detach()
                if mode == "train":
                    use_teacher = (
                        (teacher_actions >= 0)
                        & (torch.rand_like(a_t, dtype=torch.float) <= sample_ratio)
                    )
                    a_t = torch.where(use_teacher, teacher_actions, a_t)
            else:
                a_t = action_logits.argmax(dim=-1)
            cpu_a_t = a_t.cpu().numpy()

            safe_a_t = a_t.clamp(min=0, max=nav_outs["gmap_embeds"].size(1) - 1)
            selected_feats = nav_outs["gmap_embeds"][
                torch.arange(self.envs.num_envs, device=self.device), safe_a_t
            ]
            recurrent_state["prev_action_emb"] = state_model.project_action_features(selected_feats)
            recurrent_state["prev_self"] = step_outs["self_t"]
            recurrent_state["prev_rssm_h"] = step_outs["rssm_h_t"]
            recurrent_state["prev_z"] = step_outs["z_flat"]
            recurrent_state["prev_progress"] = step_outs["progress_t"].unsqueeze(-1)
            recurrent_state["prev_u_tilde"] = step_outs["u_tilde"]

            env_actions = []
            use_tryout = self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING
            for i, gmap in enumerate(self.gmaps):
                selected_idx = int(cpu_a_t[i])
                selected_vp = (
                    nav_inputs["gmap_vp_ids"][i][selected_idx]
                    if 0 <= selected_idx < len(nav_inputs["gmap_vp_ids"][i])
                    else None
                )
                selected_is_stop = selected_idx == 0
                selected_is_visited_node = (
                    selected_vp is not None
                    and not str(selected_vp).startswith("g")
                    and selected_idx != 0
                    and selected_vp in gmap.node_pos
                )
                if selected_is_stop or stepk == self.max_len - 1 or no_vp_left[i]:
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    if vp_stop_scores:
                        stop_vp = max(vp_stop_scores, key=lambda item: item[1])[0]
                        stop_pos = gmap.node_pos[stop_vp]
                    else:
                        stop_vp = cur_vp[i]
                        stop_pos = gmap.node_pos[cur_vp[i]]
                    back_path = (
                        [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]][1:]
                        if self.config.IL.back_algo == "control"
                        else None
                    )
                    env_actions.append(
                        {
                            "action": {
                                "act": 0,
                                "cur_vp": cur_vp[i],
                                "stop_vp": stop_vp,
                                "stop_pos": stop_pos,
                                "back_path": back_path,
                                "tryout": use_tryout,
                            }
                        }
                    )
                elif selected_is_visited_node:
                    target_vp = str(selected_vp)
                    target_pos = gmap.node_pos[target_vp]
                    back_path = (
                        [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][target_vp]][1:]
                        if self.config.IL.back_algo == "control" and target_vp != cur_vp[i]
                        else None
                    )
                    env_actions.append(
                        {
                            "action": {
                                "act": 5,
                                "cur_vp": cur_vp[i],
                                "target_vp": target_vp,
                                "target_pos": target_pos,
                                "back_path": back_path,
                                "tryout": use_tryout,
                            }
                        }
                    )
                    prev_vp[i] = target_vp
                else:
                    ghost_vp = selected_vp
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    back_path = (
                        [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]][1:]
                        if self.config.IL.back_algo == "control"
                        else None
                    )
                    env_actions.append(
                        {
                            "action": {
                                "act": 4,
                                "cur_vp": cur_vp[i],
                                "front_vp": front_vp,
                                "front_pos": front_pos,
                                "ghost_vp": ghost_vp,
                                "ghost_pos": ghost_pos,
                                "back_path": back_path,
                                "tryout": use_tryout,
                            }
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

            env_step_start = self._timing_now()
            outputs = self.envs.step(env_actions)
            timing_sums["env_step_sec"] += self._timing_now() - env_step_start
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            if mode == "eval":
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    if ep_id in self.stat_eps:
                        continue
                    gt_path = np.array(self.gt_data[str(ep_id)]["locations"]).astype(np.float32)
                    pred_path = np.array(info["position"]["position"])
                    distances = np.array(info["position"]["distance"])
                    metric = {
                        "steps_taken": info["steps_taken"],
                        "distance_to_goal": distances[-1],
                        "success": 1.0 if distances[-1] <= 3.0 else 0.0,
                        "oracle_success": 1.0 if (distances <= 3.0).any() else 0.0,
                        "path_length": float(np.linalg.norm(pred_path[1:] - pred_path[:-1], axis=1).sum()),
                    }
                    metric["collisions"] = info["collisions"]["count"] / max(len(pred_path), 1)
                    gt_length = distances[0]
                    metric["spl"] = metric["success"] * gt_length / max(gt_length, metric["path_length"])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric["ndtw"] = np.exp(-dtw_distance / (len(gt_path) * 3.0))
                    metric["sdtw"] = metric["ndtw"] * metric["success"]
                    if active_eval_diag is not None:
                        diag = active_eval_diag[i]
                        metric["mean_u_t_raw"] = self._mean_float_list(diag["u_raw"])
                        metric["mean_u_tilde"] = self._mean_float_list(diag["u_tilde"])
                        metric["episode_u_tilde_bar"] = metric["mean_u_tilde"]
                        metric["mean_gate"] = self._mean_float_list(diag["gate"])
                        metric["mean_progress"] = self._mean_float_list(diag["progress"])
                    self.stat_eps[ep_id] = metric
                    evaluated_eps = len(self.stat_eps)
                    eval_total = (
                        int(self.pbar.total)
                        if self.pbar is not None and self.pbar.total is not None
                        else evaluated_eps
                    )
                    running_means = {
                        "success": sum(v["success"] for v in self.stat_eps.values()) / evaluated_eps,
                        "spl": sum(v["spl"] for v in self.stat_eps.values()) / evaluated_eps,
                        "ndtw": sum(v["ndtw"] for v in self.stat_eps.values()) / evaluated_eps,
                        "sdtw": sum(v["sdtw"] for v in self.stat_eps.values()) / evaluated_eps,
                        "distance_to_goal": sum(v["distance_to_goal"] for v in self.stat_eps.values()) / evaluated_eps,
                    }
                    self._write_live_eval_progress(
                        ep_id=ep_id,
                        metric=metric,
                        running_means=running_means,
                        evaluated_eps=evaluated_eps,
                        eval_total=eval_total,
                    )
                    if self.pbar is not None:
                        self.pbar.update()

            if mode == "infer":
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            "position": info["position_infer"]["position"][0],
                            "heading": info["position_infer"]["heading"][0],
                            "stop": False,
                        }
                    ]
                    for pos, heading in zip(
                        info["position_infer"]["position"][1:],
                        info["position_infer"]["heading"][1:],
                    ):
                        if pos != self.path_eps[ep_id][-1]["position"]:
                            self.path_eps[ep_id].append(
                                {
                                    "position": pos,
                                    "heading": heading,
                                    "stop": False,
                                }
                            )
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]["stop"] = True
                    if self.pbar is not None:
                        self.pbar.update()

            done_indices = [i for i, done in enumerate(dones) if done]
            if done_indices:
                keep_indices = [i for i in range(len(dones)) if not dones[i]]
                for i in reversed(done_indices):
                    not_done_index.pop(i)
                    self.envs.pause_at(i)
                    observations.pop(i)
                    self.gmaps.pop(i)
                    prev_vp.pop(i)
                    active_pos_histories.pop(i)
                    if active_eval_diag is not None:
                        active_eval_diag.pop(i)

                if keep_indices:
                    keep_tensor = torch.tensor(keep_indices, device=self.device, dtype=torch.long)
                    for key in (
                        "prev_self",
                        "prev_rssm_h",
                        "prev_z",
                        "prev_action_emb",
                        "prev_progress",
                        "prev_u_tilde",
                    ):
                        recurrent_state[key] = recurrent_state[key].index_select(0, keep_tensor)
                    topo_bank = topo_bank.index_select(keep_indices)

            if self.envs.num_envs == 0:
                break

            for i in range(self.envs.num_envs):
                active_pos_histories[i].append(
                    torch.as_tensor(cur_pos[i], device=self.device, dtype=torch.float32)
                )

            batch_prep_start = self._timing_now()
            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                max_length=instr_max_len,
                pad_id=instr_pad_id,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)
            batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            timing_sums["batch_prep_sec"] += self._timing_now() - batch_prep_start

        if mode == "train":
            total_actions_f = max(float(total_actions), 1.0)
            nav_loss = nav_loss_sum / total_actions_f
            fe_loss_opt = fe_loss_sum / total_actions_f
            fe_loss_raw = fe_loss_raw_sum / total_actions_f
            calibrate_loss_opt = calibrate_loss_sum / total_actions_f
            calibrate_loss_raw = calibrate_loss_raw_sum / total_actions_f
            safe_loss = safe_loss_sum / total_actions_f
            mi_loss = mi_loss_sum / total_actions_f
            total_loss = (
                nav_loss
                + float(getattr(self._efes_cfg(), "lambda_fe", 1.0)) * fe_loss_opt
                + float(getattr(self._efes_cfg(), "lambda_cal", 1.0)) * calibrate_loss_opt
                + (float(getattr(self._efes_cfg(), "lambda_safe", 0.0)) * safe_loss if bool(getattr(self._efes_cfg(), "use_safe_kl", True)) else 0.0)
                + (float(getattr(self._efes_cfg(), "lambda_mi", 0.0)) * mi_loss if bool(getattr(self._efes_cfg(), "use_mi_proxy", True)) else 0.0)
            )
            self.loss = self.loss + ml_weight * total_loss

            scalar_metrics = {
                "total_loss": float(total_loss.detach().item()),
                "nav_loss": float(nav_loss.detach().item()),
                "fe_loss_opt": float(fe_loss_opt.detach().item()),
                "fe_loss_raw": float(fe_loss_raw.detach().item()),
                "calibrate_loss_opt": float(calibrate_loss_opt.detach().item()),
                "calibrate_loss_raw": float(calibrate_loss_raw.detach().item()),
                "safe_loss": float(safe_loss.detach().item()),
                "mi_loss": float(mi_loss.detach().item()),
            }
            sum_metrics = {
                "gate_sum": float(gate_sum.detach().item()),
                "gate_p90_sum": float(gate_p90_sum.detach().item()),
                "gate_p90_count": float(gate_p90_count.detach().item()),
                "lambda_hat_sum": float(lambda_hat_sum.detach().item()),
                "delta_abs_max_sum": float(delta_abs_max_sum.detach().item()),
                "u_raw_sum": float(u_raw_sum.detach().item()),
                "u_tilde_sum": float(u_tilde_sum.detach().item()),
                "kl_micro_sum": float(kl_micro_sum.detach().item()),
                "nll_macro_sum": float(nll_macro_sum.detach().item()),
                "recon_sq_error_sum": float(recon_sq_error_sum.detach().item()),
                "sigma_mean_sum": float(sigma_mean_sum.detach().item()),
                "sigma_low_sum": float(sigma_low_sum.detach().item()),
                "sigma_high_sum": float(sigma_high_sum.detach().item()),
                "progress_sum": float(progress_sum.detach().item()),
                "score_before_sum": float(score_before_sum.detach().item()),
                "score_after_sum": float(score_after_sum.detach().item()),
                "teacher_margin_delta_sum": float(teacher_margin_delta_sum.detach().item()),
                "kl_corrected_vs_base_sum": float(kl_corrected_vs_base_sum.detach().item()),
                "gate_tp_sum": float(gate_tp_sum.detach().item()),
                "gate_fp_sum": float(gate_fp_sum.detach().item()),
                "auc_sum": float(auc_sum.detach().item()),
                "auc_count_sum": float(auc_count_sum.detach().item()),
                "base_error_sum": float(base_error_sum.detach().item()),
                "mi_proxy_sum": float(mi_proxy_sum.detach().item()),
                "topo_update_sum": float(topo_update_sum.detach().item()),
            }
            count_metrics = {"total_actions": float(total_actions)}

            scalar_metrics = self._reduce_float_dict(scalar_metrics, average=True)
            sum_metrics = self._reduce_float_dict(sum_metrics, average=False)
            count_metrics = self._reduce_float_dict(count_metrics, average=False)
            timing_metrics = self._reduce_float_dict(timing_sums, average=True)

            total_actions_global = max(float(count_metrics["total_actions"]), 1.0)
            base_error_global = max(float(sum_metrics["base_error_sum"]), 1.0)
            non_error_global = max(total_actions_global - float(sum_metrics["base_error_sum"]), 1.0)
            gate_p90_count_global = max(float(sum_metrics["gate_p90_count"]), 1.0)
            auc_count_global = max(float(sum_metrics["auc_count_sum"]), 1.0)
            self._last_step_metrics = {
                **scalar_metrics,
                "contract_version": self.contract_version,
                "total_actions": float(count_metrics["total_actions"]),
                "total_topo_updates": float(sum_metrics["topo_update_sum"]),
                "u_t_raw_mean": float(sum_metrics["u_raw_sum"]) / total_actions_global,
                "u_tilde_mean": float(sum_metrics["u_tilde_sum"]) / total_actions_global,
                "kl_micro_mean": float(sum_metrics["kl_micro_sum"]) / total_actions_global,
                "nll_macro_mean": float(sum_metrics["nll_macro_sum"]) / total_actions_global,
                "recon_sq_error_mean": float(sum_metrics["recon_sq_error_sum"]) / total_actions_global,
                "sigma_mean": float(sum_metrics["sigma_mean_sum"]) / total_actions_global,
                "sigma_hit_low_rate": float(sum_metrics["sigma_low_sum"]) / total_actions_global,
                "sigma_hit_high_rate": float(sum_metrics["sigma_high_sum"]) / total_actions_global,
                "gate_mean": float(sum_metrics["gate_sum"]) / total_actions_global,
                "gate_p90": float(sum_metrics["gate_p90_sum"]) / gate_p90_count_global,
                "lambda_hat_mean": float(sum_metrics["lambda_hat_sum"]) / total_actions_global,
                "delta_abs_max": float(sum_metrics["delta_abs_max_sum"]) / total_actions_global,
                "kl_corrected_vs_base": float(sum_metrics["kl_corrected_vs_base_sum"]) / total_actions_global,
                "teacher_margin_delta": float(sum_metrics["teacher_margin_delta_sum"]) / total_actions_global,
                "base_error_rate": float(sum_metrics["base_error_sum"]) / total_actions_global,
                "gate_tp_rate": float(sum_metrics["gate_tp_sum"]) / base_error_global,
                "gate_fp_rate": float(sum_metrics["gate_fp_sum"]) / non_error_global,
                "surprise_auc_base_error": float(sum_metrics["auc_sum"]) / auc_count_global,
                "mi_proxy": float(sum_metrics["mi_proxy_sum"]) / total_actions_global,
                "progress_mean": float(sum_metrics["progress_sum"]) / total_actions_global,
                "score_before_mean": float(sum_metrics["score_before_sum"]) / total_actions_global,
                "score_after_mean": float(sum_metrics["score_after_sum"]) / total_actions_global,
                "topo_update_ratio": float(sum_metrics["topo_update_sum"]) / total_actions_global,
                **timing_metrics,
            }
            for key, value in self._last_step_metrics.items():
                if key not in {"contract_version"}:
                    self.logs[key].append(float(value))
