from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict, deque
from copy import deepcopy
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
try:
    from torch.amp import GradScaler as _TorchGradScaler

    def make_grad_scaler():
        return _TorchGradScaler("cuda")

except ImportError:
    from torch.cuda.amp import GradScaler as _CudaGradScaler

    def make_grad_scaler():
        return _CudaGradScaler()

try:
    from torch.amp import autocast as _torch_autocast

    def cuda_autocast(*, enabled: bool = True):
        return _torch_autocast("cuda", enabled=enabled)

except ImportError:
    from torch.cuda.amp import autocast as _cuda_autocast

    def cuda_autocast(*, enabled: bool = True):
        return _cuda_autocast(enabled=enabled)

try:
    from habitat import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

import tqdm
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.obs_transformers import apply_obs_transforms_batch

from fastdtw import fastdtw
from habitat_extensions.measures import NDTW
from vlnce_baselines.agents.efes_v2_agent import EFESV2Agent
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.datasets.state_label_builder import StateLabelBuilder
from vlnce_baselines.models.efes_v2.topo_state_bank import TopoStateBankV2
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.trainers.train_statenav_v6_stage1 import StateNavV6Trainer, _state_model


@baseline_registry.register_trainer(name="EFESV2")
class EFESV2Trainer(StateNavV6Trainer):
    stage_name = "EFESV2"

    def __init__(self, config=None):
        super().__init__(config)
        self._phase_name = "phase1"
        self._phase2_started = False
        self._c_micro_median_buffer: Deque[float] = deque(maxlen=1000)
        self._c_micro_median_value = 10.0
        self._c_micro_refresh_counter = 0
        self._efes_optimizer_group_names: List[str] = []
        self._router_boot_weight = 0.0

    def _efes_cfg(self):
        return self.config.EFES_V2

    def _logging_cfg(self):
        return self.config.EFES_V2.LOGGING

    def _log_prior_post_metrics_enabled(self) -> bool:
        return bool(getattr(self._logging_cfg(), "log_diag_metrics", True))

    def _log_prior_post_eval_enabled(self) -> bool:
        return bool(getattr(self._logging_cfg(), "log_diag_metrics", True))

    def _v4_trace_enabled(self) -> bool:
        return False

    def _make_statenav_components(self) -> None:
        if self.world_size > 1:
            etp_dim = self.policy.net.module.output_size
        else:
            etp_dim = self.policy.net.output_size

        self.statenav_agent = EFESV2Agent.from_config(
            self.config,
            x_dim=etp_dim,
            lang_dim=etp_dim,
            cand_dim=etp_dim,
        ).to(self.device)

        self.label_builder = StateLabelBuilder.from_config(
            self.config.STATENAV.LABEL_BUILDER,
            state_bins=int(self.config.STATENAV.state_bins),
        )

    def _get_unfrozen_etp_params(self) -> List[Tensor]:
        keywords = [str(k) for k in getattr(self._efes_cfg(), "etp_unfreeze_keywords", [])]
        if not keywords:
            return []
        params: List[Tensor] = []
        for name, param in self.policy.named_parameters():
            if any(keyword in name for keyword in keywords):
                params.append(param)
        return params

    def _set_train_phase(self, phase2: bool) -> None:
        self._phase_name = "phase2" if phase2 else "phase1"
        self._phase2_started = bool(phase2)

        for param in self.policy.parameters():
            param.requires_grad_(False)
        for param in self.statenav_agent.parameters():
            param.requires_grad_(True)
        _state_model(self.statenav_agent).apply_static_freeze()

        etp_params: List[Tensor] = []
        if phase2:
            etp_params = self._get_unfrozen_etp_params()
            for param in etp_params:
                param.requires_grad_(True)

        efes_params = [param for param in self.statenav_agent.parameters() if param.requires_grad]
        param_groups: List[Dict[str, Any]] = [
            {
                "params": efes_params,
                "lr": float(
                    self._efes_cfg().efes_lr_phase2 if phase2 else self._efes_cfg().efes_lr_phase1
                ),
                "group_name": "efes",
            }
        ]
        if phase2 and etp_params:
            param_groups.append(
                {
                    "params": etp_params,
                    "lr": 0.0,
                    "base_lr": float(self._efes_cfg().etp_lr_phase2),
                    "group_name": "etp",
                }
            )
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
        self._efes_optimizer_group_names = [str(group.get("group_name", "")) for group in self.optimizer.param_groups]

    def _configure_trainable_modules(self) -> None:
        self._set_train_phase(phase2=False)

    def _maybe_activate_phase2(self) -> None:
        if self._phase2_started:
            return
        if int(self._train_iteration) < int(self._efes_cfg().phase1_iters):
            return
        logger.info(
            "EFES switching to phase2 at iteration %d: unfreezing selected ETP layers.",
            int(self._train_iteration),
        )
        self._set_train_phase(phase2=True)

    def _apply_phase_lrs(self) -> None:
        if not self._phase2_started:
            return
        warmup_iters = max(int(self._efes_cfg().phase2_warmup_iters), 1)
        phase2_iter = max(int(self._train_iteration) - int(self._efes_cfg().phase1_iters), 0)
        warm = min(1.0, float(phase2_iter + 1) / float(warmup_iters))
        for group in self.optimizer.param_groups:
            name = str(group.get("group_name", ""))
            if name == "efes":
                group["lr"] = float(self._efes_cfg().efes_lr_phase2)
            elif name == "etp":
                base_lr = float(group.get("base_lr", self._efes_cfg().etp_lr_phase2))
                group["lr"] = base_lr * warm

    @staticmethod
    def _step_metric_columns() -> List[str]:
        return [
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
            "plan_loss",
            "kl_loss",
            "node_nll_loss",
            "pi_cal_loss",
            "router_boot_loss",
            "total_actions",
            "total_macro_updates",
            "c_micro_mean",
            "c_macro_diag_mean",
            "u_macro_mean",
            "g_t_mean",
            "a_t_mean",
            "pi_t_mean",
            "macro_valid_ratio",
            "mode_entropy_mean",
            "mode_hist_argmax",
            "etp_unfrozen_grad_norm",
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

    def _build_step_metric_record(
        self,
        interval_step: int,
        interval_size: int,
        sample_ratio: float,
        grad_norm: float,
    ) -> Dict[str, Any]:
        record = super()._build_step_metric_record(interval_step, interval_size, sample_ratio, grad_norm)
        record["phase"] = self._phase_name
        return record

    def _log_step_metric_record(self, record: Dict[str, Any]) -> None:
        if not self._is_main_process():
            return
        step_log_every = max(int(self._logging_cfg().step_log_every), 1)
        if int(record["iteration"]) % step_log_every != 0:
            return
        logger.info(
            "[EFESV2 %s %06d/%06d | interval %03d/%03d] total=%.4f plan=%.4f kl=%.4f node_nll=%.4f pi_cal=%.4f boot=%.4f grad=%.4f lr=%.2e"
            " | c=%.3f/%.3f u=%.3f g=%.3f A=%.3f pi=%.3f macro=%.3f modeH=%.3f modes=%s etp=%.4f"
            " | sec(step=%.2f roll=%.2f bw=%.2f opt=%.2f lang=%.2f wp=%.2f pano=%.2f nav=%.2f efes=%.2f env=%.2f prep=%.2f)",
            str(record.get("phase", "phase1")),
            int(record["iteration"]),
            int(self.config.IL.iters),
            int(record.get("interval_step", 0)),
            int(record.get("interval_size", 0)),
            float(record.get("total_loss", 0.0)),
            float(record.get("plan_loss", 0.0)),
            float(record.get("kl_loss", 0.0)),
            float(record.get("node_nll_loss", 0.0)),
            float(record.get("pi_cal_loss", 0.0)),
            float(record.get("router_boot_loss", 0.0)),
            float(record.get("grad_norm", 0.0)),
            float(record.get("lr", 0.0)),
            float(record.get("c_micro_mean", 0.0)),
            float(record.get("c_macro_diag_mean", 0.0)),
            float(record.get("u_macro_mean", 0.0)),
            float(record.get("g_t_mean", 0.0)),
            float(record.get("a_t_mean", 0.0)),
            float(record.get("pi_t_mean", 0.0)),
            float(record.get("macro_valid_ratio", 0.0)),
            float(record.get("mode_entropy_mean", 0.0)),
            str(record.get("mode_hist_argmax", "-")),
            float(record.get("etp_unfrozen_grad_norm", 0.0)),
            float(record.get("train_step_sec", 0.0)),
            float(record.get("rollout_sec", 0.0)),
            float(record.get("backward_sec", 0.0)),
            float(record.get("optimizer_sec", 0.0)),
            float(record.get("lang_sec", 0.0)),
            float(record.get("waypoint_sec", 0.0)),
            float(record.get("panorama_sec", 0.0)),
            float(record.get("navigation_sec", 0.0)),
            float(record.get("statenav_sec", 0.0)),
            float(record.get("env_step_sec", 0.0)),
            float(record.get("batch_prep_sec", 0.0)),
        )

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        import glob

        ckpt_dir = self.config.CHECKPOINT_FOLDER
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f"ckpt.iter{iteration}.pth")
        torch.save(
            obj={
                "policy_state_dict": self.policy.state_dict(),
                "statenav_state_dict": self.statenav_agent.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "scaler_state": None if not hasattr(self, "scaler") else self.scaler.state_dict(),
                "iteration": iteration,
                "best_metric": self.best_metric,
                "efes_phase": self._phase_name,
            },
            f=ckpt_path,
        )
        if is_best:
            best_path = os.path.join(ckpt_dir, "best_val.pth")
            torch.save(
                obj={
                    "policy_state_dict": self.policy.state_dict(),
                    "statenav_state_dict": self.statenav_agent.state_dict(),
                    "config": self.config,
                    "optim_state": self.optimizer.state_dict(),
                    "scaler_state": None if not hasattr(self, "scaler") else self.scaler.state_dict(),
                    "iteration": iteration,
                    "best_metric": self.best_metric,
                    "efes_phase": self._phase_name,
                },
                f=best_path,
            )

        max_keep = max(int(self._efes_cfg().max_keep_checkpoints), 1)
        existing = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt.iter*.pth")))
        while len(existing) > max_keep:
            oldest = existing.pop(0)
            os.remove(oldest)
            logger.info("Removed old EFES checkpoint: %s", oldest)

    def train(self):
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
        if int(start_iter) >= int(self._efes_cfg().phase1_iters):
            self._set_train_phase(phase2=True)

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

        logger.info("EFES-v2 training starts...")
        for idx in range(start_iter, total_iter, log_every):
            self._maybe_activate_phase2()
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
                    "[EFES-v2 summary %06d | %s] %s",
                    cur_iter,
                    self._phase_name,
                    ", ".join(f"{k}: {v:.4f}" for k, v in summary.items()),
                )
                for k, v in summary.items():
                    writer.add_scalar(f"loss/{k}", v, cur_iter)
                self.save_checkpoint(cur_iter, is_best=is_best)

    def _train_interval(self, interval, ml_weight, sample_ratio):
        if self._phase2_started:
            self.policy.train()
        else:
            self.policy.eval()
        self.statenav_agent.train()
        self.waypoint_predictor.eval()

        use_tqdm = bool(self._logging_cfg().use_tqdm) and self._is_main_process()
        pbar = (
            tqdm.trange(interval, leave=False, dynamic_ncols=True, desc=f"EFESV2 {self._phase_name}")
            if use_tqdm
            else range(interval)
        )
        self.logs = defaultdict(list)
        for idx in pbar:
            self._apply_phase_lrs()
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
            trainable_params = []
            unfrozen_etp_params = []
            for group in self.optimizer.param_groups:
                trainable_params.extend([param for param in group["params"] if param.grad is not None])
                if str(group.get("group_name", "")) == "etp":
                    unfrozen_etp_params.extend([param for param in group["params"] if param.grad is not None])
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=float(self._efes_cfg().grad_clip_norm),
            )
            if unfrozen_etp_params:
                etp_unfrozen_grad_norm = torch.norm(
                    torch.stack([param.grad.detach().norm(2) for param in unfrozen_etp_params]),
                    2,
                )
            else:
                etp_unfrozen_grad_norm = torch.zeros((), device=self.device)
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
                    "etp_unfrozen_grad_norm": float(etp_unfrozen_grad_norm.detach().item()),
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
                            "phase": self._phase_name,
                            "interval": f"{idx + 1}/{interval}",
                            "step": f"{self._train_iteration}/{self.config.IL.iters}",
                            "loss": f"{step_record.get('total_loss', 0.0):.4f}",
                        }
                    )
        return deepcopy(self.logs)

    def _refresh_c_micro_median(self) -> None:
        if len(self._c_micro_median_buffer) < 8:
            return
        self._c_micro_median_value = float(np.median(np.asarray(self._c_micro_median_buffer, dtype=np.float32)))

    def _update_c_micro_median(self, c_micro: Tensor) -> None:
        values = c_micro.detach().cpu().tolist()
        for value in values:
            self._c_micro_median_buffer.append(float(value))
        self._c_micro_refresh_counter += len(values)
        if self._c_micro_refresh_counter >= 100:
            self._refresh_c_micro_median()
            self._c_micro_refresh_counter = 0

    def _compute_router_boot_weight(self) -> float:
        phase1_iters = max(int(self._efes_cfg().phase1_iters), 1)
        if self._phase2_started:
            return float(self._efes_cfg().lambda_boot_end)
        progress = min(max(float(self._train_iteration) / float(phase1_iters), 0.0), 1.0)
        start = float(self._efes_cfg().lambda_boot_start)
        end = float(self._efes_cfg().lambda_boot_end)
        return start + (end - start) * progress

    def _grounding_inputs(
        self,
        progress_history: Sequence[Deque[float]],
        pos_history: Sequence[Deque[Tensor]],
        vp_history: Sequence[Deque[Any]],
        current_pos: Sequence[Any],
        current_vp: Sequence[Any],
        prev_progress: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = len(current_pos)
        progress_ref = prev_progress.detach().clone()
        is_static = torch.zeros(batch_size, device=self.device, dtype=torch.bool)
        eps_path = float(self._efes_cfg().eps_path)
        for batch_idx in range(batch_size):
            if progress_history[batch_idx]:
                progress_ref[batch_idx, 0] = float(progress_history[batch_idx][0])
            current_pos_tensor = torch.as_tensor(
                current_pos[batch_idx],
                device=self.device,
                dtype=torch.float32,
            )
            recent_positions = list(pos_history[batch_idx]) + [current_pos_tensor]
            path_delta = 0.0
            if len(recent_positions) >= 2:
                for left, right in zip(recent_positions[:-1], recent_positions[1:]):
                    path_delta += float(torch.norm(right - left, p=2).item())
            unique_vp_delta = self._unique_vp_delta(vp_history[batch_idx], current_vp[batch_idx])
            is_static[batch_idx] = bool(path_delta < eps_path and unique_vp_delta == 0)
        return progress_ref, is_static

    @staticmethod
    def _unique_vp_delta(history: Deque[Any], current_vp: Any) -> int:
        values = [vp for vp in history if vp is not None]
        if current_vp is not None:
            values.append(current_vp)
        if not values:
            return 0
        return max(len(set(values)) - 1, 0)

    @staticmethod
    def _history_mean(history: Deque[float]) -> float:
        if not history:
            return 0.0
        return float(sum(history) / len(history))

    @staticmethod
    def _recovery_hist_string(mode_counts: Dict[int, float]) -> str:
        total = max(sum(mode_counts.values()), 1.0)
        return "|".join(
            f"{mode}:{mode_counts.get(mode, 0.0) / total:.3f}" for mode in (0, 1, 2, 3)
        )

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

        if mode == "train":
            self.statenav_agent.train()
        else:
            self.statenav_agent.eval()
        policy_grad_enabled = bool(mode == "train" and self._phase2_started)
        policy_ctx = torch.enable_grad if policy_grad_enabled else torch.no_grad

        lang_start = self._timing_now()
        with policy_ctx():
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

        total_actions = 0
        total_macro_updates = 0
        plan_loss_sum = torch.zeros((), device=self.device)
        kl_loss_sum = torch.zeros((), device=self.device)
        node_nll_sum = torch.zeros((), device=self.device)
        pi_cal_loss_sum = torch.zeros((), device=self.device)
        router_boot_loss_sum = torch.zeros((), device=self.device)
        ddp_anchor_sum = torch.zeros((), device=self.device)
        c_micro_sum = torch.zeros((), device=self.device)
        c_macro_diag_sum = torch.zeros((), device=self.device)
        u_macro_sum = torch.zeros((), device=self.device)
        g_t_sum = torch.zeros((), device=self.device)
        a_t_sum = torch.zeros((), device=self.device)
        pi_t_sum = torch.zeros((), device=self.device)
        macro_valid_sum = torch.zeros((), device=self.device)
        mode_count_sum = torch.zeros(4, device=self.device)
        mode_entropy_sum = torch.zeros((), device=self.device)

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
        topo_bank = TopoStateBankV2.create(
            num_envs=self.envs.num_envs,
            max_nodes=int(self._efes_cfg().max_nodes),
            feat_dim=int(state_model.x_dim),
            device=self.device,
        )
        active_backtrack_histories = [dict() for _ in range(self.envs.num_envs)]
        active_c_micro_histories = [
            deque(maxlen=int(self._efes_cfg().history_window)) for _ in range(self.envs.num_envs)
        ]
        active_c_macro_histories = [
            deque(maxlen=int(self._efes_cfg().history_window)) for _ in range(self.envs.num_envs)
        ]
        active_progress_histories = [
            deque(maxlen=int(self._efes_cfg().history_window)) for _ in range(self.envs.num_envs)
        ]
        active_pos_histories = [
            deque(maxlen=int(self._efes_cfg().history_window)) for _ in range(self.envs.num_envs)
        ]
        active_vp_histories = [
            deque(maxlen=int(self._efes_cfg().history_window)) for _ in range(self.envs.num_envs)
        ]
        active_pending_pi: List[List[Dict[str, Any]]] = [[] for _ in range(self.envs.num_envs)]
        active_eval_diag = [
            {"a_t": [], "c_micro": [], "c_macro": [], "g_t": [], "pi_t": [], "macro_valid": []}
            for _ in range(self.envs.num_envs)
        ] if mode == "eval" else None

        for stepk in range(self.max_len):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]

            with policy_ctx():
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
                pano_denom = torch.sum(pano_masks, 1, keepdim=True).clamp_min(1.0)
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

                navigation_start = self._timing_now()
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
                timing_sums["navigation_sec"] += self._timing_now() - navigation_start

            invalid_candidate_mask = nav_inputs["gmap_masks"].logical_not()
            visited_candidate_mask = nav_inputs["gmap_visited_masks"]
            candidate_mask = invalid_candidate_mask | visited_candidate_mask
            frontier_candidate_mask, local_frontier_candidate_mask = self._build_frontier_candidate_tensors(
                nav_inputs["gmap_vp_ids"],
                cur_vp,
            )
            history_backtrack_values, history_valid_mask = self._build_history_backtrack_tensors(
                nav_inputs["gmap_vp_ids"],
                active_backtrack_histories,
                current_step=stepk,
            )
            candidate_counts = torch.tensor(
                [len(cand_vp_i) for cand_vp_i in cand_vp],
                device=self.device,
                dtype=torch.long,
            )
            current_comp_feat = state_model.macro_surprise.compress(avg_pano_embeds).detach()
            current_topo_novelty = state_model.macro_surprise.cosine_novelty(
                current_comp_feat,
                topo_bank.bank_feat,
                topo_bank.bank_mask,
            )
            topo_update_mask = topo_bank.build_update_mask(
                current_vp_ids=cur_vp,
                candidate_counts=candidate_counts,
                current_step=stepk + 1,
                topo_novelty=current_topo_novelty,
                decision_candidate_min=int(self._efes_cfg().node_decision_candidate_min),
                timeout_steps=int(self._efes_cfg().timeout_steps),
                novelty_threshold=float(self._efes_cfg().novelty_threshold),
                cooldown_steps=int(self._efes_cfg().cooldown_steps),
            )
            c_micro_recent_mean = torch.tensor(
                [self._history_mean(history) for history in active_c_micro_histories],
                device=self.device,
                dtype=avg_pano_embeds.dtype,
            )
            c_macro_recent_mean = torch.tensor(
                [self._history_mean(history) for history in active_c_macro_histories],
                device=self.device,
                dtype=avg_pano_embeds.dtype,
            )
            revisit_count = torch.tensor(
                [float(sum(1 for vp in history if vp == cur_vp_i)) for history, cur_vp_i in zip(active_vp_histories, cur_vp)],
                device=self.device,
                dtype=avg_pano_embeds.dtype,
            )
            loop_evidence = torch.tensor(
                [
                    1.0
                    if (cur_vp_i is not None and cur_vp_i in history and self._unique_vp_delta(history, cur_vp_i) <= 1)
                    else 0.0
                    for history, cur_vp_i in zip(active_vp_histories, cur_vp)
                ],
                device=self.device,
                dtype=avg_pano_embeds.dtype,
            )
            frontier_size = frontier_candidate_mask.to(avg_pano_embeds.dtype).sum(dim=-1)
            ground_progress_ref, ground_is_static = self._grounding_inputs(
                progress_history=active_progress_histories,
                pos_history=active_pos_histories,
                vp_history=active_vp_histories,
                current_pos=cur_pos,
                current_vp=cur_vp,
                prev_progress=recurrent_state["prev_progress"],
            )

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
                prev_prior_alpha=recurrent_state["prev_prior_alpha"],
                prev_topo_novelty=recurrent_state["prev_topo_novelty"],
                prev_progress_gap=recurrent_state["prev_progress_gap"],
                c_micro_recent_mean=c_micro_recent_mean,
                c_macro_recent_mean=c_macro_recent_mean,
                ground_progress_ref=ground_progress_ref,
                ground_is_static=ground_is_static,
                topo_bank_feat=topo_bank.bank_feat,
                topo_bank_mask=topo_bank.bank_mask,
                macro_valid_mask=topo_update_mask,
                candidate_mask=candidate_mask,
                frontier_size=frontier_size,
                revisit_count=revisit_count,
                loop_evidence=loop_evidence,
                history_backtrack_values=history_backtrack_values,
                history_valid_mask=history_valid_mask,
                frontier_mask=frontier_candidate_mask,
                local_frontier_mask=local_frontier_candidate_mask,
                lang_mask=txt_masks,
            )
            timing_sums["statenav_sec"] += self._timing_now() - statenav_start

            topo_bank.commit(
                update_mask=topo_update_mask,
                current_vp_ids=cur_vp,
                compressed_feat=step_outs["compressed_node_feat"],
                current_step=stepk + 1,
            )

            c_micro_sum = c_micro_sum + step_outs["C_micro"].sum()
            macro_valid_mask = step_outs["macro_valid_mask"]
            if bool(macro_valid_mask.any().item()):
                c_macro_diag_sum = c_macro_diag_sum + step_outs["C_macro_diag"][macro_valid_mask].sum()
            g_t_sum = g_t_sum + step_outs["g_t"].sum()
            a_t_sum = a_t_sum + step_outs["A_t"].sum()
            pi_t_sum = pi_t_sum + step_outs["pi_t"].sum()
            u_macro_sum = u_macro_sum + step_outs["U_macro"].sum()
            macro_valid_sum = macro_valid_sum + macro_valid_mask.to(torch.float32).sum()
            total_macro_updates += int(macro_valid_mask.sum().item())
            mode_entropy_sum = mode_entropy_sum + step_outs["mode_entropy"].sum()
            for mode_idx in range(4):
                mode_count_sum[mode_idx] = mode_count_sum[mode_idx] + step_outs["argmax_mode"].eq(mode_idx).sum()

            if mode == "eval" and active_eval_diag is not None:
                for i in range(self.envs.num_envs):
                    active_eval_diag[i]["a_t"].append(float(step_outs["A_t"][i].detach().item()))
                    active_eval_diag[i]["c_micro"].append(float(step_outs["C_micro"][i].detach().item()))
                    active_eval_diag[i]["c_macro"].append(float(step_outs["C_macro_diag"][i].detach().item()))
                    active_eval_diag[i]["g_t"].append(float(step_outs["g_t"][i].detach().item()))
                    active_eval_diag[i]["pi_t"].append(float(step_outs["pi_t"][i].detach().item()))
                    active_eval_diag[i]["macro_valid"].append(float(step_outs["macro_valid_mask"][i].detach().item()))

            if mode == "train" or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs["gmap_vp_ids"], no_vp_left)
            else:
                teacher_actions = None

            behavior_logits = step_outs["fused_logits"] if mode == "train" else step_outs["hard_logits"]
            nav_probs = F.softmax(behavior_logits, dim=1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].detach().item()

            if mode == "train":
                loss_logits = self._make_teacher_safe_logits(
                    rescored_logits=behavior_logits,
                    fallback_logits=nav_logits,
                    teacher_actions=teacher_actions,
                    invalid_candidate_mask=invalid_candidate_mask,
                )
                plan_loss_sum = plan_loss_sum + F.cross_entropy(
                    loss_logits,
                    teacher_actions.long(),
                    ignore_index=-100,
                    reduction="sum",
                )
                kl_loss_sum = kl_loss_sum + step_outs["C_micro"].clamp_min(3.0).sum()
                node_weight = step_outs["macro_valid_mask"].to(dtype=step_outs["node_nll_loss"].dtype)
                node_nll_sum = node_nll_sum + (step_outs["node_nll_loss"] * node_weight).sum()
                ddp_anchor_sum = ddp_anchor_sum + (
                    0.0 * step_outs["pi_t"].sum()
                    + 0.0 * step_outs["alpha_prior"].sum()
                    + 0.0 * step_outs["alpha_progress"].sum()
                )

                median_value = float(self._c_micro_median_value)
                future_window = max(int(self._efes_cfg().future_window), 1)
                pending_pi_values = []
                pending_targets = []
                for i in range(self.envs.num_envs):
                    updated_entries: List[Dict[str, Any]] = []
                    current_c = float(step_outs["C_micro"][i].detach().item())
                    for entry in active_pending_pi[i]:
                        entry["sum"] += current_c
                        entry["count"] += 1
                        if entry["count"] >= future_window:
                            pending_pi_values.append(entry["pi"])
                            avg_future = entry["sum"] / max(entry["count"], 1)
                            pending_targets.append(1.0 if avg_future < median_value else 0.0)
                        else:
                            updated_entries.append(entry)
                    updated_entries.append({"pi": step_outs["pi_t"][i : i + 1], "sum": 0.0, "count": 0})
                    active_pending_pi[i] = updated_entries
                if pending_pi_values:
                    with cuda_autocast(enabled=False):
                        pi_cal_loss_sum = pi_cal_loss_sum + F.binary_cross_entropy(
                            torch.cat(pending_pi_values, dim=0).float(),
                            torch.tensor(pending_targets, device=self.device, dtype=torch.float32),
                            reduction="sum",
                        )
                router_boot_loss_sum = router_boot_loss_sum + F.cross_entropy(
                    step_outs["mode_logits"],
                    step_outs["router_boot_targets"].long(),
                    reduction="sum",
                )
                self._update_c_micro_median(step_outs["C_micro"])

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
                a_t = behavior_logits.argmax(dim=-1)
            cpu_a_t = a_t.cpu().numpy()

            safe_a_t = a_t.clamp(min=0, max=nav_outs["gmap_embeds"].size(1) - 1)
            selected_feats = nav_outs["gmap_embeds"][
                torch.arange(self.envs.num_envs, device=self.device), safe_a_t
            ]
            recurrent_state["prev_action_emb"] = state_model.project_action_features(selected_feats)
            recurrent_state["prev_self"] = step_outs["h_t"]
            recurrent_state["prev_rssm_h"] = step_outs["rssm_h_t"]
            recurrent_state["prev_z"] = step_outs["z_flat"]
            recurrent_state["prev_progress"] = step_outs["progress_t"]
            recurrent_state["prev_prior_alpha"] = step_outs["alpha_prior"]
            recurrent_state["prev_topo_novelty"] = step_outs["topo_novelty"]
            recurrent_state["prev_progress_gap"] = step_outs["g_t"]

            for i in range(self.envs.num_envs):
                self._update_backtrack_history(
                    history=active_backtrack_histories[i],
                    vp_ids=nav_inputs["gmap_vp_ids"][i],
                    invalid_mask_row=invalid_candidate_mask[i],
                    visited_mask_row=visited_candidate_mask[i],
                    score_row=nav_logits[i],
                    combined_health_row=torch.zeros_like(nav_logits[i]),
                    curve_health_row=torch.zeros_like(nav_logits[i]),
                    step_index=stepk,
                )

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
                        metric["mean_a_t"] = self._mean_float_list(diag["a_t"])
                        metric["mean_c_micro"] = self._mean_float_list(diag["c_micro"])
                        metric["mean_c_macro"] = self._mean_float_list(diag["c_macro"])
                        metric["mean_g_t"] = self._mean_float_list(diag["g_t"])
                        metric["mean_pi_t"] = self._mean_float_list(diag["pi_t"])
                        metric["mean_macro_valid"] = self._mean_float_list(diag["macro_valid"])
                    self.stat_eps[ep_id] = metric
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
                    active_backtrack_histories.pop(i)
                    active_c_micro_histories.pop(i)
                    active_c_macro_histories.pop(i)
                    active_progress_histories.pop(i)
                    active_pos_histories.pop(i)
                    active_vp_histories.pop(i)
                    active_pending_pi.pop(i)
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
                        "prev_prior_alpha",
                        "prev_topo_novelty",
                        "prev_progress_gap",
                    ):
                        recurrent_state[key] = recurrent_state[key].index_select(0, keep_tensor)
                    topo_bank = topo_bank.index_select(keep_indices)

            if self.envs.num_envs == 0:
                break

            for i in range(self.envs.num_envs):
                active_c_micro_histories[i].append(float(step_outs["C_micro"][i].detach().item()))
                active_c_macro_histories[i].append(float(step_outs["C_macro_diag"][i].detach().item()))
                active_progress_histories[i].append(float(step_outs["progress_t"][i].detach().item()))
                active_pos_histories[i].append(
                    torch.as_tensor(cur_pos[i], device=self.device, dtype=torch.float32)
                )
                active_vp_histories[i].append(cur_vp[i])

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
            total_macro_updates_f = max(float(total_macro_updates), 1.0)
            plan_loss = plan_loss_sum / total_actions_f
            kl_loss = kl_loss_sum / total_actions_f
            node_nll_loss = node_nll_sum / total_macro_updates_f
            pi_cal_loss = pi_cal_loss_sum / total_actions_f
            router_boot_loss = router_boot_loss_sum / total_actions_f
            self._router_boot_weight = self._compute_router_boot_weight()
            total_loss = (
                plan_loss
                + float(self._efes_cfg().lambda_kl) * kl_loss
                + float(self._efes_cfg().lambda_node) * node_nll_loss
                + float(self._efes_cfg().lambda_pi) * pi_cal_loss
                + float(self._router_boot_weight) * router_boot_loss
                + ddp_anchor_sum
            )
            self.loss = self.loss + ml_weight * total_loss

            scalar_metrics = {
                "total_loss": float(total_loss.detach().item()),
                "plan_loss": float(plan_loss.detach().item()),
                "kl_loss": float(kl_loss.detach().item()),
                "node_nll_loss": float(node_nll_loss.detach().item()),
                "pi_cal_loss": float(pi_cal_loss.detach().item()),
                "router_boot_loss": float(router_boot_loss.detach().item()),
            }
            sum_metrics = {
                "c_micro_sum": float(c_micro_sum.detach().item()),
                "c_macro_diag_sum": float(c_macro_diag_sum.detach().item()),
                "u_macro_sum": float(u_macro_sum.detach().item()),
                "g_t_sum": float(g_t_sum.detach().item()),
                "a_t_sum": float(a_t_sum.detach().item()),
                "pi_t_sum": float(pi_t_sum.detach().item()),
                "macro_valid_sum": float(macro_valid_sum.detach().item()),
                "mode_entropy_sum": float(mode_entropy_sum.detach().item()),
                "mode_count_0": float(mode_count_sum[0].detach().item()),
                "mode_count_1": float(mode_count_sum[1].detach().item()),
                "mode_count_2": float(mode_count_sum[2].detach().item()),
                "mode_count_3": float(mode_count_sum[3].detach().item()),
            }
            count_metrics = {
                "total_actions": float(total_actions),
                "total_macro_updates": float(total_macro_updates),
            }

            scalar_metrics = self._reduce_float_dict(scalar_metrics, average=True)
            sum_metrics = self._reduce_float_dict(sum_metrics, average=False)
            count_metrics = self._reduce_float_dict(count_metrics, average=False)
            timing_metrics = self._reduce_float_dict(timing_sums, average=True)

            total_actions_global = max(float(count_metrics["total_actions"]), 1.0)
            total_macro_updates_global = max(float(count_metrics["total_macro_updates"]), 1.0)
            mode_hist = self._recovery_hist_string(
                {
                    0: float(sum_metrics["mode_count_0"]),
                    1: float(sum_metrics["mode_count_1"]),
                    2: float(sum_metrics["mode_count_2"]),
                    3: float(sum_metrics["mode_count_3"]),
                }
            )
            self._last_step_metrics = {
                **scalar_metrics,
                "total_actions": float(count_metrics["total_actions"]),
                "total_macro_updates": float(count_metrics["total_macro_updates"]),
                "c_micro_mean": float(sum_metrics["c_micro_sum"]) / total_actions_global,
                "c_macro_diag_mean": float(sum_metrics["c_macro_diag_sum"]) / total_macro_updates_global,
                "u_macro_mean": float(sum_metrics["u_macro_sum"]) / total_actions_global,
                "g_t_mean": float(sum_metrics["g_t_sum"]) / total_actions_global,
                "a_t_mean": float(sum_metrics["a_t_sum"]) / total_actions_global,
                "pi_t_mean": float(sum_metrics["pi_t_sum"]) / total_actions_global,
                "macro_valid_ratio": float(sum_metrics["macro_valid_sum"]) / total_actions_global,
                "mode_entropy_mean": float(sum_metrics["mode_entropy_sum"]) / total_actions_global,
                "mode_hist_argmax": mode_hist,
                **timing_metrics,
            }
            for key, value in scalar_metrics.items():
                self.logs[key].append(value)
            for key in (
                "c_micro_mean",
                "c_macro_diag_mean",
                "u_macro_mean",
                "g_t_mean",
                "a_t_mean",
                "pi_t_mean",
                "macro_valid_ratio",
                "mode_entropy_mean",
            ):
                self.logs[key].append(float(self._last_step_metrics[key]))
