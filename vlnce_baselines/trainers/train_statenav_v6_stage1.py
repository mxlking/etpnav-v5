import gc
import gzip
import json
import os
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re
import glob

import jsonlines
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from habitat import logger
except Exception:
    import logging

    logger = logging.getLogger(__name__)

import tqdm
import torch.distributed as distr
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from habitat_extensions.measures import NDTW
from habitat_extensions.task import ALL_ROLES_MASK, RxRVLNCEDatasetV1
from vlnce_baselines.agents.statenav_agent_v6 import StateNavAgentV6
from vlnce_baselines.common.env_utils import construct_envs
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.datasets.state_label_builder import StateLabelBuilder
from vlnce_baselines.models.graph_utils import GraphMap, heading_from_quaternion
from vlnce_baselines.models.statenav_v5.losses import (
    attention_transition_loss,
    latent_kl_loss,
    planner_loss,
    progress_curve_loss,
)
from vlnce_baselines.models.statenav_v6.topo_state_bank import TopoStateBank
from vlnce_baselines.ss_trainer_ETP import RLTrainer
from vlnce_baselines.utils import gather_list_and_concat, load_torch_checkpoint_compat

from fastdtw import fastdtw


def _state_model(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if isinstance(module, DDP) else module


_REPO_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RELEASE_R2R_CKPT = os.path.realpath(
    os.path.join(_REPO_ROOT, "data", "logs", "checkpoints", "release_r2r", "ckpt.iter12000.pth")
)


@baseline_registry.register_trainer(name="StateNavV6")
class StateNavV6Trainer(RLTrainer):
    stage_name = "Main"

    def __init__(self, config=None):
        super().__init__(config)
        self._train_iteration = 0
        self._resume_scaler_state = None
        self.best_metric = float("inf")
        self._last_step_metrics: Dict[str, float] = {}
        self._step_metrics_path: Optional[str] = None
        self._v4_trace_dir: Optional[str] = None
        self._v4_trace_written = 0
        self._v4_trace_checkpoint_index = 0

    def _topo_update_max_k(self) -> int:
        return int(getattr(self.config.STATENAV, "node_update_max_k", 5))

    def _topo_novelty_tau(self) -> float:
        return float(getattr(self.config.STATENAV, "node_novelty_tau", 0.35))

    def _topo_decision_candidate_min(self) -> int:
        return int(getattr(self.config.STATENAV, "node_decision_candidate_min", 2))

    def _load_gt_data(self) -> None:
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        gt_path = self.config.TASK_CONFIG.TASK.NDTW.GT_PATH

        if "{role}" in gt_path:
            roles = list(self.config.TASK_CONFIG.DATASET.ROLES)
            if ALL_ROLES_MASK in roles:
                roles = list(RxRVLNCEDatasetV1.annotation_roles)
            self.gt_data = {}
            for role in roles:
                with gzip.open(gt_path.format(split=split, role=role), "rt") as f:
                    self.gt_data.update(json.load(f))
        else:
            with gzip.open(gt_path.format(split=split), "rt") as f:
                self.gt_data = json.load(f)

    def _make_statenav_components(self) -> None:
        if self.world_size > 1:
            etp_dim = self.policy.net.module.output_size
        else:
            etp_dim = self.policy.net.output_size

        self.statenav_agent = StateNavAgentV6.from_config(
            self.config,
            x_dim=etp_dim,
            g_dim=etp_dim,
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
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    def _configure_trainable_modules(self) -> None:
        for param in self.policy.parameters():
            param.requires_grad_(False)

        state_model = _state_model(self.statenav_agent)
        for param in self.statenav_agent.parameters():
            param.requires_grad_(False)

        for module in (
            state_model.action_encoder,
            state_model.self_rssm_transition,
            state_model.self_rssm_correction,
            state_model.self_rssm_latent,
            state_model.node_surprise_head,
        ):
            for param in module.parameters():
                param.requires_grad_(True)

        trainable_params = [param for param in self.statenav_agent.parameters() if param.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.config.IL.lr)

    @staticmethod
    def _count_parameters(module: torch.nn.Module) -> Tuple[int, int]:
        total = 0
        trainable = 0
        for param in module.parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
        return total, trainable

    @staticmethod
    def _format_param_count(count: int) -> str:
        return f"{count:,} ({count / 1e6:.2f}M)"

    @staticmethod
    def _freeze_status(total: int, trainable: int) -> str:
        if total == 0:
            return "empty"
        if trainable == 0:
            return "frozen"
        if trainable == total:
            return "trainable"
        return "mixed"

    def _logging_cfg(self):
        return self.config.STATENAV.LOGGING

    def _log_prior_post_metrics_enabled(self) -> bool:
        return bool(getattr(self._logging_cfg(), "log_prior_post_metrics", False))

    def _log_prior_post_eval_enabled(self) -> bool:
        return bool(getattr(self._logging_cfg(), "log_prior_post_eval", False))

    def _v4_trace_enabled(self) -> bool:
        return bool(
            getattr(
                self._logging_cfg(),
                "save_v5_episode_traces",
                getattr(self._logging_cfg(), "save_v4_episode_traces", False),
            )
        )

    def _v4_trace_topk(self) -> int:
        return max(
            int(
                getattr(
                    self._logging_cfg(),
                    "v5_trace_topk",
                    getattr(self._logging_cfg(), "v4_trace_topk", 5),
                )
            ),
            1,
        )

    def _v4_trace_limit(self) -> int:
        return max(
            int(
                getattr(
                    self._logging_cfg(),
                    "v5_trace_max_episodes",
                    getattr(self._logging_cfg(), "v4_trace_max_episodes", 20),
                )
            ),
            0,
        )

    def _prepare_v4_trace_output(self, checkpoint_index: int) -> None:
        self._v4_trace_dir = None
        self._v4_trace_written = 0
        self._v4_trace_checkpoint_index = int(checkpoint_index)
        if not self._is_main_process() or not self._v4_trace_enabled():
            return
        trace_dir = os.path.join(
            self.config.RESULTS_DIR,
            str(
                getattr(
                    self._logging_cfg(),
                    "v5_trace_dirname",
                    getattr(self._logging_cfg(), "v4_trace_dirname", "v5_traces"),
                )
            ),
        )
        os.makedirs(trace_dir, exist_ok=True)
        self._v4_trace_dir = trace_dir

    @staticmethod
    def _safe_candidate_index(index: int, num_candidates: int) -> int:
        if num_candidates <= 0:
            return 0
        return int(max(min(index, num_candidates - 1), 0))

    @staticmethod
    def _masked_argmax_index(scores: Tensor, candidate_mask: Optional[Tensor]) -> int:
        if candidate_mask is None:
            return int(scores.argmax().item())
        masked_scores = scores.masked_fill(candidate_mask.bool(), -float("inf"))
        if torch.isfinite(masked_scores).any():
            return int(masked_scores.argmax().item())
        return int(scores.argmax().item())

    def _candidate_topk_trace(
        self,
        scores: Tensor,
        candidate_mask: Optional[Tensor],
        cand_health_scores: Optional[Tensor],
    ) -> List[Dict[str, float]]:
        if scores.ndim != 1:
            raise ValueError(f"scores must be 1D for trace export, got {tuple(scores.shape)}")
        if candidate_mask is not None:
            masked_scores = scores.masked_fill(candidate_mask.bool(), -float("inf"))
            valid_mask = candidate_mask.logical_not()
        else:
            masked_scores = scores
            valid_mask = torch.ones_like(scores, dtype=torch.bool)
        valid_count = int(valid_mask.sum().item())
        if valid_count <= 0:
            return []
        topk = min(self._v4_trace_topk(), valid_count)
        values, indices = torch.topk(masked_scores, k=topk, dim=0)
        entries: List[Dict[str, float]] = []
        for rank, (idx_tensor, value_tensor) in enumerate(zip(indices, values), start=1):
            idx = int(idx_tensor.item())
            entry = {
                "rank": int(rank),
                "idx": idx,
                "score": float(value_tensor.detach().item()),
            }
            if cand_health_scores is not None:
                entry["health"] = float(cand_health_scores[idx].detach().item())
            entries.append(entry)
        return entries

    @staticmethod
    def _format_curve_preview(curve: Sequence[float], limit: int = 3) -> str:
        preview = [f"{float(v):+.2f}" for v in list(curve)[: max(limit, 1)]]
        return "[" + ", ".join(preview) + "]"

    def _build_v4_video_debug_lines(
        self,
        *,
        step_index: int,
        selected_idx: int,
        top1_before_idx: int,
        top1_after_idx: int,
        teacher_idx: int,
        d_t: float,
        d_t_norm: float,
        dt_threshold_low: float,
        dt_threshold_high: float,
        health_scalar: float,
        selected_curve_health: float,
        selected_attn_health: float,
        selected_combined_health: float,
        health_bonus_mean: float,
        dt_temperature: float,
        intervention_level: float,
        score_before: float,
        score_after: float,
        progress_curve: Sequence[float],
        candidate_mask_row: Optional[Tensor],
        level3_policy: Optional[str] = None,
        level3_decision: Optional[str] = None,
        level3_target: Optional[str] = None,
    ) -> List[str]:
        valid_cands = -1
        if candidate_mask_row is not None:
            valid_cands = int(candidate_mask_row.logical_not().sum().item())
        line0 = (
            f"V6 step={int(step_index):02d} sel={int(selected_idx)} "
            f"b/a={int(top1_before_idx)}->{int(top1_after_idx)} lv={float(intervention_level):.0f}"
        )
        line1 = (
            f"D_t={float(d_t):.3f} dn={float(d_t_norm):.3f} "
            f"th={float(dt_threshold_low):.3f}/{float(dt_threshold_high):.3f} "
            f"temp={float(dt_temperature):.3f}"
        )
        line2 = (
            f"h_agent={float(health_scalar):+.3f} "
            f"h_curve={float(selected_curve_health):+.3f} "
            f"h_attn={float(selected_attn_health):+.3f} "
            f"h_sel={float(selected_combined_health):+.3f}"
        )
        line3 = (
            f"bonus={float(health_bonus_mean):+.3f} "
            f"score={float(score_before):+.2f}->{float(score_after):+.2f}"
        )
        line4 = (
            f"curve={self._format_curve_preview(progress_curve)} "
            f"teacher={int(teacher_idx)} valid={valid_cands}"
        )
        if level3_policy is not None or level3_decision is not None or level3_target is not None:
            line5 = (
                f"macro={level3_policy or '-'}:{level3_decision or '-'} "
                f"target={level3_target or '-'}"
            )
            return [line0, line1, line2, line3, line4, line5]
        return [line0, line1, line2, line3, line4]

    def _write_v4_episode_trace(
        self,
        episode_id: Any,
        metric: Dict[str, Any],
        step_trace: Sequence[Dict[str, Any]],
    ) -> None:
        if self._v4_trace_dir is None:
            return
        if self._v4_trace_written >= self._v4_trace_limit():
            return
        payload = {
            "episode_id": str(episode_id),
            "checkpoint_index": int(self._v4_trace_checkpoint_index),
            "split": str(self.config.EVAL.SPLIT),
            "metrics": metric,
            "steps": list(step_trace),
        }
        trace_path = os.path.join(
            self._v4_trace_dir,
            f"trace_ckpt_{self._v4_trace_checkpoint_index}_ep_{episode_id}.json",
        )
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._v4_trace_written += 1

    def _is_main_process(self) -> bool:
        return self.local_rank < 1

    def _distributed_ready(self) -> bool:
        return self.world_size > 1 and distr.is_available() and distr.is_initialized()

    def _reduce_float_dict(
        self,
        values: Dict[str, float],
        average: bool,
    ) -> Dict[str, float]:
        if not values:
            return {}
        if not self._distributed_ready():
            return {key: float(value) for key, value in values.items()}

        reduced: Dict[str, float] = {}
        for key, value in values.items():
            tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
            distr.all_reduce(tensor, op=distr.ReduceOp.SUM)
            if average:
                tensor /= float(self.world_size)
            reduced[key] = float(tensor.item())
        return reduced

    def _collect_module_summary_line(self, module_name: str, module: torch.nn.Module) -> str:
        total, trainable = self._count_parameters(_state_model(module))
        frozen = total - trainable
        return (
            f"[{module_name}] status={self._freeze_status(total, trainable)} | "
            f"trainable={self._format_param_count(trainable)} | "
            f"frozen={self._format_param_count(frozen)} | total={self._format_param_count(total)}"
        )

    @staticmethod
    def _step_metric_columns() -> List[str]:
        return [
            "iteration",
            "interval_step",
            "interval_size",
            "world_size",
            "envs_per_rank",
            "sample_ratio",
            "lr",
            "grad_norm",
            "total_loss",
            "plan_loss",
            "micro_kl_loss",
            "node_surprise_loss",
            "total_actions",
            "total_node_updates",
            "adapter_warmup",
            "d_t_fused_mean",
            "d_t_latent_mean",
            "d_t_node_mean",
            "d_t_norm_mean",
            "topo_novelty_mean",
            "topo_update_ratio",
            "dt_temperature_mean",
            "intervention_level_mean",
            "dt_threshold_low_mean",
            "dt_threshold_high_mean",
            "score_before_mean",
            "score_after_mean",
            "strict_candidate_count_mean",
            "expanded_candidate_count_mean",
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

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, float):
            if value == 0.0:
                return "0.000000"
            if abs(value) >= 1e3 or abs(value) < 1e-3:
                return f"{value:.6e}"
            return f"{value:.6f}"
        return str(value)

    def _prepare_step_metric_writer(self) -> None:
        self._step_metrics_path = None
        if not self._is_main_process():
            return
        if not bool(self._logging_cfg().write_step_metrics):
            return

        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        self._step_metrics_path = os.path.join(
            self.config.CHECKPOINT_FOLDER,
            str(self._logging_cfg().step_metrics_filename),
        )
        append_mode = bool(
            self.config.IL.is_requeue and os.path.exists(self._step_metrics_path)
        )
        expected_header = "\t".join(self._step_metric_columns())
        if append_mode:
            with open(self._step_metrics_path, "r", encoding="utf-8") as f:
                existing_header = f.readline().rstrip("\n")
            if existing_header != expected_header:
                backup_path = f"{self._step_metrics_path}.legacy_before_v6_schema"
                backup_idx = 1
                while os.path.exists(backup_path):
                    backup_path = (
                        f"{self._step_metrics_path}.legacy_before_v6_schema.{backup_idx}"
                    )
                    backup_idx += 1
                shutil.copy2(self._step_metrics_path, backup_path)
                logger.warning(
                    "Existing V6 step metrics header mismatch. Backed up legacy file to %s "
                    "and restarting %s with the updated V6 schema.",
                    backup_path,
                    self._step_metrics_path,
                )
                append_mode = False
        file_mode = "a" if append_mode else "w"
        needs_header = (not append_mode) or os.path.getsize(self._step_metrics_path) == 0

        with open(self._step_metrics_path, file_mode, encoding="utf-8") as f:
            if needs_header:
                f.write(expected_header + "\n")

        logger.info("Step metrics will be written to %s", self._step_metrics_path)

    def _append_step_metric_record(self, record: Dict[str, Any]) -> None:
        if not self._step_metrics_path:
            return

        columns = self._step_metric_columns()
        row = [self._format_metric_value(record.get(column, "")) for column in columns]
        with open(self._step_metrics_path, "a", encoding="utf-8") as f:
            f.write("\t".join(row) + "\n")

    def _build_step_metric_record(
        self,
        interval_step: int,
        interval_size: int,
        sample_ratio: float,
        grad_norm: float,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "iteration": int(self._train_iteration),
            "interval_step": int(interval_step),
            "interval_size": int(interval_size),
            "world_size": int(self.world_size),
            "envs_per_rank": int(self.config.NUM_ENVIRONMENTS),
            "sample_ratio": float(sample_ratio),
            "lr": float(self.optimizer.param_groups[0].get("lr", 0.0)),
            "grad_norm": float(grad_norm),
        }
        record.update(self._last_step_metrics)
        return record

    def _log_step_metric_record(self, record: Dict[str, Any]) -> None:
        if not self._is_main_process():
            return

        step_log_every = max(int(self._logging_cfg().step_log_every), 1)
        if int(record["iteration"]) % step_log_every != 0:
            return

        base_message = (
            "[step %06d/%06d | interval %03d/%03d] total=%.4f | plan=%.4f | micro_kl=%.4f | node=%.4f | "
            "grad=%.4f | lr=%.2e | adapter=%.3f | sample=%.4f | env/rank=%d | world=%d"
        )
        base_args: Tuple[Any, ...] = (
            int(record["iteration"]),
            int(self.config.IL.iters),
            int(record.get("interval_step", 0)),
            int(record.get("interval_size", 0)),
            float(record.get("total_loss", 0.0)),
            float(record.get("plan_loss", 0.0)),
            float(record.get("micro_kl_loss", 0.0)),
            float(record.get("node_surprise_loss", 0.0)),
            float(record.get("grad_norm", 0.0)),
            float(record.get("lr", 0.0)),
            float(record.get("adapter_warmup", 1.0)),
            float(record.get("sample_ratio", 0.0)),
            int(record.get("envs_per_rank", 0)),
            int(record.get("world_size", 1)),
        )
        if self._log_prior_post_metrics_enabled():
            logger.info(
                base_message
                + " | dt=%.3f dl=%.3f dn=%.3f dtn=%.3f tn=%.3f tu=%.3f tt=%.3f lv=%.2f tl=%.3f th=%.3f sc=%.2f ec=%.2f"
                + " | sec(step=%.2f roll=%.2f bw=%.2f opt=%.2f lang=%.2f wp=%.2f pano=%.2f nav=%.2f st=%.2f env=%.2f prep=%.2f)",
                *base_args,
                float(record.get("d_t_fused_mean", 0.0)),
                float(record.get("d_t_latent_mean", 0.0)),
                float(record.get("d_t_node_mean", 0.0)),
                float(record.get("d_t_norm_mean", 0.0)),
                float(record.get("topo_novelty_mean", 0.0)),
                float(record.get("topo_update_ratio", 0.0)),
                float(record.get("dt_temperature_mean", 0.0)),
                float(record.get("intervention_level_mean", 0.0)),
                float(record.get("dt_threshold_low_mean", 0.0)),
                float(record.get("dt_threshold_high_mean", 0.0)),
                float(record.get("strict_candidate_count_mean", 0.0)),
                float(record.get("expanded_candidate_count_mean", 0.0)),
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
        else:
            logger.info(
                base_message
                + " | sec(step=%.2f roll=%.2f bw=%.2f opt=%.2f lang=%.2f wp=%.2f pano=%.2f nav=%.2f st=%.2f env=%.2f prep=%.2f)",
                *base_args,
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

    def _timing_now(self) -> float:
        return time.perf_counter()

    def _collect_module_tree_lines(
        self,
        module_name: str,
        module: torch.nn.Module,
        max_depth: int = 2,
    ) -> List[str]:
        base_module = _state_model(module)
        lines: List[str] = []

        def visit(name: str, current: torch.nn.Module, depth: int) -> None:
            total, trainable = self._count_parameters(current)
            indent = "  " * depth
            lines.append(
                f"{indent}{name}: {current.__class__.__name__} | "
                f"status={self._freeze_status(total, trainable)} | "
                f"trainable={self._format_param_count(trainable)} / total={self._format_param_count(total)}"
            )
            if depth >= max_depth:
                return
            for child_name, child in current.named_children():
                next_name = f"{name}.{child_name}" if name else child_name
                visit(next_name, child, depth + 1)

        visit(module_name, base_module, 0)
        return lines

    def _collect_named_parameter_lines(
        self,
        module_name: str,
        module: torch.nn.Module,
        requires_grad: bool,
        limit: Optional[int] = None,
    ) -> List[str]:
        base_module = _state_model(module)
        lines: List[str] = []
        for name, param in base_module.named_parameters():
            if param.requires_grad != requires_grad:
                continue
            full_name = f"{module_name}.{name}" if name else module_name
            shape = tuple(param.shape)
            lines.append(
                f"{full_name}: shape={shape}, numel={param.numel():,}, dtype={param.dtype}"
            )
        if limit is not None and len(lines) > limit:
            omitted = len(lines) - limit
            lines = lines[:limit] + [f"... ({omitted} more parameters omitted)"]
        return lines

    def _log_module_report(self, module_name: str, module: torch.nn.Module) -> None:
        total, trainable = self._count_parameters(_state_model(module))
        frozen = total - trainable
        logger.info(
            "[%s] status=%s | trainable=%s | frozen=%s | total=%s",
            module_name,
            self._freeze_status(total, trainable),
            self._format_param_count(trainable),
            self._format_param_count(frozen),
            self._format_param_count(total),
        )

        logger.info("[%s] module tree:", module_name)
        for line in self._collect_module_tree_lines(module_name, module):
            logger.info("  %s", line)

        trainable_lines = self._collect_named_parameter_lines(
            module_name,
            module,
            requires_grad=True,
            limit=None,
        )
        if trainable_lines:
            logger.info("[%s] trainable parameters (%d tensors):", module_name, len(trainable_lines))
            for line in trainable_lines:
                logger.info("  %s", line)
        else:
            logger.info("[%s] trainable parameters: <none>", module_name)

    def _log_state_architecture(self, module_name: str, module: torch.nn.Module) -> None:
        base_module = _state_model(module)
        logger.info("[%s] full architecture:", module_name)
        logger.info("\n%s", base_module)

        logger.info("[%s] child modules:", module_name)
        for child_name, child in base_module.named_children():
            logger.info("  [%s.%s]\n%s", module_name, child_name, child)

    def _collect_module_report_lines(self, module_name: str, module: torch.nn.Module) -> List[str]:
        total, trainable = self._count_parameters(_state_model(module))
        frozen = total - trainable
        lines = [
            f"[{module_name}] status={self._freeze_status(total, trainable)} | "
            f"trainable={self._format_param_count(trainable)} | "
            f"frozen={self._format_param_count(frozen)} | total={self._format_param_count(total)}",
            f"[{module_name}] module tree:",
        ]
        lines.extend(f"  {line}" for line in self._collect_module_tree_lines(module_name, module))

        trainable_lines = self._collect_named_parameter_lines(
            module_name,
            module,
            requires_grad=True,
            limit=None,
        )
        if trainable_lines:
            lines.append(f"[{module_name}] trainable parameters ({len(trainable_lines)} tensors):")
            lines.extend(f"  {line}" for line in trainable_lines)
        else:
            lines.append(f"[{module_name}] trainable parameters: <none>")
        return lines

    def _collect_architecture_lines(self, module_name: str, module: torch.nn.Module) -> List[str]:
        base_module = _state_model(module)
        lines = [
            f"[{module_name}] full architecture:",
            "",
            str(base_module),
            f"[{module_name}] child modules:",
        ]
        for child_name, child in base_module.named_children():
            lines.append(f"  [{module_name}.{child_name}]")
            lines.append(str(child))
        return lines

    def _collect_optimizer_report_lines(self) -> List[str]:
        lines: List[str] = []
        unique_params = {}
        for group_idx, group in enumerate(self.optimizer.param_groups):
            group_total = 0
            group_trainable = 0
            for param in group["params"]:
                unique_params[id(param)] = param
                count = param.numel()
                group_total += count
                if param.requires_grad:
                    group_trainable += count
            lines.append(
                f"[optimizer] group={group_idx} lr={group.get('lr')} tensors={len(group['params'])} | "
                f"trainable={self._format_param_count(group_trainable)} | "
                f"total={self._format_param_count(group_total)}"
            )

        total = sum(param.numel() for param in unique_params.values())
        trainable = sum(param.numel() for param in unique_params.values() if param.requires_grad)
        lines.append(
            f"[optimizer] unique tensors={len(unique_params)} | "
            f"trainable={self._format_param_count(trainable)} | "
            f"total={self._format_param_count(total)}"
        )
        return lines

    def _write_trainability_report(self, lines: Sequence[str]) -> Optional[str]:
        if not self._is_main_process():
            return None

        report_dir = self.config.CHECKPOINT_FOLDER
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"{self.__class__.__name__}_model_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return report_path

    def _log_optimizer_report(self) -> None:
        unique_params = {}
        for group_idx, group in enumerate(self.optimizer.param_groups):
            group_total = 0
            group_trainable = 0
            for param in group["params"]:
                unique_params[id(param)] = param
                count = param.numel()
                group_total += count
                if param.requires_grad:
                    group_trainable += count
            logger.info(
                "[optimizer] group=%d lr=%s tensors=%d | trainable=%s | total=%s",
                group_idx,
                group.get("lr"),
                len(group["params"]),
                self._format_param_count(group_trainable),
                self._format_param_count(group_total),
            )

        total = sum(param.numel() for param in unique_params.values())
        trainable = sum(param.numel() for param in unique_params.values() if param.requires_grad)
        logger.info(
            "[optimizer] unique tensors=%d | trainable=%s | total=%s",
            len(unique_params),
            self._format_param_count(trainable),
            self._format_param_count(total),
        )

    def _log_trainability_report(self) -> None:
        full_lines = ["========== StateNav Trainability Report =========="]
        full_lines.extend(self._collect_module_report_lines("ETP.policy", self.policy))
        full_lines.extend(self._collect_module_report_lines("StateNav.agent", self.statenav_agent))
        full_lines.extend(self._collect_architecture_lines("ETP.policy", self.policy))
        full_lines.extend(self._collect_architecture_lines("StateNav.agent", self.statenav_agent))
        full_lines.extend(self._collect_optimizer_report_lines())
        full_lines.append("==================================================")

        report_path = self._write_trainability_report(full_lines)
        if not self._is_main_process():
            return

        summary_lines = [
            "========== StateNav Trainability Summary ==========" ,
            self._collect_module_summary_line("ETP.policy", self.policy),
            self._collect_module_summary_line("StateNav.agent", self.statenav_agent),
        ]
        summary_lines.extend(self._collect_optimizer_report_lines())
        if report_path is not None:
            summary_lines.append(f"Full model report saved to {report_path}")
        summary_lines.append("===================================================")

        for line in summary_lines:
            logger.info(line)

        if bool(self._logging_cfg().log_full_model_report_to_runtime):
            for line in full_lines:
                logger.info(line)

    @staticmethod
    def _align_state_dict_prefix(state_dict: Dict[str, Any], model: nn.Module) -> Dict[str, Any]:
        if not state_dict:
            return state_dict

        ckpt_keys = list(state_dict.keys())
        model_keys = list(model.state_dict().keys())
        if not ckpt_keys or not model_keys:
            return state_dict

        if set(ckpt_keys) == set(model_keys):
            return state_dict

        def _normalize_module_tokens(key: str) -> str:
            return ".".join(part for part in key.split(".") if part != "module")

        normalized_model_keys: Dict[str, str] = {}
        ambiguous_model_keys = set()
        for model_key in model_keys:
            normalized_key = _normalize_module_tokens(model_key)
            if normalized_key in normalized_model_keys and normalized_model_keys[normalized_key] != model_key:
                ambiguous_model_keys.add(normalized_key)
            else:
                normalized_model_keys[normalized_key] = model_key

        remapped_state_dict: Dict[str, Any] = {}
        matched_count = 0
        collision = False
        for ckpt_key, value in state_dict.items():
            normalized_key = _normalize_module_tokens(ckpt_key)
            if normalized_key in normalized_model_keys and normalized_key not in ambiguous_model_keys:
                target_key = normalized_model_keys[normalized_key]
                matched_count += 1
                if target_key in remapped_state_dict:
                    collision = True
                    break
                remapped_state_dict[target_key] = value
            else:
                if ckpt_key in remapped_state_dict:
                    collision = True
                    break
                remapped_state_dict[ckpt_key] = value

        if not collision and matched_count == len(ckpt_keys):
            return remapped_state_dict

        ckpt_has_module = all(k.startswith("module.") for k in ckpt_keys)
        model_has_module = all(k.startswith("module.") for k in model_keys)

        if ckpt_has_module and not model_has_module:
            return {k[len("module."):]: v for k, v in state_dict.items()}
        if model_has_module and not ckpt_has_module:
            return {f"module.{k}": v for k, v in state_dict.items()}
        return state_dict

    def _load_stage_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint_realpath = os.path.realpath(checkpoint_path)
        ckpt = load_torch_checkpoint_compat(
            checkpoint_realpath,
            map_location="cpu",
        )

        has_statenav_state = "statenav_state_dict" in ckpt
        if self.stage_name == "Main" and not self.config.IL.is_requeue and not has_statenav_state:
            if checkpoint_realpath != _RELEASE_R2R_CKPT:
                raise ValueError(
                    "Fresh V6 experiments in ETPNav_selected must start from "
                    f"{_RELEASE_R2R_CKPT}, but got {checkpoint_realpath}"
                )

        policy_load_msg = None
        if "policy_state_dict" in ckpt:
            policy_state_dict = self._align_state_dict_prefix(ckpt["policy_state_dict"], self.policy)
            policy_load_msg = self.policy.load_state_dict(policy_state_dict, strict=False)
        elif "state_dict" in ckpt:
            policy_state_dict = self._align_state_dict_prefix(ckpt["state_dict"], self.policy)
            policy_load_msg = self.policy.load_state_dict(policy_state_dict, strict=False)
        else:
            raise KeyError(f"Checkpoint {checkpoint_path} has neither state_dict nor policy_state_dict")

        policy_missing = len(policy_load_msg.missing_keys)
        policy_unexpected = len(policy_load_msg.unexpected_keys)
        logger.info(
            "[Policy CKPT] missing_keys(%d) unexpected_keys(%d)",
            policy_missing,
            policy_unexpected,
        )
        if policy_missing or policy_unexpected:
            raise RuntimeError(
                f"Invalid Policy checkpoint load for {checkpoint_path}: "
                f"missing_keys={policy_missing}, unexpected_keys={policy_unexpected}"
            )

        if has_statenav_state:
            statenav_state_dict = self._align_state_dict_prefix(ckpt["statenav_state_dict"], self.statenav_agent)
            load_msg = self.statenav_agent.load_state_dict(statenav_state_dict, strict=False)
            statenav_missing = len(load_msg.missing_keys)
            statenav_unexpected = len(load_msg.unexpected_keys)
            logger.info(
                "[StateNav CKPT] missing_keys(%d) unexpected_keys(%d)",
                statenav_missing,
                statenav_unexpected,
            )
            if statenav_missing or statenav_unexpected:
                raise RuntimeError(
                    f"Invalid StateNav checkpoint load for {checkpoint_path}: "
                    f"missing_keys={statenav_missing}, unexpected_keys={statenav_unexpected}"
                )
        else:
            logger.info("[StateNav CKPT] missing_keys(0) unexpected_keys(0) [fresh initialization]")

        if has_statenav_state and self.config.IL.is_requeue and "optim_state" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optim_state"])
            except Exception as exc:
                logger.warning("Unable to restore optimizer state cleanly: %s", exc)

        self._resume_scaler_state = ckpt.get("scaler_state") if has_statenav_state else None
        self.best_metric = float(ckpt.get("best_metric", self.best_metric)) if has_statenav_state else float("inf")
        start_iter = int(ckpt.get("iteration", 0)) if has_statenav_state else 0
        if has_statenav_state:
            logger.info("Loaded StageNav checkpoint: %s at iteration %d", checkpoint_path, start_iter)
        else:
            logger.info(
                "Loaded trained ETPNav checkpoint into frozen backbone: %s. StateNav modules keep fresh initialization.",
                checkpoint_path,
            )
        return start_iter

    def _initialize_policy(
        self,
        config,
        load_from_ckpt: bool,
        observation_space,
        action_space,
    ):
        start_iter = super()._initialize_policy(
            config=config,
            load_from_ckpt=False,
            observation_space=observation_space,
            action_space=action_space,
        )
        self._make_statenav_components()
        self._configure_trainable_modules()

        if not load_from_ckpt and not config.IL.is_requeue:
            raise ValueError(
                "StateNav V6 must start from a trained ETPNav navigation checkpoint. "
                "Set IL.load_from_ckpt=True and provide IL.ckpt_to_load."
            )

        # if load_from_ckpt:
        #     if config.IL.is_requeue:
        #         import glob

        #         ckpt_list = list(
        #             filter(os.path.isfile, glob.glob(os.path.join(config.CHECKPOINT_FOLDER, "*")))
        #         )
        #         ckpt_list.sort(key=os.path.getmtime)
        #         checkpoint_path = ckpt_list[-1]
        #     else:
        #         checkpoint_path = config.IL.ckpt_to_load
        #         if not checkpoint_path:
        #             raise ValueError(
        #                 "IL.ckpt_to_load is empty. Stage1 must load a trained ETPNav navigation checkpoint."
        #             )
        #         if not os.path.isfile(checkpoint_path):
        #             raise FileNotFoundError(checkpoint_path)

        if load_from_ckpt:
            if config.IL.is_requeue:
                if config.IL.ckpt_to_load and os.path.isfile(config.IL.ckpt_to_load):
                    checkpoint_path = config.IL.ckpt_to_load
                else:
                    import glob

                    ckpt_list = sorted(
                        glob.glob(os.path.join(config.CHECKPOINT_FOLDER, "*.pth")),
                        key=os.path.getmtime,
                    )
                    if not ckpt_list:
                        raise FileNotFoundError(
                            f"No .pth checkpoint found in {config.CHECKPOINT_FOLDER}"
                        )
                    checkpoint_path = ckpt_list[-1]
            else:
                checkpoint_path = config.IL.ckpt_to_load
                if not checkpoint_path:
                    raise ValueError("IL.ckpt_to_load is empty.")
                if not os.path.isfile(checkpoint_path):
                    raise FileNotFoundError(checkpoint_path)
            start_iter = self._load_stage_checkpoint(checkpoint_path)

        self._wrap_statenav_for_ddp()
        self._log_trainability_report()
        return start_iter

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
            },
            f=ckpt_path,
        )
        if is_best:
            shutil.copy2(ckpt_path, os.path.join(ckpt_dir, "best.pth"))

        max_keep = int(self.config.STATENAV.max_keep_checkpoints)
        def _iter_from_ckpt_path(path: str) -> int:
            m = re.search(r"ckpt\.iter(\d+)\.pth$", os.path.basename(path))
            return int(m.group(1)) if m else -1

        existing = sorted(
            glob.glob(os.path.join(ckpt_dir, "ckpt.iter*.pth")),
            key=_iter_from_ckpt_path,
        )
        while len(existing) > max_keep:
            oldest = existing.pop(0)
            os.remove(oldest)

    def _compute_kl_warmup(self) -> float:
        warmup_steps = max(int(self.config.STATENAV.kl_warmup_steps), 1)
        return min(1.0, float(self._train_iteration + 1) / float(warmup_steps))

    def _compute_attn_warmup(self) -> float:
        warmup_steps = int(self.config.STATENAV.attn_warmup_steps)
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self._train_iteration + 1) / float(warmup_steps))

    def _compute_adapter_warmup(self) -> float:
        warmup_steps = int(self.config.STATENAV.adapter_warmup_steps)
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(self._train_iteration + 1) / float(warmup_steps))

    def _eval_action_source(self) -> str:
        source = str(getattr(self.config.STATENAV, "eval_action_source", "statenav")).strip().lower()
        if source in {"state", "statenav", "rescored"}:
            return "statenav"
        if source in {"etp", "base", "raw"}:
            return "etp"
        logger.warning(
            "Unknown STATENAV.eval_action_source=%s. Falling back to 'statenav'.",
            source,
        )
        return "statenav"

    def _behavior_logits(
        self,
        mode: str,
        nav_logits: Tensor,
        rescored_logits: Tensor,
    ) -> Tensor:
        if mode == "eval" and self._eval_action_source() == "etp":
            return nav_logits
        return rescored_logits

    @staticmethod
    def _level3_decision_name(code: int) -> str:
        return {
            0: "none",
            1: "backtrack",
            2: "stop",
            3: "fallback_stop",
            4: "forced_stop",
            5: "continue",
        }.get(int(code), f"unknown_{int(code)}")

    def _prune_backtrack_history(self, history: Dict[str, Dict[str, float]]) -> None:
        max_items = max(int(self.config.STATENAV.backtrack_history_size), 1)
        if len(history) <= max_items:
            return
        ordered = sorted(history.items(), key=lambda item: float(item[1].get("step", -1.0)), reverse=True)
        history.clear()
        history.update(dict(ordered[:max_items]))

    def _build_history_backtrack_tensors(
        self,
        candidate_vp_ids: Sequence[Sequence[Any]],
        backtrack_histories: Sequence[Dict[str, Dict[str, float]]],
        current_step: int,
    ) -> tuple[Tensor, Tensor]:
        batch_size = len(candidate_vp_ids)
        max_candidates = max((len(vp_ids) for vp_ids in candidate_vp_ids), default=0)
        values = torch.zeros(batch_size, max_candidates, device=self.device, dtype=torch.float32)
        valid = torch.zeros(batch_size, max_candidates, device=self.device, dtype=torch.bool)
        recency_bias = float(self.config.STATENAV.backtrack_recency_bias)

        for batch_idx, vp_ids in enumerate(candidate_vp_ids):
            history = backtrack_histories[batch_idx]
            for cand_idx, vp_id in enumerate(vp_ids):
                if vp_id is None:
                    continue
                record = history.get(str(vp_id))
                if record is None:
                    continue
                age = max(float(current_step) - float(record.get("step", current_step)), 0.0)
                value = float(record.get("value", 0.0)) - recency_bias * age
                values[batch_idx, cand_idx] = float(value)
                valid[batch_idx, cand_idx] = True
        return values, valid

    def _build_frontier_candidate_tensors(
        self,
        candidate_vp_ids: Sequence[Sequence[Any]],
        current_vps: Sequence[Optional[str]],
    ) -> tuple[Tensor, Tensor]:
        batch_size = len(candidate_vp_ids)
        max_candidates = max((len(vp_ids) for vp_ids in candidate_vp_ids), default=0)
        frontier = torch.zeros(batch_size, max_candidates, device=self.device, dtype=torch.bool)
        local_frontier = torch.zeros_like(frontier)

        for batch_idx, (vp_ids, cur_vp) in enumerate(zip(candidate_vp_ids, current_vps)):
            gmap = self.gmaps[batch_idx]
            for cand_idx, vp_id in enumerate(vp_ids):
                if vp_id is None:
                    continue
                vp_name = str(vp_id)
                if not vp_name.startswith("g"):
                    continue
                frontier[batch_idx, cand_idx] = True
                if cur_vp is not None and cur_vp in gmap.ghost_fronts.get(vp_name, set()):
                    local_frontier[batch_idx, cand_idx] = True

            # If the current node has no directly attached frontier entries,
            # keep the strict candidate set non-degenerate by treating the
            # visible frontier pool as the local set for that row.
            if bool(frontier[batch_idx].any().item()) and not bool(local_frontier[batch_idx].any().item()):
                local_frontier[batch_idx] = frontier[batch_idx]

        return frontier, local_frontier

    def _update_backtrack_history(
        self,
        *,
        history: Dict[str, Dict[str, float]],
        vp_ids: Sequence[Any],
        invalid_mask_row: Tensor,
        visited_mask_row: Tensor,
        score_row: Tensor,
        combined_health_row: Tensor,
        curve_health_row: Tensor,
        step_index: int,
    ) -> None:
        for cand_idx, vp_id in enumerate(vp_ids):
            if vp_id is None:
                continue
            vp_name = str(vp_id)
            if vp_name.startswith("g"):
                continue
            if cand_idx >= invalid_mask_row.numel() or bool(invalid_mask_row[cand_idx].item()):
                continue
            if cand_idx >= visited_mask_row.numel() or not bool(visited_mask_row[cand_idx].item()):
                continue
            score_val = float(score_row[cand_idx].detach().item())
            health_val = float(combined_health_row[cand_idx].detach().item())
            curve_val = float(curve_health_row[cand_idx].detach().item())
            prior = history.get(vp_name)
            visits = 1 if prior is None else int(prior.get("visits", 0)) + 1
            if prior is None:
                avg_score = score_val
                avg_health = health_val
                avg_curve = curve_val
            else:
                avg_score = 0.5 * float(prior.get("score", score_val)) + 0.5 * score_val
                avg_health = 0.5 * float(prior.get("health", health_val)) + 0.5 * health_val
                avg_curve = 0.5 * float(prior.get("curve_health", curve_val)) + 0.5 * curve_val
            history[vp_name] = {
                "score": avg_score,
                "health": avg_health,
                "curve_health": avg_curve,
                "value": avg_score + avg_health,
                "step": float(step_index),
                "visits": visits,
            }
        self._prune_backtrack_history(history)

    @staticmethod
    def _select_label_top_candidate_id(
        teacher_action: int,
        raw_etp_logits: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> int:
        num_candidates = int(raw_etp_logits.size(0))
        if 0 <= int(teacher_action) < num_candidates:
            if int(teacher_action) == 0 or not bool(candidate_mask[int(teacher_action)].item()):
                return int(teacher_action)

        masked_logits = raw_etp_logits.masked_fill(candidate_mask.bool(), -float("inf"))
        if torch.isfinite(masked_logits).any():
            return int(masked_logits.argmax().item())

        if torch.isfinite(raw_etp_logits).any():
            return int(raw_etp_logits.argmax().item())

        return 0

    @staticmethod
    def _make_teacher_safe_logits(
        rescored_logits: torch.Tensor,
        fallback_logits: torch.Tensor,
        teacher_actions: torch.Tensor,
        invalid_candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Keep training CE finite when macro masking hides the teacher target.

        V5's frontier / level-3 controller can legally mask visited nodes or
        collapse a row down to a single macro action, but imitation supervision
        is still defined on the teacher candidate index. For loss computation we
        keep the behavior logits untouched for acting, while restoring a finite
        teacher position (and at least one finite score per row) on a copied
        tensor used only by CE.
        """
        safe_logits = rescored_logits.clone()
        fallback_logits = fallback_logits.to(dtype=safe_logits.dtype)
        safe_fallback = fallback_logits.masked_fill(invalid_candidate_mask.bool(), -float("inf"))
        row_has_finite = torch.isfinite(safe_logits).any(dim=-1)
        if (~row_has_finite).any():
            safe_logits[~row_has_finite] = safe_fallback[~row_has_finite]

        row_has_finite = torch.isfinite(safe_logits).any(dim=-1)
        if (~row_has_finite).any():
            safe_logits[~row_has_finite] = fallback_logits[~row_has_finite]

        batch_size, num_candidates = safe_logits.shape
        safe_teacher = teacher_actions.long().clamp(min=0, max=max(num_candidates - 1, 0))
        valid_teacher = (teacher_actions >= 0) & (teacher_actions < num_candidates)
        teacher_values = safe_logits[
            torch.arange(batch_size, device=safe_logits.device),
            safe_teacher,
        ]
        need_fix = valid_teacher & ~torch.isfinite(teacher_values)
        if need_fix.any():
            batch_idx = torch.arange(batch_size, device=safe_logits.device)[need_fix]
            teacher_idx = safe_teacher[need_fix]
            safe_logits[batch_idx, teacher_idx] = fallback_logits[batch_idx, teacher_idx]

        teacher_values = safe_logits[
            torch.arange(batch_size, device=safe_logits.device),
            safe_teacher,
        ]
        still_bad = valid_teacher & ~torch.isfinite(teacher_values)
        if still_bad.any():
            batch_idx = torch.arange(batch_size, device=safe_logits.device)[still_bad]
            teacher_idx = safe_teacher[still_bad]
            safe_logits[batch_idx, teacher_idx] = 0.0

        return safe_logits

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

        total_iter = self.config.IL.iters
        log_every = self.config.IL.log_every
        writer = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        if self._resume_scaler_state is not None:
            try:
                self.scaler.load_state_dict(self._resume_scaler_state)
            except Exception as exc:
                logger.warning("Unable to restore scaler state cleanly: %s", exc)

        self._prepare_step_metric_writer()

        logger.info("StateNav V6 %s training starts...", self.stage_name)
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter - idx, 0))
            cur_iter = idx + interval
            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                summary = {k: float(np.mean(v)) for k, v in logs.items()}
                save_best_by_train_loss = bool(self._logging_cfg().save_best_by_train_loss)
                total_metric = summary.get("total_loss", float("inf"))
                is_best = False
                if save_best_by_train_loss:
                    is_best = total_metric < self.best_metric
                    if is_best:
                        self.best_metric = total_metric

                message = f"[summary {cur_iter:06d}] " + ", ".join(
                    f"{k}: {v:.4f}" for k, v in summary.items()
                )
                logger.info(message)
                for k, v in summary.items():
                    writer.add_scalar(f"loss/{k}", v, cur_iter)
                self.save_checkpoint(cur_iter, is_best=is_best)

    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.eval()
        self.statenav_agent.train()
        self.waypoint_predictor.eval()

        use_tqdm = bool(self._logging_cfg().use_tqdm) and self._is_main_process()
        pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True, desc=f"StateNav {self.stage_name}") if use_tqdm else range(interval)
        self.logs = defaultdict(list)
        for idx in pbar:
            train_step_start = self._timing_now()
            self.optimizer.zero_grad(set_to_none=True)
            self.loss = torch.zeros((), device=self.device)

            rollout_start = self._timing_now()
            with autocast():
                self.rollout("train", ml_weight, sample_ratio)
            rollout_sec = self._timing_now() - rollout_start

            backward_start = self._timing_now()
            self.scaler.scale(self.loss).backward()
            self.scaler.unscale_(self.optimizer)
            trainable_params = []
            for group in self.optimizer.param_groups:
                trainable_params.extend([param for param in group["params"] if param.grad is not None])
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=float(self.config.STATENAV.grad_clip_norm),
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
                    pbar.set_postfix({
                        "interval": f"{idx + 1}/{interval}",
                        "step": f"{self._train_iteration}/{self.config.IL.iters}",
                        "loss": f"{step_record.get('total_loss', 0.0):.4f}",
                    })
        return deepcopy(self.logs)

    def _masked_mean(self, feats: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        weights = masks.float().unsqueeze(-1)
        summed = (feats * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def _init_label_contexts(self) -> List[Any]:
        contexts = []
        for episode in self.envs.current_episodes():
            ref_path = self.gt_data[str(episode.episode_id)]["locations"]
            contexts.append(self.label_builder.init_episode(ref_path))
        return contexts

    def _finalize_curve_sequences(
        self,
        active_sequences: Sequence[Dict[str, List[Any]]],
        finalized_sequences: List[Dict[str, Tensor]],
    ) -> None:
        for seq in active_sequences:
            if not seq["pred_curve"]:
                continue
            finalized_sequences.append(
                {
                    "pred_curve": torch.stack(seq["pred_curve"], dim=0),
                    "progress_ratio": torch.tensor(
                        seq["progress_ratio"],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    "valid": torch.tensor(
                        seq["valid"],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                }
            )

    def _finalize_imagination_sequences(
        self,
        active_sequences: Sequence[Dict[str, Any]],
        finalized_sequences: List[Dict[str, Tensor]],
    ) -> None:
        for seq in active_sequences:
            if not seq["h_post"]:
                continue
            finalized_sequences.append(
                {
                    "h_post": torch.stack(seq["h_post"], dim=0),
                    "z_post": torch.stack(seq["z_post"], dim=0),
                    "selected_action_feat": torch.stack(seq["selected_action_feat"], dim=0),
                    "h_prior": torch.stack(seq["h_prior"], dim=0),
                    "lang_tokens": seq["lang_tokens"],
                    "lang_mask": seq["lang_mask"],
                }
            )

    @staticmethod
    def _smooth_progress_curve(
        progress_ratio: torch.Tensor,
        window: int,
    ) -> torch.Tensor:
        if progress_ratio.numel() == 0 or window <= 1:
            return progress_ratio
        pad_left = max((window - 1) // 2, 0)
        pad_right = max(window // 2, 0)
        values = progress_ratio.view(1, 1, -1)
        padded = F.pad(values, (pad_left, pad_right), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=window, stride=1).view(-1)

    def _build_progress_curve_targets(
        self,
        progress_ratio: torch.Tensor,
    ) -> torch.Tensor:
        horizon = int(self.config.STATENAV.progress_curve_horizon)
        smooth_window = int(self.config.STATENAV.progress_curve_smooth_window)
        smooth_ratio = self._smooth_progress_curve(progress_ratio, smooth_window)
        seq_len = int(smooth_ratio.size(0))
        if seq_len == 0:
            return smooth_ratio.new_zeros((0, horizon))
        targets = []
        for step_idx in range(seq_len):
            current_value = smooth_ratio[step_idx]
            curve_values = [current_value]
            for offset in range(1, horizon):
                future_idx = min(step_idx + offset, seq_len - 1)
                curve_values.append(smooth_ratio[future_idx] - current_value)
            targets.append(torch.stack(curve_values, dim=0))
        return torch.stack(targets, dim=0)

    def _compute_progress_curve_loss(
        self,
        finalized_sequences: Sequence[Dict[str, Tensor]],
    ) -> Tuple[torch.Tensor, int]:
        curve_loss_sum = torch.zeros((), device=self.device)
        total_valid_steps = 0
        for seq in finalized_sequences:
            valid = seq["valid"].bool()
            valid_count = int(valid.sum().item())
            if valid_count == 0:
                continue
            gt_curve = self._build_progress_curve_targets(seq["progress_ratio"])
            seq_loss = progress_curve_loss(
                pred_curve=seq["pred_curve"],
                gt_curve=gt_curve,
                valid_mask=valid,
                reduction="sum",
            )
            curve_loss_sum = curve_loss_sum + seq_loss
            total_valid_steps += valid_count
        if total_valid_steps == 0:
            return curve_loss_sum, 0
        return curve_loss_sum / float(total_valid_steps), total_valid_steps

    @staticmethod
    def _mean_float_list(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        feedback = "sample" if mode == "train" else "argmax"
        trace_eval = False

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

        if mode == "train":
            self.statenav_agent.train()
        else:
            self.statenav_agent.eval()

        timing_sums = {
            "lang_sec": 0.0,
            "waypoint_sec": 0.0,
            "panorama_sec": 0.0,
            "graph_identify_sec": 0.0,
            "cand_real_pos_sec": 0.0,
            "graph_update_sec": 0.0,
            "navigation_sec": 0.0,
            "statenav_sec": 0.0,
            "label_rpc_sec": 0.0,
            "label_build_sec": 0.0,
            "env_step_sec": 0.0,
            "batch_prep_sec": 0.0,
        }

        lang_start = self._timing_now()
        with torch.no_grad():
            all_txt_ids = batch["instruction"]
            all_txt_masks = all_txt_ids != instr_pad_id
            all_txt_embeds = self.policy.net(
                mode="language",
                txt_ids=all_txt_ids,
                txt_masks=all_txt_masks,
            )
        timing_sums["lang_sec"] += self._timing_now() - lang_start

        total_actions = 0
        plan_loss_sum = torch.zeros((), device=self.device)
        kl_loss_sum = torch.zeros((), device=self.device)
        d_t_sum = torch.zeros((), device=self.device)
        health_scalar_sum = torch.zeros((), device=self.device)
        curve_health_sum = torch.zeros((), device=self.device)
        cand_health_sum = torch.zeros((), device=self.device)
        health_bonus_sum = torch.zeros((), device=self.device)
        dt_temperature_sum = torch.zeros((), device=self.device)
        d_t_norm_sum = torch.zeros((), device=self.device)
        d_t_latent_sum = torch.zeros((), device=self.device)
        d_t_node_sum = torch.zeros((), device=self.device)
        attn_health_sum = torch.zeros((), device=self.device)
        s_t_norm_sum = torch.zeros((), device=self.device)
        topo_novelty_sum = torch.zeros((), device=self.device)
        topo_update_sum = torch.zeros((), device=self.device)
        intervention_level_sum = torch.zeros((), device=self.device)
        dt_threshold_low_sum = torch.zeros((), device=self.device)
        dt_threshold_high_sum = torch.zeros((), device=self.device)
        score_before_sum = torch.zeros((), device=self.device)
        score_after_sum = torch.zeros((), device=self.device)
        strict_candidate_count_sum = torch.zeros((), device=self.device)
        expanded_candidate_count_sum = torch.zeros((), device=self.device)
        node_surprise_loss_sum = torch.zeros((), device=self.device)
        total_node_updates = 0
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
        topo_bank = TopoStateBank.create(
            num_envs=self.envs.num_envs,
            topo_state_dim=int(state_model.topo_state_dim),
            feat_dim=int(state_model.x_dim),
            device=self.device,
        )
        trace_label_contexts = self._init_label_contexts() if trace_eval else None
        active_backtrack_histories = [dict() for _ in range(self.envs.num_envs)]
        active_eval_self_stats = (
            [
                {
                    "d_t_fused": [],
                    "d_t_latent": [],
                    "d_t_node": [],
                    "topo_novelty": [],
                    "topo_update": [],
                    "dt_temperature": [],
                    "d_t_norm": [],
                    "intervention_level": [],
                    "dt_threshold_low": [],
                    "dt_threshold_high": [],
                    "score_before": [],
                    "score_after": [],
                    "strict_candidate_count": [],
                    "expanded_candidate_count": [],
                }
                for _ in range(self.envs.num_envs)
            ]
            if mode == "eval"
            else None
        )
        active_eval_step_traces = [[] for _ in range(self.envs.num_envs)] if trace_eval else None
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

            g_t = self._masked_mean(nav_outs["gmap_embeds"], nav_inputs["gmap_masks"])
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
            if bool(state_model.use_topo_bank):
                topo_novelty = topo_bank.compute_novelty(avg_pano_embeds)
                if bool(state_model.use_topo_gate):
                    topo_update_mask = topo_bank.build_update_mask(
                        current_vp_ids=cur_vp,
                        candidate_counts=candidate_counts,
                        novelty_scores=topo_novelty,
                        novelty_tau=self._topo_novelty_tau(),
                        max_micro_steps=self._topo_update_max_k(),
                        decision_candidate_min=self._topo_decision_candidate_min(),
                    )
                else:
                    topo_update_mask = torch.ones(
                        self.envs.num_envs,
                        device=self.device,
                        dtype=torch.bool,
                    )
                valid_node_update_mask = topo_bank.has_state.clone() & topo_update_mask
            else:
                topo_novelty = torch.zeros(
                    self.envs.num_envs,
                    device=self.device,
                    dtype=avg_pano_embeds.dtype,
                )
                topo_update_mask = torch.zeros(
                    self.envs.num_envs,
                    device=self.device,
                    dtype=torch.bool,
                )
                valid_node_update_mask = topo_update_mask
            adapter_warmup = self._compute_adapter_warmup() if mode == "train" else 1.0
            statenav_start = self._timing_now()
            step_outs = self.statenav_agent(
                x_t=avg_pano_embeds,
                g_t=g_t,
                lang_tokens=txt_embeds,
                cand_feats=nav_outs["gmap_embeds"],
                etp_candidate_scores=nav_logits,
                prev_action_emb=recurrent_state["prev_action_emb"],
                prev_z=recurrent_state["prev_z"],
                prev_h=recurrent_state["prev_h"],
                lang_mask=txt_masks,
                candidate_mask=candidate_mask,
                visited_mask=visited_candidate_mask,
                frontier_mask=frontier_candidate_mask,
                local_frontier_mask=local_frontier_candidate_mask,
                x_tokens=pano_embeds,
                x_token_mask=pano_masks.bool(),
                g_tokens=nav_outs["gmap_embeds"],
                g_token_mask=nav_inputs["gmap_masks"].bool(),
                history_backtrack_values=history_backtrack_values,
                history_valid_mask=history_valid_mask,
                topo_prev_state=topo_bank.topo_states,
                topo_valid_mask=topo_bank.has_state,
                topo_update_mask=topo_update_mask,
                topo_novelty=topo_novelty,
                current_step=stepk + 1,
                adapter_warmup=adapter_warmup,
                temp=float(self.config.STATENAV.gumbel_temp),
                hard=bool(self.config.STATENAV.hard_gumbel),
            )
            timing_sums["statenav_sec"] += self._timing_now() - statenav_start
            if bool(state_model.use_topo_bank):
                topo_state_t = torch.cat([step_outs["h_t"], step_outs["z_flat"]], dim=-1)
                topo_bank.commit(
                    update_mask=topo_update_mask,
                    current_vp_ids=cur_vp,
                    topo_state=topo_state_t,
                    topo_feat=avg_pano_embeds,
                )
            rescored_logits = step_outs["rescored_candidate_scores"]
            d_t_step = step_outs["D_t"]
            d_t_latent_step = step_outs["D_t_latent"]
            d_t_node_step = step_outs["D_t_node"]
            topo_novelty_step = step_outs["topo_novelty"]
            topo_update_step = topo_update_mask.to(torch.float32)
            health_scalar_step = step_outs["health_scalar"]
            curve_health_step = step_outs["curve_health_mean"]
            cand_health_step = step_outs["cand_health_mean"]
            health_bonus_step = step_outs["health_bonus_mean"]
            dt_temperature_step = step_outs["dt_temperature"]
            d_t_norm_step = step_outs["d_t_norm"]
            attn_health_step = step_outs["attn_health_mean"]
            s_t_norm_step = step_outs["s_t"].norm(dim=-1)
            strict_candidate_count_step = (
                step_outs["strict_candidate_mask"].logical_not().sum(dim=-1).to(torch.float32)
            )
            expanded_candidate_count_step = (
                step_outs["expanded_candidate_mask"].logical_not().sum(dim=-1).to(torch.float32)
            )
            intervention_level_step = step_outs["intervention_level"].to(torch.float32)
            dt_threshold_low_step = step_outs["dt_threshold_low"]
            dt_threshold_high_step = step_outs["dt_threshold_high"]
            d_t_sum = d_t_sum + d_t_step.sum()
            d_t_latent_sum = d_t_latent_sum + d_t_latent_step.sum()
            d_t_node_sum = d_t_node_sum + d_t_node_step.sum()
            health_scalar_sum = health_scalar_sum + health_scalar_step.sum()
            curve_health_sum = curve_health_sum + curve_health_step.sum()
            cand_health_sum = cand_health_sum + cand_health_step.sum()
            health_bonus_sum = health_bonus_sum + health_bonus_step.sum()
            dt_temperature_sum = dt_temperature_sum + dt_temperature_step.sum()
            d_t_norm_sum = d_t_norm_sum + d_t_norm_step.sum()
            attn_health_sum = attn_health_sum + attn_health_step.sum()
            s_t_norm_sum = s_t_norm_sum + s_t_norm_step.sum()
            topo_novelty_sum = topo_novelty_sum + topo_novelty.sum()
            topo_update_sum = topo_update_sum + topo_update_mask.to(torch.float32).sum()
            intervention_level_sum = intervention_level_sum + intervention_level_step.sum()
            dt_threshold_low_sum = dt_threshold_low_sum + dt_threshold_low_step.sum()
            dt_threshold_high_sum = dt_threshold_high_sum + dt_threshold_high_step.sum()
            score_before_sum = score_before_sum + step_outs["score_before_mean"].sum()
            score_after_sum = score_after_sum + step_outs["score_after_mean"].sum()
            strict_candidate_count_sum = strict_candidate_count_sum + strict_candidate_count_step.sum()
            expanded_candidate_count_sum = (
                expanded_candidate_count_sum + expanded_candidate_count_step.sum()
            )
            behavior_logits = self._behavior_logits(mode, nav_logits, rescored_logits)
            nav_probs = F.softmax(behavior_logits, dim=1)
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].detach().item()

            if mode == "eval":
                for i in range(self.envs.num_envs):
                    active_eval_self_stats[i]["d_t_fused"].append(float(d_t_step[i].detach().item()))
                    active_eval_self_stats[i]["d_t_latent"].append(
                        float(d_t_latent_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["d_t_node"].append(
                        float(d_t_node_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["topo_novelty"].append(
                        float(topo_novelty_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["topo_update"].append(
                        float(topo_update_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["dt_temperature"].append(
                        float(dt_temperature_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["d_t_norm"].append(
                        float(d_t_norm_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["intervention_level"].append(
                        float(intervention_level_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["dt_threshold_low"].append(
                        float(dt_threshold_low_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["dt_threshold_high"].append(
                        float(dt_threshold_high_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["score_before"].append(
                        float(step_outs["score_before_mean"][i].detach().item())
                    )
                    active_eval_self_stats[i]["score_after"].append(
                        float(step_outs["score_after_mean"][i].detach().item())
                    )
                    active_eval_self_stats[i]["strict_candidate_count"].append(
                        float(strict_candidate_count_step[i].detach().item())
                    )
                    active_eval_self_stats[i]["expanded_candidate_count"].append(
                        float(expanded_candidate_count_step[i].detach().item())
                    )

            if mode == "train" or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs["gmap_vp_ids"], no_vp_left)
            else:
                teacher_actions = None

            if mode == "train":
                loss_safe_rescored_logits = self._make_teacher_safe_logits(
                    rescored_logits=rescored_logits,
                    fallback_logits=nav_logits,
                    teacher_actions=teacher_actions,
                    invalid_candidate_mask=invalid_candidate_mask,
                )

                plan_loss_sum = plan_loss_sum + planner_loss(
                    rescored_scores=loss_safe_rescored_logits,
                    teacher_cand_label=teacher_actions,
                    reduction="sum",
                )
                kl_loss_step, _ = latent_kl_loss(
                    post_logits=step_outs["post_logits"],
                    prior_logits=step_outs["prior_logits"],
                    free_bits=float(self.config.STATENAV.free_bits),
                    reduction="sum",
                )
                kl_loss_sum = kl_loss_sum + kl_loss_step
                node_surprise_loss_sum = node_surprise_loss_sum + step_outs["D_t_node"][valid_node_update_mask].sum()
                total_node_updates += int(valid_node_update_mask.sum().item())

            if feedback == "sample":
                c = torch.distributions.Categorical(probs=nav_probs)
                a_t = c.sample().detach()
                if mode == "train":
                    use_teacher = (
                        (teacher_actions >= 0)
                        & (torch.rand_like(a_t, dtype=torch.float) <= sample_ratio)
                    )
                    a_t = torch.where(
                        use_teacher,
                        teacher_actions,
                        a_t,
                    )
            else:
                a_t = behavior_logits.argmax(dim=-1)
            cpu_a_t = a_t.cpu().numpy()

            safe_a_t = a_t.clamp(min=0, max=nav_outs["gmap_embeds"].size(1) - 1)
            selected_feats = nav_outs["gmap_embeds"][
                torch.arange(self.envs.num_envs, device=self.device), safe_a_t
            ]
            selected_action_emb = state_model.project_action_features(selected_feats)
            recurrent_state["prev_action_emb"] = selected_action_emb
            recurrent_state["prev_h"] = step_outs["h_t"]
            recurrent_state["prev_z"] = step_outs["z_flat"]

            if trace_eval:
                trace_ref_distances_batch = self.envs.call(
                    ["current_dist_to_refpath"] * self.envs.num_envs,
                    [{"path": getattr(context, "reference_path_list", getattr(context, "reference_path", getattr(context, "ref_path", None)))} for context in trace_label_contexts],
                )
                for i in range(self.envs.num_envs):
                    ep_id = self.envs.current_episodes()[i].episode_id
                    selected_idx = int(cpu_a_t[i])
                    safe_selected_idx = self._safe_candidate_index(
                        selected_idx,
                        int(nav_outs["gmap_embeds"].size(1)),
                    )
                    strict_mask_row = step_outs["strict_candidate_mask"][i]
                    expanded_mask_row = step_outs["expanded_candidate_mask"][i]
                    top1_before_idx = self._masked_argmax_index(nav_logits[i], strict_mask_row)
                    top1_after_idx = self._masked_argmax_index(rescored_logits[i], expanded_mask_row)
                    ref_distances = trace_ref_distances_batch[i]
                    trace_label = self.label_builder.build_step(
                        context=trace_label_contexts[i],
                        current_position=cur_pos[i],
                        current_heading=heading_from_quaternion(np.array(cur_ori[i])),
                        node_id=cur_vp[i],
                        top_candidate_id=safe_selected_idx,
                        ref_distances=ref_distances,
                    )

                    before_score = float(nav_logits[i, safe_selected_idx].detach().item())
                    after_score = float(rescored_logits[i, safe_selected_idx].detach().item())
                    chosen_health = float(
                        step_outs["cand_health_scores"][i, safe_selected_idx].detach().item()
                    )
                    level3_selected_idx = int(step_outs["level3_selected_idx"][i].detach().item())
                    level3_decision_code = int(step_outs["level3_decision_code"][i].detach().item())
                    level3_selected_vp = None
                    if 0 <= level3_selected_idx < len(nav_inputs["gmap_vp_ids"][i]):
                        level3_selected_vp = nav_inputs["gmap_vp_ids"][i][level3_selected_idx]
                    chosen_curve = (
                        step_outs["cand_progress_curves"][i, safe_selected_idx]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    progress_curve = step_outs["progress_curve"][i].detach().cpu().tolist()
                    trace_entry = {
                        "episode_id": str(ep_id),
                        "step_index": int(stepk),
                        "selected_idx": int(selected_idx),
                        "selected_idx_safe": int(safe_selected_idx),
                        "selected_is_stop": int(selected_idx == 0),
                        "top1_before_idx": int(top1_before_idx),
                        "top1_after_idx": int(top1_after_idx),
                        "top1_changed": int(top1_before_idx != top1_after_idx),
                        "progress_ratio": float(trace_label["progress_ratio"]),
                        "state_valid_mask": int(trace_label["state_valid_mask"]),
                        "prefix_idx": int(trace_label["prefix_idx"]),
                        "D_t": float(d_t_step[i].detach().item()),
                        "d_t_norm": float(d_t_norm_step[i].detach().item()),
                        "dt_threshold_low": float(dt_threshold_low_step[i].detach().item()),
                        "dt_threshold_high": float(dt_threshold_high_step[i].detach().item()),
                        "intervention_level": float(intervention_level_step[i].detach().item()),
                        "health_scalar": float(health_scalar_step[i].detach().item()),
                        "s_t_norm": float(s_t_norm_step[i].detach().item()),
                        "curve_health_mean": float(curve_health_step[i].detach().item()),
                        "cand_health_mean": float(cand_health_step[i].detach().item()),
                        "attn_health_mean": float(attn_health_step[i].detach().item()),
                        "health_bonus_mean": float(health_bonus_step[i].detach().item()),
                        "dt_temperature": float(dt_temperature_step[i].detach().item()),
                        "score_before_mean": float(step_outs["score_before_mean"][i].detach().item()),
                        "score_after_mean": float(step_outs["score_after_mean"][i].detach().item()),
                        "selected_score_before": before_score,
                        "selected_score_after": after_score,
                        "selected_cand_health": chosen_health,
                        "selected_curve_health": float(
                            step_outs["cand_curve_health_scores"][i, safe_selected_idx].detach().item()
                        ),
                        "selected_attn_health": float(
                            step_outs["attn_health_scores"][i, safe_selected_idx].detach().item()
                        ),
                        "selected_combined_health": chosen_health,
                        "selected_current_attn_pos": float(
                            step_outs["current_attn_position"][i].detach().item()
                        ),
                        "selected_predicted_attn_pos": float(
                            step_outs["predicted_attn_position"][i, safe_selected_idx].detach().item()
                        ),
                        "level3_policy": str(self.config.STATENAV.level3_policy),
                        "level3_selected_idx": level3_selected_idx,
                        "level3_selected_vp": None if level3_selected_vp is None else str(level3_selected_vp),
                        "level3_decision": self._level3_decision_name(level3_decision_code),
                        "progress_curve": progress_curve,
                        "selected_cand_curve": chosen_curve,
                        "strict_valid_candidates": int(
                            strict_mask_row.logical_not().sum().item()
                        ),
                        "expanded_valid_candidates": int(
                            expanded_mask_row.logical_not().sum().item()
                        ),
                        "local_frontier_candidates": int(
                            step_outs["local_frontier_candidate_mask"][i].sum().item()
                        ),
                        "frontier_candidates": int(
                            step_outs["frontier_candidate_mask"][i].sum().item()
                        ),
                        "topk_before": self._candidate_topk_trace(
                            nav_logits[i],
                            strict_mask_row,
                            None,
                        ),
                        "topk_after": self._candidate_topk_trace(
                            rescored_logits[i],
                            expanded_mask_row,
                            step_outs["cand_health_scores"][i],
                        ),
                    }
                    active_eval_step_traces[i].append(trace_entry)

            for i in range(self.envs.num_envs):
                self._update_backtrack_history(
                    history=active_backtrack_histories[i],
                    vp_ids=nav_inputs["gmap_vp_ids"][i],
                    invalid_mask_row=invalid_candidate_mask[i],
                    visited_mask_row=visited_candidate_mask[i],
                    score_row=nav_logits[i],
                    combined_health_row=step_outs["cand_health_scores"][i],
                    curve_health_row=step_outs["cand_curve_health_scores"][i],
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
                    stop_scores = [score for _, score in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    back_path = (
                        [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]][1:]
                        if self.config.IL.back_algo == "control"
                        else None
                    )
                    vis_info = {
                        "nodes": list(gmap.node_pos.values()),
                        "ghosts": list(gmap.ghost_aug_pos.values()),
                        "predict_ghost": stop_pos,
                        "debug_text_lines": self._build_v4_video_debug_lines(
                            step_index=int(stepk),
                            selected_idx=int(cpu_a_t[i]),
                            top1_before_idx=self._masked_argmax_index(
                                nav_logits[i],
                                step_outs["strict_candidate_mask"][i],
                            ),
                            top1_after_idx=self._masked_argmax_index(
                                rescored_logits[i],
                                step_outs["expanded_candidate_mask"][i],
                            ),
                            teacher_idx=int(teacher_actions[i].item()) if self.config.VIDEO_OPTION else -100,
                            d_t=float(d_t_step[i].detach().item()),
                            d_t_norm=float(d_t_norm_step[i].detach().item()),
                            dt_threshold_low=float(dt_threshold_low_step[i].detach().item()),
                            dt_threshold_high=float(dt_threshold_high_step[i].detach().item()),
                            health_scalar=float(health_scalar_step[i].detach().item()),
                            selected_curve_health=float(
                                step_outs["cand_curve_health_scores"][
                                    i,
                                    self._safe_candidate_index(
                                        int(cpu_a_t[i]),
                                        int(nav_outs["gmap_embeds"].size(1)),
                                    ),
                                ].detach().item()
                            ),
                            selected_attn_health=float(
                                step_outs["attn_health_scores"][
                                    i,
                                    self._safe_candidate_index(
                                        int(cpu_a_t[i]),
                                        int(nav_outs["gmap_embeds"].size(1)),
                                    ),
                                ].detach().item()
                            ),
                            selected_combined_health=float(
                                step_outs["cand_health_scores"][
                                    i,
                                    self._safe_candidate_index(
                                        int(cpu_a_t[i]),
                                        int(nav_outs["gmap_embeds"].size(1)),
                                    ),
                                ].detach().item()
                            ),
                            health_bonus_mean=float(health_bonus_step[i].detach().item()),
                            dt_temperature=float(dt_temperature_step[i].detach().item()),
                            intervention_level=float(intervention_level_step[i].detach().item()),
                            score_before=float(
                                nav_logits[
                                    i,
                                    self._safe_candidate_index(
                                        int(cpu_a_t[i]),
                                        int(nav_outs["gmap_embeds"].size(1)),
                                    ),
                                ]
                                .detach()
                                .item()
                            ),
                            score_after=float(
                                rescored_logits[
                                    i,
                                    self._safe_candidate_index(
                                        int(cpu_a_t[i]),
                                        int(nav_outs["gmap_embeds"].size(1)),
                                    ),
                                ]
                                .detach()
                                .item()
                            ),
                            progress_curve=step_outs["progress_curve"][i].detach().cpu().tolist(),
                            candidate_mask_row=step_outs["expanded_candidate_mask"][i],
                            level3_policy=str(self.config.STATENAV.level3_policy),
                            level3_decision=self._level3_decision_name(
                                int(step_outs["level3_decision_code"][i].detach().item())
                            ),
                            level3_target=(
                                None
                                if int(step_outs["level3_selected_idx"][i].detach().item()) < 0
                                else str(
                                    nav_inputs["gmap_vp_ids"][i][
                                        int(step_outs["level3_selected_idx"][i].detach().item())
                                    ]
                                )
                            ),
                        ),
                    }
                    env_actions.append(
                        {
                            "action": {
                                "act": 0,
                                "cur_vp": cur_vp[i],
                                "stop_vp": stop_vp,
                                "stop_pos": stop_pos,
                                "back_path": back_path,
                                "tryout": use_tryout,
                            },
                            "vis_info": vis_info,
                        }
                    )
                elif selected_is_visited_node:
                    target_vp = str(selected_vp)
                    target_pos = gmap.node_pos[target_vp]
                    if self.config.VIDEO_OPTION:
                        top1_before_idx = self._masked_argmax_index(
                            nav_logits[i],
                            step_outs["strict_candidate_mask"][i],
                        )
                        top1_after_idx = self._masked_argmax_index(
                            rescored_logits[i],
                            step_outs["expanded_candidate_mask"][i],
                        )
                        safe_selected_idx = self._safe_candidate_index(
                            selected_idx,
                            int(nav_outs["gmap_embeds"].size(1)),
                        )
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        vis_info = {
                            "nodes": list(gmap.node_pos.values()),
                            "ghosts": list(gmap.ghost_aug_pos.values()),
                            "predict_ghost": target_pos,
                            "debug_text_lines": self._build_v4_video_debug_lines(
                                step_index=int(stepk),
                                selected_idx=selected_idx,
                                top1_before_idx=top1_before_idx,
                                top1_after_idx=top1_after_idx,
                                teacher_idx=int(teacher_action_cpu),
                                d_t=float(d_t_step[i].detach().item()),
                                d_t_norm=float(d_t_norm_step[i].detach().item()),
                                dt_threshold_low=float(dt_threshold_low_step[i].detach().item()),
                                dt_threshold_high=float(dt_threshold_high_step[i].detach().item()),
                                health_scalar=float(health_scalar_step[i].detach().item()),
                                selected_curve_health=float(
                                    step_outs["cand_curve_health_scores"][i, safe_selected_idx].detach().item()
                                ),
                                selected_attn_health=float(
                                    step_outs["attn_health_scores"][i, safe_selected_idx].detach().item()
                                ),
                                selected_combined_health=float(
                                    step_outs["cand_health_scores"][i, safe_selected_idx].detach().item()
                                ),
                                health_bonus_mean=float(health_bonus_step[i].detach().item()),
                                dt_temperature=float(dt_temperature_step[i].detach().item()),
                                intervention_level=float(intervention_level_step[i].detach().item()),
                                score_before=float(nav_logits[i, safe_selected_idx].detach().item()),
                                score_after=float(rescored_logits[i, safe_selected_idx].detach().item()),
                                progress_curve=step_outs["progress_curve"][i].detach().cpu().tolist(),
                                candidate_mask_row=step_outs["expanded_candidate_mask"][i],
                                level3_policy=str(self.config.STATENAV.level3_policy),
                                level3_decision=self._level3_decision_name(
                                    int(step_outs["level3_decision_code"][i].detach().item())
                                ),
                                level3_target=(
                                    None
                                    if int(step_outs["level3_selected_idx"][i].detach().item()) < 0
                                    else str(
                                        nav_inputs["gmap_vp_ids"][i][
                                            int(step_outs["level3_selected_idx"][i].detach().item())
                                        ]
                                    )
                                ),
                            ),
                        }
                    else:
                        vis_info = None
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
                            },
                            "vis_info": vis_info,
                        }
                    )
                    prev_vp[i] = target_vp
                else:
                    ghost_vp = selected_vp
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        top1_before_idx = self._masked_argmax_index(
                            nav_logits[i],
                            step_outs["strict_candidate_mask"][i],
                        )
                        top1_after_idx = self._masked_argmax_index(
                            rescored_logits[i],
                            step_outs["expanded_candidate_mask"][i],
                        )
                        selected_idx = int(cpu_a_t[i])
                        safe_selected_idx = self._safe_candidate_index(
                            selected_idx,
                            int(nav_outs["gmap_embeds"].size(1)),
                        )
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        teacher_ghost = (
                            None
                            if teacher_action_cpu in [0, -100]
                            else gmap.ghost_aug_pos[nav_inputs["gmap_vp_ids"][i][teacher_action_cpu]]
                        )
                        vis_info = {
                            "nodes": list(gmap.node_pos.values()),
                            "ghosts": list(gmap.ghost_aug_pos.values()),
                            "predict_ghost": ghost_pos,
                            "teacher_ghost": teacher_ghost,
                            "debug_text_lines": self._build_v4_video_debug_lines(
                                step_index=int(stepk),
                                selected_idx=selected_idx,
                                top1_before_idx=top1_before_idx,
                                top1_after_idx=top1_after_idx,
                                teacher_idx=int(teacher_action_cpu),
                                d_t=float(d_t_step[i].detach().item()),
                                d_t_norm=float(d_t_norm_step[i].detach().item()),
                                dt_threshold_low=float(dt_threshold_low_step[i].detach().item()),
                                dt_threshold_high=float(dt_threshold_high_step[i].detach().item()),
                                health_scalar=float(health_scalar_step[i].detach().item()),
                                selected_curve_health=float(
                                    step_outs["cand_curve_health_scores"][i, safe_selected_idx].detach().item()
                                ),
                                selected_attn_health=float(
                                    step_outs["attn_health_scores"][i, safe_selected_idx].detach().item()
                                ),
                                selected_combined_health=float(
                                    step_outs["cand_health_scores"][i, safe_selected_idx].detach().item()
                                ),
                                health_bonus_mean=float(health_bonus_step[i].detach().item()),
                                dt_temperature=float(dt_temperature_step[i].detach().item()),
                                intervention_level=float(intervention_level_step[i].detach().item()),
                                score_before=float(nav_logits[i, safe_selected_idx].detach().item()),
                                score_after=float(rescored_logits[i, safe_selected_idx].detach().item()),
                                progress_curve=step_outs["progress_curve"][i].detach().cpu().tolist(),
                                candidate_mask_row=step_outs["expanded_candidate_mask"][i],
                                level3_policy=str(self.config.STATENAV.level3_policy),
                                level3_decision=self._level3_decision_name(
                                    int(step_outs["level3_decision_code"][i].detach().item())
                                ),
                                level3_target=(
                                    None
                                    if int(step_outs["level3_selected_idx"][i].detach().item()) < 0
                                    else str(
                                        nav_inputs["gmap_vp_ids"][i][
                                            int(step_outs["level3_selected_idx"][i].detach().item())
                                        ]
                                    )
                                ),
                            ),
                        }
                    else:
                        vis_info = None
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
                            },
                            "vis_info": vis_info,
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
                    metric["ghost_cnt"] = self.gmaps[i].ghost_cnt
                    if self._log_prior_post_eval_enabled():
                        self_stats = active_eval_self_stats[i]
                        metric["mean_d_t_fused"] = self._mean_float_list(self_stats["d_t_fused"])
                        metric["mean_d_t_latent"] = self._mean_float_list(self_stats["d_t_latent"])
                        metric["mean_d_t_node"] = self._mean_float_list(self_stats["d_t_node"])
                        metric["mean_topo_novelty"] = self._mean_float_list(self_stats["topo_novelty"])
                        metric["mean_topo_update"] = self._mean_float_list(self_stats["topo_update"])
                        metric["mean_dt_temperature"] = self._mean_float_list(self_stats["dt_temperature"])
                        metric["mean_d_t_norm"] = self._mean_float_list(self_stats["d_t_norm"])
                        metric["mean_intervention_level"] = self._mean_float_list(
                            self_stats["intervention_level"]
                        )
                        metric["mean_dt_threshold_low"] = self._mean_float_list(
                            self_stats["dt_threshold_low"]
                        )
                        metric["mean_dt_threshold_high"] = self._mean_float_list(
                            self_stats["dt_threshold_high"]
                        )
                        metric["mean_score_before"] = self._mean_float_list(self_stats["score_before"])
                        metric["mean_score_after"] = self._mean_float_list(self_stats["score_after"])
                        metric["mean_strict_candidate_count"] = self._mean_float_list(
                            self_stats["strict_candidate_count"]
                        )
                        metric["mean_expanded_candidate_count"] = self._mean_float_list(
                            self_stats["expanded_candidate_count"]
                        )
                    if trace_eval and active_eval_step_traces is not None:
                        progress_ratio_seq = torch.tensor(
                            [step["progress_ratio"] for step in active_eval_step_traces[i]],
                            device=self.device,
                            dtype=torch.float32,
                        )
                        gt_curve = self._build_progress_curve_targets(progress_ratio_seq)
                        if gt_curve.numel() > 0:
                            for step_idx, step_trace in enumerate(active_eval_step_traces[i]):
                                gt_curve_step = gt_curve[step_idx].detach().cpu().tolist()
                                pred_curve_step = step_trace["progress_curve"]
                                curve_mse = float(
                                    np.mean(
                                        (np.asarray(pred_curve_step, dtype=np.float32)
                                         - np.asarray(gt_curve_step, dtype=np.float32))
                                        ** 2
                                    )
                                )
                                step_trace["gt_progress_curve"] = gt_curve_step
                                step_trace["curve_mse"] = curve_mse
                        self._write_v4_episode_trace(
                            episode_id=ep_id,
                            metric=metric,
                            step_trace=active_eval_step_traces[i],
                        )
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
                    if self._log_prior_post_eval_enabled():
                        running_means.update(
                            {
                                "mean_d_t_fused": sum(v["mean_d_t_fused"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_d_t_latent": sum(v["mean_d_t_latent"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_d_t_node": sum(v["mean_d_t_node"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_topo_novelty": sum(v["mean_topo_novelty"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_topo_update": sum(v["mean_topo_update"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_dt_temperature": sum(v["mean_dt_temperature"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_d_t_norm": sum(v["mean_d_t_norm"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_intervention_level": sum(
                                    v["mean_intervention_level"] for v in self.stat_eps.values()
                                ) / evaluated_eps,
                                "mean_dt_threshold_low": sum(
                                    v["mean_dt_threshold_low"] for v in self.stat_eps.values()
                                ) / evaluated_eps,
                                "mean_dt_threshold_high": sum(
                                    v["mean_dt_threshold_high"] for v in self.stat_eps.values()
                                ) / evaluated_eps,
                                "mean_score_before": sum(v["mean_score_before"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_score_after": sum(v["mean_score_after"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_strict_candidate_count": sum(
                                    v["mean_strict_candidate_count"] for v in self.stat_eps.values()
                                ) / evaluated_eps,
                                "mean_expanded_candidate_count": sum(
                                    v["mean_expanded_candidate_count"] for v in self.stat_eps.values()
                                ) / evaluated_eps,
                            }
                        )
                    if self.pbar is not None and self.local_rank < 1:
                        self.pbar.set_postfix({
                            "ep": f"{evaluated_eps}/{eval_total}",
                            "succ": f"{running_means['success']:.3f}",
                            "spl": f"{running_means['spl']:.3f}",
                            "ndtw": f"{running_means['ndtw']:.3f}",
                            "sdtw": f"{running_means['sdtw']:.3f}",
                        })
                    ep_done_message = (
                        f"[EP-DONE] ep={ep_id} | succ={metric['success']:.0f} "
                        f"oracle={metric['oracle_success']:.0f} | "
                        f"d2g={metric['distance_to_goal']:.2f} "
                        f"steps={metric['steps_taken']} "
                        f"path_len={metric['path_length']:.2f} "
                        f"gt_len={gt_length:.2f} | "
                        f"spl={metric['spl']:.3f} ndtw={metric['ndtw']:.3f} sdtw={metric['sdtw']:.3f} "
                        f"ghost={metric['ghost_cnt']} collisions={metric['collisions']:.3f}"
                    )
                    if self._log_prior_post_eval_enabled():
                        ep_done_message += (
                            f" | dt={metric['mean_d_t_fused']:.3f}"
                            f" dl={metric['mean_d_t_latent']:.3f}"
                            f" dn={metric['mean_d_t_node']:.3f}"
                            f" dtn={metric['mean_d_t_norm']:.3f}"
                            f" tn={metric['mean_topo_novelty']:.3f}"
                            f" tu={metric['mean_topo_update']:.3f}"
                            f" tt={metric['mean_dt_temperature']:.3f}"
                            f" lv={metric['mean_intervention_level']:.2f}"
                            f" sc/ec={metric['mean_strict_candidate_count']:.2f}/{metric['mean_expanded_candidate_count']:.2f}"
                            f" th={metric['mean_dt_threshold_low']:.3f}/{metric['mean_dt_threshold_high']:.3f}"
                        )
                    logger.info(ep_done_message)
                    log_every_episode = max(int(getattr(self.config.EVAL, "LOG_EVERY_EPISODE", 1)), 1)
                    if self.local_rank < 1 and (
                        evaluated_eps == 1
                        or evaluated_eps % log_every_episode == 0
                        or evaluated_eps == eval_total
                    ):
                        eval_live_message = (
                            f"[EVAL-LIVE] ep={evaluated_eps}/{eval_total} | "
                            f"success={running_means['success']:.3f} "
                            f"spl={running_means['spl']:.3f} "
                            f"ndtw={running_means['ndtw']:.3f} "
                            f"sdtw={running_means['sdtw']:.3f} "
                            f"d2g={running_means['distance_to_goal']:.3f}"
                        )
                        if self._log_prior_post_eval_enabled():
                            eval_live_message += (
                                f" dt={running_means['mean_d_t_fused']:.3f}"
                                f" dl={running_means['mean_d_t_latent']:.3f}"
                                f" dn={running_means['mean_d_t_node']:.3f}"
                                f" dtn={running_means['mean_d_t_norm']:.3f}"
                                f" tn={running_means['mean_topo_novelty']:.3f}"
                                f" tu={running_means['mean_topo_update']:.3f}"
                                f" tt={running_means['mean_dt_temperature']:.3f}"
                                f" lv={running_means['mean_intervention_level']:.2f}"
                                f" sc/ec={running_means['mean_strict_candidate_count']:.2f}/{running_means['mean_expanded_candidate_count']:.2f}"
                                f" th={running_means['mean_dt_threshold_low']:.3f}/{running_means['mean_dt_threshold_high']:.3f}"
                            )
                        logger.info(eval_live_message)
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
                                {"position": pos, "heading": heading, "stop": False}
                            )
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]["stop"] = True
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
                    if mode == "eval":
                        active_eval_self_stats.pop(i)
                    if trace_eval:
                        trace_label_contexts.pop(i)
                        active_eval_step_traces.pop(i)

                if keep_indices:
                    keep_tensor = torch.tensor(keep_indices, device=self.device, dtype=torch.long)
                    recurrent_state["prev_h"] = recurrent_state["prev_h"].index_select(0, keep_tensor)
                    recurrent_state["prev_z"] = recurrent_state["prev_z"].index_select(0, keep_tensor)
                    recurrent_state["prev_action_emb"] = recurrent_state["prev_action_emb"].index_select(
                        0, keep_tensor
                    )
                    topo_bank = topo_bank.index_select(keep_indices)

            if self.envs.num_envs == 0:
                break

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
            plan_loss = plan_loss_sum / max(total_actions, 1)
            micro_kl_loss = kl_loss_sum / max(total_actions, 1)
            node_surprise_loss = node_surprise_loss_sum / max(total_node_updates, 1)
            total_loss = (
                plan_loss
                + float(self.config.STATENAV.lambda_micro_kl) * micro_kl_loss
                + float(self.config.STATENAV.lambda_node_surprise) * node_surprise_loss
            )

            self.loss = self.loss + ml_weight * total_loss

            scalar_metrics = {
                "total_loss": float(total_loss.detach().item()),
                "plan_loss": float(plan_loss.detach().item()),
                "micro_kl_loss": float(micro_kl_loss.detach().item()),
                "node_surprise_loss": float(node_surprise_loss.detach().item()),
            }
            intervention_sum_metrics = {
                "d_t_sum": float(d_t_sum.detach().item()),
                "d_t_latent_sum": float(d_t_latent_sum.detach().item()),
                "d_t_node_sum": float(d_t_node_sum.detach().item()),
                "health_scalar_sum": float(health_scalar_sum.detach().item()),
                "curve_health_sum": float(curve_health_sum.detach().item()),
                "cand_health_sum": float(cand_health_sum.detach().item()),
                "health_bonus_sum": float(health_bonus_sum.detach().item()),
                "dt_temperature_sum": float(dt_temperature_sum.detach().item()),
                "d_t_norm_sum": float(d_t_norm_sum.detach().item()),
                "attn_health_sum": float(attn_health_sum.detach().item()),
                "s_t_norm_sum": float(s_t_norm_sum.detach().item()),
                "topo_novelty_sum": float(topo_novelty_sum.detach().item()),
                "topo_update_sum": float(topo_update_sum.detach().item()),
                "intervention_level_sum": float(intervention_level_sum.detach().item()),
                "dt_threshold_low_sum": float(dt_threshold_low_sum.detach().item()),
                "dt_threshold_high_sum": float(dt_threshold_high_sum.detach().item()),
                "score_before_sum": float(score_before_sum.detach().item()),
                "score_after_sum": float(score_after_sum.detach().item()),
                "strict_candidate_count_sum": float(strict_candidate_count_sum.detach().item()),
                "expanded_candidate_count_sum": float(expanded_candidate_count_sum.detach().item()),
            }
            count_metrics = {
                "total_actions": float(total_actions),
                "total_node_updates": float(total_node_updates),
            }

            scalar_metrics = self._reduce_float_dict(scalar_metrics, average=True)
            intervention_sum_metrics = self._reduce_float_dict(intervention_sum_metrics, average=False)
            count_metrics = self._reduce_float_dict(count_metrics, average=False)
            timing_metrics = self._reduce_float_dict(timing_sums, average=True)

            total_actions_global = max(float(count_metrics["total_actions"]), 1.0)
            self._last_step_metrics = {
                **scalar_metrics,
                "total_actions": float(count_metrics["total_actions"]),
                "total_node_updates": float(count_metrics["total_node_updates"]),
                "adapter_warmup": float(self._compute_adapter_warmup()),
                **timing_metrics,
            }
            if self._log_prior_post_metrics_enabled():
                self._last_step_metrics.update(
                    {
                        "d_t_fused_mean": float(intervention_sum_metrics["d_t_sum"]) / total_actions_global,
                        "d_t_latent_mean": float(intervention_sum_metrics["d_t_latent_sum"]) / total_actions_global,
                        "d_t_node_mean": float(intervention_sum_metrics["d_t_node_sum"]) / total_actions_global,
                        "d_t_norm_mean": float(intervention_sum_metrics["d_t_norm_sum"]) / total_actions_global,
                        "topo_novelty_mean": float(intervention_sum_metrics["topo_novelty_sum"]) / total_actions_global,
                        "topo_update_ratio": float(intervention_sum_metrics["topo_update_sum"]) / total_actions_global,
                        "dt_temperature_mean": float(intervention_sum_metrics["dt_temperature_sum"]) / total_actions_global,
                        "intervention_level_mean": float(intervention_sum_metrics["intervention_level_sum"]) / total_actions_global,
                        "dt_threshold_low_mean": float(intervention_sum_metrics["dt_threshold_low_sum"]) / total_actions_global,
                        "dt_threshold_high_mean": float(intervention_sum_metrics["dt_threshold_high_sum"]) / total_actions_global,
                        "score_before_mean": float(intervention_sum_metrics["score_before_sum"]) / total_actions_global,
                        "score_after_mean": float(intervention_sum_metrics["score_after_sum"]) / total_actions_global,
                        "strict_candidate_count_mean": float(
                            intervention_sum_metrics["strict_candidate_count_sum"]
                        ) / total_actions_global,
                        "expanded_candidate_count_mean": float(
                            intervention_sum_metrics["expanded_candidate_count_sum"]
                        ) / total_actions_global,
                    }
                )

            self.logs["total_loss"].append(self._last_step_metrics["total_loss"])
            self.logs["plan_loss"].append(self._last_step_metrics["plan_loss"])
            self.logs["micro_kl_loss"].append(self._last_step_metrics["micro_kl_loss"])
            self.logs["node_surprise_loss"].append(self._last_step_metrics["node_surprise_loss"])
            self.logs["adapter_warmup"].append(self._last_step_metrics["adapter_warmup"])
            if self._log_prior_post_metrics_enabled():
                self.logs["d_t_fused_mean"].append(self._last_step_metrics["d_t_fused_mean"])
                self.logs["d_t_latent_mean"].append(self._last_step_metrics["d_t_latent_mean"])
                self.logs["d_t_node_mean"].append(self._last_step_metrics["d_t_node_mean"])
                self.logs["d_t_norm_mean"].append(self._last_step_metrics["d_t_norm_mean"])
                self.logs["topo_novelty_mean"].append(self._last_step_metrics["topo_novelty_mean"])
                self.logs["topo_update_ratio"].append(self._last_step_metrics["topo_update_ratio"])
                self.logs["dt_temperature_mean"].append(self._last_step_metrics["dt_temperature_mean"])
                self.logs["intervention_level_mean"].append(self._last_step_metrics["intervention_level_mean"])
                self.logs["dt_threshold_low_mean"].append(self._last_step_metrics["dt_threshold_low_mean"])
                self.logs["dt_threshold_high_mean"].append(self._last_step_metrics["dt_threshold_high_mean"])
                self.logs["score_before_mean"].append(self._last_step_metrics["score_before_mean"])
                self.logs["score_after_mean"].append(self._last_step_metrics["score_after_mean"])
                self.logs["strict_candidate_count_mean"].append(
                    self._last_step_metrics["strict_candidate_count_mean"]
                )
                self.logs["expanded_candidate_count_mean"].append(
                    self._last_step_metrics["expanded_candidate_count_mean"]
                )
            self.logs["sample_ratio"].append(float(sample_ratio))

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        self._load_gt_data()
        self._prepare_v4_trace_output(checkpoint_index)
        return super()._eval_checkpoint(checkpoint_path, writer, checkpoint_index)
