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
from torch.nn.parallel import DistributedDataParallel as DDP

from fastdtw import fastdtw
from habitat_extensions.measures import NDTW
from vlnce_baselines.agents.efes_self_agent import EFESSelfAgent
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.datasets.state_label_builder import StateLabelBuilder
from vlnce_baselines.models.efes_self.topo_state_bank import TopoStateBankV2
from vlnce_baselines.models.graph_utils import GraphMap
from vlnce_baselines.trainers.train_statenav_v6_stage1 import (
    StateNavV6Trainer,
    _RELEASE_R2R_CKPT,
    _state_model,
)
from vlnce_baselines.utils import load_torch_checkpoint_compat


@baseline_registry.register_trainer(name="EFESSelf")
class EFESSelfTrainer(StateNavV6Trainer):
    stage_name = "EFESSelf"
    contract_version = "v3-grouped"
    accepted_contract_versions = ("v3-grouped",)

    def __init__(self, config=None):
        super().__init__(config)
        self._phase_name = "phase1"
        self._phase2_started = False
        self._efes_optimizer_group_names: List[str] = []
        self._checkpoint_mode = "train"
        self._last_checkpoint_fingerprint: Dict[str, Any] = {}

    def _efes_cfg(self):
        return self.config.EFES_SELF

    def _inspect_checkpoint(self, checkpoint_path: str):
        checkpoint_realpath = os.path.realpath(checkpoint_path)
        ckpt = load_torch_checkpoint_compat(checkpoint_realpath, map_location="cpu")
        has_statenav_state = "statenav_state_dict" in ckpt
        fingerprint = {
            "checkpoint_path": checkpoint_path,
            "checkpoint_realpath": checkpoint_realpath,
            "checkpoint_type": "efes-full" if has_statenav_state else "bootstrap-only",
            "has_statenav_state": bool(has_statenav_state),
            "iteration": int(ckpt.get("iteration", 0)),
            "efes_contract_version": ckpt.get("efes_contract_version", None),
            "efes_phase": ckpt.get("efes_phase", None),
            "has_optim_state": "optim_state" in ckpt,
        }
        return ckpt, checkpoint_realpath, fingerprint

    def _log_checkpoint_fingerprint(self, fingerprint: Dict[str, Any], context: str) -> None:
        logger.info(
            "[EFES CKPT %s] path=%s type=%s has_statenav_state=%s iteration=%s contract=%s phase=%s optim=%s",
            context,
            fingerprint.get("checkpoint_path"),
            fingerprint.get("checkpoint_type"),
            fingerprint.get("has_statenav_state"),
            fingerprint.get("iteration"),
            fingerprint.get("efes_contract_version"),
            fingerprint.get("efes_phase"),
            fingerprint.get("has_optim_state"),
        )

    def _validate_efes_checkpoint_contract(
        self,
        fingerprint: Dict[str, Any],
        *,
        mode: str,
        is_requeue: bool,
    ) -> None:
        has_statenav_state = bool(fingerprint.get("has_statenav_state", False))
        contract_version = fingerprint.get("efes_contract_version")
        checkpoint_realpath = str(fingerprint.get("checkpoint_realpath"))

        if mode in ("eval", "inference"):
            if not has_statenav_state:
                raise RuntimeError(
                    "EFES {} only accepts EFESSelf checkpoints with statenav_state_dict. "
                    "Received bootstrap-only checkpoint {}. Use run_r2r/main.bash for baseline ETPNav evaluation."
                    .format(mode, checkpoint_realpath)
                )
            if contract_version not in self.accepted_contract_versions:
                raise RuntimeError(
                    "EFES {} rejected checkpoint {} with unsupported contract version {}. "
                    "Expected one of {}.".format(
                        mode,
                        checkpoint_realpath,
                        contract_version,
                        list(self.accepted_contract_versions),
                    )
                )
            if fingerprint.get("efes_phase") is None:
                raise RuntimeError(
                    "EFES {} rejected checkpoint {} because efes_phase is missing.".format(
                        mode, checkpoint_realpath
                    )
                )
            return

        if is_requeue:
            if not has_statenav_state:
                raise RuntimeError(
                    "EFES resume requires a full EFESSelf checkpoint with statenav_state_dict, got {}.".format(
                        checkpoint_realpath
                    )
                )
            if contract_version not in self.accepted_contract_versions:
                raise RuntimeError(
                    "EFES resume rejected checkpoint {} with unsupported contract version {}. "
                    "Expected one of {}.".format(
                        checkpoint_realpath,
                        contract_version,
                        list(self.accepted_contract_versions),
                    )
                )
            if not bool(fingerprint.get("has_optim_state", False)):
                raise RuntimeError(
                    "EFES resume requires optim_state in checkpoint {}, but it is missing.".format(
                        checkpoint_realpath
                    )
                )
            return

        if not has_statenav_state and checkpoint_realpath != _RELEASE_R2R_CKPT:
            raise RuntimeError(
                "EFES fresh training only allows bootstrap-only checkpoint {}. Got {}.".format(
                    _RELEASE_R2R_CKPT, checkpoint_realpath
                )
            )
        if has_statenav_state and contract_version not in self.accepted_contract_versions:
            raise RuntimeError(
                "EFES training rejected checkpoint {} with unsupported contract version {}. "
                "Expected one of {}.".format(
                    checkpoint_realpath,
                    contract_version,
                    list(self.accepted_contract_versions),
                )
            )

    def _get_eval_result_metadata(self) -> Dict[str, Any]:
        payload = dict(self._last_checkpoint_fingerprint) if self._last_checkpoint_fingerprint else {}
        payload.update(
            {
                "action_source": self._efes_action_source(),
                "back_algo": str(self.config.IL.back_algo),
                "conditioner_mode": str(getattr(self._efes_cfg(), "conditioner_mode", "prob_mixture")),
                "conditioner_beta_max": float(getattr(self._efes_cfg(), "conditioner_beta_max", 0.1)),
                "conditioner_stop_delta_scale": float(
                    getattr(self._efes_cfg(), "conditioner_stop_delta_scale", 1.0)
                ),
                "use_self_revision": bool(getattr(self._efes_cfg(), "use_self_revision", True)),
                "use_macro_rupture": bool(getattr(self._efes_cfg(), "use_macro_rupture", True)),
                "use_grounding_rupture": bool(getattr(self._efes_cfg(), "use_grounding_rupture", True)),
                "use_typed_rupture": bool(getattr(self._efes_cfg(), "use_typed_rupture", True)),
            }
        )
        return payload

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

        policy_load_msg = None
        if "policy_state_dict" in ckpt:
            policy_state_dict = self._align_state_dict_prefix(ckpt["policy_state_dict"], self.policy)
            policy_load_msg = self.policy.load_state_dict(policy_state_dict, strict=False)
        elif "state_dict" in ckpt:
            policy_state_dict = self._align_state_dict_prefix(ckpt["state_dict"], self.policy)
            policy_load_msg = self.policy.load_state_dict(policy_state_dict, strict=False)
        else:
            raise KeyError(
                "Checkpoint {} has neither state_dict nor policy_state_dict".format(checkpoint_path)
            )

        policy_missing = len(policy_load_msg.missing_keys)
        policy_unexpected = len(policy_load_msg.unexpected_keys)
        logger.info(
            "[Policy CKPT] missing_keys(%d) unexpected_keys(%d)",
            policy_missing,
            policy_unexpected,
        )
        if policy_missing or policy_unexpected:
            raise RuntimeError(
                "Invalid Policy checkpoint load for {}: missing_keys={}, unexpected_keys={}".format(
                    checkpoint_path, policy_missing, policy_unexpected
                )
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
            missing_keys = list(load_msg.missing_keys)
            unexpected_keys = list(load_msg.unexpected_keys)
            allowed_missing_prefixes = ("action_conditioner.", "module.action_conditioner.")
            conditioner_missing_only = bool(missing_keys) and all(
                any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
                for key in missing_keys
            )
            allow_fresh_conditioner = (
                conditioner_missing_only
                and not unexpected_keys
                and (
                    (mode == "train" and not bool(self.config.IL.is_requeue))
                    or self._efes_action_source() == "etp"
                )
            )
            if allow_fresh_conditioner:
                logger.warning(
                    "Checkpoint %s has no action_conditioner weights; initializing the conditioner freshly "
                    "for action_source=%s mode=%s.",
                    checkpoint_path,
                    self._efes_action_source(),
                    mode,
                )
            elif conditioner_missing_only and not unexpected_keys:
                raise RuntimeError(
                    "Checkpoint {} is an older EFESSelf checkpoint without action_conditioner weights. "
                    "action_source={} mode={} requires an active EFES checkpoint. "
                    "Run active training from this checkpoint first, then evaluate the newly saved active checkpoint.".format(
                        checkpoint_path,
                        self._efes_action_source(),
                        mode,
                    )
                )
            elif statenav_missing or statenav_unexpected:
                raise RuntimeError(
                    "Invalid StateNav checkpoint load for {}: missing_keys={}, unexpected_keys={}".format(
                        checkpoint_path, statenav_missing, statenav_unexpected
                    )
                )
        else:
            logger.info("[StateNav CKPT] missing_keys(0) unexpected_keys(0) [bootstrap-only initialization]")

        if has_statenav_state and self.config.IL.is_requeue and "optim_state" in ckpt:
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
                "Loaded EFESSelf checkpoint for %s: %s at iteration %d [contract=%s phase=%s]",
                mode,
                checkpoint_path,
                start_iter,
                fingerprint.get("efes_contract_version"),
                fingerprint.get("efes_phase"),
            )
        else:
            logger.info(
                "Loaded bootstrap-only ETPNav checkpoint for training: %s. StateNav modules keep fresh initialization.",
                checkpoint_path,
            )
        return start_iter

    def eval(self):
        self._checkpoint_mode = "eval"
        super().eval()

    def inference(self) -> None:
        self._checkpoint_mode = "inference"
        super().inference()

    def _eval_episode_allowlist(self) -> set:
        raw = getattr(self.config.EVAL, "EPISODE_ID_ALLOWLIST", "")
        if raw is None:
            return set()
        if isinstance(raw, str):
            raw = raw.replace(";", ",")
            values = [item.strip() for item in raw.split(",")]
        elif isinstance(raw, (list, tuple)):
            values = [str(item).strip() for item in raw]
        else:
            values = [str(raw).strip()]
        return {item for item in values if item}

    def collect_val_traj(self):
        trajectories = super().collect_val_traj()
        if getattr(self, "_checkpoint_mode", "train") != "eval":
            return trajectories

        allowlist = self._eval_episode_allowlist()
        if not allowlist:
            return trajectories

        filtered = [ep_id for ep_id in trajectories if str(ep_id) in allowlist]
        found = {str(ep_id) for ep_id in filtered}
        missing = sorted(allowlist.difference(found))
        logger.info(
            "EFES eval episode allowlist active: kept %d/%d episodes on rank %d. missing_on_rank=%s",
            len(filtered),
            len(trajectories),
            int(getattr(self.config, "local_rank", 0)),
            ",".join(missing) if missing else "<none>",
        )
        if not filtered:
            raise ValueError(
                "EVAL.EPISODE_ID_ALLOWLIST did not match any episodes on this eval rank. "
                f"Requested={sorted(allowlist)}"
            )
        return filtered

    def _logging_cfg(self):
        return self.config.EFES_SELF.LOGGING

    def _efes_action_source(self) -> str:
        source = str(getattr(self._efes_cfg(), "action_source", "etp")).strip().lower()
        if source not in {"etp", "efes_safe", "efes_hard"}:
            logger.warning("Unknown EFES_SELF.action_source=%s. Falling back to etp.", source)
            source = "etp"
        return source

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

        self.statenav_agent = EFESSelfAgent.from_config(
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

    def _maybe_activate_phase2(self) -> None:
        if self._phase2_started:
            return
        if not bool(getattr(self._efes_cfg(), "enable_phase2", False)):
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
            "plan_loss",
            "self_loss",
            "cons_loss",
            "aux_loss",
            "trust_loss",
            "bind_loss",
            "kl_loss",
            "node_nll_loss",
            "local_pred_loss",
            "topo_pred_loss",
            "prog_pred_loss",
            "clarity_loss",
            "total_actions",
            "total_macro_updates",
            "c_micro_mean",
            "c_macro_diag_mean",
            "u_macro_mean",
            "g_t_mean",
            "self_mismatch_mean",
            "kappa_self_mean",
            "self_clarity_mean",
            "r_local_mean",
            "r_topo_mean",
            "r_ground_mean",
            "rupture_conf_mean",
            "self_agency_mean",
            "self_phase_mean",
            "self_continuity_mean",
            "macro_valid_ratio",
            "mode_hist_argmax",
            "condition_gate_mean",
            "condition_delta_mean",
            "condition_policy_kl_mean",
            "score_before_mean",
            "score_after_mean",
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
        record["contract_version"] = self.contract_version
        record["phase"] = self._phase_name
        record["action_source"] = self._efes_action_source()
        return record

    def _log_step_metric_record(self, record: Dict[str, Any]) -> None:
        if not self._is_main_process():
            return
        step_log_every = max(int(self._logging_cfg().step_log_every), 1)
        if int(record["iteration"]) % step_log_every != 0:
            return
        logger.info(
            "[EFESSelf %s %s action=%s %06d/%06d | interval %03d/%03d] total=%.4f plan=%.4f self=%.4f cons=%.4f aux=%.4f grad=%.4f lr=%.2e"
            " | detail(bind=%.4f kl=%.4f node_nll=%.4f local=%.4f topo=%.4f prog=%.4f clarity=%.4f)"
            " | c=%.3f/%.3f u=%.3f g=%.3f mismatch=%.3f kappa=%.3f clarity=%.3f rupture=%.3f/%.3f/%.3f conf=%.3f self=%.3f/%.3f/%.3f macro=%.3f modes=%s cond=%.3f/%.3f score=%.3f/%.3f etp=%.4f"
            " | sec(step=%.2f roll=%.2f bw=%.2f opt=%.2f lang=%.2f wp=%.2f pano=%.2f nav=%.2f efes=%.2f env=%.2f prep=%.2f)",
            str(record.get("phase", "phase1")),
            str(record.get("contract_version", self.contract_version)),
            str(record.get("action_source", self._efes_action_source())),
            int(record["iteration"]),
            int(self.config.IL.iters),
            int(record.get("interval_step", 0)),
            int(record.get("interval_size", 0)),
            float(record.get("total_loss", 0.0)),
            float(record.get("plan_loss", 0.0)),
            float(record.get("self_loss", 0.0)),
            float(record.get("cons_loss", 0.0)),
            float(record.get("aux_loss", 0.0)),
            float(record.get("grad_norm", 0.0)),
            float(record.get("lr", 0.0)),
            float(record.get("bind_loss", 0.0)),
            float(record.get("kl_loss", 0.0)),
            float(record.get("node_nll_loss", 0.0)),
            float(record.get("local_pred_loss", 0.0)),
            float(record.get("topo_pred_loss", 0.0)),
            float(record.get("prog_pred_loss", 0.0)),
            float(record.get("clarity_loss", 0.0)),
            float(record.get("c_micro_mean", 0.0)),
            float(record.get("c_macro_diag_mean", 0.0)),
            float(record.get("u_macro_mean", 0.0)),
            float(record.get("g_t_mean", 0.0)),
            float(record.get("self_mismatch_mean", 0.0)),
            float(record.get("kappa_self_mean", 0.0)),
            float(record.get("self_clarity_mean", 0.0)),
            float(record.get("r_local_mean", 0.0)),
            float(record.get("r_topo_mean", 0.0)),
            float(record.get("r_ground_mean", 0.0)),
            float(record.get("rupture_conf_mean", 0.0)),
            float(record.get("self_agency_mean", 0.0)),
            float(record.get("self_phase_mean", 0.0)),
            float(record.get("self_continuity_mean", 0.0)),
            float(record.get("macro_valid_ratio", 0.0)),
            str(record.get("mode_hist_argmax", "-")),
            float(record.get("condition_gate_mean", 0.0)),
            float(record.get("condition_delta_mean", 0.0)),
            float(record.get("score_before_mean", 0.0)),
            float(record.get("score_after_mean", 0.0)),
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
        import re

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
                "efes_contract_version": self.contract_version,
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
                    "efes_contract_version": self.contract_version,
                },
                f=best_path,
            )

        max_keep = max(int(self._efes_cfg().max_keep_checkpoints), 1)
        def _iter_key(path: str) -> int:
            match = re.search(r"ckpt\.iter(\d+)\.pth$", path)
            return int(match.group(1)) if match else -1

        existing = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt.iter*.pth")), key=_iter_key)
        while len(existing) > max_keep:
            oldest = existing.pop(0)
            os.remove(oldest)
            logger.info("Removed old EFES checkpoint: %s", oldest)

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
        if bool(getattr(self._efes_cfg(), "enable_phase2", False)) and int(start_iter) >= int(self._efes_cfg().phase1_iters):
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

        logger.info("EFES-Self training starts...")
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
                    "[EFES-Self summary %06d | %s] %s",
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
            tqdm.trange(interval, leave=False, dynamic_ncols=True, desc=f"EFESSelf {self._phase_name}")
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
            for param in trainable_params:
                if not torch.isfinite(param.grad).all():
                    param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
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

    def _micro_kl_training_term(self, c_micro_kl_raw: Tensor) -> Tensor:
        free_nats = float(self._efes_cfg().micro_free_nats)
        kl_cap = float(self._efes_cfg().micro_kl_cap)
        return c_micro_kl_raw.clamp_min(free_nats).clamp_max(kl_cap)

    def _reality_echo_inputs(
        self,
        pos_history: Sequence[Deque[Tensor]],
        vp_history: Sequence[Deque[Any]],
        current_pos: Sequence[Any],
        current_vp: Sequence[Any],
        frontier_size: Tensor,
        prev_frontier_size: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = len(current_pos)
        path_delta = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        topo_advanced = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        frontier_delta = frontier_size.detach().to(dtype=torch.float32) - prev_frontier_size.to(dtype=torch.float32)
        for batch_idx in range(batch_size):
            current_pos_tensor = torch.as_tensor(
                current_pos[batch_idx],
                device=self.device,
                dtype=torch.float32,
            )
            if pos_history[batch_idx]:
                path_delta[batch_idx] = torch.norm(
                    current_pos_tensor - pos_history[batch_idx][-1],
                    p=2,
                )
            previous_vp = vp_history[batch_idx][-1] if vp_history[batch_idx] else None
            topo_advanced[batch_idx] = 1.0 if (previous_vp is not None and current_vp[batch_idx] != previous_vp) else 0.0
        return path_delta, topo_advanced, frontier_delta

    @staticmethod
    def _history_mean(history: Deque[float]) -> float:
        if not history:
            return 0.0
        return float(sum(history) / len(history))

    @staticmethod
    def _recovery_hist_string(mode_counts: Dict[int, float]) -> str:
        total = max(sum(mode_counts.values()), 1.0)
        return "|".join(
            f"{mode}:{mode_counts.get(mode, 0.0) / total:.3f}" for mode in (0, 1, 2)
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
        local_pred_loss_sum = torch.zeros((), device=self.device)
        topo_pred_loss_sum = torch.zeros((), device=self.device)
        prog_pred_loss_sum = torch.zeros((), device=self.device)
        clarity_loss_sum = torch.zeros((), device=self.device)
        ddp_anchor_sum = torch.zeros((), device=self.device)
        c_micro_sum = torch.zeros((), device=self.device)
        c_macro_diag_sum = torch.zeros((), device=self.device)
        u_macro_sum = torch.zeros((), device=self.device)
        g_t_sum = torch.zeros((), device=self.device)
        kappa_self_sum = torch.zeros((), device=self.device)
        self_clarity_sum = torch.zeros((), device=self.device)
        r_local_sum = torch.zeros((), device=self.device)
        r_topo_sum = torch.zeros((), device=self.device)
        r_ground_sum = torch.zeros((), device=self.device)
        rupture_conf_sum = torch.zeros((), device=self.device)
        self_agency_sum = torch.zeros((), device=self.device)
        self_phase_sum = torch.zeros((), device=self.device)
        self_continuity_sum = torch.zeros((), device=self.device)
        macro_valid_sum = torch.zeros((), device=self.device)
        mode_count_sum = torch.zeros(3, device=self.device)
        bind_loss_sum = torch.zeros((), device=self.device)
        self_mismatch_sum = torch.zeros((), device=self.device)
        condition_gate_sum = torch.zeros((), device=self.device)
        condition_delta_sum = torch.zeros((), device=self.device)
        condition_policy_kl_sum = torch.zeros((), device=self.device)
        score_before_sum = torch.zeros((), device=self.device)
        score_after_sum = torch.zeros((), device=self.device)

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
        active_pos_histories = [
            deque(maxlen=int(self._efes_cfg().history_window)) for _ in range(self.envs.num_envs)
        ]
        active_vp_histories = [
            deque(maxlen=int(self._efes_cfg().history_window)) for _ in range(self.envs.num_envs)
        ]
        prev_frontier_size = torch.zeros(self.envs.num_envs, device=self.device, dtype=torch.float32)
        active_eval_diag = [
            {"a_t": [], "c_micro": [], "c_macro": [], "g_t": [], "macro_valid": []}
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
            candidate_counts = frontier_candidate_mask.to(dtype=torch.long).sum(dim=-1)
            current_comp_feat = state_model.macro_head.compress(avg_pano_embeds).detach()
            current_topo_novelty = state_model.macro_head.cosine_novelty(
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
            path_delta, topo_advanced, frontier_delta = self._reality_echo_inputs(
                pos_history=active_pos_histories,
                vp_history=active_vp_histories,
                current_pos=cur_pos,
                current_vp=cur_vp,
                frontier_size=frontier_size,
                prev_frontier_size=prev_frontier_size,
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
                path_delta=path_delta,
                topo_advanced=topo_advanced,
                frontier_delta=frontier_delta,
                topo_bank_feat=topo_bank.bank_feat,
                topo_bank_mask=topo_bank.bank_mask,
                macro_valid_mask=topo_update_mask,
                candidate_mask=candidate_mask,
                invalid_candidate_mask=invalid_candidate_mask,
                frontier_size=frontier_size,
                revisit_count=revisit_count,
                loop_evidence=loop_evidence,
                history_backtrack_values=history_backtrack_values,
                history_valid_mask=history_valid_mask,
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
            kappa_self_sum = kappa_self_sum + step_outs["kappa_self"].sum()
            self_clarity_sum = self_clarity_sum + step_outs["self_clarity"].sum()
            r_local_sum = r_local_sum + step_outs["delta_control"].sum()
            r_topo_sum = r_topo_sum + step_outs["delta_boundary"].sum()
            r_ground_sum = r_ground_sum + step_outs["delta_progress"].sum()
            rupture_conf_sum = rupture_conf_sum + step_outs["rupture_confidence"].sum()
            self_agency_sum = self_agency_sum + step_outs["self_agency"].sum()
            self_phase_sum = self_phase_sum + step_outs["self_phase"].sum()
            self_continuity_sum = self_continuity_sum + step_outs["self_continuity"].sum()
            u_macro_sum = u_macro_sum + step_outs["U_macro"].sum()
            macro_valid_sum = macro_valid_sum + macro_valid_mask.to(torch.float32).sum()
            total_macro_updates += int(macro_valid_mask.sum().item())
            self_mismatch_sum = self_mismatch_sum + step_outs["self_mismatch"].sum()
            condition_gate_sum = condition_gate_sum + step_outs["condition_gate"].sum()
            condition_delta_sum = condition_delta_sum + step_outs["condition_delta"].sum()
            condition_policy_kl_sum = condition_policy_kl_sum + step_outs["condition_policy_kl"].sum()
            score_before_sum = score_before_sum + step_outs["score_before_mean"].sum()
            score_after_sum = score_after_sum + step_outs["score_after_mean"].sum()
            for mode_idx in range(3):
                mode_count_sum[mode_idx] = mode_count_sum[mode_idx] + step_outs["argmax_mode"].eq(mode_idx).sum()

            if mode == "eval" and active_eval_diag is not None:
                for i in range(self.envs.num_envs):
                    active_eval_diag[i]["a_t"].append(float(step_outs["A_t"][i].detach().item()))
                    active_eval_diag[i]["c_micro"].append(float(step_outs["C_micro"][i].detach().item()))
                    active_eval_diag[i]["c_macro"].append(float(step_outs["C_macro_diag"][i].detach().item()))
                    active_eval_diag[i]["g_t"].append(float(step_outs["g_t"][i].detach().item()))
                    active_eval_diag[i]["macro_valid"].append(float(step_outs["macro_valid_mask"][i].detach().item()))

            if mode == "train" or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs["gmap_vp_ids"], no_vp_left)
            else:
                teacher_actions = None

            action_source = self._efes_action_source()
            if action_source == "efes_safe":
                action_logits = step_outs["safe_logits"]
            elif action_source == "efes_hard":
                action_logits = step_outs["hard_logits"]
            else:
                action_logits = nav_logits
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
                plan_loss_sum = plan_loss_sum + F.cross_entropy(
                    loss_logits,
                    teacher_actions.long(),
                    ignore_index=-100,
                    reduction="sum",
                )
                kl_loss_sum = kl_loss_sum + self._micro_kl_training_term(step_outs["C_micro_kl_raw"]).sum()
                node_weight = step_outs["macro_valid_mask"].to(dtype=step_outs["node_nll_loss"].dtype)
                node_nll_sum = node_nll_sum + (step_outs["node_nll_loss"] * node_weight).sum()
                local_pred_loss_sum = local_pred_loss_sum + step_outs["local_pred_loss"].sum()
                topo_pred_loss_sum = topo_pred_loss_sum + step_outs["topo_pred_loss"].sum()
                prog_pred_loss_sum = prog_pred_loss_sum + step_outs["prog_pred_loss"].sum()
                local_consistency = torch.exp(
                    -(
                        step_outs["delta_control"].detach()
                        + step_outs["delta_boundary"].detach()
                        + step_outs["delta_progress"].detach()
                    ) / 3.0
                ).clamp(0.05, 0.95)
                with cuda_autocast(enabled=False):
                    bind_loss_sum = bind_loss_sum + F.binary_cross_entropy(
                        step_outs["kappa_self"].float().clamp(1e-4, 1.0 - 1e-4),
                        local_consistency.float(),
                        reduction="sum",
                    )
                    clarity_loss_sum = clarity_loss_sum + F.binary_cross_entropy(
                        step_outs["self_clarity"].float().clamp(1e-4, 1.0 - 1e-4),
                        local_consistency.float(),
                        reduction="sum",
                    )
                ddp_anchor_sum = ddp_anchor_sum + (
                    0.0 * step_outs["alpha_prior"].sum()
                    + 0.0 * step_outs["alpha_progress"].sum()
                )

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
            if bool(getattr(self._efes_cfg(), "use_self_revision", True)):
                recurrent_state["prev_self"] = step_outs["self_post_z"]
            else:
                recurrent_state["prev_self"] = step_outs["self_pred_z"]
            recurrent_state["prev_rssm_h"] = step_outs["rssm_h_t"]
            recurrent_state["prev_z"] = step_outs["z_flat"]
            recurrent_state["prev_progress"] = step_outs["self_phase"].unsqueeze(-1)
            recurrent_state["prev_prior_alpha"] = step_outs["alpha_prior"]
            recurrent_state["prev_topo_novelty"] = step_outs["topo_novelty"]
            recurrent_state["prev_progress_gap"] = step_outs["delta_progress"]

            for i in range(self.envs.num_envs):
                self._update_backtrack_history(
                    history=active_backtrack_histories[i],
                    vp_ids=nav_inputs["gmap_vp_ids"][i],
                    invalid_mask_row=invalid_candidate_mask[i],
                    visited_mask_row=visited_candidate_mask[i],
                    score_row=action_logits[i],
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
                        metric["mean_a_t"] = self._mean_float_list(diag["a_t"])
                        metric["mean_c_micro"] = self._mean_float_list(diag["c_micro"])
                        metric["mean_c_macro"] = self._mean_float_list(diag["c_macro"])
                        metric["mean_g_t"] = self._mean_float_list(diag["g_t"])
                        metric["mean_macro_valid"] = self._mean_float_list(diag["macro_valid"])
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
                    if active_eval_diag is not None:
                        running_means.update(
                            {
                                "mean_a_t": sum(v["mean_a_t"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_c_micro": sum(v["mean_c_micro"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_c_macro": sum(v["mean_c_macro"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_g_t": sum(v["mean_g_t"] for v in self.stat_eps.values()) / evaluated_eps,
                                "mean_macro_valid": sum(v["mean_macro_valid"] for v in self.stat_eps.values()) / evaluated_eps,
                            }
                        )
                    if self.pbar is not None and self.local_rank < 1:
                        self.pbar.set_postfix(
                            {
                                "ep": f"{evaluated_eps}/{eval_total}",
                                "succ": f"{running_means['success']:.3f}",
                                "spl": f"{running_means['spl']:.3f}",
                                "ndtw": f"{running_means['ndtw']:.3f}",
                                "sdtw": f"{running_means['sdtw']:.3f}",
                            }
                        )
                    logger.info(
                        "[EP-DONE] ep=%s | succ=%.0f oracle=%.0f | d2g=%.2f steps=%s "
                        "path_len=%.2f gt_len=%.2f | spl=%.3f ndtw=%.3f sdtw=%.3f",
                        ep_id,
                        metric["success"],
                        metric["oracle_success"],
                        metric["distance_to_goal"],
                        metric["steps_taken"],
                        metric["path_length"],
                        gt_length,
                        metric["spl"],
                        metric["ndtw"],
                        metric["sdtw"],
                    )
                    log_every_episode = max(int(getattr(self.config.EVAL, "LOG_EVERY_EPISODE", 1)), 1)
                    if self.local_rank < 1 and (
                        evaluated_eps == 1
                        or evaluated_eps % log_every_episode == 0
                        or evaluated_eps == eval_total
                    ):
                        logger.info(
                            "[EVAL-LIVE] ep=%d/%d | success=%.3f spl=%.3f ndtw=%.3f sdtw=%.3f d2g=%.3f",
                            evaluated_eps,
                            eval_total,
                            running_means["success"],
                            running_means["spl"],
                            running_means["ndtw"],
                            running_means["sdtw"],
                            running_means["distance_to_goal"],
                        )
                    self._write_live_eval_progress(
                        ep_id=ep_id,
                        metric=metric,
                        running_means=running_means,
                        evaluated_eps=evaluated_eps,
                        eval_total=eval_total,
                    )
                    if self.pbar is not None:
                        self.pbar.update()

            next_frontier_size = frontier_size.detach().to(dtype=torch.float32)
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
                    active_pos_histories.pop(i)
                    active_vp_histories.pop(i)
                    if active_eval_diag is not None:
                        active_eval_diag.pop(i)

                if keep_indices:
                    keep_tensor = torch.tensor(keep_indices, device=self.device, dtype=torch.long)
                    prev_frontier_size = prev_frontier_size.index_select(0, keep_tensor)
                    next_frontier_size = next_frontier_size.index_select(0, keep_tensor)
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
                active_pos_histories[i].append(
                    torch.as_tensor(cur_pos[i], device=self.device, dtype=torch.float32)
                )
                active_vp_histories[i].append(cur_vp[i])
            prev_frontier_size = next_frontier_size

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
            bind_loss = bind_loss_sum / total_actions_f
            kl_loss = kl_loss_sum / total_actions_f
            node_nll_loss = node_nll_sum / total_macro_updates_f
            local_pred_loss = local_pred_loss_sum / total_actions_f
            topo_pred_loss = topo_pred_loss_sum / total_actions_f
            prog_pred_loss = prog_pred_loss_sum / total_actions_f
            clarity_loss = clarity_loss_sum / total_actions_f
            self_loss = kl_loss
            cons_loss = (
                float(self._efes_cfg().lambda_local) * local_pred_loss
                + float(self._efes_cfg().lambda_topo) * topo_pred_loss
                + float(self._efes_cfg().lambda_prog) * prog_pred_loss
                + float(self._efes_cfg().lambda_node) * node_nll_loss
            )
            aux_loss = (
                float(self._efes_cfg().lambda_bind) * bind_loss
                + float(self._efes_cfg().lambda_clarity) * clarity_loss
            )
            condition_policy_kl = torch.nan_to_num(
                condition_policy_kl_sum / total_actions_f,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            condition_beta_mean = condition_gate_sum / total_actions_f
            condition_budget_loss = F.relu(
                condition_beta_mean - float(getattr(self._efes_cfg(), "conditioner_gate_budget", 0.08))
            ).square()
            lambda_condition_kl = float(getattr(self._efes_cfg(), "lambda_condition_kl", 0.0))
            lambda_condition_budget = float(getattr(self._efes_cfg(), "lambda_condition_budget", 0.05))
            trust_loss = lambda_condition_budget * condition_budget_loss
            if lambda_condition_kl != 0.0:
                trust_loss = trust_loss + lambda_condition_kl * condition_policy_kl
            total_loss = (
                float(self._efes_cfg().lambda_plan) * plan_loss
                + float(getattr(self._efes_cfg(), "lambda_self", 0.2)) * self_loss
                + float(getattr(self._efes_cfg(), "lambda_cons", 0.5)) * cons_loss
                + float(getattr(self._efes_cfg(), "lambda_aux", 0.0)) * aux_loss
                + trust_loss
                + ddp_anchor_sum
            )
            self.loss = self.loss + ml_weight * total_loss

            scalar_metrics = {
                "total_loss": float(total_loss.detach().item()),
                "plan_loss": float(plan_loss.detach().item()),
                "self_loss": float(self_loss.detach().item()),
                "cons_loss": float(cons_loss.detach().item()),
                "aux_loss": float(aux_loss.detach().item()),
                "trust_loss": float(trust_loss.detach().item()),
                "bind_loss": float(bind_loss.detach().item()),
                "kl_loss": float(kl_loss.detach().item()),
                "node_nll_loss": float(node_nll_loss.detach().item()),
                "local_pred_loss": float(local_pred_loss.detach().item()),
                "topo_pred_loss": float(topo_pred_loss.detach().item()),
                "prog_pred_loss": float(prog_pred_loss.detach().item()),
                "clarity_loss": float(clarity_loss.detach().item()),
                "condition_policy_kl": float(condition_policy_kl.detach().item()),
                "condition_budget_loss": float(condition_budget_loss.detach().item()),
            }
            sum_metrics = {
                "c_micro_sum": float(c_micro_sum.detach().item()),
                "c_macro_diag_sum": float(c_macro_diag_sum.detach().item()),
                "u_macro_sum": float(u_macro_sum.detach().item()),
                "g_t_sum": float(g_t_sum.detach().item()),
                "kappa_self_sum": float(kappa_self_sum.detach().item()),
                "self_clarity_sum": float(self_clarity_sum.detach().item()),
                "r_local_sum": float(r_local_sum.detach().item()),
                "r_topo_sum": float(r_topo_sum.detach().item()),
                "r_ground_sum": float(r_ground_sum.detach().item()),
                "rupture_conf_sum": float(rupture_conf_sum.detach().item()),
                "self_agency_sum": float(self_agency_sum.detach().item()),
                "self_phase_sum": float(self_phase_sum.detach().item()),
                "self_continuity_sum": float(self_continuity_sum.detach().item()),
                "self_mismatch_sum": float(self_mismatch_sum.detach().item()),
                "macro_valid_sum": float(macro_valid_sum.detach().item()),
                "condition_gate_sum": float(condition_gate_sum.detach().item()),
                "condition_delta_sum": float(condition_delta_sum.detach().item()),
                "condition_policy_kl_sum": float(condition_policy_kl_sum.detach().item()),
                "score_before_sum": float(score_before_sum.detach().item()),
                "score_after_sum": float(score_after_sum.detach().item()),
                "mode_count_0": float(mode_count_sum[0].detach().item()),
                "mode_count_1": float(mode_count_sum[1].detach().item()),
                "mode_count_2": float(mode_count_sum[2].detach().item()),
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
                }
            )
            self._last_step_metrics = {
                **scalar_metrics,
                "contract_version": self.contract_version,
                "total_actions": float(count_metrics["total_actions"]),
                "total_macro_updates": float(count_metrics["total_macro_updates"]),
                "c_micro_mean": float(sum_metrics["c_micro_sum"]) / total_actions_global,
                "c_macro_diag_mean": float(sum_metrics["c_macro_diag_sum"]) / total_macro_updates_global,
                "u_macro_mean": float(sum_metrics["u_macro_sum"]) / total_actions_global,
                "g_t_mean": float(sum_metrics["g_t_sum"]) / total_actions_global,
                "self_mismatch_mean": float(sum_metrics["self_mismatch_sum"]) / total_actions_global,
                "kappa_self_mean": float(sum_metrics["kappa_self_sum"]) / total_actions_global,
                "self_clarity_mean": float(sum_metrics["self_clarity_sum"]) / total_actions_global,
                "r_local_mean": float(sum_metrics["r_local_sum"]) / total_actions_global,
                "r_topo_mean": float(sum_metrics["r_topo_sum"]) / total_actions_global,
                "r_ground_mean": float(sum_metrics["r_ground_sum"]) / total_actions_global,
                "rupture_conf_mean": float(sum_metrics["rupture_conf_sum"]) / total_actions_global,
                "self_agency_mean": float(sum_metrics["self_agency_sum"]) / total_actions_global,
                "self_phase_mean": float(sum_metrics["self_phase_sum"]) / total_actions_global,
                "self_continuity_mean": float(sum_metrics["self_continuity_sum"]) / total_actions_global,
                "macro_valid_ratio": float(sum_metrics["macro_valid_sum"]) / total_actions_global,
                "mode_hist_argmax": mode_hist,
                "condition_gate_mean": float(sum_metrics["condition_gate_sum"]) / total_actions_global,
                "condition_delta_mean": float(sum_metrics["condition_delta_sum"]) / total_actions_global,
                "condition_policy_kl_mean": float(sum_metrics["condition_policy_kl_sum"]) / total_actions_global,
                "score_before_mean": float(sum_metrics["score_before_sum"]) / total_actions_global,
                "score_after_mean": float(sum_metrics["score_after_sum"]) / total_actions_global,
                **timing_metrics,
            }
            for key, value in scalar_metrics.items():
                self.logs[key].append(value)
            for key in (
                "c_micro_mean",
                "c_macro_diag_mean",
                "u_macro_mean",
                "g_t_mean",
                "self_mismatch_mean",
                "kappa_self_mean",
                "self_clarity_mean",
                "r_local_mean",
                "r_topo_mean",
                "r_ground_mean",
                "rupture_conf_mean",
                "self_agency_mean",
                "self_phase_mean",
                "self_continuity_mean",
                "macro_valid_ratio",
                "condition_gate_mean",
                "condition_delta_mean",
                "condition_policy_kl_mean",
                "score_before_mean",
                "score_after_mean",
            ):
                self.logs[key].append(float(self._last_step_metrics[key]))
