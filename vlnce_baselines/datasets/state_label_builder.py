from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _wrap_angle(angle: float) -> float:
    while angle <= -math.pi:
        angle += 2 * math.pi
    while angle > math.pi:
        angle -= 2 * math.pi
    return angle


def _heading_to_target(pos_from: np.ndarray, pos_to: np.ndarray) -> float:
    dx = pos_to[0] - pos_from[0]
    dz = pos_to[2] - pos_from[2]
    return math.atan2(-dx, -dz) % (2 * math.pi)


@dataclass
class StateLabelBuilderConfig:
    state_bins: int = 10
    prefix_radius: float = 3.0
    local_window: int = 1
    deviation_threshold: float = 3.0
    off_path_threshold: float = 4.5
    heading_threshold_rad: float = math.pi / 3
    repeated_window: int = 6
    repeated_ratio_threshold: float = 0.5
    candidate_instability_window: int = 4
    candidate_instability_threshold: float = 0.75
    recovery_tolerance_steps: int = 1
    max_prefix_backtrack: int = 1
    min_valid_conditions: int = 3


@dataclass
class EpisodeStateContext:
    reference_path: np.ndarray
    reference_path_list: List[List[float]]
    cumulative_arc_lengths: np.ndarray
    prefix_history: List[int] = field(default_factory=list)
    node_history: List[Any] = field(default_factory=list)
    top_candidate_history: List[Any] = field(default_factory=list)

    @property
    def prev_prefix_idx(self) -> int:
        if not self.prefix_history:
            return 0
        return int(self.prefix_history[-1])


class StateLabelBuilder:
    """Reference-trajectory-centric unified state label builder."""

    def __init__(self, config: Optional[StateLabelBuilderConfig] = None) -> None:
        self.config = config or StateLabelBuilderConfig()

    @classmethod
    def from_config(cls, config_node: Any, state_bins: int = 10) -> "StateLabelBuilder":
        cfg = StateLabelBuilderConfig(
            state_bins=int(getattr(config_node, "state_bins", state_bins)),
            prefix_radius=float(getattr(config_node, "prefix_radius", 3.0)),
            local_window=int(getattr(config_node, "local_window", 1)),
            deviation_threshold=float(getattr(config_node, "deviation_threshold", 3.0)),
            off_path_threshold=float(getattr(config_node, "off_path_threshold", 4.5)),
            heading_threshold_rad=float(getattr(config_node, "heading_threshold_rad", math.pi / 3)),
            repeated_window=int(getattr(config_node, "repeated_window", 6)),
            repeated_ratio_threshold=float(getattr(config_node, "repeated_ratio_threshold", 0.5)),
            candidate_instability_window=int(getattr(config_node, "candidate_instability_window", 4)),
            candidate_instability_threshold=float(
                getattr(config_node, "candidate_instability_threshold", 0.75)
            ),
            recovery_tolerance_steps=int(getattr(config_node, "recovery_tolerance_steps", 1)),
            max_prefix_backtrack=int(getattr(config_node, "max_prefix_backtrack", 1)),
            min_valid_conditions=int(getattr(config_node, "min_valid_conditions", 3)),
        )
        return cls(cfg)

    def init_episode(self, reference_path: Sequence[Sequence[float]]) -> EpisodeStateContext:
        ref_path = np.asarray(reference_path, dtype=np.float32)
        if ref_path.ndim != 2 or ref_path.shape[0] == 0:
            raise ValueError("reference_path must be a non-empty [T, 3] array-like value")

        deltas = ref_path[1:] - ref_path[:-1]
        seg_lengths = np.linalg.norm(deltas, axis=1)
        cumulative = np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(seg_lengths, axis=0)])
        return EpisodeStateContext(
            reference_path=ref_path,
            reference_path_list=ref_path.tolist(),
            cumulative_arc_lengths=cumulative,
        )

    def _match_reference_prefix(
        self,
        ref_distances: np.ndarray,
        prev_prefix_idx: int,
    ) -> tuple[int, bool]:
        min_allowed = max(prev_prefix_idx - self.config.max_prefix_backtrack, 0)
        max_allowed = min(
            prev_prefix_idx + self.config.local_window + 1,
            len(ref_distances) - 1,
        )
        local_indices = np.arange(min_allowed, max_allowed + 1)
        local_valid = local_indices[ref_distances[local_indices] <= self.config.prefix_radius]
        if len(local_valid) == 0:
            return prev_prefix_idx, False

        streaks: List[tuple[int, int]] = []
        streak_start = int(local_valid[0])
        streak_end = int(local_valid[0])
        for idx in local_valid[1:]:
            idx = int(idx)
            if idx == streak_end + 1:
                streak_end = idx
            else:
                streaks.append((streak_start, streak_end))
                streak_start = idx
                streak_end = idx
        streaks.append((streak_start, streak_end))

        def continuity_gap(streak: tuple[int, int]) -> int:
            start_idx, end_idx = streak
            if start_idx <= prev_prefix_idx <= end_idx:
                return 0
            if end_idx < prev_prefix_idx:
                return prev_prefix_idx - end_idx
            return start_idx - prev_prefix_idx

        best_start, best_end = min(
            streaks,
            key=lambda streak: (
                continuity_gap(streak),
                -(streak[1] - streak[0] + 1),
                -streak[1],
            ),
        )
        return int(best_end), True

    def _progress_ratio(self, context: EpisodeStateContext, prefix_idx: int) -> float:
        total_length = float(context.cumulative_arc_lengths[-1])
        if total_length <= 1e-6:
            return 1.0
        return float(context.cumulative_arc_lengths[prefix_idx] / total_length)

    def _state_bin(self, ratio: float) -> int:
        if self.config.state_bins <= 1:
            return 0
        return min(int(math.floor(ratio * self.config.state_bins)), self.config.state_bins - 1)

    def _local_window_slice(self, prefix_idx: int, length: int) -> slice:
        local_window = max(int(self.config.local_window), 0)
        start = max(prefix_idx - local_window, 0)
        stop = min(prefix_idx + local_window + 1, length)
        return slice(start, stop)

    def build_step(
        self,
        context: EpisodeStateContext,
        current_position: Sequence[float],
        current_heading: Optional[float] = None,
        node_id: Optional[Any] = None,
        top_candidate_id: Optional[Any] = None,
        ref_distances: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        current_pos = np.asarray(current_position, dtype=np.float32)
        if ref_distances is None:
            ref_distances_np = np.linalg.norm(context.reference_path - current_pos[None, :], axis=1)
        else:
            ref_distances_np = np.asarray(ref_distances, dtype=np.float32)

        prefix_idx, near_prefix = self._match_reference_prefix(
            ref_distances=ref_distances_np,
            prev_prefix_idx=context.prev_prefix_idx,
        )
        progress_ratio = self._progress_ratio(context, prefix_idx)
        state_bin = self._state_bin(progress_ratio)

        local_slice = self._local_window_slice(prefix_idx, len(ref_distances_np))
        local_ref_distances = ref_distances_np[local_slice]
        if local_ref_distances.size == 0:
            local_ref_distances = ref_distances_np[prefix_idx : prefix_idx + 1]
        local_deviation = float(np.min(local_ref_distances))
        global_nearest_ref_deviation = float(np.min(ref_distances_np))

        heading_ok = True
        heading_error = 0.0
        if current_heading is not None and prefix_idx < len(context.reference_path) - 1:
            target_heading = _heading_to_target(current_pos, context.reference_path[prefix_idx + 1])
            heading_error = abs(_wrap_angle((current_heading % (2 * math.pi)) - target_heading))
            heading_ok = heading_error <= self.config.heading_threshold_rad

        repeated_ratio = 0.0
        if node_id is not None:
            window_nodes = (context.node_history + [node_id])[-self.config.repeated_window :]
            if len(window_nodes) > 1:
                repeated_ratio = 1.0 - (len(set(window_nodes)) / float(len(window_nodes)))

        oscillation_detected = False
        if node_id is not None and len(context.node_history) >= 2:
            oscillation_detected = context.node_history[-2] == node_id and context.node_history[-1] != node_id

        candidate_instability = 0.0
        if top_candidate_id is not None:
            recent = (context.top_candidate_history + [top_candidate_id])[-self.config.candidate_instability_window :]
            if len(recent) > 1:
                changes = sum(int(a != b) for a, b in zip(recent[:-1], recent[1:]))
                candidate_instability = changes / float(len(recent) - 1)

        max_prefix_so_far = max(context.prefix_history) if context.prefix_history else prefix_idx
        recovery_conflict = prefix_idx + self.config.recovery_tolerance_steps < max_prefix_so_far

        conditions = [
            near_prefix and local_deviation <= self.config.deviation_threshold,
            heading_ok,
            local_deviation <= self.config.off_path_threshold,
            repeated_ratio <= self.config.repeated_ratio_threshold and not oscillation_detected,
            candidate_instability <= self.config.candidate_instability_threshold and not recovery_conflict,
        ]
        valid_count = sum(int(cond) for cond in conditions)
        state_valid_mask = int(
            near_prefix
            and local_deviation <= self.config.off_path_threshold
            and valid_count >= self.config.min_valid_conditions
        )

        context.prefix_history.append(prefix_idx)
        if node_id is not None:
            context.node_history.append(node_id)
        if top_candidate_id is not None:
            context.top_candidate_history.append(top_candidate_id)

        return {
            "state_bin": state_bin,
            "state_valid_mask": state_valid_mask,
            "prefix_idx": prefix_idx,
            "progress_ratio": progress_ratio,
            "metrics": {
                "prefix_deviation": local_deviation,
                "local_prefix_deviation": local_deviation,
                "nearest_ref_deviation": global_nearest_ref_deviation,
                "heading_error": heading_error,
                "repeated_ratio": repeated_ratio,
                "candidate_instability": candidate_instability,
                "recovery_conflict": int(recovery_conflict),
                "local_window_start": int(local_slice.start),
                "local_window_end": int(local_slice.stop - 1),
            },
        }

    def build_sequence(
        self,
        reference_path: Sequence[Sequence[float]],
        rollout_positions: Sequence[Sequence[float]],
        rollout_headings: Optional[Sequence[float]] = None,
        node_ids: Optional[Sequence[Any]] = None,
        top_candidate_ids: Optional[Sequence[Any]] = None,
        ref_distances_seq: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[Dict[str, Any]]:
        context = self.init_episode(reference_path)
        outputs: List[Dict[str, Any]] = []
        for step_idx, position in enumerate(rollout_positions):
            outputs.append(
                self.build_step(
                    context=context,
                    current_position=position,
                    current_heading=None if rollout_headings is None else rollout_headings[step_idx],
                    node_id=None if node_ids is None else node_ids[step_idx],
                    top_candidate_id=None if top_candidate_ids is None else top_candidate_ids[step_idx],
                    ref_distances=None if ref_distances_seq is None else ref_distances_seq[step_idx],
                )
            )
        return outputs
