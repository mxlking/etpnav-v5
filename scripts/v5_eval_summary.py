#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


AGG_KEYS = [
    "success",
    "spl",
    "ndtw",
    "distance_to_goal",
    "path_length",
    "steps_taken",
    "oracle_success",
    "sdtw",
    "ghost_cnt",
    "collisions",
]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_mean(values: Iterable[Any]) -> Optional[float]:
    vals: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def _latest_json(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: (p.stat().st_mtime, p.name))[-1]


def _round_float(value: Optional[float], ndigits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)


def _load_episode_stats(exp_dir: Path) -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for path in sorted(exp_dir.glob("stats_ep_ckpt_*.json")):
        payload = _load_json(path)
        if isinstance(payload, dict):
            for episode_id, metrics in payload.items():
                if isinstance(metrics, dict):
                    merged[str(episode_id)] = metrics
    return merged


def _load_traces(exp_dir: Path) -> Dict[str, Dict[str, Any]]:
    trace_dir = exp_dir / "v5_traces"
    if not trace_dir.is_dir():
        return {}
    traces: Dict[str, Dict[str, Any]] = {}
    for path in sorted(trace_dir.glob("trace_ckpt_*.json")):
        payload = _load_json(path)
        episode_id = str(payload.get("episode_id", path.stem))
        traces[episode_id] = payload
    return traces


def _aggregate_from_episode_metrics(episode_metrics: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    metrics_list = list(episode_metrics)
    aggregated: Dict[str, float] = {}
    for key in AGG_KEYS:
        mean_value = _safe_mean(metric.get(key) for metric in metrics_list)
        if mean_value is not None:
            aggregated[key] = mean_value
    return aggregated


def _episode_stop_source(trace_payload: Dict[str, Any]) -> Optional[str]:
    steps = trace_payload.get("steps") or []
    if not steps:
        return None
    last_step = steps[-1]
    if int(last_step.get("selected_is_stop", 0)) != 1:
        return None
    level = int(last_step.get("intervention_level", 0))
    if level >= 2:
        decision = str(last_step.get("level3_decision", "unknown"))
        return f"level3:{decision}"
    if level == 1:
        return "level2:stop"
    return "level1_or_base:stop"


def _summarize_traces(traces: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not traces:
        return {}

    trace_payloads = list(traces.values())
    trace_metrics = [payload.get("metrics", {}) for payload in trace_payloads]
    steps = [step for payload in trace_payloads for step in payload.get("steps", [])]
    total_steps = len(steps)
    total_episodes = len(trace_payloads)

    one_step_stop = 0
    level2_episode_hits = 0
    level3_episode_hits = 0
    stop_sources: Counter[str] = Counter()

    for payload in trace_payloads:
        payload_steps = payload.get("steps", [])
        if payload_steps:
            if (
                int(payload.get("metrics", {}).get("steps_taken", 0)) == 1
                and int(payload_steps[0].get("selected_is_stop", 0)) == 1
            ):
                one_step_stop += 1
            levels = [int(step.get("intervention_level", 0)) for step in payload_steps]
            if any(level == 1 for level in levels):
                level2_episode_hits += 1
            if any(level >= 2 for level in levels):
                level3_episode_hits += 1
            stop_source = _episode_stop_source(payload)
            if stop_source is not None:
                stop_sources[stop_source] += 1

    level2_step_rate = None
    level3_step_rate = None
    if total_steps > 0:
        level2_step_rate = sum(
            1 for step in steps if int(step.get("intervention_level", 0)) == 1
        ) / float(total_steps)
        level3_step_rate = sum(
            1 for step in steps if int(step.get("intervention_level", 0)) >= 2
        ) / float(total_steps)

    return {
        "trace_episode_count": total_episodes,
        "trace_step_count": total_steps,
        "one_step_stop_ratio": (
            one_step_stop / float(total_episodes) if total_episodes > 0 else None
        ),
        "level2_step_rate": level2_step_rate,
        "level3_step_rate": level3_step_rate,
        "level2_episode_rate": (
            level2_episode_hits / float(total_episodes) if total_episodes > 0 else None
        ),
        "level3_episode_rate": (
            level3_episode_hits / float(total_episodes) if total_episodes > 0 else None
        ),
        "mean_intervention_level": _safe_mean(
            metric.get("mean_intervention_level") for metric in trace_metrics
        ),
        "stop_source_counts": dict(sorted(stop_sources.items())),
    }


def summarize_experiment(exp_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "exp_name": exp_dir.name,
        "exp_dir": str(exp_dir),
        "status": "ok",
    }

    if not exp_dir.exists():
        summary["status"] = "missing_exp_dir"
        return summary

    stats_ckpt_path = _latest_json(list(exp_dir.glob("stats_ckpt_*.json")))
    stats_ep = _load_episode_stats(exp_dir)
    traces = _load_traces(exp_dir)

    summary["artifacts"] = {
        "stats_ckpt": None if stats_ckpt_path is None else str(stats_ckpt_path),
        "stats_ep_files": len(list(exp_dir.glob("stats_ep_ckpt_*.json"))),
        "trace_files": len(traces),
    }

    aggregate_metrics: Dict[str, Any] = {}
    if stats_ckpt_path is not None:
        payload = _load_json(stats_ckpt_path)
        if isinstance(payload, dict):
            aggregate_metrics = {
                key: _round_float(payload.get(key))
                for key in AGG_KEYS
                if key in payload
            }

    if not aggregate_metrics:
        if stats_ep:
            aggregate_metrics = {
                key: _round_float(value)
                for key, value in _aggregate_from_episode_metrics(stats_ep.values()).items()
            }
        elif traces:
            trace_metrics = [payload.get("metrics", {}) for payload in traces.values()]
            aggregate_metrics = {
                key: _round_float(value)
                for key, value in _aggregate_from_episode_metrics(trace_metrics).items()
            }

    trace_summary = _summarize_traces(traces)

    if not aggregate_metrics and not trace_summary and not stats_ep:
        summary["status"] = "no_artifacts"

    episode_count = len(stats_ep) if stats_ep else len(traces)
    if episode_count == 0 and stats_ckpt_path is None:
        summary["status"] = "no_episode_level_data"

    one_step_stop_ratio = trace_summary.get("one_step_stop_ratio")
    if one_step_stop_ratio is None and stats_ep:
        estimated = [
            1.0
            for metrics in stats_ep.values()
            if int(metrics.get("steps_taken", 0)) == 1 and float(metrics.get("path_length", 0.0)) == 0.0
        ]
        if stats_ep:
            one_step_stop_ratio = len(estimated) / float(len(stats_ep))

    summary["summary"] = {
        "episode_count": int(episode_count),
        "success": aggregate_metrics.get("success"),
        "spl": aggregate_metrics.get("spl"),
        "ndtw": aggregate_metrics.get("ndtw"),
        "distance_to_goal": aggregate_metrics.get("distance_to_goal"),
        "avg_traj_length": aggregate_metrics.get("path_length"),
        "avg_steps_taken": aggregate_metrics.get("steps_taken"),
        "one_step_stop_ratio": _round_float(one_step_stop_ratio),
        "level2_step_rate": _round_float(trace_summary.get("level2_step_rate")),
        "level3_step_rate": _round_float(trace_summary.get("level3_step_rate")),
        "level2_episode_rate": _round_float(trace_summary.get("level2_episode_rate")),
        "level3_episode_rate": _round_float(trace_summary.get("level3_episode_rate")),
        "mean_intervention_level": _round_float(trace_summary.get("mean_intervention_level")),
        "stop_source_counts": trace_summary.get("stop_source_counts", {}),
    }

    per_exp_output = exp_dir / "v5_diagnostic_summary.json"
    try:
        with per_exp_output.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        summary["artifacts"]["diagnostic_summary"] = str(per_exp_output)
    except PermissionError:
        summary.setdefault("warnings", []).append(
            f"permission denied when writing {per_exp_output}"
        )
        summary["artifacts"]["diagnostic_summary"] = None
    return summary


def _print_table(summaries: List[Dict[str, Any]]) -> None:
    headers = [
        "exp_name",
        "status",
        "success",
        "spl",
        "ndtw",
        "d2g",
        "one_step_stop",
        "l2_step",
        "l3_step",
        "avg_path",
        "stop_sources",
    ]
    rows = [headers]
    for item in summaries:
        s = item.get("summary", {})
        rows.append(
            [
                str(item.get("exp_name", "")),
                str(item.get("status", "")),
                "" if s.get("success") is None else f"{float(s['success']):.4f}",
                "" if s.get("spl") is None else f"{float(s['spl']):.4f}",
                "" if s.get("ndtw") is None else f"{float(s['ndtw']):.4f}",
                "" if s.get("distance_to_goal") is None else f"{float(s['distance_to_goal']):.4f}",
                "" if s.get("one_step_stop_ratio") is None else f"{float(s['one_step_stop_ratio']):.4f}",
                "" if s.get("level2_step_rate") is None else f"{float(s['level2_step_rate']):.4f}",
                "" if s.get("level3_step_rate") is None else f"{float(s['level3_step_rate']):.4f}",
                "" if s.get("avg_traj_length") is None else f"{float(s['avg_traj_length']):.4f}",
                json.dumps(s.get("stop_source_counts", {}), ensure_ascii=False),
            ]
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    for row_idx, row in enumerate(rows):
        line = " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        print(line)
        if row_idx == 0:
            print("-+-".join("-" * widths[i] for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V5 eval outputs into a fixed diagnostic template.")
    parser.add_argument("exp_dirs", nargs="+", help="One or more eval result directories under data/logs/eval_results/")
    parser.add_argument("--output-json", dest="output_json", default="", help="Optional combined summary json path.")
    args = parser.parse_args()

    summaries = [summarize_experiment(Path(exp_dir)) for exp_dir in args.exp_dirs]
    _print_table(summaries)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
