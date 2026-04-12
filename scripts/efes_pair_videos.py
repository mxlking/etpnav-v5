#!/usr/bin/env python3
"""Create side-by-side EFES comparison videos for selected episodes."""

import argparse
import glob
import os
from typing import Iterable, Optional

import cv2
import numpy as np


def _parse_episode_ids(raw: str) -> Iterable[str]:
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if item:
            yield item


def _find_video(video_dir: str, episode_id: str) -> Optional[str]:
    pattern = os.path.join(video_dir, f"*-{episode_id}-*.mp4")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    if frame.shape[0] == height:
        return frame
    width = max(int(round(frame.shape[1] * height / float(frame.shape[0]))), 1)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    bar_h = 36
    bar = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        label,
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([bar, frame], axis=0)


def _next_frame(cap: cv2.VideoCapture, last: Optional[np.ndarray]) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if ok:
        return frame
    return last


def pair_episode(
    left_path: str,
    right_path: str,
    out_path: str,
    left_label: str,
    right_label: str,
    fps: float,
) -> None:
    left_cap = cv2.VideoCapture(left_path)
    right_cap = cv2.VideoCapture(right_path)
    if not left_cap.isOpened() or not right_cap.isOpened():
        raise RuntimeError(f"Unable to open video pair: {left_path}, {right_path}")

    left_total = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    right_total = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total = max(left_total, right_total)
    left_fps = float(left_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    right_fps = float(right_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    out_fps = fps if fps > 0 else (left_fps or right_fps or 8.0)

    writer = None
    last_left = None
    last_right = None
    try:
        for _ in range(total):
            last_left = _next_frame(left_cap, last_left)
            last_right = _next_frame(right_cap, last_right)
            if last_left is None or last_right is None:
                continue
            height = max(last_left.shape[0], last_right.shape[0])
            left = _resize_to_height(last_left, height)
            right = _resize_to_height(last_right, height)
            left = _label_frame(left, left_label)
            right = _label_frame(right, right_label)
            paired = np.concatenate([left, right], axis=1)
            if writer is None:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    out_path,
                    fourcc,
                    out_fps,
                    (paired.shape[1], paired.shape[0]),
                )
            writer.write(paired)
    finally:
        left_cap.release()
        right_cap.release()
        if writer is not None:
            writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-dir", required=True)
    parser.add_argument("--right-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--episode-ids", required=True)
    parser.add_argument("--left-label", default="no_action")
    parser.add_argument("--right-label", default="active")
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()

    missing = []
    for episode_id in _parse_episode_ids(args.episode_ids):
        left = _find_video(args.left_dir, episode_id)
        right = _find_video(args.right_dir, episode_id)
        if left is None or right is None:
            missing.append((episode_id, left, right))
            continue
        out_path = os.path.join(args.out_dir, f"episode_{episode_id}_compare.mp4")
        pair_episode(left, right, out_path, args.left_label, args.right_label, args.fps)
        print(f"[paired] ep={episode_id} -> {out_path}")

    if missing:
        print("[missing]")
        for episode_id, left, right in missing:
            print(f"  ep={episode_id} left={left or 'NA'} right={right or 'NA'}")


if __name__ == "__main__":
    main()
