#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
from pathlib import Path

REPO_HINT = Path(__file__).resolve().parents[1]

BANNED_GLOBS = [
    "data/logs/*",
    "logs/*",
    "exports/*",
    "tmp_eval_ckpt_600/*",
    "__pycache__/*",
    "*.pyc",
    "*.pyo",
    "*.bak",
    "*.log",
    "*.out",
    "*.err",
    "*.pth",
    "*.pt",
    "*.ckpt",
    "*.onnx",
    "*.npy",
    "*.npz",
    "install.log",
    "get-pip.py",
]


def run_git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=REPO_HINT,
    )
    return proc.stdout


def staged_files() -> list[Path]:
    output = run_git(["diff", "--cached", "--name-only", "-z"])
    if not output:
        return []
    paths = [Path(p) for p in output.split("\0") if p]
    return paths


def matches_banned(path: str) -> bool:
    normalized = path.replace(os.sep, "/")
    for pattern in BANNED_GLOBS:
        if fnmatch.fnmatch(normalized, pattern):
            return True
    return False


def format_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.2f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check staged files before pushing to GitHub.")
    parser.add_argument("--warn-mb", type=float, default=10.0, help="Warn threshold in MB.")
    parser.add_argument("--hard-mb", type=float, default=50.0, help="Hard fail threshold in MB.")
    args = parser.parse_args()

    warn_bytes = int(args.warn_mb * 1024 * 1024)
    hard_bytes = int(args.hard_mb * 1024 * 1024)

    files = staged_files()
    if not files:
        print("No staged files.")
        return 0

    repo_root = Path(run_git(["rev-parse", "--show-toplevel"]).strip())
    ok: list[tuple[str, int]] = []
    warned: list[tuple[str, int]] = []
    blocked: list[tuple[str, int, str]] = []

    for rel_path in files:
        abs_path = repo_root / rel_path
        rel_str = rel_path.as_posix()

        if matches_banned(rel_str):
            size = abs_path.stat().st_size if abs_path.exists() and abs_path.is_file() else 0
            blocked.append((rel_str, size, "matches banned upload pattern"))
            continue

        if not abs_path.exists() or not abs_path.is_file():
            continue

        size = abs_path.stat().st_size
        if size >= hard_bytes:
            blocked.append((rel_str, size, f"exceeds hard limit {args.hard_mb:.0f} MB"))
        elif size >= warn_bytes:
            warned.append((rel_str, size))
        else:
            ok.append((rel_str, size))

    print("GitHub upload check")
    print(f"Repo: {repo_root}")
    print(f"Staged files: {len(files)}")
    print(f"Warn threshold: {args.warn_mb:.0f} MB")
    print(f"Hard threshold: {args.hard_mb:.0f} MB")

    if blocked:
        print("\nBLOCKED")
        for path, size, reason in blocked:
            print(f"- {path} [{format_mb(size)}] {reason}")

    if warned:
        print("\nWARN")
        for path, size in warned:
            print(f"- {path} [{format_mb(size)}]")

    if ok:
        print("\nOK")
        for path, size in ok[:40]:
            print(f"- {path} [{format_mb(size)}]")
        remaining = len(ok) - 40
        if remaining > 0:
            print(f"... and {remaining} more")

    if blocked:
        print("\nSuggested fix:")
        print("  git restore --staged <path>")
        return 1

    print("\nCheck passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
