#!/usr/bin/env python3
"""Sanity-check FiLM-OSG .mat files before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OSG-format .mat train/test files.")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--problem-dim", choices=["1d", "2d"], required=True)
    parser.add_argument("--expected-channels", type=int, default=None)
    parser.add_argument("--expected-time", type=int, default=None)
    parser.add_argument("--require-positive-dt", action="store_true")
    return parser.parse_args()


def check_file(path: Path, problem_dim: str, expected_channels: int | None, expected_time: int | None, require_positive_dt: bool) -> None:
    data = loadmat(path)
    missing = [key for key in ("trajectories", "dt", "coordinates") if key not in data]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")

    trajectories = np.asarray(data["trajectories"])
    dt = np.asarray(data["dt"])
    coords = np.asarray(data["coordinates"])

    if problem_dim == "1d":
        if trajectories.ndim != 4:
            raise ValueError(f"{path}: expected 1D trajectories ndim=4 (N,L,D,T), got {trajectories.shape}")
        n, l, d, t = trajectories.shape
        if coords.shape not in {(l,), (l, 1), (1, l)}:
            raise ValueError(f"{path}: coordinates shape {coords.shape} does not match L={l}")
    else:
        if trajectories.ndim != 5:
            raise ValueError(f"{path}: expected 2D trajectories ndim=5 (N,H,W,D,T), got {trajectories.shape}")
        n, h, w, d, t = trajectories.shape
        if coords.shape != (h, w, 2):
            raise ValueError(f"{path}: coordinates shape {coords.shape} does not match (H,W,2)=({h},{w},2)")

    if expected_channels is not None and d != expected_channels:
        raise ValueError(f"{path}: expected D={expected_channels}, got D={d}")
    if expected_time is not None and t != expected_time:
        raise ValueError(f"{path}: expected T={expected_time}, got T={t}")
    if dt.shape[0] != n:
        raise ValueError(f"{path}: dt first dimension {dt.shape[0]} does not match N={n}")
    if expected_time is not None and expected_time > 1 and dt.shape[-1] not in {1, expected_time - 1}:
        raise ValueError(f"{path}: dt shape {dt.shape} is incompatible with T={t}")

    for name, arr in (("trajectories", trajectories), ("dt", dt), ("coordinates", coords)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{path}: {name} contains NaN or inf")
    if require_positive_dt and not np.all(dt > 0):
        raise ValueError(f"{path}: dt must be strictly positive, got range [{dt.min()}, {dt.max()}]")

    print(f"{path}")
    print(f"  trajectories shape={trajectories.shape}, dtype={trajectories.dtype}, range=({trajectories.min():.6g}, {trajectories.max():.6g})")
    print(f"  dt shape={dt.shape}, dtype={dt.dtype}, range=({dt.min():.6g}, {dt.max():.6g})")
    print(f"  coordinates shape={coords.shape}, dtype={coords.dtype}, range=({coords.min():.6g}, {coords.max():.6g})")


def main() -> None:
    args = parse_args()
    check_file(args.train, args.problem_dim, args.expected_channels, args.expected_time, args.require_positive_dt)
    check_file(args.test, args.problem_dim, args.expected_channels, args.expected_time, args.require_positive_dt)
    print("OSG .mat sanity check passed.")


if __name__ == "__main__":
    main()
