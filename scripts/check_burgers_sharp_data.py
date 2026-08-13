#!/usr/bin/env python3
"""Check the generated steep-gradient Burgers data."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def total_variation(states: np.ndarray) -> np.ndarray:
    return np.abs(np.roll(states, -1, axis=1) - states).sum(axis=1)


def check(path: Path) -> None:
    data = loadmat(path)
    trajectories = np.asarray(data["trajectories"], dtype=np.float64)
    lags = np.asarray(data["dt"], dtype=np.float64)
    coords = np.asarray(data["coordinates"], dtype=np.float64)

    if trajectories.ndim != 4 or trajectories.shape[2] != 1:
        raise ValueError(f"{path}: expected trajectories with shape (N,L,1,T).")
    n_traj, n_grid, _, n_time = trajectories.shape
    if lags.shape != (n_traj, n_time - 1) or coords.shape != (n_grid, 1):
        raise ValueError(f"{path}: incompatible trajectory, lag, or coordinate shapes.")
    if not all(np.isfinite(array).all() for array in (trajectories, lags, coords)):
        raise ValueError(f"{path}: data contain NaN or inf.")
    if not np.all(lags > 0):
        raise ValueError(f"{path}: time lags must be positive.")

    states = trajectories[:, :, 0, :]
    mean_drift = np.abs(states.mean(axis=1) - states[:, :, 0].mean(axis=1)[:, None])
    initial_tv = total_variation(states[:, :, 0])
    final_tv = total_variation(states[:, :, -1])

    print(path)
    print(f"  trajectories: {trajectories.shape}")
    print(f"  lag range: {lags.min():.6g} .. {lags.max():.6g}")
    print(f"  value range: {states.min():.6g} .. {states.max():.6g}")
    print(f"  maximum mean drift: {mean_drift.max():.3e}")
    print(f"  mean total variation: {initial_tv.mean():.6g} -> {final_tv.mean():.6g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/burgers_sharp"))
    args = parser.parse_args()
    check(args.data_dir / "BurgersSharpOSG_train.mat")
    check(args.data_dir / "BurgersSharpOSG_test.mat")


if __name__ == "__main__":
    main()
