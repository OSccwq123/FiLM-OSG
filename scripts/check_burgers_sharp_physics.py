#!/usr/bin/env python3
"""Check sharp-front Burgers physics diagnostics.

This script complements ``check_burgers_sharp_data.py`` by focusing on
maximum-principle-style overshoot/undershoot diagnostics and mean conservation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(description="Check sharp-front Burgers overshoot, undershoot, and mean drift.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/burgers_sharp"), help="Directory with BurgersSharpOSG_train/test.mat files.")
    parser.add_argument("--splits", type=str, default="train,test", help="Comma-separated splits to check, usually train,test.")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path. Defaults to <data-dir>/physics_check.json.")
    parser.add_argument("--tol", type=float, default=1e-3, help="Threshold used for counting trajectories with visible overshoot/undershoot.")
    return parser.parse_args()


def parse_splits(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def compute_split(path: Path, tol: float) -> dict[str, float | int | list[int] | bool]:
    data = loadmat(path)
    tr = data["trajectories"].astype(np.float64)[:, :, 0, :]
    dt = data["dt"].astype(np.float64)

    init = tr[:, :, 0]
    init_min = init.min(axis=1)[:, None]
    init_max = init.max(axis=1)[:, None]
    all_min = tr.min(axis=1)
    all_max = tr.max(axis=1)
    over = np.maximum(0.0, all_max - init_max)
    under = np.maximum(0.0, init_min - all_min)
    amp = (init_max - init_min).reshape(-1)
    means = tr.mean(axis=1)
    drift = np.abs(means - means[:, :1])

    prev_min = tr[:, :, :-1].min(axis=1)
    prev_max = tr[:, :, :-1].max(axis=1)
    next_min = tr[:, :, 1:].min(axis=1)
    next_max = tr[:, :, 1:].max(axis=1)
    step_over = np.maximum(0.0, next_max - prev_max)
    step_under = np.maximum(0.0, prev_min - next_min)

    return {
        "shape": list(tr.shape),
        "dt_shape": list(dt.shape),
        "finite": bool(np.isfinite(tr).all() and np.isfinite(dt).all()),
        "dt_min": float(dt.min()),
        "dt_max": float(dt.max()),
        "value_min": float(tr.min()),
        "value_max": float(tr.max()),
        "mean_drift_max": float(drift[:, 1:].max()),
        "mean_drift_avg": float(drift[:, 1:].mean()),
        "overshoot_vs_initial_max": float(over[:, 1:].max()),
        "undershoot_vs_initial_max": float(under[:, 1:].max()),
        "overshoot_vs_initial_mean": float(over[:, 1:].mean()),
        "undershoot_vs_initial_mean": float(under[:, 1:].mean()),
        "overshoot_over_amp_max": float((over[:, 1:].max(axis=1) / (amp + 1e-12)).max()),
        "undershoot_over_amp_max": float((under[:, 1:].max(axis=1) / (amp + 1e-12)).max()),
        "num_traj_any_overshoot_gt_tol": int((over[:, 1:].max(axis=1) > tol).sum()),
        "num_traj_any_undershoot_gt_tol": int((under[:, 1:].max(axis=1) > tol).sum()),
        "step_overshoot_max": float(step_over.max()),
        "step_undershoot_max": float(step_under.max()),
        "amp_mean": float(amp.mean()),
        "amp_lt_0p3_frac": float((amp < 0.3).mean()),
        "amp_gt_0p8_frac": float((amp > 0.8).mean()),
    }


def main():
    args = parse_args()
    root = args.data_dir
    summary = {}
    for split in parse_splits(args.splits):
        path = root / f"BurgersSharpOSG_{split}.mat"
        if not path.exists():
            print("MISSING", path)
            continue
        out = compute_split(path, args.tol)
        summary[split] = out
        print("\n" + split)
        for key, value in out.items():
            print(f"  {key}: {value}")

    out_path = args.out or (root / "physics_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSAVED", out_path)


if __name__ == "__main__":
    main()
