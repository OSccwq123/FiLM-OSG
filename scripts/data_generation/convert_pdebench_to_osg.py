#!/usr/bin/env python3
"""Form the two PDEBench datasets used in the FiLM-OSG experiments.

The source HDF5 file is expected to contain one numeric group per trajectory.
Each group contains ``data`` with shape (time, x, y, channels) and the arrays
``grid/t``, ``grid/x``, and ``grid/y``.  Train and test trajectories are split
before variable-lag pairs are sampled.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert grouped two-dimensional PDEBench trajectories to OSG pairs."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--train-trajectories", type=int, default=800)
    parser.add_argument("--test-trajectories", type=int, default=200)
    parser.add_argument("--train-pairs", type=int, default=5000)
    parser.add_argument("--test-pairs", type=int, default=1000)
    parser.add_argument("--min-lag-steps", type=int, default=1)
    parser.add_argument("--max-lag-steps", type=int, default=20)
    parser.add_argument("--space-stride", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--pair-seed", type=int, default=0)
    parser.add_argument("--data-key", default="data")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def numeric_groups(h5: h5py.File) -> list[str]:
    return sorted(key for key in h5 if key.isdigit() and isinstance(h5[key], h5py.Group))


def split_trajectories(
    group_keys: list[str], n_train: int, n_test: int, seed: int
) -> tuple[list[str], list[str]]:
    total = n_train + n_test
    if n_train < 1 or n_test < 1 or total > len(group_keys):
        raise ValueError(
            f"Requested {n_train} train and {n_test} test trajectories "
            f"from {len(group_keys)} available groups."
        )
    order = np.random.default_rng(seed).permutation(len(group_keys))[:total]
    selected = np.asarray(group_keys)[order].tolist()
    train_keys = selected[:n_train]
    test_keys = selected[n_train:]
    assert set(train_keys).isdisjoint(test_keys)
    return train_keys, test_keys


def read_frame(dataset: h5py.Dataset, index: int, stride: int) -> np.ndarray:
    frame = np.asarray(dataset[index])
    if frame.ndim == 2:
        frame = frame[:, :, None]
    if frame.ndim != 3:
        raise ValueError(
            f"Expected a PDEBench frame with shape (x,y,channels), got {frame.shape}."
        )
    return frame[::stride, ::stride, :]


def balanced_schedule(
    group_keys: list[str], n_pairs: int, rng: np.random.Generator
) -> list[str]:
    if n_pairs < len(group_keys):
        raise ValueError("The number of pairs must cover every selected trajectory.")
    repeats, remainder = divmod(n_pairs, len(group_keys))
    schedule = group_keys * repeats
    if remainder:
        schedule.extend(
            np.asarray(group_keys)[rng.permutation(len(group_keys))[:remainder]].tolist()
        )
    rng.shuffle(schedule)
    return schedule


def sample_pairs(
    h5: h5py.File,
    group_keys: list[str],
    n_pairs: int,
    min_lag: int,
    max_lag: int,
    stride: int,
    data_key: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    first = h5[group_keys[0]][data_key]
    n_time = first.shape[0]
    if not 1 <= min_lag <= max_lag < n_time:
        raise ValueError(f"Invalid lag-step range [{min_lag},{max_lag}] for T={n_time}.")

    sample = read_frame(first, 0, stride)
    trajectories = np.empty((n_pairs, *sample.shape, 2), dtype=sample.dtype)
    lags = np.empty((n_pairs, 1), dtype=np.float64)

    for pair_id, group_key in enumerate(balanced_schedule(group_keys, n_pairs, rng)):
        group = h5[group_key]
        data = group[data_key]
        times = np.asarray(group["grid/t"]).squeeze()
        if data.shape[0] != n_time or times.shape != (n_time,):
            raise ValueError(f"Inconsistent time axis in trajectory {group_key}.")

        lag_steps = int(rng.integers(min_lag, max_lag + 1))
        start = int(rng.integers(0, n_time - lag_steps))
        stop = start + lag_steps
        trajectories[pair_id, ..., 0] = read_frame(data, start, stride)
        trajectories[pair_id, ..., 1] = read_frame(data, stop, stride)
        lags[pair_id, 0] = times[stop] - times[start]

    if not np.all(lags > 0):
        raise ValueError("PDEBench time coordinates must be strictly increasing.")
    return trajectories, lags


def coordinates(group: h5py.Group, stride: int) -> np.ndarray:
    x = np.asarray(group["grid/x"]).squeeze()[::stride]
    y = np.asarray(group["grid/y"]).squeeze()[::stride]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack((xx, yy), axis=-1)


def write_split(
    path: Path,
    trajectories: np.ndarray,
    lags: np.ndarray,
    coords: np.ndarray,
    dtype: str,
) -> None:
    savemat(
        path,
        {
            "trajectories": trajectories.astype(dtype, copy=False),
            "dt": lags.astype(dtype, copy=False),
            "coordinates": coords.astype(dtype, copy=False),
        },
        do_compression=True,
    )


def main() -> None:
    args = parse_args()
    if args.space_stride < 1:
        raise ValueError("space-stride must be positive.")

    with h5py.File(args.input, "r") as h5:
        groups = numeric_groups(h5)
        train_groups, test_groups = split_trajectories(
            groups, args.train_trajectories, args.test_trajectories, args.split_seed
        )
        train, train_dt = sample_pairs(
            h5,
            train_groups,
            args.train_pairs,
            args.min_lag_steps,
            args.max_lag_steps,
            args.space_stride,
            args.data_key,
            np.random.default_rng(args.pair_seed),
        )
        test, test_dt = sample_pairs(
            h5,
            test_groups,
            args.test_pairs,
            args.min_lag_steps,
            args.max_lag_steps,
            args.space_stride,
            args.data_key,
            np.random.default_rng(args.pair_seed + 1),
        )
        coords = coordinates(h5[train_groups[0]], args.space_stride)

    if train.shape[1:3] != coords.shape[:2] or test.shape[1:3] != coords.shape[:2]:
        raise ValueError("Spatial coordinates do not match the converted fields.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.prefix}_train.mat"
    test_path = args.output_dir / f"{args.prefix}_test.mat"
    write_split(train_path, train, train_dt, coords, args.dtype)
    write_split(test_path, test, test_dt, coords, args.dtype)

    print(f"train: {train_path}, trajectories={train.shape}, dt={train_dt.min():.6g}..{train_dt.max():.6g}")
    print(f"test:  {test_path}, trajectories={test.shape}, dt={test_dt.min():.6g}..{test_dt.max():.6g}")


if __name__ == "__main__":
    main()
