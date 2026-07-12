#!/usr/bin/env python3
"""Convert fixed-time PDEBench HDF5 trajectories into FiLM-OSG .mat files.

The existing FiLM-OSG dataset loader expects MATLAB files with

    trajectories: 1D -> (N, L, D, T), 2D -> (N, H, W, D, T)
    dt:           (N, T - 1)
    coordinates:  1D -> (L, 1), 2D -> (H, W, 2)

For a variable-lag task from a fixed-time PDEBench trajectory, this script
samples pairs (u(t_i), u(t_j)) and stores each pair as a two-frame trajectory.
The downstream OSG loader can then consume the file without changing the
training code: the only adjacent transition has lag t_j - t_i.

For grouped PDEBench files, the current paper protocol samples the train and
test pair sets independently from the same trajectory pool. It is therefore a
pair-level held-out split, not a trajectory-disjoint generalization test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import savemat

h5py = None


DEFAULT_DATA_KEYS = (
    "tensor",
    "data",
    "u",
    "solution",
    "solutions",
    "density",
    "Vx",
)

DEFAULT_TIME_KEYS = (
    "t-coordinate",
    "t_coordinates",
    "t",
    "time",
    "times",
)

DEFAULT_X_KEYS = (
    "x-coordinate",
    "x_coordinates",
    "x",
    "grid/x",
)

DEFAULT_Y_KEYS = (
    "y-coordinate",
    "y_coordinates",
    "y",
    "grid/y",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample variable-lag FiLM-OSG train/test .mat files from a local "
            "PDEBench HDF5/H5 shard."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="PDEBench .h5/.hdf5 file.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for .mat outputs.")
    parser.add_argument("--prefix", default="PDEBenchOSG", help="Output prefix, e.g. PDEBenchSWEOSG.")
    parser.add_argument("--problem-dim", choices=["1d", "2d"], required=True)
    parser.add_argument("--data-key", default=None, help="HDF5 key for the solution tensor.")
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="Read PDEBench group-per-trajectory files such as 0000/data, 0001/data, ...",
    )
    parser.add_argument("--group-data-key", default="data", help="Dataset name inside each trajectory group.")
    parser.add_argument("--max-trajectories", type=int, default=None, help="Optional limit on grouped trajectories.")
    parser.add_argument(
        "--layout",
        default="auto",
        choices=[
            "auto",
            "NTL",
            "NLT",
            "NTLD",
            "NLDT",
            "NLTD",
            "NTHW",
            "NHWT",
            "NTHWD",
            "NHWTD",
            "NHWDT",
            "HWD",
            "DHW",
            "LD",
            "DL",
        ],
        help=(
            "Raw tensor layout. Use an explicit value after inspecting a real "
            "PDEBench file; auto covers common PDEBench layouts but cannot "
            "resolve every ambiguous HDF5 convention."
        ),
    )
    parser.add_argument("--time-key", default=None, help="HDF5 key for time coordinates.")
    parser.add_argument("--x-key", default=None, help="HDF5 key for x coordinates.")
    parser.add_argument("--y-key", default=None, help="HDF5 key for y coordinates, for 2D data.")
    parser.add_argument("--component", type=int, default=None, help="Optional component index to keep.")
    parser.add_argument("--train-pairs", type=int, default=5000)
    parser.add_argument("--test-pairs", type=int, default=1000)
    parser.add_argument("--min-lag-steps", type=int, default=1)
    parser.add_argument("--max-lag-steps", type=int, default=None)
    parser.add_argument("--time-stride", type=int, default=1, help="Subsample input time axis before pairing.")
    parser.add_argument("--space-stride", type=int, default=1, help="Uniform spatial subsampling factor.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print detected shapes and planned output names without writing .mat files.",
    )
    return parser.parse_args()


def iter_datasets(h5: h5py.File) -> Iterable[tuple[str, h5py.Dataset]]:
    def visit(name: str, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))

    datasets: list[tuple[str, h5py.Dataset]] = []
    h5.visititems(visit)
    return datasets


def pick_key(h5: h5py.File, explicit: str | None, candidates: tuple[str, ...], *, min_ndim: int = 1) -> str:
    if explicit is not None:
        if explicit not in h5:
            raise KeyError(f"Requested key {explicit!r} not found in {list(h5.keys())}")
        return explicit
    for key in candidates:
        if key in h5 and getattr(h5[key], "ndim", 0) >= min_ndim:
            return key
    for key, dataset in iter_datasets(h5):
        if dataset.ndim >= min_ndim and not any(token in key.lower() for token in ("coord", "grid", "time")):
            return key
    raise KeyError(f"Could not infer a dataset key with ndim >= {min_ndim}.")


def read_vector(h5: h5py.File, explicit: str | None, candidates: tuple[str, ...], length: int, name: str) -> np.ndarray:
    key = explicit
    if key is None:
        for candidate in candidates:
            if candidate in h5:
                key = candidate
                break
    if key is None:
        return np.linspace(0.0, 1.0, length, dtype=np.float64)
    values = np.asarray(h5[key]).squeeze()
    if values.ndim != 1:
        raise ValueError(f"{name} coordinate {key!r} should be one-dimensional, got {values.shape}.")
    if values.shape[0] != length:
        raise ValueError(f"{name} coordinate length {values.shape[0]} does not match expected {length}.")
    return values.astype(np.float64)


def normalize_to_osg_layout(raw: np.ndarray, problem_dim: str, component: int | None, layout: str = "auto") -> np.ndarray:
    """Return data in 1D (N,L,D,T) or 2D (N,H,W,D,T) layout.

    PDEBench files have changed naming conventions over time. This heuristic
    accepts the common layouts where sample is the first axis and time is either
    the second or last axis. Ambiguous files should be inspected with --dry-run
    and converted with a small wrapper if needed.
    """
    arr = np.asarray(raw)
    if problem_dim == "1d":
        if layout == "NTL":
            arr = np.transpose(arr, (0, 2, 1))[:, :, None, :]
        elif layout == "NLT":
            arr = arr[:, :, None, :]
        elif layout == "NTLD":
            arr = np.transpose(arr, (0, 2, 3, 1))
        elif layout == "NLDT":
            pass
        elif layout == "NLTD":
            arr = np.transpose(arr, (0, 1, 3, 2))
        elif layout != "auto":
            raise ValueError(f"Layout {layout!r} is not valid for 1D data.")
        if layout != "auto":
            if component is not None:
                arr = arr[:, :, component : component + 1, :]
            return arr

        if arr.ndim == 3:
            # Prefer (N,T,L), common for PDEBench 1D files.
            n, a, b = arr.shape
            if a <= b:
                arr = np.transpose(arr, (0, 2, 1))[:, :, None, :]
            else:
                arr = arr[:, :, None, :]
        elif arr.ndim == 4:
            # (N,T,L,D) or (N,L,T,D) or already (N,L,D,T).
            if arr.shape[1] <= arr.shape[2]:
                arr = np.transpose(arr, (0, 2, 3, 1))
            elif arr.shape[-1] <= arr.shape[2]:
                arr = np.transpose(arr, (0, 1, 3, 2))
        else:
            raise ValueError(f"Unsupported 1D data shape {arr.shape}.")
        if component is not None:
            arr = arr[:, :, component : component + 1, :]
        return arr

    if layout == "NTHW":
        arr = np.transpose(arr, (0, 2, 3, 1))[:, :, :, None, :]
    elif layout == "NHWT":
        arr = arr[:, :, :, None, :]
    elif layout == "NTHWD":
        arr = np.transpose(arr, (0, 2, 3, 4, 1))
    elif layout == "NHWTD":
        arr = np.transpose(arr, (0, 1, 2, 4, 3))
    elif layout == "NHWDT":
        pass
    elif layout != "auto":
        raise ValueError(f"Layout {layout!r} is not valid for 2D data.")
    if layout != "auto":
        if component is not None:
            arr = arr[:, :, :, component : component + 1, :]
        return arr

    if arr.ndim == 4:
        # (N,T,H,W) or (N,H,W,T).
        if arr.shape[1] <= min(arr.shape[2], arr.shape[3]):
            arr = np.transpose(arr, (0, 2, 3, 1))[:, :, :, None, :]
        else:
            arr = arr[:, :, :, None, :]
    elif arr.ndim == 5:
        # (N,T,H,W,D), (N,H,W,T,D), or already (N,H,W,D,T).
        if arr.shape[1] <= min(arr.shape[2], arr.shape[3]):
            arr = np.transpose(arr, (0, 2, 3, 4, 1))
        elif arr.shape[-1] <= min(arr.shape[1], arr.shape[2]):
            arr = np.transpose(arr, (0, 1, 2, 4, 3))
    else:
        raise ValueError(f"Unsupported 2D data shape {arr.shape}.")
    if component is not None:
        arr = arr[:, :, :, component : component + 1, :]
    return arr


def downsample(data: np.ndarray, problem_dim: str, stride: int) -> np.ndarray:
    if stride <= 1:
        return data
    if problem_dim == "1d":
        return data[:, ::stride, :, :]
    return data[:, ::stride, ::stride, :, :]


def downsample_frame(frame: np.ndarray, problem_dim: str, stride: int) -> np.ndarray:
    if stride <= 1:
        return frame
    if problem_dim == "1d":
        return frame[::stride, :]
    return frame[::stride, ::stride, :]


def make_coordinates(h5: h5py.File, problem_dim: str, shape: tuple[int, ...], args: argparse.Namespace) -> np.ndarray:
    if problem_dim == "1d":
        length = shape[1]
        x = read_vector(h5, args.x_key, DEFAULT_X_KEYS, length * args.space_stride, "x")
        x = x[:: args.space_stride][:length]
        return x.reshape(length, 1)

    h, w = shape[1], shape[2]
    x = read_vector(h5, args.x_key, DEFAULT_X_KEYS, h * args.space_stride, "x")
    y = read_vector(h5, args.y_key, DEFAULT_Y_KEYS, w * args.space_stride, "y")
    x = x[:: args.space_stride][:h]
    y = y[:: args.space_stride][:w]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx, yy], axis=-1)


def sample_pairs(
    data: np.ndarray,
    times: np.ndarray,
    n_pairs: int,
    min_lag_steps: int,
    max_lag_steps: int,
    rng: np.random.Generator,
    problem_dim: str,
) -> tuple[np.ndarray, np.ndarray]:
    n_traj = data.shape[0]
    n_time = data.shape[-1]
    if n_time < 2:
        raise ValueError("Need at least two time instances.")
    max_lag_steps = min(max_lag_steps, n_time - 1)
    if min_lag_steps < 1 or min_lag_steps > max_lag_steps:
        raise ValueError(f"Invalid lag range [{min_lag_steps}, {max_lag_steps}] for T={n_time}.")

    if problem_dim == "1d":
        pairs = np.empty((n_pairs, data.shape[1], data.shape[2], 2), dtype=data.dtype)
    else:
        pairs = np.empty((n_pairs, data.shape[1], data.shape[2], data.shape[3], 2), dtype=data.dtype)
    dt = np.empty((n_pairs, 1), dtype=np.float64)

    for idx in range(n_pairs):
        traj = rng.integers(0, n_traj)
        lag = int(rng.integers(min_lag_steps, max_lag_steps + 1))
        start = int(rng.integers(0, n_time - lag))
        stop = start + lag
        pairs[..., 0][idx] = data[traj, ..., start]
        pairs[..., 1][idx] = data[traj, ..., stop]
        dt[idx, 0] = times[stop] - times[start]
    return pairs, dt


def numeric_group_keys(h5) -> list[str]:
    keys = [key for key in h5.keys() if key.isdigit() and hasattr(h5[key], "keys")]
    return sorted(keys)


def make_coordinates_grouped(h5, group_key: str, problem_dim: str, shape: tuple[int, ...], args: argparse.Namespace) -> np.ndarray:
    group = h5[group_key]
    if problem_dim == "1d":
        length = shape[0]
        x_key = args.x_key or "grid/x"
        if x_key in group:
            x = np.asarray(group[x_key]).squeeze().astype(np.float64)
        else:
            x = np.linspace(0.0, 1.0, length * args.space_stride, dtype=np.float64)
        return x[:: args.space_stride][:length].reshape(length, 1)

    h, w = shape[0], shape[1]
    x_key = args.x_key or "grid/x"
    y_key = args.y_key or "grid/y"
    if x_key in group:
        x = np.asarray(group[x_key]).squeeze().astype(np.float64)
    else:
        x = np.linspace(0.0, 1.0, h * args.space_stride, dtype=np.float64)
    if y_key in group:
        y = np.asarray(group[y_key]).squeeze().astype(np.float64)
    else:
        y = np.linspace(0.0, 1.0, w * args.space_stride, dtype=np.float64)
    x = x[:: args.space_stride][:h]
    y = y[:: args.space_stride][:w]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx, yy], axis=-1)


def normalize_group_frame(raw: np.ndarray, problem_dim: str, component: int | None, layout: str) -> np.ndarray:
    arr = np.asarray(raw)
    if problem_dim == "1d":
        if layout in {"auto", "LD"}:
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim != 2:
                raise ValueError(f"Unsupported grouped 1D frame shape {arr.shape}")
        elif layout == "DL":
            arr = np.transpose(arr, (1, 0))
        else:
            raise ValueError(f"Grouped 1D frame layout {layout!r} is not supported.")
        if component is not None:
            arr = arr[:, component : component + 1]
        return arr

    if layout in {"auto", "HWD"}:
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim != 3:
            raise ValueError(f"Unsupported grouped 2D frame shape {arr.shape}")
    elif layout == "DHW":
        arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f"Grouped 2D frame layout {layout!r} is not supported.")
    if component is not None:
        arr = arr[:, :, component : component + 1]
    return arr


def sample_pairs_grouped(h5, group_keys: list[str], args: argparse.Namespace, n_pairs: int, rng: np.random.Generator):
    first_group = h5[group_keys[0]]
    dataset = first_group[args.group_data_key]
    n_time = dataset.shape[0]
    max_lag_steps = min(args.max_lag_steps or (n_time - 1), n_time - 1)
    if args.min_lag_steps < 1 or args.min_lag_steps > max_lag_steps:
        raise ValueError(f"Invalid lag range [{args.min_lag_steps}, {max_lag_steps}] for T={n_time}.")

    first_frame = normalize_group_frame(dataset[0], args.problem_dim, args.component, "HWD" if args.layout == "auto" else args.layout)
    first_frame = downsample_frame(first_frame, args.problem_dim, args.space_stride)
    if args.problem_dim == "1d":
        pairs = np.empty((n_pairs, first_frame.shape[0], first_frame.shape[1], 2), dtype=first_frame.dtype)
    else:
        pairs = np.empty((n_pairs, first_frame.shape[0], first_frame.shape[1], first_frame.shape[2], 2), dtype=first_frame.dtype)
    dt = np.empty((n_pairs, 1), dtype=np.float64)

    for idx in range(n_pairs):
        group_key = group_keys[int(rng.integers(0, len(group_keys)))]
        group = h5[group_key]
        data = group[args.group_data_key]
        lag = int(rng.integers(args.min_lag_steps, max_lag_steps + 1))
        start = int(rng.integers(0, n_time - lag))
        stop = start + lag
        frame0 = normalize_group_frame(data[start], args.problem_dim, args.component, "HWD" if args.layout == "auto" else args.layout)
        frame1 = normalize_group_frame(data[stop], args.problem_dim, args.component, "HWD" if args.layout == "auto" else args.layout)
        frame0 = downsample_frame(frame0, args.problem_dim, args.space_stride)
        frame1 = downsample_frame(frame1, args.problem_dim, args.space_stride)
        pairs[..., 0][idx] = frame0
        pairs[..., 1][idx] = frame1
        t_key = args.time_key or "grid/t"
        if t_key in group:
            times = np.asarray(group[t_key]).squeeze().astype(np.float64)
            dt[idx, 0] = times[stop] - times[start]
        else:
            dt[idx, 0] = stop - start

    coords = make_coordinates_grouped(h5, group_keys[0], args.problem_dim, first_frame.shape, args)
    return pairs, dt, coords, n_time, max_lag_steps


def write_split(path: Path, trajectories: np.ndarray, dt: np.ndarray, coordinates: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        path,
        {
            "trajectories": trajectories.astype(dtype, copy=False),
            "dt": dt.astype(dtype, copy=False),
            "coordinates": coordinates.astype(dtype, copy=False),
        },
        do_compression=True,
    )


def main() -> None:
    args = parse_args()
    global h5py
    try:
        import h5py as _h5py
    except ImportError as exc:  # pragma: no cover - exercised only on lean envs.
        raise SystemExit(
            "convert_pdebench_to_osg.py requires h5py to read PDEBench HDF5 files. "
            "Install it in the active environment, e.g. `pip install h5py` or "
            "`conda install h5py`, then rerun this script."
        ) from exc
    h5py = _h5py
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.input, "r") as h5:
        group_keys = numeric_group_keys(h5)
        use_grouped = args.grouped or (args.data_key is None and group_keys and args.group_data_key in h5[group_keys[0]])
        if use_grouped:
            if args.max_trajectories is not None:
                group_keys = group_keys[: args.max_trajectories]
            if not group_keys:
                raise ValueError("No numeric trajectory groups found for --grouped conversion.")
            train_data, train_dt, coords, n_time, max_lag = sample_pairs_grouped(
                h5, group_keys, args, args.train_pairs, rng
            )
            test_data, test_dt, _, _, _ = sample_pairs_grouped(
                h5, group_keys, args, args.test_pairs, rng
            )
            data_key = f"<group>/{args.group_data_key}"
            metadata = {
                "input": str(args.input),
                "data_key": data_key,
                "grouped": True,
                "n_groups": len(group_keys),
                "layout": args.layout,
                "problem_dim": args.problem_dim,
                "component": args.component,
                "train_pairs": args.train_pairs,
                "test_pairs": args.test_pairs,
                "min_lag_steps": args.min_lag_steps,
                "max_lag_steps": max_lag,
                "time_stride": args.time_stride,
                "space_stride": args.space_stride,
                "seed": args.seed,
                "trajectories_shape_train": list(train_data.shape),
                "trajectories_shape_test": list(test_data.shape),
                "coordinates_shape": list(coords.shape),
                "dt_min": float(min(train_dt.min(), test_dt.min())),
                "dt_max": float(max(train_dt.max(), test_dt.max())),
            }
            train_path = args.output_dir / f"{args.prefix}_train.mat"
            test_path = args.output_dir / f"{args.prefix}_test.mat"
            print(json.dumps(metadata, indent=2))
            print(f"train -> {train_path}")
            print(f"test  -> {test_path}")
            if args.dry_run:
                return
            write_split(train_path, train_data, train_dt, coords, args.dtype)
            write_split(test_path, test_data, test_dt, coords, args.dtype)
            meta_path = args.output_dir / f"{args.prefix}_metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"meta  -> {meta_path}")
            return

        data_key = pick_key(h5, args.data_key, DEFAULT_DATA_KEYS, min_ndim=3)
        raw = np.asarray(h5[data_key])
        data = normalize_to_osg_layout(raw, args.problem_dim, args.component, args.layout)
        data = data[..., :: args.time_stride]
        data = downsample(data, args.problem_dim, args.space_stride)
        n_time = data.shape[-1]
        time_key = args.time_key
        if time_key is None:
            for candidate in DEFAULT_TIME_KEYS:
                if candidate in h5:
                    time_key = candidate
                    break
        if time_key is None:
            times = np.arange(n_time, dtype=np.float64)
        else:
            times = np.asarray(h5[time_key]).squeeze().astype(np.float64)
            times = times[:: args.time_stride][:n_time]
        coords = make_coordinates(h5, args.problem_dim, data.shape, args)

    max_lag = args.max_lag_steps or (n_time - 1)
    train_data, train_dt = sample_pairs(data, times, args.train_pairs, args.min_lag_steps, max_lag, rng, args.problem_dim)
    test_data, test_dt = sample_pairs(data, times, args.test_pairs, args.min_lag_steps, max_lag, rng, args.problem_dim)

    train_path = args.output_dir / f"{args.prefix}_train.mat"
    test_path = args.output_dir / f"{args.prefix}_test.mat"
    metadata = {
        "input": str(args.input),
        "data_key": data_key,
        "layout": args.layout,
        "problem_dim": args.problem_dim,
        "component": args.component,
        "train_pairs": args.train_pairs,
        "test_pairs": args.test_pairs,
        "min_lag_steps": args.min_lag_steps,
        "max_lag_steps": max_lag,
        "time_stride": args.time_stride,
        "space_stride": args.space_stride,
        "seed": args.seed,
        "trajectories_shape_train": list(train_data.shape),
        "trajectories_shape_test": list(test_data.shape),
        "coordinates_shape": list(coords.shape),
        "dt_min": float(min(train_dt.min(), test_dt.min())),
        "dt_max": float(max(train_dt.max(), test_dt.max())),
    }

    print(json.dumps(metadata, indent=2))
    print(f"train -> {train_path}")
    print(f"test  -> {test_path}")
    if args.dry_run:
        return
    write_split(train_path, train_data, train_dt, coords, args.dtype)
    write_split(test_path, test_data, test_dt, coords, args.dtype)
    meta_path = args.output_dir / f"{args.prefix}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"meta  -> {meta_path}")


if __name__ == "__main__":
    main()
