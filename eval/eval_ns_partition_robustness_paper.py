"""Evaluate Navier--Stokes rollouts over several partitions of the same time."""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA_DIR = REPO_ROOT / "data"
TEST_FILE = "VorticityOSG_test.mat"
DEFAULT_HORIZONS = (20, 40, 60, 80)
DEFAULT_PARTITION_SEED = 20260514


def parse_int_list(text):
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def parse_str_list(text):
    return [value.strip() for value in text.split(",") if value.strip()]


def safe_torch_load(model_path, device):
    import torch
    return torch.load(model_path, map_location=device, weights_only=False)


def model_path_for(model_name, seed, tag, root="."):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(root, f"runs_ns_{model_name}_seed{seed}{suffix}", "model")


def make_alternating_partition(terminal_time, first):
    """Return alternating 0.5/1.5 steps whose sum is terminal_time."""
    terminal_time = int(terminal_time)
    if first == 0.5:
        pair = (0.5, 1.5)
    elif first == 1.5:
        pair = (1.5, 0.5)
    else:
        raise ValueError("first must be 0.5 or 1.5")

    steps = list(pair) * (terminal_time // 2)
    if terminal_time % 2:
        steps.append(1.0)
    return np.asarray(steps, dtype=np.float32)


def make_random_pair_partition(terminal_time, rng):
    """Return fixed random pairs [a, 2-a] with both entries in [0.5, 1.5]."""
    terminal_time = int(terminal_time)
    steps = []
    for _ in range(terminal_time // 2):
        first = float(rng.uniform(0.5, 1.5))
        steps.extend((first, 2.0 - first))
    if terminal_time % 2:
        steps.append(1.0)
    return np.asarray(steps, dtype=np.float32)


def validate_partition(name, steps, terminal_time):
    total = float(np.sum(steps))
    if abs(total - float(terminal_time)) > 1e-5:
        raise RuntimeError(f"{name}: sum={total}, expected {terminal_time}")
    if float(np.min(steps)) < 0.5 - 1e-6 or float(np.max(steps)) > 1.5 + 1e-6:
        raise RuntimeError(f"{name}: time intervals fall outside [0.5, 1.5]")


def make_partitions(terminal_time, num_random=8, partition_seed=DEFAULT_PARTITION_SEED):
    """Construct the shared manuscript partition set for one terminal time."""
    terminal_time = int(terminal_time)
    rng = np.random.default_rng(int(partition_seed) + terminal_time)
    partitions = {
        "uniform_1p0": np.ones(terminal_time, dtype=np.float32),
        "fine_0p5": np.full(2 * terminal_time, 0.5, dtype=np.float32),
        "alt_0p5_1p5": make_alternating_partition(terminal_time, first=0.5),
        "alt_1p5_0p5": make_alternating_partition(terminal_time, first=1.5),
    }
    for index in range(int(num_random)):
        partitions[f"random_pair_{index:02d}"] = make_random_pair_partition(
            terminal_time, rng
        )
    for name, steps in partitions.items():
        validate_partition(name, steps, terminal_time)
    return partitions


def batch_rel_l2(first, second, denominator_reference=None, eps=1e-12):
    """Return per-sample relative L2 with an optional shared denominator."""
    first_flat = np.asarray(first).reshape(first.shape[0], -1)
    second_flat = np.asarray(second).reshape(second.shape[0], -1)
    reference = second if denominator_reference is None else denominator_reference
    reference_flat = np.asarray(reference).reshape(reference.shape[0], -1)
    numerator = np.linalg.norm(first_flat - second_flat, axis=1)
    denominator = np.linalg.norm(reference_flat, axis=1)
    return numerator / (denominator + eps)


def predict_terminal(model, initial_state, steps, device):
    import torch

    dt = np.tile(steps[None, :], (initial_state.shape[0], 1)).astype(np.float32)
    with torch.no_grad():
        prediction = model.predict(initial_state, dt, device)
    return np.asarray(prediction[..., -1], dtype=np.float32)


def evaluate_horizon(model, initial_state, truth_terminal, partitions, device):
    predictions = {}
    terminal_errors = {}
    partition_rows = []

    for name, steps in partitions.items():
        terminal = predict_terminal(model, initial_state, steps, device)
        errors = batch_rel_l2(terminal, truth_terminal)
        predictions[name] = terminal
        terminal_errors[name] = errors
        partition_rows.append(
            {
                "partition": name,
                "num_steps": int(len(steps)),
                "min_dt": float(np.min(steps)),
                "max_dt": float(np.max(steps)),
                "terminal_rel_l2_mean": float(np.mean(errors)),
                "terminal_rel_l2_median": float(np.median(errors)),
                "terminal_rel_l2_std": float(np.std(errors, ddof=0)),
            }
        )

    names = list(partitions)
    pairwise = []
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            pairwise.append(
                batch_rel_l2(
                    predictions[first_name],
                    predictions[second_name],
                    denominator_reference=truth_terminal,
                )
            )
    pairwise = np.stack(pairwise, axis=0)
    pairwise_per_sample = np.mean(pairwise, axis=0)

    uniform_prediction = predictions["uniform_1p0"]
    to_uniform = [
        batch_rel_l2(
            predictions[name],
            uniform_prediction,
            denominator_reference=truth_terminal,
        )
        for name in names
        if name != "uniform_1p0"
    ]
    to_uniform_per_sample = np.mean(np.stack(to_uniform, axis=0), axis=0)
    error_matrix = np.stack([terminal_errors[name] for name in names], axis=0)

    metrics = {
        "num_partitions": int(len(names)),
        "num_unordered_partition_pairs": int(len(names) * (len(names) - 1) // 2),
        "avg_terminal_rel_l2_over_partitions": float(np.mean(error_matrix)),
        "median_terminal_rel_l2_over_partitions": float(
            np.median(np.mean(error_matrix, axis=0))
        ),
        "std_terminal_rel_l2_across_partitions_mean": float(
            np.mean(np.std(error_matrix, axis=0, ddof=0))
        ),
        "pairwise_partition_spread_mean": float(np.mean(pairwise_per_sample)),
        "pairwise_partition_spread_median": float(np.median(pairwise_per_sample)),
        "to_uniform_partition_spread_mean": float(np.mean(to_uniform_per_sample)),
        "uniform_terminal_rel_l2_mean": float(
            np.mean(terminal_errors["uniform_1p0"])
        ),
        "fine_terminal_rel_l2_mean": float(np.mean(terminal_errors["fine_0p5"])),
        "alt_0p5_1p5_terminal_rel_l2_mean": float(
            np.mean(terminal_errors["alt_0p5_1p5"])
        ),
        "alt_1p5_0p5_terminal_rel_l2_mean": float(
            np.mean(terminal_errors["alt_1p5_0p5"])
        ),
    }
    return metrics, partition_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(seedwise_rows):
    summaries = []
    models = sorted({row["model"] for row in seedwise_rows})
    horizons = sorted({int(row["T"]) for row in seedwise_rows})
    excluded = {"model", "seed", "tag", "T"}
    for model_name in models:
        for terminal_time in horizons:
            rows = [
                row
                for row in seedwise_rows
                if row["model"] == model_name and int(row["T"]) == terminal_time
            ]
            if not rows:
                continue
            summary = {
                "model": model_name,
                "T": terminal_time,
                "num_seeds": len(rows),
            }
            for metric in (key for key in rows[0] if key not in excluded):
                values = np.asarray([row[metric] for row in rows], dtype=np.float64)
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values, ddof=0))
                summary[f"{metric}_median"] = float(np.median(values))
            summaries.append(summary)
    return summaries


def summarize_pairs(seedwise_rows, baseline="fno", conditioned="fno_film"):
    lookup = {
        (row["model"], int(row["seed"]), int(row["T"])): row
        for row in seedwise_rows
    }
    horizons = sorted({int(row["T"]) for row in seedwise_rows})
    summaries = []
    for terminal_time in horizons:
        seeds = sorted(
            {seed for model, seed, horizon in lookup if model == baseline and horizon == terminal_time}
            & {seed for model, seed, horizon in lookup if model == conditioned and horizon == terminal_time}
        )
        if not seeds:
            continue

        base_error = np.asarray(
            [lookup[(baseline, seed, terminal_time)]["avg_terminal_rel_l2_over_partitions"] for seed in seeds]
        )
        cond_error = np.asarray(
            [lookup[(conditioned, seed, terminal_time)]["avg_terminal_rel_l2_over_partitions"] for seed in seeds]
        )
        base_spread = np.asarray(
            [lookup[(baseline, seed, terminal_time)]["pairwise_partition_spread_mean"] for seed in seeds]
        )
        cond_spread = np.asarray(
            [lookup[(conditioned, seed, terminal_time)]["pairwise_partition_spread_mean"] for seed in seeds]
        )
        spread_reduction = 100.0 * (base_spread - cond_spread) / base_spread
        summaries.append(
            {
                "T": terminal_time,
                "baseline": baseline,
                "conditioned": conditioned,
                "num_paired_seeds": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "baseline_terminal_rel_l2_median": float(np.median(base_error)),
                "conditioned_terminal_rel_l2_median": float(np.median(cond_error)),
                "conditioned_terminal_error_wins": int(np.sum(cond_error < base_error)),
                "partition_spread_reduction_percent_median": float(np.median(spread_reduction)),
            }
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the manuscript NS partition-robustness metric using a shared "
            "set of admissible partitions and ||U(T)|| as the spread denominator."
        )
    )
    parser.add_argument("--models", default="fno,fno_film")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--horizons", default="20,40,60,80")
    parser.add_argument("--num-random", type=int, default=8)
    parser.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED)
    parser.add_argument("--tag", default="")
    parser.add_argument("--model-root", default=".")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--save-dir", default="./eval_outputs_ns_partition_robustness_paper")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    horizons = parse_int_list(args.horizons)
    test_path = Path(args.data_dir) / TEST_FILE

    import torch
    from scipy.io import loadmat

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    test_data = loadmat(test_path)
    trajectories = np.asarray(test_data["trajectories"], dtype=np.float32)
    initial_state = trajectories[..., 0]
    if args.max_samples is not None:
        trajectories = trajectories[: args.max_samples]
        initial_state = initial_state[: args.max_samples]

    seedwise_rows = []
    partition_rows = []
    for model_name in models:
        for seed in seeds:
            path = model_path_for(model_name, seed, args.tag, root=args.model_root)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing model: {path}")

            model = safe_torch_load(path, args.device)
            model.eval()
            for terminal_time in horizons:
                if terminal_time >= trajectories.shape[-1]:
                    raise ValueError(
                        f"T={terminal_time} exceeds {trajectories.shape[-1]} stored snapshots"
                    )
                partitions = make_partitions(
                    terminal_time,
                    num_random=args.num_random,
                    partition_seed=args.partition_seed,
                )
                metrics, rows = evaluate_horizon(
                    model,
                    initial_state,
                    trajectories[..., terminal_time],
                    partitions,
                    args.device,
                )
                seedwise_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "tag": args.tag,
                        "T": terminal_time,
                        **metrics,
                    }
                )
                for row in rows:
                    partition_rows.append(
                        {
                            "model": model_name,
                            "seed": seed,
                            "tag": args.tag,
                            "T": terminal_time,
                            **row,
                        }
                    )
                print(
                    f"{model_name} seed={seed} T={terminal_time}: "
                    f"spread={metrics['pairwise_partition_spread_mean']:.6e}",
                    flush=True,
                )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    os.makedirs(args.save_dir, exist_ok=True)
    seedwise_path = os.path.join(args.save_dir, "partition_horizon_seedwise.csv")
    partition_path = os.path.join(args.save_dir, "partition_per_partition_seedwise.csv")
    summary_path = os.path.join(args.save_dir, "partition_summary_by_model_T.csv")
    paired_path = os.path.join(args.save_dir, "partition_paired_summary.csv")
    summaries = summarize(seedwise_rows)
    paired_summaries = summarize_pairs(seedwise_rows)
    write_csv(seedwise_path, seedwise_rows)
    write_csv(partition_path, partition_rows)
    write_csv(summary_path, summaries)
    write_csv(paired_path, paired_summaries)
    print("Saved:", seedwise_path, flush=True)
    print("Saved:", partition_path, flush=True)
    print("Saved:", summary_path, flush=True)
    print("Saved:", paired_path, flush=True)


if __name__ == "__main__":
    main()
