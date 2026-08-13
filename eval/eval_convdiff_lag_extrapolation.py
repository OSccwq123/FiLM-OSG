from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_convdiff_fno import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_PAIRS,
    DEFAULT_SEEDS,
    compute_metrics,
    model_path_for,
    paired_comparison,
    parse_int_list,
    parse_str_list,
    safe_torch_load,
    summarize_metric_dicts,
    write_csv,
)

DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_TEST_GLOB = "test_data_fixed_dt_*.mat"
DEFAULT_SAVE_DIR = "./eval_outputs_convdiff_lag_extrapolation"


def resolve_test_files(data_dir, test_files):
    data_root = Path(data_dir)
    if not test_files:
        return sorted(data_root.glob(DEFAULT_TEST_GLOB))
    paths = []
    for item in parse_str_list(test_files):
        path = Path(item)
        paths.append(path if path.is_absolute() else data_root / path)
    return paths


def lag_value(dt):
    unique = np.unique(np.asarray(dt, dtype=np.float64))
    if unique.size != 1:
        raise ValueError("Each fixed-lag test file must contain a single time lag.")
    return float(unique[0])


def evaluate_model(
    model_name,
    seed,
    tag,
    model_root,
    test_data,
    device,
    eval_steps=None,
    save_mat=False,
    save_dir=DEFAULT_SAVE_DIR,
    test_stem="fixed_lag",
):
    path = model_path_for(model_name, seed, tag, root=model_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model: {path}")

    model = safe_torch_load(path, device)
    model.eval()
    x0 = test_data["trajectories"][..., 0].astype(np.float32)
    dt = test_data["dt"].astype(np.float32)
    truth = test_data["trajectories"].astype(np.float32)
    if eval_steps is not None:
        dt = dt[:, :eval_steps]
        truth = truth[..., : eval_steps + 1]

    with torch.no_grad():
        pred = model.predict(x0, dt, device)
    pred = np.asarray(pred, dtype=np.float32)
    metrics = compute_metrics(pred, truth)

    print(
        f"{test_stem:28s} {model_name:10s} seed={seed}, steps={dt.shape[1]}: "
        f"Mean Rel-L2={metrics['Mean Rel-L2']:.6e}, "
        f"Final Rel-L2={metrics['Final Rel-L2']:.6e}",
        flush=True,
    )

    if save_mat:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tag_part = f"_{tag}" if tag else ""
        out_path = output_dir / f"{test_stem}_{model_name}_seed{seed}{tag_part}_predictions.mat"
        output = {
            "prediction": pred,
            "truth": truth,
            "dt": dt,
            "metrics_MeanRelL2": np.array([[metrics["Mean Rel-L2"]]], dtype=np.float32),
            "metrics_FinalRelL2": np.array([[metrics["Final Rel-L2"]]], dtype=np.float32),
        }
        if "coordinates" in test_data:
            output["coordinates"] = test_data["coordinates"]
        savemat(out_path, output)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed-lag advection--diffusion rollouts.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--tag", default="")
    parser.add_argument("--model-root", default=".")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--test-files",
        default="",
        help="Comma-separated files; defaults to test_data_fixed_dt_*.mat under --data-dir.",
    )
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-mat", action="store_true")
    args = parser.parse_args()

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    test_files = resolve_test_files(args.data_dir, args.test_files)
    if not test_files:
        raise FileNotFoundError(
            f"No fixed-lag test files matched {Path(args.data_dir) / DEFAULT_TEST_GLOB}"
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = ("Mean Rel-L2", "Final Rel-L2")
    all_seedwise = []
    all_summary = []
    all_paired = []

    for test_file in test_files:
        if not test_file.is_file():
            raise FileNotFoundError(f"Test data not found: {test_file}")
        test_data = loadmat(test_file)
        lag = lag_value(test_data["dt"])
        seedwise = []

        for model_name in models:
            for seed in seeds:
                metrics = evaluate_model(
                    model_name,
                    seed,
                    args.tag,
                    args.model_root,
                    test_data,
                    args.device,
                    eval_steps=args.eval_steps,
                    save_mat=args.save_mat,
                    save_dir=save_dir,
                    test_stem=test_file.stem,
                )
                row = {
                    "test_file": test_file.name,
                    "lag": lag,
                    "model": model_name,
                    "seed": seed,
                    "tag": args.tag,
                }
                row.update({key: metrics[key] for key in metric_keys})
                seedwise.append(row)
                all_seedwise.append(row)

        for model_name in models:
            rows = [row for row in seedwise if row["model"] == model_name]
            summary = summarize_metric_dicts(rows, metric_keys)
            out = {
                "test_file": test_file.name,
                "lag": lag,
                "model": model_name,
                "num_seeds": len(rows),
                "seeds": ",".join(str(row["seed"]) for row in rows),
            }
            for key in metric_keys:
                out[f"{key}_mean"] = summary[key]["mean"]
                out[f"{key}_std"] = summary[key]["std"]
            all_summary.append(out)

        for row in paired_comparison(seedwise, DEFAULT_PAIRS, metric_keys):
            all_paired.append({"test_file": test_file.name, "lag": lag, **row})

    write_csv(save_dir / "convdiff_fixed_lag_seedwise.csv", all_seedwise)
    write_csv(save_dir / "convdiff_fixed_lag_summary_by_model.csv", all_summary)
    write_csv(save_dir / "convdiff_fixed_lag_paired_summary.csv", all_paired)
    print(f"\nSaved summaries to {save_dir}")


if __name__ == "__main__":
    main()
