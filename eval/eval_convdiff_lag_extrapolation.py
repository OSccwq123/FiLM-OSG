import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from eval_convdiff_fno import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_PAIRS,
    DEFAULT_SEEDS,
    TRAIN_FILE,
    compute_metrics,
    flatten_summary_rows,
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


def resolve_test_files(data_dir: str | os.PathLike[str], test_files: str):
    data_root = Path(data_dir)

    if test_files:
        resolved = []
        for item in parse_str_list(test_files):
            path = Path(item)
            if not path.is_absolute():
                path = data_root / path
            resolved.append(path)
        return resolved

    return sorted(data_root.glob(DEFAULT_TEST_GLOB))


def lag_label_from_dt(dt):
    unique = np.unique(np.asarray(dt, dtype=np.float64))
    if unique.size == 1:
        return f"{float(unique[0]):.6g}"
    return "variable"


def write_rows_csv(path, rows):
    if not rows:
        return

    fieldnames = [
        "test_file",
        "lag",
        "model",
        "seed",
        "tag",
        "MAE",
        "Rel-L1",
        "Mean Rel-L2",
        "Final Rel-L2",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_one_model_on_test_file(
    model_name,
    seed,
    tag,
    model_root,
    train_data,
    test_data,
    test_file,
    device,
    eval_steps=None,
    save_mat=False,
    save_dir=DEFAULT_SAVE_DIR,
):
    path = model_path_for(model_name, seed, tag, root=model_root)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model: {path}")

    model = safe_torch_load(path, device)
    print("loaded_model_class =", f"{type(model).__module__}.{type(model).__name__}", flush=True)
    model.eval()

    x0 = test_data["trajectories"][..., 0].astype(np.float32)
    dt = test_data["dt"].astype(np.float32)
    truth = test_data["trajectories"].astype(np.float32)

    if eval_steps is not None:
        dt = dt[:, :eval_steps]
        truth = truth[..., :eval_steps + 1]

    coordinates = test_data.get("coordinates", train_data["coordinates"])

    with torch.no_grad():
        pred = model.predict(x0, dt, device)

    pred = np.asarray(pred, dtype=np.float32)
    metrics = compute_metrics(pred, truth)

    lag_label = lag_label_from_dt(dt)
    step_tag = dt.shape[1]

    print(
        f"{Path(test_file).name:30s} {model_name:10s} seed={seed:>3d}, "
        f"dt={lag_label:>8s}, steps={step_tag:>3d} -> "
        f"MAE={metrics['MAE']:.6e}, "
        f"Rel-L1={metrics['Rel-L1']:.6e}, "
        f"Mean Rel-L2={metrics['Mean Rel-L2']:.6e}, "
        f"Final Rel-L2={metrics['Final Rel-L2']:.6e}",
        flush=True,
    )

    if save_mat:
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"steps{step_tag}" if eval_steps is not None else "full"
        tag_part = f"_{tag}" if tag else ""
        test_stem = Path(test_file).stem
        out_path = os.path.join(
            save_dir,
            f"{test_stem}_{model_name}_seed{seed}{tag_part}_{suffix}_predictions.mat",
        )
        savemat(
            out_path,
            {
                "prediction": pred,
                "truth": truth,
                "dt": dt,
                "coordinates": coordinates,
                "metrics_MAE": np.array([[metrics["MAE"]]], dtype=np.float32),
                "metrics_RelL1": np.array([[metrics["Rel-L1"]]], dtype=np.float32),
                "metrics_MeanRelL2": np.array([[metrics["Mean Rel-L2"]]], dtype=np.float32),
                "metrics_FinalRelL2": np.array([[metrics["Final Rel-L2"]]], dtype=np.float32),
            },
        )
        print(f"Saved predictions to {out_path}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model-root", type=str, default=".")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--test-files",
        type=str,
        default="",
        help=(
            "Comma-separated fixed-lag .mat files. Relative paths are resolved "
            "under --data-dir. Default: data/test_data_fixed_dt_*.mat."
        ),
    )
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--save-mat", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    data_root = Path(args.data_dir)
    train_path = data_root / TRAIN_FILE
    test_files = resolve_test_files(args.data_dir, args.test_files)

    metric_keys = ["MAE", "Rel-L1", "Mean Rel-L2", "Final Rel-L2"]

    print("=" * 80, flush=True)
    print("Convection--diffusion fixed-lag extrapolation evaluation", flush=True)
    print("Models:", models, flush=True)
    print("Seeds:", seeds, flush=True)
    print("Tag:", args.tag if args.tag else "(none)", flush=True)
    print("Model root:", args.model_root, flush=True)
    print("Data dir:", args.data_dir, flush=True)
    print("Train file:", str(train_path), flush=True)
    print("Test files:", [str(p) for p in test_files], flush=True)
    print("Device:", args.device, flush=True)
    print("Eval steps:", args.eval_steps if args.eval_steps is not None else "full", flush=True)
    print("Save dir:", args.save_dir, flush=True)
    print("=" * 80, flush=True)

    if args.check_only or args.dry_run:
        print("Check-only mode: no .mat files or model weights will be loaded.", flush=True)
        print("Train data exists:", train_path.exists(), str(train_path), flush=True)
        if not test_files:
            print(f"No fixed-lag test files matched {data_root / DEFAULT_TEST_GLOB}", flush=True)
        for test_file in test_files:
            print("Test file exists:", test_file.exists(), str(test_file), flush=True)
        for model_name in models:
            for seed in seeds:
                path = model_path_for(model_name, seed, args.tag, root=args.model_root)
                exists = os.path.exists(path)
                print(f"Model path exists={exists}: {path}", flush=True)
                if not exists and not args.skip_missing:
                    raise FileNotFoundError(f"Missing model: {path}")
        return

    if not test_files:
        raise FileNotFoundError(f"No fixed-lag test files matched {data_root / DEFAULT_TEST_GLOB}")

    os.makedirs(args.save_dir, exist_ok=True)
    train_data = loadmat(train_path)

    all_rows = []
    summary_by_test = {}
    paired_by_test = {}
    missing = []

    for test_file in test_files:
        test_data = loadmat(test_file)
        lag_label = lag_label_from_dt(test_data["dt"])
        print("\n" + "-" * 80, flush=True)
        print("Test file:", str(test_file), flush=True)
        print("Lag label:", lag_label, flush=True)
        print("Test trajectories shape:", test_data["trajectories"].shape, flush=True)
        print("Test dt shape:", test_data["dt"].shape, flush=True)

        seedwise_rows = []
        for model_name in models:
            for seed in seeds:
                path = model_path_for(model_name, seed, args.tag, root=args.model_root)
                if not os.path.exists(path):
                    msg = f"Missing model: {path}"
                    if args.skip_missing:
                        print("[SKIP]", msg, flush=True)
                        missing.append(
                            {
                                "test_file": str(test_file),
                                "model": model_name,
                                "seed": seed,
                                "path": path,
                            }
                        )
                        continue
                    raise FileNotFoundError(msg)

                metrics = evaluate_one_model_on_test_file(
                    model_name=model_name,
                    seed=seed,
                    tag=args.tag,
                    model_root=args.model_root,
                    train_data=train_data,
                    test_data=test_data,
                    test_file=test_file,
                    device=args.device,
                    eval_steps=args.eval_steps,
                    save_mat=args.save_mat,
                    save_dir=args.save_dir,
                )

                row = {
                    "test_file": Path(test_file).name,
                    "lag": lag_label,
                    "model": model_name,
                    "seed": seed,
                    "tag": args.tag,
                }
                row.update(metrics)
                seedwise_rows.append(row)
                all_rows.append(row)

        summary_by_model = {}
        for model_name in models:
            rows = [r for r in seedwise_rows if r["model"] == model_name]
            if rows:
                summary_by_model[model_name] = {
                    "model": model_name,
                    "num_seeds": len(rows),
                    "seeds": [int(r["seed"]) for r in rows],
                    "metrics": summarize_metric_dicts(rows, metric_keys),
                }

        paired_rows, paired_summary = paired_comparison(
            seedwise_rows,
            DEFAULT_PAIRS,
            metric_keys,
        )

        test_key = Path(test_file).stem
        summary_by_test[test_key] = {
            "test_file": str(test_file),
            "lag": lag_label,
            "summary_by_model": summary_by_model,
        }
        paired_by_test[test_key] = {
            "test_file": str(test_file),
            "lag": lag_label,
            "paired_seedwise": paired_rows,
            "paired_summary": paired_summary,
        }

        write_rows_csv(
            os.path.join(args.save_dir, f"{test_key}_seedwise.csv"),
            seedwise_rows,
        )
        write_csv(
            os.path.join(args.save_dir, f"{test_key}_summary_by_model.csv"),
            flatten_summary_rows(summary_by_model, metric_keys),
        )
        write_csv(
            os.path.join(args.save_dir, f"{test_key}_paired_seedwise.csv"),
            paired_rows,
        )
        write_csv(
            os.path.join(args.save_dir, f"{test_key}_paired_summary.csv"),
            paired_summary,
        )

    write_rows_csv(
        os.path.join(args.save_dir, "convdiff_lag_extrapolation_seedwise.csv"),
        all_rows,
    )

    with open(os.path.join(args.save_dir, "convdiff_lag_extrapolation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": models,
                "seeds": seeds,
                "tag": args.tag,
                "eval_steps": args.eval_steps,
                "summary_by_test": summary_by_test,
                "paired_by_test": paired_by_test,
                "missing": missing,
            },
            f,
            indent=2,
        )

    print("\nSaved outputs under:", args.save_dir, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
