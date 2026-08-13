from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.diagnostics import mean_drift_metrics


DEFAULT_DATA_DIR = REPO_ROOT / "data"
TRAIN_FILE = "VorticityOSG_train.mat"
TEST_FILE = "VorticityOSG_test.mat"
DEFAULT_MODELS = ["fno", "fno_film"]
DEFAULT_PAIRS = [("fno", "fno_film")]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_TAG = ""
DEFAULT_SAVE_DIR = "./eval_outputs_ns_fno"


def resolve_data_paths(data_dir: str | os.PathLike[str]):
    data_root = Path(data_dir)
    return data_root / TRAIN_FILE, data_root / TEST_FILE


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def safe_torch_load(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)


def model_path_for(model_name, seed, tag, root="."):
    suffix = f"_{tag}" if tag else ""
    return Path(root) / f"runs_ns_{model_name}_seed{seed}{suffix}" / "model"


def compute_metrics(pred, truth, eps=1e-12):
    pred_roll = pred[..., 1:]
    true_roll = truth[..., 1:]
    rel_l1 = []
    rel_l2 = []
    final_rel_l2 = []

    for sample in range(pred_roll.shape[0]):
        for step in range(pred_roll.shape[-1]):
            p = pred_roll[sample, ..., step].reshape(-1)
            y = true_roll[sample, ..., step].reshape(-1)
            rel_l1.append(np.linalg.norm(p - y, 1) / (np.linalg.norm(y, 1) + eps))
            rel_l2.append(np.linalg.norm(p - y, 2) / (np.linalg.norm(y, 2) + eps))

        p_final = pred_roll[sample, ..., -1].reshape(-1)
        y_final = true_roll[sample, ..., -1].reshape(-1)
        final_rel_l2.append(
            np.linalg.norm(p_final - y_final, 2) / (np.linalg.norm(y_final, 2) + eps)
        )

    return {
        "MAE": float(np.abs(pred_roll - true_roll).mean()),
        "Rel-L1": float(np.mean(rel_l1)),
        "Mean Rel-L2": float(np.mean(rel_l2)),
        "Final Rel-L2": float(np.mean(final_rel_l2)),
    }


def summarize_metric_dicts(rows, metric_keys):
    summary = {}
    for key in metric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(np.median(values)),
        }
    return summary


def evaluate_one_model(
    model_name,
    seed,
    tag,
    model_root,
    test_data,
    train_data,
    device,
    eval_steps=None,
    save_mat=False,
    save_dir=DEFAULT_SAVE_DIR,
    mean_drift=False,
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
    if mean_drift:
        metrics.update(mean_drift_metrics(pred, truth))

    print(
        f"{model_name:10s} seed={seed}, steps={dt.shape[1]}: "
        f"Mean Rel-L2={metrics['Mean Rel-L2']:.6e}, "
        f"Final Rel-L2={metrics['Final Rel-L2']:.6e}",
        flush=True,
    )

    if save_mat:
        output_dir = Path(save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        step_tag = f"steps{dt.shape[1]}" if eval_steps is not None else "full"
        tag_part = f"_{tag}" if tag else ""
        out_path = output_dir / f"{model_name}_seed{seed}{tag_part}_{step_tag}_predictions.mat"
        output = {
            "prediction": pred,
            "truth": truth,
            "dt": dt,
            "metrics_MeanRelL2": np.array([[metrics["Mean Rel-L2"]]], dtype=np.float32),
            "metrics_FinalRelL2": np.array([[metrics["Final Rel-L2"]]], dtype=np.float32),
        }
        if "coordinates" in test_data:
            output["coordinates"] = test_data["coordinates"]
        elif "coordinates" in train_data:
            output["coordinates"] = train_data["coordinates"]
        savemat(out_path, output)
        print(f"Saved predictions to {out_path}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary_rows(summary_by_model, metric_keys):
    rows = []
    for model_name, item in summary_by_model.items():
        row = {
            "model": model_name,
            "num_seeds": item["num_seeds"],
            "seeds": ",".join(str(seed) for seed in item["seeds"]),
        }
        for key in metric_keys:
            values = item["metrics"][key]
            row[f"{key}_mean"] = values["mean"]
            row[f"{key}_std"] = values["std"]
            row[f"{key}_median"] = values["median"]
        rows.append(row)
    return rows


def paired_comparison(seedwise_rows, pairs, metric_keys):
    by_key = {(row["model"], int(row["seed"])): row for row in seedwise_rows}
    summary = []
    for baseline, conditioned in pairs:
        seeds = sorted(
            {seed for model, seed in by_key if model == baseline}
            & {seed for model, seed in by_key if model == conditioned}
        )
        if not seeds:
            continue

        row = {
            "pair": f"{conditioned}_vs_{baseline}",
            "baseline": baseline,
            "conditioned": conditioned,
            "num_paired_seeds": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for key in metric_keys:
            base = np.asarray([by_key[(baseline, seed)][key] for seed in seeds])
            cond = np.asarray([by_key[(conditioned, seed)][key] for seed in seeds])
            reductions = (base - cond) / base * 100.0
            row[f"{key}_wins"] = int(np.sum(cond < base))
            row[f"{key}_mean_reduction_percent"] = float(reductions.mean())
            row[f"{key}_median_reduction_percent"] = float(np.median(reductions))
        summary.append(row)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Navier--Stokes rollouts across trained seeds.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--model-root", default=".")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-mat", action="store_true")
    parser.add_argument(
        "--mean-drift",
        action="store_true",
        help="Also report the maximum spatial-mean drift used in the projection appendix.",
    )
    args = parser.parse_args()

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    train_path, test_path = resolve_data_paths(args.data_dir)
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Expected {train_path} and {test_path}")

    metric_keys = ["Mean Rel-L2", "Final Rel-L2"]
    if args.mean_drift:
        metric_keys.append("Mean Drift Max")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_data = loadmat(train_path)
    test_data = loadmat(test_path)
    seedwise_rows = []

    print(f"Navier--Stokes evaluation: models={models}, seeds={seeds}")
    print(f"Test data: {test_path}")
    for model_name in models:
        for seed in seeds:
            metrics = evaluate_one_model(
                model_name,
                seed,
                args.tag,
                args.model_root,
                test_data,
                train_data,
                args.device,
                eval_steps=args.eval_steps,
                save_mat=args.save_mat,
                save_dir=save_dir,
                mean_drift=args.mean_drift,
            )
            row = {"model": model_name, "seed": seed, "tag": args.tag}
            row.update({key: metrics[key] for key in metric_keys})
            seedwise_rows.append(row)

    summary_by_model = {}
    for model_name in models:
        rows = [row for row in seedwise_rows if row["model"] == model_name]
        summary_by_model[model_name] = {
            "num_seeds": len(rows),
            "seeds": [int(row["seed"]) for row in rows],
            "metrics": summarize_metric_dicts(rows, metric_keys),
        }
    paired_summary = paired_comparison(seedwise_rows, DEFAULT_PAIRS, metric_keys)

    write_csv(save_dir / "ns_fno_seedwise.csv", seedwise_rows)
    write_csv(
        save_dir / "ns_fno_summary_by_model.csv",
        flatten_summary_rows(summary_by_model, metric_keys),
    )
    write_csv(save_dir / "ns_fno_paired_summary.csv", paired_summary)

    print("\nSummary by model:", flush=True)
    for model_name, item in summary_by_model.items():
        print(f"\n{model_name}, seeds={item['seeds']}", flush=True)
        for key in metric_keys:
            values = item["metrics"][key]
            print(
                f"  {key}: mean={values['mean']:.6e}, std={values['std']:.6e}, "
                f"median={values['median']:.6e}",
                flush=True,
            )

    print("\nPaired comparison:", flush=True)
    for row in paired_summary:
        print(f"\n{row['pair']}, seeds={row['seeds']}", flush=True)
        for key in metric_keys:
            print(
                f"  {key}: wins={row[f'{key}_wins']}/{row['num_paired_seeds']}, "
                f"median reduction={row[f'{key}_median_reduction_percent']:.2f}%, "
                f"mean reduction={row[f'{key}_mean_reduction_percent']:.2f}%",
                flush=True,
            )

    print(f"\nSaved summaries to {save_dir}")


if __name__ == "__main__":
    main()
