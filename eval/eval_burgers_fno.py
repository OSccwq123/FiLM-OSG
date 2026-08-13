from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATA_DIR = REPO_ROOT / "data"
TEST_FILES = {
    "original": "BurgersOSG_test.mat",
    "sharp": "BurgersSharpOSG_test.mat",
}

DEFAULT_MODELS = ["fno", "fno_film"]
DEFAULT_PAIRS = [("fno", "fno_film"), ("gl_fno", "gl_fno_film")]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_TAG = ""
DEFAULT_SAVE_DIR = "./eval_outputs_burgers_fno"


def resolve_test_path(data_dir: str | Path, dataset: str):
    return Path(data_dir) / TEST_FILES[dataset]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def load_model(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)


def model_path_for(model_name, seed, tag, root="."):
    suffix = f"_{tag}" if tag else ""
    return Path(root) / f"runs_burgers_{model_name}_seed{seed}{suffix}" / "model"


def _high_frequency_error_1d(pred_state, true_state, band_frac=1.0 / 3.0, eps=1e-12):
    err = np.asarray(pred_state - true_state)
    ref = np.asarray(true_state)
    err_hat = np.fft.rfft(err, axis=0)
    ref_hat = np.fft.rfft(ref, axis=0)
    nfreq = err_hat.shape[0]
    start = max(1, int((1.0 - band_frac) * nfreq))
    return np.linalg.norm(err_hat[start:].reshape(-1)) / (np.linalg.norm(ref_hat[start:].reshape(-1)) + eps)


def compute_metrics(pred, truth, eps=1e-12):
    pred_roll = pred[..., 1:]
    true_roll = truth[..., 1:]

    rel_l2_list = []
    final_rel_l2_list = []
    hf_rel_list = []

    N = pred_roll.shape[0]
    T = pred_roll.shape[-1]

    for n in range(N):
        for t in range(T):
            p_state = pred_roll[n, ..., t]
            y_state = true_roll[n, ..., t]
            p = p_state.reshape(-1)
            y = y_state.reshape(-1)

            rel_l2_list.append(
                np.linalg.norm(p - y, 2) / (np.linalg.norm(y, 2) + eps)
            )
            hf_rel_list.append(_high_frequency_error_1d(p_state, y_state, eps=eps))

        pT = pred_roll[n, ..., -1].reshape(-1)
        yT = true_roll[n, ..., -1].reshape(-1)
        final_rel_l2_list.append(
            np.linalg.norm(pT - yT, 2) / (np.linalg.norm(yT, 2) + eps)
        )

    return {
        "Mean Rel-L2": float(np.mean(rel_l2_list)),
        "Final Rel-L2": float(np.mean(final_rel_l2_list)),
        "HF Rel-L2": float(np.mean(hf_rel_list)),
    }


def summarize_metric_dicts(rows, metric_keys):
    out = {}
    for k in metric_keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
        }
    return out


def evaluate_one_model(
    model_name,
    seed,
    tag,
    model_root,
    test_data,
    device,
    eval_steps=None,
    save_mat=False,
    save_dir=DEFAULT_SAVE_DIR,
):
    path = model_path_for(model_name, seed, tag, root=model_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model: {path}")

    model = load_model(path, device)
    model.eval()

    x0 = test_data["trajectories"][..., 0].astype(np.float32)
    dt = test_data["dt"].astype(np.float32)
    truth = test_data["trajectories"].astype(np.float32)

    if eval_steps is not None:
        dt = dt[:, :eval_steps]
        truth = truth[..., :eval_steps + 1]

    with torch.no_grad():
        pred = model.predict(x0, dt, device)

    pred = np.asarray(pred, dtype=np.float32)
    metrics = compute_metrics(pred, truth)

    step_tag = dt.shape[1]

    print(
        f"{model_name:12s} seed={seed}, steps={step_tag}: "
        f"Mean Rel-L2={metrics['Mean Rel-L2']:.6e}, "
        f"Final Rel-L2={metrics['Final Rel-L2']:.6e}, "
        f"HF Rel-L2={metrics['HF Rel-L2']:.6e}",
        flush=True,
    )

    if save_mat:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"steps{step_tag}" if eval_steps is not None else "full"
        tag_part = f"_{tag}" if tag else ""
        out_path = save_dir / f"{model_name}_seed{seed}{tag_part}_{suffix}_predictions.mat"

        output = {
            "prediction": pred,
            "truth": truth,
            "dt": dt,
            "metrics_MeanRelL2": np.array([[metrics["Mean Rel-L2"]]], dtype=np.float32),
            "metrics_FinalRelL2": np.array([[metrics["Final Rel-L2"]]], dtype=np.float32),
            "metrics_HFRelL2": np.array([[metrics["HF Rel-L2"]]], dtype=np.float32),
        }
        if "coordinates" in test_data:
            output["coordinates"] = test_data["coordinates"]
        savemat(out_path, output)
        print(f"Saved predictions to {out_path}", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


def write_csv(path: Path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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

        for k in metric_keys:
            s = item["metrics"][k]
            row[f"{k}_mean"] = s["mean"]
            row[f"{k}_std"] = s["std"]

        rows.append(row)

    return rows


def paired_comparison(seedwise_rows, pairs, metric_keys):
    by_key = {(r["model"], int(r["seed"])): r for r in seedwise_rows}
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
        for k in metric_keys:
            base = np.array([by_key[(baseline, seed)][k] for seed in seeds])
            cond = np.array([by_key[(conditioned, seed)][k] for seed in seeds])
            reductions = (base - cond) / base * 100.0
            row[f"{k}_wins"] = int(np.sum(cond < base))
            row[f"{k}_mean_reduction_percent"] = float(np.mean(reductions))
            row[f"{k}_median_reduction_percent"] = float(np.median(reductions))
        summary.append(row)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate Burgers rollouts across trained seeds.")

    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--model-root", type=str, default=".")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--dataset",
        choices=TEST_FILES,
        default="original",
        help="Select the original or steep-gradient Burgers data files.",
    )
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-mat", action="store_true")

    args = parser.parse_args()

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    test_path = resolve_test_path(args.data_dir, args.dataset)
    if not test_path.is_file():
        raise FileNotFoundError(f"Test data not found: {test_path}")

    metric_keys = ("Mean Rel-L2", "Final Rel-L2", "HF Rel-L2")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    test_data = loadmat(test_path)

    print(f"Burgers evaluation: dataset={args.dataset}, models={models}, seeds={seeds}")
    print(f"Test data: {test_path}")

    seedwise_rows = []
    for model_name in models:
        for seed in seeds:
            metrics = evaluate_one_model(
                model_name=model_name,
                seed=seed,
                tag=args.tag,
                model_root=args.model_root,
                test_data=test_data,
                device=args.device,
                eval_steps=args.eval_steps,
                save_mat=args.save_mat,
                save_dir=save_dir,
            )

            row = {
                "model": model_name,
                "seed": seed,
                "tag": args.tag,
            }
            row.update(metrics)
            seedwise_rows.append(row)

    summary_by_model = {}
    for model_name in models:
        rows = [r for r in seedwise_rows if r["model"] == model_name]
        summary_by_model[model_name] = {
            "model": model_name,
            "num_seeds": len(rows),
            "seeds": [int(r["seed"]) for r in rows],
            "metrics": summarize_metric_dicts(rows, metric_keys),
        }

    paired_summary = paired_comparison(seedwise_rows, DEFAULT_PAIRS, metric_keys)

    seedwise_csv = save_dir / "burgers_fno_seedwise.csv"
    summary_csv = save_dir / "burgers_fno_summary_by_model.csv"
    paired_csv = save_dir / "burgers_fno_paired_summary.csv"

    write_csv(seedwise_csv, seedwise_rows)
    write_csv(summary_csv, flatten_summary_rows(summary_by_model, metric_keys))
    write_csv(paired_csv, paired_summary)

    print("\nSummary by model:", flush=True)
    for model_name, item in summary_by_model.items():
        print(f"\n{model_name}, seeds={item['seeds']}", flush=True)
        for k in metric_keys:
            s = item["metrics"][k]
            print(
                f"  {k}: "
                f"mean={s['mean']:.6e}, std={s['std']:.6e}",
                flush=True,
            )

    print("\nPaired comparison:", flush=True)
    for row in paired_summary:
        print(f"\n{row['pair']}, seeds={row['seeds']}", flush=True)
        for k in metric_keys:
            print(
                f"  {k}: "
                f"wins={row[f'{k}_wins']}/{row['num_paired_seeds']}, "
                f"median reduction={row[f'{k}_median_reduction_percent']:.2f}%, "
                f"mean reduction={row[f'{k}_mean_reduction_percent']:.2f}%",
                flush=True,
            )

    print(f"\nSaved summaries to {save_dir}")


if __name__ == "__main__":
    main()
