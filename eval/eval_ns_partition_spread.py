import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from eval.diagnostics import _band_masks_2d
except ModuleNotFoundError:
    from diagnostics import _band_masks_2d

DEFAULT_DATA_DIR = REPO_ROOT / "data"
TEST_FILE = "VorticityOSG_test.mat"


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def safe_torch_load(model_path, device):
    return torch.load(model_path, map_location=device, weights_only=False)


def model_path_for(model_name, seed, tag, root="."):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(root, f"runs_ns_{model_name}_seed{seed}{suffix}", "model")


def make_equal_partition_dt(total_dt, n_parts):
    total_dt = np.asarray(total_dt, dtype=np.float32).reshape(-1, 1)
    return np.repeat(total_dt / float(n_parts), n_parts, axis=1).astype(np.float32)


def flatten_state(x):
    return np.asarray(x).reshape(x.shape[0], -1)


def rel_l2(a, b, eps=1e-12):
    diff = flatten_state(a - b)
    denom = 0.5 * (np.linalg.norm(flatten_state(a), axis=1) + np.linalg.norm(flatten_state(b), axis=1))
    return np.linalg.norm(diff, axis=1) / (denom + eps)


def high_frequency_rel_l2(a, b, eps=1e-12):
    a_ch = np.moveaxis(a, -1, 1)
    b_ch = np.moveaxis(b, -1, 1)
    ahat = np.fft.rfft2(a_ch, axes=(-2, -1))
    bhat = np.fft.rfft2(b_ch, axes=(-2, -1))
    mask = _band_masks_2d(a.shape[1], a.shape[2])["High Band Rel-L2"]
    diff = (ahat - bhat)[:, :, mask].reshape(a.shape[0], -1)
    scale = 0.5 * (
        np.linalg.norm(ahat[:, :, mask].reshape(a.shape[0], -1), axis=1)
        + np.linalg.norm(bhat[:, :, mask].reshape(b.shape[0], -1), axis=1)
    )
    return np.linalg.norm(diff, axis=1) / (scale + eps)


def spread_metrics(final_by_partition):
    parts = sorted(final_by_partition)
    full_vals = []
    hf_vals = []
    for i, p in enumerate(parts):
        for q in parts[i + 1:]:
            a = final_by_partition[p]
            b = final_by_partition[q]
            full_vals.extend(rel_l2(a, b).tolist())
            hf_vals.extend(high_frequency_rel_l2(a, b).tolist())
    return {
        "Partition Spread Rel-L2": float(np.mean(full_vals)),
        "Partition Spread Rel-L2 Std": float(np.std(full_vals)),
        "HF Partition Spread Rel-L2": float(np.mean(hf_vals)),
        "HF Partition Spread Rel-L2 Std": float(np.std(hf_vals)),
        "Num Partition Pairs": int(len(parts) * (len(parts) - 1) // 2),
    }


def evaluate_one(model_name, seed, tag, model_root, test_data, device, partitions, max_samples=None):
    path = model_path_for(model_name, seed, tag, root=model_root)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model: {path}")

    model = safe_torch_load(path, device)
    model.eval()

    x0 = test_data["trajectories"][..., 0].astype(np.float32)
    dt_ref = test_data["dt"].astype(np.float32)
    if max_samples is not None:
        x0 = x0[:max_samples]
        dt_ref = dt_ref[:max_samples]

    total_dt = dt_ref.sum(axis=1)
    final_by_partition = {}
    with torch.no_grad():
        for n_parts in partitions:
            dt = make_equal_partition_dt(total_dt, n_parts)
            pred = model.predict(x0, dt, device)
            final_by_partition[n_parts] = np.asarray(pred[..., -1], dtype=np.float32)
            print(f"{model_name} seed={seed} partition={n_parts} done", flush=True)

    metrics = spread_metrics(final_by_partition)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="fno,fno_film")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model-root", type=str, default=".")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--save-dir", type=str, default="./eval_outputs_ns_partition_spread")
    parser.add_argument("--partitions", type=str, default="1,2,4,8")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    partitions = parse_int_list(args.partitions)
    test_path = Path(args.data_dir) / TEST_FILE

    print("NS high-frequency partition spread", flush=True)
    print("Models:", models, flush=True)
    print("Seeds:", seeds, flush=True)
    print("Partitions:", partitions, flush=True)
    print("Device:", args.device, flush=True)
    print("Test data:", test_path, flush=True)

    test_data = loadmat(test_path)
    rows = []
    for model_name in models:
        for seed in seeds:
            path = model_path_for(model_name, seed, args.tag, root=args.model_root)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing model: {path}")
            metrics = evaluate_one(model_name, seed, args.tag, args.model_root, test_data, args.device, partitions, args.max_samples)
            row = {"model": model_name, "seed": seed, "tag": args.tag}
            row.update(metrics)
            rows.append(row)

    os.makedirs(args.save_dir, exist_ok=True)
    seedwise_path = os.path.join(args.save_dir, "ns_partition_spread_seedwise.csv")
    summary_path = os.path.join(args.save_dir, "ns_partition_spread_summary.json")
    write_csv(seedwise_path, rows)

    summary = {}
    for model_name in models:
        model_rows = [r for r in rows if r["model"] == model_name]
        if not model_rows:
            continue
        metric_names = [k for k in model_rows[0] if k not in {"model", "seed", "tag"}]
        summary[model_name] = {"seeds": [int(r["seed"]) for r in model_rows], "metrics": {}}
        for key in metric_names:
            vals = np.array([r[key] for r in model_rows], dtype=np.float64)
            summary[model_name]["metrics"][key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "median": float(np.median(vals)),
                "values": [float(x) for x in vals],
            }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "partitions": partitions}, f, indent=2)

    print("Saved:", seedwise_path, flush=True)
    print("Saved:", summary_path, flush=True)
    for model_name, item in summary.items():
        print(f"\n{model_name}, seeds={item['seeds']}", flush=True)
        for key, val in item["metrics"].items():
            print(f"  {key}: mean={val['mean']:.6e}, std={val['std']:.6e}, median={val['median']:.6e}", flush=True)


if __name__ == "__main__":
    main()
