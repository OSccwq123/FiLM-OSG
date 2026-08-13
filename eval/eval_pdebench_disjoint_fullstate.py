#!/usr/bin/env python3
"""Evaluate trajectory-disjoint PDEBench-derived full-state checkpoints."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.diagnostics import band_error_metrics, spectrum_error_metrics
from eval.eval_convdiff_fno import compute_metrics, model_path_for, safe_torch_load


def parse_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_seeds(text: str) -> list[int]:
    return [int(item) for item in parse_list(text)]


def batched_prediction(model, x0, dt, device, batch_size):
    outputs = []
    for start in range(0, x0.shape[0], batch_size):
        stop = min(start + batch_size, x0.shape[0])
        outputs.append(model.predict(x0[start:stop], dt[start:stop], device))
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def evaluate(model_path, test_data, device, batch_size):
    model = safe_torch_load(model_path, device)
    model.eval()
    x0 = test_data["trajectories"][..., 0].astype(np.float32)
    dt = test_data["dt"].astype(np.float32)
    truth = test_data["trajectories"].astype(np.float32)
    with torch.no_grad():
        pred = batched_prediction(model, x0, dt, device, batch_size)
    if pred.shape != truth.shape:
        raise RuntimeError(f"prediction shape {pred.shape} != truth shape {truth.shape}")
    if not np.isfinite(pred).all():
        raise FloatingPointError("Prediction contains NaN or Inf values.")

    base = compute_metrics(pred, truth)
    bands = band_error_metrics(pred, truth)
    spectrum = spectrum_error_metrics(pred, truth)
    metrics = {
        "Mean Rel-L2": base["Mean Rel-L2"],
        "HF Rel-L2": bands["HF Rel-L2"],
        "Spectrum Error": spectrum["Spectrum Error"],
    }

    del model, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(seedwise: list[dict], models: list[str], metric_keys: list[str]):
    rows = []
    for model in models:
        selected = [row for row in seedwise if row["model"] == model]
        row = {"model": model, "num_seeds": len(selected), "seeds": ",".join(str(r["seed"]) for r in selected)}
        for metric in metric_keys:
            values = np.asarray([float(item[metric]) for item in selected], dtype=np.float64)
            row[f"{metric} mean"] = float(values.mean())
            row[f"{metric} population std"] = float(values.std(ddof=0))
        rows.append(row)
    return rows


def paired(seedwise: list[dict], pairs: list[tuple[str, str]], metric_keys: list[str]):
    lookup = {(row["model"], int(row["seed"])): row for row in seedwise}
    rows = []
    for direct, film in pairs:
        seeds = sorted(
            {seed for model, seed in lookup if model == direct}
            & {seed for model, seed in lookup if model == film}
        )
        row = {"pair": f"{film}_vs_{direct}", "seeds": ",".join(str(seed) for seed in seeds), "num_seeds": len(seeds)}
        for metric in metric_keys:
            direct_values = np.asarray([lookup[(direct, seed)][metric] for seed in seeds], dtype=np.float64)
            film_values = np.asarray([lookup[(film, seed)][metric] for seed in seeds], dtype=np.float64)
            reductions = 100.0 * (direct_values - film_values) / direct_values
            row[f"{metric} wins"] = int(np.sum(film_values < direct_values))
            row[f"{metric} mean reduction percent"] = float(reductions.mean())
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model-root", default=Path("."), type=Path)
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    models = parse_list(args.models)
    seeds = parse_seeds(args.seeds)
    test_data = loadmat(args.data_dir / "test_data.mat")
    args.save_dir.mkdir(parents=True, exist_ok=True)
    seedwise = []
    metric_keys = None
    for model in models:
        for seed in seeds:
            path = Path(model_path_for(model, seed, args.tag, root=str(args.model_root)))
            if not path.exists():
                raise FileNotFoundError(path)
            metrics = evaluate(path, test_data, args.device, args.batch_size)
            row = {"model": model, "seed": seed, "tag": args.tag}
            row.update(metrics)
            seedwise.append(row)
            metric_keys = list(metrics.keys())
            print(
                f"{model} seed={seed}: RelL2={metrics['Mean Rel-L2']:.6e}, "
                f"HF={metrics['HF Rel-L2']:.6e}, Spectrum={metrics['Spectrum Error']:.6e}",
                flush=True,
            )

    assert metric_keys is not None
    pairs = [("fno", "fno_film")]
    if "vt_fno" in models and "vt_fno_film" in models:
        pairs.append(("vt_fno", "vt_fno_film"))
    summary = summarize(seedwise, models, metric_keys)
    paired_rows = paired(seedwise, pairs, metric_keys)
    write_csv(args.save_dir / "seedwise.csv", seedwise)
    write_csv(args.save_dir / "summary_by_model.csv", summary)
    write_csv(args.save_dir / "paired_summary.csv", paired_rows)
    print(f"Saved evaluation to {args.save_dir}", flush=True)


if __name__ == "__main__":
    main()
