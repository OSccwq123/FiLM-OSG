import os
import csv
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat, savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATA_DIR = REPO_ROOT / "data"
TRAIN_FILE = "VorticityOSG_train.mat"
TEST_FILE = "VorticityOSG_test.mat"

DEFAULT_MODELS = [
    "uno",
    "uno_film",
    "transolver",
    "transolver_film",
]

DEFAULT_PAIRS = [
    ("uno", "uno_film"),
    ("transolver", "transolver_film"),
]

DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_TAG = ""
DEFAULT_SAVE_DIR = "./eval_outputs_ns_extra_backbones"


def resolve_data_paths(data_dir: str | os.PathLike[str]):
    data_root = Path(data_dir)
    return str(data_root / TRAIN_FILE), str(data_root / TEST_FILE)


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def safe_torch_load(model_path, device):
    from film_osg.compat import install_due_pickle_aliases

    compat_source = install_due_pickle_aliases()
    print("pickle_compat_source =", compat_source, flush=True)

    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def model_path_for(model_name, seed, tag, root="."):
    suffix = f"_{tag}" if tag else ""
    return os.path.join(root, f"runs_ns_{model_name}_seed{seed}{suffix}", "model")


def compute_metrics(pred, truth, eps=1e-12):
    """
    pred/truth shape: (N, H, W, C, T+1) or compatible.
    Metrics exclude initial condition at t=0.
    """
    pred_roll = pred[..., 1:]
    true_roll = truth[..., 1:]

    mae = np.abs(pred_roll - true_roll).mean()

    rel_l1_list = []
    rel_l2_list = []
    final_rel_l2_list = []

    N = pred_roll.shape[0]
    T = pred_roll.shape[-1]

    for n in range(N):
        for t in range(T):
            p = pred_roll[n, ..., t].reshape(-1)
            y = true_roll[n, ..., t].reshape(-1)

            rel_l1_list.append(
                np.linalg.norm(p - y, 1) / (np.linalg.norm(y, 1) + eps)
            )
            rel_l2_list.append(
                np.linalg.norm(p - y, 2) / (np.linalg.norm(y, 2) + eps)
            )

        pT = pred_roll[n, ..., -1].reshape(-1)
        yT = true_roll[n, ..., -1].reshape(-1)
        final_rel_l2_list.append(
            np.linalg.norm(pT - yT, 2) / (np.linalg.norm(yT, 2) + eps)
        )

    return {
        "MAE": float(mae),
        "Rel-L1": float(np.mean(rel_l1_list)),
        "Mean Rel-L2": float(np.mean(rel_l2_list)),
        "Final Rel-L2": float(np.mean(final_rel_l2_list)),
    }


def summarize_metric_dicts(rows, metric_keys):
    out = {}

    for k in metric_keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)

        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "median": float(np.median(vals)),
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "values": [float(v) for v in vals],
        }

    return out


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

    if "coordinates" in test_data:
        coordinates = test_data["coordinates"]
    else:
        coordinates = train_data["coordinates"]

    with torch.no_grad():
        pred = model.predict(x0, dt, device)

    pred = np.asarray(pred, dtype=np.float32)
    metrics = compute_metrics(pred, truth)

    step_tag = dt.shape[1]

    print(
        f"{model_name:18s} seed={seed:>3d}, steps={step_tag:>3d} -> "
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
        out_path = os.path.join(
            save_dir,
            f"{model_name}_seed{seed}{tag_part}_{suffix}_predictions.mat",
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


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary_rows(summary_by_model, metric_keys):
    rows = []

    for model_name, item in summary_by_model.items():
        row = {
            "model": model_name,
            "num_seeds": item["num_seeds"],
            "seeds": json.dumps(item["seeds"]),
        }

        for k in metric_keys:
            s = item["metrics"][k]
            row[f"{k}_mean"] = s["mean"]
            row[f"{k}_std"] = s["std"]
            row[f"{k}_median"] = s["median"]
            row[f"{k}_q25"] = s["q25"]
            row[f"{k}_q75"] = s["q75"]
            row[f"{k}_min"] = s["min"]
            row[f"{k}_max"] = s["max"]

        rows.append(row)

    return rows


def paired_comparison(seedwise_rows, pairs, metric_keys):
    """
    Within-backbone paired comparison:
      direct vs film for same seed.
    """
    by_key = {}
    for r in seedwise_rows:
        by_key[(r["model"], int(r["seed"]))] = r

    paired_rows = []
    paired_summary = []

    for direct, film in pairs:
        rows_pair = []

        seeds = sorted(
            set(seed for (model, seed) in by_key.keys() if model == direct)
            & set(seed for (model, seed) in by_key.keys() if model == film)
        )

        for seed in seeds:
            d = by_key[(direct, seed)]
            f = by_key[(film, seed)]

            row = {
                "pair": f"{film}_vs_{direct}",
                "direct": direct,
                "film": film,
                "seed": seed,
            }

            for k in metric_keys:
                direct_val = float(d[k])
                film_val = float(f[k])

                row[f"direct_{k}"] = direct_val
                row[f"film_{k}"] = film_val
                row[f"film_better_{k}"] = bool(film_val < direct_val)
                row[f"reduction_percent_{k}"] = (
                    (direct_val - film_val) / direct_val * 100.0
                    if direct_val != 0.0
                    else np.nan
                )

            paired_rows.append(row)
            rows_pair.append(row)

        summary_row = {
            "pair": f"{film}_vs_{direct}",
            "direct": direct,
            "film": film,
            "num_paired_seeds": len(rows_pair),
            "seeds": json.dumps(seeds),
        }

        for k in metric_keys:
            reductions = np.array(
                [r[f"reduction_percent_{k}"] for r in rows_pair],
                dtype=np.float64,
            )
            wins = sum(bool(r[f"film_better_{k}"]) for r in rows_pair)

            if len(reductions) > 0:
                summary_row[f"{k}_wins"] = int(wins)
                summary_row[f"{k}_total"] = int(len(reductions))
                summary_row[f"{k}_mean_reduction_percent"] = float(np.nanmean(reductions))
                summary_row[f"{k}_median_reduction_percent"] = float(np.nanmedian(reductions))
                summary_row[f"{k}_q25_reduction_percent"] = float(np.nanpercentile(reductions, 25))
                summary_row[f"{k}_q75_reduction_percent"] = float(np.nanpercentile(reductions, 75))
                summary_row[f"{k}_reductions"] = json.dumps([float(x) for x in reductions])
            else:
                summary_row[f"{k}_wins"] = 0
                summary_row[f"{k}_total"] = 0
                summary_row[f"{k}_mean_reduction_percent"] = np.nan
                summary_row[f"{k}_median_reduction_percent"] = np.nan
                summary_row[f"{k}_q25_reduction_percent"] = np.nan
                summary_row[f"{k}_q75_reduction_percent"] = np.nan
                summary_row[f"{k}_reductions"] = json.dumps([])

        paired_summary.append(summary_row)

    return paired_rows, paired_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--tag", type=str, default=DEFAULT_TAG)
    parser.add_argument("--model-root", type=str, default=".")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--save-mat", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    train_path, test_path = resolve_data_paths(args.data_dir)

    metric_keys = ["MAE", "Rel-L1", "Mean Rel-L2", "Final Rel-L2"]

    print("=" * 80, flush=True)
    print("NS extra-backbone evaluation", flush=True)
    print("Models:", models, flush=True)
    print("Seeds:", seeds, flush=True)
    print("Tag:", args.tag if args.tag else "(none)", flush=True)
    print("Model root:", args.model_root, flush=True)
    print("Device:", args.device, flush=True)
    print("Eval steps:", args.eval_steps if args.eval_steps is not None else "full", flush=True)
    print("Save dir:", args.save_dir, flush=True)
    print("=" * 80, flush=True)

    if args.check_only or args.dry_run:
        print("Check-only mode: no .mat files or model weights will be loaded.", flush=True)
        print("Train data exists:", os.path.exists(train_path), train_path, flush=True)
        print("Test data exists:", os.path.exists(test_path), test_path, flush=True)
        for model_name in models:
            for seed in seeds:
                path = model_path_for(model_name, seed, args.tag, root=args.model_root)
                exists = os.path.exists(path)
                print(f"Model path exists={exists}: {path}", flush=True)
                if not exists and not args.skip_missing:
                    raise FileNotFoundError(f"Missing model: {path}")
        return

    os.makedirs(args.save_dir, exist_ok=True)

    train_data = loadmat(train_path)
    test_data = loadmat(test_path)

    print("Train trajectories shape:", train_data["trajectories"].shape, flush=True)
    print("Test trajectories shape:", test_data["trajectories"].shape, flush=True)
    print("Test dt shape:", test_data["dt"].shape, flush=True)

    seedwise_rows = []
    missing = []

    for model_name in models:
        for seed in seeds:
            path = model_path_for(model_name, seed, args.tag, root=args.model_root)

            if not os.path.exists(path):
                msg = f"Missing model: {path}"
                if args.skip_missing:
                    print("[SKIP]", msg, flush=True)
                    missing.append({"model": model_name, "seed": seed, "path": path})
                    continue
                raise FileNotFoundError(msg)

            metrics = evaluate_one_model(
                model_name=model_name,
                seed=seed,
                tag=args.tag,
                model_root=args.model_root,
                test_data=test_data,
                train_data=train_data,
                device=args.device,
                eval_steps=args.eval_steps,
                save_mat=args.save_mat,
                save_dir=args.save_dir,
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

        if not rows:
            continue

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

    # Save outputs.
    seedwise_csv = os.path.join(args.save_dir, "extra_backbones_seedwise.csv")
    summary_csv = os.path.join(args.save_dir, "extra_backbones_summary_by_model.csv")
    summary_json = os.path.join(args.save_dir, "extra_backbones_summary_by_model.json")
    paired_csv = os.path.join(args.save_dir, "extra_backbones_paired_seedwise.csv")
    paired_summary_csv = os.path.join(args.save_dir, "extra_backbones_paired_summary.csv")
    paired_json = os.path.join(args.save_dir, "extra_backbones_paired.json")
    missing_json = os.path.join(args.save_dir, "extra_backbones_missing.json")

    write_csv(seedwise_csv, seedwise_rows)
    write_csv(summary_csv, flatten_summary_rows(summary_by_model, metric_keys))
    write_csv(paired_csv, paired_rows)
    write_csv(paired_summary_csv, paired_summary)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": models,
                "seeds": seeds,
                "tag": args.tag,
                "eval_steps": args.eval_steps,
                "summary_by_model": summary_by_model,
            },
            f,
            indent=2,
        )

    with open(paired_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pairs": DEFAULT_PAIRS,
                "paired_seedwise": paired_rows,
                "paired_summary": paired_summary,
            },
            f,
            indent=2,
        )

    if missing:
        with open(missing_json, "w", encoding="utf-8") as f:
            json.dump(missing, f, indent=2)

    print("\nSummary by model:", flush=True)
    for model_name, item in summary_by_model.items():
        print(f"\n{model_name}, seeds={item['seeds']}", flush=True)
        for k in metric_keys:
            s = item["metrics"][k]
            print(
                f"  {k}: "
                f"mean={s['mean']:.6e}, std={s['std']:.6e}, "
                f"median={s['median']:.6e}, "
                f"IQR=[{s['q25']:.6e}, {s['q75']:.6e}], "
                f"range=[{s['min']:.6e}, {s['max']:.6e}]",
                flush=True,
            )

    print("\nPaired comparison:", flush=True)
    for row in paired_summary:
        print(f"\n{row['pair']}, seeds={row['seeds']}", flush=True)
        for k in metric_keys:
            print(
                f"  {k}: "
                f"wins={row[f'{k}_wins']}/{row[f'{k}_total']}, "
                f"median reduction={row[f'{k}_median_reduction_percent']:.2f}%, "
                f"mean reduction={row[f'{k}_mean_reduction_percent']:.2f}%",
                flush=True,
            )

    print("\nSaved outputs:", flush=True)
    print(" ", seedwise_csv, flush=True)
    print(" ", summary_csv, flush=True)
    print(" ", summary_json, flush=True)
    print(" ", paired_csv, flush=True)
    print(" ", paired_summary_csv, flush=True)
    print(" ", paired_json, flush=True)
    if missing:
        print(" ", missing_json, flush=True)

    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
