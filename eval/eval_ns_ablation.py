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


TRAIN_PATH = "VorticityOSG_train.mat"
TEST_PATH = "VorticityOSG_test.mat"

DEFAULT_SEEDS = [0]
DEFAULT_VARIANTS = ["direct_nosg", "film_nosg", "direct_sg", "film_sg"]


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def get_model_path(variant_name, seed, root=".", tag=""):
    """
    Path convention for NS ablation.

    no-SG variants:
        direct_nosg -> runs_ns_direct_nosg_seed{s}
        film_nosg   -> runs_ns_film_nosg_seed{s}

    SG variants reuse the main NS FNO runs:
        direct_sg -> runs_ns_fno_seed{s}
        film_sg   -> runs_ns_fno_film_seed{s}

    Fallbacks are included in case direct_sg / film_sg were trained separately.
    """
    suffix = f"_{tag}" if tag else ""
    candidates = {
        "direct_nosg": [
            os.path.join(root, f"runs_ns_direct_nosg_seed{seed}{suffix}", "model"),
        ],
        "film_nosg": [
            os.path.join(root, f"runs_ns_film_nosg_seed{seed}{suffix}", "model"),
        ],
        "direct_sg": [
            os.path.join(root, f"runs_ns_fno_seed{seed}{suffix}", "model"),
            os.path.join(root, f"runs_ns_direct_sg_seed{seed}{suffix}", "model"),
        ],
        "film_sg": [
            os.path.join(root, f"runs_ns_fno_film_seed{seed}{suffix}", "model"),
            os.path.join(root, f"runs_ns_film_sg_seed{seed}{suffix}", "model"),
        ],
    }

    if variant_name not in candidates:
        raise ValueError(f"Unknown variant_name: {variant_name}")

    for path in candidates[variant_name]:
        if os.path.exists(path):
            return path

    msg = "\n".join(candidates[variant_name])
    raise FileNotFoundError(
        f"No model file found for variant={variant_name}, seed={seed}.\n"
        f"Checked:\n{msg}"
    )


def compute_metrics(pred, truth, eps=1e-12):
    """
    pred, truth: arrays with rollout time as the last axis.
    The first frame is the initial state and is excluded from rollout metrics.
    """
    pred_roll = pred[..., 1:]
    true_roll = truth[..., 1:]

    mae = np.abs(pred_roll - true_roll).mean()

    rel_l1_list = []
    rel_l2_list = []
    final_rel_l2_list = []

    n_test = pred_roll.shape[0]
    n_steps = pred_roll.shape[-1]

    for n in range(n_test):
        for t in range(n_steps):
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


def summarize_metrics(metrics_list, stable_final_threshold=1.0):
    keys = ["MAE", "Rel-L1", "Mean Rel-L2", "Final Rel-L2"]
    summary = {}

    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=np.float64)
        summary[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "values": [float(v) for v in vals],
        }

    final_vals = np.array([m["Final Rel-L2"] for m in metrics_list], dtype=np.float64)
    summary["stable_final_threshold"] = float(stable_final_threshold)
    summary["stable_seeds_count"] = int(np.sum(final_vals < stable_final_threshold))

    return summary


def evaluate_one_model(
    model_path,
    variant_name,
    seed,
    device,
    eval_steps=None,
    save_mat=True,
    save_dir="./eval_outputs_ns_ablation_seed012",
):
    train_data = loadmat(TRAIN_PATH)
    test_data = loadmat(TEST_PATH)

    from film_osg.compat import install_due_pickle_aliases

    compat_source = install_due_pickle_aliases()
    print("pickle_compat_source =", compat_source, flush=True)
    model = torch.load(model_path, map_location=device)
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

    pred = model.predict(x0, dt, device)
    metrics = compute_metrics(pred, truth)

    suffix = f"steps{dt.shape[1]}" if eval_steps is not None else "full"

    print(
        f"{variant_name:12s} seed={seed:3d}, {suffix:>6s} -> "
        f"MAE={metrics['MAE']:.6e}, "
        f"Rel-L1={metrics['Rel-L1']:.6e}, "
        f"Mean Rel-L2={metrics['Mean Rel-L2']:.6e}, "
        f"Final Rel-L2={metrics['Final Rel-L2']:.6e}",
        flush=True,
    )

    if save_mat:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(
            save_dir,
            f"{variant_name}_seed{seed}_{suffix}_predictions.mat",
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
                "metrics_MeanRelL2": np.array(
                    [[metrics["Mean Rel-L2"]]], dtype=np.float32
                ),
                "metrics_FinalRelL2": np.array(
                    [[metrics["Final Rel-L2"]]], dtype=np.float32
                ),
            },
        )
        print(f"Saved predictions to {out_path}", flush=True)

    return metrics


def evaluate_variant_family(
    variant_name,
    seeds,
    device,
    model_root=".",
    tag="",
    eval_steps=None,
    save_mat=True,
    save_dir="./eval_outputs_ns_ablation_seed012",
    stable_final_threshold=1.0,
    skip_missing=False,
):
    metrics_list = []
    seedwise_rows = []

    for seed in seeds:
        try:
            model_path = get_model_path(variant_name, seed, root=model_root, tag=tag)
        except FileNotFoundError as exc:
            if skip_missing:
                print("[SKIP]", exc, flush=True)
                continue
            raise
        metrics = evaluate_one_model(
            model_path=model_path,
            variant_name=variant_name,
            seed=seed,
            device=device,
            eval_steps=eval_steps,
            save_mat=save_mat,
            save_dir=save_dir,
        )
        metrics_list.append(metrics)

        row = {"variant": variant_name, "seed": seed}
        row.update(metrics)
        seedwise_rows.append(row)

    if not metrics_list:
        print(f"No completed evaluations for {variant_name}.", flush=True)
        return [], {}, []

    summary = summarize_metrics(
        metrics_list,
        stable_final_threshold=stable_final_threshold,
    )

    os.makedirs(save_dir, exist_ok=True)
    suffix = f"steps{eval_steps}" if eval_steps is not None else "full"

    summary_path = os.path.join(save_dir, f"{variant_name}_{suffix}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    seedwise_path = os.path.join(save_dir, f"{variant_name}_{suffix}_seedwise.csv")
    write_seedwise_csv(seedwise_path, seedwise_rows)

    print(f"\nSummary for {variant_name} over seeds {seeds}:")
    for k in ["MAE", "Rel-L1", "Mean Rel-L2", "Final Rel-L2"]:
        v = summary[k]
        print(
            f"  {k}: "
            f"mean={v['mean']:.6e}, std={v['std']:.6e}, "
            f"median={v['median']:.6e}, "
            f"range=[{v['min']:.6e}, {v['max']:.6e}]",
            flush=True,
        )
    print(
        f"  stable seeds by Final Rel-L2 < {stable_final_threshold}: "
        f"{summary['stable_seeds_count']}/{len(seeds)}",
        flush=True,
    )
    print(f"Saved summary to {summary_path}")
    print(f"Saved seedwise CSV to {seedwise_path}\n")

    return metrics_list, summary, seedwise_rows


def write_seedwise_csv(path, rows):
    if not rows:
        return

    fieldnames = [
        "variant",
        "seed",
        "MAE",
        "Rel-L1",
        "Mean Rel-L2",
        "Final Rel-L2",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def paired_reduction(direct_metrics, film_metrics, keys):
    out = {}

    for k in keys:
        direct_vals = np.array([m[k] for m in direct_metrics], dtype=np.float64)
        film_vals = np.array([m[k] for m in film_metrics], dtype=np.float64)

        reduction = (direct_vals - film_vals) / np.maximum(np.abs(direct_vals), 1e-12)

        out[k] = {
            "wins": int(np.sum(film_vals < direct_vals)),
            "n": int(len(direct_vals)),
            "mean_reduction_percent": float(100.0 * reduction.mean()),
            "median_reduction_percent": float(100.0 * np.median(reduction)),
            "min_reduction_percent": float(100.0 * reduction.min()),
            "max_reduction_percent": float(100.0 * reduction.max()),
            "direct_values": [float(v) for v in direct_vals],
            "film_values": [float(v) for v in film_vals],
            "reduction_values_percent": [float(100.0 * v) for v in reduction],
        }

    return out


def save_combined_csv(path, all_rows):
    if not all_rows:
        return

    fieldnames = [
        "variant",
        "seed",
        "MAE",
        "Rel-L1",
        "Mean Rel-L2",
        "Final Rel-L2",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="Comma-separated seeds, e.g. 0,1,2.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(DEFAULT_VARIANTS),
        help=(
            "Comma-separated variants. Available: "
            "direct_nosg,film_nosg,direct_sg,film_sg."
        ),
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./eval_outputs_ns_ablation_seed0",
    )
    parser.add_argument("--model-root", type=str, default=".")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=-1,
        help="Use -1 for full rollout; otherwise use the given number of rollout steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--no-save-mat",
        action="store_true",
        help="Do not save prediction .mat files.",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stable-final-threshold",
        type=float,
        default=1.0,
        help="Diagnostic threshold for counting stable seeds by Final Rel-L2.",
    )

    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    variants = parse_str_list(args.variants)
    eval_steps = None if args.eval_steps < 0 else args.eval_steps

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    save_mat = not args.no_save_mat

    print("=" * 80, flush=True)
    print("NS ablation evaluation", flush=True)
    print("Seeds:", seeds, flush=True)
    print("Variants:", variants, flush=True)
    print("Tag:", args.tag if args.tag else "(none)", flush=True)
    print("Model root:", args.model_root, flush=True)
    print("Device:", device, flush=True)
    print("Eval steps:", "full" if eval_steps is None else eval_steps, flush=True)
    print("Save dir:", args.save_dir, flush=True)
    print("Save mat:", save_mat, flush=True)
    print("=" * 80, flush=True)

    if args.check_only or args.dry_run:
        print("Check-only mode: no .mat files or model weights will be loaded.", flush=True)
        print("Train data exists:", os.path.exists(TRAIN_PATH), TRAIN_PATH, flush=True)
        print("Test data exists:", os.path.exists(TEST_PATH), TEST_PATH, flush=True)
        for variant in variants:
            for seed in seeds:
                try:
                    path = get_model_path(variant, seed, root=args.model_root, tag=args.tag)
                    print(f"Model path exists=True: {path}", flush=True)
                except FileNotFoundError as exc:
                    print("[MISSING]", exc, flush=True)
                    if not args.skip_missing:
                        raise
        return

    all_metrics = {}
    all_summaries = {}
    all_seedwise_rows = []

    for variant in variants:
        metrics_list, summary, seedwise_variant_rows = evaluate_variant_family(
            variant_name=variant,
            seeds=seeds,
            device=device,
            model_root=args.model_root,
            tag=args.tag,
            eval_steps=eval_steps,
            save_mat=save_mat,
            save_dir=args.save_dir,
            stable_final_threshold=args.stable_final_threshold,
            skip_missing=args.skip_missing,
        )
        all_metrics[variant] = metrics_list
        all_summaries[variant] = summary
        all_seedwise_rows.extend(seedwise_variant_rows)

    suffix = f"steps{eval_steps}" if eval_steps is not None else "full"

    combined_csv = os.path.join(args.save_dir, f"ns_ablation_{suffix}_seedwise.csv")
    save_combined_csv(combined_csv, all_seedwise_rows)

    combined_json = os.path.join(args.save_dir, f"ns_ablation_{suffix}_summary.json")
    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("=" * 80)
    print("Combined summary saved:")
    print(" ", combined_csv)
    print(" ", combined_json)

    keys = ["MAE", "Rel-L1", "Mean Rel-L2", "Final Rel-L2"]

    paired = {}

    if all_metrics.get("direct_sg") and all_metrics.get("film_sg"):
        paired["film_sg_vs_direct_sg"] = paired_reduction(
            all_metrics["direct_sg"],
            all_metrics["film_sg"],
            keys,
        )

    if all_metrics.get("direct_nosg") and all_metrics.get("film_nosg"):
        paired["film_nosg_vs_direct_nosg"] = paired_reduction(
            all_metrics["direct_nosg"],
            all_metrics["film_nosg"],
            keys,
        )

    if paired:
        paired_json = os.path.join(args.save_dir, f"ns_ablation_{suffix}_paired.json")
        with open(paired_json, "w", encoding="utf-8") as f:
            json.dump(paired, f, indent=2)

        print("\nPaired comparisons:")
        for pair_name, pair_summary in paired.items():
            print(f"\n{pair_name}")
            for k, v in pair_summary.items():
                print(
                    f"  {k}: wins={v['wins']}/{v['n']}, "
                    f"median reduction={v['median_reduction_percent']:.2f}%, "
                    f"mean reduction={v['mean_reduction_percent']:.2f}%, "
                    f"range=[{v['min_reduction_percent']:.2f}%, "
                    f"{v['max_reduction_percent']:.2f}%]",
                    flush=True,
                )

        print("Saved paired summary to:")
        print(" ", paired_json)

    print("=" * 80)


if __name__ == "__main__":
    main()
