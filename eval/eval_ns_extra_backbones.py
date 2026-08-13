from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_ns_fno import (
    DEFAULT_DATA_DIR,
    TEST_FILE,
    TRAIN_FILE,
    evaluate_one_model,
    flatten_summary_rows,
    paired_comparison,
    parse_int_list,
    parse_str_list,
    summarize_metric_dicts,
    write_csv,
)


DEFAULT_MODELS = ["uno", "uno_film", "transolver", "transolver_film"]
DEFAULT_PAIRS = [("uno", "uno_film"), ("transolver", "transolver_film")]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_SAVE_DIR = "./eval_outputs_ns_extra_backbones"


def main():
    parser = argparse.ArgumentParser(description="Evaluate the additional Navier--Stokes backbones.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--tag", default="")
    parser.add_argument("--model-root", default=".")
    parser.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-mat", action="store_true")
    args = parser.parse_args()

    models = parse_str_list(args.models)
    seeds = parse_int_list(args.seeds)
    data_dir = Path(args.data_dir)
    train_path = data_dir / TRAIN_FILE
    test_path = data_dir / TEST_FILE
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"Expected {train_path} and {test_path}")

    train_data = loadmat(train_path)
    test_data = loadmat(test_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = ("Mean Rel-L2", "Final Rel-L2")
    seedwise = []

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
            )
            row = {"model": model_name, "seed": seed, "tag": args.tag}
            row.update({key: metrics[key] for key in metric_keys})
            seedwise.append(row)

    summary_by_model = {}
    for model_name in models:
        rows = [row for row in seedwise if row["model"] == model_name]
        summary_by_model[model_name] = {
            "num_seeds": len(rows),
            "seeds": [int(row["seed"]) for row in rows],
            "metrics": summarize_metric_dicts(rows, metric_keys),
        }
    paired = paired_comparison(seedwise, DEFAULT_PAIRS, metric_keys)

    write_csv(save_dir / "extra_backbones_seedwise.csv", seedwise)
    write_csv(
        save_dir / "extra_backbones_summary_by_model.csv",
        flatten_summary_rows(summary_by_model, metric_keys),
    )
    write_csv(save_dir / "extra_backbones_paired_summary.csv", paired)

    for model_name, item in summary_by_model.items():
        print(f"\n{model_name}, seeds={item['seeds']}")
        for key in metric_keys:
            values = item["metrics"][key]
            print(f"  {key}: mean={values['mean']:.6e}, std={values['std']:.6e}")
    print(f"\nSaved summaries to {save_dir}")


if __name__ == "__main__":
    main()
