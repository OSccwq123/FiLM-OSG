#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="eval_outputs_evolution_time_sensitivity")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--run-prefix", default="burgers_original_seed")
    parser.add_argument("--run-suffix", default="_layerwise")
    parser.add_argument("--title", default="Original Burgers")
    parser.add_argument("--retained-modes", type=int, default=10)
    parser.add_argument("--radial-band", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="eval_outputs_evolution_time_sensitivity/burgers_original_5seed_summary",
    )
    args = parser.parse_args()

    root = Path(args.root)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    spectrum_rows = []
    for seed in seeds:
        run_dir = root / f"{args.run_prefix}{seed}{args.run_suffix}"
        for row in read_csv(run_dir / "evolution_time_sensitivity_metrics.csv"):
            row["seed"] = seed
            metric_rows.append(row)
        for row in read_csv(run_dir / "evolution_time_sensitivity_spectrum.csv"):
            row["seed"] = seed
            spectrum_rows.append(row)

    metric_names = [
        "sensitivity_rms",
        "fd_relative_error",
        "zero_mode_fraction_full",
        "nonzero_fraction_retained",
        "high_fraction_retained",
        "centroid_full",
        "centroid_retained",
    ]
    stages = ["block0_preactivation", "decoder"]
    models = ["Input-concatenation OSG-FNO", "FiLM-OSG-FNO"]
    summary = []
    for stage in stages:
        for model in models:
            selected = [row for row in metric_rows if row["stage"] == stage and row["model"] == model]
            item = {"stage": stage, "model": model, "seeds": seeds}
            for metric in metric_names:
                values = np.asarray([float(row[metric]) for row in selected])
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_std"] = float(values.std(ddof=0))
                item[f"{metric}_min"] = float(values.min())
                item[f"{metric}_max"] = float(values.max())
            summary.append(item)

    with (out_dir / "evolution_time_sensitivity_5seed_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[key for key in summary[0] if key != "seeds"])
        writer.writeheader()
        for row in summary:
            writer.writerow({key: value for key, value in row.items() if key != "seeds"})
    (out_dir / "evolution_time_sensitivity_5seed_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    colors = {"Input-concatenation OSG-FNO": "#4C78A8", "FiLM-OSG-FNO": "#D1495B"}
    titles = {"block0_preactivation": "First block, before block activation", "decoder": "Final decoder increment"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), sharey=True)
    for ax, stage in zip(axes, stages):
        for model in models:
            selected = [row for row in spectrum_rows if row["stage"] == stage]
            wavenumbers = sorted({int(row["wavenumber"]) for row in selected})
            values = np.asarray(
                [
                    [
                        float(next(row[model] for row in selected if int(row["seed"]) == seed and int(row["wavenumber"]) == k))
                        for k in wavenumbers
                    ]
                    for seed in seeds
                ]
            )
            mean = values.mean(axis=0)
            lower = values.min(axis=0)
            upper = values.max(axis=0)
            color = colors[model]
            ax.semilogy(wavenumbers, np.maximum(mean, 1e-12), marker="o", ms=3, lw=1.8, color=color, label=model)
            ax.fill_between(wavenumbers, np.maximum(lower, 1e-12), np.maximum(upper, 1e-12), color=color, alpha=0.14)
        ax.axvspan(
            -0.5,
            args.retained_modes - 0.5,
            color="#D9E6F2",
            alpha=0.35,
            label=(
                rf"low radial band $|k|<{args.retained_modes}$"
                if args.radial_band
                else "retained band"
            ),
        )
        ax.set_xlabel("Wavenumber $k$")
        ax.set_title(titles[stage])
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Normalized evolution-time sensitivity energy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    fig.suptitle(f"{args.title}: layerwise evolution-time sensitivity (five seeds)", y=1.13)
    fig.tight_layout()
    fig.savefig(out_dir / "evolution_time_sensitivity_layerwise_5seed.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "evolution_time_sensitivity_layerwise_5seed.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
