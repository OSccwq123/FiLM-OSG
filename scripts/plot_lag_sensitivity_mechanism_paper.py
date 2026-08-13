#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEEDS = range(5)
STAGES = ("block0_preactivation", "decoder")
MODELS = ("Direct-lag OSG-FNO", "FiLM-OSG-FNO")
COLORS = {"Direct-lag OSG-FNO": "#356E9F", "FiLM-OSG-FNO": "#C9495B"}
BENCHMARKS = (
    ("Original Burgers", "burgers_original_seed", "_layerwise", 10),
    ("Advection--diffusion", "ad_seed", "_layerwise", 12),
    ("Navier--Stokes", "ns_seed", "_layerwise", 12),
)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_spectra(root: Path, prefix: str, suffix: str):
    values = {stage: {model: [] for model in MODELS} for stage in STAGES}
    wavenumbers = {}
    for seed in SEEDS:
        rows = read_rows(root / f"{prefix}{seed}{suffix}" / "lag_sensitivity_spectrum.csv")
        for stage in STAGES:
            selected = [row for row in rows if row["stage"] == stage]
            selected.sort(key=lambda row: int(row["wavenumber"]))
            wavenumbers[stage] = np.asarray([int(row["wavenumber"]) for row in selected])
            for model in MODELS:
                values[stage][model].append(np.asarray([float(row[model]) for row in selected]))
    return wavenumbers, {
        stage: {model: np.stack(seed_values) for model, seed_values in by_model.items()}
        for stage, by_model in values.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Plot the five-seed lag-sensitivity spectra.")
    parser.add_argument("--input-dir", type=Path, default=Path("eval_outputs_lag_sensitivity"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.transparent": False,
        }
    )

    fig, axes = plt.subplots(3, 2, figsize=(7.25, 7.85), sharey=True)
    for row_index, (name, prefix, suffix, band_end) in enumerate(BENCHMARKS):
        wavenumbers, spectra = load_spectra(args.input_dir, prefix, suffix)
        for column_index, stage in enumerate(STAGES):
            ax = axes[row_index, column_index]
            for model in MODELS:
                seed_values = spectra[stage][model]
                mean = seed_values.mean(axis=0)
                lower = seed_values.min(axis=0)
                upper = seed_values.max(axis=0)
                k = wavenumbers[stage]
                color = COLORS[model]
                ax.semilogy(
                    k,
                    np.maximum(mean, 1e-12),
                    color=color,
                    linewidth=1.55,
                    marker="o",
                    markersize=2.35,
                    markevery=max(1, len(k) // 16),
                    label=model,
                    zorder=3,
                )
                ax.fill_between(
                    k,
                    np.maximum(lower, 1e-12),
                    np.maximum(upper, 1e-12),
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                    zorder=2,
                )

            ax.axvspan(-0.5, band_end - 0.5, color="#D8E6F2", alpha=0.42, zorder=0)
            ax.grid(True, which="major", color="#B9B9B9", alpha=0.28, linewidth=0.55)
            ax.set_ylim(5e-13, 2.0)
            ax.set_xlim(-0.5, float(wavenumbers[stage][-1]) + 0.7)
            ax.set_xlabel(r"Wavenumber $k$" if name == "Original Burgers" else r"Radial wavenumber $|k|$")
            if column_index == 0:
                ax.set_ylabel(f"{name}\nNormalized sensitivity energy")
            if row_index == 0:
                ax.set_title(
                    "First-block output before terminal activation"
                    if stage == "block0_preactivation"
                    else "Decoder output (increment rate)"
                )

            band_text = (
                r"retained modes $k<10$"
                if name == "Original Burgers"
                else r"low radial band $|k|<12$"
            )
            ax.text(
                0.975,
                0.92,
                band_text,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.1,
                color="#40566B",
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.992))
    fig.subplots_adjust(left=0.135, right=0.992, bottom=0.062, top=0.905, hspace=0.43, wspace=0.15)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "lag_sensitivity_mechanism_5seed.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "lag_sensitivity_mechanism_5seed.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
