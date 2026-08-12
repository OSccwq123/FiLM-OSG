#!/usr/bin/env python3
"""Summarize lag-sensitivity spectral energy and effective rank.

The CSV field names retain ``nonzero_fraction`` to match the evaluation output.
The reported quantity is an energy fraction: the
sum of lag-sensitivity spectral energy outside the zero mode, normalized by the
corresponding full or selected-band energy.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev


BENCHMARKS = {
    "Burgers": {
        "run_prefix": "burgers_original_seed",
        "run_suffix": "_layerwise",
        "band_limit": 10,
        "band_label": "retained/low band k<10",
    },
    "AD": {
        "run_prefix": "ad_seed",
        "run_suffix": "_layerwise",
        "band_limit": 12,
        "band_label": "conservative low radial band |k|<12",
    },
    "NS": {
        "run_prefix": "ns_seed",
        "run_suffix": "_layerwise",
        "band_limit": 12,
        "band_label": "conservative low radial band |k|<12",
    },
}

MODELS = ["Direct-lag OSG-FNO", "FiLM-OSG-FNO"]
STAGES = ["block0_preactivation", "decoder"]
STAGE_LABELS = {
    "block0_preactivation": "First-block output before terminal activation",
    "decoder": "Final decoder increment",
}


def read_spectrum(path: Path, stage: str, model: str) -> tuple[list[int], list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["stage"] == stage]
    rows.sort(key=lambda row: int(row["wavenumber"]))
    return [int(row["wavenumber"]) for row in rows], [float(row[model]) for row in rows]


def effective_rank(values: list[float]) -> float:
    total = sum(values)
    if total <= 0.0:
        return float("nan")
    entropy = 0.0
    for value in values:
        if value <= 0.0:
            continue
        p = value / total
        entropy -= p * math.log(p)
    return math.exp(entropy)


def safe_fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("nan")


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if abs(value) >= 1e3 or (abs(value) < 1e-3 and value != 0.0):
        return f"{value:.3e}"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scalar lag-sensitivity mechanism metrics.")
    parser.add_argument("--root", default="eval_outputs_lag_sensitivity")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out-dir", default="eval_outputs_lag_sensitivity/scalar_summary")
    args = parser.parse_args()

    root = Path(args.root)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]
    seed_text = ",".join(str(seed) for seed in seeds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for benchmark, cfg in BENCHMARKS.items():
        for stage in STAGES:
            for model in MODELS:
                per_seed: dict[str, list[float]] = {
                    "effective_rank": [],
                    "nonzero_fraction_full": [],
                    "low_band_fraction_full": [],
                    "nonzero_fraction_low_band": [],
                    "outside_low_band_fraction": [],
                    "high_fraction_existing_check": [],
                    "centroid_existing_check": [],
                }
                for seed in seeds:
                    run_dir = root / f"{cfg['run_prefix']}{seed}{cfg['run_suffix']}"
                    spectrum_path = run_dir / "lag_sensitivity_spectrum.csv"
                    wavenumbers, values = read_spectrum(spectrum_path, stage, model)
                    total = sum(values)
                    zero = sum(value for k, value in zip(wavenumbers, values) if k == 0)
                    low = sum(value for k, value in zip(wavenumbers, values) if k < cfg["band_limit"])
                    low_nonzero = sum(
                        value for k, value in zip(wavenumbers, values) if 0 < k < cfg["band_limit"]
                    )
                    outside_low = total - low
                    centroid = safe_fraction(
                        sum(k * value for k, value in zip(wavenumbers, values)),
                        total,
                    )
                    high = safe_fraction(outside_low, total)

                    per_seed["effective_rank"].append(effective_rank(values))
                    per_seed["nonzero_fraction_full"].append(safe_fraction(total - zero, total))
                    per_seed["low_band_fraction_full"].append(safe_fraction(low, total))
                    per_seed["nonzero_fraction_low_band"].append(safe_fraction(low_nonzero, low))
                    per_seed["outside_low_band_fraction"].append(safe_fraction(outside_low, total))
                    per_seed["high_fraction_existing_check"].append(high)
                    per_seed["centroid_existing_check"].append(centroid)

                out_row: dict[str, object] = {
                    "benchmark": benchmark,
                    "stage": stage,
                    "stage_label": STAGE_LABELS[stage],
                    "model": model,
                    "seeds": ",".join(str(seed) for seed in seeds),
                    "band_label": cfg["band_label"],
                }
                for metric, values in per_seed.items():
                    stats = summarize(values)
                    for name, value in stats.items():
                        out_row[f"{metric}_{name}"] = value
                rows.append(out_row)

    csv_path = out_dir / "lag_sensitivity_scalar_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_dir / "lag_sensitivity_scalar_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Lag-Sensitivity Scalar Mechanism Summary\n\n")
        handle.write(
            f"Values are population mean +/- standard deviation over seeds {{{seed_text}}}. "
            "For AD and NS, |k|<12 is used only as a conservative low radial band.\n\n"
        )
        handle.write(
            "| Benchmark | Stage | Model | Nonzero energy (full) | "
            "Nonzero-band energy | Effective rank | Outside low band |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['benchmark']} | {row['stage_label']} | {row['model']} "
                f"| {fmt(row['nonzero_fraction_full_mean'])} +/- {fmt(row['nonzero_fraction_full_std'])} "
                f"| {fmt(row['nonzero_fraction_low_band_mean'])} +/- {fmt(row['nonzero_fraction_low_band_std'])} "
                f"| {fmt(row['effective_rank_mean'])} +/- {fmt(row['effective_rank_std'])} "
                f"| {fmt(row['outside_low_band_fraction_mean'])} +/- {fmt(row['outside_low_band_fraction_std'])} |\n"
            )

    tex_path = out_dir / "lag_sensitivity_scalar_table_snippet.tex"
    with tex_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"% Compact scalar mechanism table snippet. Values are mean $\\pm$ population std over seeds {{{seed_text}}}.\n"
        )
        handle.write("\\begin{tabular}{llllcc}\n")
        handle.write("\\toprule\n")
        handle.write("Benchmark & Stage & Model & Band & Nonzero-band energy & Effective rank \\\\\n")
        handle.write("\\midrule\n")
        for row in rows:
            handle.write(
                f"{row['benchmark']} & {row['stage_label']} & {row['model']} & {row['band_label']} "
                f"& {fmt(row['nonzero_fraction_low_band_mean'])} $\\pm$ {fmt(row['nonzero_fraction_low_band_std'])} "
                f"& {fmt(row['effective_rank_mean'])} $\\pm$ {fmt(row['effective_rank_std'])} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")

    print(csv_path)
    print(md_path)
    print(tex_path)
    print()
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
