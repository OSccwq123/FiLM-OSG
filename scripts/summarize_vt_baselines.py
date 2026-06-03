#!/usr/bin/env python3
"""Summarize completed VT-FNO and VT-FiLM-FNO seedwise evaluations.

The VT baseline training tags used in the manuscript experiments include the seed
id, so the standard evaluation scripts are usually run one seed at a time. This
utility collects those completed `*_seedwise.csv` files and writes compact
5-seed summaries. It does not launch training or evaluation jobs.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


BURGERS_METRICS = [
    "MAE",
    "Rel-L1",
    "Mean Rel-L2",
    "Final Rel-L2",
    "HF Rel-L2",
    "TV Error",
    "Shock Loc Error",
    "Overshoot",
]
AD_METRICS = [
    "MAE",
    "Rel-L1",
    "Mean Rel-L2",
    "Final Rel-L2",
    "Mean Drift Max",
    "HF Rel-L2",
    "Spectrum Error",
    "Final Spectrum Error",
]


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="") as fh:
        return next(csv.DictReader(fh))


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def collect_rows(root: Path, benchmark: str, model: str, seeds: list[int]) -> list[dict[str, str]]:
    rows = []
    for seed in seeds:
        if benchmark == "burgers":
            path = root / "burgers" / f"{model}_seed{seed}" / "burgers_fno_seedwise.csv"
            label = "Burgers sharp-front"
        elif benchmark == "ad":
            path = root / "ad" / f"{model}_seed{seed}" / "convdiff_fno_seedwise.csv"
            label = "Advection-diffusion"
        else:
            raise ValueError(f"unknown benchmark: {benchmark}")
        if not path.exists():
            raise FileNotFoundError(path)
        row = read_one(path)
        row["benchmark"] = label
        row["seed"] = str(seed)
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]], metrics: list[str]) -> dict[str, object]:
    out: dict[str, object] = {
        "benchmark": rows[0]["benchmark"],
        "model": rows[0]["model"],
        "num_seeds": len(rows),
        "seeds": ",".join(row["seed"] for row in rows),
    }
    for metric in metrics:
        vals = [to_float(row.get(metric, "")) for row in rows]
        out[f"{metric}_mean"] = statistics.mean(vals)
        out[f"{metric}_std"] = statistics.pstdev(vals)
        out[f"{metric}_median"] = statistics.median(vals)
        out[f"{metric}_min"] = min(vals)
        out[f"{metric}_max"] = max(vals)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object) -> str:
    return f"{float(value):.3e}"


def write_markdown(path: Path, summary_rows: list[dict[str, object]]) -> None:
    md: list[str] = []
    md.append("# VT Baselines 5-Seed Summary\n\n")
    md.append("## Burgers sharp-front\n\n")
    md.append("| Model | Mean Rel-L2 | Final Rel-L2 | HF Rel-L2 | TV | Shock loc | Overshoot |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for row in [r for r in summary_rows if r["benchmark"] == "Burgers sharp-front"]:
        md.append(
            f"| {row['model']} | {fmt(row['Mean Rel-L2_mean'])} +- {fmt(row['Mean Rel-L2_std'])} "
            f"| {fmt(row['Final Rel-L2_mean'])} | {fmt(row['HF Rel-L2_mean'])} "
            f"| {fmt(row['TV Error_mean'])} | {fmt(row['Shock Loc Error_mean'])} "
            f"| {fmt(row['Overshoot_mean'])} |\n"
        )
    md.append("\n## Advection-diffusion\n\n")
    md.append("| Model | Mean Rel-L2 | Final Rel-L2 | Spectrum Error | Mean Drift Max |\n")
    md.append("|---|---:|---:|---:|---:|\n")
    for row in [r for r in summary_rows if r["benchmark"] == "Advection-diffusion"]:
        md.append(
            f"| {row['model']} | {fmt(row['Mean Rel-L2_mean'])} +- {fmt(row['Mean Rel-L2_std'])} "
            f"| {fmt(row['Final Rel-L2_mean'])} | {fmt(row['Spectrum Error_mean'])} "
            f"| {fmt(row['Mean Drift Max_mean'])} |\n"
        )
    path.write_text("".join(md))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="eval_outputs_vt_baselines_5seed")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    out_dir = Path(args.out_dir) if args.out_dir else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    seedwise: list[dict[str, str]] = []
    summary: list[dict[str, object]] = []
    for model in ["vt_fno", "vt_fno_film"]:
        rows = collect_rows(root, "burgers", model, seeds)
        seedwise.extend(rows)
        summary.append(summarize(rows, BURGERS_METRICS))
        rows = collect_rows(root, "ad", model, seeds)
        seedwise.extend(rows)
        summary.append(summarize(rows, AD_METRICS))

    write_csv(out_dir / "vt_baselines_5seed_seedwise.csv", seedwise)
    write_csv(out_dir / "vt_baselines_5seed_summary.csv", summary)
    write_markdown(out_dir / "vt_baselines_5seed_summary.md", summary)

    print(out_dir / "vt_baselines_5seed_seedwise.csv")
    print(out_dir / "vt_baselines_5seed_summary.csv")
    print(out_dir / "vt_baselines_5seed_summary.md")


if __name__ == "__main__":
    main()
