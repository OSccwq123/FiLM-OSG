#!/usr/bin/env python3
"""Create manuscript-facing PDF figures from completed FiLM-OSG experiment summaries.

The script intentionally consumes already evaluated CSV summaries. It does not
launch training or evaluation jobs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BURGERS = Path("eval_outputs_burgers_sharp_5seed_full/sharp_5seed_summary.csv")
DEFAULT_AD = Path("eval_outputs_convdiff_ad_formal7_5seed/ad_formal7_5seed_compact.csv")
DEFAULT_NS = Path("eval_outputs_ns_selected_5seed/ns_selected_5seed_compact.csv")
DEFAULT_VT = Path("eval_outputs_vt_baselines_5seed/summary/vt_baselines_5seed_summary.csv")

BURGERS_LABELS = {
    "OSG-FNO": "OSG-FNO",
    "FiLM-OSG-FNO": "FiLM-OSG-FNO",
    "OSG-FNO + projection": "OSG-FNO + proj.",
    "FiLM-OSG-FNO + projection": "FiLM-OSG-FNO + proj.",
    "GL-OSG-FNO + projection": "GL-OSG-FNO + proj.",
    "GL-FiLM global_only + projection": "GL-FiLM global + proj.",
    "GL-FiLM branchwise + projection": "GL-FiLM branch + proj.",
    "GL-FiLM global_only + HF/HF-SG + projection": "GL-FiLM global + HF-SG + proj.",
    "GL-FiLM global_only + HF-data + projection": "GL-FiLM global + HF-data + proj.",
}
BURGERS_ORDER = [
    "OSG-FNO",
    "FiLM-OSG-FNO",
    "FiLM-OSG-FNO + projection",
    "GL-OSG-FNO + projection",
    "GL-FiLM global_only + projection",
    "GL-FiLM branchwise + projection",
    "GL-FiLM global_only + HF/HF-SG + projection",
]
AD_ORDER = ["fno_proj", "film_proj", "film_loglag_proj", "gl_direct_proj", "branchwise_loglag_proj", "globalonly_loglag_hfsg_proj"]
AD_LABELS = {
    "fno_proj": "OSG-FNO + proj.",
    "film_proj": "FiLM-OSG-FNO + proj.",
    "film_loglag_proj": "FiLM log-lag + proj.",
    "gl_direct_proj": "GL-OSG-FNO + proj.",
    "gl_direct_hfdata_proj": "GL-OSG + HF-data + proj.",
    "branchwise_loglag_proj": "GL-FiLM branch log-lag + proj.",
    "globalonly_loglag_hfsg_proj": "GL-FiLM global log-lag + HF-SG + proj.",
}
NS_ORDER = ["fno_proj", "film_proj", "branchwise_gamma_stable"]
NS_LABELS = {
    "fno_proj": "OSG-FNO + proj.",
    "film_proj": "FiLM-OSG-FNO + proj.",
    "branchwise_gamma_stable": "stable GL-FiLM + proj.",
}
VT_LABELS = {
    "vt_fno": "VT-FNO",
    "vt_fno_film": "VT-FiLM-FNO",
}


COLORS = {
    "base": "#4C78A8",
    "film": "#F58518",
    "projection": "#54A24B",
    "gl": "#B279A2",
    "vt": "#E45756",
    "neutral": "#6B7280",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(row: dict[str, str], key: str, default=np.nan) -> float:
    val = row.get(key, "")
    if val in (None, ""):
        return float(default)
    return float(val)


def row_map(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {r[key]: r for r in rows}


def style_axis(ax, ylabel=None, log=True):
    if log:
        ax.set_yscale("log")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def bar_panel(ax, labels, means, stds=None, colors=None, ylabel=None, title=None, log=True):
    x = np.arange(len(labels))
    colors = colors or [COLORS["neutral"]] * len(labels)
    yerr = None if stds is None else np.asarray(stds)
    ax.bar(x, means, yerr=yerr, capsize=3, color=colors, edgecolor="#222222", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8)
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    style_axis(ax, ylabel=ylabel, log=log)


def pick_color(label: str) -> str:
    low = label.lower()
    if "vt" in low:
        return COLORS["vt"]
    if "gl" in low:
        return COLORS["gl"]
    if "proj" in low:
        return COLORS["projection"]
    if "film" in low:
        return COLORS["film"]
    return COLORS["base"]


def burgers_panels(rows: list[dict[str, str]], vt_rows: list[dict[str, str]], out_dir: Path):
    by_label = row_map(rows, "label")
    vt = [r for r in vt_rows if r.get("benchmark") == "Burgers sharp-front"]
    vt_by_model = row_map(vt, "model")

    selected = [k for k in BURGERS_ORDER if k in by_label]
    labels = [BURGERS_LABELS.get(k, k) for k in selected]
    colors = [pick_color(x) for x in labels]

    metrics = [
        ("Mean Rel-L2", "Mean Rel-L2", True),
        ("HF Rel-L2", "High-frequency Rel-L2", True),
        ("TV Error", "TV error", True),
        ("Shock Loc Error", "Shock location error", True),
        ("Overshoot", "Overshoot", True),
        ("Mean Drift Abs Max", "Mean-drift max", True),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))
    for ax, (metric, ylabel, log) in zip(axes.ravel(), metrics):
        means = [to_float(by_label[k], f"{metric}_mean") for k in selected]
        stds = [to_float(by_label[k], f"{metric}_std", 0.0) for k in selected]
        bar_panel(ax, labels, means, stds, colors, ylabel=ylabel, title=metric, log=log)
    fig.suptitle("Sharp-front Burgers: local-structure and conservation diagnostics", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_dir / "fig_burgers_sharp_metrics.pdf")
    fig.savefig(out_dir / "fig_burgers_sharp_metrics.png", dpi=180)
    plt.close(fig)

    comparison = selected[:3] + ["GL-FiLM branchwise + projection"]
    comparison = [k for k in comparison if k in by_label]
    comp_labels = [BURGERS_LABELS.get(k, k) for k in comparison]
    comp_means = [to_float(by_label[k], "Mean Rel-L2_mean") for k in comparison]
    comp_stds = [to_float(by_label[k], "Mean Rel-L2_std", 0.0) for k in comparison]
    vt_labels = []
    vt_means = []
    vt_stds = []
    for m in ["vt_fno", "vt_fno_film"]:
        if m in vt_by_model:
            vt_labels.append(VT_LABELS[m])
            vt_means.append(to_float(vt_by_model[m], "Mean Rel-L2_mean"))
            vt_stds.append(to_float(vt_by_model[m], "Mean Rel-L2_std", 0.0))
    all_labels = comp_labels + vt_labels
    all_means = comp_means + vt_means
    all_stds = comp_stds + vt_stds
    fig, ax = plt.subplots(figsize=(8.3, 4.4))
    bar_panel(ax, all_labels, all_means, all_stds, [pick_color(x) for x in all_labels], ylabel="Mean Rel-L2", title="Sharp-front Burgers with external variable-time baselines", log=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_burgers_vt_baseline_comparison.pdf")
    fig.savefig(out_dir / "fig_burgers_vt_baseline_comparison.png", dpi=180)
    plt.close(fig)


def ad_vt_panel(ad_rows: list[dict[str, str]], vt_rows: list[dict[str, str]], out_dir: Path):
    by_combo = row_map(ad_rows, "combo")
    vt = [r for r in vt_rows if r.get("benchmark") == "Advection-diffusion"]
    vt_by_model = row_map(vt, "model")

    combos = [k for k in AD_ORDER if k in by_combo]
    labels = [AD_LABELS.get(k, k) for k in combos]
    means = [to_float(by_combo[k], "Mean Rel-L2_mean") for k in combos]
    stds = [to_float(by_combo[k], "Mean Rel-L2_std", 0.0) for k in combos]
    for m in ["vt_fno", "vt_fno_film"]:
        if m in vt_by_model:
            labels.append(VT_LABELS[m])
            means.append(to_float(vt_by_model[m], "Mean Rel-L2_mean"))
            stds.append(to_float(vt_by_model[m], "Mean Rel-L2_std", 0.0))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    bar_panel(axes[0], labels, means, stds, [pick_color(x) for x in labels], ylabel="Mean Rel-L2", title="Advection-diffusion: accuracy", log=True)

    drift_labels = [AD_LABELS.get(k, k) for k in combos]
    drift_means = [to_float(by_combo[k], "Mean Drift Max_mean") for k in combos]
    drift_stds = [0.0 for _ in combos]
    for m in ["vt_fno", "vt_fno_film"]:
        if m in vt_by_model:
            drift_labels.append(VT_LABELS[m])
            drift_means.append(to_float(vt_by_model[m], "Mean Drift Max_mean"))
            drift_stds.append(to_float(vt_by_model[m], "Mean Drift Max_std", 0.0))
    bar_panel(axes[1], drift_labels, drift_means, drift_stds, [pick_color(x) for x in drift_labels], ylabel="Mean drift max", title="Projection closes the mean-drift gap", log=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ad_vt_projection.pdf")
    fig.savefig(out_dir / "fig_ad_vt_projection.png", dpi=180)
    plt.close(fig)


def cross_benchmark_projection(burgers_rows, ad_rows, ns_rows, vt_rows, out_dir: Path):
    b = row_map(burgers_rows, "label")
    a = row_map(ad_rows, "combo")
    n = row_map(ns_rows, "combo")
    vt_bench = {(r.get("benchmark"), r.get("model")): r for r in vt_rows}

    groups = [
        ("Burgers\nFiLM", b.get("FiLM-OSG-FNO"), "Mean Drift Abs Max_mean"),
        ("Burgers\nFiLM + proj.", b.get("FiLM-OSG-FNO + projection"), "Mean Drift Abs Max_mean"),
        ("AD\nVT-FiLM", vt_bench.get(("Advection-diffusion", "vt_fno_film")), "Mean Drift Max_mean"),
        ("AD\nFiLM + proj.", a.get("film_loglag_proj"), "Mean Drift Max_mean"),
        ("NS\nFNO + proj.", n.get("fno_proj"), "Mean Drift Max_mean"),
        ("NS\nFiLM + proj.", n.get("film_proj"), "Mean Drift Max_mean"),
    ]
    labels = []
    vals = []
    for label, row, key in groups:
        if row is None:
            continue
        labels.append(label)
        vals.append(to_float(row, key))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bar_panel(ax, labels, vals, None, [pick_color(x) for x in labels], ylabel="Mean-drift max", title="Mean-conservation diagnostics across benchmarks", log=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_projection_drift_across_benchmarks.pdf")
    fig.savefig(out_dir / "fig_projection_drift_across_benchmarks.png", dpi=180)
    plt.close(fig)


def ns_summary(ns_rows: list[dict[str, str]], out_dir: Path):
    by_combo = row_map(ns_rows, "combo")
    combos = [k for k in NS_ORDER if k in by_combo]
    labels = [NS_LABELS[k] for k in combos]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for ax, metric, title in [
        (axes[0], "Mean Rel-L2", "Mean Rel-L2"),
        (axes[1], "Final Rel-L2", "Final Rel-L2"),
        (axes[2], "Spectrum Error", "Spectrum error"),
    ]:
        means = [to_float(by_combo[k], f"{metric}_mean") for k in combos]
        stds = [to_float(by_combo[k], f"{metric}_std", 0.0) for k in combos]
        bar_panel(ax, labels, means, stds, [pick_color(x) for x in labels], ylabel=title, title=title, log=True)
    fig.suptitle("Navier-Stokes: GL is reported as an ablation, not the main gain", fontsize=12)
    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    fig.savefig(out_dir / "fig_ns_ablation_summary.pdf")
    fig.savefig(out_dir / "fig_ns_ablation_summary.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate paper-facing PDF figures from completed experiment summaries.")
    parser.add_argument("--burgers", type=Path, default=DEFAULT_BURGERS)
    parser.add_argument("--ad", type=Path, default=DEFAULT_AD)
    parser.add_argument("--ns", type=Path, default=DEFAULT_NS)
    parser.add_argument("--vt", type=Path, default=DEFAULT_VT)
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    burgers_rows = read_csv(args.burgers)
    ad_rows = read_csv(args.ad)
    ns_rows = read_csv(args.ns)
    vt_rows = read_csv(args.vt)

    burgers_panels(burgers_rows, vt_rows, args.out_dir)
    ad_vt_panel(ad_rows, vt_rows, args.out_dir)
    cross_benchmark_projection(burgers_rows, ad_rows, ns_rows, vt_rows, args.out_dir)
    ns_summary(ns_rows, args.out_dir)

    outputs = sorted(args.out_dir.glob("fig_*.pdf"))
    print("Generated PDFs:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
