#!/usr/bin/env python3
"""Draw the manuscript method schematic for the revised FiLM-OSG story."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


COLORS = {
    "input": "#E8F0FE",
    "lag": "#FFF3D6",
    "global": "#DFF3E4",
    "local": "#F8E6EF",
    "fusion": "#EDE7F6",
    "output": "#E6F4F1",
    "loss": "#FDECEA",
    "baseline": "#F1F3F4",
    "edge": "#253238",
    "text": "#17212B",
}


def box(ax, xy, w, h, text, fc, fontsize=10.5, lw=1.2):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=lw,
        edgecolor=COLORS["edge"],
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=COLORS["text"], linespacing=1.18)
    return patch


def arrow(ax, start, end, text=None, rad=0.0, color=None, lw=1.35, style="-|>"):
    arr = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color or COLORS["edge"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.025, text, ha="center", va="center", fontsize=8.5,
                color=color or COLORS["edge"])
    return arr


def draw_method(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Semigroup-aware global-local FiLM neural operator",
            ha="center", va="center", fontsize=16, fontweight="bold", color=COLORS["text"])
    ax.text(0.5, 0.925,
            "The queried lag modulates both the global spectral path and the local reconstruction path;\n"
            "the outer-increment form, projection, and high-frequency consistency keep the variable-time map tied to the OSG structure.",
            ha="center", va="top", fontsize=10.5, color="#41515F", linespacing=1.15)

    # Main GL-FiLM-OSG path.
    box(ax, (0.035, 0.58), 0.13, 0.15, "state\n$u(t)$", COLORS["input"], 12)
    box(ax, (0.035, 0.34), 0.13, 0.15, "query lag\n$\\Delta$", COLORS["lag"], 12)
    box(ax, (0.205, 0.58), 0.13, 0.15, "lift\n$h_0$", "#EEF4FA", 11)
    box(ax, (0.205, 0.34), 0.13, 0.15, "lag code\n$\\delta(\\Delta)$", COLORS["lag"], 11)

    box(ax, (0.395, 0.68), 0.16, 0.13, "global spectral path\nFNO modes", COLORS["global"], 10.5)
    box(ax, (0.395, 0.49), 0.16, 0.13, "local reconstruction\npooled conv correction", COLORS["local"], 10.5)
    box(ax, (0.395, 0.30), 0.16, 0.13, "FiLM lag modulation\n$\\gamma^g,\\beta^g,\\gamma^l,\\beta^l$", COLORS["lag"], 9.7)

    box(ax, (0.61, 0.56), 0.15, 0.16, "multiplicative\nglobal-local coupling", COLORS["fusion"], 10.5)
    box(ax, (0.79, 0.56), 0.15, 0.16, "outer increment\n$u + \\Delta\\,\\Pi_0\\phi_\\theta$", COLORS["output"], 10.5)
    box(ax, (0.79, 0.34), 0.15, 0.13, "mean-zero\nprojection $\\Pi_0$", COLORS["output"], 10.5)
    box(ax, (0.79, 0.15), 0.15, 0.13, "output\n$u(t+\\Delta)$", COLORS["input"], 12)

    arrow(ax, (0.165, 0.655), (0.205, 0.655))
    arrow(ax, (0.335, 0.655), (0.395, 0.745))
    arrow(ax, (0.335, 0.655), (0.395, 0.555))
    arrow(ax, (0.165, 0.415), (0.205, 0.415))
    arrow(ax, (0.335, 0.415), (0.395, 0.365))
    arrow(ax, (0.555, 0.745), (0.61, 0.64))
    arrow(ax, (0.555, 0.555), (0.61, 0.64))
    arrow(ax, (0.555, 0.365), (0.61, 0.64), text="FiLM", rad=-0.15, color="#8A5A00")
    arrow(ax, (0.76, 0.64), (0.79, 0.64))
    arrow(ax, (0.865, 0.56), (0.865, 0.47))
    arrow(ax, (0.865, 0.34), (0.865, 0.28))

    # Semigroup and HF constraints.
    box(ax, (0.395, 0.08), 0.17, 0.12, "semigroup consistency\n$\\Phi_{\\Delta_2}(\\Phi_{\\Delta_1}(u))\\approx\\Phi_{\\Delta_1+\\Delta_2}(u)$",
        COLORS["loss"], 8.8)
    box(ax, (0.60, 0.08), 0.17, 0.12, "high-frequency losses\n$L_{hf,data}+L_{hf,sg}$", COLORS["loss"], 9.5)

    # External baseline side branch.
    ax.text(0.055, 0.235, "External variable-time baseline", fontsize=10.5,
            fontweight="bold", color="#3C4043")
    box(ax, (0.035, 0.075), 0.30, 0.12,
        "VT-FNO / VT-FiLM-FNO\ndirect time-conditioned baseline\n(no semigroup loss / no projection)",
        COLORS["baseline"], 9.2)
    arrow(ax, (0.10, 0.34), (0.13, 0.195), rad=0.0, color="#6B7280", lw=1.0)

    ax.text(0.03, 0.018,
            "Representative fields/profiles go in figures; multi-seed errors and diagnostics go in tables.",
            fontsize=9.2, color="#5F6B75")

    pdf = out_dir / "fig_method_gl_film_osg.pdf"
    png = out_dir / "fig_method_gl_film_osg.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Generated", pdf)
    print("Generated", png)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()
    draw_method(args.out_dir)


if __name__ == "__main__":
    main()
