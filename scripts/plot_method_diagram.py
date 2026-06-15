#!/usr/bin/env python3
"""Draw a clean FiLM-OSG centered method schematic.

The diagram keeps the main FiLM-OSG path visually primary and shows GL as an
optional sharp-front extension without dense crossing arrows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


COLORS = {
    "input": "#EAF2FF",
    "state": "#EEF4FA",
    "lag": "#FFF4D8",
    "film": "#FFE8B3",
    "global": "#DFF3E4",
    "local": "#F9E5EF",
    "project": "#E4F5F0",
    "loss": "#FDECEA",
    "baseline": "#F1F3F4",
    "frame": "#F7FAFC",
    "edge": "#263238",
    "text": "#17212B",
    "muted": "#596873",
    "accent": "#2F6B3F",
    "brown": "#8A5A00",
}


def box(ax, xy, w, h, text, fc, fontsize=10.0, lw=1.15, dashed=False):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=lw,
        linestyle="--" if dashed else "-",
        edgecolor=COLORS["edge"],
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, color=None, lw=1.25, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color or COLORS["edge"],
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arr)
    return arr


METHOD_CAPTION = (
    "Schematic of the FiLM-OSG architecture. The queried lag is encoded into a "
    "latent time code and mapped to FiLM coefficients that modulate the global "
    "FNO backbone inside an outer-increment semigroup update. The mean-zero "
    "projection is a lightweight structure-preserving diagnostic for periodic "
    "mean-preserving benchmarks and is not part of the default update. The dashed inset shows the optional GL-FiLM "
    "extension used for sharp-front Burgers experiments, where a lag-conditioned "
    "local residual correction is coupled back into the decoded increment. "
    "Training objectives, including semigroup consistency and optional "
    "high-frequency penalties, are specified in the text rather than encoded as "
    "architecture modules."
)


def draw_method(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.6, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "FiLM-OSG",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color=COLORS["text"],
    )
    ax.text(
        0.5,
        0.915,
        "Outer-increment semigroup update with lag-conditioned FiLM modulation",
        ha="center",
        va="top",
        fontsize=9.5,
        color=COLORS["muted"],
    )

    # Core path boxes: keep the main FiLM-OSG route as a single reading line.
    box(ax, (0.025, 0.60), 0.105, 0.125, "state\n$u(t)$", COLORS["input"], 11.5)
    box(ax, (0.165, 0.60), 0.105, 0.125, "lift\n$h_0$", COLORS["state"], 10.7)
    box(ax, (0.315, 0.60), 0.165, 0.125, "FiLM-ready global\noperator blocks\n(FNO backbone)", COLORS["global"], 9.2)
    box(ax, (0.535, 0.60), 0.13, 0.125, "decoded\nincrement\n$\\phi_\\theta(u,\\delta)$", COLORS["state"], 8.9)
    box(ax, (0.700, 0.60), 0.13, 0.125, "outer\nincrement\n$u+\\Delta\\phi_\\theta$", COLORS["project"], 8.7)
    box(ax, (0.700, 0.39), 0.13, 0.105, "optional\nprojection\n$\\Pi_0$", COLORS["project"], 8.2, dashed=True)
    box(ax, (0.850, 0.60), 0.12, 0.125, "prediction\n$\\widehat u(t+\\Delta)$", COLORS["input"], 10.5)

    # Lag and FiLM path.
    box(ax, (0.025, 0.385), 0.105, 0.12, "query lag\n$\\Delta$", COLORS["lag"], 11.2)
    box(ax, (0.165, 0.385), 0.105, 0.12, "lag encoder\n$\\delta(\\Delta)$", COLORS["lag"], 9.8)
    box(ax, (0.315, 0.385), 0.165, 0.12, "FiLM coefficients\n$\\gamma^{(r)}(\\delta),\\beta^{(r)}(\\delta)$", COLORS["film"], 8.8)

    # Core arrows.
    arrow(ax, (0.13, 0.662), (0.165, 0.662))
    arrow(ax, (0.27, 0.662), (0.315, 0.662))
    arrow(ax, (0.48, 0.662), (0.535, 0.662))
    arrow(ax, (0.665, 0.662), (0.700, 0.662))
    arrow(ax, (0.830, 0.662), (0.850, 0.662))
    opt1 = arrow(ax, (0.665, 0.62), (0.700, 0.465), color="#6B7280", lw=0.85, style="->", rad=-0.18)
    opt1.set_linestyle("--")
    opt2 = arrow(ax, (0.830, 0.465), (0.700, 0.62), color="#6B7280", lw=0.85, style="->", rad=-0.18)
    opt2.set_linestyle("--")
    arrow(ax, (0.13, 0.445), (0.165, 0.445))
    arrow(ax, (0.27, 0.445), (0.315, 0.445))

    # Clean FiLM modulation arrow.
    arrow(ax, (0.397, 0.505), (0.397, 0.60), color=COLORS["brown"], lw=1.3)
    ax.text(
        0.397,
        0.555,
        "modulate",
        ha="center",
        va="center",
        fontsize=8.4,
        color=COLORS["brown"],
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )

    # Optional GL extension frame.
    frame = FancyBboxPatch(
        (0.19, 0.095), 0.535, 0.205,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.0,
        linestyle="--",
        edgecolor=COLORS["accent"],
        facecolor=COLORS["frame"],
        alpha=0.55,
    )
    ax.add_patch(frame)
    ax.text(0.205, 0.285, "Optional sharp-front GL-FiLM extension", ha="left", va="center",
            fontsize=10.2, fontweight="bold", color=COLORS["accent"])
    box(ax, (0.215, 0.13), 0.15, 0.105, "local residual\nbranch", COLORS["local"], 9.0)
    box(ax, (0.405, 0.13), 0.13, 0.105, "local FiLM\nfrom $\\delta$", COLORS["film"], 8.8)
    box(ax, (0.575, 0.13), 0.13, 0.105, "global-local\ncoupling", "#EDE7F6", 8.8)
    arrow(ax, (0.365, 0.182), (0.405, 0.182), color=COLORS["accent"])
    arrow(ax, (0.535, 0.182), (0.575, 0.182), color=COLORS["accent"])
    # Single GL return connector. This avoids implying a dense set of couplings.
    arrow(ax, (0.64, 0.235), (0.60, 0.60), color=COLORS["accent"], lw=1.05, style="->", rad=-0.16)
    ax.text(
        0.665,
        0.365,
        "local\ncorrection",
        ha="center",
        va="center",
        fontsize=8.0,
        color=COLORS["accent"],
        linespacing=1.05,
        bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    # External variable-time baselines, shown outside the OSG path.
    box(ax, (0.025, 0.11), 0.13, 0.115, "VT baselines\nVT-FNO / VT-FiLM\n(no OSG loss)", COLORS["baseline"], 7.8)
    arrow(ax, (0.078, 0.385), (0.078, 0.225), color="#6B7280", lw=0.95)

    ax.text(
        0.035,
        0.035,
        "Core FiLM-OSG uses the unprojected outer-increment update; projection is an optional mean-preservation diagnostic. GL-FiLM is a targeted sharp-front extension.",
        fontsize=8.8,
        color=COLORS["muted"],
    )

    for stem in ["fig_method_gl_film_osg", "fig_method_film_osg_optional_gl_clean_20260605"]:
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    caption_path = out_dir / "figure_captions_gl_revision.txt"
    existing = caption_path.read_text(encoding="utf-8") if caption_path.exists() else ""
    marker = "Fig. method:"
    lines = [line for line in existing.splitlines() if not line.startswith(marker)]
    lines.append(f"{marker} {METHOD_CAPTION}")
    caption_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    plt.close(fig)
    print("Generated", out_dir / "fig_method_gl_film_osg.pdf")
    print("Generated", out_dir / "fig_method_film_osg_optional_gl_clean_20260605.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures"))
    args = parser.parse_args()
    draw_method(args.out_dir)


if __name__ == "__main__":
    main()
