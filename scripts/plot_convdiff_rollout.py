#!/usr/bin/env python3
"""Plot an advection--diffusion rollout from saved evaluation predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def prediction_file(path):
    data = loadmat(path)
    return data["prediction"].astype(np.float64), data["truth"].astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct", type=Path, required=True, help="Saved input-concatenation prediction .mat file."
    )
    parser.add_argument("--film", type=Path, required=True, help="Saved FiLM prediction .mat file.")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--steps", default="10,20")
    parser.add_argument("--output", type=Path, default=Path("convdiff_rollout.pdf"))
    args = parser.parse_args()

    direct, truth = prediction_file(args.direct)
    film, film_truth = prediction_file(args.film)
    if direct.shape != truth.shape or film.shape != truth.shape:
        raise ValueError("Prediction and reference arrays must have the same shape.")
    if not np.allclose(truth, film_truth):
        raise ValueError("The two prediction files do not contain the same reference trajectories.")

    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    if not steps or min(steps) < 0 or max(steps) >= truth.shape[-1]:
        raise ValueError(f"Steps must lie between 0 and {truth.shape[-1] - 1}.")
    if not 0 <= args.case < truth.shape[0]:
        raise ValueError(f"Case must lie between 0 and {truth.shape[0] - 1}.")

    fields = (truth, direct, film)
    titles = ("Reference", "Input-concatenation OSG-FNO", "FiLM-OSG-FNO")
    fig, axes = plt.subplots(len(steps), 3, figsize=(9.2, 2.8 * len(steps)), squeeze=False)

    for row, step in enumerate(steps):
        reference = truth[args.case, :, :, 0, step]
        limit = float(np.max(np.abs(reference)))
        for col, (array, title) in enumerate(zip(fields, titles)):
            ax = axes[row, col]
            image = ax.imshow(
                array[args.case, :, :, 0, step],
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(f"step {step}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(image, ax=axes[row, :], fraction=0.018, pad=0.015)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
