#!/usr/bin/env python3
"""Plot Navier--Stokes vorticity, error, and enstrophy from saved predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def load_prediction(path):
    data = loadmat(path)
    return data["prediction"].astype(np.float64), data["truth"].astype(np.float64)


def rel_l2_curve(prediction, truth, eps=1e-12):
    error = np.moveaxis(prediction - truth, -1, 0).reshape(prediction.shape[-1], -1)
    reference = np.moveaxis(truth, -1, 0).reshape(truth.shape[-1], -1)
    return np.linalg.norm(error, axis=1) / (np.linalg.norm(reference, axis=1) + eps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--film", type=Path, required=True)
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--steps", default="50,99")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/ns"))
    args = parser.parse_args()

    direct, truth = load_prediction(args.direct)
    film, film_truth = load_prediction(args.film)
    if direct.shape != truth.shape or film.shape != truth.shape:
        raise ValueError("Prediction and reference arrays must have the same shape.")
    if not np.allclose(truth, film_truth):
        raise ValueError("The two files do not contain the same reference trajectories.")
    if not 0 <= args.case < truth.shape[0]:
        raise ValueError(f"Case must lie between 0 and {truth.shape[0] - 1}.")
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    if not steps or min(steps) < 0 or max(steps) >= truth.shape[-1]:
        raise ValueError(f"Steps must lie between 0 and {truth.shape[-1] - 1}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = (truth, direct, film)
    titles = ("Reference", "Direct-lag OSG-FNO", "FiLM-OSG-FNO")
    fig, axes = plt.subplots(len(steps), 3, figsize=(9.2, 2.8 * len(steps)), squeeze=False)
    for row, step in enumerate(steps):
        reference = truth[args.case, :, :, 0, step]
        limit = float(np.max(np.abs(reference)))
        for col, (array, title) in enumerate(zip(arrays, titles)):
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
    fig.savefig(args.output_dir / "ns_vorticity_compare.pdf", bbox_inches="tight")
    plt.close(fig)

    truth_case = truth[args.case, :, :, 0, :]
    direct_case = direct[args.case, :, :, 0, :]
    film_case = film[args.case, :, :, 0, :]
    times = np.arange(truth.shape[-1])

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(times, rel_l2_curve(direct_case, truth_case), label="Direct-lag OSG-FNO")
    ax.plot(times, rel_l2_curve(film_case, truth_case), label="FiLM-OSG-FNO")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.legend(frameon=False)
    fig.savefig(args.output_dir / "ns_relL2_curve_linear.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for array, label in zip((truth_case, direct_case, film_case), titles):
        enstrophy = np.mean(array**2, axis=(0, 1))
        ax.plot(times, enstrophy, label=label)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("discrete enstrophy")
    ax.legend(frameon=False)
    fig.savefig(args.output_dir / "ns_enstrophy_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Navier--Stokes figures to {args.output_dir}")


if __name__ == "__main__":
    main()
