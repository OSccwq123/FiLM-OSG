#!/usr/bin/env python3
"""Plot a paired OSG-FNO and FiLM-OSG-FNO Burgers rollout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_prediction(checkpoint, x0, dt, device):
    model = torch.load(checkpoint, map_location=device, weights_only=False)
    model.eval()
    with torch.no_grad():
        prediction = model.predict(x0, dt, device)
    return np.asarray(prediction)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-model", type=Path, required=True)
    parser.add_argument("--film-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/BurgersOSG_test.mat"))
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--steps", default="6,20")
    parser.add_argument("--out", type=Path, default=Path("burgers_rollout.pdf"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    data = loadmat(args.data)
    trajectories = data["trajectories"].astype(np.float32)
    lags = data["dt"].astype(np.float32)
    coordinates = data["coordinates"].reshape(-1)

    sample = args.sample % trajectories.shape[0]
    steps = [int(value) for value in args.steps.split(",") if value]
    if len(steps) != 2 or min(steps) < 1 or max(steps) > lags.shape[1]:
        raise ValueError("--steps must contain two valid rollout steps.")

    x0 = trajectories[sample : sample + 1, ..., 0]
    dt = lags[sample : sample + 1, : max(steps)]
    truth = trajectories[sample]
    direct = load_prediction(args.direct_model, x0, dt, args.device)[0]
    film = load_prediction(args.film_model, x0, dt, args.device)[0]
    times = np.concatenate(([0.0], np.cumsum(dt[0])))

    predictions = (("Direct-lag OSG-FNO", direct), ("FiLM-OSG-FNO", film))
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.6), sharex=True, sharey=True)
    for row, step in enumerate(steps):
        for col, (title, prediction) in enumerate(predictions):
            ax = axes[row, col]
            ax.plot(coordinates, truth[:, 0, step], color="black", lw=1.7, label="Reference")
            ax.plot(coordinates, prediction[:, 0, step], color="#D55E00", lw=1.5, ls="--", label="Prediction")
            ax.set_title(f"{title}, $t={times[step]:.1f}$", fontsize=10)
            ax.grid(alpha=0.2, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if col == 0:
                ax.set_ylabel("$u(x,t)$")
            if row == len(steps) - 1:
                ax.set_xlabel("$x$")

    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
