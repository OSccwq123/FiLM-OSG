#!/usr/bin/env python3
"""Generate variable-lag Burgers data with steep gradients."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def low_freq_signal(x: np.ndarray, rng: np.random.Generator, modes: int = 5) -> np.ndarray:
    """Generate a low-frequency periodic Fourier perturbation."""
    out = np.zeros_like(x, dtype=np.float64)

    for k in range(1, modes + 1):
        out += rng.uniform(-1.0 / k, 1.0 / k) * np.cos(2.0 * np.pi * k * x)
        out += rng.uniform(-1.0 / k, 1.0 / k) * np.sin(2.0 * np.pi * k * x)

    return out


def sharp_initial(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a periodic initial condition with two steep transitions."""
    xc = rng.uniform(0.0, 1.0)
    w = rng.uniform(0.08, 0.25)
    eps = rng.uniform(0.005, 0.03)

    uin = rng.uniform(-1.0, 1.0)
    uout = rng.uniform(-1.0, 1.0)

    # Periodic distance from x to xc on [0, 1].
    d = np.abs(((x - xc + 0.5) % 1.0) - 0.5)

    front = uout + 0.5 * (uin - uout) * (1.0 + np.tanh((w - d) / eps))

    return front + 0.1 * low_freq_signal(x, rng, modes=5)


def rhs_burgers_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """Evaluate the periodic finite-volume Burgers right-hand side."""
    flux = 0.5 * u * u

    up = np.roll(u, -1)
    fp = np.roll(flux, -1)

    # Rusanov / local Lax-Friedrichs interface flux F_{i+1/2}.
    a = np.maximum(np.abs(u), np.abs(up))
    f_half = 0.5 * (flux + fp) - 0.5 * a * (up - u)

    div = (f_half - np.roll(f_half, 1)) / dx

    return -div


def rk4_step(u: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """One RK4 step for the inviscid Burgers semi-discrete system."""
    k1 = rhs_burgers_periodic(u, dx)
    k2 = rhs_burgers_periodic(u + 0.5 * dt * k1, dx)
    k3 = rhs_burgers_periodic(u + 0.5 * dt * k2, dx)
    k4 = rhs_burgers_periodic(u + dt * k3, dx)

    return u + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def evolve(u: np.ndarray, delta: float, dx: float, cfl: float) -> np.ndarray:
    """Evolve inviscid Burgers over a physical lag delta."""
    t = 0.0

    while t < delta - 1e-14:
        max_speed = max(float(np.max(np.abs(u))), 1e-6)
        dt_adv = cfl * dx / max_speed
        dt = min(dt_adv, delta - t)

        u = rk4_step(u, dt, dx)
        t += dt

    return u


def conservative_average(u_fine: np.ndarray, coarse_n: int) -> np.ndarray:
    """Average fine-grid values into coarse cells.

    Requires fine_n to be divisible by coarse_n.
    """
    fine_n = u_fine.shape[0]

    if fine_n % coarse_n != 0:
        raise ValueError(
            f"fine_n={fine_n} must be divisible by coarse_n={coarse_n}."
        )

    r = fine_n // coarse_n

    return u_fine.reshape(coarse_n, r).mean(axis=1)


def make_split(
    ntraj: int,
    nsnaps: int,
    fine_n: int,
    coarse_n: int,
    lag_min: float,
    lag_max: float,
    seed: int,
    cfl: float,
) -> dict[str, np.ndarray]:
    """Generate one train/test split."""
    rng = np.random.default_rng(seed)

    xfine = np.arange(fine_n, dtype=np.float64) / fine_n
    dx = 1.0 / fine_n

    trajectories = np.zeros((ntraj, coarse_n, 1, nsnaps), dtype=np.float32)
    dt = np.zeros((ntraj, nsnaps - 1), dtype=np.float32)

    for i in range(ntraj):
        u = sharp_initial(xfine, rng).astype(np.float64)

        trajectories[i, :, 0, 0] = conservative_average(u, coarse_n).astype(np.float32)

        for k in range(nsnaps - 1):
            lag = rng.uniform(lag_min, lag_max)
            dt[i, k] = lag

            u = evolve(u, lag, dx, cfl)

            trajectories[i, :, 0, k + 1] = conservative_average(u, coarse_n).astype(
                np.float32
            )

        if (i + 1) % 50 == 0 or (i + 1) == ntraj:
            print(f"  generated {i + 1} / {ntraj} trajectories")

    coordinates = (np.arange(coarse_n, dtype=np.float32) / coarse_n).reshape(coarse_n, 1)

    return {
        "trajectories": trajectories,
        "dt": dt,
        "coordinates": coordinates,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate variable-lag sharp-front inviscid Burgers OSG data."
    )

    ap.add_argument("--out-dir", default="data")

    ap.add_argument("--train-traj", type=int, default=800)
    ap.add_argument("--test-traj", type=int, default=100)

    ap.add_argument("--train-snaps", type=int, default=11)
    ap.add_argument("--test-snaps", type=int, default=21)

    ap.add_argument("--fine-n", type=int, default=4096)
    ap.add_argument("--coarse-n", type=int, default=64)

    ap.add_argument("--lag-min", type=float, default=0.005)
    ap.add_argument("--lag-max", type=float, default=0.15)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cfl", type=float, default=0.5)

    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train_traj <= 0 or args.test_traj <= 0:
        raise ValueError("train-traj and test-traj must be positive.")

    if args.train_snaps < 2 or args.test_snaps < 2:
        raise ValueError("train-snaps and test-snaps must be at least 2.")

    if args.fine_n <= 0 or args.coarse_n <= 0:
        raise ValueError("fine-n and coarse-n must be positive.")

    if args.fine_n % args.coarse_n != 0:
        raise ValueError(
            f"fine-n={args.fine_n} must be divisible by coarse-n={args.coarse_n}."
        )

    if not (0.0 < args.lag_min < args.lag_max):
        raise ValueError("Require 0 < lag-min < lag-max.")

    if args.cfl <= 0:
        raise ValueError("cfl must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    print(vars(args))
    print("Equation: inviscid Burgers, nu = 0")
    print("Output filenames: BurgersSharpOSG_train.mat, BurgersSharpOSG_test.mat")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = make_split(
        ntraj=args.train_traj,
        nsnaps=args.train_snaps,
        fine_n=args.fine_n,
        coarse_n=args.coarse_n,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        seed=args.seed,
        cfl=args.cfl,
    )

    test = make_split(
        ntraj=args.test_traj,
        nsnaps=args.test_snaps,
        fine_n=args.fine_n,
        coarse_n=args.coarse_n,
        lag_min=args.lag_min,
        lag_max=args.lag_max,
        seed=args.seed + 100000,
        cfl=args.cfl,
    )

    train_path = out / "BurgersSharpOSG_train.mat"
    test_path = out / "BurgersSharpOSG_test.mat"

    savemat(train_path, train)
    savemat(test_path, test)

    print("saved", train_path, test_path)


if __name__ == "__main__":
    main()
