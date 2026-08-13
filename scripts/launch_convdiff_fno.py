#!/usr/bin/env python3
"""Run advection--diffusion training jobs across a list of GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "train" / "run_convdiff_fno.py"
DEFAULT_MODELS = ("fno", "fno_film")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def comma_list(text, cast=str):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def training_command(args, model, seed):
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model", model,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--data-dir", args.data_dir,
        "--save-dir", args.save_dir,
        "--learning-rate", str(args.learning_rate),
        "--sg-weight", str(args.sg_weight),
        "--modes1", str(args.modes1),
        "--modes2", str(args.modes2),
        "--depth", str(args.depth),
        "--width", str(args.width),
    ]
    if args.tag:
        command.extend(["--tag", args.tag])
    if args.overwrite:
        command.append("--overwrite")
    if args.log_lag:
        command.append("--log-lag")
    if args.conserve_mean:
        command.append("--conserve-mean")
    if args.problem_dim is not None:
        command.extend(["--problem-dim", str(args.problem_dim)])
    if args.hf_weight:
        command.extend(["--hf-weight", str(args.hf_weight)])
    if args.hf_sg_weight:
        command.extend(["--hf-sg-weight", str(args.hf_sg_weight)])
    if args.hf_weight or args.hf_sg_weight:
        command.extend(["--hf-warmup-frac", str(args.hf_warmup_frac)])
    return command


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU ids.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir", default=".")
    parser.add_argument("--log-dir", default="logs_convdiff_fno")
    parser.add_argument("--tag", default="")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sg-weight", type=float, default=1.0)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--problem-dim", type=int, default=None)
    parser.add_argument("--log-lag", action="store_true")
    parser.add_argument("--conserve-mean", action="store_true")
    parser.add_argument("--hf-weight", type=float, default=0.0)
    parser.add_argument("--hf-sg-weight", type=float, default=0.0)
    parser.add_argument("--hf-warmup-frac", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()
    models = comma_list(args.models)
    seeds = comma_list(args.seeds, int)
    gpus = comma_list(args.gpus, int)
    if not models or not seeds or not gpus:
        raise SystemExit("Models, seeds, and GPUs must be non-empty lists.")

    jobs = deque((model, seed) for seed in seeds for model in models)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    running = {}
    failures = []

    print(
        f"Launching {len(jobs)} advection--diffusion jobs: models={models}, "
        f"seeds={seeds}, GPUs={gpus}"
    )

    while jobs or running:
        for gpu in gpus:
            if gpu in running or not jobs:
                continue

            model, seed = jobs.popleft()
            tag = f"_{args.tag}" if args.tag else ""
            log_path = log_dir / f"convdiff_{model}_seed{seed}{tag}_gpu{gpu}.log"
            log_file = log_path.open("w", buffering=1)
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            process = subprocess.Popen(
                training_command(args, model, seed),
                cwd=REPO_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running[gpu] = (process, log_file, log_path, model, seed, time.time())
            print(f"GPU {gpu}: started {model}, seed {seed} ({log_path})")

        if running:
            time.sleep(args.poll_seconds)

        for gpu, job in list(running.items()):
            process, log_file, log_path, model, seed, start = job
            returncode = process.poll()
            if returncode is None:
                continue

            log_file.close()
            elapsed = (time.time() - start) / 60.0
            if returncode == 0:
                print(f"GPU {gpu}: finished {model}, seed {seed} in {elapsed:.1f} min")
            else:
                failures.append((model, seed, returncode, log_path))
                print(f"GPU {gpu}: {model}, seed {seed} failed; see {log_path}")
            del running[gpu]

    if failures:
        print("\nFailed jobs:")
        for model, seed, returncode, log_path in failures:
            print(f"  {model}, seed {seed}, exit {returncode}: {log_path}")
        raise SystemExit(1)

    print("All advection--diffusion jobs finished successfully.")


if __name__ == "__main__":
    main()
