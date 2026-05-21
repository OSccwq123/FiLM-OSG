import os
import sys
import time
import argparse
import subprocess
from collections import deque
from datetime import datetime


DEFAULT_MODELS = [
    "fno_film",
    "fno",
]

TRAIN_SCRIPT = os.path.join("train", "run_burgers_fno.py")
LOG_DIR = "./logs_burgers_fno"


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def inspect_visible_gpu(gpu_id):
    """
    Check what PyTorch sees when CUDA_VISIBLE_DEVICES=gpu_id.

    Returns:
        (available: bool, name: str)
    """
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    code = r"""
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("NO_CUDA")
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    if proc.returncode != 0:
        return False, f"INSPECTION_FAILED: {proc.stderr.strip()}"

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return False, "NO_OUTPUT"

    available = lines[0] == "True"
    name = lines[1] if len(lines) >= 2 else "UNKNOWN"
    return available, name


def validate_gpus(gpus, require_a100=True):
    """
    Validate CUDA mapping before launching jobs.
    """
    print("=" * 80, flush=True)
    print("Validating CUDA device mapping", flush=True)
    print("CUDA_DEVICE_ORDER will be set to PCI_BUS_ID for all child jobs.", flush=True)

    valid = []
    rejected = []

    for gpu in gpus:
        available, name = inspect_visible_gpu(gpu)
        print(f"GPU id {gpu}: available={available}, visible_name={name}", flush=True)

        if not available:
            rejected.append((gpu, name))
            continue

        if require_a100 and "A100" not in name:
            rejected.append((gpu, name))
            continue

        valid.append(gpu)

    if rejected:
        print("\nRejected GPU ids:", flush=True)
        for gpu, name in rejected:
            print(f"  gpu={gpu}, name={name}", flush=True)

    if not valid:
        raise RuntimeError(
            "No valid GPU remains after validation. "
            "Check CUDA_VISIBLE_DEVICES mapping or use --allow-non-a100."
        )

    print("Valid GPU ids:", valid, flush=True)
    print("=" * 80, flush=True)
    return valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--gpus", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned jobs and child commands without validating GPUs or launching.",
    )

    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="Allow GPUs whose visible name does not contain A100.",
    )

    parser.add_argument(
        "--check-gpus-only",
        action="store_true",
        help="Only check CUDA mapping and exit without launching jobs.",
    )

    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    models = parse_str_list(args.models)
    requested_gpus = parse_int_list(args.gpus)

    if args.dry_run:
        print("=" * 80, flush=True)
        print("Burgers FNO launcher dry run", flush=True)
        print("Models:", models, flush=True)
        print("Seeds:", seeds, flush=True)
        print("Requested GPUs:", requested_gpus, flush=True)
        print("Epochs:", args.epochs, flush=True)
        print("Batch size:", args.batch_size, flush=True)
        print("Tag:", args.tag if args.tag else "(none)", flush=True)
        print("No GPU validation or jobs will be launched.", flush=True)
        for seed in seeds:
            for model in models:
                cmd = [
                    sys.executable,
                    TRAIN_SCRIPT,
                    "--model", model,
                    "--seed", str(seed),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--dry-run",
                ]
                if args.tag:
                    cmd.extend(["--tag", args.tag])
                if args.no_overwrite:
                    cmd.append("--no-overwrite")
                print("CMD:", " ".join(cmd), flush=True)
        print("=" * 80, flush=True)
        return

    gpus = validate_gpus(
        requested_gpus,
        require_a100=not args.allow_non_a100,
    )

    if args.check_gpus_only:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    jobs = deque()
    for seed in seeds:
        for model in models:
            jobs.append((model, seed))

    running = {}
    failed = []

    print("=" * 80, flush=True)
    print("Burgers FNO launcher", flush=True)
    print("Models:", models, flush=True)
    print("Seeds:", seeds, flush=True)
    print("Requested GPUs:", requested_gpus, flush=True)
    print("Validated GPUs:", gpus, flush=True)
    print("Epochs:", args.epochs, flush=True)
    print("Batch size:", args.batch_size, flush=True)
    print("Tag:", args.tag if args.tag else "(none)", flush=True)
    print("Total jobs:", len(jobs), flush=True)
    print("=" * 80, flush=True)

    while jobs or running:
        for gpu in gpus:
            if gpu in running:
                continue
            if not jobs:
                break

            model, seed = jobs.popleft()

            tag_part = f"_{args.tag}" if args.tag else ""
            log_path = os.path.join(
                LOG_DIR,
                f"burgers_{model}_seed{seed}{tag_part}_gpu{gpu}.log",
            )
            log_file = open(log_path, "w", buffering=1)

            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"

            cmd = [
                sys.executable,
                TRAIN_SCRIPT,
                "--model", model,
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
            ]

            if args.tag:
                cmd.extend(["--tag", args.tag])

            if args.no_overwrite:
                cmd.append("--no-overwrite")

            print(f"[{timestamp()}] START gpu={gpu}: {model}, seed={seed}", flush=True)
            print("  log:", log_path, flush=True)
            print("  CUDA_DEVICE_ORDER=PCI_BUS_ID", flush=True)
            print(f"  CUDA_VISIBLE_DEVICES={gpu}", flush=True)

            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )

            running[gpu] = {
                "proc": proc,
                "log_file": log_file,
                "log_path": log_path,
                "model": model,
                "seed": seed,
                "start": time.time(),
            }

        time.sleep(args.poll_seconds)

        for gpu, info in list(running.items()):
            proc = info["proc"]
            ret = proc.poll()

            if ret is None:
                continue

            elapsed = time.time() - info["start"]
            info["log_file"].close()

            model = info["model"]
            seed = info["seed"]

            if ret == 0:
                print(
                    f"[{timestamp()}] DONE  gpu={gpu}: {model}, seed={seed}, "
                    f"elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
            else:
                print(
                    f"[{timestamp()}] FAIL  gpu={gpu}: {model}, seed={seed}, "
                    f"returncode={ret}, elapsed={elapsed / 60:.1f} min",
                    flush=True,
                )
                print("  see log:", info["log_path"], flush=True)
                failed.append((model, seed, gpu, ret, info["log_path"]))

            del running[gpu]

        if running:
            active = ", ".join(
                f"GPU {gpu}: {info['model']}/seed{info['seed']}"
                for gpu, info in running.items()
            )
            print(f"[{timestamp()}] ACTIVE {active}", flush=True)

    print("=" * 80, flush=True)
    if failed:
        print("Some jobs failed:", flush=True)
        for item in failed:
            print(item, flush=True)
        sys.exit(1)

    print("All jobs finished successfully.", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
