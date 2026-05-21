import os
import sys
import time
import argparse
import subprocess
from collections import deque
from datetime import datetime



DEFAULT_SEEDS = [0, 1, 2, 3, 4]


DEFAULT_MODELS = ["fno_film", "fno"]


DEFAULT_GPUS = [0, 1, 3, 4]

LOG_DIR = "./logs_ns_10seed"
TRAIN_SCRIPT = "train_ns_one.py"


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
        available: bool
        name: str
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
        return False, "INSPECTION_FAILED: " + proc.stderr.strip()

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
    print("CUDA_DEVICE_ORDER=PCI_BUS_ID will be used for all child jobs.", flush=True)

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

    print("\nValid GPU ids:", valid, flush=True)
    print("=" * 80, flush=True)
    return valid


def model_file_exists(model_name, seed):
    """
    DUE solver.save() usually stores the model at:
        ./runs_ns_{model_name}_seed{seed}/model
    """
    model_path = f"./runs_ns_{model_name}_seed{seed}/model"
    return os.path.exists(model_path)


def build_jobs(seeds, models, skip_existing=True):
    jobs = deque()

    skipped = []
    for seed in seeds:
        for model in models:
            if skip_existing and model_file_exists(model, seed):
                skipped.append((model, seed))
                continue
            jobs.append((model, seed))

    return jobs, skipped


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--gpus", type=str, default=",".join(str(g) for g in DEFAULT_GPUS))

    parser.add_argument("--poll-seconds", type=int, default=30)

    # By default, reject T1000 / unexpected GPUs.
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

    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Do not skip jobs whose ./runs_ns_{model}_seed{seed}/model already exists.",
    )

    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    models = parse_str_list(args.models)
    requested_gpus = parse_int_list(args.gpus)

    # Validate actual PyTorch-visible mapping.
    gpus = validate_gpus(
        requested_gpus,
        require_a100=not args.allow_non_a100,
    )

    if args.check_gpus_only:
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    jobs, skipped = build_jobs(
        seeds=seeds,
        models=models,
        skip_existing=not args.overwrite_existing,
    )

    running = {}
    failed = []

    print("=" * 80, flush=True)
    print("NS ten-seed completion launcher", flush=True)
    print("Seeds:", seeds, flush=True)
    print("Models:", models, flush=True)
    print("Requested GPUs:", requested_gpus, flush=True)
    print("Validated GPUs:", gpus, flush=True)
    print("Total jobs to run:", len(jobs), flush=True)

    if skipped:
        print("Skipped existing completed jobs:", skipped, flush=True)

    print("=" * 80, flush=True)

    while jobs or running:
        # Launch new jobs on free GPUs.
        for gpu in gpus:
            if gpu in running:
                continue
            if not jobs:
                break

            model, seed = jobs.popleft()
            log_path = os.path.join(LOG_DIR, f"ns_{model}_seed{seed}_gpu{gpu}.log")
            log_file = open(log_path, "w", buffering=1)

            env = os.environ.copy()

            # Critical: align CUDA ordering with nvidia-smi PCI bus order.
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)

            # Avoid CPU oversubscription when launching several processes.
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"

            cmd = [
                sys.executable,
                TRAIN_SCRIPT,
                "--model", model,
                "--seed", str(seed),
            ]

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

        # Check running jobs.
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