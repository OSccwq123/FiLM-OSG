import os
import shutil
import random
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATA_DIR = REPO_ROOT / "data"
TRAIN_FILE = "train_data.mat"
TEST_FILE = "test_data.mat"

MODEL_CHOICES = ["fno", "fno_film", "vt_fno", "vt_fno_film"]


def resolve_data_paths(data_dir: str | os.PathLike[str]):
    data_root = Path(data_dir)
    return data_root / TRAIN_FILE, data_root / TEST_FILE


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_config(
    save_path: str,
    seed: int = 0,
    epochs: int = 500,
    batch_size: int = 100,
    device: str | None = None,
    hf_weight: float = 0.0,
    hf_sg_weight: float = 0.0,
    hf_warmup_frac: float = 0.1,
    conserve_mean: bool = False,
    log_delta: bool = False,
    learning_rate: float = 1e-3,
    sg_weight: float = 1.0,
    modes1: int = 12,
    modes2: int = 12,
    depth: int = 4,
    width: int = 20,
    problem_dim: int = 1,
):
    return {
        "problem_type": "2d_regular",
        "problem_dim": int(problem_dim),
        "multiscale": log_delta,
        "dtype": "float32",

        "seed": seed,
        "device": device or default_device(),

        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "adam",
        "scheduler": "cosine",
        "verbose": 10,
        "loss": "mae",

        "nbursts": 25,
        "sg_pairing": 1,
        "sg_weight": sg_weight,

        "activation": "gelu",

        # FNO-2D backbone
        "modes1": modes1,
        "modes2": modes2,
        "depth": depth,
        "width": width,

        "hf_weight": hf_weight,
        "hf_sg_weight": hf_sg_weight,
        "hf_warmup_frac": hf_warmup_frac,
        "hf_band_frac": 1.0 / 3.0,
        "hf_power": 2.0,
        "conserve_mean": conserve_mean,

        "save_path": save_path,
    }


def build_model(model_name, vmin, vmax, tmin, tmax, config):
    from film_osg.networks.fno import (
        osg_fno2d,
        osg_fno2d_with_film,
        vt_fno2d,
        vt_fno2d_with_film,
    )

    if model_name == "vt_fno":
        return vt_fno2d(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )
    if model_name == "vt_fno_film":
        return vt_fno2d_with_film(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    if model_name == "fno":
        return osg_fno2d(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    if model_name == "fno_film":
        return osg_fno2d_with_film(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def train_one(
    model_name: str,
    seed: int,
    epochs: int = 500,
    batch_size: int = 100,
    tag: str = "",
    overwrite: bool = False,
    save_dir: str = ".",
    device: str | None = None,
    data_dir: str | os.PathLike[str] = DEFAULT_DATA_DIR,
    log_delta: bool = False,
    hf_weight: float = 0.0,
    hf_sg_weight: float = 0.0,
    hf_warmup_frac: float = 0.1,
    conserve_mean: bool = False,
    learning_rate: float = 1e-3,
    sg_weight: float = 1.0,
    modes1: int = 12,
    modes2: int = 12,
    depth: int = 4,
    width: int = 20,
    problem_dim: int | None = None,
):
    suffix = f"_{tag}" if tag else ""
    save_path = Path(save_dir) / f"runs_convdiff_{model_name}_seed{seed}{suffix}"
    train_path, _ = resolve_data_paths(data_dir)

    config = make_config(
        save_path=str(save_path),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        hf_weight=hf_weight,
        hf_sg_weight=hf_sg_weight,
        hf_warmup_frac=hf_warmup_frac,
        conserve_mean=conserve_mean,
        log_delta=log_delta,
        learning_rate=learning_rate,
        sg_weight=sg_weight,
        modes1=modes1,
        modes2=modes2,
        depth=depth,
        width=width,
        problem_dim=problem_dim if problem_dim is not None else 1,
    )

    if model_name in {"vt_fno", "vt_fno_film"}:
        config["sg_pairing"] = 0
        config["sg_weight"] = 0.0
        config["hf_weight"] = 0.0
        config["hf_sg_weight"] = 0.0
        config["conserve_mean"] = False

    from film_osg.datasets.pde import pde_dataset_osg
    from film_osg.models.pde_osg import PDE_osg

    set_seed(seed)

    if not train_path.is_file():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if save_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {save_path}. "
            "Pass --overwrite to replace it."
        )

    dataset = pde_dataset_osg(config)
    trainX, trainY, _, _, vmin, vmax, tmin, tmax, _, _ = dataset.load(
        str(train_path), None
    )
    inferred_problem_dim = int(trainY.shape[-1])
    if problem_dim is not None and int(problem_dim) != inferred_problem_dim:
        raise ValueError(
            f"Requested problem_dim={problem_dim}, but the training data contain "
            f"{inferred_problem_dim} state channels."
        )
    config["problem_dim"] = inferred_problem_dim

    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)
    with (save_path / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({"model": model_name, **config}, handle, indent=2)

    print(f"Training {model_name} with seed {seed} on {config['device']}", flush=True)
    print(f"Data: {train_path}", flush=True)
    print(f"Output: {save_path}", flush=True)

    net = build_model(model_name, vmin, vmax, tmin, tmax, config)

    solver = PDE_osg(
        trainX,
        trainY,
        network=net,
        config=config,
    )

    solver.train()
    solver.save_hist()

    print(f"Finished training {model_name}, seed={seed}, saved to {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODEL_CHOICES)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--save-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--log-delta",
        action="store_true",
        help="Use log10(dt) before affine normalization of Delta for this AD run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory for this model and seed.",
    )
    parser.add_argument("--hf-weight", type=float, default=0.0)
    parser.add_argument("--hf-sg-weight", type=float, default=0.0)
    parser.add_argument("--hf-warmup-frac", type=float, default=0.1)
    parser.add_argument("--conserve-mean", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--sg-weight", type=float, default=1.0)
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument(
        "--problem-dim",
        type=int,
        default=None,
        help="Optional state-channel assertion; otherwise inferred from trainY.",
    )
    args = parser.parse_args()

    train_one(
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        tag=args.tag,
        overwrite=args.overwrite,
        save_dir=args.save_dir,
        device=args.device,
        data_dir=args.data_dir,
        log_delta=args.log_delta,
        hf_weight=args.hf_weight,
        hf_sg_weight=args.hf_sg_weight,
        hf_warmup_frac=args.hf_warmup_frac,
        conserve_mean=args.conserve_mean,
        learning_rate=args.learning_rate,
        sg_weight=args.sg_weight,
        modes1=args.modes1,
        modes2=args.modes2,
        depth=args.depth,
        width=args.width,
        problem_dim=args.problem_dim,
    )


if __name__ == "__main__":
    main()
