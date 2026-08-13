from __future__ import annotations

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
DATASET_FILES = {
    "original": ("BurgersOSG_train.mat", "BurgersOSG_test.mat"),
    "sharp": ("BurgersSharpOSG_train.mat", "BurgersSharpOSG_test.mat"),
}

MODEL_CHOICES = ["fno", "fno_film", "gl_fno", "gl_fno_film"]


def resolve_data_paths(data_dir: str | os.PathLike[str], dataset: str):
    data_root = Path(data_dir)
    train_file, test_file = DATASET_FILES[dataset]
    return data_root / train_file, data_root / test_file


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
    epochs: int = 1000,
    batch_size: int = 100,
    device: str | None = None,
    hf_weight: float = 0.0,
    hf_sg_weight: float = 0.0,
    hf_warmup_frac: float = 0.1,
    conserve_mean: bool = False,
    gl_film_mode: str = "branchwise",
    modes: int = 10,
    width: int = 60,
):
    return {
        "problem_type": "1d_regular",
        "problem_dim": 1,
        "multiscale": True,
        "dtype": "float32",

        "seed": seed,
        "device": device or default_device(),

        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "scheduler": "cosine",
        "verbose": 10,
        "loss": "mae",

        "nbursts": 10,
        "sg_pairing": 2,
        "sg_weight": 5.0,

        "activation": "gelu",

        # FNO-1D backbone
        "modes": int(modes),
        "depth": 3,
        "width": int(width),

        "local_kernel_size": 5,
        "local_pool_factor": 2,
        "gl_layer_scale": 1e-3,
        "hf_weight": hf_weight,
        "hf_sg_weight": hf_sg_weight,
        "hf_warmup_frac": hf_warmup_frac,
        "hf_band_frac": 1.0 / 3.0,
        "hf_power": 2.0,
        "conserve_mean": conserve_mean,
        "gl_film_mode": gl_film_mode,

        "save_path": save_path,
    }


def build_model(model_name, vmin, vmax, tmin, tmax, config):
    from film_osg.networks.fno import (
        gl_osg_fno1d,
        gl_osg_fno1d_with_film,
        osg_fno1d,
        osg_fno1d_with_film,
    )

    if model_name == "fno":
        return osg_fno1d(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    if model_name == "fno_film":
        return osg_fno1d_with_film(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    if model_name == "gl_fno":
        return gl_osg_fno1d(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    if model_name == "gl_fno_film":
        return gl_osg_fno1d_with_film(
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
    epochs: int = 1000,
    batch_size: int = 100,
    tag: str = "",
    overwrite: bool = False,
    save_dir: str = ".",
    device: str | None = None,
    data_dir: str | os.PathLike[str] = DEFAULT_DATA_DIR,
    dataset_name: str = "original",
    hf_weight: float = 0.0,
    hf_sg_weight: float = 0.0,
    hf_warmup_frac: float = 0.1,
    conserve_mean: bool = False,
    gl_film_mode: str = "branchwise",
    modes: int = 10,
    width: int = 60,
):
    suffix = f"_{tag}" if tag else ""
    save_path = Path(save_dir) / f"runs_burgers_{model_name}_seed{seed}{suffix}"
    train_path, _ = resolve_data_paths(data_dir, dataset_name)

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
        gl_film_mode=gl_film_mode,
        modes=modes,
        width=width,
    )

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

    if save_path.exists():
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True)

    with (save_path / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"model": model_name, "dataset": dataset_name, **config}, f, indent=2)

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
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--save-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--dataset",
        choices=DATASET_FILES,
        default="original",
        help="Select the original or steep-gradient Burgers data files.",
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
    parser.add_argument("--gl-film-mode", type=str, default="branchwise", choices=["branchwise", "global_only"])
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--width", type=int, default=60)
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
        dataset_name=args.dataset,
        hf_weight=args.hf_weight,
        hf_sg_weight=args.hf_sg_weight,
        hf_warmup_frac=args.hf_warmup_frac,
        conserve_mean=args.conserve_mean,
        gl_film_mode=args.gl_film_mode,
        modes=args.modes,
        width=args.width,
    )


if __name__ == "__main__":
    main()
