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
TRAIN_FILE = "VorticityOSG_train.mat"
TEST_FILE = "VorticityOSG_test.mat"


MODEL_CHOICES = [
    "uno",
    "uno_film",
    "transolver",
    "transolver_film",
]


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
    batch_size: int = 20,
    device: str | None = None,
):
    return {
        "problem_type": "2d_regular",
        "problem_dim": 1,
        "multiscale": False,
        "dtype": "float32",

        "seed": seed,
        "device": device or default_device(),

        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "scheduler": "cosine",
        "verbose": 10,

        "loss": "rel_l2",

        "nbursts": 25,
        "sg_pairing": 1,
        "sg_weight": 1.0,

        "activation": "gelu",

        "depth": 4,
        "width": 20,

        "modes1": 8,
        "modes2": 8,
        "uno_norm": False,

        "n_head": 2,
        "slice_num": 32,
        "dropout": 0.0,
        "mlp_ratio": 4,

        "time_width": 20,

        "save_path": save_path,
    }


def build_model(model_name, vmin, vmax, tmin, tmax, config):
    from film_osg.networks.osg_extra_backbones import (
        osg_uno2d,
        osg_uno2d_with_film,
        osg_transolver2d,
        osg_transolver2d_with_film,
    )

    if model_name == "uno":
        return osg_uno2d(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    if model_name == "uno_film":
        return osg_uno2d_with_film(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    if model_name == "transolver":
        return osg_transolver2d(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    if model_name == "transolver_film":
        return osg_transolver2d_with_film(
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
    batch_size: int = 20,
    tag: str = "",
    overwrite: bool = False,
    save_dir: str = ".",
    device: str | None = None,
    data_dir: str | os.PathLike[str] = DEFAULT_DATA_DIR,
):
    suffix = f"_{tag}" if tag else ""
    save_path = Path(save_dir) / f"runs_ns_{model_name}_seed{seed}{suffix}"
    train_path, _ = resolve_data_paths(data_dir)

    config = make_config(
        save_path=str(save_path),
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
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
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--save-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory for this model and seed.",
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
    )


if __name__ == "__main__":
    main()
