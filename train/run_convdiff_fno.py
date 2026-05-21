import os
import shutil
import random
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TRAIN_PATH = "train_data.mat"
TEST_PATH = "test_data.mat"

MODEL_CHOICES = ["fno", "fno_film"]


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
        "loss": "mae",

        "nbursts": 25,
        "sg_pairing": 1,
        "sg_weight": 1.0,

        "activation": "gelu",

        # FNO-2D backbone
        "modes1": 12,
        "modes2": 12,
        "depth": 4,
        "width": 20,

        "save_path": save_path,
    }


def build_model(model_name, vmin, vmax, tmin, tmax, config):
    try:
        from film_osg.networks.fno import osg_fno2d, osg_fno2d_with_film
        import_source = "film_osg"
    except ImportError:
        from due.networks.fno import osg_fno2d, osg_fno2d_with_film
        import_source = "due"

    print("network_import_source =", import_source, flush=True)

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
    overwrite: bool = True,
    save_dir: str = ".",
    device: str | None = None,
    dry_run: bool = False,
):
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(save_dir, f"runs_convdiff_{model_name}_seed{seed}{suffix}")

    config = make_config(
        save_path=save_path,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )

    if dry_run:
        print("Advection-diffusion FNO training dry run", flush=True)
        print("model =", model_name, flush=True)
        print("seed =", seed, flush=True)
        print("train_path =", TRAIN_PATH, flush=True)
        print("test_path =", TEST_PATH, flush=True)
        print("save_path =", save_path, flush=True)
        print("config =", config, flush=True)
        print("No due imports, data loading, directory writes, or training were run.", flush=True)
        return

    try:
        from film_osg.datasets.pde import pde_dataset_osg
        from film_osg.models.pde_osg import PDE_osg
        runtime_import_source = "film_osg"
    except ImportError:
        from due.datasets.pde import pde_dataset_osg
        from due.models.pde_osg import PDE_osg
        runtime_import_source = "due"

    print("dataset_solver_import_source =", runtime_import_source, flush=True)

    set_seed(seed)

    if overwrite:
        shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)

    dataset = pde_dataset_osg(config)
    trainX, trainY, coords, data_test, dt_test, vmin, vmax, tmin, tmax, cmin, cmax = dataset.load(
        TRAIN_PATH, TEST_PATH
    )

    print("============================================================", flush=True)
    print(f"Model: {model_name}, seed={seed}", flush=True)
    print("save_path =", save_path, flush=True)
    print("coords.shape =", coords.shape, flush=True)
    print("trainX.shape =", trainX.shape, flush=True)
    print("trainY.shape =", trainY.shape, flush=True)
    print("config =", {
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "optimizer": config["optimizer"],
        "scheduler": config["scheduler"],
        "loss": config["loss"],
        "sg_pairing": config["sg_pairing"],
        "sg_weight": config["sg_weight"],
        "modes1": config["modes1"],
        "modes2": config["modes2"],
        "depth": config["depth"],
        "width": config["width"],
    }, flush=True)
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("torch.cuda.is_available =", torch.cuda.is_available(), flush=True)
    if torch.cuda.is_available():
        print("torch.cuda.current_device =", torch.cuda.current_device(), flush=True)
        print("torch.cuda.get_device_name =", torch.cuda.get_device_name(0), flush=True)
    print("============================================================", flush=True)

    net = build_model(model_name, vmin, vmax, tmin, tmax, config)

    solver = PDE_osg(
        trainX,
        trainY,
        osg_data=None,
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()

    train_one(
        model_name=args.model,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        tag=args.tag,
        overwrite=not args.no_overwrite,
        save_dir=args.save_dir,
        device=args.device,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
