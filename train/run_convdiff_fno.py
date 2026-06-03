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


DEFAULT_DATA_DIR = REPO_ROOT / "data"
TRAIN_FILE = "train_data.mat"
TEST_FILE = "test_data.mat"

MODEL_CHOICES = ["fno", "fno_film", "gl_fno", "gl_fno_film", "vt_fno", "vt_fno_film"]


def resolve_data_paths(data_dir: str | os.PathLike[str]):
    data_root = Path(data_dir)
    return str(data_root / TRAIN_FILE), str(data_root / TEST_FILE)


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
    gl_film_mode: str = "global_only",
    log_lag: bool = False,
):
    return {
        "problem_type": "2d_regular",
        "problem_dim": 1,
        "multiscale": log_lag,
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

        "local_kernel_size": 3,
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
        gl_osg_fno2d,
        gl_osg_fno2d_with_film,
        osg_fno2d,
        osg_fno2d_with_film,
        vt_fno2d,
        vt_fno2d_with_film,
    )

    print("network_import_source = film_osg", flush=True)

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
    if model_name == "gl_fno":
        return gl_osg_fno2d(vmin=vmin, vmax=vmax, tmin=tmin, tmax=tmax, config=config, multiscale=config["multiscale"])
    if model_name == "gl_fno_film":
        return gl_osg_fno2d_with_film(vmin=vmin, vmax=vmax, tmin=tmin, tmax=tmax, config=config, multiscale=config["multiscale"])

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
    data_dir: str | os.PathLike[str] = DEFAULT_DATA_DIR,
    log_lag: bool = False,
    dry_run: bool = False,
    hf_weight: float = 0.0,
    hf_sg_weight: float = 0.0,
    hf_warmup_frac: float = 0.1,
    conserve_mean: bool = False,
    gl_film_mode: str = "global_only",
):
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(save_dir, f"runs_convdiff_{model_name}_seed{seed}{suffix}")
    train_path, test_path = resolve_data_paths(data_dir)

    config = make_config(
        save_path=save_path,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        hf_weight=hf_weight,
        hf_sg_weight=hf_sg_weight,
        hf_warmup_frac=hf_warmup_frac,
        conserve_mean=conserve_mean,
        gl_film_mode=gl_film_mode,
        log_lag=log_lag,
    )

    if model_name in {"vt_fno", "vt_fno_film"}:
        config["sg_pairing"] = 0
        config["sg_weight"] = 0.0
        config["hf_weight"] = 0.0
        config["hf_sg_weight"] = 0.0
        config["conserve_mean"] = False

    if dry_run:
        print("Advection-diffusion FNO training dry run", flush=True)
        print("model =", model_name, flush=True)
        print("seed =", seed, flush=True)
        print("train_path =", train_path, flush=True)
        print("test_path =", test_path, flush=True)
        print("save_path =", save_path, flush=True)
        print("lag_preprocessing =", "log10_then_affine" if log_lag else "affine_only", flush=True)
        print("config =", config, flush=True)
        print("No data loading, directory writes, or training were run.", flush=True)
        return

    from film_osg.datasets.pde import pde_dataset_osg
    from film_osg.models.pde_osg import PDE_osg

    print("dataset_solver_import_source = film_osg", flush=True)

    set_seed(seed)

    if overwrite:
        shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)

    dataset = pde_dataset_osg(config)
    trainX, trainY, coords, data_test, dt_test, vmin, vmax, tmin, tmax, cmin, cmax = dataset.load(
        train_path, test_path
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
        "multiscale": config["multiscale"],
        "sg_pairing": config["sg_pairing"],
        "sg_weight": config["sg_weight"],
        "modes1": config["modes1"],
        "modes2": config["modes2"],
        "depth": config["depth"],
        "width": config["width"],
        "hf_weight": config["hf_weight"],
        "hf_sg_weight": config["hf_sg_weight"],
        "hf_warmup_frac": config["hf_warmup_frac"],
        "conserve_mean": config["conserve_mean"],
        "gl_film_mode": config["gl_film_mode"],
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
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--log-lag",
        action="store_true",
        help="Use log10(dt) before affine lag normalization for this AD run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--hf-weight", type=float, default=0.0)
    parser.add_argument("--hf-sg-weight", type=float, default=0.0)
    parser.add_argument("--hf-warmup-frac", type=float, default=0.1)
    parser.add_argument("--conserve-mean", action="store_true")
    parser.add_argument("--gl-film-mode", type=str, default="global_only", choices=["branchwise", "global_only"])
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
        data_dir=args.data_dir,
        log_lag=args.log_lag,
        dry_run=args.dry_run,
        hf_weight=args.hf_weight,
        hf_sg_weight=args.hf_sg_weight,
        hf_warmup_frac=args.hf_warmup_frac,
        conserve_mean=args.conserve_mean,
        gl_film_mode=args.gl_film_mode,
    )


if __name__ == "__main__":
    main()
