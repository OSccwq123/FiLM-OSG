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
TRAIN_FILE = "VorticityOSG_train.mat"
TEST_FILE = "VorticityOSG_test.mat"


def resolve_data_paths(data_dir: str | os.PathLike[str]):
    data_root = Path(data_dir)
    return str(data_root / TRAIN_FILE), str(data_root / TEST_FILE)


def set_seed(seed: int):
    """Set all relevant random seeds before dataset loading and training."""
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
    hf_weight: float = 0.0,
    hf_sg_weight: float = 0.0,
    hf_warmup_frac: float = 0.1,
    conserve_mean: bool = False,
    gl_film_mode: str = "global_only",
    modes1: int = 12,
    modes2: int = 12,
    depth: int = 4,
    width: int = 20,
    local_kernel_size: int = 3,
    local_pool_factor: int = 2,
    gl_layer_scale: float = 1e-3,
    gl_post_activation: bool = True,
    local_padding_mode: str = "zeros",
    gl_coupling_mode: str = "raw",
    gl_coupling_scale: float = 1.0,
    gl_local_scale: float = 1.0,
    gl_local_film_mode: str = "affine",
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

        "modes1": int(modes1),
        "modes2": int(modes2),
        "depth": int(depth),
        "width": int(width),

        "local_kernel_size": int(local_kernel_size),
        "local_pool_factor": int(local_pool_factor),
        "gl_layer_scale": float(gl_layer_scale),
        "gl_post_activation": bool(gl_post_activation),
        "local_padding_mode": str(local_padding_mode),
        "gl_coupling_mode": str(gl_coupling_mode),
        "gl_coupling_scale": float(gl_coupling_scale),
        "gl_local_scale": float(gl_local_scale),
        "gl_local_film_mode": str(gl_local_film_mode),
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
    )

    print("network_import_source = film_osg", flush=True)

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
    raise ValueError("model_name must be one of fno, fno_film, gl_fno, gl_fno_film")


def train_one(
    model_name: str,
    seed: int,
    epochs: int = 500,
    batch_size: int = 20,
    tag: str = "",
    overwrite: bool = True,
    save_dir: str = ".",
    device: str | None = None,
    data_dir: str | os.PathLike[str] = DEFAULT_DATA_DIR,
    dry_run: bool = False,
    hf_weight: float = 0.0,
    hf_sg_weight: float = 0.0,
    hf_warmup_frac: float = 0.1,
    conserve_mean: bool = False,
    gl_film_mode: str = "global_only",
    modes1: int = 12,
    modes2: int = 12,
    depth: int = 4,
    width: int = 20,
    local_kernel_size: int = 3,
    local_pool_factor: int = 2,
    gl_layer_scale: float = 1e-3,
    gl_post_activation: bool = True,
    local_padding_mode: str = "zeros",
    gl_coupling_mode: str = "raw",
    gl_coupling_scale: float = 1.0,
    gl_local_scale: float = 1.0,
    gl_local_film_mode: str = "affine",
):
    suffix = f"_{tag}" if tag else ""
    save_path = os.path.join(save_dir, f"runs_ns_{model_name}_seed{seed}{suffix}")
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
        modes1=modes1,
        modes2=modes2,
        depth=depth,
        width=width,
        local_kernel_size=local_kernel_size,
        local_pool_factor=local_pool_factor,
        gl_layer_scale=gl_layer_scale,
        gl_post_activation=gl_post_activation,
        local_padding_mode=local_padding_mode,
        gl_coupling_mode=gl_coupling_mode,
        gl_coupling_scale=gl_coupling_scale,
        gl_local_scale=gl_local_scale,
        gl_local_film_mode=gl_local_film_mode,
    )

    if dry_run:
        print("Navier-Stokes FNO training dry run", flush=True)
        print("model =", model_name, flush=True)
        print("seed =", seed, flush=True)
        print("train_path =", train_path, flush=True)
        print("test_path =", test_path, flush=True)
        print("save_path =", save_path, flush=True)
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
        "local_kernel_size": config["local_kernel_size"],
        "local_pool_factor": config["local_pool_factor"],
        "gl_layer_scale": config["gl_layer_scale"],
        "gl_post_activation": config["gl_post_activation"],
        "local_padding_mode": config["local_padding_mode"],
        "gl_coupling_mode": config["gl_coupling_mode"],
        "gl_coupling_scale": config["gl_coupling_scale"],
        "gl_local_scale": config["gl_local_scale"],
        "gl_local_film_mode": config["gl_local_film_mode"],
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
    parser.add_argument("--model", type=str, required=True, choices=["fno", "fno_film", "gl_fno", "gl_fno_film"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--save-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--hf-weight", type=float, default=0.0)
    parser.add_argument("--hf-sg-weight", type=float, default=0.0)
    parser.add_argument("--hf-warmup-frac", type=float, default=0.1)
    parser.add_argument("--conserve-mean", action="store_true")
    parser.add_argument("--gl-film-mode", type=str, default="global_only", choices=["branchwise", "global_only"])
    parser.add_argument("--modes1", type=int, default=12)
    parser.add_argument("--modes2", type=int, default=12)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--local-kernel-size", type=int, default=3)
    parser.add_argument("--local-pool-factor", type=int, default=2)
    parser.add_argument("--gl-layer-scale", type=float, default=1e-3)
    parser.add_argument("--gl-no-post-activation", action="store_true")
    parser.add_argument("--local-padding-mode", type=str, default="zeros", choices=["zeros", "circular"])
    parser.add_argument("--gl-coupling-mode", type=str, default="raw", choices=["raw", "tanh"])
    parser.add_argument("--gl-coupling-scale", type=float, default=1.0)
    parser.add_argument("--gl-local-scale", type=float, default=1.0)
    parser.add_argument("--gl-local-film-mode", type=str, default="affine", choices=["affine", "gamma"])
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
        dry_run=args.dry_run,
        hf_weight=args.hf_weight,
        hf_sg_weight=args.hf_sg_weight,
        hf_warmup_frac=args.hf_warmup_frac,
        conserve_mean=args.conserve_mean,
        gl_film_mode=args.gl_film_mode,
        modes1=args.modes1,
        modes2=args.modes2,
        depth=args.depth,
        width=args.width,
        local_kernel_size=args.local_kernel_size,
        local_pool_factor=args.local_pool_factor,
        gl_layer_scale=args.gl_layer_scale,
        gl_post_activation=not args.gl_no_post_activation,
        local_padding_mode=args.local_padding_mode,
        gl_coupling_mode=args.gl_coupling_mode,
        gl_coupling_scale=args.gl_coupling_scale,
        gl_local_scale=args.gl_local_scale,
        gl_local_film_mode=args.gl_local_film_mode,
    )


if __name__ == "__main__":
    main()
