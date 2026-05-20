import os
import shutil
import random
import numpy as np
import torch

from due.datasets.pde import pde_dataset_osg
from due.models.pde_osg import PDE_osg
from due.networks.fno import osg_fno2d, osg_fno2d_with_film


# NS 数据文件
TRAIN_PATH = "VorticityOSG_train.mat"
TEST_PATH = "VorticityOSG_test.mat"

# 三个随机种子
SEEDS = [42]


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


def make_config(save_path: str, seed: int = 0):
    """
    DUE-style NS configuration:
      - epochs = 500
      - batch_size = 20
      - Adam + cosine
      - sg_pairing = 1
      - sg_weight = 1.0
    """
    return {
        "problem_type": "2d_regular",
        "problem_dim": 1,
        "multiscale": False,
        "dtype": "float32",

        "seed": seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        "epochs": 500,
        "batch_size": 20,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "scheduler": "cosine",
        "verbose": 10,

        # Keep this aligned with your current manuscript / implementation.
        "loss": "rel_l2",

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
    if model_name == "fno":
        return osg_fno2d(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )
    elif model_name == "fno_film":
        return osg_fno2d_with_film(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )
    else:
        raise ValueError("model_name must be 'fno' or 'fno_film'")


def train_one(model_name="fno", seed=0, overwrite=True):
    """
    Train one NS model.
    If overwrite=True, the whole save directory is removed first.
    """
    set_seed(seed)

    save_path = f"./runs_ns_{model_name}_seed{seed}"

    if overwrite:
        shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)

    config = make_config(save_path=save_path, seed=seed)

    dataset = pde_dataset_osg(config)
    trainX, trainY, coords, data_test, dt_test, vmin, vmax, tmin, tmax, cmin, cmax = dataset.load(
        TRAIN_PATH, TEST_PATH
    )

    print("============================================================")
    print(f"Model: {model_name}, seed={seed}")
    print("save_path =", save_path)
    print("coords.shape =", coords.shape)
    print("trainX.shape =", trainX.shape)
    print("trainY.shape =", trainY.shape)
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
    })
    print("============================================================")

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

    print(f"Finished training {model_name}, seed={seed}, saved to {save_path}")


if __name__ == "__main__":
    for seed in SEEDS:
        train_one("fno_film", seed=seed, overwrite=True)
        train_one("fno", seed=seed, overwrite=True)