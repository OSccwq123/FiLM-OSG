import os
import random
import shutil
import numpy as np
import torch

from due.datasets.pde import pde_dataset_osg
from due.models.pde_osg import PDE_osg
from due.networks.fno import osg_fno2d, osg_fno2d_with_film


TRAIN_PATH = "VorticityOSG_train.mat"
TEST_PATH  = "VorticityOSG_test.mat"

#SEEDS = [0, 1, 42]
SEEDS = [2]
GPU_ID = 0
OVERWRITE = True


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    if torch.cuda.is_available():
        return f"cuda:{GPU_ID}"
    return "cpu"


def make_config(save_path, seed=0, use_sg=True):
    return {
        "problem_type": "2d_regular",
        "problem_dim": 1,
        "multiscale": False,
        "dtype": "float32",

        "seed": seed,
        "device": get_device(),

        "epochs": 500,
        "batch_size": 20,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "scheduler": "cosine",
        "verbose": 10,
        "loss": "rel_l2",

        "nbursts": 25,
        "sg_pairing": 1,
        "sg_weight": 1.0 if use_sg else 0.0,

        "activation": "gelu",

        # FNO-2D
        "modes1": 12,
        "modes2": 12,
        "depth": 4,
        "width": 20,

        "save_path": save_path,
    }


def build_model(model_name, vmin, vmax, tmin, tmax, config):
    if model_name == "fno":
        return osg_fno2d(
            vmin=vmin, vmax=vmax, tmin=tmin, tmax=tmax,
            config=config, multiscale=config["multiscale"]
        )
    elif model_name == "fno_film":
        return osg_fno2d_with_film(
            vmin=vmin, vmax=vmax, tmin=tmin, tmax=tmax,
            config=config, multiscale=config["multiscale"]
        )
    else:
        raise ValueError("model_name must be 'fno' or 'fno_film'")


def variant_to_flags(variant_name):
    if variant_name == "direct_nosg":
        return "fno", False
    elif variant_name == "direct_sg":
        return "fno", True
    elif variant_name == "film_nosg":
        return "fno_film", False
    elif variant_name == "film_sg":
        return "fno_film", True
    else:
        raise ValueError(f"Unknown variant_name: {variant_name}")


def train_one(variant_name="direct_sg", seed=0):
    model_name, use_sg = variant_to_flags(variant_name)

    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)

    save_path = f"./runs_ns_{variant_name}_seed{seed}"

    if OVERWRITE:
        shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)

    config = make_config(save_path=save_path, seed=seed, use_sg=use_sg)

    dataset = pde_dataset_osg(config)
    trainX, trainY, coords, data_test, dt_test, vmin, vmax, tmin, tmax, cmin, cmax = dataset.load(
        TRAIN_PATH, TEST_PATH
    )

    print("============================================================")
    print(f"Variant: {variant_name}, seed={seed}")
    print("device =", config["device"])
    print("coords.shape =", coords.shape)
    print("trainX.shape =", trainX.shape)
    print("trainY.shape =", trainY.shape)
    print("sg_weight =", config["sg_weight"])
    print("save_path =", save_path)
    print("============================================================")

    net = build_model(model_name, vmin, vmax, tmin, tmax, config)

    solver = PDE_osg(trainX, trainY, osg_data=None, network=net, config=config)
    solver.train()
    solver.save_hist()

    print(f"Finished training {variant_name}, seed={seed}, saved to {save_path}")


if __name__ == "__main__":
    variants = ["direct_nosg", "film_nosg"]

    for seed in SEEDS:
        for variant in variants:
            train_one(variant_name=variant, seed=seed)