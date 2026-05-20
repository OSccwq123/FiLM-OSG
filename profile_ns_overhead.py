import os
import csv
import json
import time
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F

from due.datasets.pde import pde_dataset_osg
from due.networks.fno import osg_fno2d, osg_fno2d_with_film

from due.networks.osg_extra_backbones import (
    osg_uno2d,
    osg_uno2d_with_film,
    osg_mambano2d,
    osg_mambano2d_with_film,
    osg_transolver2d,
    osg_transolver2d_with_film,
)


TRAIN_PATH = "VorticityOSG_train.mat"
TEST_PATH = "VorticityOSG_test.mat"


DEFAULT_MODELS = [
    "fno",
    "fno_film",
    "uno",
    "uno_film",
    "transolver",
    "transolver_film",
]

ALL_MODELS = [
    "fno",
    "fno_film",
    "uno",
    "uno_film",
    "transolver",
    "transolver_film",
    "mambano",
    "mambano_film",
]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_config(model_name: str, save_path: str, seed: int, batch_size: int):
    """
    NS profiling configuration aligned with the current training setup.

    FNO uses the current NS-FNO setting:
        modes1=modes2=12, depth=4, width=20, loss=rel_l2.

    Extra backbones use the current OSG-compatible extra-backbone setting:
        modes1=modes2=8 for U-NO-style,
        n_head=2 and slice_num=32 for Transolver-style,
        Mamba parameters as in train_ns_extra_backbones.py.
    """
    if model_name in ["fno", "fno_film"]:
        modes1 = 12
        modes2 = 12
    else:
        modes1 = 8
        modes2 = 8

    return {
        "problem_type": "2d_regular",
        "problem_dim": 1,
        "multiscale": False,
        "dtype": "float32",

        "seed": seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        "epochs": 1,
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

        "modes1": modes1,
        "modes2": modes2,
        "depth": 4,
        "width": 20,

        # Extra-backbone keys. Harmless for FNO.
        "uno_norm": False,

        "n_head": 2,
        "slice_num": 32,
        "dropout": 0.0,
        "mlp_ratio": 4,

        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,

        "time_width": 20,

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

    if model_name == "fno_film":
        return osg_fno2d_with_film(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
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

    if model_name == "mambano":
        return osg_mambano2d(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    if model_name == "mambano_film":
        return osg_mambano2d_with_film(
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            config=config,
            multiscale=config["multiscale"],
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rel_l2_loss(pred, target, eps=1e-12):
    """
    Mean per-sample relative L2 loss over a batch.
    """
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    num = torch.linalg.norm(pred_flat - target_flat, dim=1)
    den = torch.linalg.norm(target_flat, dim=1) + eps
    return torch.mean(num / den)


def decode_dt(dt_norm, tmin, tmax, multiscale=False):
    dt = dt_norm * 0.5 * (tmax - tmin) + 0.5 * (tmax + tmin)
    if multiscale:
        dt = 10.0 ** dt
    return dt


def encode_dt(dt, tmin, tmax, multiscale=False):
    if multiscale:
        dt = torch.log10(dt)
    return 2.0 * (dt - 0.5 * (tmax + tmin)) / (tmax - tmin)


def make_sg_batch(xb, tmin, tmax, multiscale=False):
    """
    Build a lightweight auxiliary semigroup batch.

    xb is expected to have shape (B,H,W,C+1), where the last channel is the
    normalized lag code. We use the state part from xb and construct two lags
    from the observed normalized lag code and a shuffled copy.
    """
    x0 = xb[..., :-1]
    dt1_norm = xb[..., -1:]

    perm = torch.randperm(xb.shape[0], device=xb.device)
    dt2_norm = xb[perm, ..., -1:]

    dt1 = decode_dt(dt1_norm, tmin, tmax, multiscale)
    dt2 = decode_dt(dt2_norm, tmin, tmax, multiscale)
    dt12_norm = encode_dt(dt1 + dt2, tmin, tmax, multiscale)

    x_step1 = torch.cat((x0, dt1_norm), dim=-1)
    x_direct = torch.cat((x0, dt12_norm), dim=-1)

    return x_step1, dt2_norm, x_direct


def training_step(model, xb, yb, optimizer, tmin, tmax, multiscale, sg_mode="aux", sg_weight=1.0):
    optimizer.zero_grad(set_to_none=True)

    pred = model(xb)
    loss_data = rel_l2_loss(pred, yb)

    if sg_mode == "none":
        loss = loss_data
    elif sg_mode == "aux":
        x_step1, dt2_norm, x_direct = make_sg_batch(
            xb,
            tmin=tmin,
            tmax=tmax,
            multiscale=multiscale,
        )

        pred_step1 = model(x_step1)

        x_step2 = torch.cat((pred_step1, dt2_norm), dim=-1)
        pred_two_step = model(x_step2)

        pred_direct = model(x_direct)

        loss_sg = rel_l2_loss(pred_two_step, pred_direct)
        loss = (loss_data + sg_weight * loss_sg) / (1.0 + sg_weight)
    else:
        raise ValueError("sg_mode must be 'none' or 'aux'.")

    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu())


def cuda_sync(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def time_training_step(
    model,
    xb,
    yb,
    optimizer,
    tmin,
    tmax,
    multiscale,
    device,
    warmup,
    iters,
    sg_mode,
    sg_weight,
):
    model.train()

    for _ in range(warmup):
        training_step(
            model,
            xb,
            yb,
            optimizer,
            tmin=tmin,
            tmax=tmax,
            multiscale=multiscale,
            sg_mode=sg_mode,
            sg_weight=sg_weight,
        )

    cuda_sync(device)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    for _ in range(iters):
        training_step(
            model,
            xb,
            yb,
            optimizer,
            tmin=tmin,
            tmax=tmax,
            multiscale=multiscale,
            sg_mode=sg_mode,
            sg_weight=sg_weight,
        )

    cuda_sync(device)

    end = time.perf_counter()
    train_ms = (end - start) * 1000.0 / iters

    if device.startswith("cuda") and torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / (1024.0 ** 3)
    else:
        peak_gb = float("nan")

    return train_ms, peak_gb


def time_inference_step(model, xb, device, warmup, iters):
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(xb)

        cuda_sync(device)

        start = time.perf_counter()

        for _ in range(iters):
            _ = model(xb)

        cuda_sync(device)

        end = time.perf_counter()

    return (end - start) * 1000.0 / iters


def profile_one_model(
    model_name,
    trainX,
    trainY,
    vmin,
    vmax,
    tmin,
    tmax,
    batch_size,
    seed,
    device,
    warmup,
    iters,
    sg_mode,
    sg_weight,
):
    set_seed(seed)

    config = make_config(
        model_name=model_name,
        save_path=f"./profile_tmp_{model_name}",
        seed=seed,
        batch_size=batch_size,
    )

    model = build_model(
        model_name=model_name,
        vmin=vmin,
        vmax=vmax,
        tmin=tmin,
        tmax=tmax,
        config=config,
    ).to(device)

    params = count_trainable_params(model)

    xb = torch.from_numpy(trainX[:batch_size]).float().to(device)
    yb = torch.from_numpy(trainY[:batch_size]).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    print("=" * 80, flush=True)
    print(f"Profiling model: {model_name}", flush=True)
    print("batch_size =", batch_size, flush=True)
    print("params =", params, flush=True)
    print("sg_mode =", sg_mode, flush=True)
    print("device =", device, flush=True)
    print("=" * 80, flush=True)

    train_ms, peak_gb = time_training_step(
        model=model,
        xb=xb,
        yb=yb,
        optimizer=optimizer,
        tmin=tmin,
        tmax=tmax,
        multiscale=config["multiscale"],
        device=device,
        warmup=warmup,
        iters=iters,
        sg_mode=sg_mode,
        sg_weight=sg_weight,
    )

    infer_ms = time_inference_step(
        model=model,
        xb=xb,
        device=device,
        warmup=warmup,
        iters=iters,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result = {
        "model": model_name,
        "params": int(params),
        "params_M": params / 1e6,
        "train_step_ms": float(train_ms),
        "inference_step_ms": float(infer_ms),
        "peak_train_memory_GB": float(peak_gb),
        "batch_size": int(batch_size),
        "warmup": int(warmup),
        "timed_iters": int(iters),
        "sg_mode": sg_mode,
    }

    print(
        f"{model_name}: "
        f"params={result['params_M']:.3f}M, "
        f"train={train_ms:.3f} ms, "
        f"infer={infer_ms:.3f} ms, "
        f"peak_mem={peak_gb:.3f} GB",
        flush=True,
    )

    del model
    del optimizer
    del xb
    del yb

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="./overhead_outputs_ns")
    parser.add_argument("--sg-mode", type=str, default="aux", choices=["none", "aux"])
    parser.add_argument("--sg-weight", type=float, default=1.0)

    args = parser.parse_args()

    models = parse_str_list(args.models)

    for m in models:
        if m not in ALL_MODELS:
            raise ValueError(f"Unknown model {m}. Choices: {ALL_MODELS}")

    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)

    # Use a FNO-compatible config only for loading the same NS data.
    load_config = make_config(
        model_name="fno",
        save_path="./profile_tmp_loader",
        seed=args.seed,
        batch_size=args.batch_size,
    )

    dataset = pde_dataset_osg(load_config)
    trainX, trainY, coords, data_test, dt_test, vmin, vmax, tmin, tmax, cmin, cmax = dataset.load(
        TRAIN_PATH,
        TEST_PATH,
    )

    print("=" * 80, flush=True)
    print("Navier--Stokes overhead profiling", flush=True)
    print("Models:", models, flush=True)
    print("Train data:", TRAIN_PATH, flush=True)
    print("Test data:", TEST_PATH, flush=True)
    print("trainX.shape:", trainX.shape, flush=True)
    print("trainY.shape:", trainY.shape, flush=True)
    print("batch_size:", args.batch_size, flush=True)
    print("warmup:", args.warmup, flush=True)
    print("timed_iters:", args.iters, flush=True)
    print("sg_mode:", args.sg_mode, flush=True)
    print("device:", args.device, flush=True)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("=" * 80, flush=True)

    rows = []

    for model_name in models:
        row = profile_one_model(
            model_name=model_name,
            trainX=trainX,
            trainY=trainY,
            vmin=vmin,
            vmax=vmax,
            tmin=tmin,
            tmax=tmax,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            warmup=args.warmup,
            iters=args.iters,
            sg_mode=args.sg_mode,
            sg_weight=args.sg_weight,
        )
        rows.append(row)

    csv_path = os.path.join(args.save_dir, "ns_overhead_profile.csv")
    json_path = os.path.join(args.save_dir, "ns_overhead_profile.json")

    write_csv(csv_path, rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark": "Navier--Stokes",
                "train_path": TRAIN_PATH,
                "test_path": TEST_PATH,
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "timed_iters": args.iters,
                "sg_mode": args.sg_mode,
                "sg_weight": args.sg_weight,
                "device": args.device,
                "results": rows,
            },
            f,
            indent=2,
        )

    print("\nSaved outputs:", flush=True)
    print(" ", csv_path, flush=True)
    print(" ", json_path, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()