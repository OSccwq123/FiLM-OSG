#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from film_osg.models.pde_osg import PDE_osg
from film_osg.networks.fno import (
    gl_osg_fno1d,
    gl_osg_fno1d_with_film,
    osg_fno1d,
    osg_fno1d_with_film,
    osg_fno2d,
    osg_fno2d_with_film,
)


def cfg1(save_path, conserve_mean=False, hf=True, gl_film_mode="global_only"):
    return {
        "problem_type": "1d_regular", "problem_dim": 1, "multiscale": False,
        "dtype": "float32", "seed": 0, "device": "cpu", "epochs": 1,
        "batch_size": 2, "learning_rate": 1e-3, "optimizer": "adam",
        "scheduler": "none", "verbose": 1, "loss": "mae",
        "nbursts": 2, "sg_pairing": 1, "sg_weight": 1.0,
        "activation": "gelu", "modes": 4, "depth": 2, "width": 8,
        "local_kernel_size": 5, "local_pool_factor": 2, "gl_layer_scale": 1e-3,
        "hf_weight": 1e-2 if hf else 0.0, "hf_sg_weight": 1e-3 if hf else 0.0,
        "hf_warmup_frac": 0.0, "hf_band_frac": 1/3, "hf_power": 2.0,
        "conserve_mean": conserve_mean, "gl_film_mode": gl_film_mode,
        "save_path": str(save_path),
    }


def cfg2(save_path, conserve_mean=False, hf=True, gl_film_mode="global_only"):
    return {
        "problem_type": "2d_regular", "problem_dim": 1, "multiscale": False,
        "dtype": "float32", "seed": 0, "device": "cpu", "epochs": 1,
        "batch_size": 2, "learning_rate": 1e-3, "optimizer": "adam",
        "scheduler": "none", "verbose": 1, "loss": "mae",
        "nbursts": 2, "sg_pairing": 1, "sg_weight": 1.0,
        "activation": "gelu", "modes1": 4, "modes2": 4, "depth": 2, "width": 8,
        "local_kernel_size": 3, "local_pool_factor": 2, "gl_layer_scale": 1e-3,
        "hf_weight": 1e-3 if hf else 0.0, "hf_sg_weight": 1e-4 if hf else 0.0,
        "hf_warmup_frac": 0.0, "hf_band_frac": 1/3, "hf_power": 2.0,
        "conserve_mean": conserve_mean, "gl_film_mode": gl_film_mode,
        "save_path": str(save_path),
    }


def finite(x):
    return bool(torch.isfinite(x).all().item())


def grad_norm(model):
    total = 0.0
    nonzero = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().norm().cpu())
        total += g
        nonzero += int(g > 0.0)
    return total, nonzero


def named_grad_norm(model, text):
    total = 0.0
    nonzero = 0
    for name, p in model.named_parameters():
        if text not in name or p.grad is None:
            continue
        g = float(p.grad.detach().norm().cpu())
        total += g
        nonzero += int(g > 0.0)
    return total, nonzero


def assert_forward_backward(name, model, x, target, dt_index_dims):
    model.train()
    x = x.clone().requires_grad_(True)
    y = model(x)
    assert y.shape == target.shape, f"{name}: shape {tuple(y.shape)} != {tuple(target.shape)}"
    assert finite(y), f"{name}: non-finite forward"
    loss = torch.nn.functional.mse_loss(y, target)
    pred01 = model(x)
    x2 = torch.cat([pred01.detach() * 0.0 + pred01, x[..., -1:]], dim=-1)
    pred012 = model(x2)
    pred2 = model(x)
    loss = loss + 0.1 * torch.nn.functional.mse_loss(pred012, pred2)
    model.zero_grad(set_to_none=True)
    loss.backward()
    total, nonzero = grad_norm(model)
    assert total > 0 and nonzero > 0, f"{name}: no parameter gradient"
    assert x.grad is not None and float(x.grad.norm()) > 0, f"{name}: no input gradient"
    if getattr(model, "conserve_mean", False):
        inc = y - x[..., :-1]
        spatial = tuple(range(1, inc.ndim - 1))
        mean_abs = float(inc.mean(dim=spatial).abs().max().detach().cpu())
        assert mean_abs < 1e-5, f"{name}: projection mean drift {mean_abs}"
    return {"loss": float(loss.detach().cpu()), "grad_norm": total, "nonzero_grads": nonzero}


def pde_hf_smoke(name, model, cfg, spatial_shape):
    n = 2
    if len(spatial_shape) == 1:
        L = spatial_shape[0]
        train_x = np.random.randn(n, L, 2).astype("float32")
        train_y = np.random.randn(n, L, 1).astype("float32")
    else:
        H, W = spatial_shape
        train_x = np.random.randn(n, H, W, 2).astype("float32")
        train_y = np.random.randn(n, H, W, 1).astype("float32")
    wrapper = PDE_osg(train_x, train_y, model, cfg)
    xx = torch.from_numpy(train_x)
    yy = torch.from_numpy(train_y)
    pred = wrapper.mynet(xx)
    hf = wrapper._high_frequency_loss(pred, yy)
    assert hf.ndim == 0 and finite(hf), f"{name}: bad hf loss"
    loss = torch.nn.functional.mse_loss(pred, yy) + cfg["hf_weight"] * hf
    wrapper.mynet.zero_grad(set_to_none=True)
    loss.backward()
    total, nonzero = grad_norm(wrapper.mynet)
    assert total > 0 and nonzero > 0, f"{name}: no gradient through HF smoke"
    return {"hf_loss": float(hf.detach()), "hf_grad_norm": total, "hf_nonzero_grads": nonzero}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default="logs/smoke_new_features.json")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    tmp = Path(tempfile.mkdtemp(prefix="film_osg_smoke_"))
    vmin = np.array([[-1.0]], dtype=np.float32)
    vmax = np.array([[1.0]], dtype=np.float32)
    tmin, tmax = 0.001, 0.1
    results = {}

    x1 = torch.randn(2, 16, 2)
    x1[..., -1:] = 0.0
    y1 = torch.randn(2, 16, 1)
    x2 = torch.randn(2, 8, 8, 2)
    x2[..., -1:] = 0.0
    y2 = torch.randn(2, 8, 8, 1)

    specs = [
        ("fno1d_projection", osg_fno1d(vmin, vmax, tmin, tmax, cfg1(tmp/"a", True)), x1, y1),
        ("film1d_projection", osg_fno1d_with_film(vmin, vmax, tmin, tmax, cfg1(tmp/"b", True)), x1, y1),
        ("gl_fno1d_projection", gl_osg_fno1d(vmin, vmax, tmin, tmax, cfg1(tmp/"c", True)), x1, y1),
        ("gl_film1d_global_only", gl_osg_fno1d_with_film(vmin, vmax, tmin, tmax, cfg1(tmp/"d", True, gl_film_mode="global_only")), x1, y1),
        ("gl_film1d_branchwise", gl_osg_fno1d_with_film(vmin, vmax, tmin, tmax, cfg1(tmp/"e", True, gl_film_mode="branchwise")), x1, y1),
        ("fno2d_projection", osg_fno2d(vmin, vmax, tmin, tmax, cfg2(tmp/"f", True)), x2, y2),
        ("film2d_projection", osg_fno2d_with_film(vmin, vmax, tmin, tmax, cfg2(tmp/"g", True)), x2, y2),
    ]
    for name, model, x, y in specs:
        results[name] = assert_forward_backward(name, model, x, y, None)
        if "gl_film" in name:
            results[name]["time_encoder_grad"] = named_grad_norm(model, "time_encoder")
        print("PASS", name, results[name], flush=True)

    results["pde_hf_1d"] = pde_hf_smoke("pde_hf_1d", gl_osg_fno1d_with_film(vmin, vmax, tmin, tmax, cfg1(tmp/"k", True)), cfg1(tmp/"k", True), (16,))
    print("PASS pde_hf_1d", results["pde_hf_1d"], flush=True)
    results["pde_hf_2d"] = pde_hf_smoke(
        "pde_hf_2d",
        osg_fno2d_with_film(vmin, vmax, tmin, tmax, cfg2(tmp/"l", True)),
        cfg2(tmp/"l", True),
        (8, 8),
    )
    print("PASS pde_hf_2d", results["pde_hf_2d"], flush=True)

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("SMOKE_OK", out, flush=True)


if __name__ == "__main__":
    main()
