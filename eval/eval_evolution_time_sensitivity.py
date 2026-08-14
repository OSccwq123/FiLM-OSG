#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def safe_torch_load(path: Path, device: torch.device):
    return torch.load(path, map_location=device, weights_only=False)


def physical_dt(model, delta: torch.Tensor) -> torch.Tensor:
    value = delta * 0.5 * (model.tmax - model.tmin) + 0.5 * (model.tmax + model.tmin)
    if bool(model.multiscale):
        value = torch.pow(10.0, value)
    return value


def encode_dt(model, dt: np.ndarray) -> np.ndarray:
    value = np.log10(dt) if bool(model.multiscale) else dt
    return 2.0 * (value - 0.5 * (model.tmax + model.tmin)) / (model.tmax - model.tmin)


def normalize_state(model, state: np.ndarray) -> torch.Tensor:
    dtype = next(model.parameters()).dtype
    state_t = torch.from_numpy(state).to(dtype=dtype)
    vmin = model.vmin.detach().cpu()
    vmax = model.vmax.detach().cpu()
    return 2.0 * (state_t - 0.5 * (vmax[..., 0] + vmin[..., 0])) / (
        vmax[..., 0] - vmin[..., 0]
    )


def model_input(model, state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    spatial_shape = state.shape[1:-1]
    dt_shape = (state.shape[0],) + (1,) * len(spatial_shape) + (1,)
    dt_channel = delta.view(dt_shape).expand((state.shape[0],) + spatial_shape + (1,))
    return torch.cat((state, dt_channel), dim=-1)


def first_block_preactivation(model, state: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    spatial_dim = state.ndim - 2
    if spatial_dim not in (1, 2):
        raise ValueError(f"Expected one or two spatial dimensions, got state shape {tuple(state.shape)}")

    if hasattr(model, "time_encoder"):
        batch_size = state.shape[0]
        film_params = model.time_encoder(delta[:, None])
        film_params = film_params.view(batch_size, model.nblocks, 2, model.hid_dim)
        raw_gamma = film_params[:, 0, 0, :]
        raw_beta = film_params[:, 0, 1, :]
        if spatial_dim == 2:
            gamma = (1.0 + 0.1 * raw_gamma).view(batch_size, model.hid_dim, 1, 1)
            beta = (0.1 * raw_beta).view(batch_size, model.hid_dim, 1, 1)
            hidden = model.en(state).permute(0, 3, 1, 2)
        else:
            gamma = raw_gamma.view(batch_size, model.hid_dim, 1)
            beta = raw_beta.view(batch_size, model.hid_dim, 1)
            hidden = model.en(state).permute(0, 2, 1)
    else:
        gamma = beta = None
        encoded = model.en(model_input(model, state, delta))
        hidden = encoded.permute(0, 2, 1) if spatial_dim == 1 else encoded.permute(0, 3, 1, 2)

    spectral = model.mlp[0](model.conv[0](hidden))
    pointwise = model.w[0](hidden)
    preactivation = spectral + pointwise
    if gamma is not None:
        preactivation = gamma * preactivation + beta
    if spatial_dim == 1:
        return preactivation.permute(0, 2, 1)
    return preactivation.permute(0, 2, 3, 1)


def radial_spectrum_2d(sensitivity: torch.Tensor) -> np.ndarray:
    spectrum = torch.fft.fft2(sensitivity, dim=(1, 2), norm="ortho")
    energy_2d = torch.sum(torch.abs(spectrum) ** 2, dim=(0, 3))
    ny, nx = energy_2d.shape
    ky = torch.fft.fftfreq(ny, d=1.0 / ny, device=energy_2d.device)
    kx = torch.fft.fftfreq(nx, d=1.0 / nx, device=energy_2d.device)
    radius = torch.round(torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).to(torch.long)
    radial = torch.zeros(int(radius.max().item()) + 1, dtype=energy_2d.dtype, device=energy_2d.device)
    radial.scatter_add_(0, radius.reshape(-1), energy_2d.reshape(-1))
    return radial.detach().cpu().numpy()


def sample_pairs(data: dict, model, count: int, seed: int, fd_eps: float):
    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
    time_intervals = np.asarray(data["dt"], dtype=np.float32)
    ntraj, steps = time_intervals.shape
    all_n, all_t = np.meshgrid(np.arange(ntraj), np.arange(steps), indexing="ij")
    all_n = all_n.reshape(-1)
    all_t = all_t.reshape(-1)
    all_dt = time_intervals.reshape(-1)
    all_delta = encode_dt(model, all_dt)
    valid = np.flatnonzero(np.abs(all_delta) <= 1.0 - 2.0 * fd_eps)
    if count > len(valid):
        raise ValueError(f"Requested {count} pairs but only {len(valid)} are FD-safe")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid, size=count, replace=False)
    states = np.stack(
        [trajectories[all_n[i], ..., all_t[i]] for i in chosen], axis=0
    ).astype(np.float32)
    return states, all_delta[chosen].astype(np.float32), all_dt[chosen].astype(np.float32)


def model_sensitivity(
    model,
    states: torch.Tensor,
    deltas: torch.Tensor,
    chunk_size: int,
    fd_eps: float,
    stage: str,
):
    energy_sum = None
    sensitivity_sq_sum = 0.0
    sensitivity_count = 0
    fd_error_sq = 0.0
    fd_reference_sq = 0.0
    elapsed = 0.0

    original_conserve_mean = bool(getattr(model, "conserve_mean", False))
    model.conserve_mean = False
    model.eval()
    captured = {}

    def capture_decoder_output(_module, _inputs, output):
        captured["rate"] = output

    hook = model.de.register_forward_hook(capture_decoder_output) if stage == "decoder" else None

    try:
        for start in range(0, states.shape[0], chunk_size):
            state = states[start : start + chunk_size]
            delta = deltas[start : start + chunk_size]

            def rate_fn(query_delta):
                if stage == "block0_preactivation":
                    return first_block_preactivation(model, state, query_delta)
                if stage == "decoder":
                    model(model_input(model, state, query_delta))
                    rate = captured["rate"]
                    if rate.ndim == 3:
                        return rate.permute(0, 2, 1)
                    if rate.ndim == 4:
                        return rate.permute(0, 2, 3, 1)
                    raise ValueError(f"Unexpected decoder output shape: {tuple(rate.shape)}")
                raise ValueError(f"Unknown stage: {stage}")

            torch.cuda.synchronize() if state.is_cuda else None
            t0 = time.perf_counter()
            _, sensitivity = torch.autograd.functional.jvp(
                rate_fn,
                (delta,),
                (torch.ones_like(delta),),
                create_graph=False,
                strict=False,
            )
            torch.cuda.synchronize() if state.is_cuda else None
            elapsed += time.perf_counter() - t0

            with torch.no_grad():
                fd = (rate_fn(delta + fd_eps) - rate_fn(delta - fd_eps)) / (2.0 * fd_eps)

            diff = sensitivity - fd
            fd_error_sq += float(torch.sum(diff * diff).cpu())
            fd_reference_sq += float(torch.sum(fd * fd).cpu())
            sensitivity_sq_sum += float(torch.sum(sensitivity * sensitivity).cpu())
            sensitivity_count += sensitivity.numel()

            if sensitivity.ndim == 3:
                spectrum = torch.fft.rfft(sensitivity, dim=1, norm="ortho")
                energy = torch.sum(torch.abs(spectrum) ** 2, dim=(0, 2)).detach().cpu().numpy()
            elif sensitivity.ndim == 4:
                energy = radial_spectrum_2d(sensitivity)
            else:
                raise ValueError(f"Unexpected sensitivity shape: {tuple(sensitivity.shape)}")
            energy_sum = energy if energy_sum is None else energy_sum + energy
    finally:
        if hook is not None:
            hook.remove()
        model.conserve_mean = original_conserve_mean

    return {
        "energy": energy_sum,
        "sensitivity_rms": float(np.sqrt(sensitivity_sq_sum / sensitivity_count)),
        "fd_relative_error": float(np.sqrt(fd_error_sq / max(fd_reference_sq, 1e-30))),
        "jvp_seconds": elapsed,
        "projection_bypassed": original_conserve_mean,
    }


def spectrum_metrics(energy: np.ndarray, modes: int):
    total = float(np.sum(energy))
    normalized = energy / max(total, 1e-30)
    retained_end = min(int(modes), len(energy))
    retained = energy[:retained_end]
    retained_total = float(np.sum(retained))
    high_start = max(1, int(np.ceil(2.0 * retained_end / 3.0)))
    k = np.arange(len(energy), dtype=np.float64)
    retained_k = np.arange(retained_end, dtype=np.float64)
    return {
        "normalized": normalized,
        "zero_mode_fraction_full": float(normalized[0]),
        "nonzero_fraction_retained": float(np.sum(retained[1:]) / max(retained_total, 1e-30)),
        "high_fraction_retained": float(np.sum(retained[high_start:]) / max(retained_total, 1e-30)),
        "centroid_full": float(np.sum(k * energy) / max(total, 1e-30)),
        "centroid_retained": float(np.sum(retained_k * retained) / max(retained_total, 1e-30)),
        "retained_end_exclusive": retained_end,
        "high_start": high_start,
    }


def main():
    parser = argparse.ArgumentParser(description="Layerwise sensitivity to evolution time")
    parser.add_argument("--benchmark", default="original_burgers")
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--direct-model", type=Path, required=True)
    parser.add_argument("--film-model", type=Path, required=True)
    parser.add_argument("--data", default="data/BurgersOSG_test.mat")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--sample-seed", type=int, default=20260620)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--fd-eps", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="eval_outputs_evolution_time_sensitivity/burgers_original_seed0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    direct = safe_torch_load(args.direct_model, device).to(device)
    film = safe_torch_load(args.film_model, device).to(device)
    data = loadmat(args.data)

    states_np, delta_np, dt_np = sample_pairs(data, direct, args.samples, args.sample_seed, args.fd_eps)
    film_delta = encode_dt(film, dt_np)
    if not np.allclose(delta_np, film_delta, rtol=1e-6, atol=1e-6):
        raise ValueError("Input-concatenation and FiLM checkpoints use different time transformations")

    direct_states = normalize_state(direct, states_np).to(device)
    film_states = normalize_state(film, states_np).to(device)
    deltas = torch.from_numpy(delta_np).to(device=device, dtype=next(direct.parameters()).dtype)

    outputs = {}
    stages = ("block0_preactivation", "decoder")
    model_specs = (
        ("Input-concatenation OSG-FNO", direct, direct_states),
        ("FiLM-OSG-FNO", film, film_states),
    )
    for stage in stages:
        outputs[stage] = {}
        for label, model, states in model_specs:
            raw = model_sensitivity(model, states, deltas, args.chunk_size, args.fd_eps, stage)
            metrics = spectrum_metrics(raw["energy"], getattr(model, "modes1", 10))
            outputs[stage][label] = {
                **{k: v for k, v in raw.items() if k != "energy"},
                **{k: v for k, v in metrics.items() if k != "normalized"},
                "energy": [float(x) for x in raw["energy"]],
                "normalized_energy": [float(x) for x in metrics["normalized"]],
            }

    metadata = {
        "benchmark": args.benchmark,
        "model_seed": args.model_seed,
        "direct_checkpoint": str(args.direct_model),
        "film_checkpoint": str(args.film_model),
        "data_path": str(Path(args.data)),
        "sample_count": args.samples,
        "sample_seed": args.sample_seed,
        "fd_eps_normalized_delta": args.fd_eps,
        "physical_dt_min": float(np.min(dt_np)),
        "physical_dt_max": float(np.max(dt_np)),
        "normalized_delta_min": float(np.min(delta_np)),
        "normalized_delta_max": float(np.max(delta_np)),
        "device": str(device),
        "stages": outputs,
    }
    (out_dir / "evolution_time_sensitivity_summary.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    labels = list(outputs[stages[0]])
    with (out_dir / "evolution_time_sensitivity_spectrum.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "wavenumber", *labels])
        nfreq = len(outputs[stages[0]][labels[0]]["normalized_energy"])
        for stage in stages:
            for k in range(nfreq):
                writer.writerow([stage, k, *[outputs[stage][label]["normalized_energy"][k] for label in labels]])

    with (out_dir / "evolution_time_sensitivity_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "stage", "model", "sensitivity_rms", "fd_relative_error",
            "zero_mode_fraction_full", "nonzero_fraction_retained",
            "high_fraction_retained", "centroid_full", "centroid_retained",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in stages:
            for label in labels:
                values = outputs[stage][label]
                writer.writerow({"stage": stage, "model": label, **{key: values[key] for key in fields[2:]}})

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), sharey=True)
    titles = {
        "block0_preactivation": "First-block output before terminal activation",
        "decoder": "Final decoder increment",
    }
    colors = ["#4C78A8", "#D1495B"]
    for ax, stage in zip(axes, stages):
        for label, color in zip(labels, colors):
            energy = np.asarray(outputs[stage][label]["normalized_energy"])
            ax.semilogy(np.arange(len(energy)), np.maximum(energy, 1e-12), marker="o", ms=3, lw=1.8, label=label, color=color)
        modes = int(outputs[stage][labels[0]]["retained_end_exclusive"])
        ax.axvspan(-0.5, modes - 0.5, color="#D9E6F2", alpha=0.35, label="retained band")
        ax.set_xlabel("Wavenumber $k$")
        ax.set_title(titles[stage])
        ax.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Normalized evolution-time sensitivity energy")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    benchmark_title = args.benchmark.replace("_", " ").title()
    fig.suptitle(
        f"{benchmark_title}: layerwise evolution-time sensitivity (seed {args.model_seed})",
        y=1.13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "evolution_time_sensitivity_layerwise.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "evolution_time_sensitivity_layerwise.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)

    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
