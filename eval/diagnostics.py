"""Shared rollout diagnostics for FiLM-OSG experiments."""

from __future__ import annotations

import numpy as np


def _roll(pred, truth):
    return np.asarray(pred)[..., 1:], np.asarray(truth)[..., 1:]


def mean_drift_metrics(pred, truth, eps=1e-12):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    spatial_axes = tuple(range(1, pred.ndim - 2))
    pred_mean = pred.mean(axis=spatial_axes)
    true_mean = truth.mean(axis=spatial_axes)
    init_mean = pred_mean[..., :1]
    drift_from_init = np.abs(pred_mean - init_mean)
    drift_from_ref = np.abs(pred_mean - true_mean)
    return {
        "Mean Drift Max": float(drift_from_init[..., 1:].max()),
        "Mean Drift Avg": float(drift_from_init[..., 1:].mean()),
        "Mean Drift Final": float(drift_from_init[..., -1].mean()),
        "Mean Ref Drift Avg": float(drift_from_ref[..., 1:].mean()),
        "Mean Ref Drift Final": float(drift_from_ref[..., -1].mean()),
    }


def _band_masks_1d(n, low_frac=1.0 / 3.0, high_frac=2.0 / 3.0):
    k = np.fft.rfftfreq(n)
    r = k / (k.max() + 1e-12)
    return {
        "Low Band Rel-L2": r < low_frac,
        "Mid Band Rel-L2": (r >= low_frac) & (r < high_frac),
        "High Band Rel-L2": r >= high_frac,
    }


def _band_masks_2d(nx, ny, low_frac=1.0 / 3.0, high_frac=2.0 / 3.0):
    kx = np.fft.fftfreq(nx)[:, None]
    ky = np.fft.rfftfreq(ny)[None, :]
    rad = np.sqrt(kx * kx + ky * ky)
    r = rad / (rad.max() + 1e-12)
    return {
        "Low Band Rel-L2": r < low_frac,
        "Mid Band Rel-L2": (r >= low_frac) & (r < high_frac),
        "High Band Rel-L2": r >= high_frac,
    }


def band_error_metrics(pred, truth, eps=1e-12):
    pred_roll, true_roll = _roll(pred, truth)
    err = pred_roll - true_roll
    ndim_spatial = err.ndim - 3
    vals = {"Low Band Rel-L2": [], "Mid Band Rel-L2": [], "High Band Rel-L2": []}
    final_vals = {"Final Low Band Rel-L2": [], "Final Mid Band Rel-L2": [], "Final High Band Rel-L2": []}
    ntraj = err.shape[0]
    steps = err.shape[-1]
    for n in range(ntraj):
        for t in range(steps):
            e_state = err[n, ..., t]
            y_state = true_roll[n, ..., t]
            if ndim_spatial == 1:
                ehat = np.fft.rfft(e_state, axis=0)
                yhat = np.fft.rfft(y_state, axis=0)
                masks = _band_masks_1d(e_state.shape[0])
                for key, mask in masks.items():
                    vals[key].append(np.linalg.norm(ehat[mask].reshape(-1)) / (np.linalg.norm(yhat[mask].reshape(-1)) + eps))
            elif ndim_spatial == 2:
                e_ch = np.moveaxis(e_state, -1, 0)
                y_ch = np.moveaxis(y_state, -1, 0)
                ehat = np.fft.rfft2(e_ch, axes=(-2, -1))
                yhat = np.fft.rfft2(y_ch, axes=(-2, -1))
                masks = _band_masks_2d(e_state.shape[0], e_state.shape[1])
                for key, mask in masks.items():
                    vals[key].append(np.linalg.norm(ehat[:, mask].reshape(-1)) / (np.linalg.norm(yhat[:, mask].reshape(-1)) + eps))
        # final state
        e_state = err[n, ..., -1]
        y_state = true_roll[n, ..., -1]
        if ndim_spatial == 1:
            ehat = np.fft.rfft(e_state, axis=0); yhat = np.fft.rfft(y_state, axis=0); masks = _band_masks_1d(e_state.shape[0])
            for key, mask in masks.items():
                final_vals["Final " + key].append(np.linalg.norm(ehat[mask].reshape(-1)) / (np.linalg.norm(yhat[mask].reshape(-1)) + eps))
        elif ndim_spatial == 2:
            e_ch = np.moveaxis(e_state, -1, 0); y_ch = np.moveaxis(y_state, -1, 0)
            ehat = np.fft.rfft2(e_ch, axes=(-2, -1)); yhat = np.fft.rfft2(y_ch, axes=(-2, -1)); masks = _band_masks_2d(e_state.shape[0], e_state.shape[1])
            for key, mask in masks.items():
                final_vals["Final " + key].append(np.linalg.norm(ehat[:, mask].reshape(-1)) / (np.linalg.norm(yhat[:, mask].reshape(-1)) + eps))
    out = {k: float(np.mean(v)) for k, v in vals.items()}
    out.update({k: float(np.mean(v)) for k, v in final_vals.items()})
    out["HF Rel-L2"] = out["High Band Rel-L2"]
    return out


def spectrum_error_metrics(pred, truth, eps=1e-12):
    pred_roll, true_roll = _roll(pred, truth)
    if pred_roll.ndim - 3 != 2:
        return {}
    vals = []
    final_vals = []
    ntraj = pred_roll.shape[0]
    steps = pred_roll.shape[-1]
    for n in range(ntraj):
        for t in range(steps):
            p = pred_roll[n, ..., t]
            y = true_roll[n, ..., t]
            p_hat = np.fft.rfft2(np.moveaxis(p, -1, 0), axes=(-2, -1))
            y_hat = np.fft.rfft2(np.moveaxis(y, -1, 0), axes=(-2, -1))
            p_energy = np.abs(p_hat) ** 2
            y_energy = np.abs(y_hat) ** 2
            val = np.abs(p_energy - y_energy).sum() / (y_energy.sum() + eps)
            vals.append(val)
        p = pred_roll[n, ..., -1]; y = true_roll[n, ..., -1]
        p_hat = np.fft.rfft2(np.moveaxis(p, -1, 0), axes=(-2, -1)); y_hat = np.fft.rfft2(np.moveaxis(y, -1, 0), axes=(-2, -1))
        final_vals.append(np.abs(np.abs(p_hat) ** 2 - np.abs(y_hat) ** 2).sum() / ((np.abs(y_hat) ** 2).sum() + eps))
    return {
        "Spectrum Error": float(np.mean(vals)),
        "Final Spectrum Error": float(np.mean(final_vals)),
    }


def tv_overshoot_shock_metrics_1d(pred, truth, eps=1e-12):
    pred_roll, true_roll = _roll(pred, truth)
    tv_err = []
    shock_err = []
    overshoot = []
    for n in range(pred_roll.shape[0]):
        for t in range(pred_roll.shape[-1]):
            p = pred_roll[n, ..., t]
            y = true_roll[n, ..., t]
            tvp = np.abs(np.roll(p, -1, axis=0) - p).sum(axis=0).mean()
            tvy = np.abs(np.roll(y, -1, axis=0) - y).sum(axis=0).mean()
            tv_err.append(abs(tvp - tvy))
            pg = np.abs(np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)).reshape(p.shape[0], -1).mean(axis=1)
            yg = np.abs(np.roll(y, -1, axis=0) - np.roll(y, 1, axis=0)).reshape(y.shape[0], -1).mean(axis=1)
            raw = abs(int(np.argmax(pg)) - int(np.argmax(yg)))
            shock_err.append(min(raw, p.shape[0] - raw) / p.shape[0])
            overshoot.append(max(0.0, float(p.max() - y.max())) + max(0.0, float(y.min() - p.min())))
    return {"TV Error": float(np.mean(tv_err)), "Shock Loc Error": float(np.mean(shock_err)), "Overshoot": float(np.mean(overshoot))}


def combined_metrics(pred, truth, include_1d_local=False):
    out = {}
    out.update(mean_drift_metrics(pred, truth))
    out.update(band_error_metrics(pred, truth))
    out.update(spectrum_error_metrics(pred, truth))
    if include_1d_local:
        out.update(tv_overshoot_shock_metrics_1d(pred, truth))
    return out
