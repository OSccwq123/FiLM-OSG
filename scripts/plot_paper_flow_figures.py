#!/usr/bin/env python3
"""Create flow/field-oriented manuscript figures from completed checkpoints.

Unlike ``plot_paper_figures.py``, this script prioritizes representative solution
profiles and local zooms. Quantitative error summaries remain in tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat, savemat
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_burgers_fno import evaluate_one_model  # noqa: E402


DEFAULT_MODELS = [
    ("OSG-FNO", "fno", "burgers_sharp_seed0_e1000"),
    ("FiLM-OSG-FNO + proj.", "fno_film", "burgers_sharp_film_proj_seed0_e1000"),
    ("VT-FNO", "vt_fno", "vt_external_seed0_burgers_sharp"),
]
COLORS = {
    "Truth": "#111111",
    "OSG-FNO": "#4C78A8",
    "FiLM-OSG-FNO + proj.": "#F58518",
    "VT-FNO": "#E45756",
}


def load_or_eval_predictions(args, label: str, model: str, tag: str, train_data, test_data):
    out_dir = args.pred_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{model}_seed{args.seed}_{tag}_full_predictions.mat"
    if not pred_path.exists() or args.force_eval:
        evaluate_one_model(
            model,
            args.seed,
            tag,
            args.model_root,
            test_data,
            train_data,
            args.device,
            eval_steps=args.eval_steps,
            save_mat=True,
            save_dir=str(out_dir),
        )
    data = loadmat(pred_path)
    return {
        "label": label,
        "model": model,
        "tag": tag,
        "prediction": data["prediction"].astype(np.float64),
    }


def periodic_gradient(u):
    return np.abs(np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / 2.0


def select_case(truth, preds):
    final = truth[:, :, 0, -1]
    grad = periodic_gradient(final).max(axis=1)
    amp = final.max(axis=1) - final.min(axis=1)
    sharp_score = grad / (amp + 1e-12)
    by_label = {p["label"]: p["prediction"][:, :, 0, -1] for p in preds}
    film = by_label.get("FiLM-OSG-FNO + proj.")
    gl = by_label.get("GL-FiLM-OSG-FNO + proj.")
    if film is not None and gl is not None:
        film_err = np.mean(np.abs(film - final), axis=1)
        gl_err = np.mean(np.abs(gl - final), axis=1)
        improvement = film_err - gl_err
    else:
        improvement = np.zeros_like(sharp_score)
    score = sharp_score / (np.median(sharp_score) + 1e-12) + improvement / (np.std(improvement) + 1e-12)
    strong = np.where(amp > np.quantile(amp, 0.5))[0]
    if len(strong):
        return int(strong[np.argmax(score[strong])])
    return int(np.argmax(score))


def zoom_window(x, u, width=0.22):
    g = np.abs(np.roll(u, -1) - np.roll(u, 1))
    center = x[int(np.argmax(g))]
    left = center - width / 2
    right = center + width / 2
    return left % 1.0, right % 1.0, center


def plot_periodic_segment(ax, x, y, left, right, **kwargs):
    if left < right:
        mask = (x >= left) & (x <= right)
        ax.plot(x[mask], y[mask], **kwargs)
    else:
        x_ext = np.concatenate([x[x >= left] - 1.0, x[x <= right]])
        y_ext = np.concatenate([y[x >= left], y[x <= right]])
        ax.plot(x_ext, y_ext, **kwargs)
        ax.set_xlim(left - 1.0, right)
        return
    ax.set_xlim(left, right)


def style(ax):
    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def burgers_flow_figure(args):
    train_data = loadmat(args.data_dir / "BurgersSharpOSG_train.mat")
    test_data = loadmat(args.data_dir / "BurgersSharpOSG_test.mat")
    preds = [load_or_eval_predictions(args, *spec, train_data, test_data) for spec in DEFAULT_MODELS]
    truth = test_data["trajectories"].astype(np.float64)
    dt = test_data["dt"].astype(np.float64)
    x = test_data.get("coordinates", train_data["coordinates"]).reshape(-1).astype(np.float64)
    case = args.case if args.case is not None else select_case(truth, preds)
    times = np.concatenate([[0.0], np.cumsum(dt[case])])
    final_truth = truth[case, :, 0, -1]
    left, right, center = zoom_window(x, final_truth, width=args.zoom_width)

    fig = plt.figure(figsize=(12.2, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1.05, 1.0], hspace=0.35, wspace=0.28)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_full = fig.add_subplot(gs[0, 1])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[1, 1])

    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, truth.shape[-1]))
    for k in range(truth.shape[-1]):
        lw = 2.0 if k in (0, truth.shape[-1] - 1) else 0.8
        alpha = 0.95 if k in (0, truth.shape[-1] - 1) else 0.45
        label = "initial" if k == 0 else ("final truth" if k == truth.shape[-1] - 1 else None)
        ax_traj.plot(x, truth[case, :, 0, k], color=cmap[k], lw=lw, alpha=alpha, label=label)
    ax_traj.set_title(f"Reference sharp-front evolution, sample {case}, T={times[-1]:.3f}")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("u")
    ax_traj.legend(frameon=False, fontsize=8)
    style(ax_traj)

    ax_full.plot(x, final_truth, color=COLORS["Truth"], lw=2.3, label="Truth")
    for p in preds:
        y = p["prediction"][case, :, 0, -1]
        ax_full.plot(x, y, lw=1.7, color=COLORS.get(p["label"], "#777777"), label=p["label"])
    ax_full.axvline(center, color="#555555", lw=0.8, ls=":")
    ax_full.set_title("Final-time prediction overlay")
    ax_full.set_xlabel("x")
    ax_full.set_ylabel("u")
    ax_full.legend(frameon=False, fontsize=7.5, ncol=1)
    style(ax_full)

    plot_periodic_segment(ax_zoom, x, final_truth, left, right, color=COLORS["Truth"], lw=2.4, label="Truth")
    for p in preds:
        y = p["prediction"][case, :, 0, -1]
        plot_periodic_segment(ax_zoom, x, y, left, right, lw=1.8, color=COLORS.get(p["label"], "#777777"), label=p["label"])
    ax_zoom.set_title("Local zoom near steep front")
    ax_zoom.set_xlabel("x")
    ax_zoom.set_ylabel("u")
    style(ax_zoom)

    for p in preds:
        y = p["prediction"][case, :, 0, -1]
        err = np.abs(y - final_truth)
        plot_periodic_segment(ax_err, x, err, left, right, lw=1.9, color=COLORS.get(p["label"], "#777777"), label=p["label"])
    ax_err.set_yscale("log")
    ax_err.set_title("Absolute error in the same zoom window")
    ax_err.set_xlabel("x")
    ax_err.set_ylabel("|prediction - truth|")
    style(ax_err)

    fig.suptitle("Sharp-front Burgers: variable-time baselines and FiLM-OSG front fidelity", fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.out_dir / "fig_burgers_sharp_flow_case.pdf"
    png = args.out_dir / "fig_burgers_sharp_flow_case.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=190)
    plt.close(fig)

    meta = {
        "case": np.array([[case]], dtype=np.int32),
        "zoom_left": np.array([[left]], dtype=np.float64),
        "zoom_right": np.array([[right]], dtype=np.float64),
        "zoom_center": np.array([[center]], dtype=np.float64),
    }
    savemat(args.out_dir / "fig_burgers_sharp_flow_case_meta.mat", meta)
    print("Generated", pdf)
    print("Generated", png)
    print("Selected sample", case, "front center", center)



from eval.eval_convdiff_fno import evaluate_one_model as evaluate_ad_model  # noqa: E402
from eval.eval_ns_fno import evaluate_one_model as evaluate_ns_model  # noqa: E402

AD_MODELS = [
    ("Truth", None, None),
    ("VT-FNO", "vt_fno", "vt_external_seed0_ad"),
    ("VT-FiLM-FNO", "vt_fno_film", "vt_film_external_seed0_ad"),
    ("FiLM-OSG-FNO + proj.", "fno_film", "ad_seed0_nohup_film_loglag_proj"),
    ("GL-FiLM-OSG + proj.", "gl_fno_film", "ad_seed0_formal_branchwise_loglag_proj"),
]
NS_MODELS = [
    ("Truth", None, None),
    ("OSG-FNO + proj.", "fno", "ns_seed0_full_fno_proj"),
    ("FiLM-OSG-FNO + proj.", "fno_film", "ns_seed0_full_film_proj"),
    ("stable GL-FiLM + proj.", "gl_fno_film", "ns_seed0_stable_branchwise_gamma"),
]
FIELD_COLORS = {
    "VT-FNO": "#E45756",
    "VT-FiLM-FNO": "#D62728",
    "FiLM-OSG-FNO + proj.": "#F58518",
    "GL-FiLM-OSG + proj.": "#B279A2",
    "OSG-FNO + proj.": "#4C78A8",
    "stable GL-FiLM + proj.": "#B279A2",
}


def load_or_eval_2d(args, label, model, tag, train_data, test_data, evaluator, prefix):
    if model is None:
        return None
    out_dir = args.pred_dir / prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{model}_seed{args.seed}_{tag}_full_predictions.mat"
    if not pred_path.exists() or args.force_eval:
        evaluator(
            model,
            args.seed,
            tag,
            args.model_root,
            test_data,
            train_data,
            args.device,
            eval_steps=args.eval_steps,
            save_mat=True,
            save_dir=str(out_dir),
        )
    return loadmat(pred_path)["prediction"].astype(np.float64)


def select_2d_case(truth, pred_candidates):
    final_truth = truth[..., -1]
    best_pred = None
    for pred in pred_candidates:
        if pred is not None and np.isfinite(pred).all():
            best_pred = pred[..., -1]
            break
    if best_pred is None:
        energy = np.mean(final_truth[..., 0] ** 2, axis=(1, 2))
        return int(np.argmax(energy))
    err = np.mean(np.abs(best_pred - final_truth), axis=(1, 2, 3))
    energy = np.mean(final_truth[..., 0] ** 2, axis=(1, 2))
    score = err / (np.median(err) + 1e-12) + 0.15 * energy / (np.median(energy) + 1e-12)
    return int(np.argmax(score))


def plot_2d_field_grid(truth, preds, labels, case, out_pdf, out_png, title, cmap="RdBu_r"):
    final_truth = truth[case, :, :, 0, -1]
    fields = [final_truth]
    errors = [None]
    for pred in preds:
        if pred is None or not np.isfinite(pred).all():
            fields.append(np.full_like(final_truth, np.nan))
            errors.append(np.full_like(final_truth, np.nan))
        else:
            final_pred = pred[case, :, :, 0, -1]
            fields.append(final_pred)
            errors.append(np.abs(final_pred - final_truth))

    vmin = np.nanmin(final_truth)
    vmax = np.nanmax(final_truth)
    lim = max(abs(vmin), abs(vmax))
    err_vals = [e for e in errors[1:] if e is not None and np.isfinite(e).any()]
    err_max = max(float(np.nanpercentile(e, 99)) for e in err_vals) if err_vals else 1.0

    fig, axes = plt.subplots(2, len(labels), figsize=(3.0 * len(labels), 5.9), constrained_layout=True)
    for j, (label, field) in enumerate(zip(labels, fields)):
        ax = axes[0, j]
        im = ax.imshow(field.T, origin="lower", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if j == len(labels) - 1:
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    axes[1, 0].axis("off")
    for j, err in enumerate(errors[1:], start=1):
        ax = axes[1, j]
        im_err = ax.imshow(err.T, origin="lower", cmap="magma", vmin=0, vmax=err_max, interpolation="nearest")
        ax.set_title("absolute error", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if j == len(labels) - 1:
            fig.colorbar(im_err, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(title + f" (sample {case}, final step)", fontsize=13)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=190)
    plt.close(fig)
    print("Generated", out_pdf)


def ad_field_figure(args):
    train_data = loadmat(args.base_data_dir / "train_data.mat")
    test_data = loadmat(args.base_data_dir / "test_data.mat")
    labels = [m[0] for m in AD_MODELS]
    preds = [None]
    for label, model, tag in AD_MODELS[1:]:
        preds.append(load_or_eval_2d(args, label, model, tag, train_data, test_data, evaluate_ad_model, "ad"))
    truth = test_data["trajectories"].astype(np.float64)
    case = args.ad_case if args.ad_case is not None else select_2d_case(truth, preds[1:])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_2d_field_grid(
        truth,
        preds[1:],
        labels,
        case,
        args.out_dir / "fig_ad_field_vt_osg_compare.pdf",
        args.out_dir / "fig_ad_field_vt_osg_compare.png",
        "Advection-diffusion: variable-time baseline and FiLM-OSG comparison",
        cmap="viridis",
    )


def ns_field_figure(args):
    train_data = loadmat(args.base_data_dir / "VorticityOSG_train.mat")
    test_data = loadmat(args.base_data_dir / "VorticityOSG_test.mat")
    labels = [m[0] for m in NS_MODELS]
    preds = [None]
    for label, model, tag in NS_MODELS[1:]:
        preds.append(load_or_eval_2d(args, label, model, tag, train_data, test_data, evaluate_ns_model, "ns"))
    truth = test_data["trajectories"].astype(np.float64)
    case = args.ns_case if args.ns_case is not None else select_2d_case(truth, preds[1:])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_2d_field_grid(
        truth,
        preds[1:],
        labels,
        case,
        args.out_dir / "fig_ns_vorticity_projection_gl_ablation.pdf",
        args.out_dir / "fig_ns_vorticity_projection_gl_ablation.png",
        "Navier-Stokes: projection models and GL ablation",
        cmap="jet",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate flow-oriented paper figures from completed checkpoints.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/burgers_sharp"))
    parser.add_argument("--base-data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures_flow"))
    parser.add_argument("--pred-dir", type=Path, default=Path("paper_figures_flow/predictions"))
    parser.add_argument("--model-root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=int, default=None)
    parser.add_argument("--ad-case", type=int, default=None)
    parser.add_argument("--ns-case", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--zoom-width", type=float, default=0.22)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--only", choices=["all", "burgers", "ad", "ns"], default="all")
    args = parser.parse_args()
    if args.only in ("all", "burgers"):
        burgers_flow_figure(args)
    if args.only in ("all", "ad"):
        ad_field_figure(args)
    if args.only in ("all", "ns"):
        ns_field_figure(args)


if __name__ == "__main__":
    main()
