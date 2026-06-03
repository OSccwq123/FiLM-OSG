import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(description='Plot sharp-front Burgers trajectory examples.')
    parser.add_argument('--data-dir', type=Path, default=Path('data/burgers_sharp'), help='Directory with BurgersSharpOSG_train/test.mat files.')
    parser.add_argument('--out', type=Path, default=None, help='Output PDF path. Defaults to <data-dir>/burgers_sharp_evolution_examples.pdf.')
    return parser.parse_args()


def pick_samples(tr, n=6):
    u0 = tr[:, :, 0, 0]
    amp = u0.max(axis=1) - u0.min(axis=1)
    grad = np.abs(np.roll(u0, -1, axis=1) - np.roll(u0, 1, axis=1)).max(axis=1) / 2
    score = grad / (amp + 1e-12)
    strong = np.where(amp > 0.8)[0]
    if len(strong) >= n:
        idx = strong[np.argsort(score[strong])[-n:]][::-1]
    else:
        idx = np.argsort(score)[-n:][::-1]
    return idx, amp, grad, score


def plot_page(pdf, data, split):
    tr = data['trajectories'].astype(float)
    dt = data['dt'].astype(float)
    x = data['coordinates'].reshape(-1)
    idx, amp, grad, score = pick_samples(tr, n=6)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, tr.shape[-1]))
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
    fig.suptitle(f'Burgers sharp-front {split}: selected high normalized-gradient samples', fontsize=14)
    for ax, i in zip(axes.ravel(), idx):
        times = np.concatenate([[0.0], np.cumsum(dt[i])])
        for k in range(tr.shape[-1]):
            lw = 2.2 if k in (0, tr.shape[-1] - 1) else 0.9
            alpha = 1.0 if k in (0, tr.shape[-1] - 1) else 0.55
            label = 't=0' if k == 0 else ('final' if k == tr.shape[-1] - 1 else None)
            ax.plot(x, tr[i, :, 0, k], color=colors[k], lw=lw, alpha=alpha, label=label)
        ax.set_title(f'{split} sample {int(i)} | amp={amp[i]:.2f}, maxgrad={grad[i]:.2f}, T={times[-1]:.2f}', fontsize=9)
        ax.set_ylabel('u')
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, 1)
    axes[-1, 0].set_xlabel('x')
    axes[-1, 1].set_xlabel('x')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f'Burgers sharp-front {split}: time-space heatmaps', fontsize=14)
    for ax, i in zip(axes, idx[:3]):
        arr = tr[i, :, 0, :].T
        times = np.concatenate([[0.0], np.cumsum(dt[i])])
        im = ax.imshow(arr, aspect='auto', origin='lower', extent=[0, 1, times[0], times[-1]], cmap='RdBu_r')
        ax.set_title(f'{split} sample {int(i)} | amp={amp[i]:.2f}, normalized grad={score[i]:.2f}', fontsize=9)
        ax.set_ylabel('time')
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    axes[-1].set_xlabel('x')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def main():
    args = parse_args()
    root = args.data_dir
    out = args.out or (root / 'burgers_sharp_evolution_examples.pdf')
    train_path = root / 'BurgersSharpOSG_train.mat'
    test_path = root / 'BurgersSharpOSG_test.mat'
    missing = [str(p) for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required sharp-front data files: ' + ', '.join(missing))
    out.parent.mkdir(parents=True, exist_ok=True)
    train = loadmat(train_path)
    test = loadmat(test_path)
    with PdfPages(out) as pdf:
        plot_page(pdf, train, 'train')
        plot_page(pdf, test, 'test')
    print(out)


if __name__ == '__main__':
    main()
