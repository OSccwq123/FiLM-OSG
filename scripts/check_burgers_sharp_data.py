import argparse
from pathlib import Path
import json

import numpy as np
from scipy.io import loadmat


def parse_args():
    parser = argparse.ArgumentParser(description='Check sharp-front Burgers .mat files.')
    parser.add_argument('--data-dir', type=Path, default=Path('data/burgers_sharp'), help='Directory with BurgersSharpOSG_train/test.mat files.')
    parser.add_argument('--base-data-dir', type=Path, default=Path('data'), help='Optional directory with baseline BurgersOSG_train/test.mat files for comparison.')
    parser.add_argument('--out', type=Path, default=None, help='Output JSON summary path. Defaults to <data-dir>/sanity_summary.json.')
    return parser.parse_args()


def tv(u):
    return np.abs(np.roll(u, -1, axis=1) - u).sum(axis=1)


def max_grad(u):
    return np.abs(np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)).max(axis=1) / 2.0


def stats(name, data):
    tr = data['trajectories'].astype(np.float64)
    dt = data['dt'].astype(np.float64)
    coords = data['coordinates']
    init = tr[..., 0].squeeze(-1) if tr.shape[-2] == 1 else tr[..., 0]
    final = tr[..., -1].squeeze(-1) if tr.shape[-2] == 1 else tr[..., -1]
    means = tr[:, :, 0, :].mean(axis=1)
    drift = np.abs(means - means[:, :1])
    init_tv = tv(init)
    final_tv = tv(final)
    init_grad = max_grad(init)
    final_grad = max_grad(final)
    out = {
        'shape_trajectories': list(tr.shape),
        'shape_dt': list(dt.shape),
        'shape_coordinates': list(coords.shape),
        'finite': bool(np.isfinite(tr).all() and np.isfinite(dt).all() and np.isfinite(coords).all()),
        'value_min': float(tr.min()),
        'value_max': float(tr.max()),
        'value_mean': float(tr.mean()),
        'value_std': float(tr.std()),
        'dt_min': float(dt.min()),
        'dt_max': float(dt.max()),
        'dt_mean': float(dt.mean()),
        'mean_drift_max': float(drift[:, 1:].max()),
        'mean_drift_avg': float(drift[:, 1:].mean()),
        'init_tv_mean': float(init_tv.mean()),
        'init_tv_median': float(np.median(init_tv)),
        'init_tv_q90': float(np.quantile(init_tv, 0.9)),
        'final_tv_mean': float(final_tv.mean()),
        'init_max_grad_mean': float(init_grad.mean()),
        'init_max_grad_median': float(np.median(init_grad)),
        'init_max_grad_q90': float(np.quantile(init_grad, 0.9)),
        'final_max_grad_mean': float(final_grad.mean()),
        'first_state_range_examples': [[float(init[i].min()), float(init[i].max())] for i in range(min(5, init.shape[0]))],
    }
    coeff = np.fft.rfft(init, axis=1)
    nfreq = coeff.shape[1]
    hi = slice(int(2 * nfreq / 3), None)
    out['init_hf_energy_frac_mean'] = float((np.abs(coeff[:, hi])**2).sum(axis=1).mean() / ((np.abs(coeff)**2).sum(axis=1).mean() + 1e-12))
    print('\n' + name)
    for k, v in out.items():
        print(f'  {k}: {v}')
    return out


def main():
    args = parse_args()
    root = args.data_dir
    files = {
        'train': root / 'BurgersSharpOSG_train.mat',
        'test': root / 'BurgersSharpOSG_test.mat',
    }
    base_files = {
        'train': args.base_data_dir / 'BurgersOSG_train.mat',
        'test': args.base_data_dir / 'BurgersOSG_test.mat',
    }

    summary = {}
    for split, path in files.items():
        if not path.exists():
            print('MISSING', path)
            continue
        summary['sharp_' + split] = stats('sharp_' + split, loadmat(path))
    for split, path in base_files.items():
        if path.exists():
            summary['base_' + split] = stats('base_' + split, loadmat(path))

    if 'sharp_train' in summary and 'base_train' in summary:
        s, b = summary['sharp_train'], summary['base_train']
        summary['comparison'] = {
            'init_tv_mean_ratio_sharp_over_base': s['init_tv_mean'] / (b['init_tv_mean'] + 1e-12),
            'init_max_grad_mean_ratio_sharp_over_base': s['init_max_grad_mean'] / (b['init_max_grad_mean'] + 1e-12),
            'init_hf_energy_frac_ratio_sharp_over_base': s['init_hf_energy_frac_mean'] / (b['init_hf_energy_frac_mean'] + 1e-12),
        }
        print('\ncomparison')
        for k, v in summary['comparison'].items():
            print(f'  {k}: {v}')

    out = args.out or (root / 'sanity_summary.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nSAVED', out)


if __name__ == '__main__':
    main()
