from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
root=Path('data/burgers_sharp')
summary={}
for split in ['train','test']:
    d=loadmat(root/f'BurgersSharpOSG_{split}.mat')
    tr=d['trajectories'].astype(np.float64)[:, :, 0, :]
    dt=d['dt'].astype(np.float64)
    init=tr[:, :, 0]
    init_min=init.min(axis=1)[:,None]
    init_max=init.max(axis=1)[:,None]
    all_min=tr.min(axis=1)
    all_max=tr.max(axis=1)
    over=np.maximum(0.0, all_max-init_max)
    under=np.maximum(0.0, init_min-all_min)
    amp=(init_max-init_min).reshape(-1)
    means=tr.mean(axis=1)
    drift=np.abs(means-means[:, :1])
    # Also check one-step monotone envelope relative to previous frame, less strict but catches spikes.
    prev_min=tr[:, :, :-1].min(axis=1)
    prev_max=tr[:, :, :-1].max(axis=1)
    next_min=tr[:, :, 1:].min(axis=1)
    next_max=tr[:, :, 1:].max(axis=1)
    step_over=np.maximum(0.0, next_max-prev_max)
    step_under=np.maximum(0.0, prev_min-next_min)
    out={
      'shape': list(tr.shape), 'dt_shape': list(dt.shape), 'finite': bool(np.isfinite(tr).all() and np.isfinite(dt).all()),
      'dt_min': float(dt.min()), 'dt_max': float(dt.max()),
      'value_min': float(tr.min()), 'value_max': float(tr.max()),
      'mean_drift_max': float(drift[:,1:].max()), 'mean_drift_avg': float(drift[:,1:].mean()),
      'overshoot_vs_initial_max': float(over[:,1:].max()), 'undershoot_vs_initial_max': float(under[:,1:].max()),
      'overshoot_vs_initial_mean': float(over[:,1:].mean()), 'undershoot_vs_initial_mean': float(under[:,1:].mean()),
      'overshoot_over_amp_max': float((over[:,1:].max(axis=1)/(amp+1e-12)).max()),
      'undershoot_over_amp_max': float((under[:,1:].max(axis=1)/(amp+1e-12)).max()),
      'num_traj_any_overshoot_gt_1e-3': int((over[:,1:].max(axis=1)>1e-3).sum()),
      'num_traj_any_undershoot_gt_1e-3': int((under[:,1:].max(axis=1)>1e-3).sum()),
      'step_overshoot_max': float(step_over.max()), 'step_undershoot_max': float(step_under.max()),
      'amp_mean': float(amp.mean()), 'amp_lt_0p3_frac': float((amp<0.3).mean()), 'amp_gt_0p8_frac': float((amp>0.8).mean()),
    }
    summary[split]=out
    print('\n'+split)
    for k,v in out.items(): print(f'  {k}: {v}')
(root/'physics_check.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('\nSAVED', root/'physics_check.json')
