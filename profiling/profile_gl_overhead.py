import argparse, csv, os, random, sys, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rel_l2_loss(pred, target, eps=1e-12):
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return torch.mean(torch.linalg.norm(pred_flat-target_flat, dim=1)/(torch.linalg.norm(target_flat, dim=1)+eps))


def data_loss(pred, target, kind):
    if kind == 'mae':
        return F.l1_loss(pred, target)
    if kind == 'rel_l2':
        return rel_l2_loss(pred, target)
    raise ValueError(kind)


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
    x0 = xb[..., :-1]
    dt1_norm = xb[..., -1:]
    perm = torch.randperm(xb.shape[0], device=xb.device)
    dt2_norm = xb[perm, ..., -1:]
    dt1 = decode_dt(dt1_norm, tmin, tmax, multiscale)
    dt2 = decode_dt(dt2_norm, tmin, tmax, multiscale)
    dt12_norm = encode_dt(dt1 + dt2, tmin, tmax, multiscale)
    return torch.cat((x0, dt1_norm), dim=-1), dt2_norm, torch.cat((x0, dt12_norm), dim=-1)


def training_step(model, xb, yb, opt, tmin, tmax, multiscale, loss_kind, sg_weight):
    opt.zero_grad(set_to_none=True)
    pred = model(xb)
    ld = data_loss(pred, yb, loss_kind)
    x_step1, dt2_norm, x_direct = make_sg_batch(xb, tmin, tmax, multiscale)
    pred_step1 = model(x_step1)
    pred_two = model(torch.cat((pred_step1, dt2_norm), dim=-1))
    pred_direct = model(x_direct)
    lsg = data_loss(pred_two, pred_direct, loss_kind)
    loss = (ld + sg_weight*lsg)/(1.0+sg_weight)
    loss.backward(); opt.step()
    return float(loss.detach().cpu())


def sync(device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_config(case, model, batch_size, seed):
    common = dict(seed=seed, device='cuda' if torch.cuda.is_available() else 'cpu', batch_size=batch_size,
                  learning_rate=1e-3, optimizer='adam', scheduler='cosine', verbose=10, activation='gelu')
    if case == 'burgers_sharp':
        common.update(problem_type='1d_regular', problem_dim=1, multiscale=True, dtype='float32', loss='mae',
                      nbursts=10, sg_pairing=2, sg_weight=5.0, modes=10, depth=3, width=60,
                      local_kernel_size=5, local_pool_factor=2, gl_layer_scale=1e-3, gl_film_mode='branchwise',
                      conserve_mean=False, save_path='./profile_tmp_burgers')
    elif case == 'ad':
        common.update(problem_type='2d_regular', problem_dim=1, multiscale=False, dtype='float32', loss='mae',
                      nbursts=25, sg_pairing=1, sg_weight=1.0, modes1=12, modes2=12, depth=4, width=20,
                      local_kernel_size=3, local_pool_factor=2, gl_layer_scale=1e-3, gl_film_mode='branchwise',
                      conserve_mean=False, save_path='./profile_tmp_ad')
    elif case == 'ns':
        common.update(problem_type='2d_regular', problem_dim=1, multiscale=False, dtype='float32', loss='rel_l2',
                      nbursts=25, sg_pairing=1, sg_weight=1.0, modes1=12, modes2=12, depth=4, width=20,
                      local_kernel_size=3, local_pool_factor=2, gl_layer_scale=1e-3, gl_film_mode='branchwise',
                      conserve_mean=False, save_path='./profile_tmp_ns')
    else:
        raise ValueError(case)
    return common


def build(case, model, vmin, vmax, tmin, tmax, config):
    from film_osg.networks.fno import osg_fno1d_with_film, gl_osg_fno1d_with_film, osg_fno2d_with_film, gl_osg_fno2d_with_film
    if case == 'burgers_sharp':
        return gl_osg_fno1d_with_film(vmin, vmax, tmin, tmax, config, config['multiscale']) if model == 'gl_film' else osg_fno1d_with_film(vmin, vmax, tmin, tmax, config, config['multiscale'])
    return gl_osg_fno2d_with_film(vmin, vmax, tmin, tmax, config, config['multiscale']) if model == 'gl_film' else osg_fno2d_with_film(vmin, vmax, tmin, tmax, config, config['multiscale'])


def paths(case):
    if case == 'burgers_sharp':
        return 'data/burgers_sharp/BurgersSharpOSG_train.mat', 'data/burgers_sharp/BurgersSharpOSG_test.mat'
    if case == 'ad':
        return 'data/train_data.mat', 'data/test_data.mat'
    if case == 'ns':
        return 'data/VorticityOSG_train.mat', 'data/VorticityOSG_test.mat'
    raise ValueError(case)


def profile(case, model_name, batch_size, seed, warmup, iters, device):
    from film_osg.datasets.pde import pde_dataset_osg
    set_seed(seed)
    cfg = make_config(case, model_name, batch_size, seed)
    tr, te = paths(case)
    ds = pde_dataset_osg(cfg)
    trainX, trainY, coords, data_test, dt_test, vmin, vmax, tmin, tmax, cmin, cmax = ds.load(tr, te)
    model = build(case, model_name, vmin, vmax, tmin, tmax, cfg).to(device)
    params = count_params(model)
    xb = torch.from_numpy(trainX[:batch_size]).float().to(device)
    yb = torch.from_numpy(trainY[:batch_size]).float().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    for _ in range(warmup):
        training_step(model, xb, yb, opt, tmin, tmax, cfg['multiscale'], cfg['loss'], cfg['sg_weight'])
    sync(device)
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        training_step(model, xb, yb, opt, tmin, tmax, cfg['multiscale'], cfg['loss'], cfg['sg_weight'])
    sync(device)
    train_ms = (time.perf_counter()-t0)*1000/iters
    peak = torch.cuda.max_memory_allocated()/(1024**3) if torch.cuda.is_available() and str(device).startswith('cuda') else float('nan')
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(xb)
        sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(xb)
        sync(device)
    infer_ms = (time.perf_counter()-t0)*1000/iters
    del model, opt, xb, yb
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return dict(case=case, model=model_name, params=params, params_M=params/1e6,
                train_step_ms=train_ms, inference_step_ms=infer_ms, peak_train_memory_GB=peak,
                batch_size=batch_size, warmup=warmup, iters=iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='burgers_sharp,ad,ns')
    ap.add_argument('--models', default='film,gl_film')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save-dir', default='overhead_outputs_gl')
    args = ap.parse_args()
    global np, torch, F
    import numpy as np
    import torch
    import torch.nn.functional as F
    os.makedirs(args.save_dir, exist_ok=True)
    rows=[]
    for case in [c for c in args.cases.split(',') if c]:
        bs = 20 if case == 'ns' else 100
        for model in [m for m in args.models.split(',') if m]:
            print(f'Profiling {case} {model} bs={bs}', flush=True)
            row = profile(case, model, bs, args.seed, args.warmup, args.iters, args.device)
            print(row, flush=True)
            rows.append(row)
    out = Path(args.save_dir)/'gl_overhead_profile.csv'
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print('wrote', out, flush=True)

if __name__ == '__main__':
    main()
