# FiLM-OSG

![reproducibility](https://img.shields.io/badge/reproducibility-draft-blue)
![python](https://img.shields.io/badge/python-3.11-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.0.1-orange)

Reproducibility code for the FiLM-OSG manuscript. This repository contains model
implementations, data-generation utilities, training entrypoints, evaluation
diagnostics, profiling scripts, and lightweight launcher helpers used to prepare
the reported experiments.

Reference branch: `codex/minimize-due-deps`
Reference commit: `add5ea7`
Release tag: pending final license and data-release decisions.

The active code path uses the local `film_osg` package and does not require the
external `due` package. Compatibility aliases are kept for older DUE-style model
files.

## Setup

```bash
conda create -n film_osg_clean python=3.11 -y
conda activate film_osg_clean
python -m pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

Install a different PyTorch wheel if required by your CUDA driver. See
`docs/environment.md` for the smoke-tested package versions.

## Data

Place benchmark `.mat` files under `data/` or pass `--data-dir`.

```text
data/BurgersOSG_train.mat
data/BurgersOSG_test.mat
data/train_data.mat
data/test_data.mat
data/VorticityOSG_train.mat
data/VorticityOSG_test.mat
```

Burgers and advection--diffusion data can be regenerated locally:

```matlab
cd data
Burgers
convection_diffusion
```

Navier--Stokes uses the public DUE benchmark data format. The repository expects
the two `VorticityOSG_*.mat` files above but does not redistribute large `.mat`
files through normal git. See `data/README.md` for shapes, checksums, and
data-release notes.

The sharp-front inviscid Burgers extension used for global-local ablations is
generated separately:

```bash
python scripts/generate_burgers_sharp_osg.py --out-dir data/burgers_sharp
```

This generator uses a fine-grid finite-volume/Rusanov solver and conservative
averaging to the learning grid. The generated files are written as
`BurgersSharpOSG_train.mat` and `BurgersSharpOSG_test.mat`; either pass
`--data-dir data/burgers_sharp` to the Burgers scripts or provide local symlinks
named `BurgersOSG_train.mat` and `BurgersOSG_test.mat`.

## Implemented Models and Options

The main FNO entrypoints expose the following model families:

```text
fno             OSG-FNO outer-increment baseline
fno_film        FiLM-OSG-FNO with lag-conditioned channel modulation
gl_fno          global-local OSG-FNO
gl_fno_film     global-local FiLM-OSG-FNO
vt_fno          direct variable-time FNO baseline without OSG structure
vt_fno_film     direct variable-time FiLM-FNO baseline without OSG structure
```

Important training options:

```text
--conserve-mean        project the learned increment to mean zero
--hf-weight            high-frequency data loss weight
--hf-sg-weight         high-frequency semigroup-consistency loss weight
--hf-warmup-frac       fraction of epochs used to warm up high-frequency losses
--gl-film-mode         global_only or branchwise FiLM modulation for GL models
--log-lag              use log-lag conditioning for advection--diffusion
```

The VT baselines intentionally disable semigroup regularization, projection, and
high-frequency losses inside the training scripts. They are included as external
variable-time baselines, not as OSG variants.

## Quick Checks

These commands should not require CUDA, data files, or model weights:

```bash
python train/run_burgers_fno.py --help
python train/run_convdiff_fno.py --help
python train/train_ns_one.py --help
python eval/eval_burgers_fno.py --help
python eval/eval_convdiff_fno.py --help
python eval/eval_ns_fno.py --help
python eval/eval_ns_partition_spread.py --help
python profiling/profile_ns_overhead.py --help
```

Dry-run/check-only examples:

```bash
python train/run_burgers_fno.py --model gl_fno_film --seed 0 --dry-run --conserve-mean --gl-film-mode branchwise
python train/run_convdiff_fno.py --model vt_fno_film --seed 0 --dry-run
python train/train_ns_one.py --model gl_fno_film --seed 0 --dry-run --conserve-mean
python eval/eval_burgers_fno.py --check-only --skip-missing
```

## Representative Experiment Commands

Full multi-seed reproduction of every manuscript table is computationally
expensive and should be scheduled on a cluster. The commands below are
representative single-job entrypoints showing the manuscript-facing settings;
choose seeds, GPUs, and job-array scheduling explicitly for your system.

Burgers sharp-front:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --tag burgers_sharp_seed0_e1000 --data-dir data/burgers_sharp --epochs 1000
python train/run_burgers_fno.py --model fno_film --seed 0 --tag burgers_sharp_film_proj_seed0_e1000 --data-dir data/burgers_sharp --epochs 1000 --conserve-mean
python train/run_burgers_fno.py --model gl_fno_film --seed 0 --tag burgers_sharp_branchwise_proj_seed0_e1000 --data-dir data/burgers_sharp --epochs 1000 --conserve-mean --gl-film-mode branchwise
python train/run_burgers_fno.py --model vt_fno --seed 0 --tag vt_external_seed0_burgers_sharp --data-dir data/burgers_sharp --epochs 1000
python train/run_burgers_fno.py --model vt_fno_film --seed 0 --tag vt_film_external_seed0_burgers_sharp --data-dir data/burgers_sharp --epochs 1000
```

Advection--diffusion:

```bash
python train/run_convdiff_fno.py --model fno --seed 0 --tag ad_seed0_fno_proj --data-dir data --epochs 500 --conserve-mean
python train/run_convdiff_fno.py --model fno_film --seed 0 --tag ad_seed0_film_loglag_proj --data-dir data --epochs 500 --log-lag --conserve-mean
python train/run_convdiff_fno.py --model gl_fno_film --seed 0 --tag ad_seed0_branchwise_loglag_proj --data-dir data --epochs 500 --log-lag --conserve-mean --gl-film-mode branchwise
python train/run_convdiff_fno.py --model vt_fno --seed 0 --tag vt_external_seed0_ad --data-dir data --epochs 500
python train/run_convdiff_fno.py --model vt_fno_film --seed 0 --tag vt_film_external_seed0_ad --data-dir data --epochs 500
```

Navier--Stokes:

```bash
python train/train_ns_one.py --model fno --seed 0 --tag ns_seed0_fno_proj --data-dir data --epochs 500 --conserve-mean
python train/train_ns_one.py --model fno_film --seed 0 --tag ns_seed0_film_proj --data-dir data --epochs 500 --conserve-mean
python train/train_ns_one.py --model gl_fno_film --seed 0 --tag ns_seed0_gl_film_ablation --data-dir data --epochs 500 --conserve-mean
```

Evaluation examples:

```bash
python eval/eval_burgers_fno.py --seeds 0,1,2,3,4 --models fno,fno_film,gl_fno,gl_fno_film --data-dir data/burgers_sharp
python eval/eval_convdiff_fno.py --seeds 0,1,2,3,4 --models fno,fno_film,gl_fno,gl_fno_film,vt_fno,vt_fno_film
python eval/eval_ns_fno.py --seeds 0,1,2,3,4 --models fno,fno_film,gl_fno,gl_fno_film
python eval/eval_ns_partition_spread.py --models fno,fno_film,gl_fno_film --seeds 0 --partitions 1,2,4,8
```

Launchers print the PyTorch-visible GPU mapping with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. Use `--require-gpu-name` only when you
intentionally want to filter by model name.

## Optional Diagnostics

The advection--diffusion lag-extrapolation diagnostic is not part of the main
benchmark table. It generates fixed-lag test sets outside the training lag
interval `[0.005, 0.5]` and evaluates already trained AD models:

```matlab
cd data
convection_diffusion_fixed_lag_extrapolation
```

```bash
python eval/eval_convdiff_lag_extrapolation.py --models fno,fno_film --seeds 0,1,2 --tag ad_affine_seed012 --data-dir data
```

## Layout

```text
film_osg/       local package used by active scripts
train/          single-job training entrypoints
eval/           evaluation scripts and shared diagnostics
profiling/      overhead profiling entrypoints
scripts/        data utilities, launchers, and small inspection helpers
data/           data-generation scripts and ignored .mat files
docs/           environment and attribution notes
```

Historical or one-off scripts are kept under `scripts/archive/` for provenance.
Generated datasets, model checkpoints, logs, evaluation outputs, plotting
artifacts, and one-off queue files are intentionally ignored by git.

## License and Attribution

The final top-level license is pending author/advisor confirmation. Until a
`LICENSE` file is added, treat this repository as shared for review and
reproducibility preparation rather than as generally licensed software.

Some local implementation files are adapted from
[AI4Equations/DUE](https://github.com/AI4Equations/due). See `NOTICE`,
`docs/third_party_attribution.md`, and `THIRD_PARTY_LICENSES/` for attribution
and release-review placeholders.
