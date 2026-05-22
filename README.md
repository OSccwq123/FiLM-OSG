# FiLM-OSG

![reproducibility](https://img.shields.io/badge/reproducibility-draft-blue)
![python](https://img.shields.io/badge/python-3.11-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.0.1-orange)

Reproducibility code for the FiLM-OSG manuscript. This repository contains
training, evaluation, profiling, and lightweight launcher entrypoints for the
reported experiments.

Reference branch: `codex/minimize-due-deps`  
Reference commit: `add5ea7`  
Release tag: pending final license and data-release decisions.

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

Navier--Stokes uses the public DUE benchmark data format. The repository
expects the two `VorticityOSG_*.mat` files above but does not redistribute
large `.mat` files through normal git. See `data/README.md` for shapes,
checksums, and data-release notes.

## Quick Checks

These commands should not require CUDA, data files, or model weights:

```bash
python train/run_burgers_fno.py --help
python train/run_convdiff_fno.py --help
python train/train_ns_one.py --help
python eval/eval_ns_fno.py --help
python profiling/profile_ns_overhead.py --help
```

Dry-run/check-only examples:

```bash
python train/run_convdiff_fno.py --model fno_film --seed 0 --dry-run
python scripts/launch_convdiff_fno.py --list-gpus
python eval/eval_burgers_fno.py --check-only --skip-missing
```

## Main Runs

Use the manuscript for the authoritative protocol. Pass seeds and GPU ids
explicitly.

```bash
python scripts/launch_burgers_fno.py --seeds 0,1,2 --gpus 0,1 --models fno,fno_film
python scripts/launch_convdiff_fno.py --seeds 0,1,2 --gpus 0,1 --models fno,fno_film
python eval/eval_burgers_fno.py --seeds 0,1,2 --models fno,fno_film
python eval/eval_convdiff_fno.py --seeds 0,1,2 --models fno,fno_film
python eval/eval_ns_fno.py --seeds 0,1,2,3,4 --models fno,fno_film
```

Navier--Stokes auxiliary diagnostics:

```bash
python scripts/launch_ns_extra_backbones.py --seeds 0,1,2 --gpus 0,1 --models uno,uno_film,transolver,transolver_film
python eval/eval_ns_extra_backbones.py --seeds 0,1,2 --models uno,uno_film,transolver,transolver_film
python eval/eval_ns_ablation.py --seeds 0
python profiling/profile_ns_overhead.py --models fno,fno_film,uno,uno_film,transolver,transolver_film
```

Launchers print the PyTorch-visible GPU mapping with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. Use `--require-gpu-name` only when you
intentionally want to filter by model name.

## Layout

```text
film_osg/       local package used by active scripts
train/          single-job training entrypoints
eval/           evaluation scripts writing CSV/JSON/MAT outputs
profiling/      overhead profiling entrypoints
scripts/        launchers and orchestration helpers
data/           data-generation scripts and ignored .mat files
docs/           environment and attribution notes
```

Plotting scripts and generated figures are intentionally ignored by git in this
draft repository.

## License and Attribution

The final top-level license is pending author/advisor confirmation. Until a
`LICENSE` file is added, treat this repository as shared for review and
reproducibility preparation rather than as generally licensed software.

Some local implementation files are adapted from
[AI4Equations/DUE](https://github.com/AI4Equations/due). See `NOTICE`,
`docs/third_party_attribution.md`, and `THIRD_PARTY_LICENSES/` for attribution
and release-review placeholders.
