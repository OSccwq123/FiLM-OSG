# FiLM-OSG

Reproducibility code for the FiLM-OSG manuscript. The repository includes
training entrypoints, evaluation scripts, overhead profiling, lightweight
multi-GPU launchers, and minimal data-generation scripts.

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

If your CUDA stack needs a different PyTorch build, install the matching PyTorch
wheel first, then install `requirements.txt`. See `docs/environment.md` for
tested versions and optional dependencies.

## Data

Place benchmark `.mat` files under `data/`:

```text
data/BurgersOSG_train.mat
data/BurgersOSG_test.mat
data/train_data.mat
data/test_data.mat
data/VorticityOSG_train.mat
data/VorticityOSG_test.mat
```

Scripts accept `--data-dir /path/to/data` for alternate locations. Burgers and
advection-diffusion data can be regenerated with:

```matlab
cd data
Burgers
convection_diffusion
```

The `.mat` files are not tracked by normal git. See `data/README.md` for file
formats, checksums, generation notes, and redistribution cautions.

## Quick Checks

These checks should not require CUDA, large datasets, model weights, or the
external `due` package:

```bash
python train/run_burgers_fno.py --help
python train/run_convdiff_fno.py --help
python train/train_ns_one.py --help
python train/train_ns_extra_backbones.py --help
python eval/eval_burgers_fno.py --help
python eval/eval_convdiff_fno.py --help
python eval/eval_ns_fno.py --help
python eval/eval_ns_extra_backbones.py --help
python eval/eval_ns_ablation.py --help
python profiling/profile_ns_overhead.py --help
```

Useful dry-run examples:

```bash
python train/run_convdiff_fno.py --model fno_film --seed 0 --dry-run
python scripts/launch_convdiff_fno.py --list-gpus
python eval/eval_burgers_fno.py --check-only --skip-missing
python profiling/profile_ns_overhead.py --check-only --models fno,fno_film
```

## Reproducing Experiments

The manuscript is the source of truth for experimental details. The entrypoints
below use manuscript-aligned defaults for model settings, losses, and training
lengths; pass seeds and GPU ids explicitly.

Main FNO training:

```bash
python scripts/launch_burgers_fno.py --seeds 0,1,2 --gpus 0,1 --models fno,fno_film
python scripts/launch_convdiff_fno.py --seeds 0,1,2 --gpus 0,1 --models fno,fno_film
python train/train_ns_one.py --model fno --seed 0
python train/train_ns_one.py --model fno_film --seed 0
```

NS extra-backbone diagnostics:

```bash
python scripts/launch_ns_extra_backbones.py --seeds 0,1,2 --gpus 0,1 --models uno,uno_film,transolver,transolver_film
```

Evaluation and profiling:

```bash
python eval/eval_burgers_fno.py --seeds 0,1,2 --models fno,fno_film
python eval/eval_convdiff_fno.py --seeds 0,1,2 --models fno,fno_film
python eval/eval_ns_fno.py --seeds 0,1,2,3,4 --models fno,fno_film
python eval/eval_ns_extra_backbones.py --seeds 0,1,2 --models uno,uno_film,transolver,transolver_film
python eval/eval_ns_ablation.py --seeds 0
python profiling/profile_ns_overhead.py --models fno,fno_film,uno,uno_film,transolver,transolver_film
```

Launchers print the PyTorch-visible GPU mapping with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. They accept any CUDA GPU by default; use
`--require-gpu-name` only if you intentionally want to filter by model name.

## Repository Layout

```text
film_osg/       local package used by active scripts
train/          single-job training entrypoints
eval/           evaluation scripts writing CSV/JSON/MAT outputs
profiling/      overhead profiling entrypoints
scripts/        launchers and orchestration helpers
data/           data-generation scripts and ignored .mat files
docs/           environment, dependency, and attribution notes
```

Historical or one-off scripts are kept under `scripts/archive/` for provenance.

## License and Attribution

The final top-level repository license is pending confirmation by the authors
and advisor. Until a `LICENSE` file is added, treat this repository as shared
for review and reproducibility preparation rather than as generally licensed
software.

Portions of `film_osg` are adapted from
[AI4Equations/DUE](https://github.com/AI4Equations/due), which is marked
upstream as LGPL-2.1 licensed. See `NOTICE`, `docs/third_party_attribution.md`,
and `THIRD_PARTY_LICENSES/` for provenance and release-review placeholders.
