# FiLM-OSG Reproducibility Code

This repository organizes the reproducibility scripts for the FiLM-OSG
manuscript. The local machine may not have CUDA, large `.mat` datasets, model
weights, predictions, or result tables, so local checks should prefer `--help`,
`--dry-run`, and `--check-only` modes.

## Layout

- `train/`: single-job training entrypoints.
- `eval/`: evaluation entrypoints that write JSON/CSV/MAT outputs when data and
  weights are available.
- `profiling/`: overhead profiling entrypoints.
- `scripts/`: multi-GPU launchers and orchestration helpers.
- `scripts/archive/`: historical, temporary, or one-off helper scripts retained
  for provenance.
- `configs/`, `docs/`: placeholders for future configuration and notes.

The root-level `fno.py`, `pde.py`, `pde_osg.py`, `utils.py`,
`osg_extra_backbones.py`, and `__init__.py` are retained as compatibility/source
snapshots. Active scripts use the local `film_osg` package and do not require
the external `due` package. Historical DUE module paths are handled through
local pickle aliases when loading older model files. See
`docs/due_dependency_minimization.md` for the dependency audit and migration
plan. See `docs/third_party_attribution.md` for attribution of local modules
adapted from AI4Equations/DUE and related FNO reference code; the repository
level `NOTICE` file records the same provenance in a standard notice format.

## Environment

The active code path does not require the external `due` package. A clean conda
environment can be created with:

```bash
conda create -n film_osg_clean python=3.11 -y
conda activate film_osg_clean
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

This matches the smoke-tested CUDA 11.7 PyTorch setup used on the server. If
your CUDA/driver stack requires another PyTorch build, install the appropriate
PyTorch wheel first, then run `python -m pip install -r requirements.txt`.

Optional `.mat` readers for MATLAB v7.3/HDF5 files can be installed with:

```bash
python -m pip install h5py mat73
```

See `docs/environment.md` for tested versions and clean-environment checks.

## Data

By default, training, evaluation, and profiling scripts read benchmark files
from `data/`. Use `--data-dir /path/to/data` to point scripts at another data
directory.

Expected files:

```text
data/BurgersOSG_train.mat
data/BurgersOSG_test.mat
data/train_data.mat
data/test_data.mat
data/VorticityOSG_train.mat
data/VorticityOSG_test.mat
```

`data/Burgers.m` and `data/convection_diffusion.m` generate the Burgers and
advection--diffusion `.mat` files. The Navier--Stokes `.mat` files are DUE
benchmark data used by our group.

See `data/README.md` for sizes, checksums, generation notes, and redistribution
notes. The `.mat` files are ignored by normal git because several exceed
GitHub's regular 100 MB file limit; use Git LFS or GitHub Release assets only
after confirming data redistribution permission.

To regenerate scripted data in MATLAB:

```matlab
cd data
Burgers
convection_diffusion
```

For headless/server runs, set `make_plots = false` near the top of each MATLAB
script before running.

## Manuscript Defaults

Main matched FNO comparisons:

- Burgers: OSG-FNO vs FiLM-OSG-FNO, seeds `0,1,2`, epochs `1000`, batch size
  `100`, SG pairing/weight `2/5.0`, multiscale log-lag preprocessing.
- Advection--diffusion: OSG-FNO vs FiLM-OSG-FNO, seeds `0,1,2`, epochs `500`,
  batch size `100`, SG pairing/weight `1/1.0`, affine lag preprocessing.
- Navier--Stokes: OSG-FNO vs FiLM-OSG-FNO, seeds `0,1,2,3,4`, epochs `500`,
  batch size `20`, SG pairing/weight `1/1.0`, affine lag preprocessing.

Additional Navier--Stokes diagnostics:

- Partition robustness: OSG-FNO vs FiLM-OSG-FNO, seeds `0,1,2,3,4`.
- Non-FNO portability: U-NO-style and Transolver-style variants, seeds `0,1,2`.
- Overhead profile: representative NS batch size `20`.
- Semigroup-loss diagnostic: direct-lag / FiLM-conditioned with and without SG,
  seed `0`.
- MambaNO-style variants are supplementary and can be selected explicitly.

## Local Checks

These commands should not require CUDA, large `.mat` files, model weights, or
the external `due` package:

```bash
python train/run_burgers_fno.py --help
python train/run_convdiff_fno.py --help
python train/train_ns_one.py --help
python train/train_ns_extra_backbones.py --help
python eval/eval_burgers_fno.py --help
python eval/eval_convdiff_fno.py --help
python eval/eval_ns_extra_backbones.py --help
python eval/eval_ns_fno.py --help
python eval/eval_ns_ablation.py --help
python profiling/profile_ns_overhead.py --help
```

Dry-run and check-only examples:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --dry-run
python train/run_convdiff_fno.py --model fno_film --seed 0 --dry-run
python scripts/launch_burgers_fno.py --dry-run --seeds 0,1,2 --gpus 0,1 --models fno,fno_film
python eval/eval_burgers_fno.py --check-only --skip-missing
python profiling/profile_ns_overhead.py --check-only --models fno,fno_film
```

Do not run full training or full evaluation locally unless the required data,
weights, and CUDA resources are available.

## Representative Commands

Main FNO training is launched through the scripts in `scripts/`:

```bash
python scripts/launch_burgers_fno.py --seeds 0,1,2 --gpus 0,1 --models fno,fno_film
python scripts/launch_convdiff_fno.py --seeds 0,1,2 --gpus 0,1 --models fno,fno_film
```

Single NS FNO jobs use:

```bash
python train/train_ns_one.py --model fno --seed 0
python train/train_ns_one.py --model fno_film --seed 0
```

Non-FNO portability jobs use U-NO-style and Transolver-style variants by default:

```bash
python scripts/launch_ns_extra_backbones.py --seeds 0,1,2 --gpus 0,1 --models uno,uno_film,transolver,transolver_film
```

Evaluation and overhead profiling should be run only when the corresponding
data and weights are present:

```bash
python eval/eval_burgers_fno.py --seeds 0,1,2 --models fno,fno_film
python eval/eval_convdiff_fno.py --seeds 0,1,2 --models fno,fno_film
python eval/eval_ns_fno.py --seeds 0,1,2,3,4 --models fno,fno_film
python eval/eval_ns_extra_backbones.py --seeds 0,1,2 --models uno,uno_film,transolver,transolver_film
python eval/eval_ns_ablation.py --seeds 0
python profiling/profile_ns_overhead.py --models fno,fno_film,uno,uno_film,transolver,transolver_film
```
