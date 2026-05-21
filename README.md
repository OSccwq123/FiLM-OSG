# FiLM-OSG Reproducibility Code

This repository contains the reproducibility code for the FiLM-OSG manuscript:
training entrypoints, evaluation scripts, overhead profiling, data-generation
helpers, and lightweight launchers. The code is organized so a clean
environment can run without the external `due` package; local compatibility
aliases are provided for older DUE-style pickle paths.

Large `.mat` datasets, model weights, predictions, and result tables are not
tracked by normal git. On machines without the full data or CUDA resources, use
`--help`, `--dry-run`, and `--check-only` modes first.

## Layout

- `train/`: single-job training entrypoints.
- `eval/`: evaluation entrypoints that write JSON/CSV/MAT outputs when data and
  weights are available.
- `profiling/`: overhead profiling entrypoints.
- `scripts/`: multi-GPU launchers and orchestration helpers.
- `scripts/archive/`: historical, temporary, or one-off helper scripts retained
  for provenance.
- `docs/`: environment notes, dependency audit, and attribution.

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

This matches the CUDA 11.7 PyTorch setup used for smoke tests. If your
CUDA/driver stack requires another PyTorch build, install the appropriate
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
advection--diffusion `.mat` files. The Navier--Stokes `.mat` files are derived
from DUE benchmark data; keep the repository `NOTICE` and
`docs/third_party_attribution.md` with any redistributed code or data package.

See `data/README.md` for sizes, checksums, generation notes, and redistribution
notes. The `.mat` files are ignored by normal git because several exceed
GitHub's regular 100 MB file limit; use Git LFS or GitHub Release assets only
after confirming data redistribution permission.

## License Status

The final repository license is pending confirmation by the authors and advisor.
Until a top-level `LICENSE` file is added, treat this repository as shared for
review and reproducibility preparation rather than as generally licensed
software. Code adapted from AI4Equations/DUE should retain the upstream
LGPL-2.1 attribution noted in `NOTICE` and `docs/third_party_attribution.md`;
the final release should confirm any additional LGPL notice or source
availability requirements before publication. `THIRD_PARTY_LICENSES/` contains
placeholders for this release review without selecting the final top-level
license.

To regenerate scripted data in MATLAB:

```matlab
cd data
Burgers
convection_diffusion
```

The MATLAB scripts are minimal headless generators: they write `.mat` files and
run shape checks, but do not create diagnostic figures.

## Manuscript Protocol Snapshot

The active entrypoints follow the current manuscript tables. This README records
the execution defaults needed to reproduce the experiments; the manuscript
remains the source of truth for interpretation and reported conclusions.

| Benchmark | Grid | Train/Test snapshots | Lag protocol | Lag preprocessing |
| --- | --- | --- | --- | --- |
| Burgers | `64` | `800 x 11` / `100 x 21` | train/test variable `Delta in [0.005, 0.15]` | `log10(Delta)` then affine |
| Advection--diffusion | `64 x 64` | `100 x 51` / `100 x 100` | train/test variable `Delta in [0.005, 0.5]` | affine |
| Navier--Stokes | `64 x 64` | `100 x 50` / `100 x 100` | train variable `Delta in [0.5, 1.5]`; test fixed `Delta = 1` | affine |

| Benchmark | Backbone config | Loss | Epochs / batch | SG pairing / weight | Reported runs |
| --- | --- | --- | --- | --- | --- |
| Burgers | modes `10`, depth `3`, width `60` | MAE | `1000 / 100` | `2 / 5.0` | seeds `0,1,2` |
| Advection--diffusion | modes `12 x 12`, depth `4`, width `20` | MAE | `500 / 100` | `1 / 1.0` | seeds `0,1,2` |
| Navier--Stokes | modes `12 x 12`, depth `4`, width `20` | Rel-L2 | `500 / 20` | `1 / 1.0` | seeds `0,1,2,3,4` |

Additional Navier--Stokes diagnostics use the same outer-increment OSG
formulation:

- Partition robustness: OSG-FNO and FiLM-OSG-FNO, seeds `0,1,2,3,4`.
- Non-FNO portability: OSG-adapted U-NO-style and Transolver-style variants,
  seeds `0,1,2`.
- Overhead profile: FNO, U-NO-style, and Transolver-style variants on a
  representative NS batch with batch size `20`.
- Semigroup-loss diagnostic: direct-lag and FiLM-conditioned FNO variants, with
  and without the auxiliary SG loss, seed `0`.
- MambaNO-style checks are supplementary and can be selected explicitly.

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

To inspect GPU ids before using a launcher:

```bash
python scripts/launch_burgers_fno.py --list-gpus
python scripts/launch_convdiff_fno.py --list-gpus
python scripts/launch_ns_extra_backbones.py --list-gpus
```

Launchers print the PyTorch-visible GPU mapping with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. They accept any CUDA GPU by default and only
filter by model name if `--require-gpu-name` is passed, for example
`--require-gpu-name A100`.

## Reproducibility Notes

- The manuscript defaults above are encoded as CLI defaults where applicable,
  but seeds are always passed explicitly in launcher commands.
- The data-generation scripts use fixed MATLAB RNG seeds and endpoint-excluded
  periodic grids where FFTs are used.
- Evaluation scripts write seedwise and summary CSV/JSON files. They do not
  fabricate missing metrics; use `--skip-missing` only for path checks or
  partial smoke tests.
- Historical scripts in `scripts/archive/` are retained for provenance and are
  not recommended as primary reproduction entrypoints.

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
