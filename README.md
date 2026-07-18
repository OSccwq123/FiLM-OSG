# FiLM-OSG

![reproducibility](https://img.shields.io/badge/reproducibility-draft-blue)
![python](https://img.shields.io/badge/python-3.11-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.0.1-orange)

Reproducibility code for the FiLM-OSG manuscript. This repository contains model
implementations, data-generation utilities, training entrypoints, evaluation
diagnostics, profiling scripts, and lightweight launcher helpers used to prepare
the reported experiments.

Reference branch: `codex/minimize-due-deps`
Reference commit: use the commit pinned in the manuscript's Data/code availability statement.
Release tag: pending final license and data-release decisions.

The active code path uses the local `film_osg` package and does not require the
external `due` package. Compatibility aliases are kept for older DUE-style model
files.

The manuscript tables use seed-indexed runs, with five-seed summaries reported
for the main Burgers, advection--diffusion, and selected sharp-front/VT controls
unless a table is explicitly labeled as a seed-0 diagnostic. Full multi-seed
reproduction is intended for cluster scheduling rather than a single bundled
launcher. The profiling scripts report model parameters, training-step time,
inference-step time, and peak CUDA memory for the hardware used by the user.

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

Burgers and advection--diffusion data can be regenerated locally from the
repository root. The MATLAB scripts write their outputs into `data/`:

```matlab
cd data
run('../scripts/data_generation/Burgers.m')
run('../scripts/data_generation/convection_diffusion.m')
```

Navier--Stokes uses the public DUE benchmark data format. The repository expects
the two `VorticityOSG_*.mat` files above but does not redistribute large `.mat`
files through normal git. See `data/README.md` for shapes, checksums, and
data-release notes.

The sharp-front inviscid Burgers extension used for targeted global-local
ablations is
generated separately. Data-generation sources live under `scripts/data_generation/`,
while `data/` is reserved for local `.mat` files and data notes:

```bash
python scripts/data_generation/generate_burgers_sharp_osg.py --out-dir data/burgers_sharp
python scripts/check_burgers_sharp_data.py --data-dir data/burgers_sharp
python scripts/plot_burgers_sharp_evolution.py --data-dir data/burgers_sharp
```

This generator uses a fine-grid finite-volume/Rusanov solver and conservative
averaging to the learning grid. The generated files are written as
`BurgersSharpOSG_train.mat` and `BurgersSharpOSG_test.mat`. The Burgers entrypoints
retain their historical compatibility names, so create local aliases before
passing `--data-dir data/burgers_sharp`:

```bash
ln -s BurgersSharpOSG_train.mat data/burgers_sharp/BurgersOSG_train.mat
ln -s BurgersSharpOSG_test.mat data/burgers_sharp/BurgersOSG_test.mat
```

Run these commands from the repository root. On systems without symbolic-link
support, copy the two files to the compatibility names instead. The check script
writes `sanity_summary.json`, and the plot script writes a PDF with selected
trajectory examples.

### PDEBench-derived variable-lag checks

The manuscript also uses the public PDEBench radial-dam-break and two-dimensional
diffusion--reaction time series; see the
[PDEBench paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html).
Download the HDF5 files from the
[PDEBench archive](https://doi.org/10.18419/darus-2986) and place them locally as

```text
data/pdebench_raw/swe/2D_rdb_NA_NA.h5
data/pdebench_raw/reacdiff2d/2D_diff-react_NA_NA.h5
```

The raw files are not redistributed by this repository. The commands below
reproduce the manuscript's trajectory-disjoint converted-data protocol. A
fixed split seed assigns 800 source trajectories to training and 200 to test;
the two trajectory-ID sets do not overlap. Pair sampling is approximately
balanced by trajectory and creates 5,000 training pairs and 1,000 test pairs
with temporal offsets from 1 to 20 stored steps. No `--component` argument is
used: SWE retains its native scalar state, whereas reaction--diffusion retains
both coupled state components.

SWE64:

```bash
python scripts/data_generation/convert_pdebench_to_osg.py \
  --input data/pdebench_raw/swe/2D_rdb_NA_NA.h5 \
  --output-dir data/pdebench_osg/swe64_disjoint_full \
  --prefix PDEBenchSWE64DisjointFullOSG \
  --problem-dim 2d --grouped --layout HWD \
  --train-trajectories 800 --test-trajectories 200 \
  --trajectory-split-seed 0 --train-pairs 5000 --test-pairs 1000 \
  --min-lag-steps 1 --max-lag-steps 20 --space-stride 2 --seed 0
python scripts/data_generation/check_osg_mat_data.py \
  --train data/pdebench_osg/swe64_disjoint_full/PDEBenchSWE64DisjointFullOSG_train.mat \
  --test data/pdebench_osg/swe64_disjoint_full/PDEBenchSWE64DisjointFullOSG_test.mat \
  --problem-dim 2d --expected-channels 1 --expected-time 2 --require-positive-dt
ln -sfn PDEBenchSWE64DisjointFullOSG_train.mat data/pdebench_osg/swe64_disjoint_full/train_data.mat
ln -sfn PDEBenchSWE64DisjointFullOSG_test.mat data/pdebench_osg/swe64_disjoint_full/test_data.mat
```

ReacDiff64:

```bash
python scripts/data_generation/convert_pdebench_to_osg.py \
  --input data/pdebench_raw/reacdiff2d/2D_diff-react_NA_NA.h5 \
  --output-dir data/pdebench_osg/reacdiff64_disjoint_full \
  --prefix PDEBenchReacDiff64DisjointFullOSG \
  --problem-dim 2d --grouped --layout HWD \
  --train-trajectories 800 --test-trajectories 200 \
  --trajectory-split-seed 0 --train-pairs 5000 --test-pairs 1000 \
  --min-lag-steps 1 --max-lag-steps 20 --space-stride 2 --seed 0
python scripts/data_generation/check_osg_mat_data.py \
  --train data/pdebench_osg/reacdiff64_disjoint_full/PDEBenchReacDiff64DisjointFullOSG_train.mat \
  --test data/pdebench_osg/reacdiff64_disjoint_full/PDEBenchReacDiff64DisjointFullOSG_test.mat \
  --problem-dim 2d --expected-channels 2 --expected-time 2 --require-positive-dt
ln -sfn PDEBenchReacDiff64DisjointFullOSG_train.mat data/pdebench_osg/reacdiff64_disjoint_full/train_data.mat
ln -sfn PDEBenchReacDiff64DisjointFullOSG_test.mat data/pdebench_osg/reacdiff64_disjoint_full/test_data.mat
```

ReacDiff128 uses the same trajectory IDs and pair seeds as ReacDiff64, but does
not spatially subsample the source grid:

```bash
python scripts/data_generation/convert_pdebench_to_osg.py \
  --input data/pdebench_raw/reacdiff2d/2D_diff-react_NA_NA.h5 \
  --output-dir data/pdebench_osg/reacdiff128_disjoint_full \
  --prefix PDEBenchReacDiff128DisjointFullOSG \
  --problem-dim 2d --grouped --layout HWD \
  --train-trajectories 800 --test-trajectories 200 \
  --trajectory-split-seed 0 --train-pairs 5000 --test-pairs 1000 \
  --min-lag-steps 1 --max-lag-steps 20 --space-stride 1 --seed 0
python scripts/data_generation/check_osg_mat_data.py \
  --train data/pdebench_osg/reacdiff128_disjoint_full/PDEBenchReacDiff128DisjointFullOSG_train.mat \
  --test data/pdebench_osg/reacdiff128_disjoint_full/PDEBenchReacDiff128DisjointFullOSG_test.mat \
  --problem-dim 2d --expected-channels 2 --expected-time 2 --require-positive-dt
ln -sfn PDEBenchReacDiff128DisjointFullOSG_train.mat data/pdebench_osg/reacdiff128_disjoint_full/train_data.mat
ln -sfn PDEBenchReacDiff128DisjointFullOSG_test.mat data/pdebench_osg/reacdiff128_disjoint_full/test_data.mat
```

On systems without symbolic-link support, copy each generated file to the
corresponding `train_data.mat` or `test_data.mat` compatibility name. The
converter writes a metadata JSON containing the selected trajectory IDs, split
seed, overlap check, channel count, pair seeds, and lag range. Each pair contains
one transition, so the resulting errors are pairwise variable-lag errors rather
than long-rollout metrics. This conversion is not the standard PDEBench
leaderboard protocol.

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
--learning-rate        initial Adam learning rate
--sg-weight            semigroup-loss weight for advection--diffusion/PDEBench
--modes1/--modes2      retained Fourier modes for the 2D backbone
--depth/--width        2D FNO depth and latent width
--problem-dim          optional assertion for the number of state channels
```

The matched VT baselines directly predict the queried next state and omit both
the OSG outer-increment parameterization and the auxiliary semigroup objective.
They share data pairs and optimization settings with the corresponding OSG runs
but are not OSG variants.

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
python eval/eval_ns_partition_robustness_paper.py --help
python profiling/profile_ns_overhead.py --help
python profiling/profile_gl_overhead.py --help
python scripts/smoke_new_features.py --help
```

Dry-run/check-only examples:

```bash
python train/run_burgers_fno.py --model gl_fno_film --seed 0 --dry-run --conserve-mean --gl-film-mode branchwise
python train/run_convdiff_fno.py --model vt_fno_film --seed 0 --dry-run
python train/train_ns_one.py --model gl_fno_film --seed 0 --dry-run --conserve-mean
python eval/eval_burgers_fno.py --check-only --skip-missing
python eval/eval_ns_partition_robustness_paper.py --check-only --skip-missing
```

## Representative Single-Seed Experiment Commands

Full multi-seed reproduction is computationally expensive and should be
scheduled explicitly on a cluster. The commands below form matched seed-0
training/evaluation examples. Repeat them with other seed values, and choose GPUs
and job-array scheduling for your system; no one-shot full-experiment launcher is
provided.

Original Burgers:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --tag original_burgers_core --data-dir data --epochs 1000
python train/run_burgers_fno.py --model fno_film --seed 0 --tag original_burgers_core --data-dir data --epochs 1000
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0 --tag original_burgers_core --data-dir data --save-dir eval_outputs_burgers_original_seed0
```

Burgers sharp-front:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --tag burgers_sharp_core --data-dir data/burgers_sharp --epochs 1000
python train/run_burgers_fno.py --model fno_film --seed 0 --tag burgers_sharp_core --data-dir data/burgers_sharp --epochs 1000
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0 --tag burgers_sharp_core --data-dir data/burgers_sharp --save-dir eval_outputs_burgers_sharp_seed0
```

The targeted projected GL check uses the same data aliases but a separate tag:

```bash
python train/run_burgers_fno.py --model gl_fno_film --seed 0 --tag burgers_sharp_gl_branchwise_proj --data-dir data/burgers_sharp --epochs 1000 --conserve-mean --gl-film-mode branchwise
python eval/eval_burgers_fno.py --models gl_fno_film --seeds 0 --tag burgers_sharp_gl_branchwise_proj --data-dir data/burgers_sharp --save-dir eval_outputs_burgers_sharp_gl_seed0
```

Advection--diffusion:

```bash
python train/run_convdiff_fno.py --model fno --seed 0 --tag ad_shared_projection --data-dir data --epochs 500 --conserve-mean
python train/run_convdiff_fno.py --model fno_film --seed 0 --tag ad_shared_projection --data-dir data --epochs 500 --conserve-mean
python eval/eval_convdiff_fno.py --models fno,fno_film --seeds 0 --tag ad_shared_projection --data-dir data --save-dir eval_outputs_ad_seed0
```

PDEBench-derived single-transition checks use `12 x 12` modes, depth 4, width
20, an Adam/cosine schedule, initial learning rate `1e-3`, and 500 epochs. The
two `64 x 64` checks use batch size 100; the `128 x 128` check uses batch size
20 for memory. OSG models use one semigroup pairing with unit weight; the
training entrypoint disables the semigroup objective for the matched direct-time
controls. Comparisons are matched within each resolution, not across resolutions.

SWE64, seed 0:

```bash
python train/run_convdiff_fno.py --model fno --seed 0 --tag pdebench_swe64_disjoint_full_seed0 --data-dir data/pdebench_osg/swe64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --sg-weight 1 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 1
python train/run_convdiff_fno.py --model fno_film --seed 0 --tag pdebench_swe64_disjoint_full_seed0 --data-dir data/pdebench_osg/swe64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --sg-weight 1 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 1
python train/run_convdiff_fno.py --model vt_fno --seed 0 --tag pdebench_swe64_disjoint_full_seed0 --data-dir data/pdebench_osg/swe64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 1
python train/run_convdiff_fno.py --model vt_fno_film --seed 0 --tag pdebench_swe64_disjoint_full_seed0 --data-dir data/pdebench_osg/swe64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 1
python eval/eval_pdebench_disjoint_fullstate.py --models fno,fno_film,vt_fno,vt_fno_film --seeds 0 --tag pdebench_swe64_disjoint_full_seed0 --data-dir data/pdebench_osg/swe64_disjoint_full --save-dir eval_outputs_pdebench_swe64_disjoint_full_seed0
```

ReacDiff64, seed 0:

```bash
python train/run_convdiff_fno.py --model fno --seed 0 --tag pdebench_reacdiff64_disjoint_full_seed0 --data-dir data/pdebench_osg/reacdiff64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --sg-weight 1 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 2
python train/run_convdiff_fno.py --model fno_film --seed 0 --tag pdebench_reacdiff64_disjoint_full_seed0 --data-dir data/pdebench_osg/reacdiff64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --sg-weight 1 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 2
python train/run_convdiff_fno.py --model vt_fno --seed 0 --tag pdebench_reacdiff64_disjoint_full_seed0 --data-dir data/pdebench_osg/reacdiff64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 2
python train/run_convdiff_fno.py --model vt_fno_film --seed 0 --tag pdebench_reacdiff64_disjoint_full_seed0 --data-dir data/pdebench_osg/reacdiff64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 2
python eval/eval_pdebench_disjoint_fullstate.py --models fno,fno_film,vt_fno,vt_fno_film --seeds 0 --tag pdebench_reacdiff64_disjoint_full_seed0 --data-dir data/pdebench_osg/reacdiff64_disjoint_full --save-dir eval_outputs_pdebench_reacdiff64_disjoint_full_seed0
```

ReacDiff128, seed 0:

```bash
python train/run_convdiff_fno.py --model fno --seed 0 --tag pdebench_reacdiff128_disjoint_full_seed0 --data-dir data/pdebench_osg/reacdiff128_disjoint_full --epochs 500 --batch-size 20 --learning-rate 1e-3 --sg-weight 1 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 2
python train/run_convdiff_fno.py --model fno_film --seed 0 --tag pdebench_reacdiff128_disjoint_full_seed0 --data-dir data/pdebench_osg/reacdiff128_disjoint_full --epochs 500 --batch-size 20 --learning-rate 1e-3 --sg-weight 1 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 2
python eval/eval_pdebench_disjoint_fullstate.py --models fno,fno_film --seeds 0 --tag pdebench_reacdiff128_disjoint_full_seed0 --data-dir data/pdebench_osg/reacdiff128_disjoint_full --save-dir eval_outputs_pdebench_reacdiff128_disjoint_full_seed0
```

The formal evaluator reports aggregate and channel-wise relative errors,
frequency-band errors, and spectrum error. Since each converted sample contains
one transition, its mean and final relative errors coincide; these commands are
not long-rollout evaluations. Repeat the matched commands with seeds 1--4 to
construct the manuscript's population summaries.

Navier--Stokes:

```bash
# Main matched non-projection protocol
python train/train_ns_one.py --model fno --seed 0 --tag ns_core --data-dir data --epochs 500
python train/train_ns_one.py --model fno_film --seed 0 --tag ns_core --data-dir data --epochs 500
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0 --tag ns_core --data-dir data --save-dir eval_outputs_ns_seed0

# Separate mean-zero projection diagnostic
python train/train_ns_one.py --model fno --seed 0 --tag ns_projection_probe --data-dir data --epochs 500 --conserve-mean
python train/train_ns_one.py --model fno_film --seed 0 --tag ns_projection_probe --data-dir data --epochs 500 --conserve-mean
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0 --tag ns_projection_probe --data-dir data --save-dir eval_outputs_ns_projection_seed0

# Smaller-learning-rate direct-lag robustness check used in the NS baseline audit
python train/train_ns_one.py --model fno --seed 0 --tag ns_directlag_lr5e4 --data-dir data --epochs 500 --learning-rate 5e-4
python eval/eval_ns_fno.py --models fno --seeds 0 --tag ns_directlag_lr5e4 --data-dir data --save-dir eval_outputs_ns_directlag_lr5e4_seed0
```

When VT baseline runs use seed-indexed tags such as
`vt_external_seed0_burgers_sharp`, evaluate each seed with the matching tag and
then collect the completed seedwise CSV files with:

```bash
python scripts/summarize_vt_baselines.py --root eval_outputs_vt_baselines_5seed
```

Manuscript-facing PDF figures can be generated from completed evaluation summaries.
The summary script creates compact metric/drift panels; the flow script creates
field/profile figures intended for manuscript visual inspection:

```bash
python scripts/plot_method_diagram.py --out-dir paper_figures
python scripts/plot_paper_figures.py --out-dir paper_figures
python scripts/plot_paper_flow_figures.py --out-dir paper_figures --pred-dir paper_figures/predictions
```

Launchers print the PyTorch-visible GPU mapping with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. Use `--require-gpu-name` only when you
intentionally want to filter by model name.

## Optional Diagnostics

### Navier--Stokes partition robustness

Two partition diagnostics are provided and should not be interchanged. The
manuscript-matched evaluation uses terminal times `20,40,60,80`, the shared
uniform/fine/alternating partition set, and eight fixed random paired
partitions. Every sub-lag lies in `[0.5,1.5]`. Its reported spread is the mean
over all unordered partition pairs of
`||U_hat_pi(T)-U_hat_pj(T)||_2 / ||U(T)||_2`, with the same partition set used
for every model and training seed:

```bash
python eval/eval_ns_partition_robustness_paper.py --models fno,fno_film --seeds 0 --tag ns_core --model-root . --data-dir data --save-dir eval_outputs_ns_partition_seed0
```

The older `eval_ns_partition_spread.py` instead compares equal partitions with
`1,2,4,8` substeps and uses a symmetric prediction-pair denominator; it is a
separate full/high-frequency spread diagnostic:

```bash
python eval/eval_ns_partition_spread.py --models fno,fno_film --seeds 0 --tag ns_core --model-root . --data-dir data --save-dir eval_outputs_ns_equal_partition_seed0
```

### Post-training lag-sensitivity spectra

The layerwise lag-sensitivity diagnostic operates on existing checkpoints and
does not retrain a model. The examples below evaluate one matched seed with 200
held-out state--lag pairs. The first measurement is taken after the spectral
branch's internal pointwise MLP activation and before the complete FNO block's
terminal activation. The internal stage key `block0_preactivation` is retained
for compatibility with existing CSV files. Use a smaller `--samples` value for a
smoke test.

Original Burgers, seed 0:

```bash
python eval/eval_lag_sensitivity_spectrum.py --benchmark original_burgers --model-seed 0 --direct-model runs_burgers_fno_seed0_burgers_seed01234_final1000/model --film-model runs_burgers_fno_film_seed0_burgers_seed01234_final1000_rerun_seed0/model --data data/BurgersOSG_test.mat --samples 200 --fd-eps 0.01 --out-dir eval_outputs_lag_sensitivity/burgers_original_seed0_layerwise
```

Advection--diffusion under the shared-projection training protocol, seed 0:

```bash
python eval/eval_lag_sensitivity_spectrum.py --benchmark advection_diffusion --model-seed 0 --direct-model runs_convdiff_fno_seed0_ad_seed0_fullcover_fno_proj/model --film-model runs_convdiff_fno_film_seed0_ad_seed0_fullcover_film_proj/model --data data/test_data.mat --samples 200 --fd-eps 0.01 --out-dir eval_outputs_lag_sensitivity/ad_seed0_layerwise
```

Navier--Stokes under the main non-projection protocol, seed 0:

```bash
python eval/eval_lag_sensitivity_spectrum.py --benchmark navier_stokes --model-seed 0 --direct-model runs_ns_fno_seed0_ns_seed0_full_fno/model --film-model runs_ns_fno_film_seed0_ns_seed0_full_film/model --data data/VorticityOSG_test.mat --samples 200 --fd-eps 0.01 --out-dir eval_outputs_lag_sensitivity/ns_seed0_layerwise
```

After the corresponding seed directories are available, aggregate their means,
standard deviations, and seed ranges with `scripts/summarize_lag_sensitivity.py`.
For example:

```bash
python scripts/summarize_lag_sensitivity.py --run-prefix burgers_original_seed --retained-modes 10 --out-dir eval_outputs_lag_sensitivity/burgers_original_5seed_summary
python scripts/summarize_lag_sensitivity.py --run-prefix ad_seed --retained-modes 12 --radial-band --out-dir eval_outputs_lag_sensitivity/ad_5seed_summary
python scripts/summarize_lag_sensitivity.py --run-prefix ns_seed --retained-modes 12 --radial-band --out-dir eval_outputs_lag_sensitivity/ns_5seed_summary
python scripts/summarize_lag_sensitivity_scalar_metrics.py --root eval_outputs_lag_sensitivity --seeds 0,1,2,3,4 --out-dir eval_outputs_lag_sensitivity/scalar_summary
python scripts/plot_lag_sensitivity_mechanism_paper.py
```

The scalar summary reports nonzero-band energy and effective rank. Burgers uses
`k<10`; advection--diffusion and Navier--Stokes use the conservative low radial
band `|k|<12`, while effective rank is computed over the full diagnostic
spectrum.

The advection--diffusion lag-extrapolation diagnostic is not part of the main
benchmark table. It generates fixed-lag test sets outside the training lag
interval `[0.005, 0.5]` and evaluates already trained AD models:

```matlab
cd data
run('../scripts/data_generation/convection_diffusion_fixed_lag_extrapolation.m')
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
scripts/        data utilities, data-generation scripts, launchers, and small inspection helpers
data/           data notes and ignored local .mat files
docs/           environment and attribution notes
```

Generated datasets, model checkpoints, logs, evaluation outputs, plotting
artifacts, and one-off queue/provenance helpers are intentionally ignored by git.
Representative commands are documented above instead of bundling a one-shot
full-experiment launcher.

## License and Attribution

The final top-level license is pending author/advisor confirmation. Until a
`LICENSE` file is added, treat this repository as shared for review and
reproducibility preparation rather than as generally licensed software.

Some local implementation files are adapted from
[AI4Equations/DUE](https://github.com/AI4Equations/due). See `NOTICE`,
`docs/third_party_attribution.md`, and `THIRD_PARTY_LICENSES/` for attribution
and release-review placeholders.

## Release Checks

Overhead profile commands:

    python profiling/profile_ns_overhead.py --save-dir overhead_outputs_ns
    python profiling/profile_gl_overhead.py --save-dir overhead_outputs_gl

CPU-only smoke test for projection, high-frequency losses, and GL/FiLM paths:

    python scripts/smoke_new_features.py --device cpu
