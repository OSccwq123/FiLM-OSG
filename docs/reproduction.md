# Reproducing the FiLM-OSG experiments

Run all commands from the repository root. The examples below use seed 0;
repeat matched experiments with seeds `0,1,2,3,4` for the five-seed summaries.
Tags identify checkpoints and must agree between training and evaluation.
Training commands do not replace an existing run directory unless
`--overwrite` is supplied.

## Burgers

Original benchmark:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --tag original_burgers_core --dataset original --data-dir data --epochs 1000
python train/run_burgers_fno.py --model fno_film --seed 0 --tag original_burgers_core --dataset original --data-dir data --epochs 1000
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0 --tag original_burgers_core --dataset original --data-dir data --save-dir eval_outputs_burgers_original_seed0
```

To run the five matched seeds across two GPUs and evaluate them together:

```bash
python scripts/launch_burgers_fno.py --gpus 0,1 --models fno,fno_film --seeds 0,1,2,3,4 --tag original_burgers_core --dataset original --data-dir data
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0,1,2,3,4 --tag original_burgers_core --dataset original --data-dir data --save-dir eval_outputs_burgers_original_5seed
```

The rollout figure for the final test trajectory is generated from a matched
checkpoint pair by

```bash
python scripts/plot_burgers_rollout.py --direct-model runs_burgers_fno_seed2_original_burgers_core/model --film-model runs_burgers_fno_film_seed2_original_burgers_core/model --data data/BurgersOSG_test.mat --sample -1 --steps 6,20 --out paper_figures/burgers_rollout.pdf
```

Additional data with steep gradients:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --tag burgers_sharp_core --dataset sharp --data-dir data/burgers_sharp --epochs 1000
python train/run_burgers_fno.py --model fno_film --seed 0 --tag burgers_sharp_core --dataset sharp --data-dir data/burgers_sharp --epochs 1000
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0 --tag burgers_sharp_core --dataset sharp --data-dir data/burgers_sharp --save-dir eval_outputs_burgers_sharp_seed0
```

Mean-zero projection:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --tag burgers_sharp_projection --dataset sharp --data-dir data/burgers_sharp --epochs 1000 --conserve-mean
python train/run_burgers_fno.py --model fno_film --seed 0 --tag burgers_sharp_projection --dataset sharp --data-dir data/burgers_sharp --epochs 1000 --conserve-mean
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0 --tag burgers_sharp_projection --dataset sharp --data-dir data/burgers_sharp --save-dir eval_outputs_burgers_sharp_projection_seed0
```

Global--local models with mean-zero projection:

```bash
python train/run_burgers_fno.py --model gl_fno --seed 0 --tag burgers_sharp_gl_direct_proj --dataset sharp --data-dir data/burgers_sharp --epochs 1000 --conserve-mean
python eval/eval_burgers_fno.py --models gl_fno --seeds 0 --tag burgers_sharp_gl_direct_proj --dataset sharp --data-dir data/burgers_sharp --save-dir eval_outputs_burgers_sharp_gl_direct_seed0

python train/run_burgers_fno.py --model gl_fno_film --seed 0 --tag burgers_sharp_gl_global_proj --dataset sharp --data-dir data/burgers_sharp --epochs 1000 --conserve-mean --gl-film-mode global_only
python eval/eval_burgers_fno.py --models gl_fno_film --seeds 0 --tag burgers_sharp_gl_global_proj --dataset sharp --data-dir data/burgers_sharp --save-dir eval_outputs_burgers_sharp_gl_global_seed0

python train/run_burgers_fno.py --model gl_fno_film --seed 0 --tag burgers_sharp_gl_branchwise_proj --dataset sharp --data-dir data/burgers_sharp --epochs 1000 --conserve-mean --gl-film-mode branchwise
python eval/eval_burgers_fno.py --models gl_fno_film --seeds 0 --tag burgers_sharp_gl_branchwise_proj --dataset sharp --data-dir data/burgers_sharp --save-dir eval_outputs_burgers_sharp_gl_seed0
```

## Advection--diffusion

The matched comparison uses mean-zero projection for both models:

```bash
python scripts/launch_convdiff_fno.py --gpus 0,1 --tag ad_shared_projection --data-dir data --conserve-mean
python eval/eval_convdiff_fno.py --models fno,fno_film --seeds 0,1,2,3,4 --tag ad_shared_projection --data-dir data --save-dir eval_outputs_ad
```

To save two seed-0 rollouts and generate the same field-panel layout:

```bash
python eval/eval_convdiff_fno.py --models fno,fno_film --seeds 0 --tag ad_shared_projection --data-dir data --save-dir eval_outputs_ad_seed0 --save-mat
python scripts/plot_convdiff_rollout.py \
  --direct eval_outputs_ad_seed0/fno_seed0_ad_shared_projection_full_predictions.mat \
  --film eval_outputs_ad_seed0/fno_film_seed0_ad_shared_projection_full_predictions.mat \
  --output figures/convdiff_rollout.pdf
```

Fixed-lag test sets outside the training interval `[0.005, 0.5]` are generated
and evaluated with:

```matlab
setenv('FILM_OSG_OUTPUT_DIR', fullfile(pwd, 'data'))
run('scripts/data_generation/convection_diffusion_fixed_lag_extrapolation.m')
```

```bash
python eval/eval_convdiff_lag_extrapolation.py --models fno,fno_film --seeds 0,1,2,3,4 --tag ad_shared_projection --data-dir data
```

## Navier--Stokes

Main comparison:

```bash
python scripts/launch_ns_fno.py --gpus 0,1 --tag ns_core --data-dir data
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0,1,2,3,4 --tag ns_core --data-dir data --save-dir eval_outputs_ns
```

Vorticity, relative-error, and enstrophy panels can be generated by

```bash
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0 --tag ns_core --data-dir data --save-dir eval_outputs_ns_seed0 --save-mat
python scripts/plot_ns_rollout.py \
  --direct eval_outputs_ns_seed0/fno_seed0_ns_core_full_predictions.mat \
  --film eval_outputs_ns_seed0/fno_film_seed0_ns_core_full_predictions.mat \
  --output-dir figures/ns
```

Use `--case` in either plotting script to select the trajectory shown in a
particular manuscript revision.

Mean-zero projection diagnostic:

```bash
python train/train_ns_one.py --model fno --seed 0 --tag ns_projection_probe --data-dir data --epochs 500 --conserve-mean
python train/train_ns_one.py --model fno_film --seed 0 --tag ns_projection_probe --data-dir data --epochs 500 --conserve-mean
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0 --tag ns_projection_probe --data-dir data --save-dir eval_outputs_ns_projection_seed0 --mean-drift
```

The manuscript partition-robustness diagnostic uses the same fixed collection
of admissible partitions for every model and seed:

```bash
python eval/eval_ns_partition_robustness_paper.py --models fno,fno_film --seeds 0,1,2,3,4 --tag ns_core --model-root . --data-dir data --save-dir eval_outputs_ns_partition
```

An additional equal-partition diagnostic uses `1,2,4,8` substeps:

```bash
python eval/eval_ns_partition_spread.py --models fno,fno_film --seeds 0 --tag ns_core --model-root . --data-dir data --save-dir eval_outputs_ns_equal_partition_seed0
```

## PDEBench-derived checks

Download the PDEBench radial-dam-break and two-dimensional diffusion--reaction
HDF5 files from the
[PDEBench archive](https://doi.org/10.18419/darus-2986). The converter uses an
800/200 trajectory-disjoint split and samples 5,000 training pairs and 1,000
test pairs with temporal offsets from 1 to 20 stored steps.

For SWE64:

```bash
python scripts/data_generation/convert_pdebench_to_osg.py --input data/pdebench_raw/swe/2D_rdb_NA_NA.h5 --output-dir data/pdebench_osg/swe64_disjoint_full --prefix PDEBenchSWE64DisjointFullOSG --train-trajectories 800 --test-trajectories 200 --train-pairs 5000 --test-pairs 1000 --min-lag-steps 1 --max-lag-steps 20 --space-stride 2 --split-seed 0 --pair-seed 0
python scripts/data_generation/check_osg_mat_data.py --train data/pdebench_osg/swe64_disjoint_full/PDEBenchSWE64DisjointFullOSG_train.mat --test data/pdebench_osg/swe64_disjoint_full/PDEBenchSWE64DisjointFullOSG_test.mat --problem-dim 2d --expected-channels 1 --expected-time 2 --require-positive-dt
ln -sfn PDEBenchSWE64DisjointFullOSG_train.mat data/pdebench_osg/swe64_disjoint_full/train_data.mat
ln -sfn PDEBenchSWE64DisjointFullOSG_test.mat data/pdebench_osg/swe64_disjoint_full/test_data.mat
```

For ReacDiff64, replace the input, output directory, and prefix by
`reacdiff2d/2D_diff-react_NA_NA.h5`, `reacdiff64_disjoint_full`, and
`PDEBenchReacDiff64DisjointFullOSG`, and use `--expected-channels 2`. For the
128-grid version, use `--space-stride 1`.

The converted datasets use the advection--diffusion training entry point. A
matched seed-0 SWE64 comparison is:

```bash
python train/run_convdiff_fno.py --model fno --seed 0 --tag pdebench_swe64_disjoint_full_seed0 --data-dir data/pdebench_osg/swe64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --sg-weight 1 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 1
python train/run_convdiff_fno.py --model fno_film --seed 0 --tag pdebench_swe64_disjoint_full_seed0 --data-dir data/pdebench_osg/swe64_disjoint_full --epochs 500 --batch-size 100 --learning-rate 1e-3 --sg-weight 1 --modes1 12 --modes2 12 --depth 4 --width 20 --problem-dim 1
python eval/eval_pdebench_disjoint_fullstate.py --models fno,fno_film --seeds 0 --tag pdebench_swe64_disjoint_full_seed0 --data-dir data/pdebench_osg/swe64_disjoint_full --save-dir eval_outputs_pdebench_swe64_disjoint_full_seed0
```

Each converted sample contains one transition, so these are variable-lag
single-transition tests rather than long-rollout evaluations.

## Other Navier--Stokes backbones

The U-NO-style and Transolver-style comparisons use seeds `0,1,2`:

```bash
python scripts/launch_ns_extra_backbones.py --gpus 0,1 --tag ns_extra --data-dir data
python eval/eval_ns_extra_backbones.py --models uno,uno_film,transolver,transolver_film --seeds 0,1,2 --tag ns_extra --data-dir data --save-dir eval_outputs_ns_extra_backbones
```

## Lag-sensitivity diagnostic

This diagnostic evaluates existing checkpoints; it does not retrain the
models. The CSV key `block0_preactivation` denotes the quantity measured before
the first block activation.

Original Burgers:

```bash
python eval/eval_lag_sensitivity_spectrum.py --benchmark original_burgers --model-seed 0 --direct-model runs_burgers_fno_seed0_original_burgers_core/model --film-model runs_burgers_fno_film_seed0_original_burgers_core/model --data data/BurgersOSG_test.mat --samples 200 --fd-eps 0.01 --out-dir eval_outputs_lag_sensitivity/burgers_original_seed0_layerwise
```

Advection--diffusion:

```bash
python eval/eval_lag_sensitivity_spectrum.py --benchmark advection_diffusion --model-seed 0 --direct-model runs_convdiff_fno_seed0_ad_shared_projection/model --film-model runs_convdiff_fno_film_seed0_ad_shared_projection/model --data data/test_data.mat --samples 200 --fd-eps 0.01 --out-dir eval_outputs_lag_sensitivity/ad_seed0_layerwise
```

Navier--Stokes:

```bash
python eval/eval_lag_sensitivity_spectrum.py --benchmark navier_stokes --model-seed 0 --direct-model runs_ns_fno_seed0_ns_core/model --film-model runs_ns_fno_film_seed0_ns_core/model --data data/VorticityOSG_test.mat --samples 200 --fd-eps 0.01 --out-dir eval_outputs_lag_sensitivity/ns_seed0_layerwise
```

Aggregate completed seed directories with:

```bash
python scripts/summarize_lag_sensitivity.py --run-prefix burgers_original_seed --retained-modes 10 --out-dir eval_outputs_lag_sensitivity/burgers_original_5seed_summary
python scripts/summarize_lag_sensitivity.py --run-prefix ad_seed --retained-modes 12 --radial-band --out-dir eval_outputs_lag_sensitivity/ad_5seed_summary
python scripts/summarize_lag_sensitivity.py --run-prefix ns_seed --retained-modes 12 --radial-band --out-dir eval_outputs_lag_sensitivity/ns_5seed_summary
python scripts/plot_lag_sensitivity_mechanism_paper.py
```

## Figures and profiling

```bash
python scripts/plot_burgers_rollout.py --direct-model runs_burgers_fno_seed2_original_burgers_core/model --film-model runs_burgers_fno_film_seed2_original_burgers_core/model --data data/BurgersOSG_test.mat --sample -1 --steps 6,20 --out paper_figures/burgers_rollout.pdf
python profiling/profile_ns_overhead.py --save-dir overhead_outputs_ns
python profiling/profile_gl_overhead.py --save-dir overhead_outputs_gl
```

The profiling scripts report parameter counts, training-step time,
inference-step time, and peak CUDA memory on the selected hardware.
