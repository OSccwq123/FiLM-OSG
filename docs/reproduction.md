# Reproducing the FiLM-OSG experiments

Run all commands from the repository root. The examples below use seed 0;
repeat matched experiments with seeds `0,1,2,3,4` for the five-seed summaries.
Tags identify checkpoints and must agree between training and evaluation.

## Burgers

Original benchmark:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --tag original_burgers_core --data-dir data --epochs 1000
python train/run_burgers_fno.py --model fno_film --seed 0 --tag original_burgers_core --data-dir data --epochs 1000
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0 --tag original_burgers_core --data-dir data --save-dir eval_outputs_burgers_original_seed0
```

Additional data with steep gradients:

```bash
python train/run_burgers_fno.py --model fno --seed 0 --tag burgers_sharp_core --data-dir data/burgers_sharp --epochs 1000
python train/run_burgers_fno.py --model fno_film --seed 0 --tag burgers_sharp_core --data-dir data/burgers_sharp --epochs 1000
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0 --tag burgers_sharp_core --data-dir data/burgers_sharp --save-dir eval_outputs_burgers_sharp_seed0
```

Global--local FiLM with mean-zero projection:

```bash
python train/run_burgers_fno.py --model gl_fno_film --seed 0 --tag burgers_sharp_gl_branchwise_proj --data-dir data/burgers_sharp --epochs 1000 --conserve-mean --gl-film-mode branchwise
python eval/eval_burgers_fno.py --models gl_fno_film --seeds 0 --tag burgers_sharp_gl_branchwise_proj --data-dir data/burgers_sharp --save-dir eval_outputs_burgers_sharp_gl_seed0
```

## Advection--diffusion

The matched comparison uses mean-zero projection for both models:

```bash
python train/run_convdiff_fno.py --model fno --seed 0 --tag ad_shared_projection --data-dir data --epochs 500 --conserve-mean
python train/run_convdiff_fno.py --model fno_film --seed 0 --tag ad_shared_projection --data-dir data --epochs 500 --conserve-mean
python eval/eval_convdiff_fno.py --models fno,fno_film --seeds 0 --tag ad_shared_projection --data-dir data --save-dir eval_outputs_ad_seed0
```

Fixed-lag test sets outside the training interval `[0.005, 0.5]` are generated
and evaluated with:

```matlab
cd data
run('../scripts/data_generation/convection_diffusion_fixed_lag_extrapolation.m')
```

```bash
python eval/eval_convdiff_lag_extrapolation.py --models fno,fno_film --seeds 0,1,2 --tag ad_affine_seed012 --data-dir data
```

## Navier--Stokes

Main comparison:

```bash
python train/train_ns_one.py --model fno --seed 0 --tag ns_core --data-dir data --epochs 500
python train/train_ns_one.py --model fno_film --seed 0 --tag ns_core --data-dir data --epochs 500
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0 --tag ns_core --data-dir data --save-dir eval_outputs_ns_seed0
```

Mean-zero projection diagnostic:

```bash
python train/train_ns_one.py --model fno --seed 0 --tag ns_projection_probe --data-dir data --epochs 500 --conserve-mean
python train/train_ns_one.py --model fno_film --seed 0 --tag ns_projection_probe --data-dir data --epochs 500 --conserve-mean
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0 --tag ns_projection_probe --data-dir data --save-dir eval_outputs_ns_projection_seed0
```

The manuscript partition-robustness diagnostic uses the same fixed collection
of admissible partitions for every model and seed:

```bash
python eval/eval_ns_partition_robustness_paper.py --models fno,fno_film --seeds 0 --tag ns_core --model-root . --data-dir data --save-dir eval_outputs_ns_partition_seed0
```

The separate equal-partition diagnostic uses `1,2,4,8` substeps:

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
python scripts/data_generation/convert_pdebench_to_osg.py --input data/pdebench_raw/swe/2D_rdb_NA_NA.h5 --output-dir data/pdebench_osg/swe64_disjoint_full --prefix PDEBenchSWE64DisjointFullOSG --problem-dim 2d --grouped --layout HWD --train-trajectories 800 --test-trajectories 200 --trajectory-split-seed 0 --train-pairs 5000 --test-pairs 1000 --min-lag-steps 1 --max-lag-steps 20 --space-stride 2 --seed 0
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

## Variable-time baselines

The `vt_fno` and `vt_fno_film` models predict the next state directly. The
training scripts disable semigroup regularization, mean projection, and
high-frequency losses for these models. Seedwise summaries can be collected
with:

```bash
python scripts/summarize_vt_baselines.py --root eval_outputs_vt_baselines_5seed
```

## Lag-sensitivity diagnostic

This diagnostic evaluates existing checkpoints; it does not retrain the
models. The CSV key `block0_preactivation` denotes the quantity measured before
the first block activation.

Original Burgers:

```bash
python eval/eval_lag_sensitivity_spectrum.py --benchmark original_burgers --model-seed 0 --direct-model runs_burgers_fno_seed0_burgers_seed01234_final1000/model --film-model runs_burgers_fno_film_seed0_burgers_seed01234_final1000_rerun_seed0/model --data data/BurgersOSG_test.mat --samples 200 --fd-eps 0.01 --out-dir eval_outputs_lag_sensitivity/burgers_original_seed0_layerwise
```

Advection--diffusion:

```bash
python eval/eval_lag_sensitivity_spectrum.py --benchmark advection_diffusion --model-seed 0 --direct-model runs_convdiff_fno_seed0_ad_seed0_fullcover_fno_proj/model --film-model runs_convdiff_fno_film_seed0_ad_seed0_fullcover_film_proj/model --data data/test_data.mat --samples 200 --fd-eps 0.01 --out-dir eval_outputs_lag_sensitivity/ad_seed0_layerwise
```

Navier--Stokes:

```bash
python eval/eval_lag_sensitivity_spectrum.py --benchmark navier_stokes --model-seed 0 --direct-model runs_ns_fno_seed0_ns_seed0_full_fno/model --film-model runs_ns_fno_film_seed0_ns_seed0_full_film/model --data data/VorticityOSG_test.mat --samples 200 --fd-eps 0.01 --out-dir eval_outputs_lag_sensitivity/ns_seed0_layerwise
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
python scripts/plot_method_diagram.py --out-dir paper_figures
python scripts/plot_paper_figures.py --out-dir paper_figures
python scripts/plot_paper_flow_figures.py --out-dir paper_figures --pred-dir paper_figures/predictions
python profiling/profile_ns_overhead.py --save-dir overhead_outputs_ns
python profiling/profile_gl_overhead.py --save-dir overhead_outputs_gl
```

The profiling scripts report parameter counts, training-step time,
inference-step time, and peak CUDA memory on the selected hardware.
