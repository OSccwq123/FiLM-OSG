# Reproducing the FiLM-OSG experiments

Run the commands below from the repository root. Training tags are arbitrary
identifiers, but the same tag must be passed to the corresponding evaluator.
The reported comparisons use seeds `0,1,2,3,4`, except for the U-NO-style and
Transolver-style experiments, which use seeds `0,1,2`. Replace `0,1` in the
launcher commands with the GPU IDs available on the local machine.

For the principal results, prepare the Burgers, advection–diffusion, and
Navier–Stokes data below and then run the three workflows under **Main matched
comparisons**. The remaining sections cover the additional experiments and
diagnostics reported in the manuscript.

The evaluation programs write a seed-wise CSV, a mean-and-standard-deviation
CSV, and a paired-comparison CSV. These files contain the scalar values used in
the manuscript tables.

## Data

### Burgers and advection–diffusion

The MATLAB programs write to `data/` by default. The environment variable is
shown here to make the destination explicit:

```matlab
setenv('FILM_OSG_OUTPUT_DIR', fullfile(pwd, 'data'))
run('data/generation/Burgers.m')
run('data/generation/convection_diffusion.m')
```

The Navier–Stokes experiments use `VorticityOSG_train.mat` and
`VorticityOSG_test.mat` from the public DUE benchmark. Place both files in
`data/`.

## Main matched comparisons

### Burgers

```bash
python scripts/launch_burgers_fno.py --gpus 0,1 --models fno,fno_film --seeds 0,1,2,3,4 --tag burgers_main --dataset original --data-dir data
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0,1,2,3,4 --tag burgers_main --dataset original --data-dir data --save-dir results/burgers_main
python scripts/plot_burgers_rollout.py --direct-model runs_burgers_fno_seed2_burgers_main/model --film-model runs_burgers_fno_film_seed2_burgers_main/model --data data/BurgersOSG_test.mat --sample -1 --steps 6,20 --out paper_figures/burgers_rollout.pdf
```

### Advection–diffusion

Both models use the mean-zero increment projection in this comparison.

```bash
python scripts/launch_convdiff_fno.py --gpus 0,1 --models fno,fno_film --seeds 0,1,2,3,4 --tag ad_main --data-dir data --conserve-mean
python eval/eval_convdiff_fno.py --models fno,fno_film --seeds 0,1,2,3,4 --tag ad_main --data-dir data --save-dir results/ad_main
python eval/eval_convdiff_fno.py --models fno,fno_film --seeds 0 --tag ad_main --data-dir data --save-dir results/ad_figure --save-mat
python scripts/plot_convdiff_rollout.py --direct results/ad_figure/fno_seed0_ad_main_full_predictions.mat --film results/ad_figure/fno_film_seed0_ad_main_full_predictions.mat --output paper_figures/convdiff_rollout.pdf
```

### Navier–Stokes

```bash
python scripts/launch_ns_fno.py --gpus 0,1 --models fno,fno_film --seeds 0,1,2,3,4 --tag ns_main --data-dir data
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0,1,2,3,4 --tag ns_main --data-dir data --save-dir results/ns_main
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0 --tag ns_main --data-dir data --save-dir results/ns_figure --save-mat
python scripts/plot_ns_rollout.py --direct results/ns_figure/fno_seed0_ns_main_full_predictions.mat --film results/ns_figure/fno_film_seed0_ns_main_full_predictions.mat --output-dir paper_figures/ns
```

## Additional experiments and diagnostics

### Additional Burgers experiments

Generate the data with steep gradients before running these comparisons:

```bash
python data/generation/generate_burgers_sharp_osg.py --out-dir data/burgers_sharp
python data/generation/check_osg_mat_data.py --train data/burgers_sharp/BurgersSharpOSG_train.mat --test data/burgers_sharp/BurgersSharpOSG_test.mat --problem-dim 1d --expected-channels 1 --require-positive-dt
```

The first comparison uses the steep-gradient data without projection; the
second applies the mean-zero increment projection to both models.

```bash
python scripts/launch_burgers_fno.py --gpus 0,1 --models fno,fno_film --seeds 0,1,2,3,4 --tag burgers_sharp --dataset sharp --data-dir data/burgers_sharp
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0,1,2,3,4 --tag burgers_sharp --dataset sharp --data-dir data/burgers_sharp --save-dir results/burgers_sharp

python scripts/launch_burgers_fno.py --gpus 0,1 --models fno,fno_film --seeds 0,1,2,3,4 --tag burgers_sharp_projection --dataset sharp --data-dir data/burgers_sharp --conserve-mean
python eval/eval_burgers_fno.py --models fno,fno_film --seeds 0,1,2,3,4 --tag burgers_sharp_projection --dataset sharp --data-dir data/burgers_sharp --save-dir results/burgers_sharp_projection
```

The global–local table uses four separately trained configurations. All use
the projection. The HF/HF-SG run uses weights `0.01` and `0.001` with a 10%
warm-up.

```bash
python scripts/launch_burgers_fno.py --gpus 0,1 --models gl_fno --seeds 0,1,2,3,4 --tag burgers_gl_input --dataset sharp --data-dir data/burgers_sharp --conserve-mean
python scripts/launch_burgers_fno.py --gpus 0,1 --models gl_fno_film --seeds 0,1,2,3,4 --tag burgers_gl_global --dataset sharp --data-dir data/burgers_sharp --conserve-mean --gl-film-mode global_only
python scripts/launch_burgers_fno.py --gpus 0,1 --models gl_fno_film --seeds 0,1,2,3,4 --tag burgers_gl_branchwise --dataset sharp --data-dir data/burgers_sharp --conserve-mean --gl-film-mode branchwise
python scripts/launch_burgers_fno.py --gpus 0,1 --models gl_fno_film --seeds 0,1,2,3,4 --tag burgers_gl_branchwise_hf --dataset sharp --data-dir data/burgers_sharp --conserve-mean --gl-film-mode branchwise --hf-weight 0.01 --hf-sg-weight 0.001 --hf-warmup-frac 0.1

python eval/eval_burgers_fno.py --models gl_fno --seeds 0,1,2,3,4 --tag burgers_gl_input --dataset sharp --data-dir data/burgers_sharp --save-dir results/burgers_gl_input
python eval/eval_burgers_fno.py --models gl_fno_film --seeds 0,1,2,3,4 --tag burgers_gl_global --dataset sharp --data-dir data/burgers_sharp --save-dir results/burgers_gl_global
python eval/eval_burgers_fno.py --models gl_fno_film --seeds 0,1,2,3,4 --tag burgers_gl_branchwise --dataset sharp --data-dir data/burgers_sharp --save-dir results/burgers_gl_branchwise
python eval/eval_burgers_fno.py --models gl_fno_film --seeds 0,1,2,3,4 --tag burgers_gl_branchwise_hf --dataset sharp --data-dir data/burgers_sharp --save-dir results/burgers_gl_branchwise_hf
```

### Fixed-time advection–diffusion evaluation

Generate the three test sets at `0.0025`, `0.75`, and `1.0`, then evaluate the
models from the main advection–diffusion experiment without retraining.

```matlab
setenv('FILM_OSG_OUTPUT_DIR', fullfile(pwd, 'data'))
run('data/generation/convection_diffusion_fixed_time.m')
```

```bash
python eval/eval_convdiff_fixed_time.py --models fno,fno_film --seeds 0,1,2,3,4 --tag ad_main --data-dir data --save-dir results/ad_fixed_time
```

### Navier–Stokes diagnostics

The partition experiment evaluates the main checkpoints at terminal times
`20,40,60,80` using the shared set of admissible partitions from the paper.

```bash
python eval/eval_ns_time_partitions.py --models fno,fno_film --seeds 0,1,2,3,4 --tag ns_main --model-root . --data-dir data --save-dir results/ns_partitions
```

The input-concatenation tuning table contains the main width-20,
learning-rate-`1e-3` run above and the following three five-seed runs:

```bash
python scripts/launch_ns_fno.py --gpus 0,1 --models fno --seeds 0,1,2,3,4 --tag ns_input_lr2e3 --data-dir data --learning-rate 2e-3
python scripts/launch_ns_fno.py --gpus 0,1 --models fno --seeds 0,1,2,3,4 --tag ns_input_lr5e4 --data-dir data --learning-rate 5e-4
python scripts/launch_ns_fno.py --gpus 0,1 --models fno --seeds 0,1,2,3,4 --tag ns_input_width32 --data-dir data --width 32
python eval/eval_ns_fno.py --models fno --seeds 0,1,2,3,4 --tag ns_input_lr2e3 --data-dir data --save-dir results/ns_input_lr2e3
python eval/eval_ns_fno.py --models fno --seeds 0,1,2,3,4 --tag ns_input_lr5e4 --data-dir data --save-dir results/ns_input_lr5e4
python eval/eval_ns_fno.py --models fno --seeds 0,1,2,3,4 --tag ns_input_width32 --data-dir data --save-dir results/ns_input_width32
```

The projection appendix uses a separate matched five-seed training run:

```bash
python scripts/launch_ns_fno.py --gpus 0,1 --models fno,fno_film --seeds 0,1,2,3,4 --tag ns_projection --data-dir data --conserve-mean
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0,1,2,3,4 --tag ns_projection --data-dir data --save-dir results/ns_projection --mean-drift
```

### Spectral response to the evolution time

This diagnostic evaluates the trained checkpoints and does not retrain them.
It is run once per seed for each main benchmark.

```bash
for model_seed in 0 1 2 3 4; do
  python eval/eval_evolution_time_sensitivity.py --benchmark original_burgers --model-seed "$model_seed" --direct-model "runs_burgers_fno_seed${model_seed}_burgers_main/model" --film-model "runs_burgers_fno_film_seed${model_seed}_burgers_main/model" --data data/BurgersOSG_test.mat --samples 200 --fd-eps 0.01 --out-dir "eval_outputs_evolution_time_sensitivity/burgers_original_seed${model_seed}_layerwise"
  python eval/eval_evolution_time_sensitivity.py --benchmark advection_diffusion --model-seed "$model_seed" --direct-model "runs_convdiff_fno_seed${model_seed}_ad_main/model" --film-model "runs_convdiff_fno_film_seed${model_seed}_ad_main/model" --data data/test_data.mat --samples 200 --fd-eps 0.01 --out-dir "eval_outputs_evolution_time_sensitivity/ad_seed${model_seed}_layerwise"
  python eval/eval_evolution_time_sensitivity.py --benchmark navier_stokes --model-seed "$model_seed" --direct-model "runs_ns_fno_seed${model_seed}_ns_main/model" --film-model "runs_ns_fno_film_seed${model_seed}_ns_main/model" --data data/VorticityOSG_test.mat --samples 200 --fd-eps 0.01 --out-dir "eval_outputs_evolution_time_sensitivity/ns_seed${model_seed}_layerwise"
done

python scripts/summarize_evolution_time_spectra.py --run-prefix burgers_original_seed --retained-modes 10 --out-dir eval_outputs_evolution_time_sensitivity/burgers_original_5seed_summary
python scripts/summarize_evolution_time_spectra.py --run-prefix ad_seed --retained-modes 12 --radial-band --out-dir eval_outputs_evolution_time_sensitivity/ad_5seed_summary
python scripts/summarize_evolution_time_spectra.py --run-prefix ns_seed --retained-modes 12 --radial-band --out-dir eval_outputs_evolution_time_sensitivity/ns_5seed_summary
python scripts/summarize_evolution_time_scalars.py --root eval_outputs_evolution_time_sensitivity --seeds 0,1,2,3,4 --out-dir eval_outputs_evolution_time_sensitivity/scalar_summary
python scripts/plot_evolution_time_sensitivity.py --input-dir eval_outputs_evolution_time_sensitivity --output-dir paper_figures
```

The scalar summary command produces CSV, Markdown, and LaTeX outputs. In
particular, `nonzero_fraction_low_band_mean` gives the values quoted in the
spectral-response discussion.

### PDEBench-derived comparisons

Download `2D_rdb_NA_NA.h5` and `2D_diff-react_NA_NA.h5` from the
[PDEBench archive](https://doi.org/10.18419/darus-2986) and place them at

```text
data/pdebench_raw/swe/2D_rdb_NA_NA.h5
data/pdebench_raw/reacdiff2d/2D_diff-react_NA_NA.h5
```

The following commands reproduce the trajectory-disjoint conversion used in
the manuscript. The same split and pair seeds are used at both
reaction–diffusion resolutions.

```bash
python data/generation/convert_pdebench_to_osg.py --input data/pdebench_raw/swe/2D_rdb_NA_NA.h5 --output-dir data/pdebench_osg/swe64 --prefix PDEBenchSWE64 --train-trajectories 800 --test-trajectories 200 --train-pairs 5000 --test-pairs 1000 --min-time-offset-steps 1 --max-time-offset-steps 20 --space-stride 2 --split-seed 0 --pair-seed 0
python data/generation/check_osg_mat_data.py --train data/pdebench_osg/swe64/PDEBenchSWE64_train.mat --test data/pdebench_osg/swe64/PDEBenchSWE64_test.mat --problem-dim 2d --expected-channels 1 --expected-time 2 --require-positive-dt
ln -sfn PDEBenchSWE64_train.mat data/pdebench_osg/swe64/train_data.mat
ln -sfn PDEBenchSWE64_test.mat data/pdebench_osg/swe64/test_data.mat

python data/generation/convert_pdebench_to_osg.py --input data/pdebench_raw/reacdiff2d/2D_diff-react_NA_NA.h5 --output-dir data/pdebench_osg/reacdiff64 --prefix PDEBenchReacDiff64 --train-trajectories 800 --test-trajectories 200 --train-pairs 5000 --test-pairs 1000 --min-time-offset-steps 1 --max-time-offset-steps 20 --space-stride 2 --split-seed 0 --pair-seed 0
python data/generation/check_osg_mat_data.py --train data/pdebench_osg/reacdiff64/PDEBenchReacDiff64_train.mat --test data/pdebench_osg/reacdiff64/PDEBenchReacDiff64_test.mat --problem-dim 2d --expected-channels 2 --expected-time 2 --require-positive-dt
ln -sfn PDEBenchReacDiff64_train.mat data/pdebench_osg/reacdiff64/train_data.mat
ln -sfn PDEBenchReacDiff64_test.mat data/pdebench_osg/reacdiff64/test_data.mat

python data/generation/convert_pdebench_to_osg.py --input data/pdebench_raw/reacdiff2d/2D_diff-react_NA_NA.h5 --output-dir data/pdebench_osg/reacdiff128 --prefix PDEBenchReacDiff128 --train-trajectories 800 --test-trajectories 200 --train-pairs 5000 --test-pairs 1000 --min-time-offset-steps 1 --max-time-offset-steps 20 --space-stride 1 --split-seed 0 --pair-seed 0
python data/generation/check_osg_mat_data.py --train data/pdebench_osg/reacdiff128/PDEBenchReacDiff128_train.mat --test data/pdebench_osg/reacdiff128/PDEBenchReacDiff128_test.mat --problem-dim 2d --expected-channels 2 --expected-time 2 --require-positive-dt
ln -sfn PDEBenchReacDiff128_train.mat data/pdebench_osg/reacdiff128/train_data.mat
ln -sfn PDEBenchReacDiff128_test.mat data/pdebench_osg/reacdiff128/test_data.mat
```

On systems without symbolic links, copy each generated file to the indicated
`train_data.mat` or `test_data.mat` name instead.

The `64 x 64` experiments use batch size 100 and include the two direct-time
controls. The `128 x 128` experiment uses batch size 20 and compares only the
two OSG models.

```bash
python scripts/launch_convdiff_fno.py --gpus 0,1 --models fno,fno_film,vt_fno,vt_fno_film --seeds 0,1,2,3,4 --tag pdebench_swe64 --data-dir data/pdebench_osg/swe64 --batch-size 100 --problem-dim 1
python eval/eval_pdebench.py --models fno,fno_film,vt_fno,vt_fno_film --seeds 0,1,2,3,4 --tag pdebench_swe64 --data-dir data/pdebench_osg/swe64 --save-dir results/pdebench_swe64

python scripts/launch_convdiff_fno.py --gpus 0,1 --models fno,fno_film,vt_fno,vt_fno_film --seeds 0,1,2,3,4 --tag pdebench_reacdiff64 --data-dir data/pdebench_osg/reacdiff64 --batch-size 100 --problem-dim 2
python eval/eval_pdebench.py --models fno,fno_film,vt_fno,vt_fno_film --seeds 0,1,2,3,4 --tag pdebench_reacdiff64 --data-dir data/pdebench_osg/reacdiff64 --save-dir results/pdebench_reacdiff64

python scripts/launch_convdiff_fno.py --gpus 0,1 --models fno,fno_film --seeds 0,1,2,3,4 --tag pdebench_reacdiff128 --data-dir data/pdebench_osg/reacdiff128 --batch-size 20 --problem-dim 2
python eval/eval_pdebench.py --models fno,fno_film --seeds 0,1,2,3,4 --tag pdebench_reacdiff128 --data-dir data/pdebench_osg/reacdiff128 --save-dir results/pdebench_reacdiff128
```

Each converted sample contains one transition, so the relative errors are
single-transition quantities rather than long-rollout errors.

### Other Navier–Stokes backbones

```bash
python scripts/launch_ns_extra_backbones.py --gpus 0,1 --models uno,uno_film,transolver,transolver_film --seeds 0,1,2 --tag ns_other_backbones --data-dir data
python eval/eval_ns_extra_backbones.py --models uno,uno_film,transolver,transolver_film --seeds 0,1,2 --tag ns_other_backbones --data-dir data --save-dir results/ns_other_backbones
```

### Computational cost

The manuscript timings use an NVIDIA A100-SXM4-40GB, batch size 20, 50 warm-up
iterations, and 200 timed iterations for the Navier–Stokes comparison.

```bash
python profiling/profile_ns_overhead.py --models fno,fno_film,uno,uno_film,transolver,transolver_film --batch-size 20 --warmup 50 --iters 200 --data-dir data --save-dir results/ns_overhead
python profiling/profile_gl_overhead.py --cases burgers_sharp --models film,gl_film --warmup 50 --iters 200 --data-dir data/burgers_sharp --save-dir results/burgers_gl_overhead
```

Timing values depend on the GPU and software environment. Parameter counts do
not.

## Scalar output files

No separate manual aggregation step is needed for the error tables. The
evaluators above write these files directly:

| Experiment | Scalar summaries |
| --- | --- |
| Burgers | `burgers_fno_summary_by_model.csv`, `burgers_fno_paired_summary.csv` |
| Advection–diffusion | `convdiff_fno_summary_by_model.csv`, `convdiff_fno_paired_summary.csv` |
| Navier–Stokes | `ns_fno_summary_by_model.csv`, `ns_fno_paired_summary.csv` |
| Fixed-time advection–diffusion | one summary and paired-summary CSV per test file |
| Partition experiment | `partition_summary_by_model_T.csv`, `partition_paired_summary.csv` |
| PDEBench-derived data | `summary_by_model.csv`, `paired_summary.csv` |
| Other backbones | `extra_backbones_summary_by_model.csv`, `extra_backbones_paired_summary.csv` |
| Spectral response | `scalar_summary/evolution_time_sensitivity_scalar_summary.csv` |

All reported standard deviations are population standard deviations across the
specified seeds.
