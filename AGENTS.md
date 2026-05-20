# AGENTS.md

This repository organizes reproducibility code for the FiLM-OSG manuscript.

## Project goal

Organize training, evaluation, overhead-profiling, and ablation scripts for the FiLM-OSG experiments. Do not redesign the method or change the scientific claims. Preserve the manuscript's current experimental protocol.

## Current manuscript protocol

Main FNO experiments:
- Burgers: OSG-FNO vs FiLM-OSG-FNO, seeds {0,1,2}, epochs 1000, batch size 100, sg_pairing=2, sg_weight=5.0, multiscale=True.
- Advection--diffusion: OSG-FNO vs FiLM-OSG-FNO, seeds {0,1,2}, lag range [0.005,0.5], epochs 500, batch size 100, sg_pairing=1, sg_weight=1.0, multiscale=False.
- Navier--Stokes: OSG-FNO vs FiLM-OSG-FNO, seeds {0,1,2,3,4}, epochs 500, batch size 20, sg_pairing=1, sg_weight=1.0, multiscale=False.

Additional Navier--Stokes diagnostics:
- Partition robustness: OSG-FNO vs FiLM-OSG-FNO, seeds {0,1,2,3,4}.
- Non-FNO portability: OSG-adapted U-NO-style and Transolver-style variants, seeds {0,1,2}.
- Overhead profile: FNO, U-NO-style, Transolver-style direct-lag and FiLM-conditioned variants, representative NS batch size 20.
- SG ablation: direct-lag / FiLM-conditioned FNO with and without auxiliary semigroup loss, target seeds {0,1,2}.
- MambaNO-style check is supplementary only, seed-sensitive, not a main portability claim.

## Important terminology

Use:
- framework for the full FiLM-OSG method;
- conditioning interface for how lag conditioning attaches to different backbones;
- wrapper only for local implementation-level code;
- lag, variable-lag, lag conditioning.

Avoid:
- timestep conditioning;
- temporal conditioning, unless referring broadly to time in background text;
- claiming exact reproduction of U-NO, Transolver, or MambaNO. These are OSG-adapted, style/inspired variants.

## Code constraints

The local machine may not have a GPU. Do not run full training locally.
Local work should focus on:
- syntax checks;
- import checks;
- CLI --help checks;
- path checks;
- dry-run modes;
- small CPU smoke tests when possible.

Do not modify model definitions unless necessary for CLI compatibility or reproducibility.
Do not change experimental hyperparameters without explicit instruction.
Do not delete historical result files.
Do not fabricate metrics.

## Expected output structure

Organize code into:
- train/
- eval/
- profiling/
- scripts/
- configs/
- docs/

Plotting scripts are optional for now. Evaluation outputs should be sufficient to reproduce tables through saved json/csv/mat files.