# DUE Dependency Minimization Audit and Migration Plan

## Current State

The active training, evaluation, and profiling entrypoints now prefer local
`film_osg` modules and retain `due` only as a fallback for compatibility.

Local replacements added:

- `film_osg.datasets.pde`: includes `pde_dataset_osg` and `pde_dataset`.
- `film_osg.models.pde`: local PDE training wrapper snapshot.
- `film_osg.models.pde_osg`: local semigroup training wrapper snapshot.
- `film_osg.networks.fno`: local FNO / FiLM-FNO network snapshot.
- `film_osg.networks.osg_extra_backbones`: local U-NO-style, Transolver-style,
  and MambaNO-style network snapshot.
- `film_osg.networks.nn`: local base network class from the supplied DUE source.
- `film_osg.compat`: pickle alias shim for models saved under old `due.*`
  module names.

## Remaining DUE Surface

Active scripts still keep fallback imports from `due` so historical environments
and old pickled models can keep working. These fallbacks should not be removed
until old result directories and server environments are checked.

Archived scripts in `scripts/archive/` still contain direct `due` imports by
design; they are retained for provenance and are not active entrypoints.

## Migration Behavior

Training/profiling import order:

1. Try `film_osg.*`.
2. Fall back to `due.*` only if the local import fails.

Evaluation loading:

1. Install local aliases for common `due.*` module names if `due` is absent.
2. Load models with `torch.load`.
3. If `due` is installed, leave it untouched so old environments behave as
   before.

This preserves old weight compatibility while allowing new smoke runs to avoid
the external `due` package for the code paths now available locally.

## Validation Checklist

Run these without data or weights:

```bash
python train/run_burgers_fno.py --help
python train/run_burgers_fno.py --model fno --seed 0 --dry-run
python eval/eval_burgers_fno.py --check-only --skip-missing
python eval/eval_ns_fno.py --check-only --skip-missing
python profiling/profile_ns_overhead.py --check-only --models fno,fno_film
```

Run these on the server after data is present:

```bash
python -c "from film_osg.datasets.pde import pde_dataset_osg; from film_osg.networks.fno import osg_fno1d; from film_osg.models.pde_osg import PDE_osg; print('film_osg imports ok')"
python train/run_burgers_fno.py --model fno --seed 0 --epochs 1 --batch-size 2 --tag localpkg_smoke
python eval/eval_burgers_fno.py --models fno --seeds 0 --tag localpkg_smoke --eval-steps 1 --save-dir eval_outputs_localpkg_smoke
python eval/eval_ns_fno.py --models fno,fno_film --seeds 0 --tag localpkg_smoke --eval-steps 1 --save-dir eval_outputs_ns_fno_localpkg_smoke
```

## Next Steps

- Keep the `due` fallback until all old `.pt`/`model` files needed for the
  paper have been loaded successfully through `film_osg.compat`.
- If any future script imports another `due.*` module, copy that source into
  `film_osg` and update this audit before removing the fallback.
- Once all smoke and seed-0 reproduction runs pass without external `due`, make
  `due` fallback optional behind a documented compatibility flag.
