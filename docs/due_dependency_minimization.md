# DUE Dependency Minimization Audit and Migration Plan

## Current State

The active training, evaluation, and profiling entrypoints use local `film_osg`
modules and do not require the external `due` package.

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

Active scripts no longer import from external `due`. Evaluation keeps a local
pickle alias shim so model files saved under old `due.*` module paths can be
loaded through local `film_osg` modules.

Archived scripts in `scripts/archive/` still contain direct `due` imports by
design; they are retained for provenance and are not active entrypoints.

## Migration Behavior

Training/profiling import behavior:

1. Import `film_osg.*`.
2. Fail fast if the local package is incomplete, rather than falling back to
   external `due`.

Evaluation loading:

1. Install local aliases for common `due.*` module names.
2. Load models with `torch.load`.
3. Historical `due.*` pickle paths resolve to local `film_osg` modules.

This preserves old weight compatibility while avoiding the external `due`
package for active code paths.

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

- If any future script imports another `due.*` module, copy that source into
  `film_osg` and update this audit instead of adding a new external `due`
  dependency.
- Keep DUE as provenance/background attribution in `NOTICE` and
  `docs/third_party_attribution.md`.
