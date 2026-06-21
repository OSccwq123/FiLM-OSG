# Burgers sharp-front GL checkpoint status

The manuscript-facing sharp-front results use the current-code `currentGL2`
checkpoints retrained after the global-local implementation was stabilized. The
five-seed evaluation is stored locally under
`eval_outputs_burgers_sharp_currentgl2_5seed/`; representative checkpoint tags
contain `sharp_currentgl2`.

The historical checkpoints described below are obsolete and are not used for
the manuscript tables or figures.

The historical 5-seed Burgers sharp-front CSV reports finite GL-FNO / GL-FiLM-FNO
metrics for checkpoints such as
`runs_burgers_gl_fno_film_seed0_burgers_sharp_branchwise_proj_seed0_e1000/model`.
A later direct reload of these full-object PyTorch checkpoints under the current
`film_osg.networks.fno` class definitions does not reproduce those metrics.

Observed on 2026-06-03:

- The checkpoint loads as `film_osg.networks.fno.gl_osg_fno1d_with_film`.
- The first rollout step on `data/burgers_sharp/BurgersSharpOSG_test.mat` is
  already unstable, with predictions roughly in `[-62, 37]` for a truth range
  near `[-1.1, 1.1]`.
- A direct forward pass on normalized training batches also produces large
  errors, even though the stored `training_history.csv` decreases to about
  `4.7e-5`.
- The same behavior appears for `gl_fno`, `global_only`, `global_only_hfsg`, and
  `branchwise` Burgers GL seed-0 checkpoints.

This indicates a checkpoint/code-path incompatibility for the historical Burgers
GL checkpoints, not a plotting failure and not a long-rollout-only instability.
Because each checkpoint was saved as a full Python object rather than a portable
state dict with an explicit config/version record, the exact historical GL
forward semantics cannot be reconstructed from the checkpoint alone. Do not use
these historical tags for evaluation; use the current-code `currentGL2`
checkpoints identified above.
