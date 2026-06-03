# Burgers sharp-front GL checkpoint reproducibility note

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
Because the checkpoint was saved as a full Python object rather than a portable
state dict with an explicit config/version record, the exact historical GL
forward semantics cannot be reconstructed from the checkpoint alone. Until these
models are retrained or the original class implementation is recovered, Burgers
GL curves should not be used in manuscript figures or as primary evidence.
