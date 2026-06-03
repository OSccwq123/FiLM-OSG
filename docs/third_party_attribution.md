# Third-Party Attribution

This repository includes local `film_osg` modules that preserve the interfaces
needed to reproduce the FiLM-OSG experiments without requiring the external
`due` package on the primary execution path.

Several local modules are adapted from the DUE project:

- Repository: https://github.com/AI4Equations/due
- License: LGPL-2.1
- Local adapted modules:
  - `film_osg/datasets/pde.py`
  - `film_osg/models/pde.py`
  - `film_osg/models/pde_osg.py`
  - `film_osg/networks/nn.py`
  - `film_osg/networks/fno.py`
  - `film_osg/utils.py`

The extra-backbone implementation in `film_osg/networks/osg_extra_backbones.py`
is local experiment code, but it intentionally follows the DUE-style model
interface so it can be trained and evaluated through the same OSG-compatible
wrappers.

The FNO implementation also follows the public neuraloperator Fourier neural
operator reference:

- https://github.com/neuraloperator/neuraloperator/blob/master/fourier_2d_time.py
- License: MIT

The U-NO-style, Transolver-style, and MambaNO-style modules in
`film_osg/networks/osg_extra_backbones.py` are local OSG-adapted architecture
variants used for portability diagnostics. The names describe paper/project
inspirations; this repository does not vendor the upstream U-NO, Transolver, or
MambaNO codebases. The optional `mamba_ssm` import, if installed by a user, is
an external state-spaces/mamba dependency distributed under Apache-2.0.

These attributions are provided to make the provenance of directly adapted,
referenced, and interface-compatible code explicit. They do not change the
manuscript protocol, model logic, hyperparameters, seed settings, or metric
definitions.

The repository-level `NOTICE` file repeats this provenance in a compact notice
format for redistribution.

The final top-level repository license is intentionally left pending until the
authors and advisor confirm the appropriate release terms. Before public
release, confirm whether additional LGPL-2.1 notices, source-availability text,
or license-file placement are needed for files adapted from AI4Equations/DUE.
Also confirm the final citation and license treatment for the neuraloperator
FNO reference, optional Mamba dependency, and architecture-reference projects.
`THIRD_PARTY_LICENSES/` contains the currently tracked third-party license texts and remaining architecture-reference release-review notes.
