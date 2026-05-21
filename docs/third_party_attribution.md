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

These attributions are provided to make the provenance of directly adapted and
interface-compatible code explicit. They do not change the manuscript protocol,
model logic, hyperparameters, seed settings, or metric definitions.
