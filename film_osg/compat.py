"""Compatibility helpers for loading historical DUE-path model files."""

from __future__ import annotations

import sys
import types


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def install_due_pickle_aliases() -> str:
    """Map common due module names to local film_osg modules for torch.load.

    This lets evaluation scripts load model files that were pickled with old
    `due.networks` / `due.models` module paths without importing the external
    `due` package.
    """

    import film_osg.datasets.pde as local_dataset_pde
    import film_osg.models.pde as local_model_pde
    import film_osg.models.pde_osg as local_pde_osg
    import film_osg.networks.fno as local_fno
    import film_osg.networks.nn as local_nn
    import film_osg.networks.osg_extra_backbones as local_extra
    import film_osg.utils as local_utils

    due_mod = _ensure_module("due")
    datasets_mod = _ensure_module("due.datasets")
    models_mod = _ensure_module("due.models")
    networks_mod = _ensure_module("due.networks")

    sys.modules.setdefault("due.utils", local_utils)
    sys.modules.setdefault("due.datasets.pde", local_dataset_pde)
    sys.modules.setdefault("due.models.pde", local_model_pde)
    sys.modules.setdefault("due.models.pde_osg", local_pde_osg)
    sys.modules.setdefault("due.networks.nn", local_nn)
    sys.modules.setdefault("due.networks.fno", local_fno)
    sys.modules.setdefault("due.networks.osg_extra_backbones", local_extra)

    due_mod.datasets = datasets_mod
    due_mod.models = models_mod
    due_mod.networks = networks_mod
    due_mod.utils = local_utils
    datasets_mod.pde = local_dataset_pde
    models_mod.pde = local_model_pde
    models_mod.pde_osg = local_pde_osg
    networks_mod.nn = local_nn
    networks_mod.fno = local_fno
    networks_mod.osg_extra_backbones = local_extra
    return "film_osg-local-aliases"
