from __future__ import annotations

import importlib
import pkgutil
from importlib.metadata import PackageNotFoundError, version
from typing import List

try:
    __version__ = version("cdi_dislo")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["__version__", "list_submodules", "import_submodules"]


def list_submodules(recursive: bool = False) -> List[str]:
    """
    Return submodules available under cdi_dislo.

    Parameters
    ----------
    recursive : bool
        If True, walk subpackages recursively.

    Returns
    -------
    list of fully qualified module names (strings)
    """
    modules: List[str] = []

    if recursive:
        for _, name, _ in pkgutil.walk_packages(__path__, prefix=f"{__name__}."):
            modules.append(name)
    else:
        for m in pkgutil.iter_modules(__path__):
            modules.append(f"{__name__}.{m.name}")

    return sorted(modules)


def import_submodules(recursive: bool = False) -> List[str]:
    """
    Import submodules (debug helper).

    Returns list of successfully imported module names.
    Raises on import errors (does not silently swallow failures).
    """
    imported: List[str] = []

    for module in list_submodules(recursive=recursive):
        importlib.import_module(module)
        imported.append(module)

    return imported
