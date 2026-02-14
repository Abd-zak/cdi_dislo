# tests/test_imports.py
from __future__ import annotations

import importlib
import importlib.util
import subprocess

import pytest


def test_import_package_and_version():
    import cdi_dislo  # noqa: F401

    assert hasattr(cdi_dislo, "__version__")
    assert isinstance(cdi_dislo.__version__, str)
    assert cdi_dislo.__version__ != ""


def test_list_modules_non_recursive():
    import cdi_dislo

    mods = cdi_dislo.list_submodules(recursive=False)
    assert isinstance(mods, list)
    assert all(isinstance(m, str) for m in mods)
    assert all(m.startswith("cdi_dislo.") for m in mods)
    assert "cdi_dislo.utils" in mods  # adjust if you ever rename


def test_list_modules_recursive():
    import cdi_dislo

    mods = cdi_dislo.list_submodules(recursive=True)
    assert isinstance(mods, list)
    assert all(isinstance(m, str) for m in mods)
    assert all(m.startswith("cdi_dislo.") for m in mods)
    # some known leaf module (adjust if you ever rename)
    assert "cdi_dislo.utils.utils" in mods


def test_import_all_modules_recursive():
    """
    Strict import test: importing every discovered module must succeed.

    If some modules are truly optional (require viz stack etc.),
    then split them into separate tests with skip conditions.
    """
    import cdi_dislo

    mods = cdi_dislo.list_submodules(recursive=True)
    # sanity: do not silently pass if discovery broke
    assert len(mods) > 0

    for m in mods:
        importlib.import_module(m)


@pytest.mark.skipif(
    importlib.util.find_spec("vtk") is None,
    reason="vtk not installed (viz extra not enabled)",
)
def test_viz_extra_vtk_imports():
    import vtk  # noqa: F401


@pytest.mark.skipif(
    importlib.util.find_spec("pyvista") is None,
    reason="pyvista not installed (viz extra not enabled)",
)
def test_viz_extra_pyvista_imports():
    import pyvista  # noqa: F401


def test_cli_list_modules_runs():
    """
    Ensure the console script entry point works.
    This assumes `[project.scripts] cdi-dislo = "cdi_dislo.cli:main"` is configured.
    """
    result = subprocess.run(
        ["cdi-dislo", "--list-modules"],
        capture_output=True,
        text=True,
        shell=False,
    )
    assert result.returncode == 0
    assert "cdi_dislo.utils" in result.stdout


def test_cli_import_modules_runs():
    """
    CLI import check (non-recursive).
    Use --recursive in CI where you install `[all]` extras.
    """
    result = subprocess.run(
        ["cdi-dislo", "--import-modules"],
        capture_output=True,
        text=True,
        shell=False,
    )
    assert result.returncode == 0
    assert "Imported" in result.stdout
