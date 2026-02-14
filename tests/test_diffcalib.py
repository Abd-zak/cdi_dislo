# tests/test_diffcalib.py
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _import_diffcalib():
    # Keep local import so tests don’t import heavy deps at collection time
    from cdi_dislo.calibration import diffcalib

    return diffcalib


def _as_1d_len_n(x, n: int) -> np.ndarray:
    """
    Convert scalar or array-like to a 1D array of length n (broadcast scalars).
    """
    arr = np.asarray(x)
    if arr.shape == ():  # scalar
        return np.full(n, float(arr), dtype=float)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    assert arr.shape == (n,), f"Expected length {n}, got {arr.shape}"
    return arr


def _has_1d_len_n_array(out, n: int) -> bool:
    """
    True if output (tuple/list/dict/ndarray/scalar) contains at least one
    1D array-like of length n.
    """
    if out is None:
        return False

    candidates = []

    if isinstance(out, dict):
        candidates.extend(out.values())
    elif isinstance(out, (tuple, list)):
        candidates.extend(out)
    else:
        candidates.append(out)

    for v in candidates:
        try:
            a = np.asarray(v)
        except Exception:
            continue
        if a.shape == (n,):
            return True
    return False


def _all_finite_1d_len_n_arrays(out, n: int) -> bool:
    """
    True if every 1D array-like of length n found inside `out` is finite.
    """
    if out is None:
        return False

    candidates = []

    if isinstance(out, dict):
        candidates.extend(out.values())
    elif isinstance(out, (tuple, list)):
        candidates.extend(out)
    else:
        candidates.append(out)

    found_any = False
    for v in candidates:
        try:
            a = np.asarray(v, dtype=float)
        except Exception:
            continue
        if a.shape == (n,):
            found_any = True
            if not np.all(np.isfinite(a)):
                return False
    return found_any


def test_extract_coefficient_and_exponent_negative():
    diffcalib = _import_diffcalib()

    coeff, exp = diffcalib.extract_coefficient_and_exponent(-0.0123)
    # -0.0123 = -1.23 * 10^-2
    assert exp == -2
    assert np.isclose(coeff, -1.23, atol=1e-12)


def test_GET_theta_lambda_energy_latticeconstant_reference_stubs_lam2en(monkeypatch):
    """
    Avoid importing real xrayutilities (may not be installed in minimal env).
    We stub xrayutilities.lam2en and the sapphire lattice predictor.

    We validate output shapes and finiteness, while allowing lam/energy to be
    either scalars (typical if wavelength is constant) or length-n arrays.
    """
    diffcalib = _import_diffcalib()

    # Stub lam2en in the module namespace (it imports inside the function).
    import sys

    fake_xu = SimpleNamespace(
        lam2en=lambda lam: 12.398419843320026 / lam
    )  # keV·Å / Å -> keV
    monkeypatch.setitem(sys.modules, "xrayutilities", fake_xu)

    # Stub lattice predictor so it returns a known lattice parameter at RT index
    def fake_pred(T_celsius, plot=False):
        return (12.9, 13.0, np.zeros(5))

    monkeypatch.setattr(
        diffcalib, "get_prediction_from_theo_sapphire_lattice", fake_pred
    )

    # Build minimal "results" array with needed columns:
    # results[:, 3]=gamma_scan, results[:, 4]=delta_scan
    # results[:, 8]=com_y, results[:, 9]=com_z
    # results[:, 12]=cchx, results[:, 13]=cchy
    # results[:, 14]=fact
    n = 5
    results = np.zeros((n, 15), dtype=float)
    results[:, 3] = 1.0  # gamma
    results[:, 4] = 2.0  # delta
    results[:, 8] = 0.1  # com_y
    results[:, 9] = -0.2  # com_z
    results[:, 12] = 0.0  # cchx
    results[:, 13] = 0.0  # cchy
    results[:, 14] = 100.0  # fact

    temp_experimental = np.array([20, 30, 40, 50, 60], dtype=float)

    theta, lam, energy, C = diffcalib.GET_theta_lambda_energy_latticeconstant_reference(
        results=results,
        temp_experimental=temp_experimental,
        experiment_name="unit-test",
    )

    assert np.asarray(theta).shape == (n,)
    assert np.asarray(C).shape == (n,)

    lam_arr = _as_1d_len_n(lam, n)
    energy_arr = _as_1d_len_n(energy, n)

    assert np.all(np.isfinite(theta))
    assert np.all(np.isfinite(lam_arr))
    assert np.all(np.isfinite(energy_arr))
    assert np.all(np.isfinite(C))


@pytest.mark.parametrize("fit_order", [1, 2, 3, 4, 5, 6, "exp"])
def test_calculate_epsilon_and_prediction_no_plot_fast(fit_order, monkeypatch):
    """
    This test is designed to reflect typical scientific usage:
    - enough experimental points to support high-order polynomial fits
    - no GUI/plotting
    - output must contain at least one length-n finite array (epsilon/prediction/etc.)
    """
    diffcalib = _import_diffcalib()
    monkeypatch.setenv("MPLBACKEND", "Agg")

    # Use enough points so polynomial order up to 6 is always feasible (>= 7 points).
    n = 25
    T_data_exp = np.linspace(0, 500, n, dtype=float)

    a0 = 12.991
    # Slight, realistic thermal expansion-like trend
    results_sapphire_d_spacing = a0 * (1.0 + 1e-5 * (T_data_exp - 20.0))

    # Suppress overflow warnings that may occur in exponential model trial fits,
    # while still failing if NaNs/infs propagate into the returned arrays.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        out = diffcalib.calculate_epsilon_and_prediction(
            T_data_exp=T_data_exp,
            results_sapphire_d_spacing=results_sapphire_d_spacing,
            fit_order=fit_order,
            plot=False,
            a0=a0,
            experiment="unit-test",
        )

    assert out is not None
    assert _has_1d_len_n_array(
        out, n
    ), "Expected at least one 1D output array of length n."
    assert _all_finite_1d_len_n_arrays(
        out, n
    ), "Found non-finite values in 1D length-n outputs."
