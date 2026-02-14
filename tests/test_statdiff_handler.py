import warnings

import numpy as np
import pytest
from scipy.optimize import OptimizeWarning

warnings.filterwarnings("ignore", category=OptimizeWarning)


def _make_3d_diffraction_like(
    shape=(20, 20, 20),
    cube_halfwidth=9.0,  # controls overall size
    truncation=14.0,  # controls {111}-like truncation: |x|+|y|+|z| <= truncation
    recip_blur_sigma=0.8,  # Gaussian blur in reciprocal space (pixels)
    background=2.0,  # constant background counts
    peak_scale=2e4,  # scales overall intensity (counts)
    poisson=True,
    seed=0,
):
    """
    Returns a 3D reciprocal-space intensity volume (non-negative floats),
    resembling a coherent diffraction pattern from a faceted particle.
    """
    rng = np.random.default_rng(seed)

    # --- Real-space faceted support (simple truncated cube / truncated octahedron-like) ---
    nz, ny, nx = shape
    z, y, x = np.indices(shape, dtype=float)
    x -= (nx - 1) / 2
    y -= (ny - 1) / 2
    z -= (nz - 1) / 2

    # cube constraint + octahedral truncation constraint
    support = (
        (np.abs(x) <= cube_halfwidth)
        & (np.abs(y) <= cube_halfwidth)
        & (np.abs(z) <= cube_halfwidth)
        & ((np.abs(x) + np.abs(y) + np.abs(z)) <= truncation)
    )

    # Uniform density inside (you can also add mild internal strain-phase if you want later)
    rho = support.astype(float)

    # --- Reciprocal-space intensity: |FFT(rho)|^2 ---
    F = np.fft.fftn(rho)
    Intensity = np.abs(np.fft.fftshift(F)) ** 2

    # Normalize and scale to counts
    Intensity = Intensity / (Intensity.max() + 1e-12)
    Intensity = peak_scale * Intensity + background

    # --- Optional reciprocal-space Gaussian blur (instrument resolution) ---
    if recip_blur_sigma and recip_blur_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
        except Exception:
            gaussian_filter = None
        if gaussian_filter is not None:
            Intensity = gaussian_filter(
                Intensity, sigma=recip_blur_sigma, mode="nearest"
            )

    # --- Poisson counting noise (more realistic for diffraction) ---
    if poisson:
        Intensity = rng.poisson(lam=np.clip(Intensity, 0, None)).astype(float)
    else:
        # small additive Gaussian noise fallback
        Intensity = Intensity + 0.01 * Intensity.max() * rng.normal(
            size=Intensity.shape
        )

    return Intensity


def test_theta_bragg_pt_matches_manual():
    from cdi_dislo.statistics import statdiff_handler as s

    lam = 1.0  # Å
    h, k, l_ = 1, 1, 1
    a0 = 3.924  # Å

    # manual: d = a0/sqrt(h^2+k^2+l^2), theta = arcsin(lam/(2d))
    d = a0 / np.sqrt(h * h + k * k + l_ * l_)
    theta_manual = np.arcsin(lam / (2.0 * d)) * 180.0 / np.pi

    theta = s.theta_bragg_pt(lam, h, k, l_, a0=a0)
    assert np.isfinite(theta)
    assert theta == pytest.approx(theta_manual, rel=0, abs=1e-12)


def test_pseudo_voigt_fwhm_scherrer_matches_formula():
    from cdi_dislo.statistics import statdiff_handler as s

    lam = 1.0  # Å
    beta = 0.01  # radians
    theta_deg = 30.0
    k = 0.9

    theta_rad = np.deg2rad(theta_deg)
    expected = (k * lam) / (beta * np.cos(theta_rad))

    out = s.pseudo_voigt_fwhm_Scherrer(lam, beta, theta_deg, k=k)
    assert np.isfinite(out)
    assert out == pytest.approx(expected, rel=0, abs=1e-12)


def test_fwhm_integral_1_basic_properties():
    from cdi_dislo.statistics import statdiff_handler as s

    x = np.linspace(-10, 10, 2001)
    y = np.exp(-0.5 * (x / 2.0) ** 2)  # sigma=2

    val = s.fwhm_integral_1(x, y)
    assert np.isfinite(val)
    assert val > 0

    # sanity: integral over FWHM region must be less than total integral
    total = np.trapz(y, x)
    assert val < total


@pytest.mark.parametrize("step_qxqyqz", [None, (1e-3, 2e-3, 3e-3)])
def test_get_plot_fwhm_and_skewness_kurtosis_runs_and_shapes(step_qxqyqz):
    # Skip cleanly if optional deps are missing in this env
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    pytest.importorskip("tabulate")
    pytest.importorskip("matplotlib")

    from cdi_dislo.statistics import statdiff_handler as s

    data = _make_3d_diffraction_like(shape=(20, 20, 20), seed=0, poisson=True)

    fwhm_xyz, fwhm_int_xyz, skew_xyz, kurt_xyz = s.get_plot_fwhm_and_skewness_kurtosis(
        data,
        plot=False,
        plot_show=False,
        step_qxqyqz=step_qxqyqz,
    )

    # returned as arrays/lists length 3
    assert len(fwhm_xyz) == 3
    assert len(fwhm_int_xyz) == 3
    assert len(skew_xyz) == 3
    assert len(kurt_xyz) == 3

    # should be finite numbers (can be 0 if fit fails, but should not be nan/inf)
    for arr in (fwhm_xyz, fwhm_int_xyz, skew_xyz, kurt_xyz):
        a = np.asarray(arr, dtype=float)
        assert np.all(np.isfinite(a))


def test_get_plot_fwhm_and_skewness_kurtosis_center_peak_does_not_crash():
    pytest.importorskip("scipy")
    pytest.importorskip("sklearn")
    pytest.importorskip("tabulate")
    pytest.importorskip("matplotlib")

    from cdi_dislo.statistics import statdiff_handler as s

    data = _make_3d_diffraction_like(shape=(20, 20, 20), seed=0, poisson=True)

    out = s.get_plot_fwhm_and_skewness_kurtosis(
        data,
        plot=False,
        plot_show=False,
        center_peak=True,
        eliminate_linear_background=False,
    )
    assert out is not None
