# tests/test_rotation.py
import numpy as np
import pytest

from cdi_dislo.rotation import rotation as rot


def _rand_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = np.linalg.norm(v)
    if n == 0:
        return _rand_unit(rng)
    return v / n


def _is_rotation_matrix(R: np.ndarray, atol: float = 1e-8) -> bool:
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3):
        return False
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, atol=atol):
        return False
    return True


def test_normalize_rotation_matrix_properties():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    R = rot.normalize_rotation_matrix(A)
    assert _is_rotation_matrix(R, atol=1e-7)


@pytest.mark.parametrize("method", ["svd", "qr"])
def test_orthogonalize_basis_returns_orthonormal(method):
    rng = np.random.default_rng(1)
    B = rng.normal(size=(3, 3))
    Q = rot.orthogonalize_basis(B, method=method)
    assert Q.shape == (3, 3)
    assert np.allclose(Q @ Q.T, np.eye(3), atol=1e-7)
    # det can be -1 depending on input; enforce |det|=1
    assert np.isclose(abs(np.linalg.det(Q)), 1.0, atol=1e-7)


def test_compute_rotation_matrix_aligns_vectors_random():
    rng = np.random.default_rng(2)
    for _ in range(50):
        v1 = _rand_unit(rng)
        v2 = _rand_unit(rng)
        Rm = rot.compute_rotation_matrix(v1, v2)
        assert Rm.shape == (3, 3)
        # Should map v1 -> v2 (within tolerance)
        v1r = Rm @ v1
        assert np.allclose(v1r, v2, atol=1e-6) or np.allclose(v1r, v2, atol=1e-6)
        # Rotation matrix validity (allow small numerical error)
        assert np.allclose(Rm.T @ Rm, np.eye(3), atol=1e-7)
        assert np.isclose(np.linalg.det(Rm), 1.0, atol=1e-7)


def test_compute_rotation_matrix_parallel_returns_identity():
    v = np.array([1.0, 2.0, -3.0])
    Rm = rot.compute_rotation_matrix(v, v)
    assert np.allclose(Rm, np.eye(3), atol=1e-8)


def test_compute_rotation_matrix_antiparallel_maps_to_negative():
    v1 = np.array([0.0, 0.0, 1.0])
    v2 = np.array([0.0, 0.0, -1.0])
    Rm = rot.compute_rotation_matrix(v1, v2)
    assert np.allclose(Rm @ (v1 / np.linalg.norm(v1)), v2, atol=1e-7)
    assert np.allclose(Rm.T @ Rm, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(Rm), 1.0, atol=1e-7)


def test_rotation_matrix_to_angles_roundtrip_xyz():
    # Use scipy Rotation for a known matrix and compare recovered angles
    angles = np.array([10.0, -20.0, 30.0])  # degrees
    from scipy.spatial.transform import Rotation as R

    Rm = R.from_euler("xyz", angles, degrees=True).as_matrix()
    out = rot.rotation_matrix_to_angles(Rm, order="xyz", degrees=True)
    assert out.shape == (3,)
    # Euler angles are not unique; compare reconstructed matrix
    Rm2 = R.from_euler("xyz", out, degrees=True).as_matrix()
    assert np.allclose(Rm2, Rm, atol=1e-8)


def test_angle_between_vectors_degrees_default():
    # Your function returns degrees by default (as you requested)
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    ang = rot.angle_between_vectors(a, b)
    assert np.isclose(ang, 90.0, atol=1e-12)


def test_cart2pol_simple_cases():
    phi, rho = rot.cart2pol(1.0, 0.0)
    assert np.isclose(rho, 1.0, atol=1e-12)
    assert np.isclose(phi, 0.0, atol=1e-12)

    phi, rho = rot.cart2pol(0.0, 1.0)
    assert np.isclose(rho, 1.0, atol=1e-12)
    assert np.isclose(phi, np.pi / 2, atol=1e-12)


def test_cartesian_to_cylind_simple_cases():
    theta, z, r = rot.cartesian_to_cylind(1.0, 0.0, 5.0)
    assert np.isclose(r, 1.0, atol=1e-12)
    assert np.isclose(theta, 0.0, atol=1e-12)
    assert np.isclose(z, 5.0, atol=1e-12)

    theta, z, r = rot.cartesian_to_cylind(0.0, 1.0, -2.0)
    assert np.isclose(r, 1.0, atol=1e-12)
    assert np.isclose(theta, np.pi / 2, atol=1e-12)
    assert np.isclose(z, -2.0, atol=1e-12)


def test_apply_rotation_to_data_identity_returns_same():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(10, 11, 12))
    out = rot.apply_rotation_to_data(data, np.eye(3), padding_factor=1.2)
    assert out.shape == data.shape
    # Interpolation + padding/crop can introduce tiny error; identity should be very close
    assert np.allclose(out, data, atol=1e-10)


def test_transform_coordinates_to_crystallographic_identity():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([3.0, 4.0, 5.0])
    z = np.array([6.0, 7.0, 8.0])
    Rm = np.eye(3)
    xc, yc, zc = rot.transform_coordinates_to_crystallographic(x, y, z, Rm)
    assert np.allclose(xc, x)
    assert np.allclose(yc, y)
    assert np.allclose(zc, z)


def test_transform_known_vector_to_crystallographic_identity():
    vx, vy, vz = 1.0, 2.0, 3.0
    Rm = np.eye(3)
    vxc, vyc, vzc = rot.transform_known_vector_to_crystallographic(vx, vy, vz, Rm)
    assert np.isclose(vxc, vx)
    assert np.isclose(vyc, vy)
    assert np.isclose(vzc, vz)


def test_normalize_vectors_3d_handles_zeros():
    vx = np.array([0.0, 3.0])
    vy = np.array([0.0, 4.0])
    vz = np.array([0.0, 0.0])
    vxn, vyn, vzn = rot.normalize_vectors_3d(vx, vy, vz)
    # First vector is zero -> should remain zero (no NaN)
    assert np.all(np.isfinite(vxn))
    assert np.all(np.isfinite(vyn))
    assert np.all(np.isfinite(vzn))
    assert np.isclose(vxn[0], 0.0)
    assert np.isclose(vyn[0], 0.0)
    assert np.isclose(vzn[0], 0.0)
    # Second vector normalized 3-4-0
    assert np.isclose(vxn[1], 3.0 / 5.0, atol=1e-12)
    assert np.isclose(vyn[1], 4.0 / 5.0, atol=1e-12)
    assert np.isclose(vzn[1], 0.0, atol=1e-12)
