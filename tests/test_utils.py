# tests/test_utils.py
import numpy as np
import pytest

from cdi_dislo.utils import utils as u


def test_center_angles_symmetry_basic():
    angles = np.array([10.0, 20.0, 30.0])
    centered = u.center_angles(angles)
    # centered should be shifted so min becomes negative half-range
    assert np.isclose(np.nanmin(centered), -np.nanmax(centered))
    assert np.isclose(np.nanmean(centered), 0.0)


def test_normalize_vector_unit():
    v = np.array([3.0, 4.0, 0.0])
    vn = u.normalize_vector(v)
    assert np.isclose(np.linalg.norm(vn), 1.0)


def test_normalize_vector_zero_safe():
    v = np.array([0.0, 0.0, 0.0])
    vn = u.normalize_vector(v)
    assert np.allclose(vn, v)  # current behavior: returns v unchanged


def test_project_vector_perpendicular_component():
    v = np.array([1.0, 2.0, 3.0])
    t = np.array([0.0, 0.0, 1.0])
    vp = u.project_vector(v, t)
    # projected vector should have no component along t
    assert np.isclose(np.dot(vp, t), 0.0)


def test_nan_to_zero():
    arr = np.array([np.nan, 1.0, -2.0])
    out = u.nan_to_zero(arr)
    assert np.array_equal(out, np.array([0.0, 1.0, -2.0]))


def test_zero_to_nan_default():
    arr = np.array([0.0, 1.0, 0.0, 2.0])
    out = u.zero_to_nan(arr)
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == 1.0 and out[3] == 2.0


def test_zero_to_nan_boolean_values():
    arr = np.array([0.0, 5.0, 0.0])
    out = u.zero_to_nan(arr, boolean_values=True)
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == 1


@pytest.mark.parametrize(
    "number, expected",
    [
        (0.0, (0, 0)),
        (1234.0, (1.234, 3)),
        # with exponent = int(log10(|x|)) -> int(-1.91...) = -1
        # coefficient = -0.0123 / 10**(-1) = -0.123
        (-0.0123, (-0.123, -1)),
    ],
)
def test_extract_coefficient_and_exponent(number, expected):
    c, e = u.extract_coefficient_and_exponent(number)
    ec, ee = expected
    assert e == ee
    assert np.isclose(c, ec, rtol=1e-12, atol=1e-12)


def test_generate_burgers_directions_contains_opposites_and_primitive():
    out = u.generate_burgers_directions(m=1, G=[1, 1, 0], hkl_max=2, sort_by_hkl=True)

    for vec, ang in out:
        # since G=[1,1,0], dot = h+k, and returned set contains both signs
        assert abs(vec[0] + vec[1]) == 2

        # primitive check: gcd of non-zero components is 1
        nz = [abs(x) for x in vec if x != 0]
        g = nz[0]
        for x in nz[1:]:
            g = np.gcd(g, x)
        assert g == 1

        assert 0.0 <= ang <= 180.0

    # verify opposites are present
    vecs = {tuple(v) for v, _ in out}
    for v in list(vecs):
        assert (-v[0], -v[1], -v[2]) in vecs


def test_transform_miller_indices_basic():
    miller = ["111", "1-10", "-101"]
    out = u.transform_miller_indices(miller)
    assert out.shape == (3, 3)
    assert np.array_equal(out[0], np.array([1.0, 1.0, 1.0]))
    assert np.array_equal(out[1], np.array([1.0, -1.0, 0.0]))
    assert np.array_equal(out[2], np.array([-1.0, 0.0, 1.0]))


def test_generate_burgers_directions_m0_returns_empty():
    out = u.generate_burgers_directions(m=0, G=[1, -1, 1])
    assert out == []


def test_find_max_and_com_3d_simple_peak():
    data = np.zeros((9, 9, 9), dtype=float)
    data[4, 5, 6] = 10.0
    max_pos, com_pos = u.find_max_and_com_3d(data, window_size=5)
    assert max_pos == (4, 5, 6)
    assert com_pos == (4, 5, 6)


def test_format_as_4digit_string():
    assert u.format_as_4digit_string(7) == "0007"
    assert u.format_as_4digit_string(123) == "0123"


@pytest.mark.parametrize("s, expected", [("3.14", True), ("-2", True), ("abc", False)])
def test_isfloat(s, expected):
    assert u.isfloat(s) is expected


def test_check_array_empty():
    assert u.check_array_empty([]) is True
    assert u.check_array_empty(np.array([])) is True
    assert u.check_array_empty(np.array([1])) is False


def test_array_to_dict_plain_array():
    arr = np.array([10, 20, 30])
    d = u.array_to_dict(arr)
    assert d == {0: 10, 1: 20, 2: 30}


def test_check_type_variants():
    assert u.check_type(3) == "int"
    assert u.check_type(3.2) == "float"
    assert u.check_type("x") == "str"
    assert u.check_type([1, 2, 3]) == "list of int"
    assert u.check_type((1.0, 2.0)) == "tuple of float"
    assert u.check_type(np.array([True, False])) == "array of bool"


def test_normalize_methods_mixed():
    methods = ("max", [128, 128, 128], np.array([1.0, 2.0, 3.0]))
    out = u.normalize_methods(methods)
    assert out[0] == "max"
    assert out[1] == (128, 128, 128)
    assert out[2] == (1.0, 2.0, 3.0)


def test_multiply_list_elements():
    assert u.multiply_list_elements([2, 3, 4]) == 24


def test_get_numbers_from_string():
    assert u.get_numbers_from_string("Run0123_LLk") == "0123"
    assert u.get_numbers_from_string("no numbers") == ""


def test_std_data_nonzero_only():
    data = np.array([0, 1, 2, 3, 0], dtype=float)
    out = u.std_data(data)
    # std of [1,2,3] is sqrt((( -1)^2 + 0^2 + 1^2)/3) = sqrt(2/3)
    assert np.isclose(out, np.sqrt(2.0 / 3.0))


def test_build_rotation_matrix_from_axes_identity():
    Rm = u.build_rotation_matrix_from_axes(0, 0, 0)
    assert np.allclose(Rm, np.eye(3), atol=1e-12)


def test_alignment_euler_angles_aligns_vectors():
    src = np.array([1.0, 0.0, 0.0])
    tgt = np.array([0.0, 0.0, 1.0])
    angles, Rm = u.alignment_euler_angles(src, tgt, order="xyz", degrees=True)
    out = Rm @ (src / np.linalg.norm(src))
    assert np.allclose(out, tgt, atol=1e-7)
    assert len(angles) == 3


def test_centered_affine_transform_identity():
    data = np.zeros((11, 11, 11), dtype=float)
    data[5, 5, 5] = 1.0
    out = u.centered_affine_transform(data, np.eye(3), order=1)
    assert np.allclose(out, data)


def test_rotation_matrix_from_angles_identity():
    Rm = u.rotation_matrix_from_angles(0, 0, 0)
    assert np.allclose(Rm, np.eye(3), atol=1e-12)


def test_get_largest_component():
    mask = np.zeros((10, 10, 10), dtype=int)
    mask[1:3, 1:3, 1:3] = 1  # size 8
    mask[6:10, 6:10, 6:10] = 1  # size 64 (largest)
    largest, lbl, size = u.get_largest_component(mask)
    assert size == 64
    assert largest.sum() == 64


def test_pad_to_shape_centering():
    arr = np.ones((3, 3, 3), dtype=int)
    out = u.pad_to_shape(arr, target_shape=(5, 5, 5), pad_value=0)
    assert out.shape == (5, 5, 5)
    assert out.sum() == arr.sum()  # padding adds zeros only


def test_optimize_cropping_min_sym_crop():
    data = np.zeros((10, 10, 10), dtype=int)
    data[2:6, 3:9, 1:4] = 1
    w = u.optimize_cropping(data, min_sym_crop=True)
    # widths: z=4, y=6, x=3 => max is 6
    assert w == 6


def test_get_size_3D_pixel():
    data = np.zeros((10, 10, 10), dtype=int)
    data[2:6, 3:9, 1:4] = 1
    w = u.get_size_3D_pixel(data)
    assert w == (4, 6, 3)


def test_mask_clusters_returns_masked_and_remaining():
    data = np.zeros((30, 30, 30), dtype=float)
    data[5, 5, 5] = 10.0  # strong peak cluster
    data[25, 25, 25] = 7.0  # second peak
    masked, remaining, dil = u.mask_clusters(
        data, threshold_factor=0.5, dilation_iterations=2
    )
    assert masked.shape == data.shape
    assert remaining.shape == data.shape
    assert dil.shape == data.shape
    # after masking first cluster, remaining should still contain the second peak
    assert remaining[25, 25, 25] > 0


def test_format_vector():
    vec = [0.00000012, 1.234567, 10000.0]
    s = u.format_vector(vec, decimal_places=4)
    assert isinstance(s, str)
    assert "," in s


def test_fit_model_regression_linear_optional():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 2.0, 4.0, 6.0])
    _, pred = u.fit_model_regression("LinearRegression", X, y)
    assert pred.shape == y.shape
    assert np.allclose(pred, y, atol=1e-8)
