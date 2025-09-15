import numpy as np
from b3_geo.utils.interpolation import (
    load_airfoil,
    interpolate_airfoil,
    linear_interpolate,
    cubic_interpolate,
    pchip_interpolate,
)


def test_load_airfoil(tmp_path):
    """Test loading airfoil data from file."""
    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("# header\n0.0 0.0\n0.5 0.1\n1.0 0.0\n")
    data = load_airfoil(str(airfoil_file))
    assert data.shape == (3, 2)
    assert data[0, 0] == 0.0
    assert data[1, 1] == 0.1


def test_interpolate_airfoil():
    """Test interpolating airfoil to fixed number of points."""
    data = np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]])
    interpolated = interpolate_airfoil(data, 5)
    assert interpolated.shape == (5, 2)
    assert interpolated[0, 0] == 0.0
    assert np.isclose(interpolated[-1, 0], 1.0)


def test_linear_interpolate():
    """Test linear interpolation."""
    points = [(0.0, 0.0), (1.0, 1.0)]
    x = np.array([0.5])
    result = linear_interpolate(points, x)
    assert result[0] == 0.5


def test_cubic_interpolate():
    """Test cubic spline interpolation."""
    points = [(0.0, 0.0), (1.0, 1.0)]
    x = np.array([0.5])
    result = cubic_interpolate(points, x)
    assert np.isclose(result[0], 0.5)


def test_pchip_interpolate():
    """Test PCHIP interpolation."""
    points = [(0.0, 0.0), (1.0, 1.0)]
    x = np.array([0.5])
    result = pchip_interpolate(points, x)
    assert np.isclose(result[0], 0.5)
