import numpy as np

from b3_geo.utils.curvature import arc_length, curvature, surface_normals


def _circle(radius: float, n: int = 400, clockwise: bool = False) -> np.ndarray:
    theta = np.linspace(0, 2 * np.pi, n)
    if clockwise:
        theta = theta[::-1]
    return np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))


def test_arc_length_circle():
    """Total arc length of a circle matches its circumference."""
    xy = _circle(0.5)
    s = arc_length(xy)
    assert s[0] == 0.0
    assert np.isclose(s[-1], 2 * np.pi * 0.5, rtol=1e-3)


def test_curvature_circle_ccw():
    """CCW circle of radius R has kappa = +1/R."""
    xy = _circle(0.5)
    kappa = curvature(xy)
    assert np.allclose(kappa[5:-5], 2.0, rtol=1e-2)


def test_curvature_circle_cw():
    """CW circle of radius R has kappa = -1/R."""
    xy = _circle(2.0, clockwise=True)
    kappa = curvature(xy)
    assert np.allclose(kappa[5:-5], -0.5, rtol=1e-2)


def test_curvature_straight_line():
    """A straight line has zero curvature."""
    xy = np.column_stack((np.linspace(0, 1, 50), np.linspace(0, 0.5, 50)))
    assert np.allclose(curvature(xy), 0.0, atol=1e-10)


def test_surface_normals_circle():
    """Left-of-travel normals on a CCW circle point inward (toward center)."""
    xy = _circle(1.0)
    n = surface_normals(xy)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0, rtol=1e-6)
    assert np.allclose(n[5:-5], -xy[5:-5], atol=1e-2)
