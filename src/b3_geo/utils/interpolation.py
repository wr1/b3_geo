import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from typing import List, Tuple
import pyvista as pv


def load_airfoil(path: str) -> np.ndarray:
    """Load airfoil data from file."""
    return np.loadtxt(path)


def interpolate_airfoil(data: np.ndarray, n_points: int) -> np.ndarray:
    """Interpolate airfoil to a fixed number of points using cubic spline on arc length."""
    diffs = np.diff(data, axis=0)
    dist = np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))
    dist = np.insert(dist, 0, 0)
    total_length = dist[-1]
    new_s = np.linspace(0, total_length, n_points)
    x_spl = CubicSpline(dist, data[:, 0])
    y_spl = CubicSpline(dist, data[:, 1])
    return np.column_stack((x_spl(new_s), y_spl(new_s)))


def linear_interpolate(points: List[Tuple[float, float]], x: np.ndarray) -> np.ndarray:
    """Linear interpolation at given x values."""
    points = sorted(points)
    xs, ys = zip(*points)
    return np.interp(x, xs, ys)


def cubic_interpolate(
    points: List[Tuple[float, float]], x: np.ndarray, bc_type: str = "clamped"
) -> np.ndarray:
    """Cubic spline interpolation at given x values."""
    points = sorted(points)
    xs, ys = zip(*points)
    spline = CubicSpline(xs, ys, bc_type=bc_type)
    return spline(x)


def pchip_interpolate(points: List[Tuple[float, float]], x: np.ndarray) -> np.ndarray:
    """PCHIP interpolation at given x values."""
    points = sorted(points)
    xs, ys = zip(*points)
    interpolator = PchipInterpolator(xs, ys)
    return interpolator(x)


def build_sections_poly(
    points: np.ndarray, np_chordwise: int, np_spanwise: int
) -> pv.StructuredGrid:
    """Build structured grid for blade sections."""
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (np_chordwise, np_spanwise, 1)
    return grid
