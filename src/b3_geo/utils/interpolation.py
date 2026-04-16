from __future__ import annotations

import numpy as np
import pyvista as pv
from pathlib import Path
from scipy.interpolate import CubicSpline, PchipInterpolator


def load_airfoil(path: str | Path, base_dir: Path | None = None) -> np.ndarray:
    """Load airfoil data from file."""
    full_path = (
        Path(base_dir or ".").absolute() / path if base_dir else Path(path).absolute()
    )
    data = np.loadtxt(full_path, skiprows=1)
    return data[:, [1, 0]]  # Switch x and y axes


def airfoil_from_coordinates(coordinates: list[list[float]]) -> np.ndarray:
    """Build airfoil array from inline [[x, y], ...] coordinates (no header, no swap)."""
    data = np.array(coordinates, dtype=float)
    return data[:, [1, 0]]  # same y/x swap convention as load_airfoil


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


def linear_interpolate(points: list[tuple[float, float]], x: np.ndarray) -> np.ndarray:
    """Linear interpolation at given x values."""
    points = sorted(points)
    xs, ys = zip(*points)
    return np.interp(x, xs, ys)


def cubic_interpolate(
    points: list[tuple[float, float]], x: np.ndarray, bc_type: str = "clamped"
) -> np.ndarray:
    """Cubic spline interpolation at given x values."""
    points = sorted(points)
    xs, ys = zip(*points)
    spline = CubicSpline(xs, ys, bc_type=bc_type)
    return spline(x)


def pchip_interpolate(points: list[tuple[float, float]], x: np.ndarray) -> np.ndarray:
    """PCHIP interpolation at given x values."""
    points = sorted(points)
    xs, ys = zip(*points)
    interpolator = PchipInterpolator(xs, ys)
    return interpolator(x)


def default_handles(
    points: list[tuple[float, float]],
) -> list[tuple[float, float, float, float]]:
    """Compute C1-matching Bezier handles from PCHIP derivatives at anchor points.

    Returns list of (in_dx, in_dy, out_dx, out_dy) offsets from each anchor.
    The resulting Bezier segments reproduce the PCHIP curve shape exactly at switch time.
    """
    pts = sorted(points)
    xs, ys = zip(*pts)
    pchip = PchipInterpolator(xs, ys)
    derivs = pchip.derivative()(xs)
    n = len(pts)
    result = []
    for k in range(n):
        in_dx, in_dy = 0.0, 0.0
        out_dx, out_dy = 0.0, 0.0
        if k > 0:
            h = pts[k][0] - pts[k - 1][0]
            in_dx = -h / 3.0
            in_dy = in_dx * float(derivs[k])
        if k < n - 1:
            h = pts[k + 1][0] - pts[k][0]
            out_dx = h / 3.0
            out_dy = out_dx * float(derivs[k])
        result.append((in_dx, in_dy, out_dx, out_dy))
    return result


def bezier_interpolate(curve, x: np.ndarray) -> np.ndarray:
    """Evaluate piecewise cubic Bezier curve at x values.

    curve may be a bezier_curve model or a plain dict with 'points' and 'handles' keys.
    handles[i] = (in_dx, in_dy, out_dx, out_dy) — offsets from anchor i.
    If handles is empty, C1-matching defaults are computed from PCHIP derivatives.
    """
    if isinstance(curve, dict):
        pts_raw = curve.get("points", [])
        handles_raw = list(curve.get("handles", []))
    else:
        pts_raw = curve.points
        handles_raw = list(curve.handles) if curve.handles else []

    pts = sorted(pts_raw)
    n = len(pts)
    if n < 2:
        return np.full(len(x), pts[0][1] if pts else 0.0)

    anchors = np.array(pts, dtype=float)
    if len(handles_raw) < n:
        handles_raw = default_handles(pts)
    handles = np.array(handles_raw, dtype=float)  # (n, 4): in_dx, in_dy, out_dx, out_dy

    result = np.empty(len(x))
    for i, xi in enumerate(x):
        if xi <= anchors[0, 0]:
            result[i] = anchors[0, 1]
            continue
        if xi >= anchors[-1, 0]:
            result[i] = anchors[-1, 1]
            continue

        seg = int(np.searchsorted(anchors[:, 0], xi)) - 1
        seg = max(0, min(seg, n - 2))

        p0 = anchors[seg]
        p3 = anchors[seg + 1]
        h0 = handles[seg]
        h3 = handles[seg + 1]
        p1 = np.array([p0[0] + h0[2], p0[1] + h0[3]])  # out handle of seg
        p2 = np.array([p3[0] + h3[0], p3[1] + h3[1]])  # in handle of seg+1

        # Newton solve: B_x(t) = xi
        span = max(p3[0] - p0[0], 1e-10)
        t = float(np.clip((xi - p0[0]) / span, 0.0, 1.0))
        for _ in range(20):
            bx = (
                (1 - t) ** 3 * p0[0]
                + 3 * (1 - t) ** 2 * t * p1[0]
                + 3 * (1 - t) * t**2 * p2[0]
                + t**3 * p3[0]
            )
            dbx = (
                3 * (1 - t) ** 2 * (p1[0] - p0[0])
                + 6 * (1 - t) * t * (p2[0] - p1[0])
                + 3 * t**2 * (p3[0] - p2[0])
            )
            f = bx - xi
            if abs(f) < 1e-10:
                break
            if abs(dbx) < 1e-14:
                break
            t = float(np.clip(t - f / dbx, 0.0, 1.0))

        result[i] = (
            (1 - t) ** 3 * p0[1]
            + 3 * (1 - t) ** 2 * t * p1[1]
            + 3 * (1 - t) * t**2 * p2[1]
            + t**3 * p3[1]
        )
    return result


def eval_curve(
    curve_or_data, x: np.ndarray, default_method: str = "pchip"
) -> np.ndarray:
    """Evaluate a planform curve at x values, dispatching by curve type.

    Accepts:
    - plain list/tuple of (x, y) pairs → uses default_method
    - dict with 'type' key → dispatches to correct interpolator
    - pchip_curve / bezier_curve Pydantic model → same dispatch

    default_method: 'pchip' | 'linear' | 'cubic_natural' | 'cubic'
    Only used when curve_or_data is a plain list (backward compat).
    """
    if isinstance(curve_or_data, (list, tuple)):
        if default_method == "linear":
            return linear_interpolate(curve_or_data, x)
        elif default_method == "cubic_natural":
            return cubic_interpolate(curve_or_data, x, bc_type="natural")
        elif default_method == "cubic":
            return cubic_interpolate(curve_or_data, x)
        return pchip_interpolate(curve_or_data, x)

    if isinstance(curve_or_data, dict):
        curve_type = curve_or_data.get("type", "pchip")
        points = curve_or_data.get("points", [])
    else:
        curve_type = curve_or_data.type
        points = curve_or_data.points

    if curve_type == "bezier":
        return bezier_interpolate(curve_or_data, x)
    elif curve_type == "linear":
        return linear_interpolate(points, x)
    elif curve_type == "cubic_natural":
        return cubic_interpolate(points, x, bc_type="natural")
    elif curve_type == "cubic":
        return cubic_interpolate(points, x)
    return pchip_interpolate(points, x)


def build_sections_poly(
    points: np.ndarray, np_chordwise: int, np_spanwise: int
) -> pv.StructuredGrid:
    """Build structured grid for blade sections."""
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (np_chordwise, np_spanwise, 1)
    return grid
