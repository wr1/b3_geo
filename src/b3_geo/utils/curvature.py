"""Signed curvature and surface normals for airfoil contours."""

from __future__ import annotations

import numpy as np


def arc_length(xy: np.ndarray) -> np.ndarray:
    """Cumulative arc length (N,) along a (N, 2) contour, starting at 0."""
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.insert(np.cumsum(seg), 0, 0.0)


def curvature(xy: np.ndarray) -> np.ndarray:
    """Signed curvature kappa(s) at each point of a (N, 2) contour.

    Uses arc-length parameterisation with central finite differences.
    kappa = x'y'' - y'x'' (positive when the curve turns left of the
    travel direction).
    """
    s = arc_length(xy)
    dx = np.gradient(xy[:, 0], s)
    dy = np.gradient(xy[:, 1], s)
    d2x = np.gradient(dx, s)
    d2y = np.gradient(dy, s)
    return dx * d2y - dy * d2x


def surface_normals(xy: np.ndarray) -> np.ndarray:
    """Unit surface normals (N, 2), left of the travel direction."""
    s = arc_length(xy)
    dx = np.gradient(xy[:, 0], s)
    dy = np.gradient(xy[:, 1], s)
    n = np.column_stack((-dy, dx))
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.where(norm == 0, 1.0, norm)
