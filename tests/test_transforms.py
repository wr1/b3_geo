"""Tests for b3_geo.transforms — coordinate frame math."""

from __future__ import annotations

import numpy as np
import pytest

from b3_geo.transforms import (
    gbcs_to_sdacs,
    sdacs_to_gbcs,
    sdacs_to_uacs,
    uacs_to_sdacs,
)


def _rect_uacs(n: int = 5) -> np.ndarray:
    """Rectangular airfoil: chord [0,1] in col0, thickness 0.1 in col1."""
    chord_frac = np.linspace(0.0, 1.0, n)
    thickness = np.full(n, 0.1)
    return np.column_stack([chord_frac, thickness])


def test_uacs_to_sdacs_x_range():
    uacs = _rect_uacs(11)
    sdacs = uacs_to_sdacs(uacs, chord=2.0)
    assert sdacs[:, 0].min() == pytest.approx(-1.0, abs=1e-12)
    assert sdacs[:, 0].max() == pytest.approx(1.0, abs=1e-12)


def test_uacs_to_sdacs_y_scaled():
    uacs = _rect_uacs(5)
    sdacs = uacs_to_sdacs(uacs, chord=2.0)
    np.testing.assert_allclose(sdacs[:, 1], uacs[:, 1] * 2.0, atol=1e-12)


def test_sdacs_to_gbcs_zero_twist_col_swap():
    """twist=0, dx=0, dy=0, z=10 → pure col-swap, col2=10."""
    sdacs = np.array([[1.0, 2.0], [3.0, 4.0]])
    gbcs = sdacs_to_gbcs(sdacs, twist=0.0, dx=0.0, dy=0.0, z=10.0)
    # col0 = flapwise = old thickness (col1)
    np.testing.assert_allclose(gbcs[:, 0], sdacs[:, 1], atol=1e-12)
    # col1 = edgewise = old chord (col0)
    np.testing.assert_allclose(gbcs[:, 1], sdacs[:, 0], atol=1e-12)
    # col2 = z
    np.testing.assert_allclose(gbcs[:, 2], 10.0, atol=1e-12)


def test_sdacs_to_gbcs_half_pi_twist():
    """twist=90°: hand-computed.

    SDACS point (c, t). After 90° rotation:
      c_rot = c*cos(90°) - t*sin(90°) = -t
      t_rot = c*sin(90°) + t*cos(90°) =  c
      flapwise (col0) = t_rot = c
      edgewise (col1) = c_rot = -t
    """
    sdacs = np.array([[1.0, 0.0], [0.0, 1.0]])
    gbcs = sdacs_to_gbcs(sdacs, twist=90.0, dx=0.0, dy=0.0, z=0.0)
    # row 0: c=1, t=0 → flapwise=1, edgewise=-0=0
    np.testing.assert_allclose(gbcs[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(gbcs[0, 1], 0.0, atol=1e-12)
    # row 1: c=0, t=1 → flapwise=0, edgewise=-1
    np.testing.assert_allclose(gbcs[1, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(gbcs[1, 1], -1.0, atol=1e-12)


def test_sdacs_to_gbcs_dx_shifts_col0():
    sdacs = np.array([[0.0, 0.0]])
    gbcs = sdacs_to_gbcs(sdacs, twist=0.0, dx=1.0, dy=0.0, z=0.0)
    np.testing.assert_allclose(gbcs[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(gbcs[0, 1], 0.0, atol=1e-12)


def test_sdacs_to_gbcs_dy_shifts_col1():
    sdacs = np.array([[0.0, 0.0]])
    gbcs = sdacs_to_gbcs(sdacs, twist=0.0, dx=0.0, dy=1.0, z=0.0)
    np.testing.assert_allclose(gbcs[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(gbcs[0, 1], 1.0, atol=1e-12)


def test_round_trip():
    """uacs → sdacs → gbcs → inv → uacs agrees to 1e-12."""
    rng = np.random.default_rng(42)
    uacs = np.column_stack([
        rng.uniform(0.0, 1.0, 20),
        rng.uniform(-0.1, 0.1, 20),
    ])
    chord = rng.uniform(1.0, 6.0)
    twist = rng.uniform(-15.0, 15.0)  # degrees
    dx = rng.uniform(-1.0, 1.0)
    dy = rng.uniform(-1.0, 1.0)
    z = rng.uniform(0.0, 60.0)

    sdacs = uacs_to_sdacs(uacs, chord)
    gbcs = sdacs_to_gbcs(sdacs, twist, dx, dy, z)
    sdacs2 = gbcs_to_sdacs(gbcs, twist, dx, dy)
    uacs2 = sdacs_to_uacs(sdacs2, chord)

    np.testing.assert_allclose(uacs2, uacs, atol=1e-12)
