"""Tests for b3_geo.wire.build_wire."""

from __future__ import annotations

import numpy as np
import pytest

from b3_aerofoil.geometry.naca import naca4
from b3_blade.airfoil_stack import AirfoilStack
from b3_blade.curve import Curve
from b3_blade.planform import Planform
from b3_geo.wire import build_wire


def _planform() -> Planform:
    return Planform(
        span=63.0,
        z=Curve.from_points([(0.0, 3.0), (1.0, 66.0)]),
        chord=Curve.from_points([(0.0, 4.0), (0.3, 5.0), (1.0, 1.0)]),
        tc=Curve.from_points([(0.0, 0.30), (0.5, 0.18), (1.0, 0.15)]),
        twist=Curve.from_points([(0.0, 10.0), (0.5, 3.0), (1.0, 0.0)]),
        dx=Curve.from_points([(0.0, 0.0), (1.0, 0.5)]),
        dy=Curve.from_points([(0.0, 0.0), (1.0, 0.1)]),
    )


def _stack() -> AirfoilStack:
    n30 = naca4("0030"); n30.metadata["thickness"] = 0.30
    n18 = naca4("0018"); n18.metadata["thickness"] = 0.18
    n15 = naca4("0015"); n15.metadata["thickness"] = 0.15
    return AirfoilStack([(0.0, n30), (0.5, n18), (1.0, n15)])


def test_point_count():
    w = build_wire(_planform(), _stack(), s=0.5, n_chord=50)
    # closed: M+1 points
    assert w.n_points == 51


def test_closed_topology():
    w = build_wire(_planform(), _stack(), s=0.5, n_chord=50)
    first = w.points[0]
    last  = w.points[-1]
    np.testing.assert_allclose(first, last, atol=1e-12)


def test_point_data_fields_present():
    w = build_wire(_planform(), _stack(), s=0.5, n_chord=50)
    for key in ("geo.abs_t", "geo.rel_t", "geo.uacs", "geo.sdacs"):
        assert key in w.point_data, f"missing {key}"


def test_field_data_scalars_present():
    w = build_wire(_planform(), _stack(), s=0.5, n_chord=50)
    for key in ("geo.s", "geo.z", "geo.chord", "geo.twist", "geo.tc", "geo.dx", "geo.dy"):
        assert key in w.field_data, f"missing {key}"


def test_rel_t_monotonic_in_01():
    w = build_wire(_planform(), _stack(), s=0.5, n_chord=50)
    rel_t = w.point_data["geo.rel_t"][:-1]  # exclude closing dup
    assert rel_t[0] == pytest.approx(0.0, abs=1e-12)
    assert rel_t[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(rel_t) >= -1e-12)


def test_abs_t_starts_zero():
    w = build_wire(_planform(), _stack(), s=0.5, n_chord=50)
    assert w.point_data["geo.abs_t"][0] == pytest.approx(0.0, abs=1e-12)


def test_uacs_col0_in_01():
    w = build_wire(_planform(), _stack(), s=0.5, n_chord=50)
    uacs = w.point_data["geo.uacs"]
    assert uacs[:, 0].min() >= -1e-10
    assert uacs[:, 0].max() <= 1.0 + 1e-10


def test_sdacs_bounded_by_chord():
    pf = _planform()
    w  = build_wire(pf, _stack(), s=0.5, n_chord=50)
    chord = float(pf.chord.at(np.array([0.5]))[0])
    sdacs = w.point_data["geo.sdacs"]
    assert np.abs(sdacs[:, 0]).max() <= chord / 2.0 + 1e-10


def test_gbcs_z_matches_field_data():
    w = build_wire(_planform(), _stack(), s=0.5, n_chord=50)
    z_field = float(w.field_data["geo.z"][0])
    np.testing.assert_allclose(w.points[:, 2], z_field, atol=1e-10)
