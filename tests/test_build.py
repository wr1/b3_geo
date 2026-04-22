"""Tests for b3_geo.build — build_w_vtk and get_wire."""

from __future__ import annotations

import numpy as np

from b3_aerofoil.geometry.naca import naca4
from b3_blade.airfoil_stack import AirfoilStack
from b3_blade.curve import Curve
from b3_blade.planform import Planform
from b3_geo.build import build_w_vtk, get_wire
from b3_geo.schema import validate_w_vtk


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


def test_validate_w_vtk_passes():
    w = build_w_vtk(_planform(), _stack(), n_sections=10, n_chord=50)
    validate_w_vtk(w)  # must not raise


def test_point_count():
    w = build_w_vtk(_planform(), _stack(), n_sections=10, n_chord=50)
    # each wire has n_chord+1 points (closed duplicate)
    assert w.n_points == 10 * 51


def test_n_polylines():
    w = build_w_vtk(_planform(), _stack(), n_sections=10, n_chord=50)
    # count polylines from cell array
    lines = w.lines
    n = 0
    i = 0
    while i < len(lines):
        n += 1
        i += int(lines[i]) + 1
    assert n == 10


def test_field_data_length():
    w = build_w_vtk(_planform(), _stack(), n_sections=10, n_chord=50)
    for key in ("geo.s", "geo.z", "geo.chord", "geo.twist", "geo.tc", "geo.dx", "geo.dy"):
        assert w.field_data[key].shape == (10,), f"{key} field_data length != 10"


def test_field_data_s_monotonic():
    w = build_w_vtk(_planform(), _stack(), n_sections=10, n_chord=50)
    s = w.field_data["geo.s"]
    assert np.all(np.diff(s) > 0)


def test_custom_sections_array():
    s_vals = np.array([0.1, 0.5, 0.9])
    w = build_w_vtk(_planform(), _stack(), n_chord=30, sections=s_vals)
    validate_w_vtk(w)
    assert w.field_data["geo.s"].shape == (3,)


def test_get_wire():
    pf = _planform()
    st = _stack()
    w = get_wire(pf, st, s=0.5, n_chord=40)
    assert w.n_points == 41
    assert "geo.abs_t" in w.point_data
