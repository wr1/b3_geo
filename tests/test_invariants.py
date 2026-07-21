"""Geometric invariants on W vtk output."""

from __future__ import annotations

import numpy as np

from b3_aerofoil.geometry.naca import naca4
from b3_blade.airfoil_stack import AirfoilStack
from b3_blade.curve import Curve
from b3_blade.planform import Planform
from b3_geo.build import build_w_vtk


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


def _build(n_sections: int = 8, n_chord: int = 60) -> object:
    return build_w_vtk(_planform(), _stack(), n_sections=n_sections, n_chord=n_chord)


def test_uacs_col0_in_unit_interval():
    w = _build()
    uacs = w.point_data["geo.uacs"]
    assert uacs[:, 0].min() >= -1e-9
    assert uacs[:, 0].max() <= 1.0 + 1e-9


def test_sdacs_col0_bounded_by_chord():
    """Per-section |sdacs[:,0]| ≤ chord/2."""
    pf = _planform()
    n_chord = 60
    n_sections = 8
    w = build_w_vtk(pf, _stack(), n_sections=n_sections, n_chord=n_chord)
    s_vals = w.field_data["geo.s"]
    chords = w.field_data["geo.chord"]
    sdacs  = w.point_data["geo.sdacs"]

    pts_per_wire = n_chord + 1  # closed duplicate
    for i in range(n_sections):
        seg = sdacs[i * pts_per_wire : (i + 1) * pts_per_wire, 0]
        half_chord = chords[i] / 2.0
        assert np.abs(seg).max() <= half_chord + 1e-9, (
            f"section {i} (s={s_vals[i]:.3f}): |sdacs[:,0]| exceeds chord/2"
        )


def test_sdacs_mean_col0_near_zero():
    """Per-section mean of sdacs col 0 ≈ 0 (mid-chord centred)."""
    n_chord = 60
    n_sections = 8
    w = build_w_vtk(_planform(), _stack(), n_sections=n_sections, n_chord=n_chord)
    sdacs = w.point_data["geo.sdacs"]
    pts_per_wire = n_chord + 1
    for i in range(n_sections):
        seg = sdacs[i * pts_per_wire : (i + 1) * pts_per_wire, 0]
        assert abs(seg.mean()) < 0.05 * w.field_data["geo.chord"][i], (
            f"section {i}: sdacs mean col0 not near zero"
        )


def test_gbcs_z_constant_within_section():
    """All points in a wire share the same z."""
    n_chord = 40
    n_sections = 6
    w = build_w_vtk(_planform(), _stack(), n_sections=n_sections, n_chord=n_chord)
    pts_per_wire = n_chord + 1
    for i in range(n_sections):
        seg_z = w.points[i * pts_per_wire : (i + 1) * pts_per_wire, 2]
        assert seg_z.max() - seg_z.min() < 1e-9, f"section {i}: z not constant"


def test_gbcs_z_monotonic_across_sections():
    w = _build()
    z_vals = w.field_data["geo.z"]
    assert np.all(np.diff(z_vals) > 0)
