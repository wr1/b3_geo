from __future__ import annotations

import numpy as np
import pytest

from b3_aerofoil.geometry.naca import naca4
from b3_blade.airfoil_stack import AirfoilStack
from b3_blade.curve import Curve
from b3_blade.planform import Planform
from b3_blade.web import PlaneWeb, RibbonWeb
from b3_geo import realise


def _make_planform(span: float = 63.0) -> Planform:
    return Planform(
        span=span,
        z=Curve.from_points([(0.0, 3.0), (1.0, span + 3.0)]),
        chord=Curve.from_points([(0.0, 4.0), (0.3, 5.0), (1.0, 1.0)]),
        tc=Curve.from_points([(0.0, 0.5), (0.2, 0.3), (1.0, 0.15)]),
        twist=Curve.from_points([(0.0, 10.0), (0.5, 3.0), (1.0, 0.0)]),
        dx=Curve.from_points([(0.0, 0.0), (1.0, 0.0)]),
        dy=Curve.from_points([(0.0, 0.0), (1.0, 0.0)]),
    )


def _make_stack() -> AirfoilStack:
    n18 = naca4("0018")
    n18.metadata["thickness"] = 0.18
    n15 = naca4("0015")
    n15.metadata["thickness"] = 0.15
    thick = naca4("0030")
    thick.metadata["thickness"] = 0.30
    return AirfoilStack([(0.0, thick), (0.5, n18), (1.0, n15)])


def test_realise_sections_shape():
    pf = _make_planform()
    stack = _make_stack()
    spans = np.linspace(0, 1, 10)
    rb = realise(pf, [], stack, spans, npchord=50)
    assert rb.sections.shape == (10, 50, 3)


def test_realise_spans_abs():
    pf = _make_planform(63.0)
    stack = _make_stack()
    spans = np.array([0.0, 1.0])
    rb = realise(pf, [], stack, spans)
    assert abs(rb.spans_abs[0] - 3.0) < 1e-6
    assert abs(rb.spans_abs[1] - 66.0) < 1e-6


def test_realise_with_plane_web():
    pf = _make_planform()
    stack = _make_stack()
    webs = [PlaneWeb(name="web0", origin=(0.0, 0.0, 0.0), orientation=(0.02, 1.0, 0.0))]
    spans = np.linspace(0, 1, 5)
    rb = realise(pf, webs, stack, spans)
    assert len(rb.webs) == 1
    assert rb.webs[0].name == "web0"


def test_realise_with_ribbon_web():
    pf = _make_planform()
    stack = _make_stack()
    webs = [
        PlaneWeb(name="spar", origin=(0.0, 0.0, 0.0), orientation=(0.02, 1.0, 0.0)),
        RibbonWeb(
            name="te_reinf",
            reference_web="spar",
            offsets=Curve.from_points([(0.0, 0.3), (1.0, 0.3)]),
        ),
    ]
    spans = np.linspace(0, 1, 5)
    rb = realise(pf, webs, stack, spans)
    assert len(rb.webs) == 2


def test_realise_fields_columns():
    pf = _make_planform()
    stack = _make_stack()
    spans = np.linspace(0, 1, 5)
    rb = realise(pf, [], stack, spans, npchord=20)
    expected = {"s", "z", "chord_frac", "panel_id", "dist_from_le", "dist_from_te"}
    assert expected.issubset(rb.fields.columns)


def test_realise_fields_size():
    pf = _make_planform()
    stack = _make_stack()
    n_spans, n_chord = 8, 20
    spans = np.linspace(0, 1, n_spans)
    rb = realise(pf, [], stack, spans, npchord=n_chord)
    assert len(rb.fields) == n_spans * n_chord
