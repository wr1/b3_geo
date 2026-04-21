"""Tests for mesh hint resolution in b3_geo."""

from __future__ import annotations

import numpy as np
import pytest

from b3_blade.curve import Curve
from b3_blade.mesh_hints import (
    BoxMask, GridAlignment, HardpointHint, MeshHints,
    RefineHP, RefinementHint, RibbonMask,
)
from b3_blade.web import PlaneWeb, RibbonWeb
from b3_geo import realise


def _make_blade():
    from b3_blade.templates.blade_mini import BladeMini
    return BladeMini()


# ── realise() without hints ───────────────────────────────────────────────────

def test_realise_without_hints_has_empty_hint_fields():
    blade = _make_blade()
    spans = np.linspace(0.0, 1.0, 20)
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(), spans)
    assert rb.hardpoint_curves == []
    assert rb.refinement_regions == []


# ── realise() with hints ──────────────────────────────────────────────────────

def test_realise_with_blade_mesh_hints_adds_hardpoints():
    blade = _make_blade()
    blade.laminates()  # populate _helper_hints
    hints = blade.mesh_hints()
    spans = np.linspace(0.0, 1.0, 20)
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(),
                 spans, mesh_hints=hints)
    assert len(rb.hardpoint_curves) >= 2  # sparcap_te_hp + sparcap_le_hp
    names = [h.name for h in rb.hardpoint_curves]
    assert any("sparcap_te" in n for n in names)
    assert any("sparcap_le" in n for n in names)


def test_realise_hardpoint_arrays_correct_shape():
    blade = _make_blade()
    blade.laminates()
    hints = blade.mesh_hints()
    spans = np.linspace(0.0, 1.0, 25)
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(),
                 spans, mesh_hints=hints)
    for hp in rb.hardpoint_curves:
        assert hp.s_array.shape == (25,)
        assert hp.chord_frac_array.shape == (25,)
        assert np.all(hp.chord_frac_array >= 0.0)
        assert np.all(hp.chord_frac_array <= 1.0)


def test_realise_with_refinement_hint_from_laminates():
    blade = _make_blade()
    blade.laminates()
    hints = blade.mesh_hints()
    spans = np.linspace(0.0, 1.0, 20)
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(),
                 spans, mesh_hints=hints)
    assert len(rb.refinement_regions) >= 1
    names = [r.name for r in rb.refinement_regions]
    assert any("te_ud" in n for n in names)


def test_realise_refinement_factor_within_bounds():
    blade = _make_blade()
    blade.laminates()
    hints = blade.mesh_hints()
    spans = np.linspace(0.0, 1.0, 20)
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(),
                 spans, mesh_hints=hints)
    for rr in rb.refinement_regions:
        assert 1.0 < rr.factor <= 5.0


# ── explicit HardpointHint with Curve ────────────────────────────────────────

def test_hardpoint_hint_with_curve_resolved():
    blade = _make_blade()
    spans = np.linspace(0.0, 1.0, 20)
    explicit_curve = Curve.from_points([(0.0, 0.4), (1.0, 0.35)])
    hints = MeshHints(hardpoints=[
        HardpointHint(curve=explicit_curve, name="explicit_hp"),
    ])
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(),
                 spans, mesh_hints=hints)
    hp_names = [h.name for h in rb.hardpoint_curves]
    assert "explicit_hp" in hp_names
    hp = next(h for h in rb.hardpoint_curves if h.name == "explicit_hp")
    expected = explicit_curve.at(spans)
    np.testing.assert_allclose(hp.chord_frac_array, expected, atol=1e-10)


# ── BoxMask refinement resolved ───────────────────────────────────────────────

def test_box_mask_refinement_resolved():
    blade = _make_blade()
    spans = np.linspace(0.0, 1.0, 20)
    hints = MeshHints(refinements=[
        RefinementHint(
            region=BoxMask(span_range=(0.2, 0.8), chord_range=(0.0, 0.3)),
            factor=3.0,
            name="test_box",
        )
    ])
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(),
                 spans, mesh_hints=hints)
    assert len(rb.refinement_regions) == 1
    rr = rb.refinement_regions[0]
    assert rr.name == "test_box"
    assert rr.factor == 3.0
    assert rr.s_range == (0.2, 0.8)
    assert rr.chord_range == (0.0, 0.3)


# ── RibbonMask refinement resolved ───────────────────────────────────────────

def test_ribbon_mask_refinement_resolved():
    blade = _make_blade()
    spans = np.linspace(0.0, 1.0, 20)
    hints = MeshHints(refinements=[
        RefinementHint(
            region=RibbonMask(ribbon_name="sparcap_te", expand=0.05),
            factor=2.0,
            name="ribbon_ref",
        )
    ])
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(),
                 spans, mesh_hints=hints)
    assert len(rb.refinement_regions) == 1
    rr = rb.refinement_regions[0]
    assert rr.name == "ribbon_ref"
    # chord_range should expand around sparcap_te chord_frac values
    assert rr.chord_range[0] < rr.chord_range[1]


def test_ribbon_mask_unknown_web_skipped():
    blade = _make_blade()
    spans = np.linspace(0.0, 1.0, 20)
    hints = MeshHints(refinements=[
        RefinementHint(
            region=RibbonMask(ribbon_name="nonexistent_web", expand=0.05),
            factor=2.0, name="bad_ref",
        )
    ])
    rb = realise(blade.planform(), blade.structure(), blade.airfoil_stack(),
                 spans, mesh_hints=hints)
    # unknown web → hint silently skipped
    assert len(rb.refinement_regions) == 0
