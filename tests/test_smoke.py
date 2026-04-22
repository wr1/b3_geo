"""End-to-end smoke test: reference blade → W.vtp → round-trip validate."""

from __future__ import annotations

import pyvista as pv
import pytest

from b3_blade.templates.blade_mini import BladeMini
from b3_geo.build import build_w_vtk
from b3_geo.schema import validate_w_vtk


@pytest.fixture(scope="module")
def reference_blade():
    return BladeMini()


def test_build_w_vtk_runs(reference_blade, tmp_path):
    pf = reference_blade.planform()
    af = reference_blade.airfoil_stack()
    w = build_w_vtk(pf, af, n_sections=80)
    path = tmp_path / "W.vtp"
    w.save(str(path))
    assert path.exists()


def test_validate_w_vtk_passes(reference_blade, tmp_path):
    pf = reference_blade.planform()
    af = reference_blade.airfoil_stack()
    w = build_w_vtk(pf, af, n_sections=80)
    validate_w_vtk(w)


def test_round_trip(reference_blade, tmp_path):
    pf = reference_blade.planform()
    af = reference_blade.airfoil_stack()
    w = build_w_vtk(pf, af, n_sections=80)
    path = tmp_path / "W.vtp"
    w.save(str(path))
    w2 = pv.read(str(path))
    validate_w_vtk(w2)


def test_file_size(reference_blade, tmp_path):
    pf = reference_blade.planform()
    af = reference_blade.airfoil_stack()
    w = build_w_vtk(pf, af, n_sections=80)
    path = tmp_path / "W.vtp"
    w.save(str(path))
    size_mb = path.stat().st_size / 1024 / 1024
    assert size_mb < 5.0, f"W.vtp is {size_mb:.2f} MB, exceeds 5 MB limit"
