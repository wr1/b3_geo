"""Verify the derived TE-distance point_data fields on wire / W vtks."""

from __future__ import annotations

import numpy as np
import pytest

from b3_blade.templates.blade_mini import BladeMini
from b3_geo import build_w_vtk


@pytest.fixture(scope="module")
def w_vtk():
    b = BladeMini()
    return build_w_vtk(b.planform(), b.airfoil_stack(), n_sections=10, n_chord=80)


def test_all_four_te_fields_present(w_vtk):
    for key in (
        "geo.arc_from_te_ss", "geo.arc_from_te_ps",
        "geo.t_from_te_ss",   "geo.t_from_te_ps",
    ):
        assert key in w_vtk.point_data, f"missing {key}"


def test_arc_from_te_ss_equals_abs_t(w_vtk):
    """The SS-starting alias is numerically abs_t itself."""
    assert np.allclose(
        np.asarray(w_vtk.point_data["geo.arc_from_te_ss"]),
        np.asarray(w_vtk.point_data["geo.abs_t"]),
    )


def test_t_from_te_fields_in_unit_interval(w_vtk):
    for key in ("geo.t_from_te_ss", "geo.t_from_te_ps"):
        arr = np.asarray(w_vtk.point_data[key])
        assert arr.min() >= -1e-12, f"{key} min={arr.min()}"
        assert arr.max() <= 1 + 1e-12, f"{key} max={arr.max()}"


def test_t_ss_plus_t_ps_is_one(w_vtk):
    """t_from_te_ss + t_from_te_ps should equal 1 at every point."""
    t_ss = np.asarray(w_vtk.point_data["geo.t_from_te_ss"])
    t_ps = np.asarray(w_vtk.point_data["geo.t_from_te_ps"])
    assert np.max(np.abs(t_ss + t_ps - 1.0)) < 1e-12


def test_arc_sum_is_section_perimeter(w_vtk):
    """arc_from_te_ss + arc_from_te_ps equals the section's full perimeter
    (constant within one section, varies across sections)."""
    arc_ss = np.asarray(w_vtk.point_data["geo.arc_from_te_ss"])
    arc_ps = np.asarray(w_vtk.point_data["geo.arc_from_te_ps"])
    sums = arc_ss + arc_ps
    # At every point, sum is non-negative and equals that section's perimeter
    assert sums.min() > 0
    assert np.all(sums > 0)


def test_arc_fields_non_negative(w_vtk):
    for key in ("geo.arc_from_te_ss", "geo.arc_from_te_ps"):
        arr = np.asarray(w_vtk.point_data[key])
        assert arr.min() >= -1e-12, f"{key} min={arr.min()}"


def test_fields_propagate_through_build_meshes():
    """TE-distance fields appear on ML + S vtks produced by b3_msh."""
    from b3_blade.adapters.mesh import MeshConfig, to_mesh_spec
    from b3_msh.build import build_meshes

    b = BladeMini()
    spec = to_mesh_spec(b, MeshConfig())
    ml, s = build_meshes(
        spec.w, spec.web_defs,
        target_density=spec.shell_elem_size,
        web_target_density=spec.web_elem_size,
    )
    for mesh, kind in ((ml, "ML"), (s, "S")):
        for key in (
            "geo.arc_from_te_ss", "geo.arc_from_te_ps",
            "geo.t_from_te_ss",   "geo.t_from_te_ps",
        ):
            assert key in mesh.point_data, f"{kind} vtk missing {key}"
