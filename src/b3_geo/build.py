"""Assemble W vtk from a list of rel-span sections."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from b3_blade.airfoil_stack import AirfoilStack
from b3_blade.planform import Planform
from b3_geo.wire import build_wire


def build_w_vtk(
    planform: Planform,
    airfoil_stack: AirfoilStack,
    n_sections: int = 80,
    n_chord: int = 200,
    sections: np.ndarray | None = None,
) -> pv.PolyData:
    """Build W vtk: all section wires in one pv.PolyData.

    If sections is None, samples np.linspace(0, 1, n_sections).
    If provided, it must be a 1-D array of rel-span values in [0, 1].

    field_data arrays have length n_sections (or len(sections)).
    point_data arrays are flat across all polylines.
    """
    s_vals = np.linspace(0.0, 1.0, n_sections) if sections is None else np.asarray(sections)
    k = len(s_vals)

    wires = [build_wire(planform, airfoil_stack, float(s), n_chord) for s in s_vals]

    # concatenate points
    all_points = np.vstack([w.points for w in wires])

    # build flat cells array: one VTK_POLY_LINE per wire
    cells = []
    offset = 0
    for w in wires:
        n_pts = w.n_points
        cell_arr = np.concatenate([[n_pts], np.arange(offset, offset + n_pts)])
        cells.append(cell_arr)
        offset += n_pts
    all_lines = np.concatenate(cells).astype(np.int_)

    poly = pv.PolyData()
    poly.points = all_points.astype(np.float64)
    poly.lines  = all_lines

    # point_data — flat across all wires
    scalar_keys = ("geo.abs_t", "geo.rel_t",
                   "geo.arc_from_te", "geo.arc_from_te_ss", "geo.arc_from_te_ps",
                   "geo.t_from_te_ss", "geo.t_from_te_ps",
                   "geo.signed_arc_from_le",
                   "geo.s", "geo.z", "geo.chord",
                   "geo.twist", "geo.tc", "geo.dx", "geo.dy",
                   "rel_span")
    for key in scalar_keys:
        poly.point_data[key] = np.concatenate(
            [w.point_data[key] for w in wires]
        ).astype(np.float64)
    for key in ("geo.uacs", "geo.sdacs"):
        poly.point_data[key] = np.vstack(
            [w.point_data[key] for w in wires]
        ).astype(np.float64)

    # field_data — one value per polyline, array length = k (programmatic access)
    for key in ("geo.s", "geo.z", "geo.chord", "geo.twist", "geo.tc", "geo.dx", "geo.dy"):
        poly.field_data[key] = np.array(
            [float(w.field_data[key][0]) for w in wires], dtype=np.float64
        )

    return poly


def get_wire(
    planform: Planform,
    airfoil_stack: AirfoilStack,
    s: float,
    n_chord: int = 200,
) -> pv.PolyData:
    """Runtime API for downstream mesher to request a single wire on demand."""
    return build_wire(planform, airfoil_stack, s, n_chord)
