import pyvista as pv
import numpy as np
from typing import Dict, Tuple, TYPE_CHECKING
from .interpolation import build_sections_poly

if TYPE_CHECKING:
    from ..core.blade import Blade


def save_blade_sections(blade: "Blade", filepath: str):
    """Save blade sections to VTP with planform data."""
    points = blade.get_sections().reshape(-1, 3)
    grid = build_sections_poly(points, blade.np_chordwise, blade.np_spanwise)
    grid.field_data["np_spanwise"] = [blade.np_spanwise]
    grid.field_data["np_chordwise"] = [blade.np_chordwise]
    # Add point_data for planform parameters
    for k in [
        "rel_span",
        "z",
        "chord",
        "thickness",
        "absolute_thickness",
        "twist",
        "dx",
        "dy",
    ]:
        val = (
            getattr(blade, k)
            if hasattr(blade, k)
            else blade.get_planform_array(blade.rel_span)[k]
        )
        grid.point_data[k] = np.repeat(val, blade.np_chordwise)
    # Convert to PolyData for VTP
    poly = pv.PolyData()
    poly.points = points
    lines = []
    for i in range(blade.np_spanwise):
        line = [blade.np_chordwise] + list(
            range(i * blade.np_chordwise, (i + 1) * blade.np_chordwise)
        )
        lines.append(line)
    poly.lines = lines
    for k, v in grid.field_data.items():
        poly.field_data[k] = v
    for k, v in grid.point_data.items():
        poly.point_data[k] = v
    # Add t coordinate
    sections = blade.get_sections()  # (n, chord, 3)
    chord = blade.get_planform_array(blade.rel_span)["chord"]
    t = sections[:, :, 0] / chord[:, None]
    poly.point_data["t"] = t.ravel()
    poly.save(filepath)


def load_blade_sections(
    filepath: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int, int]:
    """Load blade sections from VTP with planform data."""
    poly = pv.read(filepath)
    if "np_spanwise" not in poly.field_data or "np_chordwise" not in poly.field_data:
        raise ValueError("Missing metadata in file")
    ns = poly.field_data["np_spanwise"][0]
    nc = poly.field_data["np_chordwise"][0]
    if poly.n_points != ns * nc or poly.n_cells != ns * (nc - 1):
        raise ValueError("Mismatch in sizes")
    points = poly.points.reshape(ns, nc, 3)
    planform = {}
    keys = [
        "rel_span",
        "z",
        "chord",
        "thickness",
        "absolute_thickness",
        "twist",
        "dx",
        "dy",
    ]
    for k in keys:
        if k in poly.point_data:
            data = poly.point_data[k]
            reshaped = data.reshape(ns, nc)
            planform[k] = np.mean(reshaped, axis=1)
        elif k in poly.cell_data:
            data = poly.cell_data[k]
            num_cells_per_section = nc - 1
            reshaped = data.reshape(ns, num_cells_per_section)
            planform[k] = np.mean(reshaped, axis=1)
    return points, planform, nc, ns
