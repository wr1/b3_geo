import pyvista as pv
import numpy as np
from typing import TYPE_CHECKING
from .interpolation import build_sections_poly

if TYPE_CHECKING:
    from ..core.blade import Blade


def save_blade_sections(blade: "Blade", filepath: str, sections=None, rel_spans=None):
    """Save blade sections to VTP with planform data."""
    if sections is None:
        sections = blade.get_sections()
        rel_spans = blade.rel_span
    else:
        if rel_spans is None:
            rel_spans = blade.rel_span
    points = sections.reshape(-1, 3)
    grid = build_sections_poly(points, blade.np_chordwise, blade.np_spanwise)
    grid.field_data["np_spanwise"] = [len(rel_spans)]
    grid.field_data["np_chordwise"] = [blade.np_chordwise]
    # Add point_data for planform parameters
    vals = blade.get_planform_array(rel_spans)
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
        if k == "rel_span":
            grid.point_data[k] = np.repeat(rel_spans, blade.np_chordwise)
        else:
            grid.point_data[k] = np.repeat(vals[k], blade.np_chordwise)
    # Convert to PolyData for VTP
    poly = pv.PolyData()
    poly.points = points
    lines = []
    n_sections = len(rel_spans)
    for i in range(n_sections):
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
    t = np.linspace(0, 1, blade.np_chordwise)
    poly.point_data["t"] = np.tile(t, n_sections)
    # Add section_id
    poly.point_data["section_id"] = np.repeat(np.arange(n_sections), blade.np_chordwise)
    poly.save(filepath)
