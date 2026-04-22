"""b3_geo — W vtk producer: wires + coordinate-frame fields for downstream meshing."""

from __future__ import annotations

from b3_geo.build import build_w_vtk, get_wire
from b3_geo.schema import validate_w_vtk
from b3_geo import transforms

__version__ = "2.0.0"

__all__ = [
    "build_w_vtk",
    "get_wire",
    "validate_w_vtk",
    "transforms",
]
