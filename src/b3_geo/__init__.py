"""b3_geo — 3D realisation of b3_blade Planform and Web objects."""

from __future__ import annotations

from b3_geo.realise import RealisedBlade, realise
from b3_geo.web_resolve import WebGeom

__version__ = "0.1.0"

__all__ = [
    "realise",
    "RealisedBlade",
    "WebGeom",
]
