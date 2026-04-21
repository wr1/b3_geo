"""realise() — Planform + list[Web] + AirfoilStack → RealisedBlade."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from b3_blade.airfoil_stack import AirfoilStack
from b3_blade.mesh_hints import MeshHints
from b3_blade.planform import Planform
from b3_blade.web import Web
from b3_geo._section import build_sections
from b3_geo.fields import build_fields
from b3_geo.hint_resolve import (
    ResolvedHardpoint, ResolvedRefinement,
    resolve_hardpoints, resolve_refinements,
)
from b3_geo.web_resolve import WebGeom, resolve


@dataclass
class RealisedBlade:
    """Output of realise() — 3D geometry arrays ready for meshing and draping."""

    sections: np.ndarray
    spans_abs: np.ndarray
    spans_rel: np.ndarray
    webs: list[WebGeom] = field(default_factory=list)
    fields: pd.DataFrame = field(default_factory=pd.DataFrame)
    hardpoint_curves: list[ResolvedHardpoint] = field(default_factory=list)
    refinement_regions: list[ResolvedRefinement] = field(default_factory=list)

    # ------------------------------------------------------------------
    # 3D geometry queries (couplings that need sections)
    # ------------------------------------------------------------------

    def web_chord_frac(self, web_name: str) -> np.ndarray:
        """Chord fraction array for a named web, shape (N_spans,)."""
        for w in self.webs:
            if w.name == web_name:
                return w.chord_frac
        raise KeyError(f"Web '{web_name}' not found in RealisedBlade")


def realise(
    planform: Planform,
    webs: list[Web],
    airfoils: AirfoilStack,
    spans: np.ndarray,
    npchord: int = 200,
    mesh_hints: MeshHints | None = None,
) -> RealisedBlade:
    """Produce RealisedBlade from b3_blade objects.

    Port of b3_geo.core.blade.Blade._get_sections_np() semantics, consuming
    b3_blade primitives instead of b3_geo pydantic models. Same geometry maths.

    spans:      relative span grid, e.g. np.linspace(0, 1, 50)
    mesh_hints: optional MeshHints; if provided, hardpoint_curves and
                refinement_regions are resolved and attached to RealisedBlade.
    """
    sections = build_sections(planform, airfoils, spans, npchord=npchord)
    spans_abs = planform.z.at(spans)

    # resolve in declaration order so RibbonWeb can look up reference webs
    resolved_webs: list[WebGeom] = []
    resolved_by_name: dict[str, WebGeom] = {}
    for w in webs:
        rw = resolve(w, planform, spans, sections, resolved=resolved_by_name)
        resolved_webs.append(rw)
        resolved_by_name[rw.name] = rw

    web_positions = {w.name: w.chord_frac for w in resolved_webs}
    df = build_fields(sections, spans, spans_abs, web_positions=web_positions)

    hardpoints: list[ResolvedHardpoint] = []
    refinements: list[ResolvedRefinement] = []
    if mesh_hints:
        hardpoints = resolve_hardpoints(mesh_hints, resolved_webs, spans)
        refinements = resolve_refinements(mesh_hints, resolved_webs, spans)

    return RealisedBlade(
        sections=sections,
        spans_abs=spans_abs,
        spans_rel=spans,
        webs=resolved_webs,
        fields=df,
        hardpoint_curves=hardpoints,
        refinement_regions=refinements,
    )
