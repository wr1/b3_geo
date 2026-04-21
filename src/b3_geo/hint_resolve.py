"""hint_resolve — materialise MeshHints into resolved (s, chord_frac) arrays.

Called by realise() after web resolution. Produces ResolvedHardpoint and
ResolvedRefinement objects that carry explicit arrays ready for the mesher.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from b3_blade.mesh_hints import (
    BoxMask, FullChord, HardpointHint, MeshHints, RefineHP,
    RefinementHint, RibbonMask,
)
from b3_geo.web_resolve import WebGeom


@dataclass
class ResolvedHardpoint:
    """Hardpoint with curve materialised as (s, chord_frac) coordinate arrays."""
    s_array: np.ndarray          # (N,) sorted rel_span
    chord_frac_array: np.ndarray # (N,) chord fraction at each s
    refine: RefineHP | None
    name: str
    source_web: str | None = None   # which web produced this (if any)


@dataclass
class ResolvedRefinement:
    """Refinement region with explicit bounding box in (s, chord_frac) space."""
    s_range: tuple[float, float]
    chord_range: tuple[float, float]
    factor: float
    name: str


def resolve_hardpoints(
    hints: MeshHints,
    resolved_webs: list[WebGeom],
    spans: np.ndarray,
) -> list[ResolvedHardpoint]:
    """Materialise all HardpointHints into ResolvedHardpoint objects.

    Two sources:
    1. HardpointHint with a Curve → evaluate curve at spans grid.
    2. HardpointHint with curve=None → look up matching web by name suffix
       and use its chord_frac array directly.
    """
    web_by_name = {w.name: w for w in resolved_webs}
    result: list[ResolvedHardpoint] = []

    for hint in hints.hardpoints:
        if hint.curve is not None:
            # Explicit Curve — evaluate directly
            cf = hint.curve.at(spans)
            result.append(ResolvedHardpoint(
                s_array=spans.copy(),
                chord_frac_array=cf,
                refine=hint.refine,
                name=hint.name,
            ))
        else:
            # Placeholder from _default_web_hardpoints — name is "<web_name>_hp"
            # Recover the web name by stripping the "_hp" suffix
            web_name = hint.name.removesuffix("_hp") if hint.name.endswith("_hp") else hint.name
            web = web_by_name.get(web_name)
            if web is None:
                continue  # web not found (e.g. z_range excludes all spans)
            result.append(ResolvedHardpoint(
                s_array=spans.copy(),
                chord_frac_array=web.chord_frac.copy(),
                refine=hint.refine,
                name=hint.name,
                source_web=web_name,
            ))

    return result


def resolve_refinements(
    hints: MeshHints,
    resolved_webs: list[WebGeom],
    spans: np.ndarray,
) -> list[ResolvedRefinement]:
    """Materialise all RefinementHints into ResolvedRefinement objects."""
    web_by_name = {w.name: w for w in resolved_webs}
    result: list[ResolvedRefinement] = []

    for hint in hints.refinements:
        region = hint.region
        if isinstance(region, RibbonMask):
            web = web_by_name.get(region.ribbon_name)
            if web is None:
                continue
            cf = web.chord_frac
            chord_min = float(np.nanmin(cf) - region.expand)
            chord_max = float(np.nanmax(cf) + region.expand)
            s_range = (float(spans[0]), float(spans[-1]))
            chord_range = (max(0.0, chord_min), min(1.0, chord_max))
        else:
            s_range_raw, chord_range_raw = region.bounds()
            s_range = s_range_raw
            chord_range = chord_range_raw

        result.append(ResolvedRefinement(
            s_range=s_range,
            chord_range=chord_range,
            factor=hint.factor,
            name=hint.name,
        ))

    return result
