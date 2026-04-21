"""Web resolution — dispatch Web subclasses to 3D cutting geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from b3_blade.planform import Planform
from b3_blade.web import ChordFracWeb, IsoWeb, PlaneWeb, RibbonWeb, TrailingEdgeWeb, Web


@dataclass
class WebGeom:
    """Resolved web geometry for one shear web.

    upper_pts / lower_pts are the primary output — 3D points on the suction /
    pressure surface where the web intersects each span station.

    chord_frac is a derived summary (midpoint x/chord) useful for 2-D plots;
    it is NOT the input for PlaneWeb (where it varies with span due to twist).
    """

    name: str
    chord_frac: np.ndarray           # (N_spans,) derived: midpoint x/chord
    z_abs: np.ndarray                # (N_spans,) absolute z
    upper_pts: np.ndarray | None = None  # (N_spans, 3) suction-side intersection
    lower_pts: np.ndarray | None = None  # (N_spans, 3) pressure-side intersection
    origin: np.ndarray | None = None
    orientation: np.ndarray | None = None


class UnknownWebType(TypeError):
    pass


def resolve(
    web: Web,
    planform: Planform,
    spans: np.ndarray,
    sections: np.ndarray,
    resolved: dict[str, WebGeom] | None = None,
) -> WebGeom:
    """Resolve Web object → WebGeom.

    sections: (N_spans, N_chord, 3) from build_sections — required for all
              types so that intersection geometry can be computed.
    resolved: already-resolved WebGeoms by name (for RibbonWeb reference lookup).
    """
    resolved = resolved or {}
    match web:
        case PlaneWeb():
            return _resolve_plane(web, planform, spans, sections)
        case RibbonWeb():
            return _resolve_ribbon(web, planform, spans, sections, resolved)
        case TrailingEdgeWeb():
            return _resolve_trailing_edge(web, planform, spans, sections)
        case ChordFracWeb():
            return _resolve_chord_frac(web, planform, spans, sections)
        case IsoWeb():
            return _resolve_iso(web, planform, spans, sections)
        case _:
            raise UnknownWebType(f"Unknown web type: {type(web).__name__}")


# ── helpers ────────────────────────────────────────────────────────────────────

def _local_axes(planform: Planform, spans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (chord_dirs, normal_dirs) arrays, each (N_spans, 3).

    chord_dir points LE→TE in the section plane.
    normal_dir points suction-side (upward in section plane).
    """
    twists_rad = -planform.twist.at(spans) * (np.pi / 180.0)
    cos_t = np.cos(twists_rad)
    sin_t = np.sin(twists_rad)
    chord_dirs  = np.column_stack([cos_t,  sin_t,  np.zeros_like(cos_t)])
    normal_dirs = np.column_stack([-sin_t, cos_t,  np.zeros_like(cos_t)])
    return chord_dirs, normal_dirs


def _chord_frac_to_pts(
    sections: np.ndarray,
    chord_fracs: np.ndarray,
    chord_dirs: np.ndarray,
    normal_dirs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find 3D upper/lower surface points closest to chord_fracs.

    Works in local chord coordinates: projects points onto chord_dir, splits
    suction/pressure by sign of normal_dir projection, picks closest to cf_i.

    Returns (upper_pts, lower_pts) each (N_spans, 3).
    """
    n_spans = sections.shape[0]
    upper = np.full((n_spans, 3), np.nan)
    lower = np.full((n_spans, 3), np.nan)

    for i in range(n_spans):
        sec = sections[i]
        cf_i = float(np.clip(chord_fracs[i], 0.0, 1.0))
        cd, nd = chord_dirs[i], normal_dirs[i]

        proj_c = sec @ cd
        chord_len = proj_c.max() - proj_c.min()
        if chord_len < 1e-9:
            continue
        pt_cf = (proj_c - proj_c.min()) / chord_len

        le_pt = sec[np.argmin(proj_c)]
        proj_n = (sec - le_pt) @ nd

        for mask, arr in [(proj_n >= 0, upper), (proj_n < 0, lower)]:
            idxs = np.where(mask)[0]
            if len(idxs) < 2:
                continue
            j = idxs[np.argmin(np.abs(pt_cf[idxs] - cf_i))]
            arr[i] = sec[j]

    return upper, lower


def _derive_chord_frac(
    upper_pts: np.ndarray,
    lower_pts: np.ndarray,
    sections: np.ndarray,
    chord_dirs: np.ndarray,
) -> np.ndarray:
    """Compute chord_frac of the web midpoint at each span station."""
    n_spans = sections.shape[0]
    cf = np.zeros(n_spans)
    for i in range(n_spans):
        if np.any(np.isnan(upper_pts[i])) or np.any(np.isnan(lower_pts[i])):
            continue
        mid = (upper_pts[i] + lower_pts[i]) / 2.0
        cd = chord_dirs[i]
        proj_c = sections[i] @ cd
        chord_len = proj_c.max() - proj_c.min()
        if chord_len < 1e-9:
            continue
        cf[i] = ((mid @ cd) - proj_c.min()) / chord_len
    return cf


# ── PlaneWeb ──────────────────────────────────────────────────────────────────

def _resolve_plane(
    web: PlaneWeb,
    planform: Planform,
    spans: np.ndarray,
    sections: np.ndarray,
) -> WebGeom:
    """Intersect a 3D plane with each section.

    The web is a flat panel in 3D space — it does NOT follow blade twist.
    chord_frac therefore varies along span (the primary reason PlaneWeb exists:
    flat mould, variable chord position).
    """
    origin = np.array(web.origin, dtype=float)
    normal = np.array(web.orientation, dtype=float)
    normal /= np.linalg.norm(normal)
    zs = planform.z.at(spans)
    n_spans = len(spans)

    chord_dirs, normal_dirs = _local_axes(planform, spans)
    upper_pts = np.full((n_spans, 3), np.nan)
    lower_pts = np.full((n_spans, 3), np.nan)

    for i in range(n_spans):
        sec = sections[i]
        d = (sec - origin) @ normal
        n_pts = len(sec)

        crossings: list[np.ndarray] = []
        for j in range(n_pts):
            j1 = (j + 1) % n_pts
            dj, dj1 = d[j], d[j1]
            if dj * dj1 < 0:
                t = dj / (dj - dj1)
                crossings.append(sec[j] + t * (sec[j1] - sec[j]))

        if len(crossings) < 2:
            continue

        # sort by suction/pressure: suction = higher normal_dir component
        le_pt = sec[np.argmin(sec @ chord_dirs[i])]
        nd = normal_dirs[i]
        crossings.sort(key=lambda p, _le=le_pt, _nd=nd: (p - _le) @ _nd,
                       reverse=True)
        upper_pts[i] = crossings[0]
        lower_pts[i] = crossings[-1]

    chord_fracs = _derive_chord_frac(upper_pts, lower_pts, sections, chord_dirs)

    return WebGeom(
        name=web.name,
        chord_frac=chord_fracs,
        z_abs=zs,
        upper_pts=upper_pts,
        lower_pts=lower_pts,
        origin=origin,
        orientation=normal,
    )


# ── RibbonWeb ─────────────────────────────────────────────────────────────────

def _resolve_ribbon(
    web: RibbonWeb,
    planform: Planform,
    spans: np.ndarray,
    sections: np.ndarray,
    resolved: dict[str, WebGeom],
) -> WebGeom:
    """Ribbon web: chord-fraction offset from a reference web.

    offset curve gives signed chord-fraction shift from reference web's
    chord_frac at each span station.
    """
    zs = planform.z.at(spans)
    offsets = web.offsets.at(spans)

    if web.reference_web in resolved:
        ref_cf = resolved[web.reference_web].chord_frac
    else:
        ref_cf = np.full(len(spans), 0.5)

    chord_fracs = np.clip(ref_cf + offsets, 0.0, 1.0)
    chord_dirs, normal_dirs = _local_axes(planform, spans)
    upper_pts, lower_pts = _chord_frac_to_pts(
        sections, chord_fracs, chord_dirs, normal_dirs
    )

    return WebGeom(
        name=web.name,
        chord_frac=chord_fracs,
        z_abs=zs,
        upper_pts=upper_pts,
        lower_pts=lower_pts,
    )


# ── TrailingEdgeWeb ───────────────────────────────────────────────────────────

def _resolve_trailing_edge(
    web: TrailingEdgeWeb,
    planform: Planform,
    spans: np.ndarray,
    sections: np.ndarray,
) -> WebGeom:
    """Trailing-edge web: 3D plane near TE (same intersection logic as PlaneWeb)."""
    # treat as a PlaneWeb with the given orientation, origin at TE chord position
    origin = np.array(getattr(web, "origin", (1.0, 0.0, 0.0)), dtype=float)
    normal = np.array(web.orientation, dtype=float)
    normal /= np.linalg.norm(normal)
    zs = planform.z.at(spans)
    n_spans = len(spans)

    chord_dirs, normal_dirs = _local_axes(planform, spans)
    upper_pts = np.full((n_spans, 3), np.nan)
    lower_pts = np.full((n_spans, 3), np.nan)

    for i in range(n_spans):
        sec = sections[i]
        d = (sec - origin) @ normal
        n_pts = len(sec)
        crossings: list[np.ndarray] = []
        for j in range(n_pts):
            j1 = (j + 1) % n_pts
            dj, dj1 = d[j], d[j1]
            if dj * dj1 < 0:
                t = dj / (dj - dj1)
                crossings.append(sec[j] + t * (sec[j1] - sec[j]))
        if len(crossings) < 2:
            continue
        le_pt = sec[np.argmin(sec @ chord_dirs[i])]
        nd = normal_dirs[i]
        crossings.sort(key=lambda p, _le=le_pt, _nd=nd: (p - _le) @ _nd,
                       reverse=True)
        upper_pts[i] = crossings[0]
        lower_pts[i] = crossings[-1]

    chord_fracs = _derive_chord_frac(upper_pts, lower_pts, sections, chord_dirs)
    return WebGeom(
        name=web.name,
        chord_frac=chord_fracs,
        z_abs=zs,
        upper_pts=upper_pts,
        lower_pts=lower_pts,
        orientation=normal,
    )


# ── ChordFracWeb ─────────────────────────────────────────────────────────────

def _resolve_chord_frac(
    web: ChordFracWeb,
    planform: Planform,
    spans: np.ndarray,
    sections: np.ndarray,
) -> WebGeom:
    """Chord-fraction web: explicitly prescribed, follows blade twist."""
    zs = planform.z.at(spans)
    chord_fracs = np.clip(web.chord_frac.at(spans), 0.0, 1.0)
    chord_dirs, normal_dirs = _local_axes(planform, spans)
    upper_pts, lower_pts = _chord_frac_to_pts(
        sections, chord_fracs, chord_dirs, normal_dirs
    )
    return WebGeom(
        name=web.name,
        chord_frac=chord_fracs,
        z_abs=zs,
        upper_pts=upper_pts,
        lower_pts=lower_pts,
    )


# ── IsoWeb ────────────────────────────────────────────────────────────────────

def _resolve_iso(
    web: IsoWeb,
    planform: Planform,
    spans: np.ndarray,
    sections: np.ndarray,
) -> WebGeom:
    """Iso-field web: chord_frac prescribed as iso_value curve."""
    zs = planform.z.at(spans)
    chord_fracs = np.clip(web.iso_value.at(spans), 0.0, 1.0)
    chord_dirs, normal_dirs = _local_axes(planform, spans)
    upper_pts, lower_pts = _chord_frac_to_pts(
        sections, chord_fracs, chord_dirs, normal_dirs
    )
    return WebGeom(
        name=web.name,
        chord_frac=chord_fracs,
        z_abs=zs,
        upper_pts=upper_pts,
        lower_pts=lower_pts,
    )
