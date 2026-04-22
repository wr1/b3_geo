"""Build one W-vtk wire (polyline) for a single section at rel-span s."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.interpolate import interp1d

from b3_blade.airfoil_stack import AirfoilStack
from b3_blade.planform import Planform
from b3_geo.transforms import sdacs_to_gbcs, uacs_to_sdacs


def build_wire(
    planform: Planform,
    airfoil_stack: AirfoilStack,
    s: float,
    n_chord: int,
) -> pv.PolyData:
    """Build one closed polyline for the section at rel-span s.

    Steps:
      1. eval planform at s: chord, twist, tc, dx, dy, z
      2. get UACS coords (M, 2) via _uacs_at
      3. sdacs = uacs_to_sdacs(uacs, chord)
      4. gbcs  = sdacs_to_gbcs(sdacs, twist, dx, dy, z)
      5. abs_t = cumulative arc length in SDACS (physical units)
         rel_t = abs_t / abs_t[-1]
      6. build pv.PolyData with closed VTK_POLY_LINE

    Closed convention: last point duplicates first (M+1 points).
    VTK_POLY_LINE cell indices: [M+1, 0, 1, ..., M-1, 0].
    """
    s_arr = np.array([s])
    chord = float(planform.chord.at(s_arr)[0])
    twist = float(planform.twist.at(s_arr)[0])       # degrees; sdacs_to_gbcs converts internally
    tc    = float(planform.tc.at(s_arr)[0])
    dx    = float(planform.dx.at(s_arr)[0])
    dy    = float(planform.dy.at(s_arr)[0])
    z     = float(planform.z.at(s_arr)[0])

    uacs  = _uacs_at(airfoil_stack, tc, n_chord)          # (M, 2)
    sdacs = uacs_to_sdacs(uacs, chord)                     # (M, 2)
    gbcs  = sdacs_to_gbcs(sdacs, twist, dx, dy, z)        # (M, 3)

    # arc length in SDACS (physical units)
    diffs  = np.diff(sdacs, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    abs_t  = np.concatenate([[0.0], np.cumsum(seg_len)])
    rel_t  = abs_t / abs_t[-1]

    # closed polyline: duplicate first point
    m = len(gbcs)
    points_closed = np.vstack([gbcs, gbcs[[0]]])           # (M+1, 3)
    cell = np.concatenate([[m + 1], np.arange(m), [0]])    # VTK_POLY_LINE
    lines = cell.astype(np.int_)

    poly = pv.PolyData()
    poly.points = points_closed.astype(np.float64)
    poly.lines  = lines

    # point_data (M points, excluding the closing duplicate)
    abs_t_closed = np.append(abs_t, abs_t[0]).astype(np.float64)
    rel_t_closed = np.append(rel_t, rel_t[0]).astype(np.float64)
    total_arc    = float(abs_t[-1])
    poly.point_data["geo.abs_t"] = abs_t_closed
    poly.point_data["geo.rel_t"] = rel_t_closed
    # Derived TE-distance fields. SS TE is at t=0 (start of the polyline),
    # PS TE is at t=1 (end). Both are arc distances along the wire, valid
    # at every point — "from PS TE" just traverses the other way around.
    poly.point_data["geo.arc_from_te_ss"] = abs_t_closed
    poly.point_data["geo.arc_from_te_ps"] = (total_arc - abs_t_closed).astype(np.float64)
    poly.point_data["geo.t_from_te_ss"]   = rel_t_closed
    poly.point_data["geo.t_from_te_ps"]   = (1.0 - rel_t_closed).astype(np.float64)

    uacs_closed  = np.vstack([uacs,  uacs[[0]]]).astype(np.float64)
    sdacs_closed = np.vstack([sdacs, sdacs[[0]]]).astype(np.float64)
    poly.point_data["geo.uacs"]   = uacs_closed
    poly.point_data["geo.sdacs"]  = sdacs_closed

    # field_data (one scalar per polyline — programmatic access)
    poly.field_data["geo.s"]     = np.array([s],     dtype=np.float64)
    poly.field_data["geo.z"]     = np.array([z],     dtype=np.float64)
    poly.field_data["geo.chord"] = np.array([chord], dtype=np.float64)
    poly.field_data["geo.twist"] = np.array([twist], dtype=np.float64)
    poly.field_data["geo.tc"]    = np.array([tc],    dtype=np.float64)
    poly.field_data["geo.dx"]    = np.array([dx],    dtype=np.float64)
    poly.field_data["geo.dy"]    = np.array([dy],    dtype=np.float64)

    # point_data broadcast — same scalar repeated per point so ParaView can colour by it
    n_pts = m + 1
    for key, val in (
        ("geo.s", s), ("geo.z", z), ("geo.chord", chord),
        ("geo.twist", twist), ("geo.tc", tc), ("geo.dx", dx), ("geo.dy", dy),
    ):
        poly.point_data[key] = np.full(n_pts, val, dtype=np.float64)

    return poly


def _uacs_at(stack: AirfoilStack, tc: float, n_chord: int) -> np.ndarray:
    """Return UACS coords (n_chord, 2) for the airfoil nearest to tc, repanelled.

    UACS convention: col 0 = chord [0,1], col 1 = thickness (suction +y).
    Interpolates across the stack's t/c axis using interp1d (linear, clamp).
    """
    entries = stack.entries

    def _tc_of(entry: tuple) -> float:
        foil = entry[1]
        if "thickness" in foil.metadata:
            return float(foil.metadata["thickness"])
        return float(entry[0])

    sorted_entries = sorted(entries, key=_tc_of)
    tcs_arr = np.array([_tc_of(e) for e in sorted_entries])

    xs_list, ys_list = [], []
    for _, foil in sorted_entries:
        xy = _repanel_uacs(foil.xy, n_chord)
        xs_list.append(xy[:, 0])
        ys_list.append(xy[:, 1])

    x_all = np.stack(xs_list, axis=1)  # (n_chord, n_entries)
    y_all = np.stack(ys_list, axis=1)

    x_interp = interp1d(tcs_arr, x_all, axis=1, bounds_error=False,
                        fill_value=(x_all[:, 0], x_all[:, -1]))
    y_interp = interp1d(tcs_arr, y_all, axis=1, bounds_error=False,
                        fill_value=(y_all[:, 0], y_all[:, -1]))

    xs = x_interp(tc)  # (n_chord,)
    ys = y_interp(tc)
    return np.column_stack([xs, ys])


def _repanel_uacs(xy_raw: np.ndarray, n: int) -> np.ndarray:
    """Resample foil.xy to n points in UACS convention.

    Assumes foil.xy is always (col 0 = chord [0,1], col 1 = thickness) —
    the convention produced by all b3_aerofoil sources (naca4, circle_airfoil,
    load_dat). No column detection or swapping.
    """
    diffs   = np.diff(xy_raw, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    arc     = np.concatenate([[0.0], np.cumsum(seg_len)])
    arc    /= arc[-1]
    t_new   = np.linspace(0.0, 1.0, n)
    c_new   = np.interp(t_new, arc, xy_raw[:, 0])
    th_new  = np.interp(t_new, arc, xy_raw[:, 1])
    return np.column_stack([c_new, th_new])
