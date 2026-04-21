"""Section loft internals — pure numpy port of b3_geo.core.blade._get_sections_np."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from b3_blade.airfoil_stack import AirfoilStack
from b3_blade.planform import Planform


def build_sections(
    planform: Planform,
    airfoils: AirfoilStack,
    spans: np.ndarray,
    npchord: int = 200,
) -> np.ndarray:
    """Return (N_spans, npchord, 3) array of 3D section coordinates.

    Replicates b3_geo.core.blade.Blade._get_sections_np() semantics using
    b3_blade primitives. Same geometry maths, different input types.

    Section positioning per span station i:
      1. Select airfoil shape by t/c interpolation
      2. Centre: translate y by -0.5 (twist centre at chord midpoint)
      3. Scale: multiply xy by chord[i]
      4. Rotate: twist[i] degrees around z-axis (right-hand, positive nose-down)
      5. Translate: add (dx[i], dy[i], z[i])
    """
    n = len(spans)
    sections = np.zeros((n, npchord, 3))

    tcs = planform.tc.at(spans)
    chords = planform.chord.at(spans)
    twists = planform.twist.at(spans)
    dxs = planform.dx.at(spans)
    dys = planform.dy.at(spans)
    zs = planform.z.at(spans)

    xy_norm = _interpolate_airfoil_shapes(airfoils, tcs, npchord)

    for i in range(n):
        xy = xy_norm[:, i, :]
        pts = np.column_stack((xy, np.zeros(npchord)))
        pts[:, 1] -= 0.5
        pts[:, :2] *= chords[i]
        angle = -float(twists[i]) * np.pi / 180.0
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        x_rot = pts[:, 0] * cos_a - pts[:, 1] * sin_a
        y_rot = pts[:, 0] * sin_a + pts[:, 1] * cos_a
        pts[:, 0] = x_rot + float(dxs[i])
        pts[:, 1] = y_rot + float(dys[i])
        pts[:, 2] = float(zs[i])
        sections[i] = pts

    return sections


def _interpolate_airfoil_shapes(
    stack: AirfoilStack,
    tcs: np.ndarray,
    npchord: int,
) -> np.ndarray:
    """Return (npchord, N_spans, 2) array via linear interpolation over t/c.

    Matches b3_geo's interp1d approach: interpolate x and y separately
    over the t/c axis across the AirfoilStack entries.
    """
    entries = stack.entries
    entry_tcs = np.array([
        float(e[1].metadata.get("thickness", e[0]))
        for e in entries
    ])
    sort_idx = np.argsort(entry_tcs)
    sorted_tcs = entry_tcs[sort_idx]
    sorted_entries = [entries[i] for i in sort_idx]

    xs_list = []
    ys_list = []
    for _, foil in sorted_entries:
        xy = _repanel(foil.xy, npchord)
        xs_list.append(xy[:, 0])
        ys_list.append(xy[:, 1])

    x_all = np.stack(xs_list, axis=1)
    y_all = np.stack(ys_list, axis=1)

    x_interp = interp1d(sorted_tcs, x_all, axis=1, bounds_error=False,
                        fill_value=(x_all[:, 0], x_all[:, -1]))
    y_interp = interp1d(sorted_tcs, y_all, axis=1, bounds_error=False,
                        fill_value=(y_all[:, 0], y_all[:, -1]))

    xs = x_interp(tcs)
    ys = y_interp(tcs)
    return np.dstack((xs, ys))


def _repanel(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample airfoil to n points using arc-length parameterisation."""
    if xy.shape[0] == n:
        return xy
    if xy.shape[0] < 2:
        return np.tile(xy, (n, 1))[:n]
    dists = np.cumsum(np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1)])
    t = dists / dists[-1]
    t_new = np.linspace(0, 1, n)
    x_new = np.interp(t_new, t, xy[:, 0])
    y_new = np.interp(t_new, t, xy[:, 1])
    return np.column_stack((x_new, y_new))
