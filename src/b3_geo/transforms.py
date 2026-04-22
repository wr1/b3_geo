"""Coordinate frame transforms for b3_geo.

Three frames:
  UACS   col 0 = chord [0,1],  col 1 = thickness (suction +y)
  SDACS  col 0 = chord physical (mid-chord origin), col 1 = thickness physical
  GBCS   col 0 = flapwise, col 1 = edgewise (LE→TE), col 2 = span

Sole location for rotation, scaling, translation, and column-swap math.
"""

from __future__ import annotations

import numpy as np


def uacs_to_sdacs(uacs: np.ndarray, chord: float) -> np.ndarray:
    """UACS (M, 2) → SDACS (M, 2).

    x -= 0.5 to centre at mid-chord, then scale both cols by chord.
    """
    sdacs = uacs.copy()
    sdacs[:, 0] -= 0.5
    sdacs *= chord
    return sdacs


def sdacs_to_gbcs(
    sdacs: np.ndarray,
    twist: float,
    dx: float,
    dy: float,
    z: float,
) -> np.ndarray:
    """SDACS (M, 2) → GBCS (M, 3).

    twist: degrees (planform convention).
    1. Rotate (col 0, col 1) by twist around z-axis.
    2. Swap cols 0 ↔ 1  (chord→edgewise, thickness→flapwise).
    3. col 0 += dx (flapwise prebend), col 1 += dy (edgewise sweep).
    4. Append col 2 = z (span).
    """
    twist_rad = twist * np.pi / 180.0
    cos_t = np.cos(twist_rad)
    sin_t = np.sin(twist_rad)
    c = sdacs[:, 0]
    t = sdacs[:, 1]
    c_rot = c * cos_t - t * sin_t
    t_rot = c * sin_t + t * cos_t
    # swap: col0 = flapwise (was thickness), col1 = edgewise (was chord)
    flapwise = t_rot + dx
    edgewise = c_rot + dy
    span = np.full(len(sdacs), z)
    return np.column_stack([flapwise, edgewise, span])


def gbcs_to_sdacs(
    gbcs: np.ndarray,
    twist: float,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Inverse of sdacs_to_gbcs (round-trip testing only). twist in degrees."""
    flapwise = gbcs[:, 0] - dx
    edgewise = gbcs[:, 1] - dy
    c_rot = edgewise
    t_rot = flapwise
    twist_rad = twist * np.pi / 180.0
    cos_t = np.cos(twist_rad)
    sin_t = np.sin(twist_rad)
    c = c_rot * cos_t + t_rot * sin_t
    t = -c_rot * sin_t + t_rot * cos_t
    return np.column_stack([c, t])


def sdacs_to_uacs(sdacs: np.ndarray, chord: float) -> np.ndarray:
    """Inverse of uacs_to_sdacs (round-trip testing only)."""
    uacs = sdacs / chord
    uacs[:, 0] += 0.5
    return uacs
