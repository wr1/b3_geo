"""Coordinate frame transforms for b3_geo.

Three frames:
  UACS   col 0 = chord [0,1],  col 1 = thickness (suction +y)
  SDACS  col 0 = chord physical (mid-chord origin), col 1 = thickness physical
  GBCS   col 0 = flapwise, col 1 = edgewise (LE→TE), col 2 = span

Beam adapters (b3_blade.adapters.gx) consume a fourth axis convention,
GXBeam element-local: (e1, e2, e3) = (axial, thick, chord). The mapping
SDACS [chord, thick, axial] → GX-local [axial, thick, chord] is the
constant ``R3_sdacs_to_gxlocal`` permutation; twist-about-axial lives in
``R3_about_e1`` applied to K in GX-local.

Sole location for rotation, scaling, translation, column-swap, and 6-DOF
block math.
"""

from __future__ import annotations

import numpy as np


# ── 3×3 building blocks ─────────────────────────────────────────────────────

def R3_axial_rot(theta_rad: float) -> np.ndarray:
    """Rotation about SDACS axis 3 (axial) by ``theta_rad``.

    Rotates the (chord, thick) plane while keeping axial fixed.
    """
    c, s = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def R3_sdacs_to_gxlocal() -> np.ndarray:
    """Constant axis-rename matrix from SDACS [chord, thick, axial] to
    GXBeam element-local [axial, thick, chord].

    The matrix P satisfies v_gx = P · v_sdacs. Equivalently, applied to
    a 6-vector with block-diag (P, P), this is the integer permutation
    σ = (2, 1, 0, 5, 4, 3): K_gx[i,j] = K_anba[σ(i), σ(j)].
    """
    return np.array([[0.0, 0.0, 1.0],
                     [0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0]], dtype=np.float64)


def R3_about_e1(theta_rad: float) -> np.ndarray:
    """Rotation about GX-local e1 (axial) by ``theta_rad``.

    Same physical twist as ``R3_axial_rot`` but expressed in the GX-local
    basis (which puts axial first). Mixes the (e2=thick, e3=chord) pair.
    """
    c, s = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c,  -s],
                     [0.0, s,   c]], dtype=np.float64)


# ── 6×6 lift ────────────────────────────────────────────────────────────────

def T6_block(R3: np.ndarray) -> np.ndarray:
    """Lift a 3×3 rotation to the 6-DOF block-diagonal transform
    block_diag(R3, R3) — applies R3 identically to forces (top 3) and
    moments (bottom 3) of a 6-DOF stiffness/mass matrix.
    """
    T = np.zeros((6, 6), dtype=np.float64)
    T[:3, :3] = R3
    T[3:, 3:] = R3
    return T


def K_anba_to_gxlocal(K_anba: np.ndarray, twist_rad: float) -> np.ndarray:
    """Strict 6×6 transform: SDACS ANBA stiffness/mass matrix → GXBeam
    element-local matrix.

    Only the constant axis rename is applied here::

        K_gx = T_axes · K_anba · T_axes^T

    where ``T_axes = block_diag(P, P)`` with ``P = R3_sdacs_to_gxlocal()``
    sends SDACS [chord, thick, axial] → GX [axial, thick, chord]. At
    twist=0 this is the integer permutation σ=(2,1,0,5,4,3), which is the
    *strict* DOF reordering — the consistent block-diagonal axis rename.
    The previously-shipped empirical permutation [2,1,0,5,3,4] is **not**
    block-diagonal: forces with e2=thick, moments with e2=chord, which
    manifests as ~4× edge-bending softness in beam-vs-shell comparisons.

    Twist is *not* applied to K. It is encoded in the per-element world
    frame ``F_e`` (which has columns ``(tangent, thick_world(θ),
    chord_world(θ))``) so that
    ``K_world = F_e · K_gx · F_e.T = R_section_to_world(θ) · K_anba ·
    R_section_to_world(θ).T``. Putting twist on both K and F_e
    double-rotates the basis and breaks the round trip.

    The ``twist_rad`` argument is retained for API symmetry with the chain
    narrative (``SDACS → twist → axis rename → element-local → world``)
    but is currently unused — the rotation is geometric, lives in F_e.
    """
    del twist_rad  # twist is encoded in F_e, not in K_gx
    T_axes = T6_block(R3_sdacs_to_gxlocal())
    return T_axes @ np.asarray(K_anba, dtype=np.float64) @ T_axes.T


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
