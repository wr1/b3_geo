"""Tests for b3_geo.transforms — coordinate frame math."""

from __future__ import annotations

import numpy as np
import pytest

from b3_geo.transforms import (
    K_anba_to_gxlocal,
    R3_about_e1,
    R3_axial_rot,
    R3_sdacs_to_gxlocal,
    T6_block,
    gbcs_to_sdacs,
    sdacs_to_gbcs,
    sdacs_to_uacs,
    uacs_to_sdacs,
)


def _rect_uacs(n: int = 5) -> np.ndarray:
    """Rectangular airfoil: chord [0,1] in col0, thickness 0.1 in col1."""
    chord_frac = np.linspace(0.0, 1.0, n)
    thickness = np.full(n, 0.1)
    return np.column_stack([chord_frac, thickness])


def test_uacs_to_sdacs_x_range():
    uacs = _rect_uacs(11)
    sdacs = uacs_to_sdacs(uacs, chord=2.0)
    assert sdacs[:, 0].min() == pytest.approx(-1.0, abs=1e-12)
    assert sdacs[:, 0].max() == pytest.approx(1.0, abs=1e-12)


def test_uacs_to_sdacs_y_scaled():
    uacs = _rect_uacs(5)
    sdacs = uacs_to_sdacs(uacs, chord=2.0)
    np.testing.assert_allclose(sdacs[:, 1], uacs[:, 1] * 2.0, atol=1e-12)


def test_sdacs_to_gbcs_zero_twist_col_swap():
    """twist=0, dx=0, dy=0, z=10 → pure col-swap, col2=10."""
    sdacs = np.array([[1.0, 2.0], [3.0, 4.0]])
    gbcs = sdacs_to_gbcs(sdacs, twist=0.0, dx=0.0, dy=0.0, z=10.0)
    # col0 = flapwise = old thickness (col1)
    np.testing.assert_allclose(gbcs[:, 0], sdacs[:, 1], atol=1e-12)
    # col1 = edgewise = old chord (col0)
    np.testing.assert_allclose(gbcs[:, 1], sdacs[:, 0], atol=1e-12)
    # col2 = z
    np.testing.assert_allclose(gbcs[:, 2], 10.0, atol=1e-12)


def test_sdacs_to_gbcs_half_pi_twist():
    """twist=90°: hand-computed.

    SDACS point (c, t). After 90° rotation:
      c_rot = c*cos(90°) - t*sin(90°) = -t
      t_rot = c*sin(90°) + t*cos(90°) =  c
      flapwise (col0) = t_rot = c
      edgewise (col1) = c_rot = -t
    """
    sdacs = np.array([[1.0, 0.0], [0.0, 1.0]])
    gbcs = sdacs_to_gbcs(sdacs, twist=90.0, dx=0.0, dy=0.0, z=0.0)
    # row 0: c=1, t=0 → flapwise=1, edgewise=-0=0
    np.testing.assert_allclose(gbcs[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(gbcs[0, 1], 0.0, atol=1e-12)
    # row 1: c=0, t=1 → flapwise=0, edgewise=-1
    np.testing.assert_allclose(gbcs[1, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(gbcs[1, 1], -1.0, atol=1e-12)


def test_sdacs_to_gbcs_dx_shifts_col0():
    sdacs = np.array([[0.0, 0.0]])
    gbcs = sdacs_to_gbcs(sdacs, twist=0.0, dx=1.0, dy=0.0, z=0.0)
    np.testing.assert_allclose(gbcs[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(gbcs[0, 1], 0.0, atol=1e-12)


def test_sdacs_to_gbcs_dy_shifts_col1():
    sdacs = np.array([[0.0, 0.0]])
    gbcs = sdacs_to_gbcs(sdacs, twist=0.0, dx=0.0, dy=1.0, z=0.0)
    np.testing.assert_allclose(gbcs[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(gbcs[0, 1], 1.0, atol=1e-12)


# ── strict GX coordinate-chain helpers ─────────────────────────────────────

def test_R3_axial_rot_is_orthogonal():
    R = R3_axial_rot(0.37)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_R3_axial_rot_zero_is_identity():
    np.testing.assert_allclose(R3_axial_rot(0.0), np.eye(3), atol=1e-12)


def test_R3_about_e1_is_orthogonal():
    R = R3_about_e1(0.42)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_R3_sdacs_to_gxlocal_is_self_inverse_permutation():
    P = R3_sdacs_to_gxlocal()
    # Pure axis rename — all entries are 0 or 1
    assert set(P.flatten().tolist()) <= {0.0, 1.0}
    # Symmetric → its own inverse
    np.testing.assert_allclose(P @ P, np.eye(3), atol=1e-12)


def test_R3_sdacs_to_gxlocal_maps_axes_correctly():
    """Sanity: SDACS [chord, thick, axial] → GX-local [axial, thick, chord]."""
    P = R3_sdacs_to_gxlocal()
    # SDACS chord-axis (1, 0, 0) should land on GX-local position 2 (chord-slot)
    np.testing.assert_allclose(P @ np.array([1.0, 0.0, 0.0]),
                                np.array([0.0, 0.0, 1.0]), atol=1e-12)
    # SDACS thick-axis (0, 1, 0) → GX-local position 1
    np.testing.assert_allclose(P @ np.array([0.0, 1.0, 0.0]),
                                np.array([0.0, 1.0, 0.0]), atol=1e-12)
    # SDACS axial-axis (0, 0, 1) → GX-local position 0
    np.testing.assert_allclose(P @ np.array([0.0, 0.0, 1.0]),
                                np.array([1.0, 0.0, 0.0]), atol=1e-12)


def test_T6_block_is_orthogonal_for_rotation_input():
    R = R3_about_e1(0.27)
    T = T6_block(R)
    np.testing.assert_allclose(T.T @ T, np.eye(6), atol=1e-12)


def test_T6_block_top_and_bottom_independent():
    R = R3_axial_rot(0.5)
    T = T6_block(R)
    # Off-diagonal (3×3) blocks must be zero — forces and moments don't mix
    np.testing.assert_allclose(T[:3, 3:], np.zeros((3, 3)), atol=1e-15)
    np.testing.assert_allclose(T[3:, :3], np.zeros((3, 3)), atol=1e-15)
    np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)
    np.testing.assert_allclose(T[3:, 3:], R, atol=1e-12)


def test_K_anba_to_gxlocal_at_zero_twist_is_strict_perm():
    """Strict block-diagonal perm σ = (2, 1, 0, 5, 4, 3) at twist=0.

    The previously-shipped empirical permutation [2, 1, 0, 5, 3, 4] was
    *not* block-diagonal: forces with e2=thick, moments with e2=chord.
    The strict transform at twist=0 must match the consistent permutation.
    """
    rng = np.random.default_rng(0)
    K_anba = rng.standard_normal((6, 6))
    K_anba = K_anba + K_anba.T  # make symmetric
    K_gx = K_anba_to_gxlocal(K_anba, twist_rad=0.0)

    sigma = np.array([2, 1, 0, 5, 4, 3])
    K_expected = K_anba[np.ix_(sigma, sigma)]
    np.testing.assert_allclose(K_gx, K_expected, atol=1e-12)


def test_K_anba_to_gxlocal_zero_twist_NOT_empirical_perm():
    """Anti-test: confirm the strict transform does NOT reduce to the
    empirical [2, 1, 0, 5, 3, 4] perm — they differ in the bottom-right
    bending block, which is exactly the source of the 4×-soft-edge bug.
    """
    K_anba = np.diag([1.0, 2.0, 3.0, 10.0, 100.0, 1000.0])
    K_gx = K_anba_to_gxlocal(K_anba, twist_rad=0.0)

    sigma_empirical = np.array([2, 1, 0, 5, 3, 4])
    K_empirical = K_anba[np.ix_(sigma_empirical, sigma_empirical)]
    # K_gx[4,4] should be K_anba[4,4]=100 (M_thick → bend about e2_gx=thick).
    # Empirical perm puts K_anba[3,3]=10 there.
    assert K_gx[4, 4] == 100.0
    assert K_empirical[4, 4] == 10.0
    assert not np.allclose(K_gx, K_empirical)


def test_K_anba_to_gxlocal_preserves_symmetry():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 6))
    K_anba = A + A.T  # symmetric
    for twist in [0.0, 0.1, 0.7, -0.3]:
        K_gx = K_anba_to_gxlocal(K_anba, twist_rad=twist)
        np.testing.assert_allclose(K_gx, K_gx.T, atol=1e-12)


def test_K_anba_to_gxlocal_preserves_eigenvalues():
    """The strict chain is an orthogonal similarity transform → eigenvalues
    are invariant. This is the strongest invariant — it catches any
    non-orthogonal mistake in the composition."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 6))
    K_anba = A @ A.T + np.eye(6)  # SPD
    eig_anba = np.sort(np.linalg.eigvalsh(K_anba))
    for twist in [0.0, np.pi / 12, np.pi / 4, -np.pi / 6]:
        K_gx = K_anba_to_gxlocal(K_anba, twist_rad=twist)
        eig_gx = np.sort(np.linalg.eigvalsh(K_gx))
        np.testing.assert_allclose(eig_gx, eig_anba, atol=1e-10)


def test_round_trip():
    """uacs → sdacs → gbcs → inv → uacs agrees to 1e-12."""
    rng = np.random.default_rng(42)
    uacs = np.column_stack([
        rng.uniform(0.0, 1.0, 20),
        rng.uniform(-0.1, 0.1, 20),
    ])
    chord = rng.uniform(1.0, 6.0)
    twist = rng.uniform(-15.0, 15.0)  # degrees
    dx = rng.uniform(-1.0, 1.0)
    dy = rng.uniform(-1.0, 1.0)
    z = rng.uniform(0.0, 60.0)

    sdacs = uacs_to_sdacs(uacs, chord)
    gbcs = sdacs_to_gbcs(sdacs, twist, dx, dy, z)
    sdacs2 = gbcs_to_sdacs(gbcs, twist, dx, dy)
    uacs2 = sdacs_to_uacs(sdacs2, chord)

    np.testing.assert_allclose(uacs2, uacs, atol=1e-12)
