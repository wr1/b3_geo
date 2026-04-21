"""Field computers — produce spanwise/chordwise field arrays for b3_drp2."""

from __future__ import annotations

import numpy as np
import pandas as pd

FIELD_COLUMNS = [
    "s",
    "z",
    "chord_frac",
    "panel_id",
    "dist_from_le",
    "dist_from_te",
    "suction_side",
    "pressure_side",
]


def build_fields(
    sections: np.ndarray,
    spans: np.ndarray,
    zs: np.ndarray,
    web_positions: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build field DataFrame matching b3_drp2's input schema.

    sections: (N_spans, N_chord, 3)
    spans:    (N_spans,) relative span values
    zs:       (N_spans,) absolute z values
    web_positions: {web_name: (N_spans,) chord_frac array} for abs_dist_<web> columns

    Returns DataFrame with one row per (span, chord) point.
    """
    n_spans, n_chord, _ = sections.shape
    total = n_spans * n_chord

    s_vals = np.repeat(spans, n_chord)
    z_vals = np.repeat(zs, n_chord)
    chord_frac = np.tile(np.linspace(0, 1, n_chord), n_spans)
    panel_id = np.tile(np.arange(n_chord), n_spans)

    dist_le = chord_frac.copy()
    dist_te = 1.0 - chord_frac

    # Suction/pressure side: by convention, first half is suction (upper), second pressure
    midpoint = n_chord // 2
    suction = np.tile(
        np.r_[np.ones(midpoint), np.zeros(n_chord - midpoint)].astype(bool),
        n_spans,
    )

    data: dict[str, np.ndarray] = {
        "s": s_vals,
        "z": z_vals,
        "chord_frac": chord_frac,
        "panel_id": panel_id,
        "dist_from_le": dist_le,
        "dist_from_te": dist_te,
        "suction_side": suction,
        "pressure_side": ~suction,
    }

    if web_positions:
        for web_name, web_cf in web_positions.items():
            web_cf_rep = np.repeat(web_cf, n_chord)
            col = f"abs_dist_{web_name}"
            data[col] = np.abs(chord_frac - web_cf_rep)

    return pd.DataFrame(data)


def dist_from_le(sections: np.ndarray) -> np.ndarray:
    """Chordwise distance from LE as fraction [0, 1]. Shape: (N_spans, N_chord)."""
    n_chord = sections.shape[1]
    return np.tile(np.linspace(0, 1, n_chord), (sections.shape[0], 1))


def dist_from_te(sections: np.ndarray) -> np.ndarray:
    """Chordwise distance from TE as fraction [0, 1]. Shape: (N_spans, N_chord)."""
    return 1.0 - dist_from_le(sections)


def abs_dist_web(sections: np.ndarray, web_chord_frac: float | np.ndarray) -> np.ndarray:
    """Absolute chordwise distance from a web at web_chord_frac.

    sections: (N_spans, N_chord, 3)
    web_chord_frac: scalar or (N_spans,) array
    Returns: (N_spans, N_chord)
    """
    n_chord = sections.shape[1]
    cf = dist_from_le(sections)
    if np.ndim(web_chord_frac) == 0:
        return np.abs(cf - float(web_chord_frac))
    return np.abs(cf - np.asarray(web_chord_frac)[:, None])
