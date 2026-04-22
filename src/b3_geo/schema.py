"""W vtk schema declaration and validator."""

from __future__ import annotations

import pyvista as pv

W_VTK_POINT_DATA: dict[str, dict] = {
    "geo.abs_t": {"shape": "(N,)",   "dtype": "float64"},
    "geo.rel_t": {"shape": "(N,)",   "dtype": "float64"},
    # Derived TE-distance fields. arc_from_te_ss == abs_t (alias for clarity);
    # arc_from_te_ps == total_arc - abs_t (arc length going the other way);
    # t_from_te_* are the normalized [0,1] versions.
    "geo.arc_from_te_ss": {"shape": "(N,)", "dtype": "float64"},
    "geo.arc_from_te_ps": {"shape": "(N,)", "dtype": "float64"},
    "geo.t_from_te_ss":   {"shape": "(N,)", "dtype": "float64"},
    "geo.t_from_te_ps":   {"shape": "(N,)", "dtype": "float64"},
    "geo.uacs":  {"shape": "(N, 2)", "dtype": "float64"},
    "geo.sdacs": {"shape": "(N, 2)", "dtype": "float64"},
    # per-section scalars broadcast to point_data so ParaView can colour by them
    "geo.s":     {"shape": "(N,)",   "dtype": "float64"},
    "geo.z":     {"shape": "(N,)",   "dtype": "float64"},
    "geo.chord": {"shape": "(N,)",   "dtype": "float64"},
    "geo.twist": {"shape": "(N,)",   "dtype": "float64"},
    "geo.tc":    {"shape": "(N,)",   "dtype": "float64"},
    "geo.dx":    {"shape": "(N,)",   "dtype": "float64"},
    "geo.dy":    {"shape": "(N,)",   "dtype": "float64"},
}

W_VTK_FIELD_DATA: dict[str, dict] = {
    "geo.s":     {"shape": "(K,)", "dtype": "float64"},
    "geo.z":     {"shape": "(K,)", "dtype": "float64"},
    "geo.chord": {"shape": "(K,)", "dtype": "float64"},
    "geo.twist": {"shape": "(K,)", "dtype": "float64"},
    "geo.tc":    {"shape": "(K,)", "dtype": "float64"},
    "geo.dx":    {"shape": "(K,)", "dtype": "float64"},
    "geo.dy":    {"shape": "(K,)", "dtype": "float64"},
}


def validate_w_vtk(w: pv.PolyData) -> None:
    """Raise AssertionError if the W vtk does not conform to schema.

    Checks:
    - points is float64, shape (N, 3)
    - every W_VTK_POINT_DATA entry present with correct shape/dtype
    - every W_VTK_FIELD_DATA entry present with correct shape/dtype
    - N consistent with cell connectivity
    - K (field_data length) matches number of polylines
    """
    import numpy as np

    pts = w.points
    assert pts.dtype == np.float64, f"points dtype {pts.dtype!r} != float64"
    assert pts.ndim == 2 and pts.shape[1] == 3, f"points shape {pts.shape} != (N, 3)"
    n_pts = pts.shape[0]

    # count polylines and total points declared in cells
    lines = w.lines
    i = 0
    n_polylines = 0
    n_pts_in_cells = 0
    while i < len(lines):
        count = int(lines[i])
        n_pts_in_cells += count
        n_polylines += 1
        i += count + 1
    assert n_pts_in_cells == n_pts, (
        f"cell connectivity declares {n_pts_in_cells} point refs, "
        f"but poly.points has {n_pts}"
    )

    k = n_polylines

    # point_data
    for key, spec in W_VTK_POINT_DATA.items():
        assert key in w.point_data, f"point_data missing '{key}'"
        arr = w.point_data[key]
        assert arr.dtype == np.float64, f"{key} dtype {arr.dtype!r} != float64"
        if spec["shape"] == "(N,)":
            assert arr.shape == (n_pts,), f"{key} shape {arr.shape} != ({n_pts},)"
        elif spec["shape"] == "(N, 2)":
            assert arr.shape == (n_pts, 2), f"{key} shape {arr.shape} != ({n_pts}, 2)"

    # field_data
    for key, spec in W_VTK_FIELD_DATA.items():
        assert key in w.field_data, f"field_data missing '{key}'"
        arr = w.field_data[key]
        assert arr.dtype == np.float64, f"{key} dtype {arr.dtype!r} != float64"
        assert arr.shape == (k,), f"{key} shape {arr.shape} != ({k},)"
