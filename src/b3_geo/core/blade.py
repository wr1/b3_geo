from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import interp1d

from b3_geo.utils.interpolation import (
    airfoil_from_coordinates,
    eval_curve,
    interpolate_airfoil,
    load_airfoil,
    pchip_interpolate,
)

if TYPE_CHECKING:
    from b3_geo.models import BladeConfig


class Blade:
    """Represents a blade with interpolated planform and airfoils."""

    def __init__(self, config: BladeConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.np_chordwise = self.config.planform.npchord
        self._interpolate_planform()
        self.airfoils_data: dict[str, dict] = {}
        for af in self.config.airfoils:
            if af.coordinates is not None:
                data = airfoil_from_coordinates(af.coordinates)
            else:
                data = load_airfoil(af.path)
            self.airfoils_data[af.name] = {"data": data, "thickness": af.thickness}
        # Precompute interpolation functions for airfoils
        sorted_af = sorted(self.airfoils_data.values(), key=lambda d: d["thickness"])
        if len(sorted_af) == 0:
            msg = "No airfoils provided"
            raise ValueError(msg)
        self.t_sorted = np.array([d["thickness"] for d in sorted_af])
        interp_data = np.array(
            [interpolate_airfoil(d["data"], self.np_chordwise) for d in sorted_af]
        )
        x_all = interp_data[:, :, 0].T
        y_all = interp_data[:, :, 1].T
        self.x_interp = interp1d(
            self.t_sorted,
            x_all,
            axis=1,
            bounds_error=False,
            fill_value=(x_all[:, 0], x_all[:, -1]),
        )
        self.y_interp = interp1d(
            self.t_sorted,
            y_all,
            axis=1,
            bounds_error=False,
            fill_value=(y_all[:, 0], y_all[:, -1]),
        )

    def _interpolate_planform(self):
        """Interpolate planform parameters along the span."""
        self.rel_span = np.linspace(0, 1, 100)
        self.span = self.rel_span * 100
        self.z = eval_curve(
            self.config.planform.z, self.rel_span, default_method="linear"
        )
        self.chord = eval_curve(self.config.planform.chord, self.rel_span)
        self.thickness = eval_curve(
            self.config.planform.thickness,
            self.rel_span,
            default_method="cubic_natural",
        )
        self.twist = eval_curve(self.config.planform.twist, self.rel_span)
        self.dx = eval_curve(
            self.config.planform.dx, self.rel_span, default_method="cubic_natural"
        )
        self.dy = eval_curve(self.config.planform.dy, self.rel_span)
        self.absolute_thickness = self.chord * self.thickness

    def _interpolate_airfoils(self):
        """Interpolate airfoils across thicknesses using precomputed interpolators."""
        x_span = self.x_interp(self.thickness)
        y_span = self.y_interp(self.thickness)
        return x_span, y_span

    def get_planform_values(self, rel: float) -> dict:
        """Get interpolated planform values at a specific relative span."""
        rels = np.array([rel])
        return {
            "z": eval_curve(self.config.planform.z, rels, default_method="linear")[0],
            "chord": eval_curve(self.config.planform.chord, rels)[0],
            "thickness": eval_curve(
                self.config.planform.thickness, rels, default_method="cubic_natural"
            )[0],
            "twist": eval_curve(self.config.planform.twist, rels)[0],
            "dx": eval_curve(
                self.config.planform.dx, rels, default_method="cubic_natural"
            )[0],
            "dy": eval_curve(self.config.planform.dy, rels)[0],
        }

    def get_planform_array(self, rels: np.ndarray) -> dict[str, np.ndarray]:
        """Get interpolated planform values for an array of relative spans."""
        result = {
            "z": eval_curve(self.config.planform.z, rels, default_method="linear"),
            "chord": eval_curve(self.config.planform.chord, rels),
            "thickness": eval_curve(
                self.config.planform.thickness, rels, default_method="cubic_natural"
            ),
            "twist": eval_curve(self.config.planform.twist, rels),
            "dx": eval_curve(
                self.config.planform.dx, rels, default_method="cubic_natural"
            ),
            "dy": eval_curve(self.config.planform.dy, rels),
        }
        result["absolute_thickness"] = result["chord"] * result["thickness"]
        return result

    def get_airfoil_xy_norm(self, thickness: float | np.ndarray) -> np.ndarray:
        """Get normalized airfoil coordinates at specific thickness(es) using precomputed interpolators."""
        x = self.x_interp(thickness)
        y = self.y_interp(thickness)
        if np.isscalar(thickness):
            return np.column_stack((x, y))
        return np.dstack((x, y))  # (np_chordwise, n, 2)

    def plot_airfoils(self, thicknesses: np.ndarray, output_file: str):
        """Plot interpolated airfoils at given thicknesses."""
        _fig, ax = plt.subplots(figsize=(10, 8))
        for t in thicknesses:
            xy = self.get_airfoil_xy_norm(t)
            ax.plot(xy[:, 0], xy[:, 1], label=f"Thickness {t:.2f}")
        ax.set_title("Interpolated Airfoils")
        ax.set_xlabel("x/chord")
        ax.set_ylabel("y/chord")
        ax.legend()
        ax.set_aspect("equal")
        plt.savefig(output_file)
        plt.close()

    def get_sections(self, rels: np.ndarray) -> np.ndarray:
        """Compute positioned and rotated airfoil sections at given relative spans using PyVista operations."""
        vals = self.get_planform_array(rels)
        xy_norm = self.get_airfoil_xy_norm(vals["thickness"])  # (chord, n, 2)
        points_list = []
        for i in range(len(rels)):
            xy = xy_norm[:, i, :]  # (chord, 2)
            points_2d = np.column_stack((xy, np.zeros(self.np_chordwise)))  # (chord, 3)
            poly = pv.PolyData(points_2d)
            # Translate to twist center at 0
            poly.translate([0, -0.5, 0], inplace=True)
            # Scale by chord
            poly.scale([vals["chord"][i], vals["chord"][i], 1], inplace=True)
            # Rotate by twist around z-axis
            poly.rotate_z(-vals["twist"][i], inplace=True)
            # Translate by dx, dy, z
            poly.translate([vals["dx"][i], vals["dy"][i], vals["z"][i]], inplace=True)
            points_list.append(poly.points)
        return np.array(points_list)  # (n, chord, 3)

    def _get_sections_np(self, rels: np.ndarray, dy_override: np.ndarray) -> np.ndarray:
        """Build section polygons in 3D blade space with a custom dy array (pure numpy)."""
        vals = self.get_planform_array(rels)
        xy_norm = self.get_airfoil_xy_norm(vals["thickness"])  # (n_chord, n_span, 2)
        n = len(rels)
        sections = np.zeros((n, self.np_chordwise, 3))
        for i in range(n):
            xy = xy_norm[:, i, :]
            pts = np.column_stack((xy, np.zeros(self.np_chordwise)))
            pts[:, 1] -= 0.5
            pts[:, :2] *= vals["chord"][i]
            angle = -vals["twist"][i] * np.pi / 180.0
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = pts[:, 0] * cos_a - pts[:, 1] * sin_a
            y_rot = pts[:, 0] * sin_a + pts[:, 1] * cos_a
            pts[:, 0] = x_rot + vals["dx"][i]
            pts[:, 1] = y_rot + dy_override[i]
            pts[:, 2] = vals["z"][i]
            sections[i] = pts
        return sections

    @staticmethod
    def _intersect_polygon_plane(
        pts: np.ndarray, normal: np.ndarray, plane_point: np.ndarray
    ) -> list:
        """Return up to 2 intersection points of a closed polygon with a plane."""
        dists = (pts - plane_point) @ normal
        result = []
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            d0, d1 = dists[i], dists[j]
            if d0 * d1 < 0:
                t = d0 / (d0 - d1)
                result.append(pts[i] + t * (pts[j] - pts[i]))
                if len(result) == 2:
                    break
        return result

    def get_optimal_dy_for_web(
        self,
        structure: dict,
        web_index: int = 0,
        n_span: int = 40,
        n_dy_trials: int = 41,
        dy_halfwidth: float = 2.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (rel_span, optimal_dy) maximising spar-cap separation for the given web.

        Sweeps dy at each span station, intersects the section polygon with the
        (fixed, untwisted) web plane, and picks the dy giving maximum height.
        """
        webs = structure.get("webs", [])
        if web_index >= len(webs):
            msg = f"web_index {web_index} out of range ({len(webs)} webs)"
            raise ValueError(msg)
        web_def = webs[web_index]
        origin = np.array(web_def["origin"], dtype=float)
        orientation = web_def.get("orientation") or web_def.get("normal")
        if orientation is None:
            raise ValueError("Web has no orientation/normal defined")
        normal = np.array(orientation, dtype=float)
        normal /= np.linalg.norm(normal)

        rel_span = np.linspace(0.05, 0.98, n_span)
        current_dy = eval_curve(self.config.planform.dy, rel_span)
        dy_deltas = np.linspace(-dy_halfwidth, dy_halfwidth, n_dy_trials)

        heights = np.full((n_span, n_dy_trials), np.nan)
        for j, delta in enumerate(dy_deltas):
            sections = self._get_sections_np(rel_span, current_dy + delta)
            for i in range(n_span):
                pts = self._intersect_polygon_plane(sections[i], normal, origin)
                if len(pts) == 2:
                    heights[i, j] = float(np.linalg.norm(pts[0] - pts[1]))

        valid_rels, valid_opt = [], []
        for i in range(n_span):
            row = heights[i]
            if np.any(~np.isnan(row)):
                best_j = int(np.nanargmax(row))
                valid_rels.append(rel_span[i])
                valid_opt.append(current_dy[i] + dy_deltas[best_j])

        if len(valid_rels) < 2:
            return rel_span, current_dy

        smoothed = pchip_interpolate(list(zip(valid_rels, valid_opt)), rel_span)
        return rel_span, smoothed

    def z_to_rel(self, z_val: float | np.ndarray) -> float | np.ndarray:
        """Convert absolute z to relative span."""
        rels, zs = zip(*self.config.planform.z.points)
        sort_idx = np.argsort(zs)
        zs_sorted = np.array(zs)[sort_idx]
        rels_sorted = np.array(rels)[sort_idx]
        interp = interp1d(
            zs_sorted, rels_sorted, kind="linear", fill_value="extrapolate"
        )
        res = interp(z_val)
        if isinstance(z_val, (float, int)):
            return float(res)
        return res
