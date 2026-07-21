from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import interp1d

from b3_geo.utils.interpolation import (
    cubic_interpolate,
    interpolate_airfoil,
    linear_interpolate,
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
            data = load_airfoil(af.path)
            self.airfoils_data[af.name] = {"data": data, "thickness": af.thickness}
        # Precompute interpolation functions for airfoils
        sorted_items = sorted(
            self.airfoils_data.items(), key=lambda kv: kv[1]["thickness"]
        )
        sorted_af = [d for _, d in sorted_items]
        if len(sorted_af) == 0:
            msg = "No airfoils provided"
            raise ValueError(msg)
        self.names_sorted = [name for name, _ in sorted_items]
        self.t_sorted = np.array([d["thickness"] for d in sorted_af])
        interp_data = np.array(
            [interpolate_airfoil(d["data"], self.np_chordwise) for d in sorted_af]
        )
        x_all = interp_data[:, :, 0].T
        y_all = interp_data[:, :, 1].T
        if len(sorted_af) == 1:
            # interp1d needs two distinct knots; a single airfoil is constant
            self.x_interp = self._constant_interp(x_all[:, 0])
            self.y_interp = self._constant_interp(y_all[:, 0])
        else:
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

    @staticmethod
    def _constant_interp(values: np.ndarray):
        """Interpolator returning the same chordwise values for any thickness."""

        def interp(thickness):
            t_arr = np.asarray(thickness)
            if t_arr.ndim == 0:
                return values.copy()
            return np.repeat(values[:, None], t_arr.size, axis=1)

        return interp

    def _interpolate_planform(self):
        """Interpolate planform parameters along the span."""
        self.rel_span = np.linspace(0, 1, 100)
        self.span = self.rel_span * 100
        self.z = linear_interpolate(self.config.planform.z, self.rel_span)
        self.chord = pchip_interpolate(self.config.planform.chord, self.rel_span)
        self.thickness = cubic_interpolate(
            self.config.planform.thickness, self.rel_span, bc_type="natural"
        )
        self.twist = pchip_interpolate(self.config.planform.twist, self.rel_span)
        self.dx = cubic_interpolate(
            self.config.planform.dx, self.rel_span, bc_type="natural"
        )
        self.dy = cubic_interpolate(self.config.planform.dy, self.rel_span)
        self.absolute_thickness = self.chord * self.thickness

    def _interpolate_airfoils(self):
        """Interpolate airfoils across thicknesses using precomputed interpolators."""
        x_span = self.x_interp(self.thickness)
        y_span = self.y_interp(self.thickness)
        return x_span, y_span

    def get_planform_values(self, rel: float) -> dict:
        """Get interpolated planform values at a specific relative span."""
        return {
            "z": linear_interpolate(self.config.planform.z, [rel])[0],
            "chord": pchip_interpolate(self.config.planform.chord, [rel])[0],
            "thickness": cubic_interpolate(
                self.config.planform.thickness, [rel], bc_type="natural"
            )[0],
            "twist": pchip_interpolate(self.config.planform.twist, [rel])[0],
            "dx": cubic_interpolate(self.config.planform.dx, [rel], bc_type="natural")[
                0
            ],
            "dy": cubic_interpolate(self.config.planform.dy, [rel])[0],
        }

    def get_planform_array(self, rels: np.ndarray) -> dict[str, np.ndarray]:
        """Get interpolated planform values for an array of relative spans."""
        result = {
            "z": linear_interpolate(self.config.planform.z, rels),
            "chord": pchip_interpolate(self.config.planform.chord, rels),
            "thickness": cubic_interpolate(
                self.config.planform.thickness, rels, bc_type="natural"
            ),
            "twist": pchip_interpolate(self.config.planform.twist, rels),
            "dx": cubic_interpolate(self.config.planform.dx, rels, bc_type="natural"),
            "dy": cubic_interpolate(self.config.planform.dy, rels),
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
            ax.plot(xy[:, 0], xy[:, 1], label=f"t/c = {t:.2f} — {self.blend_info(t)}")
        ax.set_title("Interpolated Airfoils")
        ax.set_xlabel("x/chord")
        ax.set_ylabel("y/chord")
        ax.legend()
        ax.set_aspect("equal")
        plt.savefig(output_file)
        plt.close()

    def blend_info(self, thickness: float) -> str:
        """Describe how the airfoil at this thickness is blended from the inputs."""
        ts = self.t_sorted
        names = self.names_sorted
        knot = np.isclose(ts, thickness, rtol=0, atol=1e-6)
        if knot.any():
            return f"{names[int(np.argmax(knot))]} (input)"
        if thickness < ts[0]:
            return f"{names[0]} (clamped below t/c = {ts[0]:.2f})"
        if thickness > ts[-1]:
            return f"{names[-1]} (clamped above t/c = {ts[-1]:.2f})"
        i = int(np.searchsorted(ts, thickness))
        w = (thickness - ts[i - 1]) / (ts[i] - ts[i - 1])
        return f"{100 * (1 - w):.0f}% {names[i - 1]} + {100 * w:.0f}% {names[i]}"

    def _bracketing_airfoils(self, thickness: float) -> list[tuple[str, float]]:
        """Input airfoils (name, thickness) bracketing an interpolated thickness.

        Empty when the thickness coincides with an input airfoil or is clamped.
        """
        ts = self.t_sorted
        if np.isclose(ts, thickness, rtol=0, atol=1e-6).any():
            return []
        if thickness < ts[0] or thickness > ts[-1]:
            return []
        i = int(np.searchsorted(ts, thickness))
        return [
            (self.names_sorted[i - 1], float(ts[i - 1])),
            (self.names_sorted[i], float(ts[i])),
        ]

    def get_te_thickness(self, thickness: float | np.ndarray) -> float | np.ndarray:
        """Trailing-edge gap (chord fraction) of interpolated airfoil(s).

        Distance between the first and last contour points (Selig convention:
        the contour starts and ends at the trailing edge).
        """
        xy = self.get_airfoil_xy_norm(thickness)
        gap = np.linalg.norm(xy[0] - xy[-1], axis=-1)
        if np.isscalar(thickness):
            return float(gap)
        return gap

    def plot_te_thickness(self, output_file: str, n_sweep: int = 200):
        """Plot trailing-edge thickness vs relative thickness of the blend."""
        from b3_geo.utils.plotting import plot_te_thickness

        t_sweep = np.linspace(self.t_sorted[0], self.t_sorted[-1], n_sweep)
        inputs = {
            name: (float(t), float(self.get_te_thickness(float(t))))
            for name, t in zip(self.names_sorted, self.t_sorted)
        }
        # Zoom on the lower cluster when the thickest input sits far above it
        zoom_tc = None
        if len(self.t_sorted) >= 3 and self.t_sorted[-1] > 1.5 * self.t_sorted[-2]:
            zoom_tc = 1.2 * float(self.t_sorted[-2])
        plot_te_thickness(
            t_sweep, self.get_te_thickness(t_sweep), inputs, output_file, zoom_tc
        )

    def plot_airfoil_curvature(
        self, thicknesses: np.ndarray, output_file: str, max_tooth: float = 0.15
    ):
        """Plot curvature of interpolated airfoils at given thicknesses."""
        from b3_geo.utils.plotting import plot_airfoil_curvature

        sections = {}
        references = {}
        for t in thicknesses:
            t = float(t)
            label = f"t/c = {t:.2f} — {self.blend_info(t)}"
            sections[label] = self.get_airfoil_xy_norm(t)
            parents = self._bracketing_airfoils(t)
            if parents:
                references[label] = {
                    f"{name} (t/c = {tk:.2f})": self.get_airfoil_xy_norm(tk)
                    for name, tk in parents
                }
        plot_airfoil_curvature(
            sections, output_file, max_tooth=max_tooth, references=references
        )

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

    def z_to_rel(self, z_val: float | np.ndarray) -> float | np.ndarray:
        """Convert absolute z to relative span."""
        rels, zs = zip(*self.config.planform.z)
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
