import numpy as np
from b3_geo.models import BladeConfig
from b3_geo.utils.interpolation import (
    linear_interpolate,
    cubic_interpolate,
    pchip_interpolate,
    load_airfoil,
    interpolate_airfoil,
)
from scipy.interpolate import interp1d
from typing import Dict
import logging


class Blade:
    """Represents a blade with interpolated planform and airfoils."""

    def __init__(self, config: BladeConfig):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.np_chordwise = self.config.planform.npchord
        self.np_spanwise = self.config.planform.npspan
        self._interpolate_planform()
        self.airfoils_data: Dict[str, Dict] = {}
        for af in self.config.airfoils:
            data = load_airfoil(af.path)
            self.airfoils_data[af.name] = {"data": data, "thickness": af.thickness}
        # Precompute interpolation functions for airfoils
        sorted_af = sorted(self.airfoils_data.values(), key=lambda d: d["thickness"])
        if len(sorted_af) == 0:
            raise ValueError("No airfoils provided")
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
        self.rel_span = np.linspace(0, 1, self.np_spanwise)
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

    def get_planform_values(self, rel: float) -> Dict:
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

    def get_planform_array(self, rels: np.ndarray) -> Dict[str, np.ndarray]:
        """Get interpolated planform values for an array of relative spans."""
        return {
            "z": linear_interpolate(self.config.planform.z, rels),
            "chord": pchip_interpolate(self.config.planform.chord, rels),
            "thickness": cubic_interpolate(
                self.config.planform.thickness, rels, bc_type="natural"
            ),
            "twist": pchip_interpolate(self.config.planform.twist, rels),
            "dx": cubic_interpolate(self.config.planform.dx, rels, bc_type="natural"),
            "dy": cubic_interpolate(self.config.planform.dy, rels),
        }

    def get_airfoil_xy_norm(self, thickness: float | np.ndarray) -> np.ndarray:
        """Get normalized airfoil coordinates at specific thickness(es) using precomputed interpolators."""
        x = self.x_interp(thickness)
        y = self.y_interp(thickness)
        if np.isscalar(thickness):
            return np.column_stack((x, y))
        else:
            return np.dstack((x, y))  # (np_chordwise, n, 2)

    def get_sections(self, rels: np.ndarray = None) -> np.ndarray:
        """Compute positioned and rotated airfoil sections at given relative spans."""
        if rels is None:
            rels = self.rel_span
        vals = self.get_planform_array(rels)
        xy_norm = self.get_airfoil_xy_norm(vals["thickness"])  # (chord, n, 2)
        twist_center = 0.5
        xy = xy_norm.copy()
        xy[..., 0] -= twist_center
        xy *= vals["chord"][None, :, None]
        thetas = np.deg2rad(vals["twist"] + self.config.planform.pre_rotation)
        cos_t = np.cos(thetas)[None, :]
        sin_t = np.sin(thetas)[None, :]
        rot_x = cos_t * xy[..., 0] - sin_t * xy[..., 1]
        rot_y = sin_t * xy[..., 0] + cos_t * xy[..., 1]
        xy[..., 0] = rot_x + vals["dx"][None, :]
        xy[..., 1] = rot_y + vals["dy"][None, :]
        points = np.zeros((self.np_chordwise, len(rels), 3))
        points[..., :2] = xy
        points[..., 2] = vals["z"][None, :]
        return points.transpose(1, 0, 2)  # (n, chord, 3)

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
