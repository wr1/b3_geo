from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator

from .interpolation import interpolate_airfoil

if TYPE_CHECKING:
    import numpy as np


def plot_airfoils(airfoils_data: dict[str, dict], n_points: int, output_file: str):
    """Plot all interpolated airfoils in a single matplotlib plot."""
    _fig, ax = plt.subplots(figsize=(10, 8))
    for name, info in airfoils_data.items():
        interp_data = interpolate_airfoil(info["data"], n_points)
        ax.plot(interp_data[:, 0], interp_data[:, 1], label=name)
    ax.set_title("Interpolated Airfoils")
    ax.set_xlabel("x/chord")
    ax.set_ylabel("y/chord")
    ax.legend()
    plt.savefig(output_file)
    plt.close()


def plot_planform(
    interpolated: dict[str, np.ndarray],
    controls: dict[str, list[tuple[float, float]]],
    rel_span: np.ndarray,
    output_file: str,
):
    """Plots interpolated planform parameters."""
    _fig, axs = plt.subplots(4, 2, figsize=(12, 16))
    axs = axs.flatten()
    params = interpolated
    for i, (name, values) in enumerate(params.items()):
        if i >= len(axs):
            break
        axs[i].plot(
            rel_span, values, label="Interpolated", alpha=0.7, marker=".", markersize=2
        )
        if controls.get(name):
            ctrl_xs, ctrl_ys = zip(*sorted(controls[name]))
            axs[i].scatter(ctrl_xs, ctrl_ys, color="orange", label="Control Points")
        elif (
            name == "absolute_thickness"
            and "chord" in controls
            and "thickness" in controls
        ):
            chord_points = sorted(controls["chord"])
            thick_points = sorted(controls["thickness"])
            chord_xs, chord_ys = zip(*chord_points)
            thick_xs, thick_ys = zip(*thick_points)
            all_xs = sorted(set(chord_xs).union(thick_xs))
            chord_int = PchipInterpolator(chord_xs, chord_ys)
            thick_int = CubicSpline(thick_xs, thick_ys, bc_type="natural")
            ctrl_ys = [chord_int(x) * thick_int(x) for x in all_xs]
            axs[i].scatter(
                all_xs, ctrl_ys, color="orange", label="Derived Control Points"
            )
        axs[i].set_title(name.capitalize())
        axs[i].set_xlabel("Relative Span")
        axs[i].set_ylabel(name.capitalize())
        axs[i].legend()
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
