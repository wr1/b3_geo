from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.interpolate import CubicSpline, PchipInterpolator

from .curvature import arc_length, curvature, surface_normals
from .interpolation import interpolate_airfoil


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


def plot_airfoil_curvature(
    sections: dict[str, np.ndarray],
    output_file: str,
    max_tooth: float = 0.15,
    references: dict[str, dict[str, np.ndarray]] | None = None,
):
    """Plot curvature of airfoil contours: comb quiver + kappa(s) distribution.

    One row per section: left panel shows the contour with a curvature comb
    (tooth length proportional to |kappa|, colored by signed kappa), right
    panel shows kappa vs normalised arc length. The color scale is shared
    across all sections so interpolated shapes are directly comparable.

    Parameters
    ----------
    sections:
        Mapping of label -> (N, 2) contour coordinates in the b3_geo internal
        convention (column 0 flapwise, column 1 chordwise, normalized to
        chord). Panels are drawn with the chord horizontal.
    output_file:
        Path of the figure to write.
    max_tooth:
        Length of the tallest comb tooth as a fraction of chord.
    references:
        Optional mapping of section label -> {name: (N, 2) contour} of
        reference shapes (e.g. the input airfoils an interpolation blends
        between), drawn as dashed outlines behind the section contour.
    """
    labels = list(sections.keys())
    # Display frame: chord on the horizontal axis
    sections = {name: xy[:, ::-1] for name, xy in sections.items()}
    kappas = {name: curvature(xy) for name, xy in sections.items()}
    all_kappa = np.concatenate(list(kappas.values()))
    all_kappa = all_kappa[np.isfinite(all_kappa)]
    if all_kappa.size == 0:
        all_kappa = np.array([1.0])
    klim = float(np.percentile(np.abs(all_kappa), 97)) or 1.0
    kmax = float(np.abs(all_kappa).max()) or 1.0
    scale = max_tooth / kmax
    norm = mcolors.TwoSlopeNorm(vmin=-klim, vcenter=0.0, vmax=klim)
    cmap = plt.cm.coolwarm

    n_rows = len(labels)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(13, 3.2 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    for i, name in enumerate(labels):
        xy = sections[name]
        kappa = kappas[name]
        normals = surface_normals(xy)
        s = arc_length(xy)
        s_norm = s / s[-1] if s[-1] > 0 else s

        # Teeth point away from the center of curvature (outward on convex arcs)
        u = -kappa * normals[:, 0] * scale
        v = -kappa * normals[:, 1] * scale

        ax1, ax2 = axes[i]
        for ref_name, ref_xy in ((references or {}).get(name) or {}).items():
            ax1.plot(
                ref_xy[:, 1],
                ref_xy[:, 0],
                ls="--",
                lw=0.8,
                alpha=0.7,
                zorder=2.5,
                label=ref_name,
            )
        ax1.plot(xy[:, 0], xy[:, 1], color="#333333", lw=1.0, zorder=3)
        if (references or {}).get(name):
            ax1.legend(fontsize=7, loc="upper right")
        segs = [
            [[x0, y0], [x0 + du, y0 + dv]]
            for x0, y0, du, dv in zip(xy[:, 0], xy[:, 1], u, v)
        ]
        lc = LineCollection(segs, cmap=cmap, norm=norm, lw=0.7, zorder=2)
        lc.set_array(kappa)
        ax1.add_collection(lc)
        ax1.plot(xy[:, 0] + u, xy[:, 1] + v, "-", color="#888888", lw=0.5, zorder=1)
        ax1.set_aspect("equal")
        ax1.set_xlabel("x/chord")
        ax1.set_ylabel("y/chord")
        ax1.set_title(name)

        pts = np.column_stack([s_norm, kappa]).reshape(-1, 1, 2)
        segs2 = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc2 = LineCollection(segs2, cmap=cmap, norm=norm, lw=1.2)
        lc2.set_array(0.5 * (kappa[:-1] + kappa[1:]))
        ax2.add_collection(lc2)
        ax2.axhline(0, color="#444444", lw=0.6, ls="--")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-1.5 * klim, 1.5 * klim)
        ax2.set_xlabel("s / S (normalised arc length)")
        ax2.set_ylabel("κ (1/chord)")
        ax2.set_title("κ(s)")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(
        mappable, ax=axes.ravel().tolist(), label="κ (1/chord)", shrink=0.6
    )
    plt.savefig(output_file)
    plt.close()


def plot_te_thickness(
    t_sweep: np.ndarray,
    te_gap: np.ndarray,
    inputs: dict[str, tuple[float, float]],
    output_file: str,
    zoom_tc: float | None = None,
):
    """Plot trailing-edge thickness against relative thickness of the blend.

    Parameters
    ----------
    t_sweep:
        Relative thicknesses (t/c) the blend was evaluated at.
    te_gap:
        Trailing-edge gap (chord fraction) at each t_sweep value.
    inputs:
        Mapping of input airfoil name -> (t/c, TE gap), drawn as markers.
    output_file:
        Path of the figure to write.
    zoom_tc:
        If given, add a second panel zoomed to t/c <= zoom_tc.
    """

    def draw(ax, t_max=None):
        mask = slice(None) if t_max is None else t_sweep <= t_max
        ax.plot(t_sweep[mask], te_gap[mask], label="interpolated", zorder=2)
        shown = {
            name: (t, gap)
            for name, (t, gap) in inputs.items()
            if t_max is None or t <= t_max
        }
        if shown:
            t_in, gap_in = zip(*shown.values())
            ax.scatter(t_in, gap_in, color="orange", zorder=3, label="input airfoils")
            for name, (t, gap) in shown.items():
                ax.annotate(
                    name,
                    (t, gap),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                )
        ax.set_xlabel("t/c (relative thickness)")
        ax.set_ylabel("TE thickness / chord")
        ax.grid(True, alpha=0.4)
        ax.legend()

    n_panels = 2 if zoom_tc is not None else 1
    fig, axs = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6), squeeze=False)
    draw(axs[0, 0])
    axs[0, 0].set_title("Trailing-edge thickness of interpolated airfoils")
    if zoom_tc is not None:
        draw(axs[0, 1], t_max=zoom_tc)
        axs[0, 1].set_title(f"Zoom: t/c ≤ {zoom_tc:.2f}")
    fig.tight_layout()
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
