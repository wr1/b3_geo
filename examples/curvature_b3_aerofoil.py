#!/usr/bin/env python3
"""Interpolate between b3_aerofoil airfoils and plot curvature of the blends.

Loads airfoil coordinates from the sibling ../b3_aerofoil repository, blends
them across relative thickness with the Blade interpolators, and writes a
curvature comb + kappa(s) figure for the interpolated sections.
"""

import logging
from pathlib import Path

import numpy as np
from rich.logging import RichHandler

from b3_geo.core.blade import Blade
from b3_geo.models import Airfoil, BladeConfig, Planform

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False)],
)

AEROFOIL_DIR = (
    Path(__file__).resolve().parents[2]
    / "b3_aerofoil"
    / "src"
    / "b3_aerofoil"
    / "data"
    / "airfoils"
)


def main():
    """Blend b3_aerofoil airfoils and plot curvature of the interpolations."""
    example_dir = Path(__file__).resolve().parent

    # Airfoils from ../b3_aerofoil, thin tip section to root cylinder
    airfoils = [
        Airfoil(
            path=str(AEROFOIL_DIR / "naca0017.dat"), name="NACA 0017", thickness=0.17
        ),
        Airfoil(
            path=str(AEROFOIL_DIR / "du93_w_210.dat"),
            name="DU 93-W-210",
            thickness=0.21,
        ),
        Airfoil(
            path=str(AEROFOIL_DIR / "du91_w2_250.dat"),
            name="DU 91-W2-250",
            thickness=0.25,
        ),
        Airfoil(
            path=str(AEROFOIL_DIR / "circ_m32.dat"), name="Cylinder", thickness=1.0
        ),
    ]

    # Simple blade planform: cylinder at root blending to thin tip
    planform = Planform(
        z=[(0.0, 0.0), (1.0, -100.0)],
        chord=[(0.0, 1.0), (0.25, 1.4), (1.0, 0.6)],
        thickness=[(0.0, 1.0), (0.25, 0.4), (0.6, 0.21), (1.0, 0.17)],
        twist=[(0.0, 0.0), (1.0, 5.0)],
        dx=[(0.0, 0.0), (1.0, 0.5)],
        dy=[(0.0, 0.0), (1.0, 0.2)],
        npchord=200,
    )

    blade = Blade(BladeConfig(planform=planform, airfoils=airfoils))

    # Inputs (0.17, 0.21, 0.25, 1.0) plus blends between them
    thicknesses = np.array([0.17, 0.19, 0.21, 0.23, 0.25, 0.45, 0.7, 1.0])
    shape_plot = example_dir / "b3_aerofoil_blends.png"
    blade.plot_airfoils(thicknesses, str(shape_plot))
    logger.info(f"Interpolated airfoil plot saved to {shape_plot}")

    # Curvature comb + kappa(s) for each interpolated section
    curvature_plot = example_dir / "b3_aerofoil_curvature.png"
    blade.plot_airfoil_curvature(thicknesses, str(curvature_plot))
    logger.info(f"Curvature plot saved to {curvature_plot}")

    # Trailing-edge thickness across the interpolation range
    te_plot = example_dir / "b3_aerofoil_te_thickness.png"
    blade.plot_te_thickness(str(te_plot))
    logger.info(f"TE thickness plot saved to {te_plot}")

    logger.info("Example completed successfully.")


if __name__ == "__main__":
    main()
