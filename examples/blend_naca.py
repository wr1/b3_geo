#!/usr/bin/env python3
"""Example script for blending three NACA airfoils using the b3_geo API."""

import numpy as np
from pathlib import Path
from b3_geo.models import Planform, Airfoil, BladeConfig
from b3_geo.core.blade import Blade
from b3_geo.utils.cache import save_blade_sections
from b3_geo.utils import plot_planform
import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False)],
)


def main():
    """Main function to demonstrate blending NACA airfoils."""
    example_dir = Path("examples")

    # Define airfoils with their thicknesses
    airfoils = [
        Airfoil(path=str(example_dir / "naca0030.dat"), name="NACA0030", thickness=0.3),
        Airfoil(
            path=str(example_dir / "naca1224.dat"), name="NACA1224", thickness=0.24
        ),
        Airfoil(
            path=str(example_dir / "naca1418.dat"), name="NACA1418", thickness=0.18
        ),
    ]

    # Create a simple planform with varying thickness
    planform = Planform(
        z=[(0.0, 0.0), (1.0, -100.0)],
        chord=[(0.0, 1.0), (1.0, 0.8)],
        thickness=[
            (0.0, 0.3),
            (0.5, 0.24),
            (1.0, 0.18),
        ],  # Varying thickness to blend airfoils
        twist=[(0.0, 0.0), (1.0, 5.0)],
        dx=[(0.0, 0.0), (1.0, 1.0)],
        dy=[(0.0, 0.0), (1.0, 0.5)],
        npchord=200,
    )

    # Create blade configuration
    blade_config = BladeConfig(planform=planform, airfoils=airfoils)

    # Create Blade instance
    blade = Blade(blade_config)

    # Plot the planform
    controls = {
        "z": planform.z,
        "chord": planform.chord,
        "thickness": planform.thickness,
        "twist": planform.twist,
        "dx": planform.dx,
        "dy": planform.dy,
    }
    interpolated = {
        "rel_span": blade.rel_span,
        "z": blade.z,
        "chord": blade.chord,
        "thickness": blade.thickness,
        "twist": blade.twist,
        "dx": blade.dx,
        "dy": blade.dy,
        "absolute_thickness": blade.absolute_thickness,
    }
    planform_plot_file = example_dir / "planform.png"
    plot_planform(interpolated, controls, blade.rel_span, str(planform_plot_file))
    logger.info(f"Planform plot saved to {planform_plot_file}")

    # Plot the blended airfoils
    thicknesses = np.linspace(0.18, 0.3, 5)
    airfoil_plot_file = example_dir / "blended_naca.png"
    blade.plot_airfoils(thicknesses, str(airfoil_plot_file))
    logger.info(f"Airfoil plot saved to {airfoil_plot_file}")

    # Compute and save blade sections to VTP
    sections = blade.get_sections(blade.rel_span)
    vtp_file = example_dir / "blended_naca_blade.vtp"
    save_blade_sections(blade, str(vtp_file))
    logger.info(f"Blade sections saved to {vtp_file}")

    logger.info("Example completed successfully.")


if __name__ == "__main__":
    main()
