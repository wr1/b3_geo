#!/usr/bin/env python3
"""Example script for interpolating airfoils from a planform using the b3_geo API."""

import numpy as np
from pathlib import Path
from b3_geo.models import Planform, Airfoil, BladeConfig
from b3_geo.core.blade import Blade
from b3_geo.utils.cache import save_blade_sections
import logging
from rich.logging import RichHandler

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(show_time=False)],
)


def main():
    """Main function to demonstrate airfoil interpolation."""
    # Create a temporary directory for example data
    example_dir = Path("examples")
    example_dir.mkdir(exist_ok=True)

    # Create a dummy airfoil file
    airfoil_file = example_dir / "dummy_airfoil.dat"
    airfoil_data = """# Dummy airfoil data
0.000000  0.000000
0.012500  0.005000
0.025000  0.010000
0.050000  0.015000
0.075000  0.020000
0.100000  0.025000
0.150000  0.030000
0.200000  0.035000
0.250000  0.040000
0.300000  0.045000
0.400000  0.050000
0.500000  0.055000
0.600000  0.050000
0.700000  0.045000
0.800000  0.040000
0.900000  0.035000
0.950000  0.030000
0.975000  0.025000
1.000000  0.000000
"""
    airfoil_file.write_text(airfoil_data)

    # Define planform parameters
    planform = Planform(
        z=[(0.0, 0.0), (0.5, -50.0), (1.0, -100.0)],
        chord=[(0.0, 1.0), (0.5, 0.9), (1.0, 0.8)],
        thickness=[(0.0, 0.2), (0.5, 0.18), (1.0, 0.15)],
        twist=[(0.0, 0.0), (0.5, 2.5), (1.0, 5.0)],
        dx=[(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)],
        dy=[(0.0, 0.0), (0.5, 0.25), (1.0, 0.5)],
        pre_rotation=0.0,
        npchord=200,
        npspan=50,
    )

    # Define airfoils
    airfoils = [
        Airfoil(path=str(airfoil_file), name="dummy_af", thickness=0.2),
    ]

    # Create blade configuration
    blade_config = BladeConfig(planform=planform, airfoils=airfoils)

    # Create Blade instance
    blade = Blade(blade_config)

    # Interpolate airfoils at specific thicknesses
    thicknesses = np.array([0.15, 0.18, 0.2])
    interpolated_airfoils = blade.get_airfoil_xy_norm(thicknesses)

    logger.info(f"Interpolated airfoils shape: {interpolated_airfoils.shape}")
    logger.info("Example interpolated points for thickness 0.2:")
    logger.info(f"{interpolated_airfoils[2][:5]}")

    # Get planform values at specific relative spans
    rel_spans = np.array([0.0, 0.5, 1.0])
    planform_vals = blade.get_planform_array(rel_spans)
    logger.info("Planform values at rel_spans [0.0, 0.5, 1.0]:")
    for key, val in planform_vals.items():
        logger.info(f"{key}: {val}")

    # Compute sections
    sections = blade.get_sections()
    logger.info(f"Sections shape: {sections.shape}")

    # Save blade sections to VTP
    vtp_file = example_dir / "ex1_blade.vtp"
    save_blade_sections(blade, str(vtp_file))
    logger.info(f"Blade sections saved to {vtp_file}")

    logger.info("Example completed successfully.")


if __name__ == "__main__":
    main()
