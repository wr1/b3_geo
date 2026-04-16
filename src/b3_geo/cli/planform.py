from __future__ import annotations

from pathlib import Path


def planform_command(config_file, output=""):
    """Command to plot planform."""
    if output == "":
        output = str(Path(config_file).parent / "planform.png")
    from b3_geo.api.planform import plot_planform

    plot_planform(config_file, output)
