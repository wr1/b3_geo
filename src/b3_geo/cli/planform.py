from __future__ import annotations


def planform_command(config_file: str, output: str | None = None):
    """Command to plot planform."""
    if output is None:
        output = "planform.png"
    from src.b3_geo.api.planform import plot_planform

    plot_planform(config_file, output)
