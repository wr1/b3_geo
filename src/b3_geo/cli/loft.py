from __future__ import annotations

from b3_geo.api.loft_step import loft_step


def loft_command(config_file, file="", force=False, plot=True):
    """Command to process loft."""
    if file == "":
        file = None
    step = loft_step(config_file, output_file=file, plot=plot)
    step.run(force=force)
