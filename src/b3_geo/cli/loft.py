from __future__ import annotations
from b3_geo.api.loft_step import LoftStep


def loft_command(config_file, file="", force=False, plot=True):
    """Command to process loft."""
    if file == "":
        file = None
    step = LoftStep(config_file, output_file=file, plot=plot)
    step.run(force=force)
