from ..api.loft_step import LoftStep


def loft_command(config_file: str, file: str = None, force: bool = False, plot: bool = True):
    """Command to process loft."""
    step = LoftStep(config_file, output_file=file, plot=plot)
    step.run(force=force)
