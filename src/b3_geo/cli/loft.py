from ..api.loft_step import LoftStep


def loft_command(config_file: str, file: str = None, force: bool = False):
    """Command to process loft."""
    step = LoftStep(config_file, output_file=file)
    step.run(force=force)
