from ..api.af_step import AFStep


def af_command(config_file: str, force: bool = False):
    """Command to process airfoils."""
    step = AFStep(config_file)
    step.run(force=force)
