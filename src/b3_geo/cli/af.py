from b3_geo.api.af_step import af_step


def af_command(config_file: str, force: bool = False):
    """Command to process airfoils."""
    step = af_step(config_file)
    step.run(force=force)
