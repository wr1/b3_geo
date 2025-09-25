def planform_command(config_file: str, output: str = None):
    """Command to plot planform."""
    if output is None:
        output = "planform.png"
    from ..api.planform import plot_planform
    plot_planform(config_file, output)
