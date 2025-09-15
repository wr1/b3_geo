from ..api import process_loft


def loft_command(config_path: str, output_file: str = None):
    """Command to process loft."""
    process_loft(config_path, output_file=output_file)
