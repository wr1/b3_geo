from ..api import process_af


def af_command(config_path: str):
    """Command to process airfoils."""
    process_af(config_path)
