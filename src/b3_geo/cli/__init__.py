import rich_click as click
from .af import af_command
from .loft import loft_command
import logging
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])


@click.group()
def main():
    """b3-geo CLI for blade geometry generation.

    Run individual steps or use 'run' to chain multiple steps.
    """


@main.command(help="Load and resample airfoils, plot and export.")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--file", "-f", type=click.Path(), help="Output file path.")
def af(config_file, file):
    af_command(config_file)


@main.command(help="Create LM1 blade model and export to VTP.")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--file", "-f", type=click.Path(), help="Output file path.")
def loft(config_file, file):
    loft_command(config_file)


@main.command(
    help="""Run multiple steps in sequence.

Example: b3-geo run af loft config.yml"""
)
@click.argument("steps", nargs=-1)
@click.argument("config_file", type=click.Path(exists=True))
def run(steps, config_file):
    if not steps:
        raise click.UsageError("At least one step must be provided")
    step_map = {
        "af": af_command,
        "loft": loft_command,
    }
    for step in steps:
        if step in step_map:
            step_map[step](config_file)
        else:
            raise click.BadArgumentUsage(f"Unknown step: {step}")


if __name__ == "__main__":
    main()
