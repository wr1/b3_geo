from treeparse import cli, group, command, argument, option
import logging
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])


def af_callback(config_file, file):
    from .af import af_command

    af_command(config_file)


def loft_callback(config_file, file):
    from .loft import loft_command

    loft_command(config_file)


def run_callback(steps, config_file):
    from .af import af_command
    from .loft import loft_command

    if not steps:
        raise ValueError("At least one step must be provided")
    step_map = {
        "af": af_command,
        "loft": loft_command,
    }
    for step in steps:
        if step in step_map:
            step_map[step](config_file)
        else:
            raise ValueError(f"Unknown step: {step}")


app = cli(
    name="b3-geo",
    help="b3-geo CLI for blade geometry generation.",
    max_width=120,
    show_types=True,
    show_defaults=True,
    line_connect=True,
    theme="monochrome",
)

af_cmd = command(
    name="af",
    help="Load and resample airfoils, plot and export.",
    callback=af_callback,
    arguments=[
        argument(name="config_file", arg_type=str, help="Path to config file"),
    ],
    options=[
        option(
            flags=["--file", "-f"],
            arg_type=str,
            help="Output file path.",
        ),
    ],
)
app.commands.append(af_cmd)

loft_cmd = command(
    name="loft",
    help="Create LM1 blade model and export to VTP.",
    callback=loft_callback,
    arguments=[
        argument(name="config_file", arg_type=str, help="Path to config file"),
    ],
    options=[
        option(
            flags=["--file", "-f"],
            arg_type=str,
            help="Output file path.",
        ),
    ],
)
app.commands.append(loft_cmd)

run_cmd = command(
    name="run",
    help="Run multiple steps in sequence.",
    callback=run_callback,
    arguments=[
        argument(name="steps", arg_type=str, nargs=-1, help="Steps to run"),
        argument(name="config_file", arg_type=str, help="Path to config file"),
    ],
)
app.commands.append(run_cmd)


def main():
    app.run()


if __name__ == "__main__":
    main()
