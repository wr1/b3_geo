import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from treeparse import cli, command, argument, option

logging.basicConfig(level=logging.INFO)

from .af import af_command
from .loft import loft_command
from .clean import clean_command


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
    callback=af_command,
    arguments=[
        argument(name="config_file", arg_type=str, help="Path to config file"),
    ],
    options=[
        option(
            flags=["--file", "-f"],
            arg_type=str,
            help="Output file path.",
        ),
        option(
            flags=["--force", "-F"],
            arg_type=bool,
            help="Force rerun despite statesman checks.",
        ),
    ],
)
app.commands.append(af_cmd)

loft_cmd = command(
    name="loft",
    help="Create LM1 blade model and export to VTP.",
    callback=loft_command,
    arguments=[
        argument(name="config_file", arg_type=str, help="Path to config file"),
    ],
    options=[
        option(
            flags=["--file", "-f"],
            arg_type=str,
            help="Output file path.",
        ),
        option(
            flags=["--force", "-F"],
            arg_type=bool,
            help="Force rerun despite statesman checks.",
        ),
    ],
)
app.commands.append(loft_cmd)

clean_cmd = command(
    name="clean",
    help="Remove the b3_geo work directory.",
    callback=clean_command,
    arguments=[
        argument(name="config_file", arg_type=str, help="Path to config file"),
    ],
)
app.commands.append(clean_cmd)


def main():
    # If only one argument and it's not a command or flag, assume 'loft'
    if (
        len(sys.argv) == 2
        and not sys.argv[1].startswith("-")
        and sys.argv[1] not in ["af", "loft", "clean"]
    ):
        sys.argv.insert(1, "loft")
    app.run()


if __name__ == "__main__":
    main()
