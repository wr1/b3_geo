from treeparse import cli, command, argument, option
import logging
from rich.logging import RichHandler
import yaml
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, handlers=[RichHandler(show_time=False)])


def af_callback(config_file, file, force=False):
    from ..api.af_step import AFStep

    step = AFStep(config_file)
    step.run(force=force)


def loft_callback(config_file, file, force=False):
    from ..api.loft_step import LoftStep

    step = LoftStep(config_file)
    step.run(force=force)


def clean_callback(config_file):
    """Remove the b3_geo work directory."""
    config_data = yaml.safe_load(Path(config_file).read_text())
    config_dir = Path(config_file).parent
    workdir_str = config_data.get("workdir") or config_data.get("general", {}).get(
        "workdir", "."
    )
    workdir = config_dir / workdir_str / "b3_geo"
    if workdir.exists():
        shutil.rmtree(workdir)
        print(f"Removed {workdir}")
    else:
        print(f"Directory {workdir} does not exist")


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
    callback=clean_callback,
    arguments=[
        argument(name="config_file", arg_type=str, help="Path to config file"),
    ],
)
app.commands.append(clean_cmd)


def main():
    app.run()


if __name__ == "__main__":
    main()
