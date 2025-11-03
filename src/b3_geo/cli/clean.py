from pathlib import Path
import yaml
import shutil
import logging

logger = logging.getLogger(__name__)


def clean_command(config_file: str):
    """Clean b3_geo outputs."""
    config_data = yaml.safe_load(Path(config_file).read_text())
    config_dir = Path(config_file).parent
    workdir_str = config_data.get("workdir") or config_data.get("general", {}).get(
        "workdir", "."
    )
    workdir_path = (config_dir / workdir_str).resolve()
    try:
        workdir_path.relative_to(Path.cwd())
    except ValueError:
        logger.error("Cannot delete outside of current working directory.")
        return
    b3_geo_dir = workdir_path / "b3_geo"
    if b3_geo_dir.exists():
        shutil.rmtree(b3_geo_dir)
        logger.info(f"Removed {b3_geo_dir}")
    else:
        logger.info(f"{b3_geo_dir} does not exist.")
