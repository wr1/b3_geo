from pathlib import Path
from statesman.core.base import Statesman


class AFStep(Statesman):
    """Step for processing airfoils with statesman dependency management."""

    workdir_key = "general.workdir"
    dependent_sections = ["airfoils", "geometry"]
    output_files = ["b3_geo/airfoils.png", "b3_geo/airfoils.npz"]

    def _execute(self):
        from .af import process_af

        config_dir = Path(self.config_path).parent
        workdir = config_dir / self.config["general"]["workdir"] / "b3_geo"
        process_af(self.config_path, workdir)
