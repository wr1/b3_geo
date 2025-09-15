from pathlib import Path
from statesman.core.base import Statesman


class LoftStep(Statesman):
    """Step for processing loft with statesman dependency management."""

    workdir_key = "general.workdir"
    dependent_sections = ["geometry", "airfoils"]
    output_files = ["b3_geo/lm1.vtp"]

    def _execute(self):
        from .loft import process_loft

        config_dir = Path(self.config_path).parent
        workdir = config_dir / self.config["general"]["workdir"] / "b3_geo"
        process_loft(self.config_path, workdir)
