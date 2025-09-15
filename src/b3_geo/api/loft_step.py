from pathlib import Path
from statesman.core.base import Statesman, ManagedFile


class LoftStep(Statesman):
    """Step for processing loft with statesman dependency management."""

    workdir_key = "general.workdir"
    dependent_sections = ["geometry", "airfoils"]
    input_files = [
        ManagedFile(name="b3_geo/airfoils.npz", non_empty=True, newer_than="config")
    ]
    output_files = ["b3_geo/lm1.vtp"]

    def _execute(self):
        from .loft import process_loft

        config_dir = Path(self.config_path).parent
        workdir = config_dir / self.config["general"]["workdir"] / "b3_geo"
        process_loft(self.config_path, workdir)
