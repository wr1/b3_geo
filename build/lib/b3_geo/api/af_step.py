from pathlib import Path
from statesman import Statesman


class AFStep(Statesman):
    """Step for processing airfoils with statesman dependency management."""

    workdir_key = "workdir"
    dependent_sections = ["airfoils", "geometry"]
    output_files = ["b3_geo/airfoils.png", "b3_geo/airfoils.npz"]

    def __init__(self, config_path):
        super().__init__(config_path)
        self.force = False

    def run(self, force=False):
        self.force = force
        super().run()

    def needs_run(self):
        if self.force:
            return True
        return super().needs_run()

    def _execute(self):
        from .af import process_af

        config_dir = Path(self.config_path).parent
        workdir_str = self.config["workdir"]
        workdir = config_dir / workdir_str / "b3_geo"
        process_af(self.config_path, workdir)
