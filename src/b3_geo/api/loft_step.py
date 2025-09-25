from pathlib import Path
from statesman import Statesman


class LoftStep(Statesman):
    """Step for processing loft with statesman dependency management."""

    workdir_key = "workdir"
    dependent_sections = ["geometry", "airfoils"]
    output_files = ["b3_geo/lm1.vtp"]

    def __init__(self, config_path, output_file=None):
        super().__init__(config_path)
        self.output_file = output_file
        self.force = False

    def run(self, force=False):
        self.force = force
        super().run()

    def needs_run(self):
        if self.force:
            return True
        return super().needs_run()

    def _execute(self):
        from .loft import process_loft

        config_dir = Path(self.config_path).parent
        workdir_str = self.config.get("workdir") or self.config.get("general", {}).get(
            "workdir", "."
        )
        workdir = config_dir / workdir_str / "b3_geo"
        process_loft(self.config_path, workdir=workdir, output_file=self.output_file)
