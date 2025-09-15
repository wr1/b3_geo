from pathlib import Path
from statesman.core.base import Statesman


class LoftStep(Statesman):
    """Step for processing loft with statesman dependency management."""

    workdir_key = "general.workdir"
    dependent_sections = ["geometry", "airfoils"]
    output_files = ["b3_geo/lm1.vtp"]

    def run(self, force=False):
        self.logger.info("Starting run check.")
        if force or self.needs_run():
            self.logger.info("Step needs to run. Executing...")
            self._execute()
            self.logger.info("Execution completed.")
            self.save_state()
        else:
            self.logger.info("Step does not need to run.")

    def _execute(self):
        from .loft import process_loft

        config_dir = Path(self.config_path).parent
        workdir = config_dir / self.config["general"]["workdir"] / "b3_geo"
        process_loft(self.config_path, workdir)
