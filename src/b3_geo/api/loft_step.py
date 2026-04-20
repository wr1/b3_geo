import logging
from pathlib import Path

import yaml


class loft_step:
    workdir_key = "workdir"

    def __init__(self, config_path, output_file=None, plot=True):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = Path(config_path).resolve()
        with open(self.config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        workdir_str = self.config.get(self.workdir_key, ".")
        self.workdir = (self.config_path.parent / Path(workdir_str)).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.output_file = output_file
        self.plot = plot

    def run(self, force=False):
        self._execute()

    def _execute(self):
        from .loft import process_loft

        config_dir = Path(self.config_path).parent
        workdir_str = self.config.get("workdir") or self.config.get("general", {}).get(
            "workdir", "."
        )
        workdir = config_dir / workdir_str / "b3_geo"
        process_loft(
            self.config_path,
            workdir=workdir,
            output_file=self.output_file,
            plot=self.plot,
        )
