from pathlib import Path
from typing import ClassVar

from b3_state import b3_state


class loft_step(b3_state):
    """Step for processing loft with b3_state dependency management."""

    workdir_key = "workdir"
    dependent_sections: ClassVar[list[str]] = ["geometry", "airfoils", "mesh", "mesh3d"]

    def __init__(self, config_path, output_file=None, plot=True):
        super().__init__(config_path)
        self.output_file = output_file
        self.plot = plot
        self.force = False
        # Conditionally set output_files based on presence of mesh or mesh3d config
        self.output_files = ["b3_geo/planform.png", "b3_geo/airfoil_curvature.png"]
        if "mesh" in self.config and self.config["mesh"].get("z"):
            self.output_files.append("b3_geo/lm1_mesh.vtp")
        if "mesh3d" in self.config and self.config["mesh3d"].get("z"):
            self.output_files.append("b3_geo/lm1_mesh3d.vtp")

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
        process_loft(
            self.config_path,
            workdir=workdir,
            output_file=self.output_file,
            plot=self.plot,
        )
