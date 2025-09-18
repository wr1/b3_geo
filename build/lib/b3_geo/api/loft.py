from pathlib import Path
import yaml
import numpy as np
from typing import Optional
from b3_geo.models import Planform, Airfoil, BladeConfig
from b3_geo.core.blade import Blade
from b3_geo.utils.cache import save_blade_sections
import logging
import time

logger = logging.getLogger(__name__)


def create_lm1(blade: Blade) -> np.ndarray:
    """Create LM1 sections."""
    return blade.get_sections()


def process_loft(config_path: str, workdir: Optional[Path] = None) -> np.ndarray:
    """Process loft: create blade model and save to VTP."""
    start_time = time.time()
    logger.info("Starting loft step")
    config_data = yaml.safe_load(Path(config_path).read_text())
    config_dir = Path(config_path).parent
    logger.info(f"Config data keys: {list(config_data.keys())}")
    if workdir is None:
        workdir_str = config_data.get("workdir") or config_data.get("general", {}).get(
            "workdir", "."
        )
        workdir = config_dir / workdir_str / "b3_geo"
    workdir.mkdir(exist_ok=True, parents=True)
    geometry_data = config_data.get("geometry", {})
    planform_data_config = geometry_data.get("planform", {})
    pre_rotation = planform_data_config.get("pre_rotation", 0.0)
    logger.info(f"Pre-rotation: {pre_rotation}")
    airfoils_data = config_data.get("airfoils", [])
    logger.info(f"Airfoils data: {airfoils_data}")
    # Load planform from b3_pln
    workdir_str = config_data.get("workdir") or config_data.get("general", {}).get(
        "workdir", "."
    )
    pln_workdir = config_dir / workdir_str / "b3_pln"
    planform_npz = pln_workdir / "planform.npz"
    if not planform_npz.exists():
        raise FileNotFoundError(f"Planform file not found: {planform_npz}")
    planform_data = np.load(planform_npz)
    # Reconstruct Planform from npz data
    rel_span = planform_data["rel_span"]
    z = list(zip(rel_span, planform_data["z"]))
    chord = list(zip(rel_span, planform_data["chord"]))
    thickness = list(zip(rel_span, planform_data["thickness"]))
    twist = list(zip(rel_span, planform_data["twist"] + pre_rotation))
    dx = list(zip(rel_span, planform_data["dx"]))
    dy = list(zip(rel_span, planform_data["dy"]))
    npchord = len(rel_span)  # Assuming npchord is npspan
    npspan = len(rel_span)
    planform = Planform(
        z=z,
        chord=chord,
        thickness=thickness,
        twist=twist,
        dx=dx,
        dy=dy,
        pre_rotation=0.0,
        npchord=npchord,
        npspan=npspan,
    )
    blade_config = BladeConfig(
        planform=planform,
        airfoils=[
            Airfoil(
                path=str(config_dir / af["path"]),
                name=af["name"],
                thickness=af["thickness"],
            )
            for af in airfoils_data
        ],
    )
    blade = Blade(blade_config)
    sections = create_lm1(blade)
    vtp_file = workdir / "lm1.vtp"
    save_blade_sections(blade, str(vtp_file))
    logger.info(f"Saved blade sections to {vtp_file}")
    logger.info("Loft step completed")
    elapsed = time.time() - start_time
    logger.info(f"Loft step took {elapsed:.2f} seconds")
    return sections
