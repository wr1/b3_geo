from pathlib import Path
import yaml
import numpy as np
from typing import Optional
from b3_geo.models import Planform, Airfoil, BladeConfig
from b3_geo.core.blade import Blade
from b3_geo.utils.cache import save_blade_sections
from b3_geo.utils.interpolation import (
    linear_interpolate,
    pchip_interpolate,
    cubic_interpolate,
)
import logging
import time

logger = logging.getLogger(__name__)


def interpolate_planform(planform_data, npspan):
    """Interpolate planform parameters."""
    rel_span = np.linspace(0, 1, npspan)
    interp_plan = {
        "rel_span": rel_span,
        "z": linear_interpolate(planform_data["z"], rel_span),
        "chord": pchip_interpolate(planform_data["chord"], rel_span),
        "thickness": cubic_interpolate(
            planform_data["thickness"], rel_span, bc_type="natural"
        ),
        "twist": pchip_interpolate(planform_data["twist"], rel_span),
        "dx": cubic_interpolate(planform_data["dx"], rel_span, bc_type="natural"),
        "dy": cubic_interpolate(planform_data["dy"], rel_span),
    }
    interp_plan["absolute_thickness"] = interp_plan["chord"] * interp_plan["thickness"]
    return interp_plan


def create_lm1(blade: Blade) -> np.ndarray:
    """Create LM1 sections."""
    return blade.get_sections()


def process_loft(
    config_path: str, workdir: Optional[Path] = None, output_file: Optional[str] = None
) -> np.ndarray:
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
    npspan = planform_data_config.get("npspan", 100)
    interp_plan = interpolate_planform(planform_data_config, npspan)
    airfoils_data = config_data.get("airfoils", [])
    logger.info(f"Airfoils data: {airfoils_data}")
    # Create Planform from interpolated data
    planform = Planform(
        z=list(zip(interp_plan["rel_span"], interp_plan["z"])),
        chord=list(zip(interp_plan["rel_span"], interp_plan["chord"])),
        thickness=list(zip(interp_plan["rel_span"], interp_plan["thickness"])),
        twist=list(zip(interp_plan["rel_span"], interp_plan["twist"] + pre_rotation)),
        dx=list(zip(interp_plan["rel_span"], interp_plan["dx"])),
        dy=list(zip(interp_plan["rel_span"], interp_plan["dy"])),
        pre_rotation=0.0,
        npchord=planform_data_config.get("npchord", 200),
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
    if output_file:
        vtp_file = Path(output_file)
    else:
        vtp_file = workdir / "lm1.vtp"
    save_blade_sections(blade, str(vtp_file))
    logger.info(f"Saved blade sections to {vtp_file}")
    # Create sections at mesh.z positions
    mesh_data = config_data.get("mesh", {})
    mesh_z = mesh_data.get("z", [])
    if mesh_z:
        rels_mesh = np.array([blade.z_to_rel(z) for z in mesh_z])
        sections_mesh = blade.get_sections(rels_mesh)
        mesh_vtp_file = workdir / "lm1_mesh.vtp"
        save_blade_sections(
            blade, str(mesh_vtp_file), sections=sections_mesh, rel_spans=rels_mesh
        )
        logger.info(f"Saved mesh sections to {mesh_vtp_file}")
    logger.info("Loft step completed")
    elapsed = time.time() - start_time
    logger.info(f"Loft step took {elapsed:.2f} seconds")
    return sections
