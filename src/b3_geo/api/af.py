from pathlib import Path
import yaml
import numpy as np
from typing import Dict, List
from b3_geo.models import Airfoil
from b3_geo.utils.interpolation import load_airfoil, interpolate_airfoil
from b3_geo.utils.plotting import plot_airfoils
import logging
import time

logger = logging.getLogger(__name__)


def resample_airfoils(airfoils: List[Airfoil], npchord: int) -> Dict[str, Dict]:
    """Resample airfoils to uniform points."""
    airfoils_dict = {}
    for af in airfoils:
        data = load_airfoil(af.path)
        resampled = interpolate_airfoil(data, npchord)
        airfoils_dict[af.name] = {"data": resampled, "thickness": af.thickness}
    return airfoils_dict


def process_af(config_path: str, workdir: Path = None) -> Dict[str, Dict]:
    """Process airfoils: load, resample, plot, and save."""
    start_time = time.time()
    logger.info("Starting af step")
    config_data = yaml.safe_load(Path(config_path).read_text())
    config_dir = Path(config_path).parent
    logger.info(f"Config data keys: {list(config_data.keys())}")
    if workdir is None:
        workdir = (
            config_dir / config_data.get("general", {}).get("workdir", ".") / "b3_geo"
        )
    workdir.mkdir(exist_ok=True, parents=True)
    # Load planform from b3_pln
    pln_workdir = (
        config_dir / config_data.get("general", {}).get("workdir", ".") / "b3_pln"
    )
    planform_npz = pln_workdir / "planform.npz"
    if not planform_npz.exists():
        raise FileNotFoundError(f"Planform file not found: {planform_npz}")
    planform_data = np.load(planform_npz)
    npchord = planform_data["chord"].shape[0]  # Assuming npchord is the length
    airfoils_data = config_data.get("airfoils", [])
    logger.info(f"Airfoils data: {airfoils_data}")
    airfoils = [
        Airfoil(
            path=str(config_dir / af["path"]),
            name=af["name"],
            thickness=af["thickness"],
        )
        for af in airfoils_data
    ]
    airfoils_dict = resample_airfoils(airfoils, npchord)
    plot_file = workdir / "airfoils.png"
    plot_airfoils(airfoils_dict, npchord, str(plot_file))
    logger.info(f"Saved airfoils plot to {plot_file}")
    npz_file = workdir / "airfoils.npz"
    names = list(airfoils_dict.keys())
    thicknesses = [af["thickness"] for af in airfoils_dict.values()]
    data = [af["data"] for af in airfoils_dict.values()]
    np.savez(
        npz_file,
        names=np.array(names),
        thicknesses=np.array(thicknesses),
        data=np.array(data),
    )
    logger.info(f"Saved airfoils data to {npz_file}")
    logger.info("Af step completed")
    elapsed = time.time() - start_time
    logger.info(f"Af step took {elapsed:.2f} seconds")
    return airfoils_dict
