from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from b3_geo.utils.interpolation import (
    cubic_interpolate,
    linear_interpolate,
    pchip_interpolate,
)


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


def process_planform(config: str | Path | dict) -> dict:
    """Process planform from config file or dict."""
    if isinstance(config, (str, Path)):
        config_data = yaml.safe_load(Path(config).read_text())
        planform_data = config_data.get("geometry", {}).get("planform", {})
    elif isinstance(config, dict):
        planform_data = config.get("geometry", {}).get("planform", {})
    else:
        msg = "config must be path or dict"
        raise ValueError(msg)
    npspan = planform_data.get("npspan", 100)
    return interpolate_planform(planform_data, npspan)


def plot_planform(config: str | Path | dict, output_file: str):
    """Plot planform from config."""
    if isinstance(config, (str, Path)):
        config_data = yaml.safe_load(Path(config).read_text())
        planform_data = config_data.get("geometry", {}).get("planform", {})
    elif isinstance(config, dict):
        planform_data = config.get("geometry", {}).get("planform", {})
    else:
        msg = "config must be path or dict"
        raise ValueError(msg)
    npspan = planform_data.get("npspan", 100)
    interp_plan = interpolate_planform(planform_data, npspan)
    controls = {
        k: planform_data.get(k, [])
        for k in ["z", "chord", "thickness", "twist", "dx", "dy"]
    }
    from b3_geo.utils.plotting import plot_planform

    plot_planform(interp_plan, controls, interp_plan["rel_span"], output_file)
