import yaml
import numpy as np
from b3_geo.api.planform import interpolate_planform, process_planform, plot_planform
from pathlib import Path


def test_interpolate_planform():
    """Test interpolate_planform function."""
    planform_data = {
        "z": [(0.0, 0.0), (1.0, -100.0)],
        "chord": [(0.0, 1.0), (1.0, 0.8)],
        "thickness": [(0.0, 0.2), (1.0, 0.15)],
        "twist": [(0.0, 0.0), (1.0, 5.0)],
        "dx": [(0.0, 0.0), (1.0, 1.0)],
        "dy": [(0.0, 0.0), (1.0, 0.5)],
    }
    npspan = 10
    result = interpolate_planform(planform_data, npspan)
    assert "rel_span" in result
    assert len(result["rel_span"]) == npspan
    assert result["z"].shape == (npspan,)
    assert result["chord"][0] == 1.0
    assert result["chord"][-1] == 0.8


def test_process_planform_from_dict():
    """Test process_planform from dict."""
    config = {
        "geometry": {
            "planform": {
                "npspan": 10,
                "z": [(0.0, 0.0), (1.0, -100.0)],
                "chord": [(0.0, 1.0), (1.0, 0.8)],
                "thickness": [(0.0, 0.2), (1.0, 0.15)],
                "twist": [(0.0, 0.0), (1.0, 5.0)],
                "dx": [(0.0, 0.0), (1.0, 1.0)],
                "dy": [(0.0, 0.0), (1.0, 0.5)],
            }
        }
    }
    result = process_planform(config)
    assert "rel_span" in result
    assert len(result["rel_span"]) == 10


def test_process_planform_from_file(tmp_path):
    """Test process_planform from file."""
    config_data = {
        "geometry": {
            "planform": {
                "npspan": 10,
                "z": [[0.0, 0.0], [1.0, -100.0]],
                "chord": [[0.0, 1.0], [1.0, 0.8]],
                "thickness": [[0.0, 0.2], [1.0, 0.15]],
                "twist": [[0.0, 0.0], [1.0, 5.0]],
                "dx": [[0.0, 0.0], [1.0, 1.0]],
                "dy": [[0.0, 0.0], [1.0, 0.5]],
            }
        }
    }
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    result = process_planform(str(config_file))
    assert "rel_span" in result
    assert len(result["rel_span"]) == 10


def test_plot_planform_from_file(tmp_path):
    """Test plot_planform from file."""
    config_data = {
        "geometry": {
            "planform": {
                "npspan": 10,
                "z": [[0.0, 0.0], [1.0, -100.0]],
                "chord": [[0.0, 1.0], [1.0, 0.8]],
                "thickness": [[0.0, 0.2], [1.0, 0.15]],
                "twist": [[0.0, 0.0], [1.0, 5.0]],
                "dx": [[0.0, 0.0], [1.0, 1.0]],
                "dy": [[0.0, 0.0], [1.0, 0.5]],
            }
        }
    }
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    output_file = str(tmp_path / "planform.png")
    plot_planform(str(config_file), output_file)
    assert (tmp_path / "planform.png").exists()
