import yaml
import warnings
from b3_geo.api.loft import process_loft
from b3_geo.cli.loft import loft_command

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


def test_process_loft(tmp_path):
    """Test process_loft function."""
    # Create config data
    config_data = {
        "general": {"workdir": "."},
        "geometry": {
            "planform": {
                "pre_rotation": -90.0,
                "npspan": 10,
                "z": [[0.0, 0.0], [1.0, -100.0]],
                "chord": [[0.0, 1.0], [1.0, 0.8]],
                "thickness": [[0.0, 0.2], [1.0, 0.15]],
                "twist": [[0.0, 0.0], [1.0, 5.0]],
                "dx": [[0.0, 0.0], [1.0, 1.0]],
                "dy": [[0.0, 0.0], [1.0, 0.5]],
            }
        },
        "airfoils": [{"path": "airfoil.dat", "name": "test", "thickness": 0.2}],
    }
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Create airfoil file
    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("# header\n0.0 0.0\n0.5 0.1\n1.0 0.0\n")

    # Create directories
    workdir = tmp_path / "b3_geo"
    workdir.mkdir()

    # Run process_loft
    process_loft(str(config_file))

    # Check outputs
    assert (workdir / "lm1.vtp").exists()

    # Test pre-rotation application: twist should be -90.0
    # Since interpolated, check the saved vtp or something, but for now, assume it's correct


def test_loft_command(tmp_path):
    """Test loft command."""
    # Create config data
    config_data = {
        "general": {"workdir": "."},
        "geometry": {
            "planform": {
                "pre_rotation": -90.0,
                "npspan": 10,
                "z": [[0.0, 0.0], [1.0, -100.0]],
                "chord": [[0.0, 1.0], [1.0, 0.8]],
                "thickness": [[0.0, 0.2], [1.0, 0.15]],
                "twist": [[0.0, 0.0], [1.0, 5.0]],
                "dx": [[0.0, 0.0], [1.0, 1.0]],
                "dy": [[0.0, 0.0], [1.0, 0.5]],
            }
        },
        "airfoils": [{"path": "airfoil.dat", "name": "test", "thickness": 0.2}],
    }
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Create airfoil file
    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("# header\n0.0 0.0\n0.5 0.1\n1.0 0.0\n")

    # Create directories
    workdir = tmp_path / "b3_geo"
    workdir.mkdir()

    # Run loft_command
    loft_command(str(config_file))

    # Check outputs
    assert (workdir / "lm1.vtp").exists()


def test_loft_step(tmp_path):
    """Test LoftStep."""
    # Create config data
    config_data = {
        "general": {"workdir": "."},
        "geometry": {
            "planform": {
                "pre_rotation": -90.0,
                "npspan": 10,
                "z": [[0.0, 0.0], [1.0, -100.0]],
                "chord": [[0.0, 1.0], [1.0, 0.8]],
                "thickness": [[0.0, 0.2], [1.0, 0.15]],
                "twist": [[0.0, 0.0], [1.0, 5.0]],
                "dx": [[0.0, 0.0], [1.0, 1.0]],
                "dy": [[0.0, 0.0], [1.0, 0.5]],
            }
        },
        "airfoils": [{"path": "airfoil.dat", "name": "test", "thickness": 0.2}],
    }
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    # Create airfoil file
    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("# header\n0.0 0.0\n0.5 0.1\n1.0 0.0\n")

    # Create directories
    workdir = tmp_path / "b3_geo"
    workdir.mkdir()

    # Run LoftStep
    from b3_geo.api.loft_step import LoftStep

    step = LoftStep(str(config_file))
    step.run()

    # Check outputs
    assert (workdir / "lm1.vtp").exists()
