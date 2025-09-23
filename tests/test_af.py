import yaml
from b3_geo.api.af import process_af
from b3_geo.cli.af import af_command


def test_process_af(tmp_path):
    """Test process_af function."""
    # Create config data
    config_data = {
        "general": {"workdir": "."},
        "geometry": {
            "planform": {
                "npchord": 10,
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

    # Run process_af
    process_af(str(config_file))

    # Check outputs
    assert (workdir / "airfoils.png").exists()
    assert (workdir / "airfoils.npz").exists()


def test_af_command(tmp_path):
    """Test af command."""
    # Create config data
    config_data = {
        "general": {"workdir": "."},
        "geometry": {
            "planform": {
                "npchord": 10,
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

    # Run af_command
    af_command(str(config_file))

    # Check outputs
    assert (workdir / "airfoils.png").exists()
    assert (workdir / "airfoils.npz").exists()
