import yaml

from b3_geo.cli.planform import planform_command


def test_cli_planform(tmp_path):
    """Test CLI planform command."""
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

    planform_command(str(config_file))

    assert (tmp_path / "planform.png").exists()
