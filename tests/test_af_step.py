import yaml

from b3_geo.api.af_step import AFStep


def test_af_step(tmp_path):
    """Test AFStep execution."""
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

    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("# header\n0.0 0.0\n0.5 0.1\n1.0 0.0\n")

    step = AFStep(str(config_file))
    step.run()

    workdir = tmp_path / "b3_geo"
    assert (workdir / "airfoils.png").exists()
    assert (workdir / "airfoils.npz").exists()
