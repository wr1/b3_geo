import yaml

from b3_geo.api.loft_step import LoftStep


def test_loft_step(tmp_path):
    """Test LoftStep execution."""
    config_data = {
        "general": {"workdir": "."},
        "geometry": {
            "planform": {
                "z": [[0.0, 0.0], [1.0, -100.0]],
                "chord": [[0.0, 1.0], [1.0, 0.8]],
                "thickness": [[0.0, 0.2], [1.0, 0.15]],
                "twist": [[0.0, 0.0], [1.0, 5.0]],
                "dx": [[0.0, 0.0], [1.0, 1.0]],
                "dy": [[0.0, 0.0], [1.0, 0.5]],
            }
        },
        "airfoils": [{"path": "airfoil.dat", "name": "test", "thickness": 0.2}],
        "mesh": {"z": [{"type": "linspace", "values": [0.0, -100.0], "num": 10}]},
    }
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("# header\n0.0 0.0\n0.5 0.1\n1.0 0.0\n")

    step = LoftStep(str(config_file))
    assert step.workdir_key == "workdir"
    assert step.dependent_sections == ["geometry", "airfoils"]
    assert len(step.output_files) == 2
    assert step.output_file is None
    assert step.plot is True
    assert step.force is False
    step.run()

    workdir = tmp_path / "b3_geo"
    assert (workdir / "planform.png").exists()
    assert (workdir / "lm1_mesh.vtp").exists()
