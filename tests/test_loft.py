import yaml
import numpy as np
import warnings
from b3_geo.api.loft import process_loft
from b3_geo.cli.loft import loft_command

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


def test_process_loft(tmp_path):
    """Test process_loft function."""
    # Create config data
    config_data = {
        "general": {"workdir": "."},
        "geometry": {"planform": {"pre_rotation": 0.0}},
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
    pln_workdir = tmp_path / "b3_pln"
    pln_workdir.mkdir()

    # Create planform.npz
    rel_span = np.linspace(0, 1, 10)
    np.savez(
        pln_workdir / "planform.npz",
        rel_span=rel_span,
        chord=np.ones_like(rel_span),
        thickness=np.full_like(rel_span, 0.2),
        twist=np.zeros_like(rel_span),
        dx=np.zeros_like(rel_span),
        dy=np.zeros_like(rel_span),
        z=np.linspace(0, -100, 10),
    )

    # Run process_loft
    process_loft(str(config_file))

    # Check outputs
    assert (workdir / "lm1.vtp").exists()


def test_loft_command(tmp_path):
    """Test loft command."""
    # Create config data
    config_data = {
        "general": {"workdir": "."},
        "geometry": {"planform": {"pre_rotation": 0.0}},
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
    pln_workdir = tmp_path / "b3_pln"
    pln_workdir.mkdir()

    # Create planform.npz
    rel_span = np.linspace(0, 1, 10)
    np.savez(
        pln_workdir / "planform.npz",
        rel_span=rel_span,
        chord=np.ones_like(rel_span),
        thickness=np.full_like(rel_span, 0.2),
        twist=np.zeros_like(rel_span),
        dx=np.zeros_like(rel_span),
        dy=np.zeros_like(rel_span),
        z=np.linspace(0, -100, 10),
    )

    # Run loft_command
    loft_command(str(config_file))

    # Check outputs
    assert (workdir / "lm1.vtp").exists()
