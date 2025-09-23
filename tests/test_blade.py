import numpy as np
from b3_geo.core.blade import Blade
from b3_geo.models import Planform, Airfoil, BladeConfig


def test_blade_init(tmp_path):
    """Test Blade initialization."""
    # Create dummy airfoil file
    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("# header\n0.0 0.0\n0.5 0.1\n1.0 0.0\n")

    # Create planform
    planform = Planform(
        z=[(0.0, 0.0), (1.0, -100.0)],
        chord=[(0.0, 1.0), (1.0, 0.8)],
        thickness=[(0.0, 0.2), (1.0, 0.15)],
        twist=[(0.0, 0.0), (1.0, 5.0)],
        dx=[(0.0, 0.0), (1.0, 1.0)],
        dy=[(0.0, 0.0), (1.0, 0.5)],
        pre_rotation=0.0,
        npchord=10,
        npspan=10,
    )

    # Create airfoil
    airfoil = Airfoil(path=str(airfoil_file), name="test", thickness=0.2)
    blade_config = BladeConfig(planform=planform, airfoils=[airfoil])

    blade = Blade(blade_config)

    assert blade.np_chordwise == 10
    assert blade.np_spanwise == 10
    assert blade.rel_span.shape == (10,)


def test_blade_plot_airfoils(tmp_path):
    """Test plotting airfoils from Blade."""
    # Create dummy airfoil file
    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("# header\n0.0 0.0\n0.5 0.1\n1.0 0.0\n")

    # Create planform
    planform = Planform(
        z=[(0.0, 0.0), (1.0, -100.0)],
        chord=[(0.0, 1.0), (1.0, 0.8)],
        thickness=[(0.0, 0.2), (1.0, 0.15)],
        twist=[(0.0, 0.0), (1.0, 5.0)],
        dx=[(0.0, 0.0), (1.0, 1.0)],
        dy=[(0.0, 0.0), (1.0, 0.5)],
        pre_rotation=0.0,
        npchord=10,
        npspan=10,
    )

    # Create airfoil
    airfoil = Airfoil(path=str(airfoil_file), name="test", thickness=0.2)
    blade_config = BladeConfig(planform=planform, airfoils=[airfoil])

    blade = Blade(blade_config)

    output_file = str(tmp_path / "airfoils.png")
    thicknesses = np.array([0.2])
    blade.plot_airfoils(thicknesses, output_file)

    assert (tmp_path / "airfoils.png").exists()
