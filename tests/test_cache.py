import warnings
from b3_geo.core.blade import Blade
from b3_geo.models import Planform, Airfoil, BladeConfig
from b3_geo.utils.cache import save_blade_sections

warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")


def test_save_blade_sections(tmp_path):
    """Test saving blade sections."""
    # Create dummy airfoil file
    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("0.0 0.0\n0.5 0.1\n1.0 0.0\n")

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

    filepath = str(tmp_path / "test.vtk")
    save_blade_sections(blade, filepath)

    assert (tmp_path / "test.vtk").exists()
