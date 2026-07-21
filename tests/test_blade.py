import numpy as np

from b3_geo.core.blade import Blade
from b3_geo.models import Airfoil, BladeConfig, Planform


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
        npchord=10,
    )

    # Create airfoil
    airfoil = Airfoil(path=str(airfoil_file), name="test", thickness=0.2)
    blade_config = BladeConfig(planform=planform, airfoils=[airfoil])

    blade = Blade(blade_config)

    assert blade.np_chordwise == 10
    assert len(blade.rel_span) == 100


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
        npchord=10,
    )

    # Create airfoil
    airfoil = Airfoil(path=str(airfoil_file), name="test", thickness=0.2)
    blade_config = BladeConfig(planform=planform, airfoils=[airfoil])

    blade = Blade(blade_config)

    output_file = str(tmp_path / "airfoils.png")
    thicknesses = np.array([0.2])
    blade.plot_airfoils(thicknesses, output_file)

    assert (tmp_path / "airfoils.png").exists()


def test_blade_blend_info(tmp_path):
    """Test blend description for input, interpolated, and clamped thicknesses."""
    af1 = tmp_path / "af1.dat"
    af1.write_text("# header\n0.0 0.0\n0.5 0.1\n1.0 0.0\n")
    af2 = tmp_path / "af2.dat"
    af2.write_text("# header\n0.0 0.0\n0.5 0.15\n1.0 0.0\n")

    planform = Planform(
        z=[(0.0, 0.0), (1.0, -100.0)],
        chord=[(0.0, 1.0), (1.0, 0.8)],
        thickness=[(0.0, 0.3), (1.0, 0.2)],
        twist=[(0.0, 0.0), (1.0, 5.0)],
        dx=[(0.0, 0.0), (1.0, 1.0)],
        dy=[(0.0, 0.0), (1.0, 0.5)],
        npchord=10,
    )
    blade = Blade(
        BladeConfig(
            planform=planform,
            airfoils=[
                Airfoil(path=str(af1), name="thin", thickness=0.2),
                Airfoil(path=str(af2), name="thick", thickness=0.3),
            ],
        )
    )

    assert blade.blend_info(0.2) == "thin (input)"
    assert blade.blend_info(0.3) == "thick (input)"
    assert blade.blend_info(0.25) == "50% thin + 50% thick"
    assert "clamped" in blade.blend_info(0.1)
    assert "clamped" in blade.blend_info(0.4)
    assert blade._bracketing_airfoils(0.25) == [("thin", 0.2), ("thick", 0.3)]
    assert blade._bracketing_airfoils(0.2) == []


def test_blade_te_thickness(tmp_path):
    """TE gap of blended airfoils interpolates linearly between the inputs."""
    # Selig-like diamond contours with known TE gaps of 0.02 and 0.04
    af1 = tmp_path / "af1.dat"
    af1.write_text("# header\n1.0 0.01\n0.5 0.1\n0.0 0.0\n0.5 -0.1\n1.0 -0.01\n")
    af2 = tmp_path / "af2.dat"
    af2.write_text("# header\n1.0 0.02\n0.5 0.15\n0.0 0.0\n0.5 -0.15\n1.0 -0.02\n")

    planform = Planform(
        z=[(0.0, 0.0), (1.0, -100.0)],
        chord=[(0.0, 1.0), (1.0, 0.8)],
        thickness=[(0.0, 0.3), (1.0, 0.2)],
        twist=[(0.0, 0.0), (1.0, 5.0)],
        dx=[(0.0, 0.0), (1.0, 1.0)],
        dy=[(0.0, 0.0), (1.0, 0.5)],
        npchord=50,
    )
    blade = Blade(
        BladeConfig(
            planform=planform,
            airfoils=[
                Airfoil(path=str(af1), name="thin", thickness=0.2),
                Airfoil(path=str(af2), name="thick", thickness=0.3),
            ],
        )
    )

    assert np.isclose(blade.get_te_thickness(0.2), 0.02)
    assert np.isclose(blade.get_te_thickness(0.3), 0.04)
    assert np.isclose(blade.get_te_thickness(0.25), 0.03)
    gaps = blade.get_te_thickness(np.array([0.2, 0.25, 0.3]))
    assert np.allclose(gaps, [0.02, 0.03, 0.04])

    output_file = str(tmp_path / "te_thickness.png")
    blade.plot_te_thickness(output_file)
    assert (tmp_path / "te_thickness.png").exists()


def test_blade_plot_airfoil_curvature(tmp_path):
    """Test curvature plot of interpolated airfoils from Blade."""
    airfoil_file = tmp_path / "airfoil.dat"
    airfoil_file.write_text("# header\n0.0 0.0\n0.25 0.08\n0.5 0.1\n0.75 0.08\n1.0 0.0\n")

    planform = Planform(
        z=[(0.0, 0.0), (1.0, -100.0)],
        chord=[(0.0, 1.0), (1.0, 0.8)],
        thickness=[(0.0, 0.2), (1.0, 0.15)],
        twist=[(0.0, 0.0), (1.0, 5.0)],
        dx=[(0.0, 0.0), (1.0, 1.0)],
        dy=[(0.0, 0.0), (1.0, 0.5)],
        npchord=50,
    )

    airfoil = Airfoil(path=str(airfoil_file), name="test", thickness=0.2)
    blade_config = BladeConfig(planform=planform, airfoils=[airfoil])

    blade = Blade(blade_config)

    output_file = str(tmp_path / "airfoil_curvature.png")
    blade.plot_airfoil_curvature(np.array([0.15, 0.2]), output_file)

    assert (tmp_path / "airfoil_curvature.png").exists()
