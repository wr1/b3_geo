import numpy as np

from b3_geo.utils.plotting import (
    plot_airfoil_curvature,
    plot_airfoils,
    plot_planform,
)


def test_plot_airfoils(tmp_path):
    """Test plotting airfoils."""
    airfoils_data = {
        "test": {
            "data": np.array([[0.0, 0.0], [0.5, 0.1], [1.0, 0.0]]),
            "thickness": 0.2,
        }
    }
    output_file = str(tmp_path / "plot.png")
    plot_airfoils(airfoils_data, 3, output_file)
    assert (tmp_path / "plot.png").exists()


def test_plot_airfoil_curvature(tmp_path):
    """Test curvature plot for a set of contours."""
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = 0.5 * np.column_stack((np.cos(theta), np.sin(theta)))
    ellipse = np.column_stack((0.5 * np.cos(theta), 0.1 * np.sin(theta)))
    sections = {"circle": circle, "ellipse": ellipse}
    output_file = str(tmp_path / "curvature.png")
    plot_airfoil_curvature(sections, output_file)
    assert (tmp_path / "curvature.png").exists()


def test_plot_planform(tmp_path):
    """Test plotting planform."""
    interpolated = {
        "rel_span": np.linspace(0, 1, 10),
        "z": np.linspace(0, -100, 10),
        "chord": np.ones(10),
        "thickness": np.full(10, 0.2),
        "twist": np.zeros(10),
        "dx": np.zeros(10),
        "dy": np.zeros(10),
        "absolute_thickness": np.full(10, 0.2),
    }
    controls = {
        "z": [(0.0, 0.0), (1.0, -100.0)],
        "chord": [(0.0, 1.0), (1.0, 0.8)],
        "thickness": [(0.0, 0.2), (1.0, 0.15)],
        "twist": [(0.0, 0.0), (1.0, 5.0)],
        "dx": [(0.0, 0.0), (1.0, 1.0)],
        "dy": [(0.0, 0.0), (1.0, 0.5)],
    }
    output_file = str(tmp_path / "planform.png")
    plot_planform(interpolated, controls, interpolated["rel_span"], output_file)
    assert (tmp_path / "planform.png").exists()
