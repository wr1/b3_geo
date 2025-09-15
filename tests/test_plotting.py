import numpy as np
from b3_geo.utils.plotting import plot_airfoils


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
