import matplotlib.pyplot as plt
from typing import Dict
from .interpolation import interpolate_airfoil


def plot_airfoils(airfoils_data: Dict[str, Dict], n_points: int, output_file: str):
    """Plot all interpolated airfoils in a single matplotlib plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, info in airfoils_data.items():
        interp_data = interpolate_airfoil(info["data"], n_points)
        ax.plot(interp_data[:, 0], interp_data[:, 1], label=name)
    ax.set_title("Interpolated Airfoils")
    ax.set_xlabel("x/chord")
    ax.set_ylabel("y/chord")
    ax.legend()
    plt.savefig(output_file)
    plt.close()
