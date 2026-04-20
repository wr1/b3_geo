"""ex01 — Airfoil profiles: plot several NACA 4-digit sections.

Demonstrates plot_airfoils with five airfoils spanning the typical
thickness range used along a wind turbine blade (12% – 40%).
"""

import os
import numpy as np
from pathlib import Path
from b3_geo.utils.plotting import plot_airfoils


def _naca_symmetric(t: float, n: int = 100) -> np.ndarray:
    """NACA 4-digit symmetric airfoil as [[x, y], ...] going LE→upper→TE→lower→LE."""
    x = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n)))
    yt = (t / 0.2) * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )
    upper = np.column_stack([x, yt])
    lower = np.column_stack([x[::-1], -yt[::-1]])
    return np.vstack([upper, lower[1:]])


airfoils_data = {
    "NACA 0012 (t=12%)": {"data": _naca_symmetric(0.12), "thickness": 0.12},
    "NACA 0018 (t=18%)": {"data": _naca_symmetric(0.18), "thickness": 0.18},
    "NACA 0024 (t=24%)": {"data": _naca_symmetric(0.24), "thickness": 0.24},
    "NACA 0030 (t=30%)": {"data": _naca_symmetric(0.30), "thickness": 0.30},
    "NACA 0040 (t=40%)": {"data": _naca_symmetric(0.40), "thickness": 0.40},
}

out = Path("examples/ex01_airfoils.png")
out.parent.mkdir(exist_ok=True)
plot_airfoils(airfoils_data, n_points=120, output_file=str(out))
print(f"  plot → file://{out.resolve()}")
