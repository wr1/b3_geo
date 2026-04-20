"""ex02 — Blade planform: interpolated curves with control points.

Demonstrates plot_planform with a realistic 62 m blade parameterisation.
"""

import os
import numpy as np
from pathlib import Path
from b3_geo.utils.plotting import plot_planform

rel_span = np.linspace(0, 1, 120)

interpolated = {
    "z":                  rel_span * 62.0,
    "chord":              4.5 * np.exp(-2.0 * rel_span) + 1.0 * (1.0 - np.exp(-2.0 * rel_span)),
    "thickness":          0.24 - (0.24 - 0.12) * rel_span,
    "twist":              13.0 * np.exp(-3.5 * rel_span),
    "dx":                 np.zeros(120),
    "dy":                 np.zeros(120),
    "absolute_thickness": (0.24 - (0.24 - 0.12) * rel_span) * (4.5 * np.exp(-2.0 * rel_span) + 1.0 * (1.0 - np.exp(-2.0 * rel_span))),
}

controls = {
    "z":       [(0.0, 0.0),   (0.5, 31.0), (1.0, 62.0)],
    "chord":   [(0.0, 4.5),   (0.1, 4.2),  (0.4, 2.8), (1.0, 1.2)],
    "thickness":[(0.0, 0.24), (0.5, 0.18), (1.0, 0.12)],
    "twist":   [(0.0, 13.0),  (0.2, 8.0),  (0.6, 2.5), (1.0, 0.0)],
    "dx":      [(0.0, 0.0),   (1.0, 0.0)],
    "dy":      [(0.0, 0.0),   (1.0, 0.0)],
}

out = Path("examples/ex02_planform.png")
out.parent.mkdir(exist_ok=True)
plot_planform(interpolated, controls, rel_span, output_file=str(out))
print(f"  plot → file://{out.resolve()}")
