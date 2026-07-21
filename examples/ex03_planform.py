"""ex03 — planform curves: the spanwise distributions behind the W vtk.

Samples the template's Planform curves (chord, twist, t/c, prebend, sweep)
over relative span.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from b3_blade.templates.blade_mini import BladeMini
from b3_plot._style import configure, style_ax

configure()
pf = BladeMini().planform()
s = np.linspace(0.0, 1.0, 300)

curves = [
    ("chord", "chord (m)"),
    ("twist", "twist (deg)"),
    ("tc", "t/c"),
    ("dx", "dx — prebend (m)"),
    ("dy", "dy — sweep (m)"),
]
fig, axes = plt.subplots(len(curves), 1, figsize=(8, 9), sharex=True)
for ax, (name, label) in zip(axes, curves, strict=True):
    ax.plot(s, getattr(pf, name).at(s), lw=1.4)
    ax.set_ylabel(label, fontsize=9)
    style_ax(ax)
axes[-1].set_xlabel("rel. span s")
fig.suptitle("Planform curves — BladeMini", fontsize=11)
fig.tight_layout()

out = Path("examples/ex03_planform.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"  plot → file://{out.resolve()}")
