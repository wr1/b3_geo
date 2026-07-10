"""ex05 — airfoil stack: the section shapes a blade blends between.

UACS shapes sampled at several span stations, stacked with a vertical
offset (root at the bottom, tip at the top).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from b3_blade.templates.blade_mini import BladeMini
from b3_geo import get_wire
from b3_plot._style import configure, style_ax

configure()
blade = BladeMini()
pf, stack = blade.planform(), blade.airfoil_stack()

fig, ax = plt.subplots(figsize=(8, 5))
stations = [0.0, 0.25, 0.5, 0.75, 1.0]
for k, s in enumerate(stations):
    uacs = get_wire(pf, stack, s=s).point_data["geo.uacs"]
    ax.plot(uacs[:, 0], uacs[:, 1] + 0.35 * k, lw=1.2, label=f"s={s:.2f}")
ax.set_aspect("equal")
ax.set_xlabel("chord fraction")
ax.legend(fontsize=8, loc="upper right")
ax.set_title("Airfoil stack — UACS shapes along span", fontsize=11)
style_ax(ax)
fig.tight_layout()

out = Path("examples/ex05_airfoil_stack.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"  plot → file://{out.resolve()}")
