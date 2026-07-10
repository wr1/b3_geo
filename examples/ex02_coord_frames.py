"""ex02 — one section in its three frames: UACS, SDACS, GBCS.

UACS: unit airfoil (chord in [0,1]). SDACS: scaled + detwisted physical
section. GBCS points: twist baked in, placed at the section's z.
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
wire = get_wire(blade.planform(), blade.airfoil_stack(), s=0.5)

uacs = wire.point_data["geo.uacs"]
sdacs = wire.point_data["geo.sdacs"]
gbcs = wire.points

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, xy, title in (
    (axes[0], uacs, "UACS — unit airfoil"),
    (axes[1], sdacs, "SDACS — scaled, detwisted"),
    (axes[2], gbcs[:, :2], "GBCS — twist baked in"),
):
    ax.plot(xy[:, 0], xy[:, 1], lw=1.2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    style_ax(ax)

fig.suptitle("Section s=0.5 across coordinate frames", fontsize=11)
fig.tight_layout()

out = Path("examples/ex02_coord_frames.png")
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(f"  plot → file://{out.resolve()}")
