"""ex04 — TE distance fields: geo.arc_from_te_ss on the section wires.

Every W-vtk point carries arc distances from the trailing edge (per side)
— the fields laminate masks and web filters are built on.
"""

from pathlib import Path

from b3_blade.templates.blade_mini import BladeMini
from b3_geo import build_w_vtk
from b3_plot.three_d.wire import plot_wire

blade = BladeMini()
w = build_w_vtk(blade.planform(), blade.airfoil_stack(), n_sections=40)

out = Path("examples/ex04_te_fields.png")
out.parent.mkdir(exist_ok=True)
plot_wire(w, out=out, scalar="geo.arc_from_te_ss")
print(f"  plot → file://{out.resolve()}")
