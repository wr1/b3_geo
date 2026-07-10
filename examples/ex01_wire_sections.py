"""ex01 — W vtk: all section wires of a blade in one PolyData.

The W vtk is b3_geo's product: one closed polyline per span station,
coloured here by relative span.
"""

from pathlib import Path

from b3_blade.templates.blade_mini import BladeMini
from b3_geo import build_w_vtk
from b3_plot.three_d.wire import plot_wire

blade = BladeMini()
w = build_w_vtk(blade.planform(), blade.airfoil_stack(), n_sections=40)

out = Path("examples/ex01_wire_sections.png")
out.parent.mkdir(exist_ok=True)
plot_wire(w, out=out, scalar="rel_span")
print(f"  plot → file://{out.resolve()}")
