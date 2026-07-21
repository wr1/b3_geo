"""Build and open W vtk for blade_r1_te in ParaView."""

import sys
import numpy as np
import yaml

from b3_blade.curve import Curve
from b3_blade.planform import Planform
from b3_blade.airfoil_stack import AirfoilStack
from b3_aerofoil import load_dat

from b3_geo.build import build_w_vtk
from b3_geo.schema import validate_w_vtk

YML  = "/home/wr1/b3mc_work/blade_r1_te.yml"
WDIR = "/home/wr1/b3mc_work"
OUT  = "/tmp/r1_te.vtp"

with open(YML) as f:
    data = yaml.safe_load(f)

pf_data = data["geometry"]["planform"]

def curve(key):
    return Curve.from_points([tuple(p) for p in pf_data[key]])

z_pts = pf_data["z"]
span = float(z_pts[-1][1]) - float(z_pts[0][1])

planform = Planform(
    span  = span,
    chord = curve("chord"),
    tc    = curve("thickness"),
    twist = curve("twist"),
    dx    = curve("dx"),
    dy    = curve("dy"),
    z     = curve("z"),
)

# Build airfoil stack
import os
foils = []
for entry in data["airfoils"]:
    path = os.path.join(WDIR, entry["path"])
    foil = load_dat(path)
    tc   = entry["thickness"]
    foils.append((tc, foil))

stack = AirfoilStack(foils)

print("Building W vtk …")
w = build_w_vtk(planform, stack, n_sections=80, n_chord=200)
validate_w_vtk(w)
w.save(OUT)
print(f"Saved → {OUT}")
print(f"  {w.n_points} points, {w.n_lines} polylines")
