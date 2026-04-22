"""Plot planform curves with control-point markers and interpolated lines."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from b3_blade.curve import Curve

yml = "/home/wr1/b3mc_work/blade_r1_te.yml"
out = "/tmp/planform_r1_te.png"

with open(yml) as f:
    data = yaml.safe_load(f)

pf = data["geometry"]["planform"]

# Each entry in pf is a list of [s, v] pairs
def load(key):
    pts = pf[key]  # [[s, v], ...]
    return np.array(pts), Curve.from_points([tuple(p) for p in pts])

keys = [
    ("chord",     "Chord (m)"),
    ("thickness", "t/c"),
    ("twist",     "Twist (deg)"),
    ("dx",        "dx — flapwise prebend (m)"),
    ("dy",        "dy — edgewise sweep (m)"),
]

fig, axes = plt.subplots(len(keys), 1, figsize=(9, 12), sharex=True)

s_fine = np.linspace(0.0, 1.0, 500)

for ax, (key, label) in zip(axes, keys):
    pts_arr, curve = load(key)
    s_pts = pts_arr[:, 0]
    v_pts = pts_arr[:, 1]

    v_fine = curve.at(s_fine)

    ax.plot(s_fine, v_fine, "b-", lw=1.5, label="interpolated")
    ax.scatter(s_pts, v_pts, color="orange", s=40, zorder=5, label="control pts")
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, lw=0.4, alpha=0.5)
    ax.legend(fontsize=7, loc="best")

axes[-1].set_xlabel("Rel. span s")
fig.suptitle("Planform — blade_r1_te", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"saved → {out}")
