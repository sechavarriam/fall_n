#!/usr/bin/env python3
"""Generate roof-displacement time-history figure for ch68 (L-shaped RC building)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
csv_path = Path(r"C:\MyLibs\fall_n\data\output\lshaped_rc\roof_displacement.csv")
out_dir  = Path(r"C:\MyLibs\fall_n\doc\figures\lshaped_rc")

# ── load ───────────────────────────────────────────────────────────────
data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
t   = data[:, 0]          # time [s]
# columns: node48_dof0, node48_dof1, node51_dof0, node51_dof1,
#          node56_dof0, node56_dof1, node58_dof0, node58_dof1

# convert to mm
ux48, uy48 = data[:, 1] * 1e3, data[:, 2] * 1e3   # node (0,0,roof)
ux51, uy51 = data[:, 3] * 1e3, data[:, 4] * 1e3   # node (3,0,roof)
ux56, uy56 = data[:, 5] * 1e3, data[:, 6] * 1e3   # node (0,2,roof)
ux58, uy58 = data[:, 7] * 1e3, data[:, 8] * 1e3   # node (2,2,roof)

# ── style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 0.7,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

node_labels = {
    "48": "(0,0)",
    "51": "(3,0)",
    "56": "(0,2)",
    "58": "(2,2)",
}

colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

# ── figure : X and Y displacement histories (2 subplots) ──────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 4.5), sharex=True)

for ax, comp, traces, ylabel in [
    (ax1, "X", [(ux48, "48"), (ux51, "51"), (ux56, "56"), (ux58, "58")],
     r"$u_x$ [mm]"),
    (ax2, "Y", [(uy48, "48"), (uy51, "51"), (uy56, "56"), (uy58, "58")],
     r"$u_y$ [mm]"),
]:
    for (u, nid), c in zip(traces, colors):
        ax.plot(t, u, color=c, label=f"node {node_labels[nid]}")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", ncol=2, framealpha=0.8)

ax2.set_xlabel("Time [s]")
ax1.set_title("Roof displacement time histories — L-shaped RC building")
fig.tight_layout()
fig.savefig(out_dir / "roof_displacement.pdf", bbox_inches="tight")
fig.savefig(out_dir / "roof_displacement.png", bbox_inches="tight", dpi=200)
print(f"Saved: {out_dir / 'roof_displacement.pdf'}")
print(f"Saved: {out_dir / 'roof_displacement.png'}")

# ── figure 2 : X-Y orbit at one node ──────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(4.0, 4.0))
ax3.plot(ux51, uy51, color=colors[1], linewidth=0.5, alpha=0.8)
ax3.plot(ux51[0], uy51[0], "o", color="green", ms=5, zorder=5, label="start")
ax3.plot(ux51[-1], uy51[-1], "s", color="red", ms=5, zorder=5, label="end")
ax3.set_xlabel(r"$u_x$ [mm]")
ax3.set_ylabel(r"$u_y$ [mm]")
ax3.set_title(r"Roof orbit — node (3,0)")
ax3.set_aspect("equal", adjustable="datalim")
ax3.legend(loc="upper left", fontsize=8)
fig2.tight_layout()
fig2.savefig(out_dir / "roof_orbit.pdf", bbox_inches="tight")
fig2.savefig(out_dir / "roof_orbit.png", bbox_inches="tight", dpi=200)
print(f"Saved: {out_dir / 'roof_orbit.pdf'}")
print(f"Saved: {out_dir / 'roof_orbit.png'}")

# ── figure 3 : damage evolution ────────────────────────────────────────
dmg_csv = Path(r"C:\MyLibs\fall_n\data\output\lshaped_rc\damage_evolution.csv")
import csv
t_dmg, d_dmg, phase = [], [], []
with open(dmg_csv, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        t_dmg.append(float(row["time"]))
        d_dmg.append(float(row["peak_dmg"]))
        phase.append(row["phase"].strip() if row["phase"] else "phase1")

t_dmg = np.array(t_dmg)
d_dmg = np.array(d_dmg)
phase = np.array(phase)

fig3, ax4 = plt.subplots(figsize=(6.5, 3.0))
mask1 = phase == "phase1"
mask2 = phase == "phase2"
ax4.plot(t_dmg[mask1], d_dmg[mask1], "o-", color="#1f77b4", ms=2.5,
         linewidth=0.9, label="Phase 1 (global)")
ax4.plot(t_dmg[mask2], d_dmg[mask2], "s-", color="#d62728", ms=3.5,
         linewidth=0.9, label="Phase 2 (sub-models)")
ax4.axhline(0.80, color="gray", ls="--", lw=0.7, label=r"$d_{\mathrm{thr}}=0.80$")
ax4.set_xlabel("Time [s]")
ax4.set_ylabel(r"Peak damage index $d$")
ax4.set_title("Damage evolution — L-shaped RC building")
ax4.legend(loc="upper left", fontsize=8)
ax4.set_ylim(-0.02, 1.0)
fig3.tight_layout()
fig3.savefig(out_dir / "damage_evolution.pdf", bbox_inches="tight")
fig3.savefig(out_dir / "damage_evolution.png", bbox_inches="tight", dpi=200)
print(f"Saved: {out_dir / 'damage_evolution.pdf'}")
print(f"Saved: {out_dir / 'damage_evolution.png'}")

plt.close("all")
