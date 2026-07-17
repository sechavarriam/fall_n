# Curvas de convergencia del programa CA: best/mean por generacion para cada
# campana con ca_history.csv (replay monolitico ca2/ca3 y afinador FE2).
# Salida: data/output/ca_convergence.png
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\MyLibs\fall_n\data\output"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

CAMPAIGNS = [
    ("ca2_w112 (2 fuentes)", "#4a3aa7", r"ca2_w112\ca_frac_1.0\ca_history.csv"),
    ("ca2_w156 (2 fuentes)", "#e34948", r"ca2_w156\ca_frac_1.0\ca_history.csv"),
    ("ca3_w112 (5 fuentes)", "#1f7a5a", r"ca3_w112_5src\ca_frac_1.0\ca_history.csv"),
    ("ca3_w156 (5 fuentes)", "#b3701c", r"ca3_w156_5src\ca_frac_1.0\ca_history.csv"),
    ("ca3_w156 (semilla B)", "#7a3a8f", r"ca3_w156_seedB\ca_frac_1.0\ca_history.csv"),
    ("CA-FE2 (acople)", "#2a6db0", r"ca_fe2_night\ca_history.csv"),
]


def load(rel):
    gens, best, mean = [], [], []
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        return gens, best, mean
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            try:
                gens.append(int(row["gen"]))
                best.append(float(row["best"]))
                mean.append(float(row["mean"]))
            except (ValueError, KeyError):
                break
    return gens, best, mean


fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=170)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(True, color=GRID, linewidth=0.6, zorder=0)

n_drawn = 0
for label, color, rel in CAMPAIGNS:
    gens, best, mean = load(rel)
    if not gens:
        continue
    n_drawn += 1
    ax.plot(gens, best, color=color, linewidth=1.6, label=label, zorder=3)
    ax.plot(gens, mean, color=color, linewidth=0.9, linestyle=(0, (3, 2)),
            alpha=0.65, zorder=2)

ax.set_xlabel("generación", color=INK2, fontsize=10)
ax.set_ylabel("aptitud (— mejor, ‑‑ media)", color=INK2, fontsize=10)
ax.set_title("Programa CA — convergencia por generación "
             "(aptitud anclada a la meseta física)",
             color=INK, fontsize=11.5)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.tick_params(colors=MUTED, labelsize=8.5)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")

out = os.path.join(ROOT, "ca_convergence.png")
fig.tight_layout()
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("figura:", out, f"({n_drawn} campañas)")
