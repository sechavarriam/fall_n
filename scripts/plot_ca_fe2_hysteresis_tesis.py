# Figura VERSION TESIS: lazos de histeresis del retorno bidireccional.
# Referencia estructural Timoshenko (gris, detras), acople afinado a mano
# (caso D, limitador 0.25 + tope 12) y optimo encontrado por el CA
# (limitador 0.165, relax 0.769, tope 19, enganche 39).
# Salida: PhD_Thesis/Figuras/validacion/ca_fe2_histeresis.pdf
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\MyLibs\fall_n\data\output"
OUT = r"c:\MyLibs\fall_n\PhD_Thesis\Figuras\validacion\ca_fe2_histeresis.pdf"

SERIES = [
    ("referencia estructural (Timoshenko)", "#9a988f", 1.6, "-", 1,
     r"fe2_col\macro_only\fe2_column_hysteresis.csv"),
    ("acople afinado a mano (caso D)", "#b3701c", 0.9, (0, (4, 2)), 2,
     r"fe2_col\two_way_delay_gap25\fe2_column_hysteresis.csv"),
    ("óptimo del algoritmo cultural", "#2a6db0", 1.1, "-", 3,
     r"ca_fe2_best_full\fe2_column_hysteresis.csv"),
]


def load(rel):
    xs, ys = [], []
    with open(os.path.join(ROOT, rel), newline="") as f:
        for row in csv.DictReader(f):
            try:
                xs.append(float(row["drift_m"]) * 1e3)
                ys.append(-float(row["base_shear_MN"]) * 1e3)
            except (ValueError, KeyError):
                break
    return xs, ys


plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=300)
ax.grid(True, color="#e1e0d9", linewidth=0.5, zorder=0)
ax.axhline(0.0, color="#c3c2b7", linewidth=0.7, zorder=1)
ax.axvline(0.0, color="#c3c2b7", linewidth=0.7, zorder=1)

for label, color, lw, ls, z, rel in SERIES:
    xs, ys = load(rel)
    ax.plot(xs, ys, color=color, linewidth=lw, linestyle=ls,
            label=label, zorder=1 + z)

ax.set_xlabel("deriva de techo [mm]")
ax.set_ylabel(r"cortante basal $V=-R_x$ [kN]")
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.legend(frameon=False, loc="upper left")

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("figura tesis:", OUT)
