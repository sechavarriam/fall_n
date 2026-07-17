# Figura VERSION TESIS de las curvas de convergencia del programa CA.
# Dos paneles con escalas propias: (a) replicas de ventana monoliticas
# (ca2/ca3, aptitud v2 anclada a la meseta fisica), (b) afinador del
# acople FE2. Linea continua = mejor aptitud, discontinua = media
# poblacional. Salida: PhD_Thesis/Figuras/validacion/ca_convergencia.pdf
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:\MyLibs\fall_n\data\output"
OUT = r"c:\MyLibs\fall_n\PhD_Thesis\Figuras\validacion\ca_convergencia.pdf"

MONO = [
    ("ventana 112:12, 2 fuentes", "#4a3aa7",
     r"ca2_w112\ca_frac_1.0\ca_history.csv"),
    ("ventana 156:12, 2 fuentes", "#e34948",
     r"ca2_w156\ca_frac_1.0\ca_history.csv"),
    ("ventana 112:12, 5 fuentes", "#1f7a5a",
     r"ca3_w112_5src\ca_frac_1.0\ca_history.csv"),
    ("ventana 156:12, 5 fuentes", "#b3701c",
     r"ca3_w156_5src\ca_frac_1.0\ca_history.csv"),
    ("ventana 156:12, semilla alterna", "#898781",
     r"ca3_w156_seedB\ca_frac_1.0\ca_history.csv"),
]
FE2 = [("acople FE² (4 rasgos)", "#2a6db0", r"ca_fe2_night\ca_history.csv")]


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


plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.6,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6.6, 2.9), dpi=300,
    gridspec_kw={"width_ratios": [1.35, 1.0]})

for ax in (ax1, ax2):
    ax.grid(True, color="#e1e0d9", linewidth=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

for label, color, rel in MONO:
    g, b, m = load(rel)
    if not g:
        continue
    ax1.plot(g, b, color=color, linewidth=1.3, label=label, zorder=3)
    ax1.plot(g, m, color=color, linewidth=0.7, linestyle=(0, (3, 2)),
             alpha=0.55, zorder=2)

for label, color, rel in FE2:
    g, b, m = load(rel)
    if not g:
        continue
    ax2.plot(g, b, color=color, linewidth=1.3, label=label, zorder=3)
    ax2.plot(g, m, color=color, linewidth=0.7, linestyle=(0, (3, 2)),
             alpha=0.55, zorder=2)

ax1.set_title("(a) réplicas de ventana del solucionador")
ax2.set_title("(b) afinador del acople FE$^2$")
for ax in (ax1, ax2):
    ax.set_xlabel("generación")
ax1.set_ylabel("aptitud")
ax1.legend(frameon=False, loc="lower right")
ax2.legend(frameon=False, loc="lower right")

fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print("figura tesis:", OUT)
