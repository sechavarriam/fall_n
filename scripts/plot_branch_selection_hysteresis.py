# Lazos de histeresis comparativos: G0 baseline vs los tres remedios env-gated
# de la extension de seleccion de rama (bitacora cap. 105, Fase I).
#
# Pequenos multiplos (un remedio por panel), G0 en gris de referencia al fondo,
# misma escala en los tres paneles. El CSV guarda la REACCION del apoyo; se
# voltea el signo para la convencion estructural (V positivo con deriva
# positiva).
#
# Entradas (gitignoradas, regenerables con los scripts kobathe_*):
#   data/output/g0_baseline/cyc/lmnewton_hysteresis.csv
#   data/output/p1_deflation_final/cyc/lmnewton_hysteresis.csv
#   data/output/p2_tao/cyc/lmnewton_hysteresis.csv
#   data/output/p3_ca/full/lmnewton_hysteresis.csv
# Salida: data/output/branch_selection_hysteresis.png
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                 "output"))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#008300", "#e87ba4"   # slots categoricos 1-3


def load(rel):
    xs, ys = [], []
    with open(os.path.join(ROOT, rel), newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["drift_m"]) * 1e3)           # mm
            ys.append(-float(row["base_shear_MN"]) * 1e3)    # kN (accion)
    return xs, ys


g0 = load(os.path.join("g0_baseline", "cyc", "lmnewton_hysteresis.csv"))
runs = [
    ("Deflación (retry + fallback)", S1,
     load(os.path.join("p1_deflation_final", "cyc",
                       "lmnewton_hysteresis.csv")),
     "noconv 80 · pico 50.8 kN — bit-idéntica a G0"),
    ("Híbrido TAO en reversas", S2,
     load(os.path.join("p2_tao", "cyc", "lmnewton_hysteresis.csv")),
     "noconv 77 · pico 37.7 kN — espiga del paso 117 domada"),
    ("LM afinado por CA (genoma 116:6)", S3,
     load(os.path.join("p3_ca", "full", "lmnewton_hysteresis.csv")),
     "noconv 66 · pico 48.1 kN — costo cero por paso"),
]

fig, axes = plt.subplots(
    1, 3, figsize=(13.8, 4.9), dpi=200, sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, (title, color, (xs, ys), stats) in zip(axes, runs):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.plot(g0[0], g0[1], color=MUTED, linewidth=1.0, alpha=0.9, zorder=2,
            label="G0 baseline")
    ax.plot(xs, ys, color=color, linewidth=1.7, zorder=3, label=title)
    ax.set_title(title, color=INK, fontsize=11.5, pad=16)
    ax.text(0.5, 1.015, stats, transform=ax.transAxes, ha="center",
            va="bottom", color=INK2, fontsize=8.6)
    ax.set_xlabel("deriva lateral  [mm]", color=INK2, fontsize=9.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)

axes[0].set_ylabel("cortante basal  V = −reacción  [kN]", color=INK2,
                   fontsize=9.5)
axes[0].set_ylim(-62, 62)

# Anotacion selectiva: la espiga espuria de G0 (paso 117) en el panel TAO
# (el remedio que la doma), en la banda superior vacia.
gx, gy = g0
axes[1].annotate(
    "espiga espuria G0\n(paso 117, 47.7 kN)",
    xy=(gx[117], gy[117]), xytext=(-30, 52),
    color=INK2, fontsize=8.2, ha="center", va="center",
    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkB=4))

handles = [
    plt.Line2D([], [], color=MUTED, linewidth=1.2,
               label="G0 baseline (referencia)"),
    plt.Line2D([], [], color=S1, linewidth=1.8, label="deflación"),
    plt.Line2D([], [], color=S2, linewidth=1.8, label="híbrido TAO"),
    plt.Line2D([], [], color=S3, linewidth=1.8, label="LM afinado por CA"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))

fig.suptitle(
    "Lazos de histéresis del protocolo ±50/100/150/200 mm — "
    "baseline G0 vs remedios de selección de rama",
    color=INK, fontsize=13, y=0.99)

fig.tight_layout(rect=(0, 0.05, 1, 0.90))
out = os.path.join(ROOT, "branch_selection_hysteresis.png")
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("escrita:", out)
