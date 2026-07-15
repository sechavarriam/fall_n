# Lazos de histeresis comparativos, SEGUNDA RONDA de remedios (bitacora
# cap. 106): line search de energia en el LM, hibrido TAO y el combo
# TAO+pulido+line search, cada uno contra el baseline G0 en gris.
# Mismo sistema visual que plot_branch_selection_hysteresis.py; los colores
# siguen a la ENTIDAD (orden fijo de la paleta: deflacion=1, TAO=2, CA=3,
# line search=4, combo=5).
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
S_TAO = "#008300"     # slot 2 (TAO conserva su color de la ronda 1)
S_LS = "#eda100"      # slot 4 (line search)
S_COMBO = "#1baf7a"   # slot 5 (combo)


def load(rel):
    xs, ys = [], []
    with open(os.path.join(ROOT, rel), newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["drift_m"]) * 1e3)           # mm
            ys.append(-float(row["base_shear_MN"]) * 1e3)    # kN (accion)
    return xs, ys


g0 = load(os.path.join("g0_baseline", "cyc", "lmnewton_hysteresis.csv"))
runs = [
    ("LM + line search de energía", S_LS,
     load(os.path.join("p1_linesearch", "cyc", "lmnewton_hysteresis.csv")),
     "noconv 78 · pico 49.3 kN · espiga 117: 10.7 kN — costo ~cero"),
    ("Híbrido TAO en reversas", S_TAO,
     load(os.path.join("p2_tao", "cyc", "lmnewton_hysteresis.csv")),
     "noconv 77 · pico 37.7 kN · espiga 117: 18.9 kN"),
    ("TAO + pulido + line search", S_COMBO,
     load(os.path.join("p2_tao_ls_combo", "cyc", "lmnewton_hysteresis.csv")),
     "noconv 77 · pico 55.4 kN (en +200 mm de carga) · espiga 117: 20.2 kN"),
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
            va="bottom", color=INK2, fontsize=8.2)
    ax.set_xlabel("deriva lateral  [mm]", color=INK2, fontsize=9.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)

axes[0].set_ylabel("cortante basal  V = −reacción  [kN]", color=INK2,
                   fontsize=9.5)
axes[0].set_ylim(-62, 62)

gx, gy = g0
axes[0].annotate(
    "espiga espuria G0\n(paso 117, 47.7 kN)",
    xy=(gx[117], gy[117]), xytext=(-30, 52),
    color=INK2, fontsize=8.2, ha="center", va="center",
    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.7, shrinkB=4))

handles = [
    plt.Line2D([], [], color=MUTED, linewidth=1.2,
               label="G0 baseline (referencia)"),
    plt.Line2D([], [], color=S_LS, linewidth=1.8, label="line search"),
    plt.Line2D([], [], color=S_TAO, linewidth=1.8, label="híbrido TAO"),
    plt.Line2D([], [], color=S_COMBO, linewidth=1.8, label="combo"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))

fig.suptitle(
    "Lazos de histéresis — segunda ronda de remedios de selección de rama "
    "vs baseline G0",
    color=INK, fontsize=13, y=0.99)

fig.tight_layout(rect=(0, 0.05, 1, 0.90))
out = os.path.join(ROOT, "branch_selection_hysteresis_round2.png")
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("escrita:", out)
