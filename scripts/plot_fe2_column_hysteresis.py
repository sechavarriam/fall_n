# Lazos de histeresis del experimento FE2 de columna (bitacora cap. 106):
# macro fibra (= one-way), two-way con reacciones de borde (fix
# evolve_locals_in_hybrid, protocolo completo) y two-way con promedio
# volumetrico (aborta en el paso 40: par (f_vol, D_bnd) inconsistente).
# Referencia gris: el monolitico continuo 3D g0.
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
S_MACRO = "#eb6834"   # slot 6 (macro fibra / one-way)
S_TWB = "#4a3aa7"     # slot 7 (two-way, reacciones de borde)
S_TWV = "#e34948"     # slot 8 (two-way, promedio volumetrico)


def load(path):
    xs, ys = [], []
    with open(os.path.join(ROOT, path), newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["drift_m"]) * 1e3)
            ys.append(-float(row["base_shear_MN"]) * 1e3)
    return xs, ys


g0 = load(os.path.join("g0_baseline", "cyc", "lmnewton_hysteresis.csv"))
macro = load(os.path.join("fe2_col", "macro_only",
                          "fe2_column_hysteresis.csv"))
twb = load(os.path.join("fe2_col", "two_way_boundary",
                        "fe2_column_hysteresis.csv"))
twv = load(os.path.join("fe2_col", "two_way_volume",
                        "fe2_column_hysteresis.csv"))

twl = load(os.path.join("fe2_col", "two_way_layers",
                        "fe2_column_hysteresis.csv"))

runs = [
    ("Macro fibra (= one-way)", S_MACRO, macro, False,
     "196/196 · 0 no conv · pico 23.7 kN — suave, sin fidelidad 3D"),
    ("Two-way, caras extremas", S_TWB, twb, False,
     "196/196 · 0 no conv · transitorio inicial (5-19) · lazo global más liso"),
    ("Two-way, capas interiores (B1)", S_TWV, twl, False,
     "196/196 · 49 estrictos · local impecable (energy-lm idéntico) · "
     "global más rugoso: sobre-restricción"),
]

fig, axes = plt.subplots(
    1, 3, figsize=(13.8, 4.9), dpi=200, sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, (title, color, (xs, ys), partial, stats) in zip(axes, runs):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.plot(g0[0], g0[1], color=MUTED, linewidth=1.0, alpha=0.9, zorder=2)
    ax.plot(xs, ys, color=color, linewidth=1.7, zorder=3)
    if partial:
        ax.plot(xs[-1], ys[-1], "o", color=color, markersize=6, zorder=4)
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
axes[0].set_ylim(-70, 70)

handles = [
    plt.Line2D([], [], color=MUTED, linewidth=1.2,
               label="monolítico G0 (continuo 3D, referencia)"),
    plt.Line2D([], [], color=S_MACRO, linewidth=1.8, label="macro fibra"),
    plt.Line2D([], [], color=S_TWB, linewidth=1.8, label="two-way caras"),
    plt.Line2D([], [], color=S_TWV, linewidth=1.8, label="two-way capas"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))

fig.suptitle(
    "Experimento FE² de columna (macro Timoshenko + RVE KoBathe en la rótula,"
    " evolve_locals_in_hybrid) vs monolítico G0",
    color=INK, fontsize=12.5, y=0.99)

fig.tight_layout(rect=(0, 0.05, 1, 0.90))
out = os.path.join(ROOT, "fe2_column_hysteresis.png")
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("escrita:", out)
