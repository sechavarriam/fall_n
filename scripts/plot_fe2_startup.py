# Estrategias de arranque del acople two-way FE2 (cap. 107): inmediato,
# warmup one-way corto (8), warmup largo (20) y retraso puro con enganche
# virgen en la reversa (20). Leccion empirica: CUANDO se engancha importa
# mas que COMO -- el mejor punto es una reversa (rama elastica unica), y la
# historia trackeada en one-way llega INCONSISTENTE con el equilibrio
# acoplado (warmup largo = el peor).
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
C_IMM = "#4a3aa7"   # slot 7: two-way caras (entidad establecida)
C_W8 = "#1baf7a"    # slot 5: warmup corto
C_W20 = "#eda100"   # slot 4: warmup largo
C_D20 = "#008300"   # slot 2: enganche en reversa (el ganador)


def load(rel):
    xs, ys = [], []
    with open(os.path.join(ROOT, rel), newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["drift_m"]) * 1e3)
            ys.append(-float(row["base_shear_MN"]) * 1e3)
    return xs, ys


g0 = load(os.path.join("g0_baseline", "cyc", "lmnewton_hysteresis.csv"))
runs = [
    ("Inmediato (paso 1)", C_IMM,
     load(os.path.join("fe2_col", "two_way_boundary",
                       "fe2_column_hysteresis.csv")),
     "pico 64 kN · 5 pasos >30 kN · rugosidad 1.15 MN"),
    ("Warmup one-way corto (8)", C_W8,
     load(os.path.join("fe2_col", "two_way_warmup8",
                       "fe2_column_hysteresis.csv")),
     "pico 29 kN · 0 pasos >30 kN · rugosidad 0.77 MN"),
    ("Warmup one-way largo (20)", C_W20,
     load(os.path.join("fe2_col", "two_way_warmup20",
                       "fe2_column_hysteresis.csv")),
     "pico 128 kN · historia one-way INCONSISTENTE al enganchar"),
    ("Enganche virgen en la reversa (20)", C_D20,
     load(os.path.join("fe2_col", "two_way_delay20",
                       "fe2_column_hysteresis.csv")),
     "pico 27 kN · 0 pasos >30 kN · rugosidad 0.60 MN — el más suave"),
]

fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.3), dpi=200,
                         sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

for ax, (title, color, (xs, ys), stats) in zip(axes, runs):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.plot(g0[0], g0[1], color=MUTED, linewidth=0.9, alpha=0.9, zorder=2)
    ax.plot(xs, ys, color=color, linewidth=1.5, zorder=3)
    ax.set_title(title, color=INK, fontsize=10.5, pad=28)
    ax.text(0.5, 1.015, stats, transform=ax.transAxes, ha="center",
            va="bottom", color=INK2, fontsize=7.4)
    ax.set_xlabel("deriva  [mm]", color=INK2, fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.0)

axes[0].set_ylabel("V = −reacción  [kN]", color=INK2, fontsize=9)
axes[0].set_ylim(-108, 132)

fig.legend(handles=[
    plt.Line2D([], [], color=MUTED, linewidth=1.1,
               label="baseline monolítico g0 (referencia)")],
    loc="lower center", ncol=1, frameon=False, fontsize=8.5,
    labelcolor=INK2, bbox_to_anchor=(0.5, -0.006))

fig.suptitle(
    "Estrategias de arranque del two-way FE² — cuándo enganchar importa "
    "más que cómo",
    color=INK, fontsize=12.5, y=1.0)

fig.tight_layout(rect=(0, 0.05, 1, 0.90))
out = os.path.join(ROOT, "fe2_startup_strategies.png")
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("escrita:", out)
