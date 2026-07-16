# Matriz de decision del merge: remedios sobre las DOS fisicas
# (tau_o transcrito viejo vs octaedrico corregido). Barras agrupadas.
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_OLD = "#898781"   # fisica vieja = referencia gris (entidad ya establecida)
C_NEW = "#e34948"   # slot 8: fisica corregida tau-oct

cats = ["baseline", "line search", "TAO", "combo"]
noconv_old = [80, 78, 77, 77]
noconv_new = [100, 97, 96, 92]
peak_old = [50.8, 49.3, 37.7, 55.4]
peak_new = [84.4, 58.8, 123.4, 58.2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE)

x = np.arange(len(cats))
w = 0.36

for ax, old, new, title, ylab in (
    (ax1, noconv_old, noconv_new, "Pasos regularizados (noconv) de 196",
     "noconv  [–]"),
    (ax2, peak_old, peak_new, "Pico global de |V|", "|V|max  [kN]"),
):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6, zorder=0)
    b1 = ax.bar(x - w / 2, old, w, color=C_OLD, zorder=3,
                label="física vieja (τ_o transcrito)")
    b2 = ax.bar(x + w / 2, new, w, color=C_NEW, zorder=3,
                label="física corregida (τ_o octaédrico)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1,
                    f"{r.get_height():.0f}", ha="center", va="bottom",
                    color=INK2, fontsize=8.0)
    ax.set_xticks(x, cats, color=INK2, fontsize=9.5)
    ax.set_title(title, color=INK, fontsize=11.5)
    ax.set_ylabel(ylab, color=INK2, fontsize=9.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)

ax1.set_ylim(0, 118)
ax2.set_ylim(0, 140)

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))

fig.suptitle(
    "Matriz de decisión — remedios del monolítico sobre las dos físicas "
    "(protocolo ±50/100/150/200 mm)",
    color=INK, fontsize=12.5, y=0.99)

fig.tight_layout(rect=(0, 0.06, 1, 0.92))
out = r"c:\MyLibs\fall_n\data\output\two_physics_matrix.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("escrita:", out)
