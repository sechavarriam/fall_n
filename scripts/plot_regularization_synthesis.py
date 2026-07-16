# Figuras de sintesis del programa de regularizacion de reversas Ko-Bathe
# (bitacora caps. 99-107). Genera dos PNG en data/output:
#   regularization_principles.png : los 4 principios (evitacion, descenso,
#                                   control, ajuste) vs baseline g0
#   fe2_control_dial.png          : el dial de control cinematico del FE2
#                                   (one-way -> caras -> capas) con metricas
# Colores por ENTIDAD, consistentes con las figuras previas de la campana.
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                 "output"))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_DEFL = "#2a78d6"    # slot 1: deflacion
C_LS = "#eda100"      # slot 4: line search
C_CA = "#e87ba4"      # slot 3: CA
C_MACRO = "#eb6834"   # slot 6: macro fibra
C_FACES = "#4a3aa7"   # slot 7: two-way caras
C_LAYERS = "#e34948"  # slot 8: two-way capas


def load(rel):
    xs, ys = [], []
    with open(os.path.join(ROOT, rel), newline="") as f:
        for row in csv.DictReader(f):
            xs.append(float(row["drift_m"]) * 1e3)
            ys.append(-float(row["base_shear_MN"]) * 1e3)
    return xs, ys


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8.0)


def loop_panel(ax, g0, series, color, title, stats):
    style_axis(ax)
    ax.axhline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.axvline(0.0, color=BASELINE, linewidth=0.8, zorder=1)
    ax.plot(g0[0], g0[1], color=MUTED, linewidth=0.9, alpha=0.9, zorder=2)
    ax.plot(series[0], series[1], color=color, linewidth=1.5, zorder=3)
    ax.set_title(title, color=INK, fontsize=10.5, pad=30)
    ax.text(0.5, 1.015, stats, transform=ax.transAxes, ha="center",
            va="bottom", color=INK2, fontsize=7.4)


g0 = load(os.path.join("g0_baseline", "cyc", "lmnewton_hysteresis.csv"))

# ── Figura 1: los cuatro principios ─────────────────────────────────────────
defl = load(os.path.join("p1_deflation_final", "cyc",
                         "lmnewton_hysteresis.csv"))
ls = load(os.path.join("p1_linesearch", "cyc", "lmnewton_hysteresis.csv"))
twb = load(os.path.join("fe2_col", "two_way_boundary",
                        "fe2_column_hysteresis.csv"))
ca = load(os.path.join("p3_ca", "full", "lmnewton_hysteresis.csv"))

fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.2), dpi=200,
                         sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

loop_panel(axes[0], g0, defl, C_DEFL,
           "1 · EVITACIÓN — deflación",
           "resultado negativo: cuencas vecinas peores (hasta 14.7 MN);\n"
           "el fallback reproduce el base — evitar no es seleccionar")
loop_panel(axes[1], g0, ls, C_LS,
           "2 · DESCENSO — line search de energía",
           "espiga dominante 48 → 11 kN a costo ~cero;\n"
           "único remedio robusto al cambio de física")
loop_panel(axes[2], g0, twb, C_FACES,
           "3 · CONTROL — FE² two-way",
           "0 pasos no convergidos en 196; el lazo sigue al\n"
           "monolítico con pinching — el macro selecciona la rama")
loop_panel(axes[3], g0, ca, C_CA,
           "4 · AJUSTE — CA-tuner",
           "noconv 80 → 66 sin costo por paso; descubre\n"
           "subdivisión + μ lento (acoplamiento no obvio)")

for ax in axes:
    ax.set_xlabel("deriva  [mm]", color=INK2, fontsize=9)
axes[0].set_ylabel("V = −reacción  [kN]", color=INK2, fontsize=9)
axes[0].set_ylim(-62, 70)

fig.legend(handles=[
    plt.Line2D([], [], color=MUTED, linewidth=1.1,
               label="baseline monolítico g0 (referencia)")],
    loc="lower center", ncol=1, frameon=False, fontsize=8.5,
    labelcolor=INK2, bbox_to_anchor=(0.5, -0.006))

fig.suptitle(
    "Cuatro principios de regularización de las reversas Ko–Bathe — "
    "veredicto empírico (física de g0)",
    color=INK, fontsize=12.5, y=1.0)

fig.tight_layout(rect=(0, 0.05, 1, 0.90))
out1 = os.path.join(ROOT, "regularization_principles.png")
fig.savefig(out1, facecolor=SURFACE, bbox_inches="tight")
print("escrita:", out1)

# ── Figura 2: el dial de control cinemático del FE² ─────────────────────────
macro = load(os.path.join("fe2_col", "macro_only",
                          "fe2_column_hysteresis.csv"))
twl = load(os.path.join("fe2_col", "two_way_layers",
                        "fe2_column_hysteresis.csv"))

fig2 = plt.figure(figsize=(15.6, 4.6), dpi=200)
fig2.patch.set_facecolor(SURFACE)
gs = fig2.add_gridspec(2, 4, wspace=0.32, hspace=0.55,
                       height_ratios=[1.0, 1.0])
ax_a = fig2.add_subplot(gs[:, 0])
ax_b = fig2.add_subplot(gs[:, 1], sharey=ax_a)
ax_c = fig2.add_subplot(gs[:, 2], sharey=ax_a)
ax_d = fig2.add_subplot(gs[0, 3])
ax_e = fig2.add_subplot(gs[1, 3])

loop_panel(ax_a, g0, macro, C_MACRO,
           "sin acople (fibra)",
           "suave trivial: el macro nunca ve\nla patología · mitad de resistencia")
loop_panel(ax_b, g0, twb, C_FACES,
           "control débil: 2 caras",
           "20 estrictos · transitorio inicial ·\nlazo global más liso")
loop_panel(ax_c, g0, twl, C_LAYERS,
           "control fuerte: 5 capas",
           "49 estrictos · local sin ambigüedad\n(energy-lm idéntico) · global rugoso")
for ax in (ax_a, ax_b, ax_c):
    ax.set_xlabel("deriva  [mm]", color=INK2, fontsize=9)
ax_a.set_ylabel("V = −reacción  [kN]", color=INK2, fontsize=9)
ax_a.set_ylim(-108, 108)

# El trade-off local/global en dos mini-paneles de EJE ÚNICO.
modes = ["fibra", "caras", "capas"]
colors = [C_MACRO, C_FACES, C_LAYERS]
strict = [0, 20, 49]            # pasos estrictos (fibra: sin acople)
rough = [0.62, 1.15, 1.39]      # variación total de V [MN]
x = np.arange(3)

style_axis(ax_d)
ax_d.bar(x, strict, 0.55, color=colors, zorder=3)
for xi, v in zip(x, strict):
    ax_d.text(xi, v + 1.5, str(v), ha="center", va="bottom",
              color=INK2, fontsize=8)
ax_d.set_xticks(x, modes, color=INK2, fontsize=8.5)
ax_d.set_ylim(0, 62)
ax_d.set_title("acople estricto  (↑ bien-puesto)", color=INK, fontsize=9.5)
ax_d.set_ylabel("pasos  [–]", color=INK2, fontsize=8.5)

style_axis(ax_e)
ax_e.bar(x, rough, 0.55, color=colors, zorder=3)
for xi, v in zip(x, rough):
    ax_e.text(xi, v + 0.04, f"{v:.2f}", ha="center", va="bottom",
              color=INK2, fontsize=8)
ax_e.set_xticks(x, modes, color=INK2, fontsize=8.5)
ax_e.set_ylim(0, 1.75)
ax_e.set_title("rugosidad global ΣΔV  (↓ fidelidad)", color=INK,
               fontsize=9.5)
ax_e.set_ylabel("[MN]", color=INK2, fontsize=8.5)

fig2.suptitle(
    "El control cinemático como dial de regularización — FE² de columna, "
    "de la libertad total al plano por capa",
    color=INK, fontsize=12.5, y=1.02)

fig2.tight_layout(rect=(0, 0.02, 1, 0.88))
out2 = os.path.join(ROOT, "fe2_control_dial.png")
fig2.savefig(out2, facecolor=SURFACE, bbox_inches="tight")
print("escrita:", out2)
