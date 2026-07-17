# GRAN CHART: la curva histeretica de CADA corrida (evaluacion) de las 3
# campanas CA-FE2 mejoradas. Un panel por eval, ordenados por aptitud
# (mejor arriba-izquierda), cada uno con la referencia Timoshenko por
# detras y etiquetado con su configuracion (los 4 rasgos) y sus valores CA
# (fit, env, pico, zigzag). Re-ejecutable: lee lo que exista.
# Salida: data/output/ca_fe2_all_runs.png
import csv
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

ROOT = r"c:\MyLibs\fall_n\data\output"
CAMPAIGNS = {
    "s16": "ca_fe2_env_s20260716",
    "s17": "ca_fe2_env_s20260717",
    "s18": "ca_fe2_env_s20260718",
}
REF = r"fe2_col\macro_only\fe2_column_hysteresis.csv"      # Timoshenko
OLD = r"ca_fe2_best_full\fe2_column_hysteresis.csv"        # optimo viejo (defecto)


def load_xy(path):
    xs, ys = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                xs.append(float(row["drift_m"]) * 1e3)
                ys.append(-float(row["base_shear_MN"]) * 1e3)
            except (ValueError, KeyError):
                break
    return xs, ys


def load_manifest(cdir):
    p = os.path.join(ROOT, cdir, "runs", "manifest.csv")
    rows = {}
    if not os.path.isfile(p):
        return rows
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows[int(r["eval"])] = {k: float(r[k]) for k in
                                        ("gap", "relax", "fit", "env", "peak", "zz")}
                rows[int(r["eval"])]["stag"] = int(float(r["stag"]))
                rows[int(r["eval"])]["start"] = int(float(r["start"]))
            except (ValueError, KeyError):
                pass
    return rows


# reunir todas las corridas de las 3 campanas
runs = []
for tag, cdir in CAMPAIGNS.items():
    man = load_manifest(cdir)
    for csvp in sorted(glob.glob(os.path.join(ROOT, cdir, "runs", "run_*.csv"))):
        ev = int(os.path.basename(csvp)[4:7])
        meta = man.get(ev, {})
        xs, ys = load_xy(csvp)
        if len(xs) < 10:
            continue
        runs.append({"tag": tag, "eval": ev, "xs": xs, "ys": ys, **meta})

refx, refy = load_xy(os.path.join(ROOT, REF))

if not runs:
    print("aun no hay corridas persistidas (runs/run_*.csv); reintenta luego.")
    raise SystemExit(0)

# ordenar por aptitud descendente (mejores primero)
runs.sort(key=lambda r: r.get("fit", -99), reverse=True)

ncol = 6
nrow = (len(runs) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.3 * nrow),
                         dpi=150, squeeze=False)

fits = [r.get("fit", 0) for r in runs]
norm = colors.Normalize(vmin=min(fits), vmax=max(fits))
cmap = cm.get_cmap("viridis")
TAGCOL = {"s16": "#4a3aa7", "s17": "#e34948", "s18": "#1f7a5a"}

for idx in range(nrow * ncol):
    ax = axes[idx // ncol][idx % ncol]
    if idx >= len(runs):
        ax.axis("off")
        continue
    r = runs[idx]
    ax.plot(refx, refy, color="#c9c8c0", linewidth=1.0, zorder=1)
    ax.plot(r["xs"], r["ys"], color=cmap(norm(r.get("fit", 0))),
            linewidth=0.9, zorder=2)
    ax.axhline(0, color="#e1e0d9", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#e1e0d9", linewidth=0.4, zorder=0)
    ax.set_xlim(-215, 215)
    ax.set_ylim(-75, 75)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("bottom", "left", "top", "right"):
        ax.spines[sp].set_edgecolor(TAGCOL[r["tag"]])
        ax.spines[sp].set_linewidth(1.3)
    title = (f"{r['tag']}·e{r['eval']:02d}   fit {r.get('fit',0):.3f}\n"
             f"gap {r.get('gap',0):.2f}  relax {r.get('relax',0):.2f}  "
             f"stag {r.get('stag',0)}  start {r.get('start',0)}")
    sub = (f"env {r.get('env',0)*1e3:.1f}kN  "
           f"|V|max {r.get('peak',0)*1e3:.0f}kN  zz {r.get('zz',0)*100:.0f}%")
    ax.set_title(title, fontsize=6.6, color="#0b0b0b", pad=12)
    ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6.0, color="#52514e")

fig.suptitle(
    "Programa CA-FE² mejorado (fitness con seguimiento de envolvente) — "
    "lazo histéretico de cada evaluación, ordenadas por aptitud\n"
    "referencia Timoshenko en gris; borde por semilla "
    "(s16 violeta / s17 rojo / s18 verde); color de traza por aptitud",
    fontsize=11, y=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.985))
out = os.path.join(ROOT, "ca_fe2_all_runs.png")
fig.savefig(out, bbox_inches="tight")
print(f"figura: {out}  ({len(runs)} corridas)")
