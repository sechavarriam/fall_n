#!/usr/bin/env python3
# =============================================================================
#  plot_table_multiscale.py — Postprocessing for the 4-legged table example
# =============================================================================
#
#  Generates:
#    1. Roof displacement time histories (X, Y, Z) + X-Y orbit
#    2. Fiber hysteresis (concrete + steel) at global scale
#    3. Crack evolution at sub-model scale
#    4. Rebar strain histories at sub-model scale
#    5. Global history (displacement norm + damage index)
#
#  Usage:
#    python plot_table_multiscale.py --data <recorder_dir> --figures <fig_dir>
#
# =============================================================================

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN  = "#2ca02c"
RED    = "#d62728"
PURPLE = "#9467bd"
COLORS = [BLUE, ORANGE, GREEN, RED, PURPLE, "#8c564b", "#e377c2", "#7f7f7f"]


def load_csv(path: Path) -> np.ndarray:
    """Load CSV with header, return structured array."""
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return np.genfromtxt(path, delimiter=",", names=True,
                         dtype=None, encoding="utf-8")


def save(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")


# =============================================================================
#  1. Roof displacement
# =============================================================================
def plot_roof_displacement(data: Path, figs: Path) -> None:
    csv = data / "roof_displacement.csv"
    if not csv.exists():
        print("  [skip] roof_displacement.csv not found")
        return

    d = load_csv(csv)
    cols = list(d.dtype.names)
    t = d[cols[0]]

    # Extract displacement columns per node
    disp_cols = [c for c in cols[1:] if c.lower().startswith("n")]
    x_cols = [c for c in disp_cols if c.endswith("_0") or c.endswith("_dof0")]
    y_cols = [c for c in disp_cols if c.endswith("_1") or c.endswith("_dof1")]
    z_cols = [c for c in disp_cols if c.endswith("_2") or c.endswith("_dof2")]

    if not x_cols:
        x_cols = disp_cols[0::3]
        y_cols = disp_cols[1::3]
        z_cols = disp_cols[2::3]

    # Time histories: X, Y, Z
    fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for i, c in enumerate(x_cols):
        lbl = c.split("_")[0]
        axes[0].plot(t, d[c] * 1e3, lw=0.7, color=COLORS[i % len(COLORS)],
                     label=lbl)
    axes[0].set_ylabel("$u_x$ [mm]")
    axes[0].legend(fontsize=7, ncol=4)

    for i, c in enumerate(y_cols):
        axes[1].plot(t, d[c] * 1e3, lw=0.7, color=COLORS[i % len(COLORS)])
    axes[1].set_ylabel("$u_y$ [mm]")

    for i, c in enumerate(z_cols):
        axes[2].plot(t, d[c] * 1e3, lw=0.7, color=COLORS[i % len(COLORS)])
    axes[2].set_ylabel("$u_z$ [mm]")
    axes[2].set_xlabel("Time [s]")

    fig.suptitle("Roof corner displacements — 4-legged table", fontsize=10)
    fig.tight_layout()
    save(fig, figs, "roof_displacement")
    plt.close(fig)

    # X-Y orbit (first corner)
    if x_cols and y_cols:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(d[x_cols[0]] * 1e3, d[y_cols[0]] * 1e3, lw=0.4, color=BLUE)
        ax.set_xlabel("$u_x$ [mm]")
        ax.set_ylabel("$u_y$ [mm]")
        ax.set_title("Roof orbit (corner node)")
        ax.set_aspect("equal")
        fig.tight_layout()
        save(fig, figs, "roof_orbit")
        plt.close(fig)


# =============================================================================
#  2. Fiber hysteresis (global scale)
# =============================================================================
def plot_fiber_hysteresis(data: Path, figs: Path, tag: str) -> None:
    csv = data / f"fiber_hysteresis_{tag.lower()}.csv"
    if not csv.exists():
        print(f"  [skip] fiber_hysteresis_{tag.lower()}.csv not found")
        return

    d = load_csv(csv)
    cols = list(d.dtype.names)

    strain_cols = [c for c in cols if "strain" in c.lower()]
    stress_cols = [c for c in cols if "stress" in c.lower()]
    n = min(len(strain_cols), len(stress_cols), 4)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3), squeeze=False)
    for i in range(n):
        ax = axes[0, i]
        ax.plot(d[strain_cols[i]], d[stress_cols[i]], lw=0.4,
                color=COLORS[i % len(COLORS)])
        ax.set_xlabel("$\\varepsilon$")
        if i == 0:
            ax.set_ylabel("$\\sigma$ [MPa]")
        ax.set_title(f"Fiber {i+1}", fontsize=8)
    fig.suptitle(f"{tag} fiber hysteresis (global scale)", fontsize=10)
    fig.tight_layout()
    save(fig, figs, f"hysteresis_{tag.lower()}")
    plt.close(fig)


# =============================================================================
#  3. Crack evolution (sub-model scale)
# =============================================================================
def plot_crack_evolution(data: Path, figs: Path) -> None:
    csv = data / "crack_evolution.csv"
    if not csv.exists():
        print("  [skip] crack_evolution.csv not found")
        return

    d = load_csv(csv)
    cols = list(d.dtype.names)
    t = d[cols[0]]

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)

    if "total_cracked_gps" in cols:
        axes[0, 0].plot(t, d["total_cracked_gps"], lw=0.8, color=BLUE)
    axes[0, 0].set_ylabel("Cracked GPs")
    axes[0, 0].set_title("Cracked Gauss points", fontsize=8)

    if "total_cracks" in cols:
        axes[0, 1].plot(t, d["total_cracks"], lw=0.8, color=ORANGE)
    axes[0, 1].set_ylabel("Total cracks")
    axes[0, 1].set_title("Accumulated cracks", fontsize=8)

    if "max_damage" in cols:
        axes[1, 0].plot(t, d["max_damage"], lw=0.8, color=RED)
    axes[1, 0].set_ylabel("Max damage")
    axes[1, 0].set_xlabel("Time [s]")
    axes[1, 0].set_title("Peak damage index", fontsize=8)

    if "max_opening" in cols:
        axes[1, 1].plot(t, d["max_opening"] * 1e3, lw=0.8, color=GREEN)
    axes[1, 1].set_ylabel("Max opening [mm]")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_title("Max crack opening", fontsize=8)

    fig.suptitle("Crack evolution (sub-model scale)", fontsize=10)
    fig.tight_layout()
    save(fig, figs, "crack_evolution")
    plt.close(fig)

    # Per-sub-model comparison
    sub_cols = [c for c in cols if c.startswith("sub") and "cracks" in c]
    if len(sub_cols) >= 2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for i, c in enumerate(sub_cols):
            ax.plot(t, d[c], lw=0.7, color=COLORS[i % len(COLORS)],
                    label=c.replace("_", " "))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Cracks")
        ax.set_title("Per-sub-model crack count")
        ax.legend(fontsize=7)
        fig.tight_layout()
        save(fig, figs, "crack_submodel_comparison")
        plt.close(fig)


# =============================================================================
#  4. Rebar strain histories (sub-model scale)
# =============================================================================
def plot_rebar_strains(data: Path, figs: Path) -> None:
    csv = data / "rebar_strains.csv"
    if not csv.exists():
        print("  [skip] rebar_strains.csv not found")
        return

    d = load_csv(csv)
    cols = list(d.dtype.names)
    t = d[cols[0]]
    bar_cols = [c for c in cols[1:] if "bar" in c]

    if not bar_cols:
        return

    # Group by sub-model
    subs = {}
    for c in bar_cols:
        sub_id = c.split("_bar")[0]
        subs.setdefault(sub_id, []).append(c)

    n_subs = len(subs)
    fig, axes = plt.subplots(1, n_subs, figsize=(5 * n_subs, 3.5),
                             squeeze=False, sharey=True)

    for j, (sub_id, bcols) in enumerate(subs.items()):
        ax = axes[0, j]
        for i, c in enumerate(bcols):
            ax.plot(t, d[c] * 1e3, lw=0.5, color=COLORS[i % len(COLORS)],
                    label=f"bar {i}")
        ax.set_xlabel("Time [s]")
        if j == 0:
            ax.set_ylabel("Axial strain [‰]")
        ax.set_title(sub_id.replace("_", " "), fontsize=8)
        ax.legend(fontsize=6, ncol=2)

    fig.suptitle("Embedded rebar strains (sub-model scale)", fontsize=10)
    fig.tight_layout()
    save(fig, figs, "rebar_strains")
    plt.close(fig)


# =============================================================================
#  5. Global history (displacement + damage)
# =============================================================================
def plot_global_history(data: Path, figs: Path) -> None:
    csv = data / "global_history.csv"
    if not csv.exists():
        print("  [skip] global_history.csv not found")
        return

    d = load_csv(csv)
    cols = list(d.dtype.names)
    t = d["time"]
    u = d["u_inf"]
    dmg = d["peak_damage"]

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)

    axes[0].plot(t, u * 1e3, lw=0.7, color=BLUE)
    axes[0].set_ylabel("$||u||_\\infty$ [mm]")
    axes[0].set_title("Peak displacement", fontsize=8)

    axes[1].plot(t, dmg, lw=0.7, color=RED)
    axes[1].axhline(1.0, ls="--", lw=0.5, color="gray", label="yield")
    axes[1].set_ylabel("Damage index")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_title("Peak damage index", fontsize=8)
    axes[1].legend(fontsize=7)

    fig.suptitle("Global response history — 4-legged table", fontsize=10)
    fig.tight_layout()
    save(fig, figs, "global_history")
    plt.close(fig)


# =============================================================================
#  Entry point
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="4-legged table multiscale postprocessing")
    parser.add_argument("--data", required=True, type=Path,
                        help="Path to recorders/ directory")
    parser.add_argument("--figures", required=True, type=Path,
                        help="Output directory for figures")
    args = parser.parse_args()

    data: Path = args.data
    figs: Path = args.figures

    if not data.exists():
        print(f"ERROR: data directory not found: {data}", file=sys.stderr)
        sys.exit(1)

    figs.mkdir(parents=True, exist_ok=True)
    print(f"Data    : {data}")
    print(f"Figures : {figs}\n")

    print("[1/6] Roof displacement...")
    plot_roof_displacement(data, figs)

    print("[2/6] Concrete fiber hysteresis...")
    plot_fiber_hysteresis(data, figs, "Concrete")

    print("[3/6] Steel fiber hysteresis...")
    plot_fiber_hysteresis(data, figs, "Steel")

    print("[4/6] Crack evolution...")
    plot_crack_evolution(data, figs)

    print("[5/6] Rebar strains...")
    plot_rebar_strains(data, figs)

    print("[6/6] Global history...")
    plot_global_history(data, figs)

    print("\nDone — all figures saved.")


if __name__ == "__main__":
    main()
