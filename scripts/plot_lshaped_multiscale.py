#!/usr/bin/env python3
"""
Plot L-shaped Multiscale Seismic Analysis Results
==================================================
Reads CSV files produced by ``fall_n_lshaped_multiscale`` and generates
publication-quality figures (PDF + PNG) suitable for a doctoral thesis.

Expected input files (under DATA_DIR):
    recorders/roof_displacement.csv
    recorders/fiber_hysteresis_concrete.csv
    recorders/fiber_hysteresis_steel.csv

Usage:
    python scripts/plot_lshaped_multiscale.py

Output directory:
    doc/figures/lshaped_multiscale/
"""

import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("matplotlib not found. Install with: pip install matplotlib",
          file=sys.stderr)
    sys.exit(1)


# ── Paths ─────────────────────────────────────────────────────────────

REPO  = Path(__file__).resolve().parent.parent
DATA  = REPO / "data" / "output" / "lshaped_multiscale" / "recorders"
FIGS  = REPO / "doc" / "figures" / "lshaped_multiscale"
FIGS.mkdir(parents=True, exist_ok=True)


# ── Style ─────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   10,
    "legend.fontsize":  8,
    "lines.linewidth":  0.7,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "figure.dpi":       150,
    "savefig.dpi":      300,
})

BLUE    = "#1f77b4"
RED     = "#d62728"
GREEN   = "#2ca02c"
PURPLE  = "#9467bd"
ORANGE  = "#ff7f0e"
PALETTE = [BLUE, RED, GREEN, PURPLE, ORANGE, "#8c564b", "#e377c2", "#7f7f7f"]


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")
    print(f"  Saved: {FIGS / name}.pdf/png")


# =====================================================================
#  1. Roof displacement time histories
# =====================================================================

def plot_roof_displacement():
    csv_path = DATA / "roof_displacement.csv"
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found"); return

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    t = data[:, 0]

    # Read header to get column names
    with open(csv_path) as f:
        header = f.readline().strip().split(",")

    # Group columns by node: each node has dof0 (X) and dof1 (Y)
    nodes = {}
    for i, col in enumerate(header[1:], start=1):
        # col format: node<id>_dof<d>
        parts = col.split("_")
        nid = parts[0].replace("node", "")
        dof = int(parts[1].replace("dof", ""))
        nodes.setdefault(nid, {})[dof] = i

    # ── Figure: X and Y displacement histories ────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)

    for idx, (nid, dofs) in enumerate(sorted(nodes.items())):
        c = PALETTE[idx % len(PALETTE)]
        lbl = f"Node {nid}"
        if 0 in dofs:
            ax1.plot(t, data[:, dofs[0]] * 1e3, color=c, label=lbl)
        if 1 in dofs:
            ax2.plot(t, data[:, dofs[1]] * 1e3, color=c, label=lbl)

    ax1.set_ylabel(r"$u_x$ [mm]")
    ax2.set_ylabel(r"$u_y$ [mm]")
    ax2.set_xlabel("Time [s]")
    ax1.set_title("Roof displacement — L-shaped RC building (multiscale)")
    ax1.legend(loc="upper right", ncol=2, framealpha=0.8)
    ax2.legend(loc="upper right", ncol=2, framealpha=0.8)
    fig.tight_layout()
    save(fig, "roof_displacement")

    # ── Figure: X-Y orbit at first node ───────────────────────────────
    first_nid = sorted(nodes.keys())[0]
    dofs = nodes[first_nid]
    if 0 in dofs and 1 in dofs:
        ux = data[:, dofs[0]] * 1e3
        uy = data[:, dofs[1]] * 1e3

        fig2, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot(ux, uy, color=BLUE, lw=0.5, alpha=0.8)
        ax.plot(ux[0], uy[0], "o", color=GREEN, ms=6, zorder=5, label="Start")
        ax.plot(ux[-1], uy[-1], "s", color=RED, ms=6, zorder=5, label="End")
        ax.set_xlabel(r"$u_x$ [mm]")
        ax.set_ylabel(r"$u_y$ [mm]")
        ax.set_title(f"Roof orbit — Node {first_nid}")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper left")
        fig2.tight_layout()
        save(fig2, "roof_orbit")

    plt.close("all")


# =====================================================================
#  2. Fiber hysteresis loops (concrete + steel)
# =====================================================================

def plot_hysteresis(mat_name, csv_path, stress_unit="MPa"):
    if not csv_path.exists():
        print(f"  [skip] {csv_path} not found"); return

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    with open(csv_path) as f:
        header = f.readline().strip().split(",")

    # Pair strain/stress columns
    fibers = []
    i = 1
    while i + 1 < len(header):
        strain_col = header[i]
        stress_col = header[i + 1]
        label = strain_col.rsplit("_strain", 1)[0]
        fibers.append((label, i, i + 1))
        i += 2

    n_fibers = len(fibers)
    if n_fibers == 0:
        return

    ncols = min(n_fibers, 3)
    nrows = (n_fibers + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.8 * ncols, 3.2 * nrows),
                             squeeze=False)

    for idx, (label, si, sti) in enumerate(fibers):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        strain = data[:, si]
        stress = data[:, sti]
        ax.plot(strain, stress, color=PALETTE[idx % len(PALETTE)], lw=0.5)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(rf"$\sigma$ [{stress_unit}]")
        ax.set_title(label.replace("_", " "), fontsize=8)

    # Hide unused subplots
    for idx in range(n_fibers, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Fiber hysteresis — {mat_name}", fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, f"hysteresis_{mat_name.lower()}")
    plt.close(fig)


# =====================================================================
#  3. Inter-story drift envelopes (derived from roof displacement)
# =====================================================================

def plot_drift_envelope():
    csv_path = DATA / "roof_displacement.csv"
    if not csv_path.exists():
        return

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    with open(csv_path) as f:
        header = f.readline().strip().split(",")

    # Extract max absolute X displacement per node → approximate drift
    max_ux = {}
    for i, col in enumerate(header[1:], start=1):
        parts = col.split("_")
        nid = parts[0].replace("node", "")
        dof = int(parts[1].replace("dof", ""))
        if dof == 0:
            max_ux[nid] = np.max(np.abs(data[:, i])) * 1e3  # mm

    if len(max_ux) < 2:
        return

    nids = sorted(max_ux.keys())
    vals = [max_ux[n] for n in nids]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.barh(range(len(nids)), vals, color=BLUE, alpha=0.7)
    ax.set_yticks(range(len(nids)))
    ax.set_yticklabels([f"Node {n}" for n in nids])
    ax.set_xlabel("Peak $|u_x|$ [mm]")
    ax.set_title("Peak lateral displacement envelope")
    fig.tight_layout()
    save(fig, "peak_displacement_envelope")
    plt.close(fig)


# =====================================================================
#  Main
# =====================================================================

def main():
    print(f"Data directory: {DATA}")
    print(f"Output directory: {FIGS}\n")

    if not DATA.exists():
        print(f"ERROR: Data directory not found: {DATA}")
        print("Run the simulation first: build/fall_n_lshaped_multiscale.exe")
        sys.exit(1)

    print("[1/4] Roof displacement time histories...")
    plot_roof_displacement()

    print("[2/4] Concrete fiber hysteresis...")
    plot_hysteresis("Concrete", DATA / "fiber_hysteresis_concrete.csv")

    print("[3/4] Steel fiber hysteresis...")
    plot_hysteresis("Steel", DATA / "fiber_hysteresis_steel.csv")

    print("[4/4] Peak displacement envelope...")
    plot_drift_envelope()

    print("\nDone.")


if __name__ == "__main__":
    main()
