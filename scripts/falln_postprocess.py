#!/usr/bin/env python3
# =============================================================================
#  falln_postprocess.py — Reusable postprocessing library for fall_n outputs
# =============================================================================
#
#  Usage:
#    python falln_postprocess.py --data <recorder_dir> --figures <output_dir>
#
#  Can also be imported as a module:
#    from falln_postprocess import load_csv, plot_roof_displacement, ...
#
# =============================================================================

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
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
COLORS = [BLUE, ORANGE, GREEN, RED, "#9467bd", "#8c564b", "#e377c2"]


# =============================================================================
#  Utilities
# =============================================================================
def load_csv(path: str | Path, *, skip_comments: bool = True) -> np.ndarray:
    """Load a CSV file with optional comment-line skipping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    lines = path.read_text().splitlines()
    header_idx = 0
    for i, ln in enumerate(lines):
        if not ln.strip().startswith("#"):
            header_idx = i
            break
    return np.genfromtxt(path, delimiter=",", skip_header=header_idx,
                         names=True, dtype=None, encoding="utf-8")


def _save(fig: plt.Figure, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{name}.{ext}")


# =============================================================================
#  Roof displacement
# =============================================================================
def plot_roof_displacement(data_dir: Path, fig_dir: Path,
                           csv_name: str = "roof_displacement.csv") -> None:
    csv = data_dir / csv_name
    d = load_csv(csv)

    cols = list(d.dtype.names)
    time_col = cols[0]
    t = d[time_col]

    disp_cols = [c for c in cols[1:] if c.lower().startswith("n")]
    x_cols = [c for c in disp_cols if c.endswith("_0") or c.endswith("_dof0")]
    y_cols = [c for c in disp_cols if c.endswith("_1") or c.endswith("_dof1")]

    if not x_cols:
        x_cols = disp_cols[0::3]
        y_cols = disp_cols[1::3]

    # ── Time histories ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)
    for i, c in enumerate(x_cols):
        axes[0].plot(t, d[c] * 1e3, lw=0.6, color=COLORS[i % len(COLORS)],
                     label=c.split("_")[0])
    axes[0].set_ylabel("$u_x$ [mm]")
    axes[0].legend(fontsize=7, ncol=3)
    for i, c in enumerate(y_cols):
        axes[1].plot(t, d[c] * 1e3, lw=0.6, color=COLORS[i % len(COLORS)])
    axes[1].set_ylabel("$u_y$ [mm]")
    axes[1].set_xlabel("Time [s]")
    fig.suptitle("Roof displacement", fontsize=10)
    fig.tight_layout()
    _save(fig, fig_dir, "roof_displacement")
    plt.close(fig)

    # ── X-Y orbit ───────────────────────────────────────────────────────
    if x_cols and y_cols:
        fig, ax = plt.subplots(figsize=(4. , 4.))
        ax.plot(d[x_cols[0]] * 1e3, d[y_cols[0]] * 1e3, lw=0.4, color=BLUE)
        ax.set_xlabel("$u_x$ [mm]")
        ax.set_ylabel("$u_y$ [mm]")
        ax.set_title("Roof orbit")
        ax.set_aspect("equal")
        fig.tight_layout()
        _save(fig, fig_dir, "roof_orbit")
        plt.close(fig)


# =============================================================================
#  Fiber hysteresis
# =============================================================================
def plot_hysteresis(data_dir: Path, fig_dir: Path,
                    material_tag: str = "Concrete",
                    csv_name: str | None = None) -> None:
    if csv_name is None:
        csv_name = f"fiber_hysteresis_{material_tag.lower()}.csv"
    csv = data_dir / csv_name
    if not csv.exists():
        return

    d = load_csv(csv)
    cols = list(d.dtype.names)

    strain_cols = [c for c in cols if "strain" in c.lower()]
    stress_cols = [c for c in cols if "stress" in c.lower()]
    n = min(len(strain_cols), len(stress_cols), 5)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    for i in range(n):
        ax = axes[0, i]
        ax.plot(d[strain_cols[i]], d[stress_cols[i]], lw=0.4,
                color=COLORS[i % len(COLORS)])
        ax.set_xlabel("$\\varepsilon$")
        if i == 0:
            ax.set_ylabel("$\\sigma$ [MPa]")
        ax.set_title(f"Fiber {i+1}", fontsize=8)
    fig.suptitle(f"{material_tag} fiber hysteresis", fontsize=10)
    fig.tight_layout()
    _save(fig, fig_dir, f"hysteresis_{material_tag.lower()}")
    plt.close(fig)


# =============================================================================
#  Crack evolution
# =============================================================================
def plot_crack_evolution(data_dir: Path, fig_dir: Path,
                         csv_name: str = "crack_evolution.csv") -> None:
    csv = data_dir / csv_name
    if not csv.exists():
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

    fig.suptitle("Crack evolution", fontsize=10)
    fig.tight_layout()
    _save(fig, fig_dir, "crack_evolution")
    plt.close(fig)

    # ── Per-sub-model comparison ────────────────────────────────────────
    sub_cracks_cols = [c for c in cols if c.startswith("sub") and "cracks" in c]
    if len(sub_cracks_cols) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, c in enumerate(sub_cracks_cols):
        ax.plot(t, d[c], lw=0.7, color=COLORS[i % len(COLORS)],
                label=c.replace("_", " "))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Cracks")
    ax.set_title("Per-sub-model crack count")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, fig_dir, "crack_submodel_comparison")
    plt.close(fig)


# =============================================================================
#  Interstory drift envelope (from roof displacement CSV)
# =============================================================================
def plot_interstory_drift(data_dir: Path, fig_dir: Path,
                          story_height: float = 3.2,
                          csv_name: str = "roof_displacement.csv") -> None:
    csv = data_dir / csv_name
    if not csv.exists():
        return

    d = load_csv(csv)
    cols = list(d.dtype.names)
    x_cols = [c for c in cols[1:] if c.endswith("_0") or c.endswith("_dof0")]
    if not x_cols:
        x_cols = [c for c in cols[1:]][0::3]

    # Peak |u_x| per node
    peaks = [np.max(np.abs(d[c])) for c in x_cols]
    nids = [c.split("_")[0] for c in x_cols]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.barh(range(len(nids)), [p * 1e3 for p in peaks],
            color=BLUE, alpha=0.7)
    ax.set_yticks(range(len(nids)))
    ax.set_yticklabels([f"Node {n}" for n in nids])
    ax.set_xlabel("Peak $|u_x|$ [mm]")
    ax.set_title("Peak lateral displacement envelope")
    fig.tight_layout()
    _save(fig, fig_dir, "peak_displacement_envelope")
    plt.close(fig)


# =============================================================================
#  Entry point (CLI)
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="fall_n postprocessing — generate plots from CSV output")
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

    print("[1/5] Roof displacement...")
    plot_roof_displacement(data, figs)

    print("[2/5] Concrete fiber hysteresis...")
    plot_hysteresis(data, figs, "Concrete")

    print("[3/5] Steel fiber hysteresis...")
    plot_hysteresis(data, figs, "Steel")

    print("[4/5] Crack evolution...")
    plot_crack_evolution(data, figs)

    print("[5/5] Peak displacement envelope...")
    plot_interstory_drift(data, figs)

    print("\nDone — all figures saved.")


if __name__ == "__main__":
    main()
