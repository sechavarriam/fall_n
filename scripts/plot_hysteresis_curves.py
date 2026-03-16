#!/usr/bin/env python3
"""
Phase 4 — Hysteresis Curve Plotter
===================================
Reads CSV files produced by test_phase4_hysteretic_cycles and generates
stress-strain hysteresis plots using matplotlib.

Usage:
    python3 scripts/plot_hysteresis_curves.py

Output:
    data/output/hysteretic_cycles/hysteresis_curves.pdf
    data/output/hysteretic_cycles/hysteresis_curves.png
"""

import csv
import os
import sys
from pathlib import Path

# ── Locate data directory ─────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR  = REPO_ROOT / "data" / "output" / "hysteretic_cycles"

CURVES = {
    "steel_menegotto_pinto_cyclic":        "Menegotto-Pinto Steel — Increasing Amplitude",
    "steel_menegotto_pinto_3cycles":       "Menegotto-Pinto Steel — 3 Constant Cycles",
    "concrete_kent_park_cyclic":           "Kent-Park Concrete — Unconfined Cyclic",
    "concrete_kent_park_confined_cyclic":  "Kent-Park Concrete — Confined Cyclic",
    "j2_uniaxial_cyclic":                 "J₂ Plasticity — Symmetric Cyclic",
}


def read_csv(filepath: Path):
    """Read a 2-column (strain, stress) CSV and return lists."""
    strain, stress = [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strain.append(float(row["strain"]))
            stress.append(float(row["stress"]))
    return strain, stress


def read_combined_csv(filepath: Path):
    """Read the combined steel+concrete CSV."""
    strain, steel_stress, concrete_stress = [], [], []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strain.append(float(row["strain"]))
            steel_stress.append(float(row["steel_stress"]))
            concrete_stress.append(float(row["concrete_stress"]))
    return strain, steel_stress, concrete_stress


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib",
              file=sys.stderr)
        sys.exit(1)

    if not DATA_DIR.is_dir():
        print(f"Data directory not found: {DATA_DIR}", file=sys.stderr)
        print("Run the test first: ./build/fall_n_phase4_hysteretic_test", file=sys.stderr)
        sys.exit(1)

    # ── Collect available curves ──────────────────────────────────────────

    available = {}
    for key, title in CURVES.items():
        csv_path = DATA_DIR / f"{key}.csv"
        if csv_path.exists():
            available[key] = (title, csv_path)

    combined_path = DATA_DIR / "combined_steel_concrete_cyclic.csv"
    has_combined = combined_path.exists()

    n_panels = len(available) + (1 if has_combined else 0)
    if n_panels == 0:
        print("No CSV files found in", DATA_DIR, file=sys.stderr)
        sys.exit(1)

    # ── Layout ────────────────────────────────────────────────────────────

    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols

    fig = plt.figure(figsize=(7.5 * ncols, 5.5 * nrows))
    fig.suptitle("Phase 4 — Cyclic Constitutive Model Verification",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.30)

    panel_idx = 0

    # ── Individual hysteresis curves ──────────────────────────────────────

    colors = {
        "steel_menegotto_pinto_cyclic":       "#1f77b4",
        "steel_menegotto_pinto_3cycles":      "#2ca02c",
        "concrete_kent_park_cyclic":          "#d62728",
        "concrete_kent_park_confined_cyclic": "#ff7f0e",
        "j2_uniaxial_cyclic":                "#9467bd",
    }

    for key, (title, csv_path) in available.items():
        row, col = divmod(panel_idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        strain, stress = read_csv(csv_path)
        color = colors.get(key, "#333333")
        ax.plot(strain, stress, linewidth=0.8, color=color, alpha=0.9)
        ax.axhline(0, color="gray", linewidth=0.3, zorder=0)
        ax.axvline(0, color="gray", linewidth=0.3, zorder=0)
        ax.set_xlabel("Strain ε", fontsize=10)
        ax.set_ylabel("Stress σ (MPa)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)
        panel_idx += 1

    # ── Combined comparison panel ─────────────────────────────────────────

    if has_combined:
        row, col = divmod(panel_idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        strain, steel_s, concrete_s = read_combined_csv(combined_path)
        ax.plot(strain, steel_s, linewidth=0.8, color="#1f77b4",
                label="Steel (Menegotto-Pinto)", alpha=0.9)
        ax.plot(strain, concrete_s, linewidth=0.8, color="#d62728",
                label="Concrete (Kent-Park)", alpha=0.9)
        ax.axhline(0, color="gray", linewidth=0.3, zorder=0)
        ax.axvline(0, color="gray", linewidth=0.3, zorder=0)
        ax.set_xlabel("Strain ε", fontsize=10)
        ax.set_ylabel("Stress σ (MPa)", fontsize=10)
        ax.set_title("Steel vs Concrete — Same Protocol", fontsize=11,
                      fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)

    # ── Save ──────────────────────────────────────────────────────────────

    out_pdf = DATA_DIR / "hysteresis_curves.pdf"
    out_png = DATA_DIR / "hysteresis_curves.png"

    fig.savefig(str(out_pdf), dpi=150, bbox_inches="tight")
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to:")
    print(f"  {out_pdf}")
    print(f"  {out_png}")


if __name__ == "__main__":
    main()
