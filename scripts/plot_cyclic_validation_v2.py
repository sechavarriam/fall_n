#!/usr/bin/env python3
"""
plot_cyclic_validation_v2.py

Comprehensive comparative plots for the V2 cyclic validation suite.

Column: 4.0 m × 0.50 m × 0.30 m, f'c = 28 MPa, fy = 420 MPa.

Phases:
  1. Material-level: Kent-Park concrete, Menegotto-Pinto steel σ-ε hysteresis
  2. Section-level: M-κ monotonic + cyclic (strong & weak axes)
  3. Beam convergence: Timoshenko<N> (N=2..10) force-displacement
  4. Continuum mesh convergence: Hex8/Hex20/Hex27 × coarse/medium/fine
  5. Cross-scale comparison: beam vs continuum selected models
  6-7. FE² one-way / two-way vs references

Usage:
  python scripts/plot_cyclic_validation_v2.py
  python scripts/plot_cyclic_validation_v2.py --summary-only
  python scripts/plot_cyclic_validation_v2.py --phases 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

plt = None
np = None


def ensure_deps():
    global plt, np
    if plt is not None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np
    plt = _plt
    np = _np
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  CSV loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_csv(path: Path) -> dict[str, list[float]]:
    """Load a CSV file into a dict of column_name → list of float values."""
    if not path.exists():
        return {}
    with open(path) as f:
        reader = csv.DictReader(f)
        data = {col: [] for col in reader.fieldnames}
        for row in reader:
            for col in reader.fieldnames:
                try:
                    data[col].append(float(row[col]))
                except (ValueError, TypeError):
                    data[col].append(float("nan"))
    return data


def load_hysteresis(base: Path, subdir: str) -> dict:
    return load_csv(base / subdir / "hysteresis.csv")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 1: Material-level plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase1(base: Path, fig_dir: Path):
    ensure_deps()
    mat_dir = base / "material"

    # --- Kent-Park concrete ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for label, fname, ax in [
        ("Unconfined", "kent_park_unconfined.csv", axes[0]),
        ("Confined", "kent_park_confined.csv", axes[1]),
    ]:
        data = load_csv(mat_dir / fname)
        if data:
            ax.plot(data["strain"], data["stress_MPa"], "b-", lw=0.8)
            ax.set_xlabel(r"Strain $\varepsilon$")
            ax.set_ylabel(r"Stress $\sigma$ (MPa)")
            ax.set_title(f"Kent-Park Concrete ({label})")
            ax.axhline(0, color="k", lw=0.3)
            ax.axvline(0, color="k", lw=0.3)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "material_concrete_hysteresis.pdf")
    plt.close(fig)
    print(f"  [Phase 1] material_concrete_hysteresis.pdf")

    # --- Menegotto-Pinto steel ---
    data = load_csv(mat_dir / "menegotto_pinto.csv")
    if data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data["strain"], data["stress_MPa"], "r-", lw=0.8)
        ax.set_xlabel(r"Strain $\varepsilon$")
        ax.set_ylabel(r"Stress $\sigma$ (MPa)")
        ax.set_title("Menegotto-Pinto Steel — Cyclic Response")
        ax.axhline(0, color="k", lw=0.3)
        ax.axvline(0, color="k", lw=0.3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "material_steel_hysteresis.pdf")
        plt.close(fig)
        print(f"  [Phase 1] material_steel_hysteresis.pdf")

    # --- Energy comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for fname, label, color in [
        ("kent_park_unconfined.csv", "KP Unconfined", "blue"),
        ("kent_park_confined.csv", "KP Confined", "green"),
    ]:
        data = load_csv(mat_dir / fname)
        if data:
            ax.plot(data["strain"], data["cumulative_energy"],
                    color=color, lw=1.0, label=label)
    data = load_csv(mat_dir / "menegotto_pinto.csv")
    if data:
        ax.plot(data["strain"], data["cumulative_energy"],
                color="red", lw=1.0, label="MP Steel")
    ax.set_xlabel(r"Strain $\varepsilon$")
    ax.set_ylabel("Cumulative Energy Density")
    ax.set_title("Energy Dissipation Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "material_energy_comparison.pdf")
    plt.close(fig)
    print(f"  [Phase 1] material_energy_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 2: Section-level M-κ plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase2(base: Path, fig_dir: Path):
    ensure_deps()
    sec_dir = base / "section"

    # --- Monotonic M-κ ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis_name, fname, ax in [
        ("Strong Axis", "mk_monotonic_strong.csv", axes[0]),
        ("Weak Axis", "mk_monotonic_weak.csv", axes[1]),
    ]:
        data = load_csv(sec_dir / fname)
        if data:
            ax.plot(data["curvature_1pm"], data["moment_kNm"], "b-", lw=1.2)
            ax.set_xlabel(r"Curvature $\kappa$ (1/m)")
            ax.set_ylabel(r"Moment $M$ (kN·m)")
            ax.set_title(f"Monotonic M-κ ({axis_name})")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "section_mk_monotonic.pdf")
    plt.close(fig)
    print(f"  [Phase 2] section_mk_monotonic.pdf")

    # --- Cyclic M-κ ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis_name, fname, ax in [
        ("Strong Axis", "mk_cyclic_strong.csv", axes[0]),
        ("Weak Axis", "mk_cyclic_weak.csv", axes[1]),
    ]:
        data = load_csv(sec_dir / fname)
        if data:
            ax.plot(data["curvature_1pm"], data["moment_kNm"], "b-", lw=0.6)
            ax.set_xlabel(r"Curvature $\kappa$ (1/m)")
            ax.set_ylabel(r"Moment $M$ (kN·m)")
            ax.set_title(f"Cyclic M-κ ({axis_name})")
            ax.axhline(0, color="k", lw=0.3)
            ax.axvline(0, color="k", lw=0.3)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "section_mk_cyclic.pdf")
    plt.close(fig)
    print(f"  [Phase 2] section_mk_cyclic.pdf")

    # --- Fiber strain history from cyclic ---
    data = load_csv(sec_dir / "mk_cyclic_strong.csv")
    if data and "max_concrete_strain" in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = data["step"]
        ax.plot(steps, data["max_concrete_strain"], "b-", lw=0.8,
                label="Max concrete strain")
        ax.plot(steps, data["max_steel_strain"], "r-", lw=0.8,
                label="Max steel strain")
        ax.set_xlabel("Step")
        ax.set_ylabel("Fiber Strain")
        ax.set_title("Extreme Fiber Strain History (Cyclic, Strong Axis)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "section_fiber_history.pdf")
        plt.close(fig)
        print(f"  [Phase 2] section_fiber_history.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 3: Beam convergence plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase3(base: Path, fig_dir: Path):
    ensure_deps()

    # Overlay: selected N values
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 9))
    fig, ax = plt.subplots(figsize=(10, 6))

    peak_forces = {}
    for i, N in enumerate(range(2, 11)):
        subdir = f"beam_N{N:02d}"
        data = load_hysteresis(base, subdir)
        if not data:
            continue
        drift_mm = [d * 1e3 for d in data["drift_m"]]
        shear_kN = [v * 1e3 for v in data["base_shear_MN"]]  # MN → kN
        ax.plot(drift_mm, shear_kN, color=colors[i], lw=0.7,
                label=f"N={N}", alpha=0.8)
        peak_forces[N] = max(abs(v) for v in shear_kN)

    ax.set_xlabel("Drift (mm)")
    ax.set_ylabel("Base Shear (kN)")
    ax.set_title("Beam Element Convergence — Timoshenko⟨N⟩")
    ax.legend(ncol=3, loc="upper left")
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "beam_convergence_overlay.pdf")
    plt.close(fig)
    print(f"  [Phase 3] beam_convergence_overlay.pdf")

    # Convergence curve: peak force vs N
    if peak_forces:
        fig, ax = plt.subplots(figsize=(8, 5))
        Ns = sorted(peak_forces.keys())
        peaks = [peak_forces[n] for n in Ns]
        ax.plot(Ns, peaks, "ko-", lw=1.5, markersize=6)
        if len(Ns) >= 2:
            ref = peaks[-1]
            ax.axhline(ref, color="gray", ls="--", lw=0.8, label=f"N=10 ref")
            ax.axhline(ref * 1.02, color="red", ls=":", lw=0.6,
                       label="±2% band")
            ax.axhline(ref * 0.98, color="red", ls=":", lw=0.6)
        ax.set_xlabel("Number of Nodes N")
        ax.set_ylabel("Peak Base Shear (kN)")
        ax.set_title("Beam Element Convergence Curve")
        ax.legend()
        ax.set_xticks(Ns)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "beam_convergence_curve.pdf")
        plt.close(fig)
        print(f"  [Phase 3] beam_convergence_curve.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 4: Continuum mesh convergence plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase4(base: Path, fig_dir: Path):
    ensure_deps()

    hex_types = {
        "Hex8": ["hex8_coarse", "hex8_medium", "hex8_fine"],
        "Hex20": ["hex20_coarse", "hex20_medium", "hex20_fine"],
        "Hex27": ["hex27_coarse", "hex27_medium", "hex27_fine"],
    }

    density_colors = {"coarse": "C0", "medium": "C1", "fine": "C2"}

    for hex_name, dirs in hex_types.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for subdir in dirs:
            density = subdir.split("_")[1]
            data = load_hysteresis(base, subdir)
            if not data:
                continue
            drift_mm = [d * 1e3 for d in data["drift_m"]]
            shear_kN = [v * 1e3 for v in data["base_shear_MN"]]
            ax.plot(drift_mm, shear_kN, color=density_colors[density],
                    lw=0.7, label=density.capitalize())

        ax.set_xlabel("Drift (mm)")
        ax.set_ylabel("Base Shear (kN)")
        ax.set_title(f"{hex_name} Mesh Convergence")
        ax.legend()
        ax.axhline(0, color="k", lw=0.3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = f"{hex_name.lower()}_mesh_convergence.pdf"
        fig.savefig(fig_dir / fname)
        plt.close(fig)
        print(f"  [Phase 4] {fname}")

    # Cross-element comparison (finest mesh of each)
    fig, ax = plt.subplots(figsize=(10, 6))
    for hex_name, dirs in hex_types.items():
        data = load_hysteresis(base, dirs[-1])  # finest
        if not data:
            continue
        drift_mm = [d * 1e3 for d in data["drift_m"]]
        shear_kN = [v * 1e3 for v in data["base_shear_MN"]]
        ax.plot(drift_mm, shear_kN, lw=0.8, label=f"{hex_name} (fine)")

    # Add beam reference (N=10)
    data = load_hysteresis(base, "beam_N10")
    if data:
        drift_mm = [d * 1e3 for d in data["drift_m"]]
        shear_kN = [v * 1e3 for v in data["base_shear_MN"]]
        ax.plot(drift_mm, shear_kN, "k--", lw=1.0, label="Beam N=10 (ref)")

    ax.set_xlabel("Drift (mm)")
    ax.set_ylabel("Base Shear (kN)")
    ax.set_title("Cross-Element Comparison (Finest Mesh)")
    ax.legend()
    ax.axhline(0, color="k", lw=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "continuum_cross_comparison.pdf")
    plt.close(fig)
    print(f"  [Phase 4] continuum_cross_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Phase 6-7: FE² validation plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_phase67(base: Path, fig_dir: Path):
    ensure_deps()

    fig, ax = plt.subplots(figsize=(10, 6))
    refs = [
        ("beam_N10", "Beam N=10", "k--"),
        ("hex20_fine", "Hex20 (fine)", "C1-"),
        ("fe2_oneway", r"FE$^2$ one-way", "C3-"),
        ("fe2_twoway", r"FE$^2$ two-way", "C4-"),
    ]
    for subdir, label, style in refs:
        data = load_hysteresis(base, subdir)
        if not data:
            continue
        drift_mm = [d * 1e3 for d in data["drift_m"]]
        shear_kN = [v * 1e3 for v in data["base_shear_MN"]]
        ax.plot(drift_mm, shear_kN, style, lw=0.8, label=label)

    ax.set_xlabel("Drift (mm)")
    ax.set_ylabel("Base Shear (kN)")
    ax.set_title(r"FE$^2$ Multiscale Validation — Full Comparison")
    ax.legend()
    ax.axhline(0, color="k", lw=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fe2_twoway_vs_all.pdf")
    plt.close(fig)
    print(f"  [Phase 6-7] fe2_twoway_vs_all.pdf")


# ═══════════════════════════════════════════════════════════════════════════
#  Summary table output
# ═══════════════════════════════════════════════════════════════════════════

def print_summary_tables(base: Path):
    print("\n" + "=" * 72)
    print("  SUMMARY TABLES")
    print("=" * 72)

    # Beam convergence
    print("\n  Beam Convergence (Timoshenko<N>):")
    print(f"  {'N':>4s}  {'Records':>8s}  {'Peak V (kN)':>12s}")
    print("  " + "-" * 28)
    for N in range(2, 11):
        data = load_hysteresis(base, f"beam_N{N:02d}")
        if not data:
            continue
        peak = max(abs(v) * 1e3 for v in data["base_shear_MN"])
        print(f"  {N:4d}  {len(data['step']):8d}  {peak:12.2f}")

    # Continuum convergence
    print("\n  Continuum Mesh Convergence:")
    print(f"  {'Config':>16s}  {'Records':>8s}  {'Peak V (kN)':>12s}")
    print("  " + "-" * 40)
    for label in [
        "hex8_coarse", "hex8_medium", "hex8_fine",
        "hex20_coarse", "hex20_medium", "hex20_fine",
        "hex27_coarse", "hex27_medium", "hex27_fine",
    ]:
        data = load_hysteresis(base, label)
        if not data:
            continue
        peak = max(abs(v) * 1e3 for v in data["base_shear_MN"])
        print(f"  {label:>16s}  {len(data['step']):8d}  {peak:12.2f}")

    # FE² comparison
    print("\n  FE² Comparison:")
    print(f"  {'Model':>16s}  {'Records':>8s}  {'Peak V (kN)':>12s}")
    print("  " + "-" * 40)
    for label in ["beam_N10", "hex20_fine", "fe2_oneway", "fe2_twoway"]:
        data = load_hysteresis(base, label)
        if not data:
            continue
        peak = max(abs(v) * 1e3 for v in data["base_shear_MN"])
        print(f"  {label:>16s}  {len(data['step']):8d}  {peak:12.2f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="V2 comprehensive cyclic validation plots")
    parser.add_argument("--base", type=str, default=None,
                        help="Base output directory (default: auto-detect)")
    parser.add_argument("--fig-dir", type=str, default=None,
                        help="Figure output directory")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print summary tables only, no plots")
    parser.add_argument("--phases", nargs="+", type=int, default=None,
                        help="Which phases to plot (1,2,3,4,67)")
    args = parser.parse_args()

    # Auto-detect base directory
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    base = Path(args.base) if args.base else repo_root / "data" / "output" / "cyclic_validation_v2"
    fig_dir = Path(args.fig_dir) if args.fig_dir else repo_root / "doc" / "figures" / "cyclic_validation_v2"

    if not base.exists():
        print(f"ERROR: Base directory not found: {base}")
        print("Run the validation suite first:")
        print("  ./fall_n_table_cyclic_validation --case v2_all")
        sys.exit(1)

    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Base: {base}")
    print(f"  Figures: {fig_dir}")

    phases = args.phases or [1, 2, 3, 4, 67]

    if not args.summary_only:
        if 1 in phases:
            plot_phase1(base, fig_dir)
        if 2 in phases:
            plot_phase2(base, fig_dir)
        if 3 in phases:
            plot_phase3(base, fig_dir)
        if 4 in phases:
            plot_phase4(base, fig_dir)
        if 67 in phases:
            plot_phase67(base, fig_dir)

    print_summary_tables(base)


if __name__ == "__main__":
    main()
