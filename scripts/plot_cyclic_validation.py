#!/usr/bin/env python3
"""
plot_cyclic_validation.py

Comparative plots for the cyclic validation suite.

Supported protocol presets:
  - legacy20   : geometric amplitudes {2.5, 5, 10, 20} mm
  - extended50 : extended amplitudes {2.5, 5, 10, 20, 35, 50} mm

Cases:
  0 (elastic), 1a-1i (beam N=2..10), 2a-2c (continuum),
  3 (table), 4 (FE2 one-way), 5 (FE2 two-way).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

plt = None
np = None
pd = None


def ensure_plot_deps() -> None:
    global plt, np, pd
    if plt is not None and np is not None and pd is not None:
        return

    try:
        import matplotlib.pyplot as _plt
        import numpy as _np
        import pandas as _pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "plotting dependencies are unavailable; use --summary-only "
            "for textual output") from exc

    plt = _plt
    np = _np
    pd = _pd


PROTOCOLS = {
    "legacy20": [0.0025, 0.005, 0.010, 0.020],
    "extended50": [0.0025, 0.005, 0.010, 0.020, 0.035, 0.050],
}


def require_matplotlib() -> None:
    ensure_plot_deps()


def cyclic_displacement(p: float, amps: list[float]) -> float:
    n_seg = 3 * len(amps)
    t = np.clip(p, 0.0, 1.0) * n_seg
    seg = int(np.clip(t, 0, n_seg - 1))
    f = t - seg
    level = seg // 3
    phase = seg % 3
    A = amps[level]
    if phase == 0:
        return f * A
    if phase == 1:
        return A * (1.0 - 2.0 * f)
    return -A * (1.0 - f)


def cyclic_displacement_v(p: np.ndarray, amps: list[float]) -> np.ndarray:
    return np.vectorize(lambda x: cyclic_displacement(x, amps))(p)


def protocol_label_mm(amps: list[float]) -> str:
    mm = []
    for a in amps:
        val = a * 1e3
        if abs(val - round(val)) < 1e-12:
            mm.append(str(int(round(val))))
        else:
            mm.append(f"{val:.1f}")
    return r"$\pm\{" + ", ".join(mm) + r"\}$ mm"


def load_case(base_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = base_dir / f"case{case_id}" / "hysteresis.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    return df["drift_m"].values, df["base_shear_MN"].values


def split_accepted_rows(df):
    if "accepted" not in df.columns:
        return df, df.iloc[0:0]
    accepted = pd.to_numeric(df["accepted"], errors="coerce").fillna(1.0) != 0.0
    return df[accepted], df[~accepted]


def plot_protocol(base_dir: Path, amps: list[float], protocol_name: str) -> list[str]:
    require_matplotlib()
    p = np.linspace(0, 1, 1000)
    d = cyclic_displacement_v(p, amps) * 1e3

    p_steps = np.linspace(0, 1, 3 * len(amps) * 10 + 1)
    d_steps = cyclic_displacement_v(p_steps, amps) * 1e3

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(p, d, "k-", linewidth=1.2)
    ax.plot(p_steps, d_steps, "o", color="#1f77b4", markersize=2, alpha=0.55)
    ax.set_xlabel("Control parameter $p$")
    ax.set_ylabel("Drift $d(p)$ [mm]")
    ax.set_title(f"Cyclic protocol {protocol_name}: {protocol_label_mm(amps)}")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = base_dir / "protocol.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [out.name]


def plot_group(base_dir: Path,
               case_ids: list[str],
               labels: list[str],
               colors: list[str],
               title: str,
               filename: str) -> list[str]:
    require_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for cid, lbl, col in zip(case_ids, labels, colors):
        data = load_case(base_dir, cid)
        if data is None:
            continue
        drift_mm = data[0] * 1e3
        shear_kN = data[1] * 1e3
        ax.plot(drift_mm, shear_kN, color=col, linewidth=1.0,
                label=lbl, alpha=0.88)

    ax.set_xlabel("Lateral drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title(title)
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.5)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = base_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [out.name]


def plot_case_fiber_hysteresis(base_dir: Path, case_id: str, material: str) -> list[str]:
    require_matplotlib()
    path = base_dir / f"case{case_id}" / "recorders" / f"fiber_hysteresis_{material}.csv"
    if not path.is_file():
        return []

    df = pd.read_csv(path)
    strain_cols = [c for c in df.columns if c.endswith("_strain")]
    stress_cols = [c for c in df.columns if c.endswith("_stress")]
    if not strain_cols or len(strain_cols) != len(stress_cols):
        return []

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(strain_cols)))
    for i, (eps_col, sig_col) in enumerate(zip(strain_cols, stress_cols)):
        label = eps_col.replace("_strain", "")
        ax.plot(df[eps_col], df[sig_col], color=colors[i], linewidth=1.0,
                alpha=0.9, label=label)

    ax.set_xlabel("Fiber strain")
    ax.set_ylabel("Fiber stress [MPa]")
    ax.set_title(f"Case {case_id}: {material.capitalize()} fiber hysteresis")
    ax.axhline(0.0, color="gray", linewidth=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out = base_dir / f"case{case_id}_fiber_{material}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [out.name]


def plot_case_crack_evolution(base_dir: Path, case_id: str) -> list[str]:
    require_matplotlib()
    path = base_dir / f"case{case_id}" / "recorders" / "crack_evolution.csv"
    if not path.is_file():
        return []

    df_all = pd.read_csv(path)
    df, failed = split_accepted_rows(df_all)
    if df.empty:
        df = df_all
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.2))
    drift_mm = df["drift_m"] * 1e3 if "drift_m" in df.columns else df["step"]
    failed_drift_mm = (
        failed["drift_m"] * 1e3 if "drift_m" in failed.columns else failed["step"]
    )

    axes[0, 0].plot(drift_mm, df["total_cracked_gps"], color="#1f77b4", lw=1.0)
    axes[0, 0].set_title("Cracked Gauss points", fontsize=9)
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].plot(drift_mm, df["total_cracks"], color="#ff7f0e", lw=1.0)
    axes[0, 1].set_title("Accumulated cracks", fontsize=9)

    if "peak_damage_scalar" in df.columns:
        axes[1, 0].plot(drift_mm, df["peak_damage_scalar"],
                        color="#d62728", lw=1.0, label="peak_damage_scalar")
    if "max_tau_o_max" in df.columns:
        axes[1, 0].plot(drift_mm, df["max_tau_o_max"],
                        color="#9467bd", lw=1.0, label="max_tau_o_max")
    if "most_compressive_sigma_o_max" in df.columns:
        axes[1, 0].plot(drift_mm, np.abs(df["most_compressive_sigma_o_max"]),
                        color="#8c564b", lw=1.0,
                        label="|most_compressive_sigma_o_max|")
    elif "max_damage" in df.columns:
        axes[1, 0].plot(drift_mm, df["max_damage"],
                        color="#d62728", lw=1.0, label="max_damage")
    axes[1, 0].set_title("Fracture indicators", fontsize=9)
    axes[1, 0].set_ylabel("Envelope value")
    axes[1, 0].set_xlabel("Drift [mm]" if "drift_m" in df.columns else "Step")
    axes[1, 0].legend(fontsize=7, loc="best")

    axes[1, 1].plot(drift_mm, df["max_opening"], color="#2ca02c", lw=1.0)
    axes[1, 1].set_title("Peak crack opening", fontsize=9)
    axes[1, 1].set_xlabel("Drift [mm]" if "drift_m" in df.columns else "Step")

    if not failed.empty:
        axes[0, 0].scatter(
            failed_drift_mm, failed["total_cracked_gps"],
            color="#1f77b4", marker="x", s=28, linewidths=1.0,
            label="failed FE2 attempt")
        axes[0, 1].scatter(
            failed_drift_mm, failed["total_cracks"],
            color="#ff7f0e", marker="x", s=28, linewidths=1.0)
        if "peak_damage_scalar" in failed.columns:
            axes[1, 0].scatter(
                failed_drift_mm, failed["peak_damage_scalar"],
                color="#d62728", marker="x", s=28, linewidths=1.0)
        if "max_tau_o_max" in failed.columns:
            axes[1, 0].scatter(
                failed_drift_mm, failed["max_tau_o_max"],
                color="#9467bd", marker="x", s=28, linewidths=1.0)
        if "most_compressive_sigma_o_max" in failed.columns:
            axes[1, 0].scatter(
                failed_drift_mm, np.abs(failed["most_compressive_sigma_o_max"]),
                color="#8c564b", marker="x", s=28, linewidths=1.0)
        axes[1, 1].scatter(
            failed_drift_mm, failed["max_opening"],
            color="#2ca02c", marker="x", s=28, linewidths=1.0)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.axvline(0.0, color="gray", linewidth=0.4)

    fig.suptitle(f"Case {case_id}: crack evolution", fontsize=11)
    fig.tight_layout()
    out = base_dir / f"case{case_id}_crack_evolution.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [out.name]


def plot_case_global_history(base_dir: Path, case_id: str) -> list[str]:
    require_matplotlib()
    path = base_dir / f"case{case_id}" / "recorders" / "global_history.csv"
    if not path.is_file():
        return []

    df_all = pd.read_csv(path)
    if "peak_damage" not in df_all.columns:
        return []
    df, failed = split_accepted_rows(df_all)
    if df.empty:
        df = df_all

    x = df["drift_m"] * 1e3 if "drift_m" in df.columns else df["step"]
    failed_x = (
        failed["drift_m"] * 1e3 if "drift_m" in failed.columns else failed["step"]
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.8), sharex=True)

    axes[0].plot(x, df["peak_damage"], color="#d62728", lw=1.0)
    axes[0].set_ylabel("Peak damage")
    axes[0].set_title("Structural damage envelope", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    if "total_cracks" in df.columns:
        axes[1].plot(x, df["total_cracks"], color="#1f77b4", lw=1.0,
                     label="total_cracks")
    if "peak_submodel_damage_scalar" in df.columns:
        axes[1].plot(x, df["peak_submodel_damage_scalar"],
                     color="#ff7f0e", lw=1.0,
                     label="peak_submodel_damage_scalar")
    elif "peak_submodel_damage" in df.columns:
        axes[1].plot(x, df["peak_submodel_damage"], color="#ff7f0e", lw=1.0,
                     label="peak_submodel_damage")
    if "max_submodel_tau_o_max" in df.columns:
        axes[1].plot(x, df["max_submodel_tau_o_max"], color="#9467bd", lw=1.0,
                     label="max_submodel_tau_o_max")
    if "most_compressive_submodel_sigma_o_max" in df.columns:
        axes[1].plot(x, np.abs(df["most_compressive_submodel_sigma_o_max"]),
                     color="#8c564b", lw=1.0,
                     label="|most_compressive_submodel_sigma_o_max|")
    if "fe2_iterations" in df.columns:
        axes[1].plot(x, df["fe2_iterations"], color="#2ca02c", lw=1.0,
                     label="fe2_iterations")
    if not failed.empty:
        if "total_cracks" in failed.columns:
            axes[1].scatter(
                failed_x, failed["total_cracks"], color="#1f77b4",
                marker="x", s=28, linewidths=1.0, label="failed total_cracks")
        if "peak_submodel_damage_scalar" in failed.columns:
            axes[1].scatter(
                failed_x, failed["peak_submodel_damage_scalar"],
                color="#ff7f0e", marker="x", s=28, linewidths=1.0,
                label="failed peak_submodel_damage_scalar")
        if "max_submodel_tau_o_max" in failed.columns:
            axes[1].scatter(
                failed_x, failed["max_submodel_tau_o_max"],
                color="#9467bd", marker="x", s=28, linewidths=1.0)
        if "most_compressive_submodel_sigma_o_max" in failed.columns:
            axes[1].scatter(
                failed_x, np.abs(failed["most_compressive_submodel_sigma_o_max"]),
                color="#8c564b", marker="x", s=28, linewidths=1.0)
    axes[1].set_xlabel("Drift [mm]" if "drift_m" in df.columns else "Step")
    axes[1].set_ylabel("Response metrics")
    axes[1].set_title("Multiscale response indicators", fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8, loc="best")

    fig.suptitle(f"Case {case_id}: global history", fontsize=11)
    fig.tight_layout()
    out = base_dir / f"case{case_id}_global_history.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [out.name]


def plot_case_solver_diagnostics(base_dir: Path, case_id: str) -> list[str]:
    require_matplotlib()
    path = base_dir / f"case{case_id}" / "recorders" / "solver_diagnostics.csv"
    if not path.is_file():
        return []

    df = pd.read_csv(path)
    if df.empty or "step" not in df.columns:
        return []

    def max_of_suffix(suffix: str) -> np.ndarray | None:
        cols = [c for c in df.columns if c.endswith(suffix)]
        if not cols:
            return None
        return df[cols].max(axis=1).to_numpy()

    x = df["drift_m"] * 1e3 if "drift_m" in df.columns else df["step"]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.4), sharex=True)

    axes[0, 0].plot(x, df["failed_submodels"], color="#d62728", lw=1.0,
                    label="failed_submodels")
    if "regularized_submodels" in df.columns:
        axes[0, 0].plot(x, df["regularized_submodels"], color="#ff7f0e", lw=1.0,
                        label="regularized_submodels")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Step outcome")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")

    max_iters = max_of_suffix("_snes_iterations")
    if max_iters is not None:
        axes[0, 1].plot(x, max_iters, color="#1f77b4", lw=1.0)
    axes[0, 1].set_ylabel("Iterations")
    axes[0, 1].set_title("Max local SNES iterations")
    axes[0, 1].grid(True, alpha=0.3)

    max_fnorm = max_of_suffix("_function_norm")
    if max_fnorm is not None:
        axes[1, 0].plot(x, np.maximum(max_fnorm, 1e-16), color="#2ca02c", lw=1.0)
        axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel(r"$||F||$")
    axes[1, 0].set_xlabel("Drift [mm]" if "drift_m" in df.columns else "Step")
    axes[1, 0].set_title("Max local residual norm")
    axes[1, 0].grid(True, alpha=0.3)

    achieved = max_of_suffix("_achieved_fraction")
    if achieved is not None:
        axes[1, 1].plot(x, achieved, color="#9467bd", lw=1.0,
                        label="max achieved fraction")
    bisections = max_of_suffix("_adaptive_bisections")
    if bisections is not None:
        axes[1, 1].plot(x, bisections, color="#8c564b", lw=1.0,
                        label="max adaptive bisections")
    axes[1, 1].set_xlabel("Drift [mm]" if "drift_m" in df.columns else "Step")
    axes[1, 1].set_ylabel("Adaptive solve metrics")
    axes[1, 1].set_title("Local solve progression")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8, loc="best")

    fig.suptitle(f"Case {case_id}: solver diagnostics", fontsize=11)
    fig.tight_layout()
    out = base_dir / f"case{case_id}_solver_diagnostics.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [out.name]


def copy_pdfs(base_dir: Path, fig_dir: Path, names: list[str]) -> None:
    for name in names:
        src = base_dir / name
        if src.is_file():
            dst = fig_dir / name
            shutil.copy2(src, dst)
            print(f"  Copied -> {dst}")


def print_case_summary(base_dir: Path, case_id: str) -> None:
    case_dir = base_dir / f"case{case_id}"
    hyst_path = case_dir / "hysteresis.csv"
    max_drift_mm = 0.0
    max_shear_kN = 0.0
    num_records = 0
    if hyst_path.is_file():
        with hyst_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                num_records += 1
                max_drift_mm = max(max_drift_mm, abs(float(row["drift_m"])) * 1e3)
                max_shear_kN = max(max_shear_kN, abs(float(row["base_shear_MN"])) * 1e3)

    peak_damage = None
    max_total_cracks = None
    max_opening_mm = None
    max_active_crack_history_points = None
    max_num_cracks_at_point = None
    failed_attempts = 0
    global_path = case_dir / "recorders" / "global_history.csv"
    if global_path.is_file():
        global_records = 0
        with global_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                global_records += 1
                accepted = row.get("accepted", "1")
                if accepted and accepted not in ("1", "1.0", "True", "true"):
                    failed_attempts += 1
                if "drift_m" in row and row["drift_m"]:
                    drift = float(row["drift_m"])
                    if math.isfinite(drift):
                        max_drift_mm = max(max_drift_mm, abs(drift) * 1e3)
                if "base_shear_MN" in row and row["base_shear_MN"]:
                    shear = float(row["base_shear_MN"])
                    if math.isfinite(shear):
                        max_shear_kN = max(max_shear_kN, abs(shear) * 1e3)
                if "peak_damage" in row and row["peak_damage"]:
                    value = float(row["peak_damage"])
                    if math.isfinite(value):
                        peak_damage = value if peak_damage is None else max(peak_damage, value)
                if "total_cracks" in row and row["total_cracks"]:
                    value = int(float(row["total_cracks"]))
                    max_total_cracks = value if max_total_cracks is None else max(max_total_cracks, value)
                if "total_active_crack_history_points" in row and row["total_active_crack_history_points"]:
                    value = int(float(row["total_active_crack_history_points"]))
                    max_active_crack_history_points = value if max_active_crack_history_points is None else max(max_active_crack_history_points, value)
                if "max_num_cracks_at_point" in row and row["max_num_cracks_at_point"]:
                    value = int(float(row["max_num_cracks_at_point"]))
                    max_num_cracks_at_point = value if max_num_cracks_at_point is None else max(max_num_cracks_at_point, value)
                if "max_opening" in row and row["max_opening"]:
                    opening = float(row["max_opening"])
                    if math.isfinite(opening):
                        value = abs(opening) * 1e3
                        max_opening_mm = value if max_opening_mm is None else max(max_opening_mm, value)
        num_records = max(num_records, global_records)

    if num_records == 0:
        return

    print(f"Case {case_id}: records={num_records}  max|drift|={max_drift_mm:.3f} mm  "
          f"max|V|={max_shear_kN:.3f} kN", end="")
    if peak_damage is not None:
        print(f"  peak_damage={peak_damage:.5f}", end="")
    if max_total_cracks is not None:
        print(f"  max_cracks={max_total_cracks}", end="")
    if max_active_crack_history_points is not None:
        print(f"  active_crack_history_pts={max_active_crack_history_points}", end="")
    if max_num_cracks_at_point is not None:
        print(f"  max_cracks_per_point={max_num_cracks_at_point}", end="")
    if max_opening_mm is not None:
        print(f"  max_opening={max_opening_mm:.4f} mm", end="")
    if failed_attempts:
        print(f"  failed_attempts={failed_attempts}", end="")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cyclic validation hysteresis and recorder outputs")
    parser.add_argument(
        "--dir", default="data/output/cyclic_validation",
        help="Base output directory")
    parser.add_argument(
        "--figures-dir", default="doc/figures/cyclic_validation",
        help="LaTeX figures directory")
    parser.add_argument(
        "--protocol", choices=sorted(PROTOCOLS.keys()), default="extended50",
        help="Protocol preset used by the validation run")
    parser.add_argument(
        "--summary-only", action="store_true",
        help="Print textual case summaries even when plotting dependencies are unavailable")
    args = parser.parse_args()

    base = Path(args.dir)
    if not base.is_dir():
        print(f"Directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    amps = PROTOCOLS[args.protocol]
    print(f"Plotting from: {base}")
    print(f"Protocol: {args.protocol} -> {protocol_label_mm(amps)}")

    for case_id in ["0", "1a", "1b", "1c", "1d", "2a", "2b", "2c", "3", "4", "5"]:
        print_case_summary(base, case_id)

    if args.summary_only:
        print("Summary-only mode: skipping PDF generation.")
        return

    require_matplotlib()

    generated: list[str] = []
    generated += plot_protocol(base, amps, args.protocol)
    generated += plot_group(
        base,
        case_ids=["0", "1a", "1b", "1c", "1d"],
        labels=[
            "Case 0: Elastic (N=3)",
            "Case 1a: N=2 (1 GP)",
            "Case 1b: N=3 (2 GPs)",
            "Case 1c: N=4 (3 GPs)",
            "Case 1d: N=5 (4 GPs)",
        ],
        colors=["#ba4b4b", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        title="Beam convergence study (single cantilever column)",
        filename="comparison_beam.pdf")
    generated += plot_group(
        base,
        case_ids=["0", "2a", "2b", "2c"],
        labels=[
            "Case 0: Elastic reference",
            "Case 2a: Hex8 (reinforced)",
            "Case 2b: Hex20 (reinforced)",
            "Case 2c: Hex27 (reinforced)",
        ],
        colors=["#7f7f7f", "#d62728", "#9467bd", "#8c564b"],
        title="Reinforced continuum column: hex order comparison",
        filename="comparison_continuum.pdf")
    generated += plot_group(
        base,
        case_ids=["3", "4", "5"],
        labels=[
            "Case 3: Table (fiber beams)",
            "Case 4: Table + FE2 (one-way)",
            "Case 5: Table + FE2 (two-way)",
        ],
        colors=["#000000", "#e377c2", "#17becf"],
        title="Structural model vs FE2 multiscale coupling",
        filename="comparison_fe2.pdf")
    generated += plot_group(
        base,
        case_ids=["0", "1a", "1b", "1c", "1d", "2a", "2b", "2c"],
        labels=[
            "0: Elastic", "1a: Beam N=2", "1b: Beam N=3",
            "1c: Beam N=4", "1d: Beam N=5",
            "2a: Hex8", "2b: Hex20", "2c: Hex27",
        ],
        colors=[
            "#ba4b4b", "#1f77b4", "#ff7f0e",
            "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2",
        ],
        title="Single-column cases comparison",
        filename="comparison_all.pdf")

    for case_id in ["3", "4", "5"]:
        generated += plot_case_fiber_hysteresis(base, case_id, "concrete")
        generated += plot_case_fiber_hysteresis(base, case_id, "steel")
        generated += plot_case_global_history(base, case_id)
    for case_id in ["4", "5"]:
        generated += plot_case_crack_evolution(base, case_id)
        generated += plot_case_solver_diagnostics(base, case_id)

    copy_pdfs(base, fig_dir, generated)
    print("Done.")


if __name__ == "__main__":
    main()
