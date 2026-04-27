#!/usr/bin/env python3
"""
Freeze the first Ko-Bathe cracking audit for the reduced RC continuum column.

The goal of this audit is not to claim continuum convergence yet. The goal is
to validate the first physically coherent host/rebar pairings that can already
show cracking with embedded reinforcement:

* Hex20 + automatic rebar interpolation -> validated default resolves to 2-node
  bars on the serendipity host.
* Hex27 + automatic rebar interpolation -> validated default promotes to 3-node
  bars on the full triquadratic host.

Both cases run the same monotonic precompressed column slice and emit:
* runtime manifests
* control/hysteresis histories
* crack_state.csv
* VTK/PVD outputs for mesh, Gauss cloud, and crack planes
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.bbox": "tight",
        "figure.dpi": 140,
        "savefig.dpi": 300,
    }
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the first reduced RC continuum Ko-Bathe crack audit."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-exe",
        type=Path,
        default=repo_root
        / "build"
        / "fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "doc" / "figures" / "validation_reboot",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=repo_root / "PhD_Thesis" / "Figuras" / "validation_reboot",
    )
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for row in reader:
            rows.append(dict(row))
        return rows


@dataclass(frozen=True)
class ContinuumCrackAuditCase:
    key: str
    hex_order: str
    nx: int
    ny: int
    nz: int
    monotonic_tip_mm: float
    monotonic_steps: int
    axial_compression_mn: float
    axial_preload_steps: int
    timeout_seconds: int
    solver_policy: str = "newton-l2-only"
    continuation: str = "monolithic"
    reinforcement_mode: str = "embedded-longitudinal-bars"
    rebar_interpolation: str = "automatic"
    vtk_stride: int = 2


@dataclass(frozen=True)
class ContinuumCrackAuditRow:
    key: str
    hex_order: str
    mesh: str
    completed_successfully: bool
    solve_wall_seconds: float
    total_wall_seconds: float
    process_wall_seconds: float
    peak_base_shear_mn: float
    peak_cracked_gauss_points: int
    max_crack_opening: float
    first_crack_runtime_step: int
    first_crack_drift_mm: float
    output_dir: str


def build_command(
    exe: Path, case: ContinuumCrackAuditCase, output_dir: Path, print_progress: bool
) -> list[str]:
    command = [
        str(exe),
        "--output-dir",
        str(output_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        "nonlinear",
        "--reinforcement-mode",
        case.reinforcement_mode,
        "--rebar-interpolation",
        case.rebar_interpolation,
        "--hex-order",
        case.hex_order,
        "--nx",
        str(case.nx),
        "--ny",
        str(case.ny),
        "--nz",
        str(case.nz),
        "--solver-policy",
        case.solver_policy,
        "--continuation",
        case.continuation,
        "--axial-compression-mn",
        f"{case.axial_compression_mn}",
        "--axial-preload-steps",
        str(case.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{case.monotonic_tip_mm}",
        "--monotonic-steps",
        str(case.monotonic_steps),
        "--max-bisections",
        "8",
        "--write-vtk",
        "--vtk-stride",
        str(case.vtk_stride),
    ]
    if print_progress:
        command.append("--print-progress")
    return command


def first_crack_drift_mm(
    control_rows: list[dict[str, object]], first_runtime_step: int
) -> float:
    if first_runtime_step < 0:
        return math.nan
    for row in control_rows:
        try:
            runtime_step = int(float(row["runtime_step"]))
        except (KeyError, TypeError, ValueError):
            continue
        if runtime_step == first_runtime_step:
            return 1.0e3 * safe_float(row.get("target_drift_m"))
    return math.nan


def build_row(case: ContinuumCrackAuditCase, output_dir: Path) -> ContinuumCrackAuditRow:
    manifest = read_json(output_dir / "runtime_manifest.json")
    control_rows = read_csv_rows(output_dir / "control_state.csv")
    observables = manifest.get("observables") or {}
    timing = manifest.get("timing") or {}
    first_runtime_step = int(observables.get("first_crack_runtime_step", -1) or -1)
    return ContinuumCrackAuditRow(
        key=case.key,
        hex_order=case.hex_order,
        mesh=f"{case.nx}x{case.ny}x{case.nz}",
        completed_successfully=bool(manifest.get("completed_successfully")),
        solve_wall_seconds=safe_float(timing.get("solve_wall_seconds")),
        total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
        process_wall_seconds=safe_float(timing.get("total_wall_seconds")),
        peak_base_shear_mn=safe_float(observables.get("max_abs_base_shear_mn")),
        peak_cracked_gauss_points=int(
            observables.get("peak_cracked_gauss_points", 0) or 0
        ),
        max_crack_opening=safe_float(observables.get("max_crack_opening")),
        first_crack_runtime_step=first_runtime_step,
        first_crack_drift_mm=first_crack_drift_mm(control_rows, first_runtime_step),
        output_dir=str(output_dir),
    )


def run_case(
    exe: Path,
    root: Path,
    case: ContinuumCrackAuditCase,
    reuse_existing: bool,
    print_progress: bool,
) -> ContinuumCrackAuditRow:
    output_dir = root / case.key
    ensure_dir(output_dir)
    manifest_path = output_dir / "runtime_manifest.json"
    if reuse_existing and manifest_path.exists():
        return build_row(case, output_dir)

    command = build_command(exe, case, output_dir, print_progress)
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=case.timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - start

    (output_dir / "runner_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "runner_stderr.log").write_text(proc.stderr, encoding="utf-8")

    if not manifest_path.exists():
        raise RuntimeError(
            f"Continuum crack audit did not produce {manifest_path}.\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )

    row = build_row(case, output_dir)
    return ContinuumCrackAuditRow(
        **{**asdict(row), "process_wall_seconds": elapsed}
    )


def write_csv(path: Path, rows: list[ContinuumCrackAuditRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def merged_histories(row: ContinuumCrackAuditRow) -> list[dict[str, float]]:
    output_dir = Path(row.output_dir)
    control_rows = read_csv_rows(output_dir / "control_state.csv")
    crack_rows = {
        int(float(rec["runtime_step"])): rec
        for rec in read_csv_rows(output_dir / "crack_state.csv")
    }
    merged: list[dict[str, float]] = []
    for control in control_rows:
        runtime_step = int(float(control["runtime_step"]))
        crack = crack_rows.get(runtime_step)
        merged.append(
            {
                "target_drift_mm": 1.0e3 * safe_float(control.get("target_drift_m")),
                "base_shear_kn": 1.0e3 * safe_float(control.get("base_shear")),
                "cracked_gp": 0.0
                if crack is None
                else safe_float(crack.get("cracked_gauss_point_count")),
                "max_crack_opening": 0.0
                if crack is None
                else safe_float(crack.get("max_crack_opening")),
            }
        )
    return merged


def plot_base_shear(
    rows: list[ContinuumCrackAuditRow], primary: Path, secondary: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for row in rows:
        history = merged_histories(row)
        ax.plot(
            [rec["target_drift_mm"] for rec in history],
            [rec["base_shear_kn"] for rec in history],
            linewidth=2.0,
            label=f"{row.hex_order} auto",
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Reduced RC continuum monotonic response with Ko-Bathe cracking")
    ax.legend(frameon=False)
    fig.tight_layout()
    ensure_dir(primary.parent)
    fig.savefig(primary)
    ensure_dir(secondary.parent)
    fig.savefig(secondary)
    plt.close(fig)


def plot_crack_progression(
    rows: list[ContinuumCrackAuditRow], primary: Path, secondary: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    for row in rows:
        history = merged_histories(row)
        axes[0].plot(
            [rec["target_drift_mm"] for rec in history],
            [rec["cracked_gp"] for rec in history],
            linewidth=2.0,
            label=f"{row.hex_order} auto",
        )
        axes[1].plot(
            [rec["target_drift_mm"] for rec in history],
            [1.0e3 * rec["max_crack_opening"] for rec in history],
            linewidth=2.0,
            label=f"{row.hex_order} auto",
        )

    axes[0].set_xlabel("Tip drift [mm]")
    axes[0].set_ylabel("Cracked Gauss points [-]")
    axes[0].set_title("Crack onset and spread")

    axes[1].set_xlabel("Tip drift [mm]")
    axes[1].set_ylabel("Max crack opening x1e3 [-]")
    axes[1].set_title("Peak opening strain surrogate")

    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    ensure_dir(primary.parent)
    fig.savefig(primary)
    ensure_dir(secondary.parent)
    fig.savefig(secondary)
    plt.close(fig)


def plot_timing(
    rows: list[ContinuumCrackAuditRow], primary: Path, secondary: Path
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    labels = [f"{row.hex_order}\n{row.mesh}" for row in rows]
    values = [row.solve_wall_seconds for row in rows]
    bars = ax.bar(labels, values, color=["#0b5fa5", "#c2410c"])
    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{row.peak_cracked_gauss_points} gp",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Solve wall time [s]")
    ax.set_title("Continuum cracking audit timing")
    fig.tight_layout()
    ensure_dir(primary.parent)
    fig.savefig(primary)
    ensure_dir(secondary.parent)
    fig.savefig(secondary)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    cases = [
        ContinuumCrackAuditCase(
            key="hex20_auto_monotonic_20mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            axial_compression_mn=0.02,
            axial_preload_steps=4,
            timeout_seconds=240,
        ),
        ContinuumCrackAuditCase(
            key="hex27_auto_monotonic_20mm",
            hex_order="hex27",
            nx=2,
            ny=2,
            nz=2,
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            axial_compression_mn=0.02,
            axial_preload_steps=4,
            timeout_seconds=360,
        ),
    ]

    rows = [
        run_case(
            args.benchmark_exe,
            args.output_dir,
            case,
            args.reuse_existing,
            args.print_progress,
        )
        for case in cases
    ]

    summary = {
        "status": "completed",
        "case_count": len(rows),
        "completed_case_count": sum(1 for row in rows if row.completed_successfully),
        "hex20_peak_base_shear_kn": 1.0e3 * rows[0].peak_base_shear_mn,
        "hex27_peak_base_shear_kn": 1.0e3 * rows[1].peak_base_shear_mn,
        "hex20_peak_cracked_gp": rows[0].peak_cracked_gauss_points,
        "hex27_peak_cracked_gp": rows[1].peak_cracked_gauss_points,
        "hex20_max_crack_opening": rows[0].max_crack_opening,
        "hex27_max_crack_opening": rows[1].max_crack_opening,
        "hex20_first_crack_drift_mm": rows[0].first_crack_drift_mm,
        "hex27_first_crack_drift_mm": rows[1].first_crack_drift_mm,
        "hex20_over_hex27_peak_base_shear_ratio": (
            rows[0].peak_base_shear_mn / rows[1].peak_base_shear_mn
            if rows[1].peak_base_shear_mn > 0.0
            else math.nan
        ),
        "hex20_over_hex27_solve_time_ratio": (
            rows[0].solve_wall_seconds / rows[1].solve_wall_seconds
            if rows[1].solve_wall_seconds > 0.0
            else math.nan
        ),
        "validated_default_policy": {
            "hex20": "automatic -> two_node_linear rebar",
            "hex27": "automatic -> three_node_quadratic rebar",
        },
        "rows": [asdict(row) for row in rows],
    }

    write_json(args.output_dir / "continuum_crack_audit_summary.json", summary)
    write_csv(args.output_dir / "continuum_crack_audit_cases.csv", rows)

    plot_base_shear(
        rows,
        args.figures_dir / "reduced_rc_continuum_ko_bathe_base_shear.png",
        args.secondary_figures_dir
        / "reduced_rc_continuum_ko_bathe_base_shear.png",
    )
    plot_crack_progression(
        rows,
        args.figures_dir / "reduced_rc_continuum_ko_bathe_crack_progression.png",
        args.secondary_figures_dir
        / "reduced_rc_continuum_ko_bathe_crack_progression.png",
    )
    plot_timing(
        rows,
        args.figures_dir / "reduced_rc_continuum_ko_bathe_timing.png",
        args.secondary_figures_dir / "reduced_rc_continuum_ko_bathe_timing.png",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
