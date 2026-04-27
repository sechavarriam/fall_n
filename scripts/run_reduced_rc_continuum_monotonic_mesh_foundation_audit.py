#!/usr/bin/env python3
"""
Freeze a monotonic mesh-foundation audit for the promoted RC continuum local model.

This audit is intentionally narrow and physically motivated:

1. same nonlinear concrete profile and embedded interior rebar model;
2. same axial preload and monotonic pushover to 20 mm;
3. explicit separation between:
   - current promoted baseline       : Hex20 4x4x2 uniform,
   - transverse refinement control   : Hex20 6x6x2 uniform,
   - longitudinal refinement control : Hex20 4x4x4 bias=3 fixed-end.

The goal is to answer a concrete validation question before returning to larger
cyclic amplitudes: is the current local baseline already in a physically
reasonable regime, or is it still too coarse in the section and/or along the
column axis?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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

BLUE = "#0b5fa5"
ORANGE = "#d97706"
GREEN = "#2f855a"
PURPLE = "#7c3aed"


@dataclass(frozen=True)
class StructuralCase:
    key: str
    label: str
    output_subdir: str


@dataclass(frozen=True)
class ContinuumCase:
    key: str
    label: str
    output_subdir: str
    nx: int
    ny: int
    nz: int
    longitudinal_bias_power: float
    characteristic_length_mode: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the reduced RC continuum monotonic mesh-foundation audit."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--continuum-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
    )
    parser.add_argument(
        "--structural-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_reference_benchmark.exe",
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
    parser.add_argument("--monotonic-tip-mm", type=float, default=20.0)
    parser.add_argument("--monotonic-steps", type=int, default=12)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def scaled_optional(value: object, scale: float) -> float | None:
    parsed = safe_float(value)
    return scale * parsed if parsed is not None else None


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def run_or_reuse(
    command: list[str],
    bundle_dir: Path,
    manifest_name: str,
    reuse_existing: bool,
) -> tuple[float, dict[str, Any]]:
    ensure_dir(bundle_dir)
    manifest_path = bundle_dir / manifest_name
    if reuse_existing and manifest_path.exists():
        return math.nan, read_json(manifest_path)

    elapsed, proc = run_command(command, bundle_dir.parent)
    (bundle_dir / "runner_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (bundle_dir / "runner_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit {proc.returncode}:\n{' '.join(command)}\n\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if not manifest_path.exists():
        raise RuntimeError(f"Missing runtime manifest after successful run: {manifest_path}")
    return elapsed, read_json(manifest_path)


def structural_case() -> StructuralCase:
    return StructuralCase(
        key="structural_clamped_lobatto10_20mm",
        label="Structural clamped Lobatto N=10",
        output_subdir="structural_clamped_lobatto10_20mm",
    )


def continuum_cases() -> list[ContinuumCase]:
    return [
        ContinuumCase(
            key="hex20_4x4x2_uniform",
            label="Hex20 4x4x2 uniform",
            output_subdir="hex20_4x4x2_uniform_20mm",
            nx=4,
            ny=4,
            nz=2,
            longitudinal_bias_power=1.0,
            characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
        ),
        ContinuumCase(
            key="hex20_6x6x2_uniform",
            label="Hex20 6x6x2 uniform",
            output_subdir="hex20_6x6x2_uniform_20mm",
            nx=6,
            ny=6,
            nz=2,
            longitudinal_bias_power=1.0,
            characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
        ),
        ContinuumCase(
            key="hex20_4x4x4_bias3_fixedend",
            label="Hex20 4x4x4 bias=3 fixed-end",
            output_subdir="hex20_4x4x4_bias3_fixedend_20mm",
            nx=4,
            ny=4,
            nz=4,
            longitudinal_bias_power=3.0,
            characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
        ),
    ]


def structural_command(exe: Path, bundle_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        str(exe.resolve()),
        "--output-dir",
        str(bundle_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        "nonlinear",
        "--solver-policy",
        "newton-l2-only",
        "--beam-nodes",
        "10",
        "--beam-integration",
        "lobatto",
        "--clamp-top-bending-rotation",
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{args.monotonic_tip_mm}",
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--max-bisections",
        "8",
    ]


def continuum_command(
    exe: Path,
    bundle_dir: Path,
    case: ContinuumCase,
    args: argparse.Namespace,
) -> list[str]:
    return [
        str(exe.resolve()),
        "--output-dir",
        str(bundle_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        "nonlinear",
        "--hex-order",
        "hex20",
        "--reinforcement-mode",
        "embedded-longitudinal-bars",
        "--rebar-layout",
        "structural-matched-eight-bar",
        "--rebar-interpolation",
        "automatic",
        "--host-concrete-zoning-mode",
        "cover-core-split",
        "--transverse-mesh-mode",
        "cover-aligned",
        "--nx",
        str(case.nx),
        "--ny",
        str(case.ny),
        "--nz",
        str(case.nz),
        "--transverse-cover-subdivisions-x-each-side",
        "1",
        "--transverse-cover-subdivisions-y-each-side",
        "1",
        "--longitudinal-bias-power",
        f"{case.longitudinal_bias_power}",
        "--concrete-profile",
        "production-stabilized",
        "--concrete-tangent-mode",
        "fracture-secant",
        "--concrete-characteristic-length-mode",
        case.characteristic_length_mode,
        "--solver-policy",
        "newton-l2-only",
        "--predictor-policy",
        "hybrid-secant-linearized",
        "--continuation",
        "reversal-guarded",
        "--continuation-segment-substep-factor",
        "2",
        "--max-bisections",
        "8",
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{args.monotonic_tip_mm}",
        "--monotonic-steps",
        str(args.monotonic_steps),
    ]


def read_curve(bundle_dir: Path) -> list[tuple[float, float]]:
    rows = read_csv_rows(bundle_dir / "hysteresis.csv")
    curve: list[tuple[float, float]] = []
    for row in rows:
        drift_m = safe_float(row.get("drift_m"))
        base_shear_mn = safe_float(row.get("base_shear_MN"))
        if drift_m is None or base_shear_mn is None:
            continue
        curve.append((1.0e3 * drift_m, 1.0e3 * base_shear_mn))
    return sorted(curve, key=lambda item: item[0])


def interpolate_curve(curve: list[tuple[float, float]], x: float) -> float | None:
    if not curve:
        return None
    if x < curve[0][0] - 1.0e-12 or x > curve[-1][0] + 1.0e-12:
        return None
    for i in range(len(curve) - 1):
        x0, y0 = curve[i]
        x1, y1 = curve[i + 1]
        if abs(x - x0) <= 1.0e-12:
            return y0
        if x0 <= x <= x1:
            if abs(x1 - x0) <= 1.0e-12:
                return y0
            alpha = (x - x0) / (x1 - x0)
            return (1.0 - alpha) * y0 + alpha * y1
    return curve[-1][1]


def relative_curve_error(
    structural_curve: list[tuple[float, float]],
    continuum_curve: list[tuple[float, float]],
) -> tuple[float, float]:
    rel_errors: list[float] = []
    for drift_mm, base_shear_kn in continuum_curve:
        structural_value = interpolate_curve(structural_curve, drift_mm)
        if structural_value is None:
            continue
        scale = max(abs(structural_value), abs(base_shear_kn), 1.0)
        rel_errors.append(abs(base_shear_kn - structural_value) / scale)
    if not rel_errors:
        return math.nan, math.nan
    return max(rel_errors), math.sqrt(sum(value * value for value in rel_errors) / len(rel_errors))


def max_abs_from_rows(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [abs(v) for row in rows if (v := safe_float(row.get(key))) is not None]
    return max(values, default=None)


def first_crack_drift_mm(bundle_dir: Path) -> float | None:
    crack_rows = read_csv_rows(bundle_dir / "crack_state.csv")
    control_rows = read_csv_rows(bundle_dir / "control_state.csv")
    control_by_runtime_step = {
        int(safe_float(row.get("runtime_step"))): row
        for row in control_rows
        if safe_float(row.get("runtime_step")) is not None
    }
    for row in crack_rows:
        cracked = safe_float(row.get("cracked_gauss_point_count"))
        if cracked is not None and cracked > 0.0:
            runtime_step = int(safe_float(row.get("runtime_step")) or -1)
            control = control_by_runtime_step.get(runtime_step)
            drift_m = safe_float(control.get("actual_tip_lateral_displacement")) if control else None
            return 1.0e3 * drift_m if drift_m is not None else None
    return None


def figure_paths(stem: str, figures_dir: Path, secondary_dir: Path) -> list[Path]:
    return [
        figures_dir / f"{stem}.png",
        secondary_dir / f"{stem}.png",
    ]


def save(fig: plt.Figure, paths: list[Path]) -> None:
    for path in paths:
        ensure_dir(path.parent)
        fig.savefig(path)
    plt.close(fig)


def make_figures(
    structural_curve: list[tuple[float, float]],
    rows: list[dict[str, Any]],
    figures_dir: Path,
    secondary_dir: Path,
) -> None:
    label_colors = {
        "Structural clamped Lobatto N=10": BLUE,
        "Hex20 4x4x2 uniform": ORANGE,
        "Hex20 6x6x2 uniform": GREEN,
        "Hex20 4x4x4 bias=3 fixed-end": PURPLE,
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        [x for x, _ in structural_curve],
        [y for _, y in structural_curve],
        color=BLUE,
        linewidth=2.0,
        label="Structural clamped Lobatto N=10",
    )
    for row in rows:
        curve = read_curve(Path(row["output_dir"]))
        ax.plot(
            [x for x, _ in curve],
            [y for _, y in curve],
            linewidth=1.8,
            color=label_colors[row["label"]],
            label=row["label"],
        )
    ax.set_title("Promoted continuum monotonic mesh foundation")
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.legend(frameon=False)
    save(
        fig,
        figure_paths(
            "reduced_rc_continuum_monotonic_mesh_foundation_overlay",
            figures_dir,
            secondary_dir,
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax0, ax1 = axes
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    ax0.bar(
        x,
        [float(row["process_wall_seconds"]) for row in rows],
        color=[label_colors[label] for label in labels],
    )
    ax0.set_xticks(x, labels, rotation=18, ha="right")
    ax0.set_ylabel("Process wall time [s]")
    ax0.set_title("Monotonic mesh cost")

    ax1.bar(
        x,
        [float(row["rms_rel_vs_structural"]) for row in rows],
        color=[label_colors[label] for label in labels],
    )
    ax1.set_xticks(x, labels, rotation=18, ha="right")
    ax1.set_ylabel("RMS relative base-shear error")
    ax1.set_title("Monotonic mesh error vs structural control")
    save(
        fig,
        figure_paths(
            "reduced_rc_continuum_monotonic_mesh_foundation_overview",
            figures_dir,
            secondary_dir,
        ),
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    structural = structural_case()
    structural_dir = output_dir / structural.output_subdir
    _, structural_manifest = run_or_reuse(
        structural_command(args.structural_exe, structural_dir, args),
        structural_dir,
        "runtime_manifest.json",
        args.reuse_existing,
    )
    structural_curve = read_curve(structural_dir)

    rows: list[dict[str, Any]] = []
    cases_summary: dict[str, Any] = {}
    for case in continuum_cases():
        case_dir = output_dir / case.output_subdir
        elapsed, manifest = run_or_reuse(
            continuum_command(args.continuum_exe, case_dir, case, args),
            case_dir,
            "runtime_manifest.json",
            args.reuse_existing,
        )
        curve = read_curve(case_dir)
        max_rel, rms_rel = relative_curve_error(structural_curve, curve)
        row = {
            "key": case.key,
            "label": case.label,
            "mesh": f"{case.nx}x{case.ny}x{case.nz}",
            "longitudinal_bias_power": case.longitudinal_bias_power,
            "characteristic_length_mode": case.characteristic_length_mode,
            "process_wall_seconds": elapsed if math.isfinite(elapsed) else safe_float(manifest.get("timing", {}).get("total_wall_seconds")),
            "reported_total_wall_seconds": safe_float(manifest.get("timing", {}).get("total_wall_seconds")),
            "reported_solve_wall_seconds": safe_float(manifest.get("timing", {}).get("solve_wall_seconds")),
            "peak_base_shear_kn": scaled_optional(manifest.get("observables", {}).get("max_abs_base_shear_mn"), 1.0e3),
            "peak_cracked_gauss_points": safe_float(manifest.get("observables", {}).get("peak_cracked_gauss_points")),
            "max_crack_opening_mm": scaled_optional(manifest.get("observables", {}).get("max_crack_opening"), 1.0e3),
            "peak_rebar_stress_mpa": safe_float(manifest.get("observables", {}).get("max_abs_rebar_stress_mpa")),
            "peak_rebar_strain": safe_float(manifest.get("observables", {}).get("max_abs_rebar_strain")),
            "max_embedding_gap_mm": scaled_optional(manifest.get("observables", {}).get("max_embedding_gap_norm_m"), 1.0e3),
            "first_crack_drift_mm": first_crack_drift_mm(case_dir),
            "max_rel_vs_structural": max_rel,
            "rms_rel_vs_structural": rms_rel,
            "output_dir": str(case_dir),
        }
        rows.append(row)
        cases_summary[case.key] = row

    write_csv(output_dir / "continuum_monotonic_mesh_foundation_cases.csv", rows)
    summary = {
        "structural_reference": {
            "key": structural.key,
            "label": structural.label,
            "output_dir": str(structural_dir),
            "reported_total_wall_seconds": safe_float(structural_manifest.get("timing", {}).get("total_wall_seconds")),
            "reported_solve_wall_seconds": safe_float(structural_manifest.get("timing", {}).get("analysis_wall_seconds")),
        },
        "continuum_cases": cases_summary,
        "artifacts": {
            "cases_csv": str(output_dir / "continuum_monotonic_mesh_foundation_cases.csv"),
        },
    }
    write_json(output_dir / "continuum_monotonic_mesh_foundation_summary.json", summary)

    make_figures(
        structural_curve,
        rows,
        args.figures_dir.resolve(),
        args.secondary_figures_dir.resolve(),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
