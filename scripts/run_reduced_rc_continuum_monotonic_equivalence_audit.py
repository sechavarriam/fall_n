#!/usr/bin/env python3
"""
Freeze a monotonic continuum-equivalence audit for the reduced RC column.

This audit asks a more basic question than the cyclic structural-vs-continuum
bridge:

1. How much of the continuum/beam gap is explained by benchmark kinematics,
   i.e. by comparing a continuum with the full top face translated against a
   beam slice whose tip rotation is still free?
2. How much changes when the continuum is refined primarily along the column
   axis and clustered toward the fixed base?
3. How much of the remaining gap belongs to the embedded-bar formulation rather
   than the concrete host alone?

The output is intentionally monotonic and precompressed before any direction
changes are revisited.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
RED = "#b91c1c"
PURPLE = "#7c3aed"
BROWN = "#8b5e3c"


@dataclass(frozen=True)
class StructuralCase:
    key: str
    material_mode: str
    monotonic_tip_mm: float
    monotonic_steps: int
    clamp_top_bending_rotation: bool
    timeout_seconds: int = 1200


@dataclass(frozen=True)
class ContinuumCase:
    key: str
    material_mode: str
    reinforcement_mode: str
    nx: int
    ny: int
    nz: int
    longitudinal_bias_power: float
    monotonic_tip_mm: float
    monotonic_steps: int
    concrete_profile: str = "benchmark-reference"
    solver_policy: str = "newton-l2-only"
    hex_order: str = "hex20"
    timeout_seconds: int = 2400


@dataclass(frozen=True)
class StructuralRow:
    key: str
    material_mode: str
    clamp_top_bending_rotation: bool
    monotonic_tip_mm: float
    status: str
    completed_successfully: bool
    timed_out: bool
    process_wall_seconds: float
    reported_total_wall_seconds: float | None
    peak_base_shear_kn: float | None
    output_dir: str


@dataclass(frozen=True)
class ContinuumRow:
    key: str
    material_mode: str
    reinforcement_mode: str
    mesh: str
    longitudinal_bias_power: float
    monotonic_tip_mm: float
    status: str
    completed_successfully: bool
    timed_out: bool
    process_wall_seconds: float
    reported_total_wall_seconds: float | None
    peak_base_shear_kn: float | None
    max_embedding_gap_m: float | None
    peak_cracked_gauss_points: int | None
    first_crack_drift_mm: float | None
    max_rel_error_vs_structural_free: float | None
    rms_rel_error_vs_structural_free: float | None
    max_rel_error_vs_structural_clamped: float | None
    rms_rel_error_vs_structural_clamped: float | None
    output_dir: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the reduced RC continuum monotonic equivalence audit."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--structural-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument(
        "--continuum-exe",
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
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--beam-nodes", type=int, default=10)
    parser.add_argument("--beam-integration", default="lobatto")
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--print-progress", action="store_true")
    parser.add_argument(
        "--case-filter",
        action="append",
        default=[],
        help="Only run cases whose key contains one of these substrings.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def case_selected(key: str, filters: list[str]) -> bool:
    if not filters:
        return True
    key_lower = key.lower()
    return any(token.lower() in key_lower for token in filters)


def terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        proc.kill()


def run_command(
    command: list[str], cwd: Path, timeout_seconds: int
) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        terminate_process_tree(proc)
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout_seconds,
            output=(exc.output or "") + (stdout or ""),
            stderr=(exc.stderr or "") + (stderr or ""),
        ) from exc
    completed = subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    return time.perf_counter() - start, completed


def structural_command(
    exe: Path,
    output_dir: Path,
    args: argparse.Namespace,
    case: StructuralCase,
) -> list[str]:
    command = [
        str(exe),
        "--output-dir",
        str(output_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        case.material_mode,
        "--beam-nodes",
        str(args.beam_nodes),
        "--beam-integration",
        args.beam_integration,
        "--solver-policy",
        "newton-l2-only",
        "--continuation",
        "monolithic",
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{case.monotonic_tip_mm}",
        "--monotonic-steps",
        str(case.monotonic_steps),
        "--max-bisections",
        "8",
    ]
    if case.clamp_top_bending_rotation:
        command.append("--clamp-top-bending-rotation")
    if args.print_progress:
        command.append("--print-progress")
    return command


def continuum_command(
    exe: Path,
    output_dir: Path,
    args: argparse.Namespace,
    case: ContinuumCase,
) -> list[str]:
    command = [
        str(exe),
        "--output-dir",
        str(output_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        case.material_mode,
        "--concrete-profile",
        case.concrete_profile,
        "--reinforcement-mode",
        case.reinforcement_mode,
        "--hex-order",
        case.hex_order,
        "--nx",
        str(case.nx),
        "--ny",
        str(case.ny),
        "--nz",
        str(case.nz),
        "--longitudinal-bias-power",
        f"{case.longitudinal_bias_power}",
        "--solver-policy",
        case.solver_policy,
        "--continuation",
        "monolithic",
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{case.monotonic_tip_mm}",
        "--monotonic-steps",
        str(case.monotonic_steps),
        "--max-bisections",
        "8",
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def structural_cases() -> list[StructuralCase]:
    return [
        StructuralCase(
            key="structural_elastic_free_rotation_0p5mm",
            material_mode="elasticized",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            clamp_top_bending_rotation=False,
            timeout_seconds=600,
        ),
        StructuralCase(
            key="structural_elastic_clamped_rotation_0p5mm",
            material_mode="elasticized",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            clamp_top_bending_rotation=True,
            timeout_seconds=600,
        ),
        StructuralCase(
            key="structural_nonlinear_free_rotation_20mm",
            material_mode="nonlinear",
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            clamp_top_bending_rotation=False,
            timeout_seconds=1800,
        ),
        StructuralCase(
            key="structural_nonlinear_clamped_rotation_20mm",
            material_mode="nonlinear",
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            clamp_top_bending_rotation=True,
            timeout_seconds=1800,
        ),
    ]


def continuum_cases() -> list[ContinuumCase]:
    return [
        ContinuumCase(
            key="continuum_embedded_elastic_hex20_2x2x2_uniform_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            nx=2,
            ny=2,
            nz=2,
            longitudinal_bias_power=1.0,
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=900,
        ),
        ContinuumCase(
            key="continuum_embedded_elastic_hex20_2x2x10_uniform_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            nx=2,
            ny=2,
            nz=10,
            longitudinal_bias_power=1.0,
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=1200,
        ),
        ContinuumCase(
            key="continuum_embedded_elastic_hex20_2x2x10_bias3_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            nx=2,
            ny=2,
            nz=10,
            longitudinal_bias_power=3.0,
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=1200,
        ),
        ContinuumCase(
            key="continuum_embedded_nonlinear_hex20_2x2x2_uniform_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            nx=2,
            ny=2,
            nz=2,
            longitudinal_bias_power=1.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=1800,
        ),
        ContinuumCase(
            key="continuum_host_nonlinear_hex20_2x2x2_uniform_20mm",
            material_mode="nonlinear",
            reinforcement_mode="continuum-only",
            nx=2,
            ny=2,
            nz=2,
            longitudinal_bias_power=1.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=1800,
        ),
        ContinuumCase(
            key="continuum_embedded_nonlinear_hex20_2x2x10_uniform_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            nx=2,
            ny=2,
            nz=10,
            longitudinal_bias_power=1.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=2400,
        ),
        ContinuumCase(
            key="continuum_host_nonlinear_hex20_2x2x10_uniform_20mm",
            material_mode="nonlinear",
            reinforcement_mode="continuum-only",
            nx=2,
            ny=2,
            nz=10,
            longitudinal_bias_power=1.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=2400,
        ),
        ContinuumCase(
            key="continuum_embedded_nonlinear_hex20_2x2x10_bias3_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            nx=2,
            ny=2,
            nz=10,
            longitudinal_bias_power=3.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=2400,
        ),
        ContinuumCase(
            key="continuum_host_nonlinear_hex20_2x2x10_bias3_20mm",
            material_mode="nonlinear",
            reinforcement_mode="continuum-only",
            nx=2,
            ny=2,
            nz=10,
            longitudinal_bias_power=3.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=2400,
        ),
    ]


def read_monotonic_curve(bundle_dir: Path) -> list[tuple[float, float]]:
    rows = read_csv_rows(bundle_dir / "hysteresis.csv")
    curve: list[tuple[float, float]] = []
    for row in rows:
        drift = safe_float(row.get("drift_m"))
        base_shear = safe_float(row.get("base_shear_MN"))
        if drift is None or base_shear is None:
            continue
        curve.append((1.0e3 * drift, 1.0e3 * base_shear))
    curve.sort(key=lambda item: item[0])
    return curve


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
    return curve[-1][1] if abs(x - curve[-1][0]) <= 1.0e-12 else None


def curve_error_metrics(
    candidate: list[tuple[float, float]],
    reference: list[tuple[float, float]],
) -> tuple[float | None, float | None]:
    if not candidate or not reference:
        return (None, None)
    ref_scale = max(abs(y) for _, y in reference)
    if ref_scale <= 1.0e-12:
        ref_scale = 1.0
    rel_errors: list[float] = []
    for x, y in candidate:
        y_ref = interpolate_curve(reference, x)
        if y_ref is None:
            continue
        rel_errors.append(abs(y - y_ref) / ref_scale)
    if not rel_errors:
        return (None, None)
    rms = math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors))
    return (max(rel_errors), rms)


def structural_row_from_bundle(
    case: StructuralCase,
    bundle_dir: Path,
    elapsed: float | None,
    *,
    status: str = "completed",
    timed_out: bool = False,
) -> StructuralRow:
    manifest = read_json(bundle_dir / "runtime_manifest.json")
    observables = manifest.get("observables", {})
    timing = manifest.get("timing", {})
    curve = read_monotonic_curve(bundle_dir)
    peak_base_shear_kn = (
        max((abs(y) for _, y in curve), default=None)
        if curve
        else None
    )
    manifest_peak = safe_float(observables.get("max_abs_base_shear_mn"))
    if manifest_peak is not None:
        peak_base_shear_kn = 1.0e3 * manifest_peak
    return StructuralRow(
        key=case.key,
        material_mode=case.material_mode,
        clamp_top_bending_rotation=case.clamp_top_bending_rotation,
        monotonic_tip_mm=case.monotonic_tip_mm,
        status=status,
        completed_successfully=bool(
            manifest.get("completed_successfully", manifest.get("status") == "completed")
        ),
        timed_out=timed_out,
        process_wall_seconds=elapsed if elapsed is not None else float(timing.get("total_wall_seconds", math.nan)),
        reported_total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
        peak_base_shear_kn=peak_base_shear_kn,
        output_dir=str(bundle_dir),
    )


def continuum_row_from_bundle(
    case: ContinuumCase,
    bundle_dir: Path,
    elapsed: float | None,
    structural_free_curve: list[tuple[float, float]],
    structural_clamped_curve: list[tuple[float, float]],
    *,
    status: str = "completed",
    timed_out: bool = False,
) -> ContinuumRow:
    manifest = read_json(bundle_dir / "runtime_manifest.json")
    timing = manifest.get("timing", {})
    observables = manifest.get("observables", {})
    curve = read_monotonic_curve(bundle_dir)
    max_free, rms_free = curve_error_metrics(curve, structural_free_curve)
    max_clamped, rms_clamped = curve_error_metrics(curve, structural_clamped_curve)
    control_rows = read_csv_rows(bundle_dir / "control_state.csv") if (bundle_dir / "control_state.csv").exists() else []
    first_crack_runtime_step = int(observables.get("first_crack_runtime_step", -1) or -1)
    first_crack_drift_mm: float | None = None
    if first_crack_runtime_step >= 0:
        for row in control_rows:
            runtime_step = safe_float(row.get("runtime_step"))
            target_drift = safe_float(row.get("target_drift_m"))
            if runtime_step is not None and int(runtime_step) == first_crack_runtime_step and target_drift is not None:
                first_crack_drift_mm = 1.0e3 * target_drift
                break
    return ContinuumRow(
        key=case.key,
        material_mode=case.material_mode,
        reinforcement_mode=case.reinforcement_mode,
        mesh=f"{case.nx}x{case.ny}x{case.nz}",
        longitudinal_bias_power=case.longitudinal_bias_power,
        monotonic_tip_mm=case.monotonic_tip_mm,
        status=status,
        completed_successfully=bool(manifest.get("completed_successfully")),
        timed_out=timed_out,
        process_wall_seconds=elapsed if elapsed is not None else float(timing.get("total_wall_seconds", math.nan)),
        reported_total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
        peak_base_shear_kn=(1.0e3 * safe_float(observables.get("max_abs_base_shear_mn"))) if safe_float(observables.get("max_abs_base_shear_mn")) is not None else None,
        max_embedding_gap_m=safe_float(observables.get("max_abs_top_rebar_face_gap_m")),
        peak_cracked_gauss_points=int(observables.get("peak_cracked_gauss_points", 0) or 0),
        first_crack_drift_mm=first_crack_drift_mm,
        max_rel_error_vs_structural_free=max_free,
        rms_rel_error_vs_structural_free=rms_free,
        max_rel_error_vs_structural_clamped=max_clamped,
        rms_rel_error_vs_structural_clamped=rms_clamped,
        output_dir=str(bundle_dir),
    )


def structural_failure_row(
    case: StructuralCase,
    bundle_dir: Path,
    *,
    status: str,
    timed_out: bool,
    elapsed: float,
) -> StructuralRow:
    return StructuralRow(
        key=case.key,
        material_mode=case.material_mode,
        clamp_top_bending_rotation=case.clamp_top_bending_rotation,
        monotonic_tip_mm=case.monotonic_tip_mm,
        status=status,
        completed_successfully=False,
        timed_out=timed_out,
        process_wall_seconds=elapsed,
        reported_total_wall_seconds=None,
        peak_base_shear_kn=None,
        output_dir=str(bundle_dir),
    )


def continuum_failure_row(
    case: ContinuumCase,
    bundle_dir: Path,
    *,
    status: str,
    timed_out: bool,
    elapsed: float,
) -> ContinuumRow:
    return ContinuumRow(
        key=case.key,
        material_mode=case.material_mode,
        reinforcement_mode=case.reinforcement_mode,
        mesh=f"{case.nx}x{case.ny}x{case.nz}",
        longitudinal_bias_power=case.longitudinal_bias_power,
        monotonic_tip_mm=case.monotonic_tip_mm,
        status=status,
        completed_successfully=False,
        timed_out=timed_out,
        process_wall_seconds=elapsed,
        reported_total_wall_seconds=None,
        peak_base_shear_kn=None,
        max_embedding_gap_m=None,
        peak_cracked_gauss_points=None,
        first_crack_drift_mm=None,
        max_rel_error_vs_structural_free=None,
        rms_rel_error_vs_structural_free=None,
        max_rel_error_vs_structural_clamped=None,
        rms_rel_error_vs_structural_clamped=None,
        output_dir=str(bundle_dir),
    )


def run_structural_case(
    exe: Path,
    root: Path,
    repo_root: Path,
    args: argparse.Namespace,
    case: StructuralCase,
) -> StructuralRow:
    bundle_dir = root / case.key
    ensure_dir(bundle_dir)
    manifest = bundle_dir / "runtime_manifest.json"
    if args.reuse_existing and manifest.exists():
        return structural_row_from_bundle(case, bundle_dir, None, status="reused_cached_bundle")
    try:
        elapsed, completed = run_command(
            structural_command(exe, bundle_dir, args, case),
            cwd=repo_root,
            timeout_seconds=case.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        (bundle_dir / "runner_stdout.log").write_text(
            (exc.output or ""), encoding="utf-8"
        )
        (bundle_dir / "runner_stderr.log").write_text(
            (exc.stderr or ""), encoding="utf-8"
        )
        if manifest.exists():
            return structural_row_from_bundle(
                case,
                bundle_dir,
                None,
                status="completed_beyond_timeout_budget",
                timed_out=True,
            )
        return structural_failure_row(
            case,
            bundle_dir,
            status="timeout_budget_exceeded",
            timed_out=True,
            elapsed=float(case.timeout_seconds),
        )
    (bundle_dir / "runner_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (bundle_dir / "runner_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        if manifest.exists():
            return structural_row_from_bundle(
                case,
                bundle_dir,
                elapsed,
                status="completed_with_runner_warning",
            )
        return structural_failure_row(
            case,
            bundle_dir,
            status="runner_failed",
            timed_out=False,
            elapsed=elapsed,
        )
    if not manifest.exists():
        return structural_failure_row(
            case,
            bundle_dir,
            status="missing_runtime_manifest",
            timed_out=False,
            elapsed=elapsed,
        )
    return structural_row_from_bundle(case, bundle_dir, elapsed)


def run_continuum_case(
    exe: Path,
    root: Path,
    repo_root: Path,
    args: argparse.Namespace,
    case: ContinuumCase,
    structural_free_curve: list[tuple[float, float]],
    structural_clamped_curve: list[tuple[float, float]],
) -> ContinuumRow:
    bundle_dir = root / case.key
    ensure_dir(bundle_dir)
    manifest = bundle_dir / "runtime_manifest.json"
    if args.reuse_existing and manifest.exists():
        return continuum_row_from_bundle(
            case,
            bundle_dir,
            None,
            structural_free_curve,
            structural_clamped_curve,
            status="reused_cached_bundle",
        )
    try:
        elapsed, completed = run_command(
            continuum_command(exe, bundle_dir, args, case),
            cwd=repo_root,
            timeout_seconds=case.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        (bundle_dir / "runner_stdout.log").write_text(
            (exc.output or ""), encoding="utf-8"
        )
        (bundle_dir / "runner_stderr.log").write_text(
            (exc.stderr or ""), encoding="utf-8"
        )
        if manifest.exists():
            return continuum_row_from_bundle(
                case,
                bundle_dir,
                None,
                structural_free_curve,
                structural_clamped_curve,
                status="completed_beyond_timeout_budget",
                timed_out=True,
            )
        return continuum_failure_row(
            case,
            bundle_dir,
            status="timeout_budget_exceeded",
            timed_out=True,
            elapsed=float(case.timeout_seconds),
        )
    (bundle_dir / "runner_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (bundle_dir / "runner_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        if manifest.exists():
            return continuum_row_from_bundle(
                case,
                bundle_dir,
                elapsed,
                structural_free_curve,
                structural_clamped_curve,
                status="completed_with_runner_warning",
            )
        return continuum_failure_row(
            case,
            bundle_dir,
            status="runner_failed",
            timed_out=False,
            elapsed=elapsed,
        )
    if not manifest.exists():
        return continuum_failure_row(
            case,
            bundle_dir,
            status="missing_runtime_manifest",
            timed_out=False,
            elapsed=elapsed,
        )
    return continuum_row_from_bundle(
        case,
        bundle_dir,
        elapsed,
        structural_free_curve,
        structural_clamped_curve,
    )


def plot_base_shear_overlay(
    structural_rows: list[StructuralRow],
    continuum_rows: list[ContinuumRow],
    primary: Path,
    secondary: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))

    structural_styles = {
        "structural_elastic_free_rotation_0p5mm": ("Structural elastic free rot.", BLUE, "-"),
        "structural_elastic_clamped_rotation_0p5mm": ("Structural elastic clamped rot.", ORANGE, "--"),
        "structural_nonlinear_free_rotation_20mm": ("Structural nonlinear free rot.", GREEN, "-"),
        "structural_nonlinear_clamped_rotation_20mm": ("Structural nonlinear clamped rot.", RED, "--"),
    }
    for row in structural_rows:
        if not row.completed_successfully:
            continue
        curve = read_monotonic_curve(Path(row.output_dir))
        label, color, linestyle = structural_styles[row.key]
        ax.plot(
            [x for x, _ in curve],
            [y for _, y in curve],
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            label=label,
        )

    continuum_styles = {
        "continuum_embedded_elastic_hex20_2x2x2_uniform_0p5mm": ("Hex20 emb. 2x2x2 elastic", PURPLE),
        "continuum_embedded_elastic_hex20_2x2x10_uniform_0p5mm": ("Hex20 emb. 2x2x10 elastic", BROWN),
        "continuum_embedded_elastic_hex20_2x2x10_bias3_0p5mm": ("Hex20 emb. 2x2x10 bias=3 elastic", "#6b7280"),
        "continuum_embedded_nonlinear_hex20_2x2x2_uniform_20mm": ("Hex20 emb. 2x2x2 nonlinear", PURPLE),
        "continuum_embedded_nonlinear_hex20_2x2x10_uniform_20mm": ("Hex20 emb. 2x2x10 nonlinear", BROWN),
        "continuum_embedded_nonlinear_hex20_2x2x10_bias3_20mm": ("Hex20 emb. 2x2x10 bias=3 nonlinear", "#6b7280"),
        "continuum_host_nonlinear_hex20_2x2x2_uniform_20mm": ("Hex20 host 2x2x2 nonlinear", "#8b5cf6"),
        "continuum_host_nonlinear_hex20_2x2x10_uniform_20mm": ("Hex20 host 2x2x10 nonlinear", "#a16207"),
        "continuum_host_nonlinear_hex20_2x2x10_bias3_20mm": ("Hex20 host 2x2x10 bias=3 nonlinear", "#4b5563"),
    }
    for row in continuum_rows:
        if not row.completed_successfully:
            continue
        curve = read_monotonic_curve(Path(row.output_dir))
        label, color = continuum_styles[row.key]
        ax.plot(
            [x for x, _ in curve],
            [y for _, y in curve],
            color=color,
            linewidth=1.8,
            alpha=0.95,
            label=label,
        )

    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Reduced RC monotonic structural-vs-continuum equivalence audit")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(primary)
    fig.savefig(secondary)
    plt.close(fig)


def plot_continuum_error_bars(
    continuum_rows: list[ContinuumRow],
    primary: Path,
    secondary: Path,
) -> None:
    labels = [row.key.replace("continuum_", "").replace("_", "\n") for row in continuum_rows]
    free_vals = [row.rms_rel_error_vs_structural_free or math.nan for row in continuum_rows]
    clamped_vals = [row.rms_rel_error_vs_structural_clamped or math.nan for row in continuum_rows]
    x = list(range(len(continuum_rows)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar([i - width / 2 for i in x], free_vals, width=width, color=BLUE, label="vs structural free")
    ax.bar([i + width / 2 for i in x], clamped_vals, width=width, color=ORANGE, label="vs structural clamped")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("RMS relative base-shear path error")
    ax.set_title("Continuum monotonic gap vs structural free/clamped references")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(primary)
    fig.savefig(secondary)
    plt.close(fig)


def plot_timing(
    structural_rows: list[StructuralRow],
    continuum_rows: list[ContinuumRow],
    primary: Path,
    secondary: Path,
) -> None:
    structural_label_map = {
        "structural_elastic_free_rotation_0p5mm": ("beam free el.", BLUE),
        "structural_elastic_clamped_rotation_0p5mm": ("beam clamp el.", ORANGE),
        "structural_nonlinear_free_rotation_20mm": ("beam free nl.", GREEN),
        "structural_nonlinear_clamped_rotation_20mm": ("beam clamp nl.", RED),
    }
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for row in structural_rows:
        label, color = structural_label_map.get(
            row.key, (row.key.replace("structural_", "").replace("_", "\n"), BLUE)
        )
        labels.append(label)
        values.append(row.reported_total_wall_seconds or math.nan)
        colors.append(color)
    for row in continuum_rows:
        labels.append(row.key.replace("continuum_", "").replace("_", "\n"))
        values.append(row.reported_total_wall_seconds or math.nan)
        colors.append(PURPLE if row.completed_successfully else "#9ca3af")

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Reported total wall time [s]")
    ax.set_title("Monotonic equivalence audit timing")
    fig.tight_layout()
    fig.savefig(primary)
    fig.savefig(secondary)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)
    ensure_dir(args.secondary_figures_dir)

    structural_results: list[StructuralRow] = []
    for case in [case for case in structural_cases() if case_selected(case.key, args.case_filter)]:
        structural_results.append(
            run_structural_case(
                args.structural_exe, args.output_dir, repo_root, args, case
            )
        )

    structural_by_key = {
        row.key: row for row in structural_results if row.completed_successfully
    }
    if "structural_nonlinear_free_rotation_20mm" not in structural_by_key or \
       "structural_nonlinear_clamped_rotation_20mm" not in structural_by_key:
        raise RuntimeError(
            "The monotonic equivalence audit requires completed structural nonlinear "
            "free/clamped references before continuum errors can be evaluated."
        )
    structural_free_curve = read_monotonic_curve(Path(structural_by_key["structural_nonlinear_free_rotation_20mm"].output_dir))
    structural_clamped_curve = read_monotonic_curve(Path(structural_by_key["structural_nonlinear_clamped_rotation_20mm"].output_dir))

    continuum_results: list[ContinuumRow] = []
    for case in [case for case in continuum_cases() if case_selected(case.key, args.case_filter)]:
        continuum_results.append(
            run_continuum_case(
                args.continuum_exe,
                args.output_dir,
                repo_root,
                args,
                case,
                structural_free_curve,
                structural_clamped_curve,
            )
        )

    summary_status = (
        "completed"
        if all(row.completed_successfully for row in structural_results + continuum_results)
        else "completed_with_partial_failures"
    )
    summary = {
        "status": summary_status,
        "structural_rows": [asdict(row) for row in structural_results],
        "continuum_rows": [asdict(row) for row in continuum_results],
        "key_findings": {
            "structural_clamped_rotation_note": (
                "The clamped-rotation beam slice is used only as a kinematic "
                "equivalence control against the continuum top-face displacement "
                "boundary condition; it does not replace the free-rotation beam "
                "reference for the reduced-column structural family."
            ),
            "continuum_focus_note": (
                "The continuum bridge is audited monotonically before any direction "
                "changes, with explicit longitudinal refinement and base-biased "
                "z-level clustering."
            ),
        },
    }
    write_json(args.output_dir / "continuum_monotonic_equivalence_summary.json", summary)
    write_csv(
        args.output_dir / "continuum_monotonic_equivalence_rows.csv",
        [asdict(row) for row in structural_results + continuum_results],
    )

    plot_base_shear_overlay(
        structural_results,
        continuum_results,
        args.figures_dir / "reduced_rc_continuum_monotonic_equivalence_overlay.png",
        args.secondary_figures_dir / "reduced_rc_continuum_monotonic_equivalence_overlay.png",
    )
    plot_continuum_error_bars(
        continuum_results,
        args.figures_dir / "reduced_rc_continuum_monotonic_equivalence_error.png",
        args.secondary_figures_dir / "reduced_rc_continuum_monotonic_equivalence_error.png",
    )
    plot_timing(
        structural_results,
        continuum_results,
        args.figures_dir / "reduced_rc_continuum_monotonic_equivalence_timing.png",
        args.secondary_figures_dir / "reduced_rc_continuum_monotonic_equivalence_timing.png",
    )
    write_csv(
        args.output_dir / "continuum_monotonic_equivalence_structural_rows.csv",
        [asdict(row) for row in structural_results],
    )
    write_csv(
        args.output_dir / "continuum_monotonic_equivalence_continuum_rows.csv",
        [asdict(row) for row in continuum_results],
    )


if __name__ == "__main__":
    main()
