#!/usr/bin/env python3
"""
Reduced RC column monotonic continuum amplitude audit.

This pass pushes the continuum pilot in amplitude while holding the structural
comparison on the clamped-rotation beam control that was already identified as
the right kinematic benchmark for the monotonic solid-vs-beam bridge.

The audit is designed to be operationally robust:
  - each case is independent;
  - timeouts are recorded as case status instead of aborting the whole bundle;
  - the summary is written even when some high-cost cases remain open.
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
GRAY = "#6b7280"


@dataclass(frozen=True)
class StructuralAmplitudeCase:
    key: str
    monotonic_tip_mm: float
    monotonic_steps: int
    timeout_seconds: int


@dataclass(frozen=True)
class ContinuumAmplitudeCase:
    key: str
    family_key: str
    family_label: str
    reinforcement_mode: str
    nx: int
    ny: int
    nz: int
    longitudinal_bias_power: float
    monotonic_tip_mm: float
    monotonic_steps: int
    concrete_profile: str
    concrete_tangent_mode: str
    concrete_characteristic_length_mode: str
    solver_policy: str
    timeout_seconds: int


@dataclass(frozen=True)
class StructuralAmplitudeRow:
    key: str
    monotonic_tip_mm: float
    status: str
    completed_successfully: bool
    timed_out: bool
    process_wall_seconds: float
    reported_total_wall_seconds: float | None
    peak_base_shear_kn: float | None
    output_dir: str


@dataclass(frozen=True)
class ContinuumAmplitudeRow:
    key: str
    family_key: str
    family_label: str
    reinforcement_mode: str
    mesh: str
    longitudinal_bias_power: float
    monotonic_tip_mm: float
    concrete_profile: str
    concrete_tangent_mode: str
    concrete_characteristic_length_mode: str
    status: str
    completed_successfully: bool
    timed_out: bool
    process_wall_seconds: float
    reported_total_wall_seconds: float | None
    peak_base_shear_kn: float | None
    max_rel_error_vs_structural_clamped: float | None
    rms_rel_error_vs_structural_clamped: float | None
    max_embedding_gap_m: float | None
    peak_cracked_gauss_points: int | None
    first_crack_drift_mm: float | None
    output_dir: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the reduced RC continuum monotonic amplitude audit."
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
        "--amplitudes-mm",
        default="2.5,5,10,15,20",
        help="Comma-separated monotonic amplitudes in mm.",
    )
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


def safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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
    return time.perf_counter() - start, subprocess.CompletedProcess(
        command, proc.returncode, stdout, stderr
    )


def parse_amplitudes_mm(raw: str) -> list[float]:
    amps: list[float] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        amps.append(float(stripped))
    if not amps:
        raise ValueError("At least one amplitude is required.")
    return amps


def monotonic_steps_for_amplitude(amplitude_mm: float) -> int:
    if amplitude_mm <= 5.0:
        return 6
    if amplitude_mm <= 10.0:
        return 8
    if amplitude_mm <= 15.0:
        return 10
    return 12


def structural_timeout_for_amplitude(amplitude_mm: float) -> int:
    if amplitude_mm <= 10.0:
        return 900
    if amplitude_mm <= 20.0:
        return 1800
    return 2400


def continuum_timeout_for_case(amplitude_mm: float, nz: int) -> int:
    base = 1800 if amplitude_mm <= 10.0 else 2400
    if nz >= 10:
        base += 1200
    return base


def structural_cases(amplitudes_mm: list[float]) -> list[StructuralAmplitudeCase]:
    return [
        StructuralAmplitudeCase(
            key=f"structural_clamped_{str(amplitude).replace('.', 'p')}mm",
            monotonic_tip_mm=amplitude,
            monotonic_steps=monotonic_steps_for_amplitude(amplitude),
            timeout_seconds=structural_timeout_for_amplitude(amplitude),
        )
        for amplitude in amplitudes_mm
    ]


def continuum_cases(amplitudes_mm: list[float]) -> list[ContinuumAmplitudeCase]:
    cases: list[ContinuumAmplitudeCase] = []
    families = [
        (
            "embedded_uniform_control",
            "Hex20 emb. 2x2x2 uniforme",
            "embedded-longitudinal-bars",
            2,
            2,
            2,
            1.0,
            "production-stabilized",
            "fracture-secant",
            "mean-longitudinal-host-edge-mm",
        ),
        (
            "embedded_promoted_bias3_fixedend",
            "Hex20 emb. 2x2x2 sesgo=3, lb fijo",
            "embedded-longitudinal-bars",
            2,
            2,
            2,
            3.0,
            "production-stabilized",
            "fracture-secant",
            "fixed-end-longitudinal-host-edge-mm",
        ),
        (
            "host_promoted_bias3_fixedend",
            "Hex20 host 2x2x2 sesgo=3, lb fijo",
            "continuum-only",
            2,
            2,
            2,
            3.0,
            "production-stabilized",
            "fracture-secant",
            "fixed-end-longitudinal-host-edge-mm",
        ),
    ]
    for amplitude in amplitudes_mm:
        for (
            family_key,
            family_label,
            reinforcement_mode,
            nx,
            ny,
            nz,
            bias,
            concrete_profile,
            concrete_tangent_mode,
            concrete_characteristic_length_mode,
        ) in families:
            cases.append(
                ContinuumAmplitudeCase(
                    key=(
                        f"continuum_{family_key}_hex20_{nx}x{ny}x{nz}_"
                        f"{str(amplitude).replace('.', 'p')}mm"
                    ),
                    family_key=family_key,
                    family_label=family_label,
                    reinforcement_mode=reinforcement_mode,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    longitudinal_bias_power=bias,
                    monotonic_tip_mm=amplitude,
                    monotonic_steps=monotonic_steps_for_amplitude(amplitude),
                    concrete_profile=concrete_profile,
                    concrete_tangent_mode=concrete_tangent_mode,
                    concrete_characteristic_length_mode=concrete_characteristic_length_mode,
                    solver_policy="newton-l2-only",
                    timeout_seconds=continuum_timeout_for_case(amplitude, nz),
                )
            )
    return cases


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


def structural_command(
    exe: Path,
    output_dir: Path,
    args: argparse.Namespace,
    case: StructuralAmplitudeCase,
) -> list[str]:
    command = [
        str(exe),
        "--output-dir",
        str(output_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        "nonlinear",
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
        "--clamp-top-bending-rotation",
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def continuum_command(
    exe: Path,
    output_dir: Path,
    args: argparse.Namespace,
    case: ContinuumAmplitudeCase,
) -> list[str]:
    command = [
        str(exe),
        "--output-dir",
        str(output_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        "nonlinear",
        "--concrete-profile",
        case.concrete_profile,
        "--concrete-tangent-mode",
        case.concrete_tangent_mode,
        "--concrete-characteristic-length-mode",
        case.concrete_characteristic_length_mode,
        "--reinforcement-mode",
        case.reinforcement_mode,
        "--hex-order",
        "hex20",
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


def structural_row_from_bundle(
    case: StructuralAmplitudeCase,
    bundle_dir: Path,
    elapsed: float | None,
    *,
    status: str = "completed",
    timed_out: bool = False,
) -> StructuralAmplitudeRow:
    manifest = read_json(bundle_dir / "runtime_manifest.json")
    timing = manifest.get("timing", {})
    curve = read_monotonic_curve(bundle_dir)
    peak = max((abs(y) for _, y in curve), default=None) if curve else None
    return StructuralAmplitudeRow(
        key=case.key,
        monotonic_tip_mm=case.monotonic_tip_mm,
        status=status,
        completed_successfully=bool(
            manifest.get("completed_successfully", manifest.get("status") == "completed")
        ),
        timed_out=timed_out,
        process_wall_seconds=elapsed if elapsed is not None else float(timing.get("total_wall_seconds", math.nan)),
        reported_total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
        peak_base_shear_kn=peak,
        output_dir=str(bundle_dir),
    )


def continuum_row_from_bundle(
    case: ContinuumAmplitudeCase,
    bundle_dir: Path,
    elapsed: float | None,
    structural_curve: list[tuple[float, float]],
    *,
    status: str = "completed",
    timed_out: bool = False,
) -> ContinuumAmplitudeRow:
    manifest = read_json(bundle_dir / "runtime_manifest.json")
    timing = manifest.get("timing", {})
    observables = manifest.get("observables", {})
    curve = read_monotonic_curve(bundle_dir)
    max_err, rms_err = curve_error_metrics(curve, structural_curve)
    control_rows = (
        read_csv_rows(bundle_dir / "control_state.csv")
        if (bundle_dir / "control_state.csv").exists()
        else []
    )
    first_crack_runtime_step = int(observables.get("first_crack_runtime_step", -1) or -1)
    first_crack_drift_mm: float | None = None
    if first_crack_runtime_step >= 0:
        for row in control_rows:
            runtime_step = safe_float(row.get("runtime_step"))
            target_drift = safe_float(row.get("target_drift_m"))
            if runtime_step is not None and int(runtime_step) == first_crack_runtime_step and target_drift is not None:
                first_crack_drift_mm = 1.0e3 * target_drift
                break

    return ContinuumAmplitudeRow(
        key=case.key,
        family_key=case.family_key,
        family_label=case.family_label,
        reinforcement_mode=case.reinforcement_mode,
        mesh=f"{case.nx}x{case.ny}x{case.nz}",
        longitudinal_bias_power=case.longitudinal_bias_power,
        monotonic_tip_mm=case.monotonic_tip_mm,
        concrete_profile=case.concrete_profile,
        concrete_tangent_mode=case.concrete_tangent_mode,
        concrete_characteristic_length_mode=case.concrete_characteristic_length_mode,
        status=status,
        completed_successfully=bool(manifest.get("completed_successfully")),
        timed_out=timed_out,
        process_wall_seconds=elapsed if elapsed is not None else float(timing.get("total_wall_seconds", math.nan)),
        reported_total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
        peak_base_shear_kn=max((abs(y) for _, y in curve), default=None) if curve else None,
        max_rel_error_vs_structural_clamped=max_err,
        rms_rel_error_vs_structural_clamped=rms_err,
        max_embedding_gap_m=safe_float(observables.get("max_abs_top_rebar_face_gap_m")),
        peak_cracked_gauss_points=int(observables.get("peak_cracked_gauss_points", 0) or 0)
        if observables
        else None,
        first_crack_drift_mm=first_crack_drift_mm,
        output_dir=str(bundle_dir),
    )


def structural_failure_row(
    case: StructuralAmplitudeCase,
    bundle_dir: Path,
    *,
    status: str,
    timed_out: bool,
    elapsed: float,
) -> StructuralAmplitudeRow:
    return StructuralAmplitudeRow(
        key=case.key,
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
    case: ContinuumAmplitudeCase,
    bundle_dir: Path,
    *,
    status: str,
    timed_out: bool,
    elapsed: float,
) -> ContinuumAmplitudeRow:
    return ContinuumAmplitudeRow(
        key=case.key,
        family_key=case.family_key,
        family_label=case.family_label,
        reinforcement_mode=case.reinforcement_mode,
        mesh=f"{case.nx}x{case.ny}x{case.nz}",
        longitudinal_bias_power=case.longitudinal_bias_power,
        monotonic_tip_mm=case.monotonic_tip_mm,
        concrete_profile=case.concrete_profile,
        concrete_tangent_mode=case.concrete_tangent_mode,
        concrete_characteristic_length_mode=case.concrete_characteristic_length_mode,
        status=status,
        completed_successfully=False,
        timed_out=timed_out,
        process_wall_seconds=elapsed,
        reported_total_wall_seconds=None,
        peak_base_shear_kn=None,
        max_rel_error_vs_structural_clamped=None,
        rms_rel_error_vs_structural_clamped=None,
        max_embedding_gap_m=None,
        peak_cracked_gauss_points=None,
        first_crack_drift_mm=None,
        output_dir=str(bundle_dir),
    )


def run_structural_case(
    exe: Path,
    root: Path,
    repo_root: Path,
    args: argparse.Namespace,
    case: StructuralAmplitudeCase,
) -> StructuralAmplitudeRow:
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
        (bundle_dir / "runner_stdout.log").write_text((exc.output or ""), encoding="utf-8")
        (bundle_dir / "runner_stderr.log").write_text((exc.stderr or ""), encoding="utf-8")
        if manifest.exists():
            return structural_row_from_bundle(
                case, bundle_dir, None, status="completed_beyond_timeout_budget", timed_out=True
            )
        return structural_failure_row(
            case, bundle_dir, status="timeout_budget_exceeded", timed_out=True, elapsed=float(case.timeout_seconds)
        )

    (bundle_dir / "runner_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (bundle_dir / "runner_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        if manifest.exists():
            return structural_row_from_bundle(
                case, bundle_dir, elapsed, status="completed_with_runner_warning"
            )
        return structural_failure_row(
            case, bundle_dir, status="runner_failed", timed_out=False, elapsed=elapsed
        )
    if not manifest.exists():
        return structural_failure_row(
            case, bundle_dir, status="missing_runtime_manifest", timed_out=False, elapsed=elapsed
        )
    return structural_row_from_bundle(case, bundle_dir, elapsed)


def run_continuum_case(
    exe: Path,
    root: Path,
    repo_root: Path,
    args: argparse.Namespace,
    case: ContinuumAmplitudeCase,
    structural_curve: list[tuple[float, float]],
) -> ContinuumAmplitudeRow:
    bundle_dir = root / case.key
    ensure_dir(bundle_dir)
    manifest = bundle_dir / "runtime_manifest.json"
    if args.reuse_existing and manifest.exists():
        return continuum_row_from_bundle(
            case, bundle_dir, None, structural_curve, status="reused_cached_bundle"
        )
    try:
        elapsed, completed = run_command(
            continuum_command(exe, bundle_dir, args, case),
            cwd=repo_root,
            timeout_seconds=case.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        (bundle_dir / "runner_stdout.log").write_text((exc.output or ""), encoding="utf-8")
        (bundle_dir / "runner_stderr.log").write_text((exc.stderr or ""), encoding="utf-8")
        if manifest.exists():
            return continuum_row_from_bundle(
                case,
                bundle_dir,
                None,
                structural_curve,
                status="completed_beyond_timeout_budget",
                timed_out=True,
            )
        return continuum_failure_row(
            case, bundle_dir, status="timeout_budget_exceeded", timed_out=True, elapsed=float(case.timeout_seconds)
        )

    (bundle_dir / "runner_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (bundle_dir / "runner_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        if manifest.exists():
            return continuum_row_from_bundle(
                case, bundle_dir, elapsed, structural_curve, status="completed_with_runner_warning"
            )
        return continuum_failure_row(
            case, bundle_dir, status="runner_failed", timed_out=False, elapsed=elapsed
        )
    if not manifest.exists():
        return continuum_failure_row(
            case, bundle_dir, status="missing_runtime_manifest", timed_out=False, elapsed=elapsed
        )
    return continuum_row_from_bundle(case, bundle_dir, elapsed, structural_curve)


def plot_error_vs_amplitude(
    continuum_rows: list[ContinuumAmplitudeRow],
    primary: Path,
    secondary: Path,
) -> None:
    families = {
        "embedded_uniform_control": ("Hex20 emb. 2x2x2 uniforme", PURPLE),
        "embedded_promoted_bias3_fixedend": (
            "Hex20 emb. 2x2x2 sesgo=3, lb fijo",
            ORANGE,
        ),
        "host_promoted_bias3_fixedend": (
            "Hex20 host 2x2x2 sesgo=3, lb fijo",
            BLUE,
        ),
    }
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for family_key, (label, color) in families.items():
        points = [
            row
            for row in continuum_rows
            if row.family_key == family_key
            and row.completed_successfully
            and row.rms_rel_error_vs_structural_clamped is not None
        ]
        points.sort(key=lambda row: row.monotonic_tip_mm)
        if not points:
            continue
        ax.plot(
            [row.monotonic_tip_mm for row in points],
            [row.rms_rel_error_vs_structural_clamped for row in points],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
    ax.set_xlabel("Monotonic tip amplitude [mm]")
    ax.set_ylabel("RMS relative base-shear path error vs clamped beam")
    ax.set_title("Continuum monotonic amplitude audit: error vs clamped beam control")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(primary)
    fig.savefig(secondary)
    plt.close(fig)


def plot_timing_vs_amplitude(
    continuum_rows: list[ContinuumAmplitudeRow],
    structural_rows: list[StructuralAmplitudeRow],
    primary: Path,
    secondary: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    structural = [row for row in structural_rows if row.completed_successfully]
    structural.sort(key=lambda row: row.monotonic_tip_mm)
    if structural:
        ax.plot(
            [row.monotonic_tip_mm for row in structural],
            [row.reported_total_wall_seconds for row in structural],
            marker="s",
            linewidth=2.0,
            color=GREEN,
            label="Beam clamped reference",
        )
    families = {
        "embedded_uniform_control": ("Hex20 emb. 2x2x2 uniforme", PURPLE),
        "embedded_promoted_bias3_fixedend": (
            "Hex20 emb. 2x2x2 sesgo=3, lb fijo",
            ORANGE,
        ),
        "host_promoted_bias3_fixedend": (
            "Hex20 host 2x2x2 sesgo=3, lb fijo",
            BLUE,
        ),
    }
    for family_key, (label, color) in families.items():
        points = [
            row
            for row in continuum_rows
            if row.family_key == family_key
            and row.completed_successfully
            and row.reported_total_wall_seconds is not None
        ]
        points.sort(key=lambda row: row.monotonic_tip_mm)
        if not points:
            continue
        ax.plot(
            [row.monotonic_tip_mm for row in points],
            [row.reported_total_wall_seconds for row in points],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
    ax.set_xlabel("Monotonic tip amplitude [mm]")
    ax.set_ylabel("Reported total wall time [s]")
    ax.set_title("Continuum monotonic amplitude audit: timing")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(primary)
    fig.savefig(secondary)
    plt.close(fig)


def plot_base_shear_envelopes(
    structural_rows: list[StructuralAmplitudeRow],
    continuum_rows: list[ContinuumAmplitudeRow],
    primary: Path,
    secondary: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    structural = [row for row in structural_rows if row.completed_successfully and row.peak_base_shear_kn is not None]
    structural.sort(key=lambda row: row.monotonic_tip_mm)
    if structural:
        ax.plot(
            [row.monotonic_tip_mm for row in structural],
            [row.peak_base_shear_kn for row in structural],
            marker="s",
            linewidth=2.0,
            color=GREEN,
            label="Beam clamped reference",
        )
    families = {
        "embedded_uniform_control": ("Hex20 emb. 2x2x2 uniforme", PURPLE),
        "embedded_promoted_bias3_fixedend": (
            "Hex20 emb. 2x2x2 sesgo=3, lb fijo",
            ORANGE,
        ),
        "host_promoted_bias3_fixedend": (
            "Hex20 host 2x2x2 sesgo=3, lb fijo",
            BLUE,
        ),
    }
    for family_key, (label, color) in families.items():
        points = [
            row
            for row in continuum_rows
            if row.family_key == family_key
            and row.completed_successfully
            and row.peak_base_shear_kn is not None
        ]
        points.sort(key=lambda row: row.monotonic_tip_mm)
        if not points:
            continue
        ax.plot(
            [row.monotonic_tip_mm for row in points],
            [row.peak_base_shear_kn for row in points],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
    ax.set_xlabel("Monotonic tip amplitude [mm]")
    ax.set_ylabel("Peak base shear [kN]")
    ax.set_title("Continuum monotonic amplitude audit: peak base shear")
    ax.legend(frameon=False, ncol=2)
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

    amplitudes_mm = parse_amplitudes_mm(args.amplitudes_mm)

    structural_results: list[StructuralAmplitudeRow] = []
    structural_curve_by_amplitude: dict[float, list[tuple[float, float]]] = {}
    for case in [case for case in structural_cases(amplitudes_mm) if case_selected(case.key, args.case_filter)]:
        row = run_structural_case(args.structural_exe, args.output_dir, repo_root, args, case)
        structural_results.append(row)
        if row.completed_successfully:
            structural_curve_by_amplitude[case.monotonic_tip_mm] = read_monotonic_curve(
                Path(row.output_dir)
            )

    continuum_results: list[ContinuumAmplitudeRow] = []
    for case in [case for case in continuum_cases(amplitudes_mm) if case_selected(case.key, args.case_filter)]:
        structural_curve = structural_curve_by_amplitude.get(case.monotonic_tip_mm)
        if structural_curve is None:
            continuum_results.append(
                continuum_failure_row(
                    case,
                    args.output_dir / case.key,
                    status="missing_structural_reference",
                    timed_out=False,
                    elapsed=0.0,
                )
            )
            continue
        continuum_results.append(
            run_continuum_case(
                args.continuum_exe,
                args.output_dir,
                repo_root,
                args,
                case,
                structural_curve,
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
            "reference_note": (
                "The structural comparator is the clamped-rotation beam control "
                "identified in the monotonic equivalence audit as the right "
                "kinematic benchmark for the translated-top-face continuum slice."
            ),
            "promotion_note": (
                "The amplitude sweep now promotes the production-stabilized + "
                "fracture-secant continuum chain explicitly. The uniform "
                "Hex20 2x2x2 slice remains the cheapest nonlinear control, "
                "while the fixed-end biased embedded slice is the promoted "
                "local-validation baseline for this stage."
            ),
            "frontier_note": (
                "Rows marked with timeout or partial failure define the current "
                "operational frontier of the nonlinear continuum amplitude sweep."
            ),
        },
    }
    write_json(args.output_dir / "continuum_monotonic_amplitude_summary.json", summary)
    write_csv(
        args.output_dir / "continuum_monotonic_amplitude_rows.csv",
        [asdict(row) for row in structural_results + continuum_results],
    )

    plot_base_shear_envelopes(
        structural_results,
        continuum_results,
        args.figures_dir / "reduced_rc_continuum_monotonic_amplitude_base_shear.png",
        args.secondary_figures_dir / "reduced_rc_continuum_monotonic_amplitude_base_shear.png",
    )
    plot_error_vs_amplitude(
        continuum_results,
        args.figures_dir / "reduced_rc_continuum_monotonic_amplitude_error.png",
        args.secondary_figures_dir / "reduced_rc_continuum_monotonic_amplitude_error.png",
    )
    plot_timing_vs_amplitude(
        continuum_results,
        structural_results,
        args.figures_dir / "reduced_rc_continuum_monotonic_amplitude_timing.png",
        args.secondary_figures_dir / "reduced_rc_continuum_monotonic_amplitude_timing.png",
    )


if __name__ == "__main__":
    main()
