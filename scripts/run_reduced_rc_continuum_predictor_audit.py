#!/usr/bin/env python3
"""
Reduced RC continuum predictor audit.

This audit measures whether carrying a better accepted-step seed into the next
increment buys a real improvement for the promoted local continuum slice,
especially across fracture onset. The intent is to validate the predictor seam
as an architectural improvement, not as a hidden benchmark-specific tweak.
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
class PredictorCase:
    key: str
    amplitude_mm: float
    steps: int
    predictor_policy: str
    timeout_seconds: int


@dataclass(frozen=True)
class PredictorRow:
    key: str
    predictor_policy: str
    amplitude_mm: float
    status: str
    completed_successfully: bool
    timed_out: bool
    process_wall_seconds: float
    reported_total_wall_seconds: float | None
    reported_solve_wall_seconds: float | None
    accepted_runtime_steps: int | None
    failed_attempt_count: int | None
    peak_cracked_gauss_points: int | None
    first_crack_runtime_step: int | None
    first_crack_drift_mm: float | None
    max_abs_base_shear_kn: float | None
    max_newton_iterations: float | None
    mean_newton_iterations: float | None
    max_bisection_level: int | None
    output_dir: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the reduced RC continuum predictor audit."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument(
        "--amplitudes-mm",
        default="2.5,5,10,20",
        help="Comma-separated monotonic amplitudes in mm.",
    )
    parser.add_argument(
        "--predictor-policies",
        default="current-state-only,secant,adaptive-secant,linearized-equilibrium,hybrid-secant-linearized",
        help="Comma-separated predictor policies.",
    )
    parser.add_argument(
        "--case-filter",
        action="append",
        default=[],
        help="Only run cases whose key contains one of these substrings.",
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


def safe_int(value: object) -> int | None:
    parsed = safe_float(value)
    return None if parsed is None else int(parsed)


def parse_amplitudes_mm(raw: str) -> list[float]:
    amps: list[float] = []
    for token in raw.split(","):
        stripped = token.strip()
        if stripped:
            amps.append(float(stripped))
    if not amps:
        raise ValueError("At least one amplitude is required.")
    return amps


def parse_predictor_policies(raw: str) -> list[str]:
    policies = [token.strip() for token in raw.split(",") if token.strip()]
    if not policies:
        raise ValueError("At least one predictor policy is required.")
    return policies


def monotonic_steps_for_amplitude(amplitude_mm: float) -> int:
    if amplitude_mm <= 5.0:
        return 6
    if amplitude_mm <= 10.0:
        return 8
    if amplitude_mm <= 15.0:
        return 10
    return 12


def timeout_for_amplitude(amplitude_mm: float) -> int:
    if amplitude_mm <= 5.0:
        return 1200
    if amplitude_mm <= 10.0:
        return 1800
    return 2400


def case_selected(key: str, filters: list[str]) -> bool:
    if not filters:
        return True
    lowered = key.lower()
    return any(token.lower() in lowered for token in filters)


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


def predictor_cases(
    amplitudes_mm: list[float], predictor_policies: list[str]
) -> list[PredictorCase]:
    cases: list[PredictorCase] = []
    for amplitude in amplitudes_mm:
        for predictor_policy in predictor_policies:
            key = (
                "continuum_predictor_"
                f"{predictor_policy.replace('-', '_')}_"
                f"{str(amplitude).replace('.', 'p')}mm"
            )
            cases.append(
                PredictorCase(
                    key=key,
                    amplitude_mm=amplitude,
                    steps=monotonic_steps_for_amplitude(amplitude),
                    predictor_policy=predictor_policy,
                    timeout_seconds=timeout_for_amplitude(amplitude),
                )
            )
    return cases


def continuum_command(
    exe: Path,
    output_dir: Path,
    args: argparse.Namespace,
    case: PredictorCase,
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
        "production-stabilized",
        "--concrete-tangent-mode",
        "fracture-secant",
        "--concrete-characteristic-length-mode",
        "fixed-end-longitudinal-host-edge-mm",
        "--reinforcement-mode",
        "embedded-longitudinal-bars",
        "--hex-order",
        "hex20",
        "--nx",
        "2",
        "--ny",
        "2",
        "--nz",
        "2",
        "--longitudinal-bias-power",
        "3",
        "--solver-policy",
        "newton-l2-only",
        "--predictor-policy",
        case.predictor_policy,
        "--continuation",
        "monolithic",
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{case.amplitude_mm}",
        "--monotonic-steps",
        str(case.steps),
        "--max-bisections",
        "8",
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def row_from_bundle(
    case: PredictorCase,
    bundle_dir: Path,
    elapsed: float | None,
    *,
    status: str = "completed",
    timed_out: bool = False,
) -> PredictorRow:
    manifest = read_json(bundle_dir / "runtime_manifest.json")
    timing = manifest.get("timing", {})
    observables = manifest.get("observables", {})
    solve_summary = manifest.get("solve_summary", {})
    control_rows = (
        read_csv_rows(bundle_dir / "control_state.csv")
        if (bundle_dir / "control_state.csv").exists()
        else []
    )
    newton_values = [
        value
        for row in control_rows
        if (value := safe_float(row.get("newton_iterations"))) is not None
    ]
    bisection_values = [
        value
        for row in control_rows
        if (value := safe_int(row.get("max_bisection_level"))) is not None
    ]
    first_crack_runtime_step = safe_int(observables.get("first_crack_runtime_step"))
    first_crack_drift_mm: float | None = None
    if first_crack_runtime_step is not None:
        for row in control_rows:
            runtime_step = safe_int(row.get("runtime_step"))
            target_drift = safe_float(row.get("target_drift_m"))
            if runtime_step == first_crack_runtime_step and target_drift is not None:
                first_crack_drift_mm = 1.0e3 * target_drift
                break

    return PredictorRow(
        key=case.key,
        predictor_policy=case.predictor_policy,
        amplitude_mm=case.amplitude_mm,
        status=status,
        completed_successfully=bool(manifest.get("completed_successfully", False)),
        timed_out=timed_out,
        process_wall_seconds=(
            elapsed
            if elapsed is not None
            else float(timing.get("total_wall_seconds", math.nan))
        ),
        reported_total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
        reported_solve_wall_seconds=safe_float(timing.get("solve_wall_seconds")),
        accepted_runtime_steps=safe_int(solve_summary.get("accepted_runtime_steps")),
        failed_attempt_count=safe_int(solve_summary.get("failed_attempt_count")),
        peak_cracked_gauss_points=safe_int(observables.get("peak_cracked_gauss_points")),
        first_crack_runtime_step=first_crack_runtime_step,
        first_crack_drift_mm=first_crack_drift_mm,
        max_abs_base_shear_kn=(
            None
            if (value := safe_float(observables.get("max_abs_base_shear_mn"))) is None
            else 1.0e3 * value
        ),
        max_newton_iterations=(max(newton_values) if newton_values else None),
        mean_newton_iterations=(
            sum(newton_values) / len(newton_values) if newton_values else None
        ),
        max_bisection_level=(max(bisection_values) if bisection_values else None),
        output_dir=str(bundle_dir),
    )


def timeout_row(case: PredictorCase, bundle_dir: Path) -> PredictorRow:
    return PredictorRow(
        key=case.key,
        predictor_policy=case.predictor_policy,
        amplitude_mm=case.amplitude_mm,
        status="timeout",
        completed_successfully=False,
        timed_out=True,
        process_wall_seconds=float(case.timeout_seconds),
        reported_total_wall_seconds=None,
        reported_solve_wall_seconds=None,
        accepted_runtime_steps=None,
        failed_attempt_count=None,
        peak_cracked_gauss_points=None,
        first_crack_runtime_step=None,
        first_crack_drift_mm=None,
        max_abs_base_shear_kn=None,
        max_newton_iterations=None,
        mean_newton_iterations=None,
        max_bisection_level=None,
        output_dir=str(bundle_dir),
    )


def failed_row(
    case: PredictorCase,
    bundle_dir: Path,
    elapsed: float,
    completed: subprocess.CompletedProcess[str],
) -> PredictorRow:
    manifest_path = bundle_dir / "runtime_manifest.json"
    if manifest_path.exists():
        return row_from_bundle(case, bundle_dir, elapsed, status="failed")
    message_path = bundle_dir / "subprocess_failure.txt"
    message_path.write_text(
        f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    return PredictorRow(
        key=case.key,
        predictor_policy=case.predictor_policy,
        amplitude_mm=case.amplitude_mm,
        status=f"returncode_{completed.returncode}",
        completed_successfully=False,
        timed_out=False,
        process_wall_seconds=elapsed,
        reported_total_wall_seconds=None,
        reported_solve_wall_seconds=None,
        accepted_runtime_steps=None,
        failed_attempt_count=None,
        peak_cracked_gauss_points=None,
        first_crack_runtime_step=None,
        first_crack_drift_mm=None,
        max_abs_base_shear_kn=None,
        max_newton_iterations=None,
        mean_newton_iterations=None,
        max_bisection_level=None,
        output_dir=str(bundle_dir),
    )


def policy_color(policy: str) -> str:
    return {
        "current-state-only": BLUE,
        "secant": ORANGE,
        "adaptive-secant": GREEN,
    }.get(policy, GRAY)


def plot_timing(rows: list[PredictorRow], path: Path) -> None:
    completed = [row for row in rows if row.completed_successfully]
    if not completed:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for policy in sorted({row.predictor_policy for row in completed}):
        policy_rows = sorted(
            (row for row in completed if row.predictor_policy == policy),
            key=lambda row: row.amplitude_mm,
        )
        ax.plot(
            [row.amplitude_mm for row in policy_rows],
            [row.process_wall_seconds for row in policy_rows],
            marker="o",
            linewidth=1.6,
            color=policy_color(policy),
            label=policy,
        )
    ax.set_xlabel("Monotonic tip amplitude [mm]")
    ax.set_ylabel("Process wall time [s]")
    ax.set_title("Continuum predictor timing audit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_newton(rows: list[PredictorRow], path: Path) -> None:
    completed = [row for row in rows if row.completed_successfully]
    if not completed:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for policy in sorted({row.predictor_policy for row in completed}):
        policy_rows = sorted(
            (row for row in completed if row.predictor_policy == policy),
            key=lambda row: row.amplitude_mm,
        )
        ax.plot(
            [row.amplitude_mm for row in policy_rows],
            [row.mean_newton_iterations for row in policy_rows],
            marker="o",
            linewidth=1.6,
            color=policy_color(policy),
            label=policy,
        )
    ax.set_xlabel("Monotonic tip amplitude [mm]")
    ax.set_ylabel("Mean Newton iterations per accepted step")
    ax.set_title("Continuum predictor Newton workload")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_fracture(rows: list[PredictorRow], path: Path) -> None:
    completed = [
        row
        for row in rows
        if row.completed_successfully and row.first_crack_drift_mm is not None
    ]
    if not completed:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for policy in sorted({row.predictor_policy for row in completed}):
        policy_rows = sorted(
            (
                row
                for row in completed
                if row.predictor_policy == policy and row.first_crack_drift_mm is not None
            ),
            key=lambda row: row.amplitude_mm,
        )
        ax.plot(
            [row.amplitude_mm for row in policy_rows],
            [row.first_crack_drift_mm for row in policy_rows],
            marker="o",
            linewidth=1.6,
            color=policy_color(policy),
            label=policy,
        )
    ax.set_xlabel("Monotonic tip amplitude [mm]")
    ax.set_ylabel("First crack drift [mm]")
    ax.set_title("Continuum predictor fracture onset audit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)
    ensure_dir(args.secondary_figures_dir)

    amplitudes_mm = parse_amplitudes_mm(args.amplitudes_mm)
    predictor_policies = parse_predictor_policies(args.predictor_policies)
    rows: list[PredictorRow] = []

    for case in predictor_cases(amplitudes_mm, predictor_policies):
        if not case_selected(case.key, args.case_filter):
            continue
        bundle_dir = args.output_dir / case.key
        ensure_dir(bundle_dir)
        manifest_path = bundle_dir / "runtime_manifest.json"
        if args.reuse_existing and manifest_path.exists():
            if args.print_progress:
                print(f"[reuse] {case.key}")
            rows.append(row_from_bundle(case, bundle_dir, None, status="reused"))
            continue

        command = continuum_command(args.continuum_exe, bundle_dir, args, case)
        if args.print_progress:
            print(f"[run] {case.key}")
        try:
            elapsed, completed = run_command(
                command,
                cwd=Path(__file__).resolve().parent.parent,
                timeout_seconds=case.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            rows.append(timeout_row(case, bundle_dir))
            continue

        if completed.returncode == 0 and manifest_path.exists():
            rows.append(row_from_bundle(case, bundle_dir, elapsed))
        else:
            rows.append(failed_row(case, bundle_dir, elapsed, completed))

    rows.sort(key=lambda row: (row.amplitude_mm, row.predictor_policy))

    summary = {
        "audit": "reduced_rc_continuum_predictor_audit",
        "cases": [asdict(row) for row in rows],
        "promoted_slice": {
            "hex_order": "hex20",
            "mesh": "2x2x2",
            "reinforcement_mode": "embedded-longitudinal-bars",
            "longitudinal_bias_power": 3.0,
            "concrete_profile": "production-stabilized",
            "concrete_tangent_mode": "fracture-secant",
            "concrete_characteristic_length_mode": "fixed-end-longitudinal-host-edge-mm",
            "solver_policy_kind": "newton-l2-only",
        },
        "closing_note": (
            "This audit promotes predictor policies only if they reduce the "
            "solve cost of the promoted continuum slice without moving the "
            "fracture onset or the macroscopic response in a benchmark-specific way."
        ),
    }
    write_json(args.output_dir / "continuum_predictor_audit_summary.json", summary)
    write_csv(
        args.output_dir / "continuum_predictor_audit_cases.csv",
        [asdict(row) for row in rows],
    )

    figures = {
        "timing": "reduced_rc_continuum_predictor_timing.png",
        "newton": "reduced_rc_continuum_predictor_newton.png",
        "fracture": "reduced_rc_continuum_predictor_fracture_onset.png",
    }
    plot_timing(rows, args.figures_dir / figures["timing"])
    plot_newton(rows, args.figures_dir / figures["newton"])
    plot_fracture(rows, args.figures_dir / figures["fracture"])
    for name in figures.values():
        source = args.figures_dir / name
        if source.exists():
            (args.secondary_figures_dir / name).write_bytes(source.read_bytes())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
