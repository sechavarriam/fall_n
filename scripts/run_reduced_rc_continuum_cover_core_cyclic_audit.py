#!/usr/bin/env python3
"""
Freeze the first cover/core-aware cyclic audit of the reduced RC continuum.

This pass keeps the protocol intentionally narrow and physically interpretable:

1. same axial preload;
2. same short cyclic drift window;
3. same Hex20 host family;
4. explicit comparison between:
   - plain cover/core concrete,
   - embedded interior bars,
   - embedded boundary bars; and
5. one strongly base-biased embedded branch to test whether the corrected
   fixed-end characteristic-length policy remains operational under reversal.
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
RED = "#c53030"
PURPLE = "#7c3aed"
GRAY = "#4b5563"


@dataclass(frozen=True)
class ContinuumCyclicCase:
    key: str
    label: str
    reinforcement_mode: str
    rebar_layout: str
    nx: int
    ny: int
    nz: int
    longitudinal_bias_power: float
    concrete_characteristic_length_mode: str
    timeout_seconds: int
    amplitudes_mm: tuple[float, ...] = (1.25, 2.5, 5.0, 10.0)
    steps_per_segment: int = 3
    material_mode: str = "nonlinear"
    concrete_profile: str = "production-stabilized"
    concrete_tangent_mode: str = "fracture-secant"
    reinforcement_solver_policy: str = "newton-l2-only"
    predictor_policy: str = "hybrid-secant-linearized"
    host_concrete_zoning_mode: str = "cover-core-split"
    transverse_mesh_mode: str = "cover-aligned"
    transverse_cover_subdivisions_x_each_side: int = 1
    transverse_cover_subdivisions_y_each_side: int = 1
    axial_compression_mn: float = 0.02
    axial_preload_steps: int = 4
    continuation: str = "reversal-guarded"
    continuation_segment_substep_factor: int = 2
    max_bisections: int = 8
    hex_order: str = "hex20"


@dataclass(frozen=True)
class ContinuumCyclicRow:
    key: str
    label: str
    reinforcement_mode: str
    rebar_layout: str
    mesh: str
    longitudinal_bias_power: float
    characteristic_length_mode: str
    status: str
    completed_successfully: bool
    timed_out: bool
    process_wall_seconds: float
    reported_total_wall_seconds: float | None
    peak_base_shear_kn: float | None
    peak_cracked_gauss_points: int | None
    max_crack_opening: float | None
    first_crack_drift_mm: float | None
    peak_rebar_stress_mpa: float | None
    peak_rebar_strain: float | None
    max_host_bar_strain_gap: float | None
    output_dir: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the reduced RC continuum cover/core cyclic audit."
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
    parser.add_argument(
        "--amplitudes-mm",
        default="1.25,2.5,5,10",
        help="Comma-separated cyclic drift amplitudes in mm applied to all cases.",
    )
    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=3,
        help="Accepted continuation steps per segment applied to all cases.",
    )
    parser.add_argument(
        "--case-filter",
        action="append",
        default=[],
        help="Only run cases whose key contains one of these substrings.",
    )
    parser.add_argument(
        "--solver-policy",
        default="newton-l2-only",
        help="Nonlinear solver policy forwarded to all cyclic continuum cases.",
    )
    parser.add_argument(
        "--predictor-policy",
        default="hybrid-secant-linearized",
        help="Increment predictor policy forwarded to all cyclic continuum cases.",
    )
    parser.add_argument(
        "--continuation",
        default="reversal-guarded",
        help="Continuation mode forwarded to all cyclic continuum cases.",
    )
    parser.add_argument(
        "--continuation-segment-substep-factor",
        type=int,
        default=2,
        help="Continuation substep factor forwarded to all cyclic continuum cases.",
    )
    parser.add_argument(
        "--max-bisections",
        type=int,
        default=8,
        help="Maximum bisections forwarded to all cyclic continuum cases.",
    )
    parser.add_argument(
        "--plain-timeout-seconds",
        type=int,
        default=1800,
        help="Timeout budget for the plain cover/core branch.",
    )
    parser.add_argument(
        "--interior-timeout-seconds",
        type=int,
        default=2400,
        help="Timeout budget for the promoted embedded-interior branch.",
    )
    parser.add_argument(
        "--boundary-timeout-seconds",
        type=int,
        default=2400,
        help="Timeout budget for the boundary-bar control branch.",
    )
    parser.add_argument(
        "--biased-timeout-seconds",
        type=int,
        default=4800,
        help="Timeout budget for the high-cost biased 4x4x4 frontier branch.",
    )
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


def parse_csv_doubles(raw: str) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in raw.split(",") if token.strip())
    if not values:
        raise ValueError("--amplitudes-mm must contain at least one value.")
    return values


def amplitude_suffix(amplitudes_mm: tuple[float, ...]) -> str:
    peak_mm = max(amplitudes_mm)
    label = f"{peak_mm:.3f}".rstrip("0").rstrip(".")
    return label.replace(".", "p") + "mm"


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


def case_selected(key: str, filters: list[str]) -> bool:
    if not filters:
        return True
    key_lower = key.lower()
    return any(token.lower() in key_lower for token in filters)


def build_command(exe: Path, output_dir: Path, case: ContinuumCyclicCase) -> list[str]:
    return [
        str(exe),
        "--output-dir",
        str(output_dir),
        "--analysis",
        "cyclic",
        "--material-mode",
        case.material_mode,
        "--concrete-profile",
        case.concrete_profile,
        "--concrete-tangent-mode",
        case.concrete_tangent_mode,
        "--concrete-characteristic-length-mode",
        case.concrete_characteristic_length_mode,
        "--reinforcement-mode",
        case.reinforcement_mode,
        "--rebar-layout",
        case.rebar_layout,
        "--host-concrete-zoning-mode",
        case.host_concrete_zoning_mode,
        "--transverse-mesh-mode",
        case.transverse_mesh_mode,
        "--transverse-cover-subdivisions-x-each-side",
        str(case.transverse_cover_subdivisions_x_each_side),
        "--transverse-cover-subdivisions-y-each-side",
        str(case.transverse_cover_subdivisions_y_each_side),
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
        case.reinforcement_solver_policy,
        "--predictor-policy",
        case.predictor_policy,
        "--axial-compression-mn",
        f"{case.axial_compression_mn}",
        "--axial-preload-steps",
        str(case.axial_preload_steps),
        "--continuation",
        case.continuation,
        "--continuation-segment-substep-factor",
        str(case.continuation_segment_substep_factor),
        "--max-bisections",
        str(case.max_bisections),
        "--amplitudes-mm",
        ",".join(f"{value:.6f}" for value in case.amplitudes_mm),
        "--steps-per-segment",
        str(case.steps_per_segment),
    ]


def read_case_row(
    case: ContinuumCyclicCase, output_dir: Path, process_wall_seconds: float, timed_out: bool
) -> ContinuumCyclicRow:
    manifest = read_json(output_dir / "runtime_manifest.json")
    observables = manifest.get("observables") or {}
    timing = manifest.get("timing") or {}
    manifest_termination = str(manifest.get("termination_reason", ""))
    manifest_timed_out = bool(manifest.get("timed_out")) or (
        manifest_termination == "timeout_budget_exceeded"
    )
    effective_timed_out = timed_out or manifest_timed_out
    mesh = f"{case.nx}x{case.ny}x{case.nz}"
    return ContinuumCyclicRow(
        key=case.key,
        label=case.label,
        reinforcement_mode=case.reinforcement_mode,
        rebar_layout=case.rebar_layout,
        mesh=mesh,
        longitudinal_bias_power=case.longitudinal_bias_power,
        characteristic_length_mode=case.concrete_characteristic_length_mode,
        status=(
            "completed"
            if bool(manifest.get("completed_successfully"))
            else ("timeout_budget_exceeded" if effective_timed_out else "aborted")
        ),
        completed_successfully=bool(manifest.get("completed_successfully")),
        timed_out=effective_timed_out,
        process_wall_seconds=process_wall_seconds,
        reported_total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
        peak_base_shear_kn=(
            safe_float(observables.get("max_abs_base_shear_mn")) * 1.0e3
            if safe_float(observables.get("max_abs_base_shear_mn")) is not None
            else None
        ),
        peak_cracked_gauss_points=(
            int(observables.get("peak_cracked_gauss_points"))
            if observables.get("peak_cracked_gauss_points") is not None
            else None
        ),
        max_crack_opening=safe_float(observables.get("max_crack_opening")),
        first_crack_drift_mm=safe_float(observables.get("first_crack_drift_mm")),
        peak_rebar_stress_mpa=safe_float(observables.get("max_abs_rebar_stress_mpa")),
        peak_rebar_strain=safe_float(observables.get("max_abs_rebar_strain")),
        max_host_bar_strain_gap=safe_float(
            observables.get("max_abs_host_rebar_axial_strain_gap")
        ),
        output_dir=str(output_dir),
    )


def run_case(
    exe: Path,
    root: Path,
    case: ContinuumCyclicCase,
    reuse_existing: bool,
    print_progress: bool,
) -> ContinuumCyclicRow:
    output_dir = root / case.key
    ensure_dir(output_dir)
    manifest_path = output_dir / "runtime_manifest.json"
    if reuse_existing and manifest_path.exists():
        return read_case_row(case, output_dir, 0.0, False)

    command = build_command(exe, output_dir, case)
    if print_progress:
        print(f"[continuum-cyclic] {case.key}")
    try:
        process_wall_seconds, completed = run_command(
            command, Path(__file__).resolve().parent.parent, case.timeout_seconds
        )
        if completed.returncode != 0:
            (output_dir / "runner_stdout.log").write_text(
                completed.stdout or "", encoding="utf-8"
            )
            (output_dir / "runner_stderr.log").write_text(
                completed.stderr or "", encoding="utf-8"
            )
            if not manifest_path.exists():
                raise RuntimeError(
                    f"{case.key} failed without runtime manifest (return code {completed.returncode})."
                )
        return read_case_row(case, output_dir, process_wall_seconds, False)
    except subprocess.TimeoutExpired as exc:
        (output_dir / "runner_stdout.log").write_text(
            exc.output or "", encoding="utf-8"
        )
        (output_dir / "runner_stderr.log").write_text(
            exc.stderr or "", encoding="utf-8"
        )
        if manifest_path.exists():
            return read_case_row(case, output_dir, case.timeout_seconds, True)

        write_json(
            manifest_path,
            {
                "completed_successfully": False,
                "termination_reason": "timeout_budget_exceeded",
                "timed_out": True,
                "timing": {"total_wall_seconds": float(case.timeout_seconds)},
            },
        )
        return ContinuumCyclicRow(
            key=case.key,
            label=case.label,
            reinforcement_mode=case.reinforcement_mode,
            rebar_layout=case.rebar_layout,
            mesh=f"{case.nx}x{case.ny}x{case.nz}",
            longitudinal_bias_power=case.longitudinal_bias_power,
            characteristic_length_mode=case.concrete_characteristic_length_mode,
            status="timeout_budget_exceeded",
            completed_successfully=False,
            timed_out=True,
            process_wall_seconds=float(case.timeout_seconds),
            reported_total_wall_seconds=None,
            peak_base_shear_kn=None,
            peak_cracked_gauss_points=None,
            max_crack_opening=None,
            first_crack_drift_mm=None,
            peak_rebar_stress_mpa=None,
            peak_rebar_strain=None,
            max_host_bar_strain_gap=None,
            output_dir=str(output_dir),
        )


def read_hysteresis(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def save(fig: plt.Figure, out_dirs: list[Path], stem: str) -> list[str]:
    saved: list[str] = []
    for out_dir in out_dirs:
        ensure_dir(out_dir)
        target = out_dir / f"{stem}.png"
        fig.savefig(target)
        saved.append(str(target))
    plt.close(fig)
    return saved


def plot_hysteresis(
    rows: list[ContinuumCyclicRow], out_dirs: list[Path], suffix: str
) -> list[str]:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    colors = [BLUE, ORANGE, GREEN, PURPLE, RED]
    for color, row in zip(colors, rows):
        if not row.completed_successfully:
            continue
        hysteresis = read_hysteresis(Path(row.output_dir) / "hysteresis.csv")
        if not hysteresis:
            continue
        drift_mm = [
            float(record.get("avg_top_face_total_dx_m", 0.0)) * 1.0e3
            if "avg_top_face_total_dx_m" in record
            else float(record.get("target_drift_m", 0.0)) * 1.0e3
            for record in hysteresis
        ]
        # StepRecord CSV uses force_residual_norm and similar; control_state is more reliable.
        if not any(abs(value) > 0.0 for value in drift_mm):
            control_rows = read_csv_rows(Path(row.output_dir) / "control_state.csv")
            drift_mm = [float(record.get("avg_top_face_total_dx_m", 0.0)) * 1.0e3 for record in control_rows]
            shear_kn = [float(record.get("base_shear_MN", 0.0)) * 1.0e3 for record in control_rows]
        else:
            shear_kn = [
                float(record.get("base_shear_MN", 0.0)) * 1.0e3
                for record in hysteresis
            ]
        ax.plot(drift_mm, shear_kn, linewidth=1.6, color=color, label=row.label)

    ax.set_title(f"Continuum cyclic hysteresis on the cover/core foundation ({suffix})")
    ax.set_xlabel("Top drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.legend(frameon=True, fontsize=8)
    return save(fig, out_dirs, f"reduced_rc_continuum_cover_core_cyclic_hysteresis_{suffix}")


def plot_crack_opening(
    rows: list[ContinuumCyclicRow], out_dirs: list[Path], suffix: str
) -> list[str]:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    colors = [BLUE, ORANGE, GREEN, PURPLE, RED]
    for color, row in zip(colors, rows):
        if not row.completed_successfully:
            continue
        control_rows = read_csv_rows(Path(row.output_dir) / "control_state.csv")
        crack_rows = read_csv_rows(Path(row.output_dir) / "crack_state.csv")
        if not control_rows or not crack_rows:
            continue
        drift_by_step = {
            int(record["runtime_step"]): float(record.get("avg_top_face_total_dx_m", 0.0)) * 1.0e3
            for record in control_rows
        }
        drift_mm: list[float] = []
        max_opening: list[float] = []
        for record in crack_rows:
            runtime_step = int(record.get("runtime_step", -1))
            if runtime_step not in drift_by_step:
                continue
            drift_mm.append(drift_by_step[runtime_step])
            max_opening.append(float(record.get("max_crack_opening", 0.0)) * 1.0e3)
        ax.plot(drift_mm, max_opening, linewidth=1.6, color=color, label=row.label)

    ax.set_title(f"Maximum crack opening along the cyclic window ({suffix})")
    ax.set_xlabel("Top drift [mm]")
    ax.set_ylabel("Max crack opening [mm]")
    ax.legend(frameon=True, fontsize=8)
    return save(
        fig,
        out_dirs,
        f"reduced_rc_continuum_cover_core_cyclic_crack_opening_{suffix}",
    )


def plot_rebar_stress(
    rows: list[ContinuumCyclicRow], out_dirs: list[Path], suffix: str
) -> list[str]:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    colors = [ORANGE, GREEN, PURPLE, RED]
    plotted = False
    for color, row in zip(colors, rows):
        if not row.completed_successfully:
            continue
        rebar_path = Path(row.output_dir) / "rebar_history.csv"
        if not rebar_path.exists():
            continue
        control_rows = read_csv_rows(Path(row.output_dir) / "control_state.csv")
        rebar_rows = read_csv_rows(rebar_path)
        if not control_rows or not rebar_rows:
            continue
        drift_by_step = {
            int(record["runtime_step"]): float(record.get("avg_top_face_total_dx_m", 0.0)) * 1.0e3
            for record in control_rows
        }
        peak_by_step: dict[int, float] = {}
        for record in rebar_rows:
            runtime_step = int(record.get("runtime_step", -1))
            peak_by_step[runtime_step] = max(
                peak_by_step.get(runtime_step, 0.0),
                abs(float(record.get("axial_stress", 0.0))),
            )
        drift_mm = []
        stress_mpa = []
        for runtime_step in sorted(peak_by_step.keys()):
            if runtime_step not in drift_by_step:
                continue
            drift_mm.append(drift_by_step[runtime_step])
            stress_mpa.append(peak_by_step[runtime_step])
        if drift_mm:
            plotted = True
            ax.plot(drift_mm, stress_mpa, linewidth=1.6, color=color, label=row.label)

    if not plotted:
        plt.close(fig)
        return []
    ax.set_title(f"Peak rebar stress along the cyclic window ({suffix})")
    ax.set_xlabel("Top drift [mm]")
    ax.set_ylabel("Peak |steel stress| [MPa]")
    ax.legend(frameon=True, fontsize=8)
    return save(
        fig,
        out_dirs,
        f"reduced_rc_continuum_cover_core_cyclic_rebar_stress_{suffix}",
    )


def build_cases(
    amplitudes_mm: tuple[float, ...],
    steps_per_segment: int,
    *,
    solver_policy: str,
    predictor_policy: str,
    continuation: str,
    continuation_segment_substep_factor: int,
    max_bisections: int,
    plain_timeout_seconds: int,
    interior_timeout_seconds: int,
    boundary_timeout_seconds: int,
    biased_timeout_seconds: int,
) -> list[ContinuumCyclicCase]:
    suffix = amplitude_suffix(amplitudes_mm)
    return [
        ContinuumCyclicCase(
            key=f"plain_covercore_hex20_4x4x2_cyclic_{suffix}",
            label=f"Hex20 4x4x2 cover/core plain ({suffix})",
            reinforcement_mode="continuum-only",
            rebar_layout="structural-matched-eight-bar",
            nx=4,
            ny=4,
            nz=2,
            longitudinal_bias_power=1.0,
            concrete_characteristic_length_mode="mean-longitudinal-host-edge-mm",
            timeout_seconds=plain_timeout_seconds,
            amplitudes_mm=amplitudes_mm,
            steps_per_segment=steps_per_segment,
            reinforcement_solver_policy=solver_policy,
            predictor_policy=predictor_policy,
            continuation=continuation,
            continuation_segment_substep_factor=continuation_segment_substep_factor,
            max_bisections=max_bisections,
        ),
        ContinuumCyclicCase(
            key=f"embedded_covercore_interior_hex20_4x4x2_cyclic_{suffix}",
            label=f"Hex20 4x4x2 cover/core embedded interior ({suffix})",
            reinforcement_mode="embedded-longitudinal-bars",
            rebar_layout="structural-matched-eight-bar",
            nx=4,
            ny=4,
            nz=2,
            longitudinal_bias_power=1.0,
            concrete_characteristic_length_mode="mean-longitudinal-host-edge-mm",
            timeout_seconds=interior_timeout_seconds,
            amplitudes_mm=amplitudes_mm,
            steps_per_segment=steps_per_segment,
            reinforcement_solver_policy=solver_policy,
            predictor_policy=predictor_policy,
            continuation=continuation,
            continuation_segment_substep_factor=continuation_segment_substep_factor,
            max_bisections=max_bisections,
        ),
        ContinuumCyclicCase(
            key=f"embedded_covercore_boundary_hex20_4x4x2_cyclic_{suffix}",
            label=f"Hex20 4x4x2 cover/core boundary bars ({suffix})",
            reinforcement_mode="embedded-longitudinal-bars",
            rebar_layout="boundary-matched-eight-bar",
            nx=4,
            ny=4,
            nz=2,
            longitudinal_bias_power=1.0,
            concrete_characteristic_length_mode="mean-longitudinal-host-edge-mm",
            timeout_seconds=boundary_timeout_seconds,
            amplitudes_mm=amplitudes_mm,
            steps_per_segment=steps_per_segment,
            reinforcement_solver_policy=solver_policy,
            predictor_policy=predictor_policy,
            continuation=continuation,
            continuation_segment_substep_factor=continuation_segment_substep_factor,
            max_bisections=max_bisections,
        ),
        ContinuumCyclicCase(
            key=f"embedded_covercore_interior_hex20_4x4x4_bias3_fixedend_cyclic_{suffix}",
            label=f"Hex20 4x4x4 bias3 fixed-end lb embedded interior ({suffix})",
            reinforcement_mode="embedded-longitudinal-bars",
            rebar_layout="structural-matched-eight-bar",
            nx=4,
            ny=4,
            nz=4,
            longitudinal_bias_power=3.0,
            concrete_characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
            timeout_seconds=biased_timeout_seconds,
            amplitudes_mm=amplitudes_mm,
            steps_per_segment=steps_per_segment,
            reinforcement_solver_policy=solver_policy,
            predictor_policy=predictor_policy,
            continuation=continuation,
            continuation_segment_substep_factor=continuation_segment_substep_factor,
            max_bisections=max_bisections,
        ),
    ]


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.figures_dir = args.figures_dir.resolve()
    args.secondary_figures_dir = args.secondary_figures_dir.resolve()
    args.benchmark_exe = args.benchmark_exe.resolve()
    ensure_dir(args.output_dir)
    amplitudes_mm = parse_csv_doubles(args.amplitudes_mm)
    steps_per_segment = max(int(args.steps_per_segment), 1)

    rows: list[ContinuumCyclicRow] = []
    for case in build_cases(
        amplitudes_mm,
        steps_per_segment,
        solver_policy=args.solver_policy,
        predictor_policy=args.predictor_policy,
        continuation=args.continuation,
        continuation_segment_substep_factor=max(
            int(args.continuation_segment_substep_factor), 1
        ),
        max_bisections=max(int(args.max_bisections), 0),
        plain_timeout_seconds=max(int(args.plain_timeout_seconds), 1),
        interior_timeout_seconds=max(int(args.interior_timeout_seconds), 1),
        boundary_timeout_seconds=max(int(args.boundary_timeout_seconds), 1),
        biased_timeout_seconds=max(int(args.biased_timeout_seconds), 1),
    ):
        if not case_selected(case.key, args.case_filter):
            continue
        rows.append(
            run_case(
                args.benchmark_exe,
                args.output_dir,
                case,
                args.reuse_existing,
                args.print_progress,
            )
        )

    out_dirs = [args.figures_dir, args.secondary_figures_dir]
    figure_suffix = amplitude_suffix(amplitudes_mm)
    hysteresis_paths = plot_hysteresis(rows, out_dirs, figure_suffix)
    crack_paths = plot_crack_opening(rows, out_dirs, figure_suffix)
    rebar_paths = plot_rebar_stress(rows, out_dirs, figure_suffix)

    completed_rows = [row for row in rows if row.completed_successfully]
    timed_out_rows = [row for row in rows if row.timed_out]
    summary = {
        "status": "completed",
        "case_count": len(rows),
        "completed_case_count": len(completed_rows),
        "timed_out_case_count": len(timed_out_rows),
        "cases": [asdict(row) for row in rows],
        "key_findings": {
            "plain_vs_embedded_note": (
                "The first short cyclic window compares the same cover/core-aware "
                "Hex20 host with no steel, interior embedded bars, and boundary bars."
            ),
            "interface_equivalence_note": (
                "An explicit interface-bar probe collapses to the same short cyclic "
                "response as the promoted structural matched eight-bar branch for "
                "the current reduced-column geometry, because the canonical steel "
                "positions already lie on the cover-core interfaces."
            ),
            "boundary_bar_note": (
                "Boundary bars remain an explicit physical branch, not a silent substitute "
                "for the structural interior-bar baseline."
            ),
            "bias_frontier_note": (
                "The biased 4x4x4 branch keeps the corrected fixed-end characteristic-length "
                "policy and is tracked as the first higher-cost cyclic frontier."
            ),
        },
        "artifacts": {
            "amplitude_suffix": figure_suffix,
            "hysteresis_overlay_figures": hysteresis_paths,
            "crack_opening_figures": crack_paths,
            "rebar_stress_figures": rebar_paths,
        },
    }

    write_csv(
        args.output_dir / "continuum_cover_core_cyclic_cases.csv",
        [asdict(row) for row in rows],
    )
    write_json(
        args.output_dir / "continuum_cover_core_cyclic_summary.json",
        summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
