#!/usr/bin/env python3
"""
Freeze a compact structural-vs-continuum audit matrix for:

1. spatial refinement of the promoted Hex20 embedded-bar continuum slice; and
2. nonlinear solver-policy sensitivity on the same local steel bridge.

The worker reuses the already-audited structural bundle and delegates each
continuum case to the canonical steel-hysteresis bridge runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
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


@dataclass(frozen=True)
class AuditCase:
    key: str
    group: str
    hex_order: str
    nx: int
    ny: int
    nz: int
    solver_policy: str
    continuation: str = "reversal-guarded"
    continuation_segment_substep_factor: int = 2
    amplitudes_mm: str = "5,10,15,20"
    steps_per_segment: int = 2
    timeout_seconds: int = 1800


@dataclass(frozen=True)
class AuditRow:
    key: str
    group: str
    hex_order: str
    mesh: str
    solver_policy: str
    continuation: str
    status: str
    completed_successfully: bool
    timed_out: bool
    process_wall_seconds: float | None
    reported_total_wall_seconds: float | None
    reported_solve_wall_seconds: float | None
    global_max_rel_base_shear_error: float | None
    global_rms_rel_base_shear_error: float | None
    steel_max_rel_stress_error: float | None
    steel_rms_rel_stress_error: float | None
    steel_max_rel_strain_error: float | None
    steel_rms_rel_strain_error: float | None
    selected_structural_fiber_index: int | None
    selected_structural_fiber_y: float | None
    selected_structural_fiber_z: float | None
    selected_bar_index: int | None
    selected_bar_y: float | None
    selected_bar_z: float | None
    output_dir: str


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the structural-vs-continuum refinement and solver-policy audit."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--worker-script",
        type=Path,
        default=repo_root / "scripts" / "run_reduced_rc_structural_continuum_steel_hysteresis_audit.py",
    )
    parser.add_argument(
        "--structural-bundle-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_steel_hysteresis_audit_hex20_2x2x2_20mm"
        / "structural",
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
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=repo_root / "data" / "output" / "cyclic_validation",
        help="Search existing worker bundles here and reuse equivalent cases when available.",
    )
    parser.add_argument("--python-launcher", default="py -3.11")
    parser.add_argument("--beam-nodes", type=int, default=10)
    parser.add_argument("--beam-integration", default="lobatto")
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_launcher(raw: str) -> list[str]:
    return [token for token in raw.split() if token.strip()]


def parse_csv_doubles(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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


def build_cases() -> list[AuditCase]:
    return [
        AuditCase(
            key="mesh_hex20_2x2x2_newton_l2",
            group="mesh_refinement",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            solver_policy="newton-l2-only",
            timeout_seconds=1200,
        ),
        AuditCase(
            key="mesh_hex20_3x3x2_newton_l2",
            group="mesh_refinement",
            hex_order="hex20",
            nx=3,
            ny=3,
            nz=2,
            solver_policy="newton-l2-only",
            timeout_seconds=1800,
        ),
        AuditCase(
            key="mesh_hex20_2x2x4_newton_l2",
            group="mesh_refinement",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=4,
            solver_policy="newton-l2-only",
            timeout_seconds=1800,
        ),
        AuditCase(
            key="solver_hex20_2x2x2_canonical",
            group="solver_policy",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            solver_policy="canonical-cascade",
            timeout_seconds=1200,
        ),
        AuditCase(
            key="solver_hex20_2x2x2_newton_l2",
            group="solver_policy",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            solver_policy="newton-l2-only",
            timeout_seconds=1200,
        ),
        AuditCase(
            key="solver_hex20_2x2x2_dogleg",
            group="solver_policy",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            solver_policy="newton-trust-region-dogleg-only",
            timeout_seconds=1200,
        ),
        AuditCase(
            key="solver_hex20_2x2x2_anderson",
            group="solver_policy",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            solver_policy="anderson-only",
            timeout_seconds=1200,
        ),
        AuditCase(
            key="solver_hex20_2x2x2_nonlinear_cg",
            group="solver_policy",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            solver_policy="nonlinear-cg-only",
            timeout_seconds=1200,
        ),
        AuditCase(
            key="solver_hex20_2x2x2_nonlinear_gmres",
            group="solver_policy",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            solver_policy="nonlinear-gmres-only",
            timeout_seconds=1200,
        ),
        AuditCase(
            key="solver_hex20_2x2x2_newton_l2_arc_length",
            group="solver_policy",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            solver_policy="newton-l2-only",
            continuation="arc-length",
            timeout_seconds=1200,
        ),
    ]


def solver_policy_manifest_key(value: str) -> str:
    mapping = {
        "canonical-cascade": "canonical_newton_profile_cascade",
        "newton-backtracking-only": "newton_backtracking_only",
        "newton-l2-only": "newton_l2_only",
        "newton-l2-lu-symbolic-reuse-only": "newton_l2_lu_symbolic_reuse_only",
        "newton-l2-gmres-ilu1-only": "newton_l2_gmres_ilu1_only",
        "newton-trust-region-only": "newton_trust_region_only",
        "newton-trust-region-dogleg-only": "newton_trust_region_dogleg_only",
        "quasi-newton-only": "quasi_newton_only",
        "nonlinear-gmres-only": "nonlinear_gmres_only",
        "nonlinear-cg-only": "nonlinear_conjugate_gradient_only",
        "anderson-only": "anderson_acceleration_only",
        "nonlinear-richardson-only": "nonlinear_richardson_only",
    }
    return mapping.get(value, value.replace("-", "_"))


def continuation_manifest_key(value: str) -> str:
    mapping = {
        "monolithic": "monolithic_incremental_displacement_control",
        "segmented": "segmented_incremental_displacement_control",
        "reversal-guarded": "reversal_guarded_incremental_displacement_control",
        "arc-length": "arc_length_continuation_candidate",
    }
    return mapping.get(value, value.replace("-", "_"))


def matches_equivalent_bundle(
    summary: dict[str, object],
    runtime_manifest: dict[str, object],
    case: AuditCase,
    args: argparse.Namespace,
) -> bool:
    protocol = summary.get("protocol") or {}
    continuum_case = ((summary.get("continuum_cases") or {}).get(case.hex_order) or {})
    mesh = runtime_manifest.get("mesh") or {}
    if not continuum_case.get("completed_successfully", False):
        return False
    return (
        protocol.get("amplitudes_mm") == parse_csv_doubles(case.amplitudes_mm)
        and int(protocol.get("steps_per_segment", -1)) == case.steps_per_segment
        and int(protocol.get("continuation_segment_substep_factor", -1))
        == case.continuation_segment_substep_factor
        and str(protocol.get("continuation", "")) == case.continuation
        and safe_float(protocol.get("axial_compression_mn")) == args.axial_compression_mn
        and int(protocol.get("axial_preload_steps", -1)) == args.axial_preload_steps
        and str(runtime_manifest.get("hex_order", "")).lower() == case.hex_order
        and int(mesh.get("nx", -1)) == case.nx
        and int(mesh.get("ny", -1)) == case.ny
        and int(mesh.get("nz", -1)) == case.nz
        and str(runtime_manifest.get("solver_policy_kind", ""))
        == solver_policy_manifest_key(case.solver_policy)
        and str(runtime_manifest.get("continuation_kind", ""))
        == continuation_manifest_key(case.continuation)
    )


def find_equivalent_cached_bundle(
    cache_root: Path,
    current_output_root: Path,
    case: AuditCase,
    args: argparse.Namespace,
) -> Path | None:
    if not cache_root.exists():
        return None
    for summary_path in cache_root.rglob("structural_continuum_steel_hysteresis_summary.json"):
        bundle_dir = summary_path.parent
        if current_output_root in bundle_dir.parents:
            continue
        runtime_manifest_path = bundle_dir / case.hex_order / "runtime_manifest.json"
        if not runtime_manifest_path.exists():
            continue
        try:
            summary = read_json(summary_path)
            runtime_manifest = read_json(runtime_manifest_path)
        except json.JSONDecodeError:
            continue
        if matches_equivalent_bundle(summary, runtime_manifest, case, args):
            return bundle_dir
    return None


def build_worker_command(
    worker_script: Path,
    structural_bundle_dir: Path,
    output_dir: Path,
    case: AuditCase,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        *parse_launcher(args.python_launcher),
        str(worker_script),
        "--output-dir",
        str(output_dir),
        "--structural-bundle-dir",
        str(structural_bundle_dir),
        "--beam-nodes",
        str(args.beam_nodes),
        "--beam-integration",
        args.beam_integration,
        "--hex-orders",
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
        "--continuation-segment-substep-factor",
        str(case.continuation_segment_substep_factor),
        "--amplitudes-mm",
        case.amplitudes_mm,
        "--steps-per-segment",
        str(case.steps_per_segment),
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--skip-figure-export",
    ]
    if args.reuse_existing:
        command.append("--reuse-existing")
    if args.print_progress:
        command.append("--print-progress")
    return command


def run_case(
    worker_script: Path,
    structural_bundle_dir: Path,
    root: Path,
    case: AuditCase,
    args: argparse.Namespace,
) -> AuditRow:
    case_dir = root / case.key
    ensure_dir(case_dir)
    summary_path = case_dir / "structural_continuum_steel_hysteresis_summary.json"
    if args.reuse_existing and summary_path.exists():
        summary = read_json(summary_path)
        return row_from_summary(case, case_dir, summary, status="completed", timed_out=False)

    cached_bundle = find_equivalent_cached_bundle(
        args.cache_root.resolve(),
        root.resolve(),
        case,
        args,
    )
    if args.reuse_existing and cached_bundle is not None:
        shutil.copytree(cached_bundle, case_dir, dirs_exist_ok=True)
        summary = read_json(summary_path)
        return row_from_summary(
            case,
            case_dir,
            summary,
            status="reused_equivalent_bundle",
            timed_out=False,
        )

    command = build_worker_command(
        worker_script, structural_bundle_dir, case_dir, case, args
    )
    start = time.perf_counter()
    try:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        proc = subprocess.Popen(
            command,
            cwd=str(root.parent.parent.parent.parent),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        stdout, stderr = proc.communicate(timeout=case.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        terminate_process_tree(proc)
        stdout, stderr = proc.communicate()
        (case_dir / "wrapper_stdout.log").write_text(
            (exc.stdout or "") + (stdout or ""),
            encoding="utf-8",
        )
        (case_dir / "wrapper_stderr.log").write_text(
            (exc.stderr or "") + (stderr or ""),
            encoding="utf-8",
        )
        delayed_summary_path = case_dir / "structural_continuum_steel_hysteresis_summary.json"
        if delayed_summary_path.exists():
            summary = read_json(delayed_summary_path)
            return row_from_summary(
                case,
                case_dir,
                summary,
                status="completed_beyond_timeout_budget",
                timed_out=True,
            )
        runtime_manifest_path = case_dir / case.hex_order / "runtime_manifest.json"
        if runtime_manifest_path.exists():
            runtime_manifest = read_json(runtime_manifest_path)
            completed = bool(runtime_manifest.get("completed_successfully", False))
            return row_from_runtime_manifest(
                case,
                case_dir,
                runtime_manifest,
                status=(
                    "continuum_failed_before_trace_export"
                    if not completed
                    else "continuum_completed_beyond_timeout_budget"
                ),
                timed_out=not completed,
                process_wall_seconds=time.perf_counter() - start,
            )
        return AuditRow(
            key=case.key,
            group=case.group,
            hex_order=case.hex_order,
            mesh=f"{case.nx}x{case.ny}x{case.nz}",
            solver_policy=case.solver_policy,
            continuation=case.continuation,
            status="timeout_budget_exceeded",
            completed_successfully=False,
            timed_out=True,
            process_wall_seconds=time.perf_counter() - start,
            reported_total_wall_seconds=None,
            reported_solve_wall_seconds=None,
            global_max_rel_base_shear_error=None,
            global_rms_rel_base_shear_error=None,
            steel_max_rel_stress_error=None,
            steel_rms_rel_stress_error=None,
            steel_max_rel_strain_error=None,
            steel_rms_rel_strain_error=None,
            selected_structural_fiber_index=None,
            selected_structural_fiber_y=None,
            selected_structural_fiber_z=None,
            selected_bar_index=None,
            selected_bar_y=None,
            selected_bar_z=None,
            output_dir=str(case_dir),
        )

    elapsed = time.perf_counter() - start
    (case_dir / "wrapper_stdout.log").write_text(stdout, encoding="utf-8")
    (case_dir / "wrapper_stderr.log").write_text(stderr, encoding="utf-8")
    if proc.returncode != 0 or not summary_path.exists():
        runtime_manifest_path = case_dir / case.hex_order / "runtime_manifest.json"
        if runtime_manifest_path.exists():
            runtime_manifest = read_json(runtime_manifest_path)
            return row_from_runtime_manifest(
                case,
                case_dir,
                runtime_manifest,
                status="continuum_failed_before_trace_export",
                timed_out=False,
                process_wall_seconds=elapsed,
            )
        return AuditRow(
            key=case.key,
            group=case.group,
            hex_order=case.hex_order,
            mesh=f"{case.nx}x{case.ny}x{case.nz}",
            solver_policy=case.solver_policy,
            continuation=case.continuation,
            status="runner_failed",
            completed_successfully=False,
            timed_out=False,
            process_wall_seconds=elapsed,
            reported_total_wall_seconds=None,
            reported_solve_wall_seconds=None,
            global_max_rel_base_shear_error=None,
            global_rms_rel_base_shear_error=None,
            steel_max_rel_stress_error=None,
            steel_rms_rel_stress_error=None,
            steel_max_rel_strain_error=None,
            steel_rms_rel_strain_error=None,
            selected_structural_fiber_index=None,
            selected_structural_fiber_y=None,
            selected_structural_fiber_z=None,
            selected_bar_index=None,
            selected_bar_y=None,
            selected_bar_z=None,
            output_dir=str(case_dir),
        )

    summary = read_json(summary_path)
    row = row_from_summary(case, case_dir, summary, status="completed", timed_out=False)
    return AuditRow(
        **{
            **asdict(row),
            "process_wall_seconds": row.process_wall_seconds
            if row.process_wall_seconds is not None
            else elapsed,
        }
    )


def row_from_runtime_manifest(
    case: AuditCase,
    case_dir: Path,
    runtime_manifest: dict[str, object],
    status: str,
    timed_out: bool,
    process_wall_seconds: float | None,
) -> AuditRow:
    mesh = runtime_manifest.get("mesh") or {}
    return AuditRow(
        key=case.key,
        group=case.group,
        hex_order=case.hex_order,
        mesh=f"{int(mesh.get('nx', case.nx))}x{int(mesh.get('ny', case.ny))}x{int(mesh.get('nz', case.nz))}",
        solver_policy=case.solver_policy,
        continuation=case.continuation,
        status=status,
        completed_successfully=bool(runtime_manifest.get("completed_successfully", False)),
        timed_out=timed_out,
        process_wall_seconds=process_wall_seconds,
        reported_total_wall_seconds=safe_float(
            ((runtime_manifest.get("timing") or {}).get("total_wall_seconds"))
        ),
        reported_solve_wall_seconds=safe_float(
            ((runtime_manifest.get("timing") or {}).get("solve_wall_seconds"))
        ),
        global_max_rel_base_shear_error=None,
        global_rms_rel_base_shear_error=None,
        steel_max_rel_stress_error=None,
        steel_rms_rel_stress_error=None,
        steel_max_rel_strain_error=None,
        steel_rms_rel_strain_error=None,
        selected_structural_fiber_index=None,
        selected_structural_fiber_y=None,
        selected_structural_fiber_z=None,
        selected_bar_index=None,
        selected_bar_y=None,
        selected_bar_z=None,
        output_dir=str(case_dir),
    )


def row_from_summary(
    case: AuditCase,
    case_dir: Path,
    summary: dict[str, object],
    status: str,
    timed_out: bool,
) -> AuditRow:
    continuum_case = ((summary.get("continuum_cases") or {}).get(case.hex_order) or {})
    global_case = ((summary.get("global_comparison") or {}).get(case.hex_order) or {})
    steel_case = ((summary.get("steel_local_comparison") or {}).get(case.hex_order) or {})
    selected_structural_fiber = continuum_case.get("selected_structural_fiber") or {}
    selected_bar = continuum_case.get("selected_bar") or {}
    return AuditRow(
        key=case.key,
        group=case.group,
        hex_order=case.hex_order,
        mesh=f"{case.nx}x{case.ny}x{case.nz}",
        solver_policy=case.solver_policy,
        continuation=case.continuation,
        status=status,
        completed_successfully=bool(continuum_case.get("completed_successfully", False)),
        timed_out=timed_out,
        process_wall_seconds=safe_float(continuum_case.get("process_wall_seconds")),
        reported_total_wall_seconds=safe_float(
            continuum_case.get("reported_total_wall_seconds")
        ),
        reported_solve_wall_seconds=safe_float(
            continuum_case.get("reported_solve_wall_seconds")
        ),
        global_max_rel_base_shear_error=safe_float(
            global_case.get("max_rel_base_shear_error")
        ),
        global_rms_rel_base_shear_error=safe_float(
            global_case.get("rms_rel_base_shear_error")
        ),
        steel_max_rel_stress_error=safe_float(
            steel_case.get("max_rel_stress_vs_drift_error")
        ),
        steel_rms_rel_stress_error=safe_float(
            steel_case.get("rms_rel_stress_vs_drift_error")
        ),
        steel_max_rel_strain_error=safe_float(
            steel_case.get("max_rel_strain_vs_drift_error")
        ),
        steel_rms_rel_strain_error=safe_float(
            steel_case.get("rms_rel_strain_vs_drift_error")
        ),
        selected_structural_fiber_index=(
            int(selected_structural_fiber["fiber_index"])
            if "fiber_index" in selected_structural_fiber
            else None
        ),
        selected_structural_fiber_y=safe_float(selected_structural_fiber.get("y")),
        selected_structural_fiber_z=safe_float(selected_structural_fiber.get("z")),
        selected_bar_index=(
            int(selected_bar["bar_index"]) if "bar_index" in selected_bar else None
        ),
        selected_bar_y=safe_float(selected_bar.get("bar_y")),
        selected_bar_z=safe_float(selected_bar.get("bar_z")),
        output_dir=str(case_dir),
    )


def save(fig: plt.Figure, out_dirs: list[Path], stem: str) -> None:
    for out_dir in out_dirs:
        ensure_dir(out_dir)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def plot_mesh_refinement(
    rows: list[AuditRow],
    out_dirs: list[Path],
    structural_bundle_dir: Path,
) -> None:
    mesh_rows = [row for row in rows if row.group == "mesh_refinement"]
    completed = [row for row in mesh_rows if row.completed_successfully]
    if not completed:
        return

    structural_hist = read_csv_rows(structural_bundle_dir / "hysteresis.csv")
    palette = {
        "2x2x2": BLUE,
        "3x3x2": ORANGE,
        "2x2x4": RED,
    }

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(
        [1.0e3 * float(row["drift_m"]) for row in structural_hist],
        [1.0e3 * float(row["base_shear_MN"]) for row in structural_hist],
        color="black",
        linewidth=1.8,
        label="Structural reference",
    )
    for row in completed:
        hist = read_csv_rows(Path(row.output_dir) / row.hex_order / "hysteresis.csv")
        ax.plot(
            [1.0e3 * float(item["drift_m"]) for item in hist],
            [1.0e3 * float(item["base_shear_MN"]) for item in hist],
            color=palette.get(row.mesh, "#666666"),
            linewidth=1.4,
            linestyle="--",
            label=f"Continuum {row.mesh}",
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Continuum mesh refinement vs structural hysteresis")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dirs, "reduced_rc_structural_continuum_mesh_refinement_hysteresis")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    x = range(len(mesh_rows))
    ax.bar(
        x,
        [
            row.steel_rms_rel_stress_error if row.steel_rms_rel_stress_error is not None else math.nan
            for row in mesh_rows
        ],
        color=[palette.get(row.mesh, "#666666") for row in mesh_rows],
    )
    ax.set_xticks(list(x), [row.mesh for row in mesh_rows], rotation=15)
    ax.set_ylabel(r"RMS relative steel-stress error")
    ax.set_title("Continuum mesh refinement: local steel mismatch")
    fig.tight_layout()
    save(fig, out_dirs, "reduced_rc_structural_continuum_mesh_refinement_steel_error")


def plot_solver_policies(rows: list[AuditRow], out_dirs: list[Path]) -> None:
    solver_rows = [row for row in rows if row.group == "solver_policy"]
    if not solver_rows:
        return

    def label(row: AuditRow) -> str:
        if row.continuation == "reversal-guarded":
            return row.solver_policy
        return f"{row.solver_policy} + {row.continuation}"

    colors = {
        "canonical-cascade": BLUE,
        "newton-l2-only": ORANGE,
        "newton-l2-lu-symbolic-reuse-only": "#0891b2",
        "newton-l2-gmres-ilu1-only": "#7c3aed",
        "newton-trust-region-dogleg-only": GREEN,
        "anderson-only": PURPLE,
        "nonlinear-cg-only": "#a16207",
        "nonlinear-gmres-only": RED,
        "newton-l2-only + arc-length": "#0f766e",
    }

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = range(len(solver_rows))
    ax.bar(
        x,
        [
            row.reported_total_wall_seconds
            if row.reported_total_wall_seconds is not None
            else math.nan
            for row in solver_rows
        ],
        color=[colors.get(label(row), colors.get(row.solver_policy, "#666666")) for row in solver_rows],
    )
    ax.set_xticks(list(x), [label(row) for row in solver_rows], rotation=30, ha="right")
    ax.set_ylabel("Reported total wall time [s]")
    ax.set_title("Continuum solver-policy timing on Hex20 2x2x2")
    fig.tight_layout()
    save(fig, out_dirs, "reduced_rc_structural_continuum_solver_policy_timing")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.bar(
        x,
        [
            row.steel_rms_rel_stress_error
            if row.steel_rms_rel_stress_error is not None
            else math.nan
            for row in solver_rows
        ],
        color=[colors.get(label(row), colors.get(row.solver_policy, "#666666")) for row in solver_rows],
    )
    ax.set_xticks(list(x), [label(row) for row in solver_rows], rotation=30, ha="right")
    ax.set_ylabel(r"RMS relative steel-stress error")
    ax.set_title("Continuum solver-policy effect on local steel bridge")
    fig.tight_layout()
    save(fig, out_dirs, "reduced_rc_structural_continuum_solver_policy_steel_error")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    out_dirs = [args.figures_dir.resolve(), args.secondary_figures_dir.resolve()]

    cases = build_cases()
    rows: list[AuditRow] = []
    worker_script = args.worker_script.resolve()
    structural_bundle_dir = args.structural_bundle_dir.resolve()
    for case in cases:
        row = run_case(worker_script, structural_bundle_dir, output_dir, case, args)
        rows.append(row)

    rows_dict = [asdict(row) for row in rows]
    write_csv(output_dir / "structural_continuum_refinement_solver_audit_cases.csv", rows_dict)

    summary = {
        "structural_bundle_dir": str(structural_bundle_dir),
        "cases": rows_dict,
        "mesh_refinement": {
            "completed_case_count": sum(
                1 for row in rows if row.group == "mesh_refinement" and row.completed_successfully
            ),
            "timed_out_case_count": sum(
                1 for row in rows if row.group == "mesh_refinement" and row.timed_out
            ),
        },
        "solver_policy": {
            "completed_case_count": sum(
                1 for row in rows if row.group == "solver_policy" and row.completed_successfully
            ),
            "failed_case_count": sum(
                1
                for row in rows
                if row.group == "solver_policy" and not row.completed_successfully
            ),
        },
        "artifacts": {
            "mesh_refinement_hysteresis": str(
                args.figures_dir
                / "reduced_rc_structural_continuum_mesh_refinement_hysteresis.png"
            ),
            "mesh_refinement_steel_error": str(
                args.figures_dir
                / "reduced_rc_structural_continuum_mesh_refinement_steel_error.png"
            ),
            "solver_policy_timing": str(
                args.figures_dir
                / "reduced_rc_structural_continuum_solver_policy_timing.png"
            ),
            "solver_policy_steel_error": str(
                args.figures_dir
                / "reduced_rc_structural_continuum_solver_policy_steel_error.png"
            ),
        },
    }
    write_json(output_dir / "structural_continuum_refinement_solver_audit_summary.json", summary)

    plot_mesh_refinement(rows, out_dirs, structural_bundle_dir)
    plot_solver_policies(rows, out_dirs)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
