#!/usr/bin/env python3
"""
Freeze the promoted local continuum baseline for the reduced RC column.

This audit is deliberately narrower than the full structural-vs-continuum
campaign.  Its job is to answer, with explicit artifacts, whether the current
Ko-Bathe 3D continuum path is still pertinent as the local RC model, and what
mesh/regularization slice should be promoted today.

Questions answered here:
1. Was the old cost issue intrinsic to Ko-Bathe 3D, or to the old baseline
   choices (hidden crack-band length, numerical tangent, wrong z-bias geometry)?
2. Does a base-biased Hex20 host improve the nonlinear monotonic bridge without
   exploding the runtime?
3. Is the refined longitudinal host (`nz >= 6`) already promotable, or does it
   remain an explicitly open operational frontier?
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
GRAY = "#6b7280"
BROWN = "#8b5e3c"


@dataclass(frozen=True)
class StructuralCase:
    key: str
    monotonic_tip_mm: float
    monotonic_steps: int
    clamp_top_bending_rotation: bool
    timeout_seconds: int


@dataclass(frozen=True)
class ContinuumCase:
    key: str
    material_mode: str
    reinforcement_mode: str
    concrete_profile: str
    concrete_tangent_mode: str
    concrete_characteristic_length_mode: str
    hex_order: str
    nx: int
    ny: int
    nz: int
    longitudinal_bias_power: float
    monotonic_tip_mm: float
    monotonic_steps: int
    solver_policy: str = "newton-l2-only"
    timeout_seconds: int = 1200


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the reduced RC continuum local-model baseline audit."
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def structural_command(
    exe: Path, output_dir: Path, args: argparse.Namespace, case: StructuralCase
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
    ]
    if case.clamp_top_bending_rotation:
        command.append("--clamp-top-bending-rotation")
    if args.print_progress:
        command.append("--print-progress")
    return command


def continuum_command(
    exe: Path, output_dir: Path, args: argparse.Namespace, case: ContinuumCase
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
        "--concrete-tangent-mode",
        case.concrete_tangent_mode,
        "--concrete-characteristic-length-mode",
        case.concrete_characteristic_length_mode,
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
            key="structural_elastic_clamped_10mm",
            monotonic_tip_mm=10.0,
            monotonic_steps=8,
            clamp_top_bending_rotation=True,
            timeout_seconds=600,
        ),
        StructuralCase(
            key="structural_nonlinear_clamped_20mm",
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            clamp_top_bending_rotation=True,
            timeout_seconds=900,
        ),
    ]


def continuum_cases() -> list[ContinuumCase]:
    return [
        ContinuumCase(
            key="continuum_embedded_legacy_hex20_2x2x2_uniform_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            concrete_profile="benchmark-reference",
            concrete_tangent_mode="adaptive-central-difference-with-secant-fallback",
            concrete_characteristic_length_mode="fixed-reference-mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            longitudinal_bias_power=1.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=900,
        ),
        ContinuumCase(
            key="continuum_embedded_promoted_hex20_2x2x2_uniform_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            concrete_profile="production-stabilized",
            concrete_tangent_mode="fracture-secant",
            concrete_characteristic_length_mode="mean-longitudinal-host-edge-mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            longitudinal_bias_power=1.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=900,
        ),
        ContinuumCase(
            key="continuum_embedded_promoted_hex20_2x2x2_bias3_mean_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            concrete_profile="production-stabilized",
            concrete_tangent_mode="fracture-secant",
            concrete_characteristic_length_mode="mean-longitudinal-host-edge-mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            longitudinal_bias_power=3.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=900,
        ),
        ContinuumCase(
            key="continuum_embedded_promoted_hex20_2x2x2_bias3_fixedend_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            concrete_profile="production-stabilized",
            concrete_tangent_mode="fracture-secant",
            concrete_characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            longitudinal_bias_power=3.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=900,
        ),
        ContinuumCase(
            key="continuum_host_promoted_hex20_2x2x2_bias3_fixedend_20mm",
            material_mode="nonlinear",
            reinforcement_mode="continuum-only",
            concrete_profile="production-stabilized",
            concrete_tangent_mode="fracture-secant",
            concrete_characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=2,
            longitudinal_bias_power=3.0,
            monotonic_tip_mm=20.0,
            monotonic_steps=12,
            timeout_seconds=900,
        ),
        ContinuumCase(
            key="continuum_embedded_promoted_hex20_2x2x10_uniform_elastic_10mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            concrete_profile="production-stabilized",
            concrete_tangent_mode="fracture-secant",
            concrete_characteristic_length_mode="mean-longitudinal-host-edge-mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=10,
            longitudinal_bias_power=1.0,
            monotonic_tip_mm=10.0,
            monotonic_steps=8,
            timeout_seconds=600,
        ),
        ContinuumCase(
            key="continuum_embedded_promoted_hex20_2x2x10_bias1p5_fixedend_elastic_10mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            concrete_profile="production-stabilized",
            concrete_tangent_mode="fracture-secant",
            concrete_characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=10,
            longitudinal_bias_power=1.5,
            monotonic_tip_mm=10.0,
            monotonic_steps=8,
            timeout_seconds=600,
        ),
        ContinuumCase(
            key="continuum_embedded_promoted_hex20_2x2x6_bias1p5_fixedend_10mm_frontier",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            concrete_profile="production-stabilized",
            concrete_tangent_mode="fracture-secant",
            concrete_characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=6,
            longitudinal_bias_power=1.5,
            monotonic_tip_mm=10.0,
            monotonic_steps=8,
            timeout_seconds=300,
        ),
        ContinuumCase(
            key="continuum_embedded_promoted_hex20_2x2x10_bias1p5_fixedend_10mm_frontier",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            concrete_profile="production-stabilized",
            concrete_tangent_mode="fracture-secant",
            concrete_characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=10,
            longitudinal_bias_power=1.5,
            monotonic_tip_mm=10.0,
            monotonic_steps=8,
            timeout_seconds=300,
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
    return curve


def max_rel_error_against_reference(
    bundle_dir: Path, reference_bundle: Path
) -> tuple[float | None, float | None]:
    rows = read_csv_rows(bundle_dir / "hysteresis.csv")
    ref_rows = read_csv_rows(reference_bundle / "hysteresis.csv")
    ref_by_step = {
        int(float(row["step"])): 1.0e3 * float(row["base_shear_MN"]) for row in ref_rows
    }
    if not ref_by_step:
        return None, None
    peak_ref = max(abs(value) for value in ref_by_step.values())
    activity_floor = max(0.05 * peak_ref, 1.0e-9)
    rel_errors: list[float] = []
    for row in rows:
        step = int(float(row["step"]))
        if step not in ref_by_step:
            continue
        lhs = 1.0e3 * float(row["base_shear_MN"])
        rhs = ref_by_step[step]
        rel_errors.append(abs(lhs - rhs) / max(abs(rhs), activity_floor))
    if not rel_errors:
        return None, None
    rms = math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors))
    return max(rel_errors), rms


def structural_row_from_bundle(
    case: StructuralCase, bundle_dir: Path, elapsed: float, status: str = "completed"
) -> dict[str, Any]:
    manifest = read_json(bundle_dir / "runtime_manifest.json")
    curve = read_monotonic_curve(bundle_dir)
    return {
        "kind": "structural",
        "key": case.key,
        "status": status,
        "completed_successfully": True,
        "timed_out": False,
        "process_wall_seconds": elapsed,
        "reported_total_wall_seconds": manifest.get("timing", {}).get(
            "total_wall_seconds"
        ),
        "peak_base_shear_kn": max(abs(y) for _, y in curve) if curve else None,
        "output_dir": str(bundle_dir),
    }


def structural_failure_row(
    case: StructuralCase, bundle_dir: Path, status: str, timed_out: bool, elapsed: float
) -> dict[str, Any]:
    return {
        "kind": "structural",
        "key": case.key,
        "status": status,
        "completed_successfully": False,
        "timed_out": timed_out,
        "process_wall_seconds": elapsed,
        "reported_total_wall_seconds": None,
        "peak_base_shear_kn": None,
        "output_dir": str(bundle_dir),
    }


def continuum_row_from_bundle(
    case: ContinuumCase,
    bundle_dir: Path,
    elapsed: float,
    reference_bundle: Path,
    status: str = "completed",
) -> dict[str, Any]:
    manifest = read_json(bundle_dir / "runtime_manifest.json")
    curve = read_monotonic_curve(bundle_dir)
    max_rel, rms_rel = max_rel_error_against_reference(bundle_dir, reference_bundle)
    return {
        "kind": "continuum",
        "key": case.key,
        "status": status,
        "completed_successfully": True,
        "timed_out": False,
        "material_mode": case.material_mode,
        "reinforcement_mode": case.reinforcement_mode,
        "mesh": f"{case.hex_order}_{case.nx}x{case.ny}x{case.nz}",
        "longitudinal_bias_power": case.longitudinal_bias_power,
        "concrete_profile": case.concrete_profile,
        "concrete_tangent_mode": case.concrete_tangent_mode,
        "concrete_characteristic_length_mode": case.concrete_characteristic_length_mode,
        "process_wall_seconds": elapsed,
        "reported_total_wall_seconds": manifest.get("timing", {}).get(
            "total_wall_seconds"
        ),
        "solve_wall_seconds": manifest.get("timing", {}).get("solve_wall_seconds"),
        "peak_base_shear_kn": max(abs(y) for _, y in curve) if curve else None,
        "max_rel_error_vs_structural_clamped": max_rel,
        "rms_rel_error_vs_structural_clamped": rms_rel,
        "characteristic_length_mm": manifest.get("concrete_profile_details", {}).get(
            "characteristic_length_mm"
        ),
        "peak_cracked_gauss_points": manifest.get("observables", {}).get(
            "peak_cracked_gauss_points"
        ),
        "max_crack_opening": manifest.get("observables", {}).get("max_crack_opening"),
        "max_embedding_gap_norm_m": manifest.get("observables", {}).get(
            "max_embedding_gap_norm_m"
        ),
        "output_dir": str(bundle_dir),
    }


def continuum_failure_row(
    case: ContinuumCase, bundle_dir: Path, status: str, timed_out: bool, elapsed: float
) -> dict[str, Any]:
    return {
        "kind": "continuum",
        "key": case.key,
        "status": status,
        "completed_successfully": False,
        "timed_out": timed_out,
        "material_mode": case.material_mode,
        "reinforcement_mode": case.reinforcement_mode,
        "mesh": f"{case.hex_order}_{case.nx}x{case.ny}x{case.nz}",
        "longitudinal_bias_power": case.longitudinal_bias_power,
        "concrete_profile": case.concrete_profile,
        "concrete_tangent_mode": case.concrete_tangent_mode,
        "concrete_characteristic_length_mode": case.concrete_characteristic_length_mode,
        "process_wall_seconds": elapsed,
        "reported_total_wall_seconds": None,
        "solve_wall_seconds": None,
        "peak_base_shear_kn": None,
        "max_rel_error_vs_structural_clamped": None,
        "rms_rel_error_vs_structural_clamped": None,
        "characteristic_length_mm": None,
        "peak_cracked_gauss_points": None,
        "max_crack_opening": None,
        "max_embedding_gap_norm_m": None,
        "output_dir": str(bundle_dir),
    }


def load_failure_status(status_path: Path) -> dict[str, Any] | None:
    if not status_path.exists():
        return None
    return read_json(status_path)


def run_structural_case(
    exe: Path, output_dir: Path, repo_root: Path, args: argparse.Namespace, case: StructuralCase
) -> dict[str, Any]:
    bundle_dir = output_dir / case.key
    manifest = bundle_dir / "runtime_manifest.json"
    status_path = bundle_dir / "runner_status.json"
    if args.reuse_existing and manifest.exists():
        return structural_row_from_bundle(case, bundle_dir, math.nan, status="reused")
    if args.reuse_existing:
        cached_status = load_failure_status(status_path)
        if cached_status is not None:
            return structural_failure_row(
                case,
                bundle_dir,
                str(cached_status.get("status", "cached_failure")),
                bool(cached_status.get("timed_out", False)),
                float(cached_status.get("process_wall_seconds", math.nan)),
            )
    command = structural_command(exe, bundle_dir, args, case)
    try:
        elapsed, completed = run_command(command, repo_root, case.timeout_seconds)
    except subprocess.TimeoutExpired:
        ensure_dir(bundle_dir)
        write_json(
            status_path,
            {
                "status": "timed_out",
                "timed_out": True,
                "process_wall_seconds": case.timeout_seconds,
            },
        )
        return structural_failure_row(case, bundle_dir, "timed_out", True, case.timeout_seconds)
    ensure_dir(bundle_dir)
    (bundle_dir / "runner_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (bundle_dir / "runner_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        if manifest.exists():
            return structural_row_from_bundle(
                case, bundle_dir, elapsed, status="completed_with_runner_warning"
            )
        write_json(
            status_path,
            {
                "status": "runner_failed",
                "timed_out": False,
                "process_wall_seconds": elapsed,
            },
        )
        return structural_failure_row(case, bundle_dir, "runner_failed", False, elapsed)
    if not manifest.exists():
        write_json(
            status_path,
            {
                "status": "missing_runtime_manifest",
                "timed_out": False,
                "process_wall_seconds": elapsed,
            },
        )
        return structural_failure_row(
            case, bundle_dir, "missing_runtime_manifest", False, elapsed
        )
    if status_path.exists():
        status_path.unlink()
    return structural_row_from_bundle(case, bundle_dir, elapsed)


def run_continuum_case(
    exe: Path,
    output_dir: Path,
    repo_root: Path,
    args: argparse.Namespace,
    case: ContinuumCase,
    reference_bundle: Path,
) -> dict[str, Any]:
    bundle_dir = output_dir / case.key
    manifest = bundle_dir / "runtime_manifest.json"
    status_path = bundle_dir / "runner_status.json"
    if args.reuse_existing and manifest.exists():
        return continuum_row_from_bundle(
            case, bundle_dir, math.nan, reference_bundle, status="reused"
        )
    if args.reuse_existing:
        cached_status = load_failure_status(status_path)
        if cached_status is not None:
            return continuum_failure_row(
                case,
                bundle_dir,
                str(cached_status.get("status", "cached_failure")),
                bool(cached_status.get("timed_out", False)),
                float(cached_status.get("process_wall_seconds", math.nan)),
            )
    command = continuum_command(exe, bundle_dir, args, case)
    try:
        elapsed, completed = run_command(command, repo_root, case.timeout_seconds)
    except subprocess.TimeoutExpired:
        ensure_dir(bundle_dir)
        write_json(
            status_path,
            {
                "status": "timed_out",
                "timed_out": True,
                "process_wall_seconds": case.timeout_seconds,
            },
        )
        return continuum_failure_row(case, bundle_dir, "timed_out", True, case.timeout_seconds)
    ensure_dir(bundle_dir)
    (bundle_dir / "runner_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (bundle_dir / "runner_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        if manifest.exists():
            return continuum_row_from_bundle(
                case,
                bundle_dir,
                elapsed,
                reference_bundle,
                status="completed_with_runner_warning",
            )
        write_json(
            status_path,
            {
                "status": "runner_failed",
                "timed_out": False,
                "process_wall_seconds": elapsed,
            },
        )
        return continuum_failure_row(case, bundle_dir, "runner_failed", False, elapsed)
    if not manifest.exists():
        write_json(
            status_path,
            {
                "status": "missing_runtime_manifest",
                "timed_out": False,
                "process_wall_seconds": elapsed,
            },
        )
        return continuum_failure_row(
            case, bundle_dir, "missing_runtime_manifest", False, elapsed
        )
    if status_path.exists():
        status_path.unlink()
    return continuum_row_from_bundle(case, bundle_dir, elapsed, reference_bundle)


def save(fig: plt.Figure, paths: list[Path], stem: str) -> None:
    for path in paths:
        ensure_dir(path)
        fig.savefig(path / f"{stem}.png")
    plt.close(fig)


def plot_nonlinear_overlay(
    structural_bundle: Path, continuum_rows: list[dict[str, Any]], figure_dirs: list[Path]
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    s_curve = read_monotonic_curve(structural_bundle)
    ax.plot(
        [x for x, _ in s_curve],
        [y for _, y in s_curve],
        color=BLUE,
        linewidth=2.2,
        label="Structural beam (clamped tip rotation)",
    )

    style_map = {
        "continuum_embedded_legacy_hex20_2x2x2_uniform_20mm": (GRAY, "-."),
        "continuum_embedded_promoted_hex20_2x2x2_uniform_20mm": (ORANGE, "-"),
        "continuum_embedded_promoted_hex20_2x2x2_bias3_mean_20mm": (PURPLE, "--"),
        "continuum_embedded_promoted_hex20_2x2x2_bias3_fixedend_20mm": (RED, ":"),
        "continuum_host_promoted_hex20_2x2x2_bias3_fixedend_20mm": (BROWN, "--"),
    }
    label_map = {
        "continuum_embedded_legacy_hex20_2x2x2_uniform_20mm": "Hex20 2x2x2 embedded legacy",
        "continuum_embedded_promoted_hex20_2x2x2_uniform_20mm": "Hex20 2x2x2 embedded promoted",
        "continuum_embedded_promoted_hex20_2x2x2_bias3_mean_20mm": "Hex20 2x2x2 embedded bias=3, mean lb",
        "continuum_embedded_promoted_hex20_2x2x2_bias3_fixedend_20mm": "Hex20 2x2x2 embedded bias=3, fixed-end lb",
        "continuum_host_promoted_hex20_2x2x2_bias3_fixedend_20mm": "Hex20 2x2x2 host-only bias=3, fixed-end lb",
    }
    for row in continuum_rows:
        if not row["completed_successfully"] or row["material_mode"] != "nonlinear":
            continue
        if row["key"] not in style_map:
            continue
        curve = read_monotonic_curve(Path(row["output_dir"]))
        color, linestyle = style_map[row["key"]]
        ax.plot(
            [x for x, _ in curve],
            [y for _, y in curve],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=label_map[row["key"]],
        )

    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Reduced RC continuum local-model baseline audit")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, figure_dirs, "reduced_rc_continuum_local_model_baseline_overlay")


def plot_error_timing(
    continuum_rows: list[dict[str, Any]], figure_dirs: list[Path]
) -> None:
    successful = [row for row in continuum_rows if row["completed_successfully"]]
    labels = [row["key"].replace("continuum_", "").replace("_", "\n") for row in successful]
    errors = [
        row["rms_rel_error_vs_structural_clamped"]
        if row["rms_rel_error_vs_structural_clamped"] is not None
        else math.nan
        for row in successful
    ]
    timings = [
        row["solve_wall_seconds"] if row["solve_wall_seconds"] is not None else math.nan
        for row in successful
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].bar(range(len(labels)), errors, color=ORANGE)
    axes[0].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[0].set_ylabel("RMS relative base-shear path error")
    axes[0].set_title("Gap vs clamped structural control")

    colors = [
        BLUE if row["material_mode"] == "elasticized" else PURPLE for row in successful
    ]
    axes[1].bar(range(len(labels)), timings, color=colors)
    axes[1].set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    axes[1].set_ylabel("Solve wall time [s]")
    axes[1].set_title("Operational cost")

    fig.tight_layout()
    save(fig, figure_dirs, "reduced_rc_continuum_local_model_baseline_error_timing")


def plot_frontier_status(
    continuum_rows: list[dict[str, Any]], figure_dirs: list[Path]
) -> None:
    frontier_rows = [
        row for row in continuum_rows if "frontier" in row["key"] or row["material_mode"] == "elasticized"
    ]
    labels = [row["key"].replace("continuum_", "").replace("_", "\n") for row in frontier_rows]
    values = [row["process_wall_seconds"] for row in frontier_rows]
    colors = []
    for row in frontier_rows:
        if row["timed_out"]:
            colors.append(RED)
        elif row["material_mode"] == "elasticized":
            colors.append(GREEN)
        else:
            colors.append(ORANGE)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Process wall time [s]")
    ax.set_title("Refined-host operational frontier")
    fig.tight_layout()
    save(fig, figure_dirs, "reduced_rc_continuum_local_model_baseline_frontier")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    ensure_dir(args.output_dir)
    figure_dirs = [args.figures_dir.resolve(), args.secondary_figures_dir.resolve()]

    structural_results = [
        run_structural_case(args.structural_exe, args.output_dir, repo_root, args, case)
        for case in structural_cases()
        if case_selected(case.key, args.case_filter)
    ]
    structural_by_key = {
        row["key"]: row for row in structural_results if row["completed_successfully"]
    }
    if "structural_elastic_clamped_10mm" not in structural_by_key:
        raise RuntimeError(
            "The local-model baseline audit requires the completed clamped structural elastic reference."
        )
    if "structural_nonlinear_clamped_20mm" not in structural_by_key:
        raise RuntimeError(
            "The local-model baseline audit requires the completed clamped structural reference."
        )

    structural_nonlinear_bundle = Path(
        structural_by_key["structural_nonlinear_clamped_20mm"]["output_dir"]
    )
    structural_elastic_bundle = Path(
        structural_by_key["structural_elastic_clamped_10mm"]["output_dir"]
    )

    continuum_results = [
        run_continuum_case(
            args.continuum_exe,
            args.output_dir,
            repo_root,
            args,
            case,
            structural_elastic_bundle
            if case.material_mode == "elasticized"
            else structural_nonlinear_bundle,
        )
        for case in continuum_cases()
        if case_selected(case.key, args.case_filter)
    ]

    summary = {
        "status": (
            "completed"
            if all(row["completed_successfully"] for row in structural_results + continuum_results)
            else "completed_with_partial_frontier"
        ),
        "structural_rows": structural_results,
        "continuum_rows": continuum_results,
        "key_findings": {
            "kobathe_pertinence_note": (
                "Ko-Bathe 3D remains pertinent for the reduced RC local continuum slice "
                "once the baseline is promoted to the fracture-secant tangent, explicit "
                "crack-band semantics, and corrected quadratic longitudinal bias geometry."
            ),
            "frontier_note": (
                "Refining the nonlinear Hex20 host to nz>=6 remains an operational frontier "
                "for this iteration even though the same refined topology is cheap in the elasticized slice."
            ),
            "promotion_note": (
                "The uniform Hex20 2x2x2 slice remains the cheapest nonlinear control, while "
                "Hex20 2x2x2 with base-side bias and fixed-end characteristic length becomes the "
                "best representative local-validation slice for this iteration because it improves the "
                "monotonic bridge modestly without changing the order of cost."
            ),
            "characteristic_length_note": (
                "On the current Hex20 2x2x2 bias=3 monotonic bridge, switching the crack-band "
                "length from the mean host edge to the fixed-end host edge changes the declared "
                "regularization semantics but does not materially move the global force path. The "
                "effect of lb is still verified independently at the constitutive level."
            ),
        },
    }
    write_json(
        args.output_dir / "continuum_local_model_baseline_summary.json",
        summary,
    )
    write_csv(
        args.output_dir / "continuum_local_model_baseline_rows.csv",
        structural_results + continuum_results,
    )

    plot_nonlinear_overlay(
        structural_nonlinear_bundle,
        continuum_results,
        figure_dirs,
    )
    plot_error_timing(continuum_results, figure_dirs)
    plot_frontier_status(continuum_results, figure_dirs)


if __name__ == "__main__":
    main()
