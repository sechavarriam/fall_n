#!/usr/bin/env python3
"""Run and compare global XFEM mesh-refinement branches against the structural reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REDUCED_RC_COLUMN_HEIGHT_M = 1.2


@dataclass(frozen=True)
class XFEMCase:
    name: str
    nx: int
    ny: int
    nz: int
    bias_power: float
    bias_location: str


DEFAULT_CASES = (
    XFEMCase("nx1_ny1_nz2_uniform", 1, 1, 2, 1.0, "fixed-end"),
    XFEMCase("nx1_ny1_nz4_uniform", 1, 1, 4, 1.0, "fixed-end"),
    XFEMCase("nx2_ny2_nz2_uniform", 2, 2, 2, 1.0, "fixed-end"),
    XFEMCase("nx1_ny1_nz2_fixed_bias2", 1, 1, 2, 2.0, "fixed-end"),
    XFEMCase("nx1_ny1_nz4_fixed_bias2", 1, 1, 4, 2.0, "fixed-end"),
    XFEMCase("nx2_ny2_nz2_fixed_bias2", 2, 2, 2, 2.0, "fixed-end"),
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run a global XFEM secant mesh-refinement matrix."
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=repo / "build/fall_n_reduced_rc_xfem_reference_benchmark.exe",
    )
    parser.add_argument(
        "--structural-dir",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
        "reboot_structural_reference_n10_lobatto_200mm_for_xfem_secant_compare",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
        "xfem_structural_refinement_matrix_200mm",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo / "doc/figures/validation_reboot",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=repo / "PhD_Thesis/Figuras/validation_reboot",
    )
    parser.add_argument(
        "--matrix-stem",
        default="xfem_global_secant_structural_refinement_matrix_200mm",
        help="Basename used for the matrix figure and JSON summary artifacts.",
    )
    parser.add_argument("--steps-per-segment", type=int, default=8)
    parser.add_argument(
        "--global-xfem-cohesive-tangent",
        default="central-fallback",
        choices=("secant", "central", "central-fallback"),
        help="Algorithmic tangent used by the XFEM cohesive interface.",
    )
    parser.add_argument(
        "--global-xfem-shear-cap-mpa",
        type=float,
        default=0.05,
        help="Tangential cohesive traction cap passed to the XFEM benchmark.",
    )
    parser.add_argument(
        "--global-xfem-kinematic-formulation",
        default="small-strain",
        choices=(
            "small-strain",
            "corotational",
            "total-lagrangian",
            "updated-lagrangian",
        ),
        help="Kinematic policy injected into the shifted-Heaviside solid.",
    )
    parser.add_argument(
        "--allow-guarded-xfem-finite-kinematics",
        action="store_true",
        help=(
            "Forward the benchmark opt-in for guarded TL/UL XFEM audit runs."
        ),
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-rebar-mode",
        default="none",
        help=(
            "Optional localized rebar bridge across the XFEM crack: none, "
            "axial, dowel-x, dowel-y, axial+dowel-x, or axial+dowel-y."
        ),
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-rebar-area-scale",
        type=float,
        default=0.0,
        help=(
            "Area scale for the localized crack-crossing rebar bridge. "
            "The baseline matrix keeps this at zero to avoid implicit "
            "double counting; calibration runs can opt in explicitly."
        ),
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-gauge-length-mm",
        type=float,
        default=100.0,
        help="Gauge length used by the localized crack-crossing bridge.",
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-bridge-law",
        default="material",
        choices=("material", "material-strain", "bounded-slip", "bounded", "dowel-slip"),
        help=(
            "Constitutive law for the localized crack-crossing bridge. "
            "Use bounded-slip to cap dowel/bond transfer by force rather "
            "than by an artificial strain stress."
        ),
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-yield-slip-mm",
        type=float,
        default=0.25,
        help="Yield slip used by the bounded-slip crack-crossing bridge.",
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-yield-force-mn",
        type=float,
        default=None,
        help=(
            "Optional per-bridge yield force for bounded-slip. If omitted, "
            "the executable derives it from the shear cap and section area."
        ),
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-hardening-ratio",
        type=float,
        default=0.0,
        help="Post-yield tangent ratio for the bounded-slip bridge.",
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-force-cap-mn",
        type=float,
        default=None,
        help="Optional per-bridge absolute force cap for bounded-slip.",
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-axis-frame",
        default="fixed-global",
        choices=("fixed-global", "corotational-host"),
        help="Frame used to project the crack-crossing bridge slip.",
    )
    parser.add_argument(
        "--global-xfem-crack-crossing-host-axis-tangent",
        default="frozen",
        choices=("frozen", "finite-difference"),
        help=(
            "Optional directional tangent for the corotational-host bridge "
            "axis. Keep frozen for promoted evidence unless convergence "
            "diagnostics require the extra columns."
        ),
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=900.0,
        help="Maximum wall time per XFEM case before marking it as timed out.",
    )
    parser.add_argument(
        "--global-xfem-solver-profile",
        default="backtracking",
        help=(
            "PETSc nonlinear profile passed to the benchmark. The default is "
            "paired with --global-xfem-solver-cascade for backwards-compatible "
            "matrix runs; use l2 for the promoted guarded mixed-control path."
        ),
    )
    parser.add_argument(
        "--disable-global-xfem-solver-cascade",
        action="store_true",
        help="Do not pass --global-xfem-solver-cascade to the executable.",
    )
    parser.add_argument(
        "--global-xfem-continuation",
        default="fixed-increment",
        choices=(
            "fixed-increment",
            "mixed-arc-length",
            "mixed-control",
            "bordered-fixed-control",
            "bordered-fixed-control-hybrid",
        ),
        help=(
            "Continuation policy used by the global XFEM benchmark. "
            "mixed-arc-length enables the guarded observable-arc driver; "
            "bordered-fixed-control is an experimental PETSc bordered "
            "diagnostic path; bordered-fixed-control-hybrid falls back to "
            "the selected SNES profile after repeated bordered stalls."
        ),
    )
    parser.add_argument(
        "--global-xfem-bordered-hybrid-disable-streak",
        type=int,
        default=3,
        help=(
            "Disable the bordered attempt temporarily after this many "
            "consecutive bordered failures in hybrid mode."
        ),
    )
    parser.add_argument(
        "--global-xfem-bordered-hybrid-retry-interval",
        type=int,
        default=12,
        help="Number of protocol targets to skip before retrying bordered mode.",
    )
    parser.add_argument(
        "--global-xfem-mixed-arc-target",
        type=float,
        default=0.50,
        help="Target mixed observable arc length for mixed-control runs.",
    )
    parser.add_argument(
        "--global-xfem-mixed-arc-reject-factor",
        type=float,
        default=1.50,
        help="Reject factor for guarded mixed observable arc-length runs.",
    )
    parser.add_argument(
        "--global-xfem-mixed-arc-reaction-scale-mn",
        type=float,
        default=0.02,
        help="Base-shear scale used by the mixed-control observable metric.",
    )
    parser.add_argument(
        "--global-xfem-mixed-arc-damage-weight",
        type=float,
        default=0.10,
        help="Damage-observable weight used by the mixed-control metric.",
    )
    parser.add_argument(
        "--crack-z-m",
        type=float,
        default=0.6,
        help=(
            "Physical XFEM crack-plane height. Keeping this fixed separates "
            "mesh convergence from moving-hinge artifacts."
        ),
    )
    parser.add_argument(
        "--column-height-m",
        type=float,
        default=REDUCED_RC_COLUMN_HEIGHT_M,
        help=(
            "Column height used only for script-side crack/mesh sanity checks. "
            "The executable remains the source of truth for the physical model."
        ),
    )
    parser.add_argument(
        "--auto-crack-z",
        action="store_true",
        help="Use the executable default: crack at the first host element midpoint.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only-case",
        action="append",
        default=[],
        help="Run only matching case names; may be passed more than once.",
    )
    parser.add_argument(
        "--include-2x2x4",
        action="store_true",
        help="Also run a heavier 2x2x4 fixed-end biased case.",
    )
    parser.add_argument(
        "--extra-case",
        action="append",
        default=[],
        metavar="NAME:NX:NY:NZ:BIAS:LOCATION",
        help=(
            "Append a custom XFEM mesh case without editing the script. "
            "Example: nx1_ny1_nz8_fixed_bias2:1:1:8:2.0:fixed-end"
        ),
    )
    parser.add_argument(
        "--promotion-rms-limit",
        type=float,
        default=0.10,
        help="Peak-normalized RMS base-shear limit for XFEM promotion.",
    )
    parser.add_argument(
        "--promotion-max-error-limit",
        type=float,
        default=0.30,
        help="Peak-normalized maximum base-shear error limit for XFEM promotion.",
    )
    parser.add_argument(
        "--promotion-peak-ratio-min",
        type=float,
        default=0.90,
        help="Lower bound for XFEM/structural peak base-shear ratio.",
    )
    parser.add_argument(
        "--promotion-peak-ratio-max",
        type=float,
        default=1.15,
        help="Upper bound for XFEM/structural peak base-shear ratio.",
    )
    parser.add_argument(
        "--promotion-max-failures",
        type=int,
        default=0,
        help=(
            "Maximum failed/timed-out matrix cases allowed before the XFEM "
            "branch is considered promotable."
        ),
    )
    return parser.parse_args()


def parse_extra_case(spec: str) -> XFEMCase:
    parts = spec.split(":")
    if len(parts) != 6:
        raise ValueError(
            "--extra-case must have NAME:NX:NY:NZ:BIAS:LOCATION format."
        )
    name, nx, ny, nz, bias, location = parts
    return XFEMCase(
        name=name,
        nx=int(nx),
        ny=int(ny),
        nz=int(nz),
        bias_power=float(bias),
        bias_location=location,
    )


def crack_plane_coincides_with_uniform_mesh(
    args: argparse.Namespace,
    case: XFEMCase,
) -> bool:
    if args.auto_crack_z:
        return False
    if abs(case.bias_power - 1.0) > 1.0e-12:
        return False
    if case.bias_location != "fixed-end":
        return False
    if case.nz <= 0 or args.column_height_m <= 0.0:
        return False
    normalized = args.crack_z_m / args.column_height_m
    nearest_level = round(normalized * case.nz)
    if nearest_level <= 0 or nearest_level >= case.nz:
        return False
    return abs(normalized - nearest_level / case.nz) <= 1.0e-10


def read_csv(path: Path) -> list[dict[str, Any]]:
    def coerce(value: str) -> Any:
        if value == "":
            return math.nan
        try:
            return float(value)
        except ValueError:
            return value

    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_case_label(name: str) -> str:
    return (
        name.replace("nx", "")
        .replace("_ny", "x")
        .replace("_nz", "x")
        .replace("_fixed_bias2", " bias2")
        .replace("_uniform", " uniform")
    )


def interpolate_by_p(rows: list[dict[str, float]], p: float, key: str) -> float:
    ps = [row["p"] for row in rows]
    index = bisect_left(ps, p)
    if index <= 0:
        return rows[0][key]
    if index >= len(rows):
        return rows[-1][key]
    lo = rows[index - 1]
    hi = rows[index]
    span = hi["p"] - lo["p"]
    if abs(span) < 1.0e-15:
        return hi[key]
    t = (p - lo["p"]) / span
    return (1.0 - t) * lo[key] + t * hi[key]


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def peak_abs(values: list[float]) -> float:
    return max(abs(value) for value in values) if values else math.nan


def loop_work(rows: list[dict[str, float]], shear_key: str, drift_key: str) -> float:
    work = 0.0
    for previous, current in zip(rows, rows[1:]):
        work += (
            0.5
            * (previous[shear_key] + current[shear_key])
            * (current[drift_key] - previous[drift_key])
        )
    return work


def passes_xfem_promotion_gate(
    args: argparse.Namespace,
    summary: dict[str, Any],
) -> bool:
    ratio = float(summary["xfem_to_structural_peak_base_shear_ratio"])
    return (
        bool(summary.get("completed_successfully"))
        and float(summary["peak_normalized_rms_base_shear_error"])
        <= args.promotion_rms_limit
        and float(summary["peak_normalized_max_base_shear_error"])
        <= args.promotion_max_error_limit
        and args.promotion_peak_ratio_min
        <= ratio
        <= args.promotion_peak_ratio_max
        and float(summary["peak_abs_steel_stress_MPa"]) >= 420.0
        and float(summary["max_host_damage"]) > 0.0
    )


def make_promotion_gate_summary(
    args: argparse.Namespace,
    summaries: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    passing = [
        summary
        for summary in summaries
        if passes_xfem_promotion_gate(args, summary)
    ]
    matrix_failure_ok = len(failures) <= args.promotion_max_failures
    promoted = bool(passing) and matrix_failure_ok
    best = min(
        summaries,
        key=lambda row: row["peak_normalized_rms_base_shear_error"],
        default=None,
    )
    if promoted:
        diagnosis = (
            "At least one XFEM case satisfies the response gate and the matrix "
            "has no excessive failures; this branch may be promoted to the "
            "next local-model calibration stage."
        )
    elif passing:
        diagnosis = (
            "A response-quality case exists, but the matrix still contains "
            "failed or timed-out cases. Treat XFEM as the primary candidate, "
            "not as a closed multiscale local model."
        )
    else:
        diagnosis = (
            "No case satisfies the full response gate. Continue with tangent, "
            "crack-position, cohesive-energy and shear-transfer calibration "
            "before promoting XFEM."
        )
    return {
        "status": "promoted_to_next_stage" if promoted else "not_promoted",
        "criteria": {
            "max_peak_normalized_rms_base_shear_error": args.promotion_rms_limit,
            "max_peak_normalized_max_base_shear_error": args.promotion_max_error_limit,
            "min_peak_base_shear_ratio": args.promotion_peak_ratio_min,
            "max_peak_base_shear_ratio": args.promotion_peak_ratio_max,
            "min_peak_steel_stress_MPa": 420.0,
            "max_allowed_failure_count": args.promotion_max_failures,
        },
        "matrix_failure_count": len(failures),
        "matrix_failure_ok": matrix_failure_ok,
        "passing_case_count": len(passing),
        "passing_cases": [summary["case"] for summary in passing],
        "best_case_by_rms": best["case"] if best else None,
        "diagnosis": diagnosis,
    }


def run_case(args: argparse.Namespace, case: XFEMCase) -> Path:
    out_dir = args.output_root / case.name
    manifest = out_dir / "global_xfem_newton_manifest.json"
    failure_marker = out_dir / "xfem_case_failure.json"
    if manifest.exists() and not args.force:
        data = read_json(manifest)
        physics = data.get("physics", {})
        crack_matches = True
        if not args.auto_crack_z:
            crack_matches = (
                abs(float(physics.get("crack_z_m", math.nan)) - args.crack_z_m)
                <= 1.0e-10
                and physics.get("crack_z_source")
                == "user_prescribed_physical_position"
            )
        if data.get("completed_successfully") and crack_matches:
            print(f"[skip] {case.name}: completed artifact exists")
            return out_dir
    if failure_marker.exists() and not args.force:
        failure = read_json(failure_marker)
        print(f"[skip] {case.name}: previous {failure.get('status', 'failure')}")
        return out_dir
    if crack_plane_coincides_with_uniform_mesh(args, case):
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "case": case.name,
            "status": "invalid_crack_mesh_coincidence",
            "crack_z_m": args.crack_z_m,
            "column_height_m": args.column_height_m,
            "nz": case.nz,
            "bias_power": case.bias_power,
            "bias_location": case.bias_location,
            "diagnosis": (
                "The prescribed XFEM crack plane lies on a uniform mesh level. "
                "That degenerates the shifted-Heaviside split and can produce "
                "a false converged zero-crack response."
            ),
        }
        (out_dir / "xfem_case_failure.json").write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        print(f"[invalid] {case.name}: crack plane coincides with a mesh level")
        return out_dir

    cmd = [
        str(args.exe),
        "--output-dir",
        str(out_dir),
        "--amplitudes-mm",
        "50,100,150,200",
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--section-cells-x",
        "2",
        "--section-cells-y",
        "2",
        "--tangential-slip-drift-ratio",
        "0.0",
        "--global-xfem-concrete-material",
        "cyclic-crack-band",
        "--global-xfem-crack-band-tangent",
        "secant",
        "--global-xfem-cohesive-tangent",
        args.global_xfem_cohesive_tangent,
        "--global-xfem-shear-cap-mpa",
        f"{args.global_xfem_shear_cap_mpa:g}",
        "--global-xfem-kinematic-formulation",
        args.global_xfem_kinematic_formulation,
        "--global-xfem-crack-crossing-rebar-mode",
        args.global_xfem_crack_crossing_rebar_mode,
        "--global-xfem-crack-crossing-rebar-area-scale",
        f"{args.global_xfem_crack_crossing_rebar_area_scale:g}",
        "--global-xfem-crack-crossing-gauge-length-mm",
        f"{args.global_xfem_crack_crossing_gauge_length_mm:g}",
        "--global-xfem-crack-crossing-bridge-law",
        args.global_xfem_crack_crossing_bridge_law,
        "--global-xfem-crack-crossing-yield-slip-mm",
        f"{args.global_xfem_crack_crossing_yield_slip_mm:g}",
        "--global-xfem-crack-crossing-hardening-ratio",
        f"{args.global_xfem_crack_crossing_hardening_ratio:g}",
        "--global-xfem-crack-crossing-axis-frame",
        args.global_xfem_crack_crossing_axis_frame,
        "--global-xfem-crack-crossing-host-axis-tangent",
        args.global_xfem_crack_crossing_host_axis_tangent,
        "--global-xfem-solver-max-iterations",
        "120",
        "--global-xfem-solver-profile",
        args.global_xfem_solver_profile,
        "--global-xfem-adaptive-increments",
        "--global-xfem-nx",
        str(case.nx),
        "--global-xfem-ny",
        str(case.ny),
        "--global-xfem-nz",
        str(case.nz),
        "--global-xfem-bias-power",
        f"{case.bias_power:g}",
        "--global-xfem-bias-location",
        case.bias_location,
    ]
    if not args.disable_global_xfem_solver_cascade:
        cmd += ["--global-xfem-solver-cascade"]
    if args.allow_guarded_xfem_finite_kinematics:
        cmd += ["--allow-guarded-xfem-finite-kinematics"]
    if args.global_xfem_continuation != "fixed-increment":
        cmd += [
            "--global-xfem-continuation",
            args.global_xfem_continuation,
            "--global-xfem-mixed-arc-target",
            f"{args.global_xfem_mixed_arc_target:g}",
            "--global-xfem-mixed-arc-reject-factor",
            f"{args.global_xfem_mixed_arc_reject_factor:g}",
            "--global-xfem-mixed-arc-reaction-scale-mn",
            f"{args.global_xfem_mixed_arc_reaction_scale_mn:g}",
            "--global-xfem-mixed-arc-damage-weight",
            f"{args.global_xfem_mixed_arc_damage_weight:g}",
            "--global-xfem-bordered-hybrid-disable-streak",
            str(args.global_xfem_bordered_hybrid_disable_streak),
            "--global-xfem-bordered-hybrid-retry-interval",
            str(args.global_xfem_bordered_hybrid_retry_interval),
        ]
    if args.global_xfem_crack_crossing_yield_force_mn is not None:
        cmd += [
            "--global-xfem-crack-crossing-yield-force-mn",
            f"{args.global_xfem_crack_crossing_yield_force_mn:g}",
        ]
    if args.global_xfem_crack_crossing_force_cap_mn is not None:
        cmd += [
            "--global-xfem-crack-crossing-force-cap-mn",
            f"{args.global_xfem_crack_crossing_force_cap_mn:g}",
        ]
    if not args.auto_crack_z:
        cmd += ["--global-xfem-crack-z-m", f"{args.crack_z_m:g}"]
    print(f"[run] {case.name}: {' '.join(cmd)}")
    tic = time.perf_counter()
    try:
        subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            timeout=args.case_timeout_seconds,
        )
        print(f"[done] {case.name}: {time.perf_counter() - tic:.1f} s")
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - tic
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "case": case.name,
            "status": "timeout",
            "timeout_seconds": args.case_timeout_seconds,
            "elapsed_seconds": elapsed,
            "command": cmd,
        }
        (out_dir / "xfem_case_failure.json").write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        print(f"[timeout] {case.name}: {elapsed:.1f} s")
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - tic
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "case": case.name,
            "status": "failed",
            "returncode": exc.returncode,
            "elapsed_seconds": elapsed,
            "command": cmd,
        }
        (out_dir / "xfem_case_failure.json").write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        print(f"[failed] {case.name}: rc={exc.returncode}, {elapsed:.1f} s")
    return out_dir


def compare_case(
    case: XFEMCase,
    out_dir: Path,
    structural_rows: list[dict[str, float]],
) -> dict[str, Any]:
    xfem_rows = read_csv(out_dir / "global_xfem_newton_hysteresis.csv")
    manifest = read_json(out_dir / "global_xfem_newton_manifest.json")
    structural_for_compare = [
        {
            "p": row["p"],
            "drift_mm": 1000.0 * interpolate_by_p(structural_rows, row["p"], "drift_m"),
            "base_shear_MN": interpolate_by_p(
                structural_rows,
                row["p"],
                "base_shear_MN",
            ),
        }
        for row in xfem_rows
    ]
    xfem_shear = [row["base_shear_MN"] for row in xfem_rows]
    structural_shear = [row["base_shear_MN"] for row in structural_for_compare]
    raw_errors = [candidate - reference for candidate, reference in zip(xfem_shear, structural_shear)]
    flipped_errors = [candidate + reference for candidate, reference in zip(xfem_shear, structural_shear)]
    raw_rms = rms(raw_errors)
    flipped_rms = rms(flipped_errors)
    sign_factor = -1.0 if flipped_rms < raw_rms else 1.0
    aligned_structural = [sign_factor * value for value in structural_shear]
    aligned_errors = [
        candidate - reference for candidate, reference in zip(xfem_shear, aligned_structural)
    ]
    peak_xfem = peak_abs(xfem_shear)
    peak_structural = peak_abs(aligned_structural)
    normalization = max(peak_xfem, peak_structural, 1.0e-12)
    aligned_structural_rows = [
        {
            "drift_mm": row["drift_mm"],
            "base_shear_MN": sign_factor * row["base_shear_MN"],
        }
        for row in structural_for_compare
    ]
    xfem_work_rows = [
        {"drift_mm": row["drift_mm"], "base_shear_MN": row["base_shear_MN"]}
        for row in xfem_rows
    ]
    return {
        "case": case.name,
        "output_dir": str(out_dir),
        "completed_successfully": bool(manifest.get("completed_successfully")),
        "nx": case.nx,
        "ny": case.ny,
        "nz": case.nz,
        "bias_power": case.bias_power,
        "bias_location": case.bias_location,
        "crack_z_m": manifest["physics"].get("crack_z_m", math.nan),
        "crack_z_source": manifest["physics"].get("crack_z_source", "unknown"),
        "solver_global_dofs": manifest["dofs"]["solver_global_dofs"],
        "host_element_count": manifest["mesh"]["element_count"],
        "enriched_node_count": manifest["mesh"]["enriched_node_count"],
        "point_count": manifest["protocol"]["point_count"],
        "wall_seconds": manifest["timing"]["total_wall_seconds"],
        "peak_abs_structural_base_shear_MN": peak_structural,
        "peak_abs_xfem_base_shear_MN": peak_xfem,
        "xfem_to_structural_peak_base_shear_ratio": peak_xfem
        / max(peak_structural, 1.0e-12),
        "raw_rms_error_MN": raw_rms,
        "sign_factor_applied_to_structural": sign_factor,
        "rms_base_shear_error_MN": rms(aligned_errors),
        "max_abs_base_shear_error_MN": peak_abs(aligned_errors),
        "peak_normalized_rms_base_shear_error": rms(aligned_errors) / normalization,
        "peak_normalized_max_base_shear_error": peak_abs(aligned_errors) / normalization,
        "structural_loop_work_MN_mm": loop_work(
            aligned_structural_rows,
            "base_shear_MN",
            "drift_mm",
        ),
        "xfem_loop_work_MN_mm": loop_work(
            xfem_work_rows,
            "base_shear_MN",
            "drift_mm",
        ),
        "peak_abs_steel_stress_MPa": manifest["observables"]["peak_abs_steel_stress_mpa"],
        "max_host_damage": manifest["observables"]["max_host_damage"],
        "max_damaged_host_points": manifest["observables"]["max_damaged_host_points"],
        "crack_crossing_rebar_mode": manifest.get("reinforcement", {}).get(
            "crack_crossing_rebar_mode",
            "none",
        ),
        "crack_crossing_bridge_law": manifest.get("reinforcement", {}).get(
            "crack_crossing_bridge_law",
            "material",
        ),
        "crack_crossing_rebar_area_scale": manifest.get("reinforcement", {}).get(
            "crack_crossing_rebar_area_scale",
            0.0,
        ),
        "crack_crossing_yield_slip_mm": manifest.get("reinforcement", {}).get(
            "crack_crossing_yield_slip_mm",
            math.nan,
        ),
        "crack_crossing_yield_force_mn": manifest.get("reinforcement", {}).get(
            "crack_crossing_yield_force_mn",
            math.nan,
        ),
        "crack_crossing_force_cap_mn": manifest.get("reinforcement", {}).get(
            "crack_crossing_force_cap_mn",
            math.nan,
        ),
        "crack_crossing_rebar_element_count": manifest.get("reinforcement", {}).get(
            "crack_crossing_rebar_element_count",
            0,
        ),
    }


def save_figure(fig: Any, figures_dir: Path, secondary_dir: Path, stem: str) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for extension in ("png", "pdf"):
        path = figures_dir / f"{stem}.{extension}"
        try:
            fig.savefig(
                path,
                dpi=300 if extension == "png" else None,
                bbox_inches="tight",
            )
        except PermissionError:
            print(f"[warn] locked figure not overwritten: {path}")
            artifacts[f"{extension}_locked"] = str(path)
            continue
        shutil.copy2(path, secondary_dir / path.name)
        artifacts[extension] = str(path)
        artifacts[f"secondary_{extension}"] = str(secondary_dir / path.name)
    return artifacts


def plot_matrix(
    args: argparse.Namespace,
    structural_rows: list[dict[str, float]],
    summaries: list[dict[str, Any]],
) -> dict[str, str]:
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
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    sign = summaries[0]["sign_factor_applied_to_structural"] if summaries else -1.0
    axes[0].plot(
        [1000.0 * row["drift_m"] for row in structural_rows],
        [1000.0 * sign * row["base_shear_MN"] for row in structural_rows],
        color="#111827",
        lw=1.8,
        label="Structural N=10 Lobatto",
    )
    colors = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#4d7c0f"]
    for index, summary in enumerate(summaries):
        rows = read_csv(Path(summary["output_dir"]) / "global_xfem_newton_hysteresis.csv")
        axes[0].plot(
            [row["drift_mm"] for row in rows],
            [1000.0 * row["base_shear_MN"] for row in rows],
            lw=1.0,
            color=colors[index % len(colors)],
            label=compact_case_label(summary["case"]),
        )
    axes[0].set_title("Hysteresis overlays")
    axes[0].set_xlabel("Tip drift [mm]")
    axes[0].set_ylabel("Base shear [kN]")
    axes[0].legend(fontsize=6)

    axes[1].scatter(
        [summary["solver_global_dofs"] for summary in summaries],
        [summary["peak_normalized_rms_base_shear_error"] for summary in summaries],
        c=[summary["wall_seconds"] for summary in summaries],
        cmap="viridis",
        s=70,
        edgecolor="#111827",
    )
    for summary in summaries:
        axes[1].annotate(
            compact_case_label(summary["case"]).replace(" uniform", " uni"),
            (
                summary["solver_global_dofs"],
                summary["peak_normalized_rms_base_shear_error"],
            ),
            fontsize=6,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[1].set_title("Error vs global DOFs")
    axes[1].set_xlabel("PETSc reduced DOFs")
    axes[1].set_ylabel("Peak-normalized RMS error")
    fig.suptitle(
        "Global XFEM secant mesh-refinement matrix vs structural reference",
        y=1.04,
        fontsize=12,
    )
    return save_figure(
        fig,
        args.figures_dir,
        args.secondary_figures_dir,
        args.matrix_stem,
    )


def main() -> int:
    args = parse_args()
    cases = list(DEFAULT_CASES)
    if args.include_2x2x4:
        cases.append(XFEMCase("nx2_ny2_nz4_fixed_bias2", 2, 2, 4, 2.0, "fixed-end"))
    for extra in args.extra_case:
        cases.append(parse_extra_case(extra))
    if args.only_case:
        selected = set(args.only_case)
        cases = [case for case in cases if case.name in selected]
    args.output_root.mkdir(parents=True, exist_ok=True)

    structural_rows = read_csv(args.structural_dir / "comparison_hysteresis.csv")
    summaries: list[dict[str, Any]] = []
    for case in cases:
        out_dir = run_case(args, case)
        try:
            summaries.append(compare_case(case, out_dir, structural_rows))
        except FileNotFoundError:
            print(f"[warn] {case.name}: missing completed comparison artifacts")

    summaries.sort(key=lambda row: (row["nx"] * row["ny"] * row["nz"], row["case"]))
    completed_summaries = [
        summary for summary in summaries if summary["completed_successfully"]
    ]
    incomplete_summaries = [
        summary for summary in summaries if not summary["completed_successfully"]
    ]
    write_csv(args.output_root / "xfem_structural_refinement_matrix.csv", summaries)
    artifacts = plot_matrix(args, structural_rows, completed_summaries)
    failures = [
        read_json(failure)
        for failure in sorted(args.output_root.glob("*/xfem_case_failure.json"))
    ]
    matrix_blockers = failures + [
        {
            "case": summary["case"],
            "status": "incomplete_protocol",
            "point_count": summary["point_count"],
            "output_dir": summary["output_dir"],
        }
        for summary in incomplete_summaries
    ]

    best = min(
        completed_summaries,
        key=lambda row: row["peak_normalized_rms_base_shear_error"],
        default=None,
    )
    summary = {
        "scope": "global_xfem_secant_structural_refinement_matrix_200mm",
        "status": "completed",
        "structural_bundle": str(args.structural_dir),
        "output_root": str(args.output_root),
        "steps_per_segment": args.steps_per_segment,
        "crack_z_m": None if args.auto_crack_z else args.crack_z_m,
        "crack_z_policy": "auto_first_element_midpoint"
        if args.auto_crack_z
        else "fixed_physical_position",
        "case_count": len(summaries),
        "failure_count": len(failures),
        "incomplete_case_count": len(incomplete_summaries),
        "best_case_by_peak_normalized_rms": best,
        "cases": summaries,
        "failures": failures,
        "incomplete_cases": incomplete_summaries,
        "promotion_gate": make_promotion_gate_summary(
            args,
            completed_summaries,
            matrix_blockers,
        ),
        "interpretation": (
            "Uniform and fixed-end-biased XFEM meshes are compared against the "
            "same N=10 Lobatto structural reference using protocol-time "
            "alignment and explicit base-reaction sign alignment."
        ),
        "artifacts": artifacts,
    }
    summary_path = args.figures_dir / f"{args.matrix_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, args.secondary_figures_dir / summary_path.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
