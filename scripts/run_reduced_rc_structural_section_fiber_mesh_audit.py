#!/usr/bin/env python3
"""Audit structural RC-section fiber refinement without changing steel layout.

The continuum model is currently softer than the validated structural column.
Before attributing that gap to continuum kinematics or concrete damage, this
script isolates one very mundane but important source of uncertainty: the fiber
discretization of the structural section itself.

Every profile uses the same gross section, cover, longitudinal steel area and
bar coordinates. Only the concrete patch subdivision changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

try:
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
except ModuleNotFoundError:
    plt = None


PROFILE_ORDER = ("coarse", "canonical", "fine", "ultra")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run structural RC-section fiber refinement audit."
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_section_fiber_mesh_audit",
    )
    parser.add_argument("--profiles", default="coarse,canonical,fine,ultra")
    parser.add_argument("--reference-profile", default="ultra")
    parser.add_argument("--analysis", choices=("monotonic", "cyclic"), default="cyclic")
    parser.add_argument("--beam-nodes", type=int, default=10)
    parser.add_argument(
        "--beam-integration",
        choices=("legendre", "lobatto", "radau-left", "radau-right"),
        default="lobatto",
    )
    parser.add_argument(
        "--top-rotation-mode",
        choices=("free", "clamped"),
        default="free",
        help=(
            "Use free for the current lateral-only continuum comparator; "
            "use clamped only as an explicit guided-cap control."
        ),
    )
    parser.add_argument(
        "--solver-policy",
        default="newton-l2-only",
        choices=(
            "canonical-cascade",
            "newton-backtracking-only",
            "newton-l2-only",
            "newton-l2-lu-symbolic-reuse-only",
            "newton-l2-gmres-ilu1-only",
            "newton-trust-region-only",
            "newton-trust-region-dogleg-only",
            "quasi-newton-only",
            "nonlinear-gmres-only",
            "nonlinear-cg-only",
            "anderson-only",
            "nonlinear-richardson-only",
        ),
    )
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--continuation", default="reversal-guarded")
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--monotonic-tip-mm", type=float, default=200.0)
    parser.add_argument("--monotonic-steps", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument(
        "--force-rerun",
        action="store_false",
        dest="reuse_existing",
        help="Regenerate bundles even if runtime_manifest.json already exists.",
    )
    parser.add_argument("--skip-figures", action="store_true")
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
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def split_csv(raw: str) -> list[str]:
    return [token.strip().lower() for token in raw.split(",") if token.strip()]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def command_for_profile(args: argparse.Namespace, profile: str, out_dir: Path) -> list[str]:
    command = [
        str(args.exe.resolve()),
        "--output-dir",
        str(out_dir),
        "--analysis",
        args.analysis,
        "--material-mode",
        "nonlinear",
        "--beam-nodes",
        str(args.beam_nodes),
        "--beam-integration",
        args.beam_integration,
        "--solver-policy",
        args.solver_policy,
        "--continuation",
        args.continuation,
        "--continuation-segment-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
        "--monotonic-tip-mm",
        f"{args.monotonic_tip_mm}",
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--section-fiber-profile",
        profile,
    ]
    if args.top_rotation_mode == "clamped":
        command.append("--clamp-top-bending-rotation")
    if args.print_progress:
        command.append("--print-progress")
    return command


def run_case(
    command: list[str],
    out_dir: Path,
    timeout_seconds: int,
    reuse_existing: bool,
) -> tuple[float | None, dict[str, Any]]:
    manifest_path = out_dir / "runtime_manifest.json"
    if reuse_existing and manifest_path.exists():
        return None, read_json(manifest_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - start
    (out_dir / "stdout.log").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / "stderr.log").write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0 and not manifest_path.exists():
        raise RuntimeError(
            f"Profile run failed with exit code {proc.returncode}: {' '.join(command)}"
        )
    if not manifest_path.exists():
        raise RuntimeError(f"Profile run did not write {manifest_path}")
    return elapsed, read_json(manifest_path)


def timing_value(manifest: dict[str, Any], key: str) -> float:
    timing = manifest.get("timing")
    if isinstance(timing, dict):
        value = timing.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return math.nan


def completed_successfully(manifest: dict[str, Any]) -> bool:
    completed = manifest.get("completed_successfully")
    if isinstance(completed, bool):
        return completed
    return str(manifest.get("status", "")).lower() == "completed"


def loop_work(rows: list[dict[str, str]]) -> float:
    work = 0.0
    for prev, curr in zip(rows[:-1], rows[1:]):
        du = float(curr["drift_m"]) - float(prev["drift_m"])
        avg_force = 0.5 * (
            float(curr["base_shear_MN"]) + float(prev["base_shear_MN"])
        )
        work += avg_force * du
    return work


def section_invariants(layout_rows: list[dict[str, str]]) -> dict[str, Any]:
    concrete_area = 0.0
    steel_area = 0.0
    first_moment_y = 0.0
    first_moment_z = 0.0
    steel_positions: list[tuple[float, float, float]] = []
    for row in layout_rows:
        area = float(row["area"])
        y = float(row["y"])
        z = float(row["z"])
        first_moment_y += area * y
        first_moment_z += area * z
        if row["material_role"] == "reinforcing_steel":
            steel_area += area
            steel_positions.append((y, z, area))
        else:
            concrete_area += area
    steel_positions.sort()
    return {
        "concrete_area_m2": concrete_area,
        "steel_area_m2": steel_area,
        "first_moment_y_m3": first_moment_y,
        "first_moment_z_m3": first_moment_z,
        "steel_bar_count": len(steel_positions),
        "steel_positions": [
            {"y": y, "z": z, "area": area} for y, z, area in steel_positions
        ],
    }


def matched_error(
    rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
) -> dict[str, float]:
    reference_by_step = {
        (int(float(row["step"])), round(float(row["p"]), 12)): row
        for row in reference_rows
    }
    reference_peak = max(
        (abs(float(row["base_shear_MN"])) for row in reference_rows),
        default=0.0,
    )
    denom = max(reference_peak, 1.0e-12)
    errors: list[float] = []
    signed_errors: list[float] = []
    for row in rows:
        key = (int(float(row["step"])), round(float(row["p"]), 12))
        ref = reference_by_step.get(key)
        if ref is None:
            continue
        diff = float(row["base_shear_MN"]) - float(ref["base_shear_MN"])
        signed_errors.append(diff)
        errors.append(abs(diff) / denom)
    if not errors:
        return {
            "matched_point_count": 0,
            "peak_normalized_max_base_shear_error": math.nan,
            "peak_normalized_rms_base_shear_error": math.nan,
            "mean_signed_base_shear_error_MN": math.nan,
        }
    return {
        "matched_point_count": len(errors),
        "peak_normalized_max_base_shear_error": max(errors),
        "peak_normalized_rms_base_shear_error": math.sqrt(
            sum(value * value for value in errors) / len(errors)
        ),
        "mean_signed_base_shear_error_MN": sum(signed_errors) / len(signed_errors),
    }


def profile_sort_key(profile: str) -> int:
    try:
        return PROFILE_ORDER.index(profile)
    except ValueError:
        return len(PROFILE_ORDER)


def make_figures(
    profiles: list[str],
    output_dir: Path,
    figure_dirs: list[Path],
    summary_rows: list[dict[str, Any]],
) -> None:
    if plt is None:
        print("matplotlib is not available; skipping figure export.")
        return

    for figure_dir in figure_dirs:
        figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    colors = {
        "coarse": "#9a3412",
        "canonical": "#0b5fa5",
        "fine": "#2f855a",
        "ultra": "#111827",
    }
    for profile in profiles:
        rows = read_csv_rows(output_dir / profile / "hysteresis.csv")
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [1.0e3 * float(row["base_shear_MN"]) for row in rows],
            linewidth=1.35,
            color=colors.get(profile, None),
            label=profile,
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Structural fiber-section refinement hysteresis")
    ax.legend(frameon=False)
    fig.tight_layout()
    for figure_dir in figure_dirs:
        fig.savefig(figure_dir / "reduced_rc_structural_section_fiber_mesh_hysteresis.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    labels = [str(row["profile"]) for row in summary_rows]
    values = [
        clean_number(row["peak_normalized_rms_base_shear_error_vs_reference"]) or 0.0
        for row in summary_rows
    ]
    ax.bar(labels, values, color=[colors.get(label, "#555555") for label in labels])
    ax.set_ylabel("RMS error vs reference / peak ref shear")
    ax.set_title("Section refinement convergence")
    fig.tight_layout()
    for figure_dir in figure_dirs:
        fig.savefig(figure_dir / "reduced_rc_structural_section_fiber_mesh_convergence.png")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = sorted(set(split_csv(args.profiles)), key=profile_sort_key)
    if args.reference_profile not in profiles:
        raise RuntimeError("--reference-profile must be included in --profiles.")

    case_payloads: dict[str, dict[str, Any]] = {}
    histories: dict[str, list[dict[str, str]]] = {}
    invariants: dict[str, dict[str, Any]] = {}

    for profile in profiles:
        case_dir = output_dir / profile
        elapsed, manifest = run_case(
            command_for_profile(args, profile, case_dir),
            case_dir,
            args.timeout_seconds,
            args.reuse_existing,
        )
        histories[profile] = read_csv_rows(case_dir / "hysteresis.csv")
        invariants[profile] = section_invariants(read_csv_rows(case_dir / "section_layout.csv"))
        case_payloads[profile] = {
            "bundle_dir": str(case_dir),
            "process_wall_seconds": clean_number(elapsed),
            "runtime_manifest": manifest,
            "section_invariants": invariants[profile],
        }

    reference_history = histories[args.reference_profile]
    summary_rows: list[dict[str, Any]] = []
    for profile in profiles:
        manifest = case_payloads[profile]["runtime_manifest"]
        fiber_mesh = manifest.get("section_fiber_mesh", {})
        rows = histories[profile]
        shear_values = [float(row["base_shear_MN"]) for row in rows]
        drift_values = [float(row["drift_m"]) for row in rows]
        errors = matched_error(rows, reference_history)
        summary_rows.append(
            {
                "profile": profile,
                "completed_successfully": completed_successfully(manifest),
                "concrete_fiber_count": (
                    fiber_mesh.get("concrete_fiber_count")
                    if isinstance(fiber_mesh, dict)
                    else None
                ),
                "total_section_fiber_count": (
                    fiber_mesh.get("total_section_fiber_count")
                    if isinstance(fiber_mesh, dict)
                    else None
                ),
                "hysteresis_point_count": len(rows),
                "peak_abs_drift_mm": 1.0e3 * max(
                    (abs(value) for value in drift_values),
                    default=0.0,
                ),
                "peak_abs_base_shear_kN": 1.0e3 * max(
                    (abs(value) for value in shear_values),
                    default=0.0,
                ),
                "signed_loop_work_MN_m": loop_work(rows),
                "reported_total_wall_seconds": timing_value(
                    manifest, "total_wall_seconds"
                ),
                "reported_solve_wall_seconds": timing_value(
                    manifest, "analysis_wall_seconds"
                ),
                "process_wall_seconds": case_payloads[profile]["process_wall_seconds"],
                "matched_point_count_vs_reference": errors["matched_point_count"],
                "peak_normalized_max_base_shear_error_vs_reference": errors[
                    "peak_normalized_max_base_shear_error"
                ],
                "peak_normalized_rms_base_shear_error_vs_reference": errors[
                    "peak_normalized_rms_base_shear_error"
                ],
                "mean_signed_base_shear_error_MN_vs_reference": errors[
                    "mean_signed_base_shear_error_MN"
                ],
                "concrete_area_m2": invariants[profile]["concrete_area_m2"],
                "steel_area_m2": invariants[profile]["steel_area_m2"],
                "first_moment_y_m3": invariants[profile]["first_moment_y_m3"],
                "first_moment_z_m3": invariants[profile]["first_moment_z_m3"],
                "steel_bar_count": invariants[profile]["steel_bar_count"],
            }
        )

    write_csv(output_dir / "structural_section_fiber_mesh_audit_summary.csv", summary_rows)
    write_json(
        output_dir / "structural_section_fiber_mesh_audit_summary.json",
        {
            "protocol": {
                "analysis": args.analysis,
                "beam_nodes": args.beam_nodes,
                "beam_integration": args.beam_integration,
                "top_rotation_mode": args.top_rotation_mode,
                "solver_policy": args.solver_policy,
                "axial_compression_mn": args.axial_compression_mn,
                "axial_preload_steps": args.axial_preload_steps,
                "continuation": args.continuation,
                "continuation_segment_substep_factor": args.continuation_segment_substep_factor,
                "amplitudes_mm": args.amplitudes_mm,
                "steps_per_segment": args.steps_per_segment,
                "max_bisections": args.max_bisections,
                "reference_profile": args.reference_profile,
            },
            "summary_rows": summary_rows,
            "case_payloads": case_payloads,
            "invariant_statement": (
                "Across section-fiber profiles, steel area and bar coordinates "
                "are held fixed; only concrete patch subdivisions vary."
            ),
        },
    )

    if not args.skip_figures:
        make_figures(
            profiles,
            output_dir,
            [args.figures_dir.resolve(), args.secondary_figures_dir.resolve()],
            summary_rows,
        )

    print(f"Wrote {output_dir / 'structural_section_fiber_mesh_audit_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
