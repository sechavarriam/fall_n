#!/usr/bin/env python3
"""
Canonical fall_n vs OpenSees reduced RC-column benchmark runner.

This script does not replace the internal validation studies. Its role is
smaller and more precise: run one declared reduced-column slice through the
fall_n structural baseline and through the external OpenSeesPy bridge, then
freeze both observable comparison and compute-time comparison in one bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from python_launcher_utils import python_launcher_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical fall_n vs OpenSees reduced-column benchmark bundle."
    )
    parser.add_argument("--analysis", choices=("monotonic", "cyclic"), default="cyclic")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--material-mode",
        choices=("nonlinear", "elasticized"),
        default="nonlinear",
        help="Structural material family for both fall_n and OpenSees slices.",
    )
    parser.add_argument(
        "--solver-policy",
        choices=(
            "canonical-cascade",
            "newton-backtracking-only",
            "newton-l2-only",
            "newton-trust-region-only",
            "newton-trust-region-dogleg-only",
            "quasi-newton-only",
            "nonlinear-gmres-only",
            "nonlinear-cg-only",
            "anderson-only",
            "nonlinear-richardson-only",
        ),
        default="canonical-cascade",
        help="fall_n nonlinear solve policy used by the internal structural slice.",
    )
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=Path("build/fall_n_reduced_rc_column_reference_benchmark.exe"),
        help="Path to the fall_n reduced-column benchmark executable.",
    )
    parser.add_argument(
        "--python-launcher",
        default="py -3.12",
        help="Python launcher used to invoke the OpenSeesPy bridge.",
    )
    parser.add_argument("--beam-nodes", type=int, default=4)
    parser.add_argument(
        "--beam-integration",
        choices=("legendre", "lobatto", "radau-left", "radau-right"),
        default="legendre",
    )
    parser.add_argument(
        "--beam-element-family",
        choices=("disp", "force", "elastic-timoshenko"),
        default="disp",
        help=(
            "OpenSees structural formulation. `disp` is the parity anchor "
            "against the current fall_n displacement-based beam benchmark; "
            "`force` is kept as a sensitivity path; `elastic-timoshenko` is "
            "an elastic-only stiffness-equivalent control."
        ),
    )
    parser.add_argument(
        "--integration-points",
        type=int,
        help="OpenSees integration points. Defaults to beam_nodes - 1.",
    )
    parser.add_argument(
        "--geom-transf",
        choices=("linear", "pdelta"),
        default="linear",
        help="Use `linear` for the current small-strain parity slice; keep `pdelta` as a sensitivity path.",
    )
    parser.add_argument(
        "--mapping-policy",
        choices=("elasticized-parity", "monotonic-reference", "cyclic-diagnostic"),
        default=None,
        help=(
            "Declared OpenSees constitutive policy. Defaults to "
            "`monotonic-reference` for monotonic nonlinear runs, "
            "`cyclic-diagnostic` for cyclic nonlinear runs, and "
            "`elasticized-parity` for elasticized control runs."
        ),
    )
    parser.add_argument("--concrete-model", choices=("Elastic", "Concrete01", "Concrete02"), default=None)
    parser.add_argument("--concrete-lambda", type=float, default=None)
    parser.add_argument("--concrete-ft-ratio", type=float, default=None)
    parser.add_argument("--concrete-softening-multiplier", type=float, default=None)
    parser.add_argument("--concrete-unconfined-residual-ratio", type=float, default=None)
    parser.add_argument("--concrete-confined-residual-ratio", type=float, default=None)
    parser.add_argument("--concrete-ultimate-strain", type=float, default=None)
    parser.add_argument("--steel-r0", type=float, default=None)
    parser.add_argument("--steel-cr1", type=float, default=None)
    parser.add_argument("--steel-cr2", type=float, default=None)
    parser.add_argument("--steel-a1", type=float, default=None)
    parser.add_argument("--steel-a2", type=float, default=None)
    parser.add_argument("--steel-a3", type=float, default=None)
    parser.add_argument("--steel-a4", type=float, default=None)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument(
        "--continuation",
        choices=("monolithic", "segmented", "reversal-guarded", "arc-length"),
        default=None,
        help="fall_n continuation policy. Defaults to monolithic for monotonic and reversal-guarded for cyclic.",
    )
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument("--monotonic-tip-mm", type=float, default=2.5)
    parser.add_argument("--monotonic-steps", type=int, default=8)
    parser.add_argument("--amplitudes-mm", default="1.25,2.50")
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--reversal-substep-factor", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def relative_ratio(lhs: float, rhs: float) -> float:
    if abs(rhs) <= 1.0e-12:
        return math.nan
    return lhs / rhs


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    return time.perf_counter() - start, proc


def build_falln_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    continuation = args.continuation
    if continuation is None:
        continuation = "monolithic" if args.analysis == "monotonic" else "reversal-guarded"

    command = [
        str(args.falln_exe),
        "--analysis",
        args.analysis,
        "--output-dir",
        str(out_dir),
        "--material-mode",
        args.material_mode,
        "--solver-policy",
        args.solver_policy,
        "--beam-nodes",
        str(args.beam_nodes),
        "--beam-integration",
        args.beam_integration,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--continuation",
        continuation,
        "--continuation-segment-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--monotonic-tip-mm",
        str(args.monotonic_tip_mm),
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def build_opensees_command(args: argparse.Namespace, out_dir: Path, repo_root: Path) -> list[str]:
    integration_points = args.integration_points or max(args.beam_nodes - 1, 1)
    command = [
        *python_launcher_command(args.python_launcher),
        str(repo_root / "scripts/opensees_reduced_rc_column_reference.py"),
        "--analysis",
        args.analysis,
        "--output-dir",
        str(out_dir),
        "--material-mode",
        args.material_mode,
        "--beam-integration",
        args.beam_integration,
        "--beam-element-family",
        args.beam_element_family,
        "--integration-points",
        str(integration_points),
        "--geom-transf",
        args.geom_transf,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        str(args.monotonic_tip_mm),
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--reversal-substep-factor",
        str(args.reversal_substep_factor),
        "--max-bisections",
        str(args.max_bisections),
    ]
    if args.mapping_policy:
        command.extend(("--mapping-policy", args.mapping_policy))
    for flag, value in (
        ("--concrete-model", args.concrete_model),
        ("--concrete-lambda", args.concrete_lambda),
        ("--concrete-ft-ratio", args.concrete_ft_ratio),
        ("--concrete-softening-multiplier", args.concrete_softening_multiplier),
        (
            "--concrete-unconfined-residual-ratio",
            args.concrete_unconfined_residual_ratio,
        ),
        (
            "--concrete-confined-residual-ratio",
            args.concrete_confined_residual_ratio,
        ),
        ("--concrete-ultimate-strain", args.concrete_ultimate_strain),
        ("--steel-r0", args.steel_r0),
        ("--steel-cr1", args.steel_cr1),
        ("--steel-cr2", args.steel_cr2),
        ("--steel-a1", args.steel_a1),
        ("--steel-a2", args.steel_a2),
        ("--steel-a3", args.steel_a3),
        ("--steel-a4", args.steel_a4),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    return command


def write_timing_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=(
                "tool",
                "status",
                "process_wall_seconds",
                "analysis_wall_seconds",
                "output_write_wall_seconds",
                "reported_total_wall_seconds",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    falln_dir = root_dir / "fall_n"
    opensees_dir = root_dir / "opensees"
    ensure_dir(root_dir)
    ensure_dir(falln_dir)
    ensure_dir(opensees_dir)

    falln_command = build_falln_command(args, falln_dir)
    falln_elapsed, falln_proc = run_command(falln_command, repo_root)
    (falln_dir / "stdout.log").write_text(falln_proc.stdout, encoding="utf-8")
    (falln_dir / "stderr.log").write_text(falln_proc.stderr, encoding="utf-8")
    if falln_proc.returncode != 0:
        write_json(
            root_dir / "benchmark_summary.json",
            {
                "status": "failed",
                "failed_stage": "fall_n",
                "command": falln_command,
                "return_code": falln_proc.returncode,
            },
        )
        return falln_proc.returncode

    falln_manifest = read_json(falln_dir / "runtime_manifest.json")
    falln_hysteresis = falln_dir / "comparison_hysteresis.csv"
    falln_mk = falln_dir / "comparison_moment_curvature_base.csv"
    falln_section_response = falln_dir / "section_response.csv"
    falln_control_state = falln_dir / "control_state.csv"
    falln_section_layout = falln_dir / "section_layout.csv"
    falln_station_layout = falln_dir / "section_station_layout.csv"
    falln_section_fiber_history = (
        falln_dir / "comparison_section_fiber_state_history.csv"
    )

    opensees_command = build_opensees_command(args, opensees_dir, repo_root)
    opensees_command.extend(
        [
            "--falln-hysteresis",
            str(falln_hysteresis),
            "--falln-moment-curvature",
            str(falln_mk),
            "--falln-section-response",
            str(falln_section_response),
            "--falln-control-state",
            str(falln_control_state),
            "--falln-section-layout",
            str(falln_section_layout),
            "--falln-station-layout",
            str(falln_station_layout),
            "--falln-section-fiber-history",
            str(falln_section_fiber_history),
        ]
    )
    opensees_elapsed, opensees_proc = run_command(opensees_command, repo_root)
    (opensees_dir / "stdout.log").write_text(opensees_proc.stdout, encoding="utf-8")
    (opensees_dir / "stderr.log").write_text(opensees_proc.stderr, encoding="utf-8")
    if opensees_proc.returncode != 0:
        write_json(
            root_dir / "benchmark_summary.json",
            {
                "status": "failed",
                "failed_stage": "opensees",
                "command": opensees_command,
                "return_code": opensees_proc.returncode,
                "fall_n_status": falln_manifest.get("status"),
            },
        )
        return opensees_proc.returncode

    opensees_manifest = read_json(opensees_dir / "reference_manifest.json")
    comparison_summary = read_json(opensees_dir / "comparison_summary.json")

    falln_timing = dict(falln_manifest.get("timing", {}))
    opensees_timing = dict(opensees_manifest.get("timing", {}))
    timing_rows = [
        {
            "tool": "fall_n",
            "status": falln_manifest.get("status", "unknown"),
            "process_wall_seconds": falln_elapsed,
            "analysis_wall_seconds": falln_timing.get("analysis_wall_seconds", math.nan),
            "output_write_wall_seconds": falln_timing.get("output_write_wall_seconds", math.nan),
            "reported_total_wall_seconds": falln_timing.get("total_wall_seconds", math.nan),
        },
        {
            "tool": "OpenSeesPy",
            "status": opensees_manifest.get("status", "unknown"),
            "process_wall_seconds": opensees_elapsed,
            "analysis_wall_seconds": opensees_timing.get("analysis_wall_seconds", math.nan),
            "output_write_wall_seconds": opensees_timing.get("output_write_wall_seconds", math.nan),
            "reported_total_wall_seconds": opensees_timing.get("total_wall_seconds", math.nan),
        },
    ]
    write_timing_csv(root_dir / "timing_summary.csv", timing_rows)

    summary = {
        "status": "completed",
        "analysis": args.analysis,
        "material_mode": args.material_mode,
        "benchmark_scope": "reduced_rc_column_external_computational_reference",
        "fall_n": {
            "dir": str(falln_dir),
            "manifest": falln_manifest,
            "process_wall_seconds": falln_elapsed,
        },
        "opensees": {
            "dir": str(opensees_dir),
            "manifest": opensees_manifest,
            "process_wall_seconds": opensees_elapsed,
        },
        "comparison": comparison_summary,
        "timing_comparison": {
            "fall_n_over_opensees_process_ratio": relative_ratio(falln_elapsed, opensees_elapsed),
            "fall_n_over_opensees_reported_total_ratio": relative_ratio(
                float(falln_timing.get("total_wall_seconds", math.nan)),
                float(opensees_timing.get("total_wall_seconds", math.nan)),
            ),
        },
        "artifacts": {
            "timing_summary_csv": str(root_dir / "timing_summary.csv"),
            "fall_n_hysteresis": str(falln_hysteresis),
            "fall_n_moment_curvature": str(falln_mk),
            "fall_n_section_response": str(falln_section_response),
            "fall_n_control_state": str(falln_control_state),
            "fall_n_preload_state": str(falln_dir / "preload_state.json"),
            "fall_n_section_layout": str(falln_section_layout),
            "fall_n_station_layout": str(falln_station_layout),
            "fall_n_section_fiber_history": str(falln_section_fiber_history),
            "opensees_hysteresis": str(opensees_dir / "hysteresis.csv"),
            "opensees_moment_curvature": str(opensees_dir / "moment_curvature_base.csv"),
            "opensees_section_response": str(opensees_dir / "section_response.csv"),
            "opensees_control_state": str(opensees_dir / "control_state.csv"),
            "opensees_preload_state": str(opensees_dir / "preload_state.json"),
            "opensees_section_layout": str(opensees_dir / "section_layout.csv"),
            "opensees_station_layout": str(opensees_dir / "section_station_layout.csv"),
            "opensees_section_fiber_history": str(opensees_dir / "section_fiber_state_history.csv"),
            "comparison_summary": str(opensees_dir / "comparison_summary.json"),
        },
    }
    write_json(root_dir / "benchmark_summary.json", summary)

    print(
        "Benchmark completed:",
        f"fall_n total={falln_timing.get('total_wall_seconds', math.nan):.6f}s,",
        f"OpenSees total={opensees_timing.get('total_wall_seconds', math.nan):.6f}s,",
        f"process ratio={summary['timing_comparison']['fall_n_over_opensees_process_ratio']:.4f}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
