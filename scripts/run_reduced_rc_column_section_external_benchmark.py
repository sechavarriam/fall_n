#!/usr/bin/env python3
"""
Canonical fall_n vs OpenSees section-level benchmark for the reduced RC column.

This isolates the constitutive and fiber-section gap from the beam-element and
continuation gap by driving the same audited section ingredients through:
  1. fall_n's internal section moment-curvature baseline, and
  2. an OpenSeesPy zeroLengthSection reference.
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
        description="Run the canonical fall_n vs OpenSees section benchmark bundle."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis", choices=("monotonic", "cyclic"), default="monotonic")
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=Path("build/fall_n_reduced_rc_column_section_reference_benchmark.exe"),
    )
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument(
        "--material-mode",
        choices=("nonlinear", "elasticized"),
        default="nonlinear",
    )
    parser.add_argument(
        "--mapping-policy",
        choices=("elasticized-parity", "monotonic-reference", "cyclic-diagnostic"),
        default=None,
        help=(
            "Declared OpenSees constitutive policy. Defaults to "
            "`monotonic-reference` for nonlinear section runs and "
            "`elasticized-parity` for elasticized parity runs."
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
    parser.add_argument("--max-curvature-y", type=float, default=0.03)
    parser.add_argument(
        "--section-amplitudes-curvature-y",
        default="",
        help="Comma-separated curvature amplitudes for cyclic section benchmarks.",
    )
    parser.add_argument("--section-steps", type=int, default=120)
    parser.add_argument("--steps-per-segment", type=int, default=4)
    parser.add_argument("--reversal-substep-factor", type=int, default=1)
    parser.add_argument("--max-bisections", type=int, default=4)
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ratio(lhs: float, rhs: float) -> float:
    return math.nan if abs(rhs) <= 1.0e-12 else lhs / rhs


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


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
    ensure_dir(falln_dir)
    ensure_dir(opensees_dir)
    effective_steps_per_segment = max(args.steps_per_segment, 1) * max(
        args.reversal_substep_factor, 1
    )

    falln_command = [
        str(args.falln_exe),
        "--output-dir",
        str(falln_dir),
        "--analysis",
        args.analysis,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--material-mode",
        args.material_mode,
        "--max-curvature-y",
        str(args.max_curvature_y),
        "--steps",
        str(args.section_steps if args.analysis == "monotonic" else max(args.section_steps, 1)),
        "--steps-per-segment",
        str(effective_steps_per_segment),
        "--print-progress" if args.print_progress else "",
    ]
    if args.section_amplitudes_curvature_y:
        falln_command.extend(
            ("--amplitudes-curvature-y", args.section_amplitudes_curvature_y)
        )
    falln_command = [token for token in falln_command if token]
    falln_elapsed, falln_proc = run_command(falln_command, repo_root)
    (falln_dir / "stdout.log").write_text(falln_proc.stdout, encoding="utf-8")
    (falln_dir / "stderr.log").write_text(falln_proc.stderr, encoding="utf-8")
    if falln_proc.returncode != 0:
        write_json(
            root_dir / "benchmark_summary.json",
            {
                "status": "failed",
                "failed_stage": "fall_n_section",
                "command": falln_command,
                "return_code": falln_proc.returncode,
                "effective_steps_per_segment": effective_steps_per_segment,
                "reversal_substep_factor": args.reversal_substep_factor,
            },
        )
        return falln_proc.returncode

    falln_manifest = read_json(falln_dir / "runtime_manifest.json")
    falln_section_csv = falln_dir / "section_moment_curvature_baseline.csv"
    falln_section_fiber_history = falln_dir / "section_fiber_state_history.csv"
    falln_section_control_trace = falln_dir / "section_control_trace.csv"
    falln_section_layout = falln_dir / "section_layout.csv"
    falln_station_layout = falln_dir / "section_station_layout.csv"

    opensees_command = [
        *python_launcher_command(args.python_launcher),
        str(repo_root / "scripts/opensees_reduced_rc_column_reference.py"),
        "--model-kind",
        "section",
        "--analysis",
        args.analysis,
        "--material-mode",
        args.material_mode,
        "--output-dir",
        str(opensees_dir),
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--max-curvature-y",
        str(args.max_curvature_y),
        "--section-steps",
        str(args.section_steps),
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--reversal-substep-factor",
        str(args.reversal_substep_factor),
        "--max-bisections",
        str(args.max_bisections),
        "--falln-section-baseline",
        str(falln_section_csv),
        "--falln-section-fiber-history",
        str(falln_section_fiber_history),
        "--falln-section-control-trace",
        str(falln_section_control_trace),
        "--falln-section-layout",
        str(falln_section_layout),
        "--falln-station-layout",
        str(falln_station_layout),
    ]
    if args.section_amplitudes_curvature_y:
        opensees_command.extend(
            ("--section-amplitudes-curvature-y", args.section_amplitudes_curvature_y)
        )
    if args.mapping_policy:
        opensees_command.extend(("--mapping-policy", args.mapping_policy))
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
            opensees_command.extend((flag, str(value)))
    opensees_elapsed, opensees_proc = run_command(opensees_command, repo_root)
    (opensees_dir / "stdout.log").write_text(opensees_proc.stdout, encoding="utf-8")
    (opensees_dir / "stderr.log").write_text(opensees_proc.stderr, encoding="utf-8")
    if opensees_proc.returncode != 0:
        failure_manifest_path = opensees_dir / "reference_manifest.json"
        failure_manifest = (
            read_json(failure_manifest_path)
            if failure_manifest_path.exists()
            else {}
        )
        write_json(
            root_dir / "benchmark_summary.json",
            {
                "status": "failed",
                "failed_stage": "opensees_section",
                "command": opensees_command,
                "return_code": opensees_proc.returncode,
                "fall_n_status": falln_manifest.get("status"),
                "effective_steps_per_segment": effective_steps_per_segment,
                "reversal_substep_factor": args.reversal_substep_factor,
                "failure_message": failure_manifest.get("failure_message"),
                "failure_step": failure_manifest.get("failure_step"),
                "failure_target_curvature_y": failure_manifest.get(
                    "failure_target_curvature_y"
                ),
            },
        )
        return opensees_proc.returncode

    opensees_manifest = read_json(opensees_dir / "reference_manifest.json")
    comparison = read_json(opensees_dir / "comparison_summary.json")
    falln_timing = dict(falln_manifest.get("timing", {}))
    opensees_timing = dict(opensees_manifest.get("timing", {}))
    write_timing_csv(
        root_dir / "timing_summary.csv",
        [
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
        ],
    )

    summary = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_column_section_external_computational_reference",
        "analysis": args.analysis,
        "material_mode": args.material_mode,
        "effective_steps_per_segment": effective_steps_per_segment,
        "reversal_substep_factor": args.reversal_substep_factor,
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
        "comparison": comparison,
        "timing_comparison": {
            "fall_n_over_opensees_process_ratio": ratio(falln_elapsed, opensees_elapsed),
            "fall_n_over_opensees_reported_total_ratio": ratio(
                float(falln_timing.get("total_wall_seconds", math.nan)),
                float(opensees_timing.get("total_wall_seconds", math.nan)),
            ),
        },
        "artifacts": {
            "timing_summary_csv": str(root_dir / "timing_summary.csv"),
            "fall_n_section_baseline": str(falln_section_csv),
            "fall_n_section_fiber_history": str(falln_section_fiber_history),
            "fall_n_section_control_trace": str(falln_section_control_trace),
            "fall_n_section_tangent_diagnostics": str(
                falln_dir / "section_tangent_diagnostics.csv"
            ),
            "fall_n_section_layout": str(falln_section_layout),
            "fall_n_section_station_layout": str(falln_station_layout),
            "opensees_section_baseline": str(opensees_dir / "section_moment_curvature_baseline.csv"),
            "opensees_section_fiber_history": str(
                opensees_dir / "section_fiber_state_history.csv"
            ),
            "opensees_section_control_trace": str(
                opensees_dir / "section_control_trace.csv"
            ),
            "opensees_section_tangent_diagnostics": str(
                opensees_dir / "section_tangent_diagnostics.csv"
            ),
            "opensees_section_layout": str(opensees_dir / "section_layout.csv"),
            "opensees_section_station_layout": str(opensees_dir / "section_station_layout.csv"),
            "opensees_section_fiber_anchor_total_summary": str(
                opensees_dir / "section_fiber_anchor_total_summary.csv"
            ),
            "opensees_section_fiber_anchor_material_role_summary": str(
                opensees_dir / "section_fiber_anchor_material_role_summary.csv"
            ),
            "opensees_section_fiber_anchor_zone_summary": str(
                opensees_dir / "section_fiber_anchor_zone_summary.csv"
            ),
            "comparison_summary": str(opensees_dir / "comparison_summary.json"),
        },
    }
    write_json(root_dir / "benchmark_summary.json", summary)

    print(
        "Section benchmark completed:",
        f"fall_n total={falln_timing.get('total_wall_seconds', math.nan):.6f}s,",
        f"OpenSees total={opensees_timing.get('total_wall_seconds', math.nan):.6f}s,",
        f"process ratio={summary['timing_comparison']['fall_n_over_opensees_process_ratio']:.4f}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
