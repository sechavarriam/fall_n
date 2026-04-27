#!/usr/bin/env python3
"""
Canonical fall_n vs OpenSees uniaxial-material benchmark bundle for the reduced
RC-column reboot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterialBenchmarkCase:
    name: str
    material: str
    protocol: str
    monotonic_target_strain: float = 0.0


CASES = (
    MaterialBenchmarkCase("steel_monotonic", "steel", "monotonic", 0.03),
    MaterialBenchmarkCase("steel_cyclic", "steel", "cyclic"),
    MaterialBenchmarkCase("concrete_monotonic", "concrete", "monotonic", -0.006),
    MaterialBenchmarkCase("concrete_cyclic", "concrete", "cyclic"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical fall_n vs OpenSees uniaxial-material benchmark bundle."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=Path("build/fall_n_reduced_rc_material_reference_benchmark.exe"),
    )
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument(
        "--case",
        choices=("all", *(case.name for case in CASES)),
        default="all",
    )
    parser.add_argument("--steps-per-branch", type=int, default=40)
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
                "case",
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


def selected_cases(case_name: str) -> tuple[MaterialBenchmarkCase, ...]:
    return CASES if case_name == "all" else tuple(case for case in CASES if case.name == case_name)


def build_case_command(case: MaterialBenchmarkCase, args: argparse.Namespace, out_dir: Path) -> dict[str, list[str]]:
    falln = [
        str(args.falln_exe),
        "--output-dir",
        str(out_dir / "fall_n"),
        "--material",
        case.material,
        "--protocol",
        case.protocol,
        "--steps-per-branch",
        str(args.steps_per_branch),
    ]
    opensees = [
        *shlex.split(args.python_launcher),
        str(Path("scripts") / "opensees_reduced_rc_material_reference.py"),
        "--output-dir",
        str(out_dir / "opensees"),
        "--material",
        case.material,
        "--protocol",
        case.protocol,
        "--steps-per-branch",
        str(args.steps_per_branch),
    ]
    if abs(case.monotonic_target_strain) > 0.0:
        monotonic_args = ["--monotonic-target-strain", str(case.monotonic_target_strain)]
        falln.extend(monotonic_args)
        opensees.extend(monotonic_args)
    if args.print_progress:
        falln.append("--print-progress")
    return {"falln": falln, "opensees": opensees}


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    ensure_dir(root_dir)

    case_summaries: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []

    for case in selected_cases(args.case):
        case_dir = root_dir / case.name
        falln_dir = case_dir / "fall_n"
        opensees_dir = case_dir / "opensees"
        ensure_dir(falln_dir)
        ensure_dir(opensees_dir)

        commands = build_case_command(case, args, case_dir)

        falln_elapsed, falln_proc = run_command(commands["falln"], repo_root)
        (falln_dir / "stdout.log").write_text(falln_proc.stdout, encoding="utf-8")
        (falln_dir / "stderr.log").write_text(falln_proc.stderr, encoding="utf-8")
        if falln_proc.returncode != 0:
            write_json(
                root_dir / "benchmark_summary.json",
                {
                    "status": "failed",
                    "failed_stage": f"{case.name}_fall_n",
                    "command": commands["falln"],
                    "return_code": falln_proc.returncode,
                },
            )
            return falln_proc.returncode

        falln_manifest = read_json(falln_dir / "runtime_manifest.json")
        falln_response = falln_dir / "uniaxial_response.csv"

        opensees_command = [
            *commands["opensees"],
            "--falln-response",
            str(falln_response),
        ]
        opensees_elapsed, opensees_proc = run_command(opensees_command, repo_root)
        (opensees_dir / "stdout.log").write_text(opensees_proc.stdout, encoding="utf-8")
        (opensees_dir / "stderr.log").write_text(opensees_proc.stderr, encoding="utf-8")
        if opensees_proc.returncode != 0:
            write_json(
                root_dir / "benchmark_summary.json",
                {
                    "status": "failed",
                    "failed_stage": f"{case.name}_opensees",
                    "command": opensees_command,
                    "return_code": opensees_proc.returncode,
                },
            )
            return opensees_proc.returncode

        opensees_manifest = read_json(opensees_dir / "reference_manifest.json")
        comparison = read_json(opensees_dir / "comparison_summary.json")
        falln_timing = dict(falln_manifest.get("timing", {}))
        opensees_timing = dict(opensees_manifest.get("timing", {}))

        timing_rows.extend(
            [
                {
                    "case": case.name,
                    "tool": "fall_n",
                    "status": falln_manifest.get("status", "unknown"),
                    "process_wall_seconds": falln_elapsed,
                    "analysis_wall_seconds": falln_timing.get("analysis_wall_seconds", math.nan),
                    "output_write_wall_seconds": falln_timing.get("output_write_wall_seconds", math.nan),
                    "reported_total_wall_seconds": falln_timing.get("total_wall_seconds", math.nan),
                },
                {
                    "case": case.name,
                    "tool": "OpenSeesPy",
                    "status": opensees_manifest.get("status", "unknown"),
                    "process_wall_seconds": opensees_elapsed,
                    "analysis_wall_seconds": opensees_timing.get("analysis_wall_seconds", math.nan),
                    "output_write_wall_seconds": opensees_timing.get("output_write_wall_seconds", math.nan),
                    "reported_total_wall_seconds": opensees_timing.get("total_wall_seconds", math.nan),
                },
            ]
        )

        case_summaries.append(
            {
                "case": case.name,
                "material": case.material,
                "protocol": case.protocol,
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
            }
        )

    write_timing_csv(root_dir / "timing_summary.csv", timing_rows)
    summary = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_column_uniaxial_material_external_computational_reference",
        "completed_case_count": len(case_summaries),
        "cases": case_summaries,
        "artifacts": {
            "timing_summary_csv": str(root_dir / "timing_summary.csv"),
        },
    }
    write_json(root_dir / "benchmark_summary.json", summary)
    print(
        "Material benchmark completed:",
        f"cases={len(case_summaries)}",
        f"output={root_dir}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
