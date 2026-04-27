#!/usr/bin/env python3
"""
Audit the constitutive mapping gap behind the reduced RC-column reboot.

This bundle does not reopen section or beam effects. It keeps the benchmark at
uniaxial level and asks two sharper questions:

  1. How much of the remaining steel cyclic gap can be reduced within the
     Steel02 family by varying the transition parameters?
  2. Which external concrete family/profile behaves more comparably to the
     current Kent-Park baseline: Concrete01 or Concrete02?
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
class AuditCase:
    name: str
    material: str
    protocol: str
    monotonic_target_strain: float = 0.0


STEEL_CYCLIC = AuditCase("steel_cyclic", "steel", "cyclic")
CONCRETE_MONOTONIC = AuditCase("concrete_monotonic", "concrete", "monotonic", -0.006)
CONCRETE_CYCLIC = AuditCase("concrete_cyclic", "concrete", "cyclic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the constitutive-mapping audit for the reduced RC-column reboot."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=Path("build/fall_n_reduced_rc_material_reference_benchmark.exe"),
    )
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument("--steps-per-branch", type=int, default=24)
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        raw_rows = list(csv.DictReader(fh))
    return [
        {
            key: (float(value) if key != "step" else int(value))
            for key, value in row.items()
        }
        for row in raw_rows
    ]


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def run_falln_case(repo_root: Path, out_dir: Path, case: AuditCase, args: argparse.Namespace) -> dict[str, object]:
    ensure_dir(out_dir)
    command = [
        str(args.falln_exe),
        "--output-dir",
        str(out_dir),
        "--material",
        case.material,
        "--protocol",
        case.protocol,
        "--steps-per-branch",
        str(args.steps_per_branch),
    ]
    if abs(case.monotonic_target_strain) > 0.0:
        command.extend(("--monotonic-target-strain", str(case.monotonic_target_strain)))
    if args.print_progress:
        command.append("--print-progress")
    elapsed, proc = run_command(command, repo_root)
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"fall_n material baseline failed for {case.name}: exit {proc.returncode}")
    return {
        "process_wall_seconds": elapsed,
        "manifest": read_json(out_dir / "runtime_manifest.json"),
        "response_csv": out_dir / "uniaxial_response.csv",
    }


def run_opensees_case(
    repo_root: Path,
    out_dir: Path,
    case: AuditCase,
    args: argparse.Namespace,
    falln_response: Path,
    extra: list[str],
) -> dict[str, object]:
    ensure_dir(out_dir)
    command = [
        *shlex.split(args.python_launcher),
        str(Path("scripts") / "opensees_reduced_rc_material_reference.py"),
        "--output-dir",
        str(out_dir),
        "--material",
        case.material,
        "--protocol",
        case.protocol,
        "--steps-per-branch",
        str(args.steps_per_branch),
        "--falln-response",
        str(falln_response),
        *extra,
    ]
    if abs(case.monotonic_target_strain) > 0.0:
        command.extend(("--monotonic-target-strain", str(case.monotonic_target_strain)))
    elapsed, proc = run_command(command, repo_root)
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        return {
            "status": "process_failed",
            "process_wall_seconds": elapsed,
            "return_code": proc.returncode,
            "command": command,
        }
    return {
        "status": "completed",
        "process_wall_seconds": elapsed,
        "manifest": read_json(out_dir / "reference_manifest.json"),
        "comparison": read_json(out_dir / "comparison_summary.json"),
        "response_csv": out_dir / "uniaxial_response.csv",
    }


def composite_score(comparison: dict[str, object]) -> float:
    metric = lambda key, field: float(comparison.get(key, {}).get(field, math.inf))
    penalties = (
        metric("stress", "rms_rel_error"),
        metric("tangent", "rms_rel_error"),
        0.25 * metric("energy_density", "rms_rel_error"),
        10.0 * (
            metric("stress", "nonfinite_step_count")
            + metric("tangent", "nonfinite_step_count")
            + metric("energy_density", "nonfinite_step_count")
        ),
    )
    return sum(penalties)


def extract_turning_points(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    if len(rows) < 3:
        return []
    turning_points: list[dict[str, float]] = []
    for left, mid, right in zip(rows, rows[1:], rows[2:]):
        d0 = float(mid["strain"]) - float(left["strain"])
        d1 = float(right["strain"]) - float(mid["strain"])
        if d0 == 0.0 or d1 == 0.0 or d0 * d1 >= 0.0:
            continue
        turning_points.append(
            {
                "step": int(mid["step"]),
                "strain": float(mid["strain"]),
                "stress_mpa": float(mid["stress_MPa"]),
                "tangent_mpa": float(mid["tangent_MPa"]),
            }
        )
    return turning_points


def extract_landmarks(rows: list[dict[str, float]], protocol: str) -> dict[str, object]:
    if not rows:
        return {}

    peaks = {
        "peak_tension": max(rows, key=lambda row: float(row["stress_MPa"])),
        "peak_compression": min(rows, key=lambda row: float(row["stress_MPa"])),
        "max_tangent": max(rows, key=lambda row: abs(float(row["tangent_MPa"]))),
        "final": rows[-1],
    }
    summary: dict[str, object] = {
        "initial_tangent_mpa": float(rows[min(1, len(rows) - 1)]["tangent_MPa"]),
        "peak_tension_stress_mpa": float(peaks["peak_tension"]["stress_MPa"]),
        "peak_tension_strain": float(peaks["peak_tension"]["strain"]),
        "peak_compression_stress_mpa": float(peaks["peak_compression"]["stress_MPa"]),
        "peak_compression_strain": float(peaks["peak_compression"]["strain"]),
        "max_abs_tangent_mpa": abs(float(peaks["max_tangent"]["tangent_MPa"])),
        "final_energy_density_mpa": float(peaks["final"]["energy_density_MPa"]),
    }
    if protocol != "cyclic":
        return summary

    turning_points = extract_turning_points(rows)
    summary["turning_points"] = turning_points[:4]
    if turning_points:
        last_turning_step = int(turning_points[min(1, len(turning_points) - 1)]["step"])
        candidates = [row for row in rows if int(row["step"]) >= last_turning_step]
        if candidates:
            zero_return = min(candidates, key=lambda row: abs(float(row["strain"])))
            summary["first_zero_return"] = {
                "step": int(zero_return["step"]),
                "strain": float(zero_return["strain"]),
                "stress_mpa": float(zero_return["stress_MPa"]),
                "tangent_mpa": float(zero_return["tangent_MPa"]),
            }
    return summary


def relative_delta(lhs: float, rhs: float) -> float:
    scale = max(abs(rhs), 1.0e-12)
    return abs(lhs - rhs) / scale


def compare_landmarks(lhs: dict[str, object], rhs: dict[str, object]) -> dict[str, float]:
    comparable_keys = (
        "initial_tangent_mpa",
        "peak_tension_stress_mpa",
        "peak_tension_strain",
        "peak_compression_stress_mpa",
        "peak_compression_strain",
        "max_abs_tangent_mpa",
        "final_energy_density_mpa",
    )
    deltas = {}
    for key in comparable_keys:
        if key not in lhs or key not in rhs:
            continue
        lhs_value = float(lhs[key])
        rhs_value = float(rhs[key])
        if abs(lhs_value) < 1.0e-12 and abs(rhs_value) < 1.0e-12:
            continue
        deltas[key] = relative_delta(lhs_value, rhs_value)
    if "first_zero_return" in lhs and "first_zero_return" in rhs:
        pairs = (
            ("first_zero_return_stress_mpa", "stress_mpa"),
            ("first_zero_return_tangent_mpa", "tangent_mpa"),
        )
        for label, field in pairs:
            lhs_value = float(lhs["first_zero_return"][field])
            rhs_value = float(rhs["first_zero_return"][field])
            if abs(lhs_value) < 1.0e-12 and abs(rhs_value) < 1.0e-12:
                continue
            deltas[label] = relative_delta(lhs_value, rhs_value)
    return deltas


def concrete_profiles() -> tuple[tuple[str, list[str]], ...]:
    return (
        ("Concrete02_reference", ["--concrete-model", "Concrete02"]),
        (
            "Concrete02_low_tension",
            [
                "--concrete-model",
                "Concrete02",
                "--concrete-ft-ratio",
                "0.02",
                "--concrete-softening-multiplier",
                "0.50",
                "--concrete-residual-ratio",
                "0.20",
                "--concrete-ultimate-strain",
                "-0.006",
                "--concrete-lambda",
                "0.10",
            ],
        ),
        (
            "Concrete01_ksp_like",
            [
                "--concrete-model",
                "Concrete01",
                "--concrete-residual-ratio",
                "0.20",
                "--concrete-ultimate-strain",
                "-0.006",
            ],
        ),
    )


def steel_parameter_grid() -> tuple[tuple[str, list[str]], ...]:
    grid: list[tuple[str, list[str]]] = []
    for r0 in (8.0, 12.0, 20.0, 30.0):
        for cr1 in (8.0, 18.5, 30.0):
            for cr2 in (0.05, 0.15, 0.30):
                label = f"Steel02_r0_{r0:g}_cr1_{cr1:g}_cr2_{cr2:g}"
                grid.append(
                    (
                        label,
                        [
                            "--steel-r0",
                            f"{r0}",
                            "--steel-cr1",
                            f"{cr1}",
                            "--steel-cr2",
                            f"{cr2}",
                        ],
                    )
                )
    return tuple(grid)


def audit_rank(
    repo_root: Path,
    root_dir: Path,
    case: AuditCase,
    falln: dict[str, object],
    args: argparse.Namespace,
    candidates: tuple[tuple[str, list[str]], ...],
) -> dict[str, object]:
    falln_rows = read_csv_rows(Path(falln["response_csv"]))
    falln_landmarks = extract_landmarks(falln_rows, case.protocol)
    ranked: list[dict[str, object]] = []

    for label, extra in candidates:
        out_dir = root_dir / case.name / label
        result = run_opensees_case(
            repo_root,
            out_dir,
            case,
            args,
            Path(falln["response_csv"]),
            extra,
        )
        if result["status"] != "completed":
            ranked.append(
                {
                    "label": label,
                    "status": result["status"],
                    "score": math.inf,
                    "process_wall_seconds": result["process_wall_seconds"],
                    "return_code": result["return_code"],
                }
            )
            continue

        opensees_rows = read_csv_rows(Path(result["response_csv"]))
        opensees_landmarks = extract_landmarks(opensees_rows, case.protocol)
        comparison = dict(result["comparison"])
        landmark_deltas = compare_landmarks(opensees_landmarks, falln_landmarks)
        ranked.append(
            {
                "label": label,
                "status": "completed",
                "score": composite_score(comparison),
                "process_wall_seconds": result["process_wall_seconds"],
                "manifest": result["manifest"],
                "comparison": comparison,
                "fall_n_landmarks": falln_landmarks,
                "opensees_landmarks": opensees_landmarks,
                "landmark_relative_deltas": landmark_deltas,
            }
        )

    ranked.sort(key=lambda item: float(item["score"]))

    ranking_rows = []
    for item in ranked:
        comparison = item.get("comparison", {})
        ranking_rows.append(
            {
                "label": item["label"],
                "status": item["status"],
                "score": item["score"],
                "process_wall_seconds": item["process_wall_seconds"],
                "stress_rms_rel_error": comparison.get("stress", {}).get("rms_rel_error", math.nan),
                "stress_max_rel_error": comparison.get("stress", {}).get("max_rel_error", math.nan),
                "tangent_rms_rel_error": comparison.get("tangent", {}).get("rms_rel_error", math.nan),
                "tangent_max_rel_error": comparison.get("tangent", {}).get("max_rel_error", math.nan),
                "energy_rms_rel_error": comparison.get("energy_density", {}).get("rms_rel_error", math.nan),
                "energy_max_rel_error": comparison.get("energy_density", {}).get("max_rel_error", math.nan),
            }
        )

    write_csv(
        root_dir / case.name / "ranking_summary.csv",
        (
            "label",
            "status",
            "score",
            "process_wall_seconds",
            "stress_rms_rel_error",
            "stress_max_rel_error",
            "tangent_rms_rel_error",
            "tangent_max_rel_error",
            "energy_rms_rel_error",
            "energy_max_rel_error",
        ),
        ranking_rows,
    )

    return {
        "case": case.name,
        "material": case.material,
        "protocol": case.protocol,
        "fall_n": {
            "manifest": falln["manifest"],
            "process_wall_seconds": falln["process_wall_seconds"],
            "landmarks": falln_landmarks,
        },
        "candidate_count": len(ranked),
        "ranking_summary_csv": str(root_dir / case.name / "ranking_summary.csv"),
        "best_candidate": ranked[0] if ranked else {},
        "ranked_candidates": ranked[: min(8, len(ranked))],
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    ensure_dir(root_dir)

    falln_root = root_dir / "fall_n"
    ensure_dir(falln_root)
    falln_cases = {
        case.name: run_falln_case(repo_root, falln_root / case.name, case, args)
        for case in (STEEL_CYCLIC, CONCRETE_MONOTONIC, CONCRETE_CYCLIC)
    }

    steel_audit = audit_rank(
        repo_root,
        root_dir / "steel_profiles",
        STEEL_CYCLIC,
        falln_cases[STEEL_CYCLIC.name],
        args,
        steel_parameter_grid(),
    )
    concrete_monotonic_audit = audit_rank(
        repo_root,
        root_dir / "concrete_profiles_monotonic",
        CONCRETE_MONOTONIC,
        falln_cases[CONCRETE_MONOTONIC.name],
        args,
        concrete_profiles(),
    )
    concrete_cyclic_audit = audit_rank(
        repo_root,
        root_dir / "concrete_profiles_cyclic",
        CONCRETE_CYCLIC,
        falln_cases[CONCRETE_CYCLIC.name],
        args,
        concrete_profiles(),
    )

    summary = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_column_constitutive_mapping_audit",
        "steps_per_branch": args.steps_per_branch,
        "steel_cyclic_audit": steel_audit,
        "concrete_monotonic_audit": concrete_monotonic_audit,
        "concrete_cyclic_audit": concrete_cyclic_audit,
    }
    write_json(root_dir / "audit_summary.json", summary)
    print(f"Material mapping audit completed: output={root_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
