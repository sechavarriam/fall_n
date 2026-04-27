#!/usr/bin/env python3
"""
Audit external concrete-mapping profiles on the reduced RC section slice.

This stays deliberately below the structural column. The goal is to identify
which OpenSees concrete bridge best balances:

  - section moment-curvature parity,
  - section tangent parity,
  - zero-curvature anchor behavior,
  - and problematic-fiber closure.

The fall_n section slice is kept fixed; only the external concrete bridge is
varied. This avoids confounding beam/continuation effects while we decide what
to promote into the larger-amplitude structural comparison.
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
class Candidate:
    label: str
    args: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reduced RC section mapping-policy audit."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--max-curvature-y", type=float, default=0.03)
    parser.add_argument("--section-amplitudes-curvature-y", default="0.01")
    parser.add_argument("--steps-per-segment", type=int, default=4)
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "label",
        "status",
        "moment_max",
        "moment_rms",
        "tangent_max",
        "tangent_rms",
        "tangent_branch_max",
        "tangent_anchor_max",
        "fiber_stress_anchor_max",
        "fiber_tangent_anchor_max",
        "process_ratio",
        "process_wall_seconds",
        "score",
    )
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric(payload: dict[str, object], *keys: str) -> float:
    node: object = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return math.nan
        node = node[key]
    try:
        value = float(node)
    except (TypeError, ValueError):
        return math.nan
    return value


def score(row: dict[str, float]) -> float:
    return (
        row["moment_rms"]
        + 0.75 * row["tangent_rms"]
        + 0.50 * row["tangent_anchor_max"]
        + 0.25 * row["fiber_stress_anchor_max"]
    )


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def candidates() -> tuple[Candidate, ...]:
    return (
        Candidate("cyclic_diagnostic", ("--mapping-policy", "cyclic-diagnostic")),
        Candidate(
            "concrete02_ft_0p015",
            (
                "--mapping-policy",
                "cyclic-diagnostic",
                "--concrete-model",
                "Concrete02",
                "--concrete-ft-ratio",
                "0.015",
                "--concrete-softening-multiplier",
                "0.50",
                "--concrete-unconfined-residual-ratio",
                "0.20",
                "--concrete-confined-residual-ratio",
                "0.20",
                "--concrete-ultimate-strain",
                "-0.006",
                "--concrete-lambda",
                "0.10",
            ),
        ),
        Candidate(
            "concrete02_ft_0p01",
            (
                "--mapping-policy",
                "cyclic-diagnostic",
                "--concrete-model",
                "Concrete02",
                "--concrete-ft-ratio",
                "0.01",
                "--concrete-softening-multiplier",
                "0.50",
                "--concrete-unconfined-residual-ratio",
                "0.20",
                "--concrete-confined-residual-ratio",
                "0.20",
                "--concrete-ultimate-strain",
                "-0.006",
                "--concrete-lambda",
                "0.10",
            ),
        ),
        Candidate(
            "concrete02_ft_0p005",
            (
                "--mapping-policy",
                "cyclic-diagnostic",
                "--concrete-model",
                "Concrete02",
                "--concrete-ft-ratio",
                "0.005",
                "--concrete-softening-multiplier",
                "0.50",
                "--concrete-unconfined-residual-ratio",
                "0.20",
                "--concrete-confined-residual-ratio",
                "0.20",
                "--concrete-ultimate-strain",
                "-0.006",
                "--concrete-lambda",
                "0.10",
            ),
        ),
        Candidate(
            "concrete02_no_tension",
            (
                "--mapping-policy",
                "cyclic-diagnostic",
                "--concrete-model",
                "Concrete02",
                "--concrete-ft-ratio",
                "0.0",
                "--concrete-softening-multiplier",
                "0.50",
                "--concrete-unconfined-residual-ratio",
                "0.20",
                "--concrete-confined-residual-ratio",
                "0.20",
                "--concrete-ultimate-strain",
                "-0.006",
                "--concrete-lambda",
                "0.10",
            ),
        ),
        Candidate(
            "concrete02_ft_0p01_lambda_0p3",
            (
                "--mapping-policy",
                "cyclic-diagnostic",
                "--concrete-model",
                "Concrete02",
                "--concrete-ft-ratio",
                "0.01",
                "--concrete-softening-multiplier",
                "0.50",
                "--concrete-unconfined-residual-ratio",
                "0.20",
                "--concrete-confined-residual-ratio",
                "0.20",
                "--concrete-ultimate-strain",
                "-0.006",
                "--concrete-lambda",
                "0.30",
            ),
        ),
        Candidate(
            "concrete02_ft_0p02_lambda_0p3",
            (
                "--mapping-policy",
                "cyclic-diagnostic",
                "--concrete-model",
                "Concrete02",
                "--concrete-ft-ratio",
                "0.02",
                "--concrete-softening-multiplier",
                "0.50",
                "--concrete-unconfined-residual-ratio",
                "0.20",
                "--concrete-confined-residual-ratio",
                "0.20",
                "--concrete-ultimate-strain",
                "-0.006",
                "--concrete-lambda",
                "0.30",
            ),
        ),
        Candidate(
            "concrete01_ksp_like",
            (
                "--mapping-policy",
                "cyclic-diagnostic",
                "--concrete-model",
                "Concrete01",
                "--concrete-unconfined-residual-ratio",
                "0.20",
                "--concrete-confined-residual-ratio",
                "0.20",
                "--concrete-ultimate-strain",
                "-0.006",
            ),
        ),
    )


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    ensure_dir(root_dir)

    runner = repo_root / "scripts" / "run_reduced_rc_column_section_external_benchmark.py"
    rows: list[dict[str, object]] = []
    ranked: list[dict[str, object]] = []

    for candidate in candidates():
        out_dir = root_dir / candidate.label
        ensure_dir(out_dir)
        command = [
            *shlex.split(args.python_launcher),
            str(runner),
            "--output-dir",
            str(out_dir),
            "--analysis",
            "cyclic",
            "--material-mode",
            "nonlinear",
            "--axial-compression-mn",
            str(args.axial_compression_mn),
            "--max-curvature-y",
            str(args.max_curvature_y),
            "--section-amplitudes-curvature-y",
            args.section_amplitudes_curvature_y,
            "--steps-per-segment",
            str(args.steps_per_segment),
            *candidate.args,
        ]
        elapsed, proc = run_command(command, repo_root)
        (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")

        if proc.returncode != 0:
            row = {
                "label": candidate.label,
                "status": "failed",
                "moment_max": math.nan,
                "moment_rms": math.nan,
                "tangent_max": math.nan,
                "tangent_rms": math.nan,
                "tangent_branch_max": math.nan,
                "tangent_anchor_max": math.nan,
                "fiber_stress_anchor_max": math.nan,
                "fiber_tangent_anchor_max": math.nan,
                "process_ratio": math.nan,
                "score": math.inf,
            }
            rows.append(row)
            continue

        summary = read_json(out_dir / "benchmark_summary.json")
        comparison = summary["comparison"]
        row = {
            "label": candidate.label,
            "status": "completed",
            "moment_max": metric(
                comparison, "section_moment_curvature", "max_rel_moment_error"
            ),
            "moment_rms": metric(
                comparison, "section_moment_curvature", "rms_rel_moment_error"
            ),
            "tangent_max": metric(
                comparison, "section_tangent", "max_rel_tangent_error"
            ),
            "tangent_rms": metric(
                comparison, "section_tangent", "rms_rel_tangent_error"
            ),
            "tangent_branch_max": metric(
                comparison, "section_tangent_branch_only", "max_rel_tangent_error"
            ),
            "tangent_anchor_max": metric(
                comparison,
                "section_tangent_zero_curvature_anchor_only",
                "max_rel_tangent_error",
            ),
            "fiber_stress_anchor_max": metric(
                comparison,
                "section_fiber_stress_zero_curvature_anchor_only",
                "max_rel_fiber_stress_error",
            ),
            "fiber_tangent_anchor_max": metric(
                comparison,
                "section_fiber_tangent_zero_curvature_anchor_only",
                "max_rel_fiber_tangent_error",
            ),
            "process_ratio": metric(
                summary, "timing_comparison", "fall_n_over_opensees_process_ratio"
            ),
        }
        row["score"] = score(row)
        row["process_wall_seconds"] = elapsed
        rows.append(row)
        ranked.append(
            {
                "label": candidate.label,
                "command": command,
                "summary": summary,
                "row": row,
            }
        )
        if args.print_progress:
            print(
                "Section mapping candidate completed:",
                candidate.label,
                f"score={row['score']:.6f}",
            )

    rows.sort(key=lambda item: float(item["score"]))
    ranked.sort(key=lambda item: float(item["row"]["score"]))
    write_csv(root_dir / "ranking_summary.csv", rows)
    write_json(
        root_dir / "audit_summary.json",
        {
            "status": "completed",
            "benchmark_scope": "reduced_rc_section_mapping_policy_audit",
            "best_candidate": ranked[0] if ranked else {},
            "ranked_candidates": ranked,
            "ranking_summary_csv": str(root_dir / "ranking_summary.csv"),
        },
    )
    print(f"Section mapping audit completed: output={root_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
