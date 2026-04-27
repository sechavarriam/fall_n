#!/usr/bin/env python3
"""
Aggregate nonlinear solver-policy benchmarks over representative structural slices.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

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


POLICY_ORDER = [
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
]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced RC-column nonlinear solver-policy benchmark over "
            "a representative structural case matrix."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-launcher", default="py -3.11")
    parser.add_argument("--beam-nodes", default="2,4,10")
    parser.add_argument("--quadratures", default="legendre,lobatto")
    parser.add_argument("--solver-policies", default=",".join(POLICY_ORDER))
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
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
    parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="Force rerun of every per-case solver-policy worker even if a completed summary already exists.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class PolicyCaseRow:
    case_id: str
    beam_nodes: int
    beam_integration: str
    fastest_completed_policy: str
    completed_policy_count: int
    failed_policy_count: int
    canonical_completed: bool
    newton_l2_completed: bool
    canonical_process_wall_seconds: float
    newton_l2_process_wall_seconds: float
    best_completed_process_wall_seconds: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv(raw: str, caster=float):
    return [caster(token.strip()) for token in raw.split(",") if token.strip()]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)


def csv_value(summary: dict[str, object], policy: str, field: str) -> float:
    rows = summary.get("rows", [])
    for row in rows:
        if row.get("solver_policy") == policy:
            try:
                return float(row.get(field, math.nan))
            except (TypeError, ValueError):
                return math.nan
    return math.nan


def write_rows_csv(path: Path, rows: list[PolicyCaseRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def save(fig: plt.Figure, stem: str, figures_dir: Path, secondary_dir: Path | None) -> list[str]:
    outputs: list[str] = []
    for base in [figures_dir, *( [secondary_dir] if secondary_dir else [] )]:
        ensure_dir(base)
        for ext in ("png", "pdf"):
            path = base / f"{stem}.{ext}"
            fig.savefig(path)
            outputs.append(str(path))
    plt.close(fig)
    return outputs


def plot_rows(rows: list[PolicyCaseRow], figures_dir: Path, secondary_dir: Path | None) -> list[str]:
    outputs: list[str] = []
    case_labels = [row.case_id for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].bar(range(len(rows)), [row.completed_policy_count for row in rows], color="#0b5fa5")
    axes[0].set_xticks(range(len(rows)), case_labels, rotation=20, ha="right")
    axes[0].set_ylabel("Completed policies")
    axes[0].set_title("Nonlinear solver robustness by representative case")

    axes[1].bar(range(len(rows)), [row.best_completed_process_wall_seconds for row in rows], color="#2f855a")
    axes[1].set_xticks(range(len(rows)), case_labels, rotation=20, ha="right")
    axes[1].set_ylabel("Best process wall time [s]")
    axes[1].set_title("Best completed nonlinear policy by case")
    outputs += save(fig, "reduced_rc_solver_policy_case_matrix_overview", figures_dir, secondary_dir)

    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    xs = range(len(rows))
    ax.plot(xs, [row.canonical_process_wall_seconds for row in rows], marker="o", color="#0b5fa5", label="canonical cascade")
    ax.plot(xs, [row.newton_l2_process_wall_seconds for row in rows], marker="s", color="#d97706", label="newton l2")
    ax.set_xticks(range(len(rows)), case_labels, rotation=20, ha="right")
    ax.set_ylabel("Process wall time [s]")
    ax.set_title("Canonical Newton cascade vs newton-l2-only")
    ax.legend()
    outputs += save(fig, "reduced_rc_solver_policy_case_matrix_timing", figures_dir, secondary_dir)
    return outputs


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    ensure_dir(root_dir)
    worker = repo_root / "scripts" / "run_reduced_rc_nonlinear_solver_policy_benchmark.py"

    reuse_existing = not args.no_reuse_existing
    rows: list[PolicyCaseRow] = []
    for beam_nodes in parse_csv(args.beam_nodes, int):
        for quadrature in parse_csv(args.quadratures, str):
            case_id = f"n{beam_nodes:02d}_{quadrature.replace('-', '_')}"
            out_dir = root_dir / case_id
            ensure_dir(out_dir)
            summary_path = out_dir / "solver_policy_benchmark_summary.json"
            if reuse_existing and summary_path.exists():
                summary = read_json(summary_path)
            else:
                command = [
                    *shlex.split(args.runner_launcher),
                    str(worker),
                    "--output-dir",
                    str(out_dir),
                    "--beam-nodes",
                    str(beam_nodes),
                    "--beam-integration",
                    quadrature,
                    "--solver-policies",
                    args.solver_policies,
                    "--amplitudes-mm",
                    args.amplitudes_mm,
                    *(["--print-progress"] if args.print_progress else []),
                ]
                proc = run_command(command, repo_root)
                (out_dir / "wrapper_stdout.log").write_text(proc.stdout, encoding="utf-8")
                (out_dir / "wrapper_stderr.log").write_text(proc.stderr, encoding="utf-8")
                summary = read_json(summary_path)
            rows.append(
                PolicyCaseRow(
                    case_id=case_id,
                    beam_nodes=beam_nodes,
                    beam_integration=quadrature,
                    fastest_completed_policy=str(summary.get("fastest_completed_policy", "")),
                    completed_policy_count=int(summary.get("completed_policy_count", 0) or 0),
                    failed_policy_count=int(summary.get("failed_policy_count", 0) or 0),
                    canonical_completed=math.isfinite(csv_value(summary, "canonical-cascade", "process_wall_seconds")),
                    newton_l2_completed=math.isfinite(csv_value(summary, "newton-l2-only", "process_wall_seconds")),
                    canonical_process_wall_seconds=csv_value(summary, "canonical-cascade", "process_wall_seconds"),
                    newton_l2_process_wall_seconds=csv_value(summary, "newton-l2-only", "process_wall_seconds"),
                    best_completed_process_wall_seconds=min(
                        (
                            float(row.get("process_wall_seconds", math.inf))
                            for row in summary.get("rows", [])
                            if row.get("status") == "completed"
                        ),
                        default=math.nan,
                    ),
                )
            )
            if args.print_progress:
                print(
                    f"[{case_id}] fastest={rows[-1].fastest_completed_policy} "
                    f"completed={rows[-1].completed_policy_count}"
                )

    write_rows_csv(root_dir / "solver_policy_case_matrix.csv", rows)
    figures = plot_rows(rows, args.figures_dir, args.secondary_figures_dir)
    summary = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_solver_policy_case_matrix",
        "cases": [asdict(row) for row in rows],
        "artifacts": {
            "csv": str(root_dir / "solver_policy_case_matrix.csv"),
            "figures": figures,
        },
    }
    write_json(root_dir / "solver_policy_case_matrix_summary.json", summary)
    shutil.copy2(root_dir / "solver_policy_case_matrix_summary.json", args.figures_dir / "solver_policy_case_matrix_summary.json")
    if args.secondary_figures_dir:
        ensure_dir(args.secondary_figures_dir)
        shutil.copy2(root_dir / "solver_policy_case_matrix_summary.json", args.secondary_figures_dir / "solver_policy_case_matrix_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
