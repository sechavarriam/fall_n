#!/usr/bin/env python3
"""Benchmark nonlinear solver policies on the promoted global XFEM local model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from run_xfem_structural_refinement_matrix import (
    XFEMCase,
    compare_case,
    read_csv,
    read_json,
)


@dataclass(frozen=True)
class SolverCase:
    name: str
    profile: str
    description: str


DEFAULT_SOLVERS = (
    SolverCase(
        "newton_backtracking",
        "backtracking",
        "PETSc SNES Newton line-search with backtracking.",
    ),
    SolverCase(
        "newton_l2",
        "l2",
        "PETSc SNES Newton line-search with L2 merit search.",
    ),
    SolverCase(
        "newton_trust_region",
        "trust-region",
        "PETSc SNES Newton trust-region.",
    ),
    SolverCase(
        "quasi_newton",
        "quasi-newton",
        "PETSc SNES quasi-Newton candidate.",
    ),
    SolverCase(
        "nonlinear_gmres",
        "ngmres",
        "PETSc SNES nonlinear GMRES acceleration.",
    ),
    SolverCase(
        "anderson",
        "anderson",
        "PETSc SNES Anderson acceleration.",
    ),
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run a solver-policy benchmark on the promoted bounded-dowel "
            "global XFEM column and compare each run against the structural "
            "N=10 Lobatto reference."
        )
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
        "xfem_promoted_bounded_dowel_solver_policy_benchmark_200mm",
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
        default="xfem_promoted_bounded_dowel_solver_policy_benchmark_200mm",
    )
    parser.add_argument(
        "--solver",
        action="append",
        default=[],
        help=(
            "Solver profile token to run. May be passed more than once. "
            "Defaults to a representative Newton/non-Newton matrix."
        ),
    )
    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--continuation",
        default="fixed-increment",
        choices=("fixed-increment", "mixed-arc-length"),
        help=(
            "Continuation policy passed to the XFEM benchmark. "
            "Use mixed-arc-length to activate the checkpointed observable-arc "
            "guard while preserving protocol reversal points."
        ),
    )
    parser.add_argument(
        "--mixed-arc-target",
        type=float,
        default=0.50,
        help="Target scaled observable arc for mixed-arc-length runs.",
    )
    parser.add_argument(
        "--mixed-arc-reaction-scale-mn",
        type=float,
        default=0.02,
        help="Base-reaction scale used by the mixed-control arc metric.",
    )
    parser.add_argument(
        "--mixed-arc-damage-weight",
        type=float,
        default=0.10,
        help="Weight of the max-damage observable in the mixed-control arc.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=720.0,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def case_from_solver_token(token: str) -> SolverCase:
    normalized = token.replace("_", "-").lower()
    for case in DEFAULT_SOLVERS:
        if normalized == case.profile or normalized == case.name.replace("_", "-"):
            return case
    return SolverCase(
        normalized.replace("-", "_"),
        normalized,
        f"User-selected PETSc nonlinear profile '{normalized}'.",
    )


def selected_solver_cases(args: argparse.Namespace) -> list[SolverCase]:
    if args.solver:
        seen: set[str] = set()
        cases: list[SolverCase] = []
        for token in args.solver:
            case = case_from_solver_token(token)
            if case.profile not in seen:
                cases.append(case)
                seen.add(case.profile)
        return cases
    return list(DEFAULT_SOLVERS)


def run_solver_case(args: argparse.Namespace, case: SolverCase) -> Path:
    out_dir = args.output_root / case.name
    manifest = out_dir / "global_xfem_newton_manifest.json"
    failure_marker = out_dir / "xfem_solver_case_failure.json"
    if manifest.exists() and not args.force:
        data = read_json(manifest)
        solve_control = data.get("solve_control", {})
        if solve_control.get("solver_profile") == case.profile:
            status = "completed" if data.get("completed_successfully") else "incomplete"
            print(f"[skip] {case.name}: {status} artifact exists")
            return out_dir
    if failure_marker.exists() and not args.force:
        failure = read_json(failure_marker)
        print(f"[skip] {case.name}: previous {failure.get('status', 'failure')}")
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
        "central-fallback",
        "--global-xfem-shear-cap-mpa",
        "0.02",
        "--global-xfem-crack-crossing-rebar-mode",
        "dowel-x",
        "--global-xfem-crack-crossing-rebar-area-scale",
        "1.0",
        "--global-xfem-crack-crossing-gauge-length-mm",
        "100",
        "--global-xfem-crack-crossing-bridge-law",
        "bounded-slip",
        "--global-xfem-crack-crossing-yield-slip-mm",
        "0.25",
        "--global-xfem-crack-crossing-yield-force-mn",
        "0.00190",
        "--global-xfem-crack-crossing-force-cap-mn",
        "0.00190",
        "--global-xfem-solver-profile",
        case.profile,
        "--global-xfem-continuation",
        args.continuation,
        "--global-xfem-mixed-arc-target",
        str(args.mixed_arc_target),
        "--global-xfem-mixed-arc-reaction-scale-mn",
        str(args.mixed_arc_reaction_scale_mn),
        "--global-xfem-mixed-arc-damage-weight",
        str(args.mixed_arc_damage_weight),
        "--global-xfem-solver-max-iterations",
        "120",
        "--global-xfem-adaptive-increments",
        "--global-xfem-nx",
        "1",
        "--global-xfem-ny",
        "1",
        "--global-xfem-nz",
        "4",
        "--global-xfem-bias-power",
        "1.0",
        "--global-xfem-bias-location",
        "fixed-end",
        "--global-xfem-crack-z-m",
        "0.60",
    ]
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
            "profile": case.profile,
            "status": "timeout",
            "timeout_seconds": args.case_timeout_seconds,
            "elapsed_seconds": elapsed,
            "command": cmd,
        }
        failure_marker.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(f"[timeout] {case.name}: {elapsed:.1f} s")
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - tic
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "case": case.name,
            "profile": case.profile,
            "status": "failed",
            "returncode": exc.returncode,
            "elapsed_seconds": elapsed,
            "command": cmd,
        }
        failure_marker.write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(f"[failed] {case.name}: rc={exc.returncode}, {elapsed:.1f} s")
    return out_dir


def add_solver_diagnostics(summary: dict[str, Any], case: SolverCase) -> dict[str, Any]:
    manifest = read_json(Path(summary["output_dir"]) / "global_xfem_newton_manifest.json")
    solve = manifest.get("solve_control", {})
    summary = dict(summary)
    summary.update(
        {
            "solver_case": case.name,
            "solver_profile": case.profile,
            "solver_description": case.description,
            "total_accepted_substeps": solve.get("total_accepted_substeps", math.nan),
            "total_nonlinear_iterations": solve.get("total_nonlinear_iterations", math.nan),
            "total_failed_attempts": solve.get("total_failed_attempts", math.nan),
            "total_solver_profile_attempts": solve.get("total_solver_profile_attempts", math.nan),
            "max_requested_step_nonlinear_iterations": solve.get(
                "max_requested_step_nonlinear_iterations", math.nan
            ),
            "max_bisection_level": solve.get("max_bisection_level", math.nan),
            "hard_step_count": solve.get("hard_step_count", math.nan),
        }
    )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: Any, figures_dir: Path, secondary_dir: Path, stem: str) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for extension in ("png", "pdf"):
        path = figures_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        shutil.copy2(path, secondary_dir / path.name)
        artifacts[extension] = str(path)
        artifacts[f"secondary_{extension}"] = str(secondary_dir / path.name)
    return artifacts


def plot_solver_matrix(
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
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    sign = summaries[0]["sign_factor_applied_to_structural"] if summaries else -1.0
    axes[0].plot(
        [1000.0 * row["drift_m"] for row in structural_rows],
        [1000.0 * sign * row["base_shear_MN"] for row in structural_rows],
        color="#111827",
        lw=1.8,
        label="Structural N=10",
    )
    colors = ["#2563eb", "#0f766e", "#d97706", "#7c3aed", "#dc2626", "#0891b2"]
    for index, summary in enumerate(summaries):
        rows = read_csv(Path(summary["output_dir"]) / "global_xfem_newton_hysteresis.csv")
        axes[0].plot(
            [row["drift_mm"] for row in rows],
            [1000.0 * row["base_shear_MN"] for row in rows],
            lw=1.0,
            color=colors[index % len(colors)],
            label=summary["solver_profile"],
        )
    axes[0].set_title("Hysteresis invariance")
    axes[0].set_xlabel("Tip drift [mm]")
    axes[0].set_ylabel("Base shear [kN]")
    axes[0].legend(fontsize=6)

    x = list(range(len(summaries)))
    labels = [summary["solver_profile"] for summary in summaries]
    iter_axis = axes[1].twinx()
    axes[1].bar(
        [value - 0.18 for value in x],
        [summary["wall_seconds"] for summary in summaries],
        width=0.36,
        label="wall [s]",
        color="#64748b",
    )
    iter_axis.bar(
        [value + 0.18 for value in x],
        [summary["total_nonlinear_iterations"] for summary in summaries],
        width=0.36,
        label="iterations",
        color="#f97316",
    )
    axes[1].set_title("Cost")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylabel("Wall time [s]", color="#64748b")
    iter_axis.set_ylabel("Nonlinear iterations", color="#f97316")
    axes[1].legend(loc="upper left", fontsize=7)
    iter_axis.legend(loc="upper right", fontsize=7)

    axes[2].scatter(
        [summary["peak_normalized_rms_base_shear_error"] for summary in summaries],
        [summary["hard_step_count"] for summary in summaries],
        s=[
            40.0 + 6.0 * max(float(summary["total_failed_attempts"]), 0.0)
            for summary in summaries
        ],
        c=[summary["wall_seconds"] for summary in summaries],
        cmap="viridis",
        edgecolor="#111827",
    )
    for summary in summaries:
        axes[2].annotate(
            summary["solver_profile"],
            (
                summary["peak_normalized_rms_base_shear_error"],
                summary["hard_step_count"],
            ),
            fontsize=6,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[2].set_title("Robustness")
    axes[2].set_xlabel("Peak-normalized RMS error")
    axes[2].set_ylabel("Hard requested steps")
    fig.suptitle("Promoted bounded-dowel XFEM solver-policy benchmark", y=1.04)
    return save_figure(fig, args.figures_dir, args.secondary_figures_dir, args.matrix_stem)


def main() -> int:
    args = parse_args()
    solver_cases = selected_solver_cases(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    structural_rows = read_csv(args.structural_dir / "comparison_hysteresis.csv")

    summaries: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    xfem_case = XFEMCase("solver_policy", 1, 1, 4, 1.0, "fixed-end")
    for solver_case in solver_cases:
        out_dir = run_solver_case(args, solver_case)
        failure_path = out_dir / "xfem_solver_case_failure.json"
        if failure_path.exists():
            failures.append(read_json(failure_path))
            continue
        try:
            raw = compare_case(xfem_case, out_dir, structural_rows)
            diagnosed = add_solver_diagnostics(raw, solver_case)
            if diagnosed.get("completed_successfully"):
                summaries.append(diagnosed)
            else:
                incomplete.append(diagnosed)
        except FileNotFoundError as exc:
            failures.append(
                {
                    "case": solver_case.name,
                    "profile": solver_case.profile,
                    "status": "missing_artifacts",
                    "message": str(exc),
                }
            )

    summaries.sort(
        key=lambda row: (
            row.get("wall_seconds", math.inf),
            row.get("total_nonlinear_iterations", math.inf),
        )
    )
    incomplete.sort(key=lambda row: row.get("solver_profile", ""))
    write_csv(args.output_root / "xfem_solver_policy_benchmark.csv", summaries)
    artifacts = plot_solver_matrix(args, structural_rows, summaries) if summaries else {}
    best = min(
        summaries,
        key=lambda row: (
            row["hard_step_count"],
            row["wall_seconds"],
            row["total_nonlinear_iterations"],
        ),
        default=None,
    )
    summary = {
        "scope": "promoted_bounded_dowel_xfem_solver_policy_benchmark_200mm",
        "status": "completed" if summaries else "no_completed_cases",
        "output_root": str(args.output_root),
        "continuation": {
            "kind": args.continuation,
            "mixed_arc_target": args.mixed_arc_target,
            "mixed_arc_reaction_scale_mn": args.mixed_arc_reaction_scale_mn,
            "mixed_arc_damage_weight": args.mixed_arc_damage_weight,
        },
        "case_count": len(solver_cases),
        "completed_case_count": len(summaries),
        "incomplete_case_count": len(incomplete),
        "failure_count": len(failures),
        "best_case_by_robust_cost": best,
        "cases": summaries,
        "incomplete_cases": incomplete,
        "failures": failures,
        "interpretation": (
            "The benchmark keeps the promoted XFEM physical model fixed and "
            "changes only the PETSc nonlinear solver profile. This separates "
            "algorithmic robustness from physical calibration before the "
            "local model is used in the multiscale stage."
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
