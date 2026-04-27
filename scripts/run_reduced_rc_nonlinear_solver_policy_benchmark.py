#!/usr/bin/env python3
"""
Benchmark nonlinear solver policies on the internal reduced RC-column slice.

The goal is not to crown a universal winner but to compare robustness, time,
and incremental behavior on the exact same cyclic protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
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

POLICY_STYLE = {
    "canonical-cascade": {"label": "canonical cascade", "color": "#0b5fa5", "marker": "o"},
    "newton-backtracking-only": {"label": "newton bt", "color": "#2f855a", "marker": "s"},
    "newton-l2-only": {"label": "newton l2", "color": "#d97706", "marker": "^"},
    "newton-l2-lu-symbolic-reuse-only": {"label": "newton l2 + LU reuse", "color": "#0891b2", "marker": ">"},
    "newton-l2-gmres-ilu1-only": {"label": "newton l2 + GMRES/ILU(1)", "color": "#7c3aed", "marker": "P"},
    "newton-trust-region-only": {"label": "newton tr", "color": "#7c3aed", "marker": "D"},
    "newton-trust-region-dogleg-only": {"label": "newton trdc", "color": "#6b46c1", "marker": "d"},
    "quasi-newton-only": {"label": "quasi-Newton", "color": "#c53030", "marker": "P"},
    "nonlinear-gmres-only": {"label": "NGMRES", "color": "#4a5568", "marker": "X"},
    "nonlinear-cg-only": {"label": "nonlinear CG", "color": "#2b6cb0", "marker": "v"},
    "anderson-only": {"label": "Anderson", "color": "#dd6b20", "marker": "h"},
    "nonlinear-richardson-only": {"label": "N-Richardson", "color": "#b83280", "marker": "*"},
}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare internal nonlinear solve policies on the reduced RC-column "
            "cyclic benchmark."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument(
        "--solver-policies",
        default=(
            "canonical-cascade,"
            "newton-backtracking-only,"
            "newton-l2-only,"
            "newton-l2-lu-symbolic-reuse-only,"
            "newton-l2-gmres-ilu1-only,"
            "newton-trust-region-only,"
            "newton-trust-region-dogleg-only,"
            "quasi-newton-only,"
            "nonlinear-gmres-only,"
            "nonlinear-cg-only,"
            "anderson-only,"
            "nonlinear-richardson-only"
        ),
    )
    parser.add_argument("--beam-nodes", type=int, default=4)
    parser.add_argument(
        "--beam-integration",
        choices=("legendre", "lobatto", "radau-left", "radau-right"),
        default="legendre",
    )
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument(
        "--continuation",
        choices=("monolithic", "segmented", "reversal-guarded", "arc-length"),
        default="reversal-guarded",
    )
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
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
    return parser.parse_args()


@dataclass(frozen=True)
class PolicyRow:
    solver_policy: str
    bundle_dir: str
    status: str
    return_code: int
    reported_status: str
    process_wall_seconds: float
    reported_total_wall_seconds: float
    reported_analysis_wall_seconds: float
    max_abs_tip_drift_mm: float
    max_abs_base_shear_kn: float
    hysteresis_point_count: int
    control_state_count: int
    max_newton_iterations: float
    avg_newton_iterations: float
    max_bisection_level: int
    avg_bisection_level: float
    active_solver_profiles: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(fh):
            converted: dict[str, object] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
        return rows


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def save(fig: plt.Figure, paths: list[Path]) -> None:
    for path in paths:
        ensure_dir(path.parent)
        fig.savefig(path)
    plt.close(fig)


def figure_paths(stem: str, figures_dir: Path, secondary_dir: Path | None) -> list[Path]:
    paths = [
        figures_dir / f"{stem}.png",
        figures_dir / f"{stem}.pdf",
    ]
    if secondary_dir is not None:
        paths.extend(
            [
                secondary_dir / f"{stem}.png",
                secondary_dir / f"{stem}.pdf",
            ]
        )
    return paths


def copy_bundle_artifact(source: Path, dest_dir: Path) -> None:
    if source.exists():
        ensure_dir(dest_dir)
        shutil.copy2(source, dest_dir / source.name)


def parse_policies(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def build_command(args: argparse.Namespace, out_dir: Path, solver_policy: str) -> list[str]:
    return [
        str(args.falln_exe),
        "--analysis",
        "cyclic",
        "--output-dir",
        str(out_dir),
        "--material-mode",
        "nonlinear",
        "--solver-policy",
        solver_policy,
        "--beam-nodes",
        str(args.beam_nodes),
        "--beam-integration",
        args.beam_integration,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--continuation",
        args.continuation,
        "--continuation-segment-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
        *(["--print-progress"] if args.print_progress else []),
    ]


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.output_dir.resolve()
    ensure_dir(out_dir)
    policies = parse_policies(args.solver_policies)

    rows: list[PolicyRow] = []
    successful_hysteresis: dict[str, list[dict[str, object]]] = {}

    for solver_policy in policies:
        policy_dir = out_dir / solver_policy.replace("-", "_")
        ensure_dir(policy_dir)
        command = build_command(args, policy_dir, solver_policy)
        process_wall_seconds, proc = run_command(command, repo_root)
        (policy_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (policy_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")

        manifest_path = policy_dir / "runtime_manifest.json"
        manifest = read_json(manifest_path) if manifest_path.exists() else {}
        hysteresis = (
            read_csv_rows(policy_dir / "comparison_hysteresis.csv")
            if (policy_dir / "comparison_hysteresis.csv").exists()
            else []
        )
        control_state = (
            read_csv_rows(policy_dir / "control_state.csv")
            if (policy_dir / "control_state.csv").exists()
            else []
        )

        max_abs_tip_drift_mm = (
            max(abs(1.0e3 * float(row["drift_m"])) for row in hysteresis)
            if hysteresis
            else math.nan
        )
        max_abs_base_shear_kn = (
            max(abs(1.0e3 * float(row["base_shear_MN"])) for row in hysteresis)
            if hysteresis
            else math.nan
        )
        max_newton_iterations = (
            max(float(row["newton_iterations"]) for row in control_state)
            if control_state
            else math.nan
        )
        avg_newton_iterations = (
            sum(float(row["newton_iterations"]) for row in control_state) / len(control_state)
            if control_state
            else math.nan
        )
        max_bisection_level = (
            max(int(float(row["max_bisection_level"])) for row in control_state)
            if control_state
            else 0
        )
        avg_bisection_level = (
            sum(float(row["max_bisection_level"]) for row in control_state) / len(control_state)
            if control_state
            else math.nan
        )
        active_profiles = sorted(
            {
                str(row["solver_profile_label"])
                for row in control_state
                if str(row["solver_profile_label"]).strip()
            }
        )
        rows.append(
            PolicyRow(
                solver_policy=solver_policy,
                bundle_dir=str(policy_dir),
                status="completed" if proc.returncode == 0 else "failed",
                return_code=proc.returncode,
                reported_status=str(manifest.get("status", "missing_manifest")),
                process_wall_seconds=process_wall_seconds,
                reported_total_wall_seconds=float(
                    (manifest.get("timing") or {}).get("total_wall_seconds", math.nan)
                ),
                reported_analysis_wall_seconds=float(
                    (manifest.get("timing") or {}).get("analysis_wall_seconds", math.nan)
                ),
                max_abs_tip_drift_mm=max_abs_tip_drift_mm,
                max_abs_base_shear_kn=max_abs_base_shear_kn,
                hysteresis_point_count=len(hysteresis),
                control_state_count=len(control_state),
                max_newton_iterations=max_newton_iterations,
                avg_newton_iterations=avg_newton_iterations,
                max_bisection_level=max_bisection_level,
                avg_bisection_level=avg_bisection_level,
                active_solver_profiles=";".join(active_profiles),
            )
        )
        if proc.returncode == 0 and hysteresis:
            successful_hysteresis[solver_policy] = hysteresis

    summary = {
        "rows": [asdict(row) for row in rows],
        "completed_policy_count": sum(1 for row in rows if row.status == "completed"),
        "failed_policy_count": sum(1 for row in rows if row.status != "completed"),
        "best_process_wall_seconds": min(
            (row.process_wall_seconds for row in rows if row.status == "completed"),
            default=math.nan,
        ),
        "fastest_completed_policy": next(
            (
                row.solver_policy
                for row in sorted(rows, key=lambda item: item.process_wall_seconds)
                if row.status == "completed"
            ),
            "",
        ),
    }
    write_json(out_dir / "solver_policy_benchmark_summary.json", summary)

    with (out_dir / "solver_policy_benchmark.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(asdict(rows[0]).keys()) if rows else ["solver_policy"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    if rows:
        completed_rows = [row for row in rows if row.status == "completed"]

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ordered = rows
        ax.bar(
            range(len(ordered)),
            [row.process_wall_seconds for row in ordered],
            color=[POLICY_STYLE.get(row.solver_policy, {}).get("color", "#0b5fa5") for row in ordered],
            alpha=0.9,
        )
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(
            [
                POLICY_STYLE.get(row.solver_policy, {}).get("label", row.solver_policy)
                for row in ordered
            ],
            rotation=20,
            ha="right",
        )
        ax.set_ylabel("Process wall time [s]")
        ax.set_title("Internal nonlinear solver policy timing benchmark")
        save(
            fig,
            figure_paths(
                "reduced_rc_internal_solver_policy_timing",
                args.figures_dir,
                args.secondary_figures_dir,
            ),
        )

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.bar(
            range(len(ordered)),
            [1.0 if row.status == "completed" else 0.0 for row in ordered],
            color=[POLICY_STYLE.get(row.solver_policy, {}).get("color", "#0b5fa5") for row in ordered],
            alpha=0.9,
        )
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(
            [
                POLICY_STYLE.get(row.solver_policy, {}).get("label", row.solver_policy)
                for row in ordered
            ],
            rotation=20,
            ha="right",
        )
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("Completed full 200 mm run")
        ax.set_title("Internal nonlinear solver policy robustness")
        save(
            fig,
            figure_paths(
                "reduced_rc_internal_solver_policy_robustness",
                args.figures_dir,
                args.secondary_figures_dir,
            ),
        )

        if completed_rows:
            fig, ax = plt.subplots(figsize=(6.0, 4.4))
            for row in completed_rows:
                style = POLICY_STYLE.get(row.solver_policy, {})
                hysteresis = successful_hysteresis[row.solver_policy]
                ax.plot(
                    [1.0e3 * float(point["drift_m"]) for point in hysteresis],
                    [1.0e3 * float(point["base_shear_MN"]) for point in hysteresis],
                    color=style.get("color", "#0b5fa5"),
                    lw=1.3,
                    label=style.get("label", row.solver_policy),
                )
            ax.set_xlabel("Tip drift [mm]")
            ax.set_ylabel("Base shear [kN]")
            ax.set_title("Internal hysteresis overlay by nonlinear solver policy")
            ax.legend(loc="best")
            save(
                fig,
                figure_paths(
                    "reduced_rc_internal_solver_policy_hysteresis_overlay",
                    args.figures_dir,
                    args.secondary_figures_dir,
                ),
            )

        copy_bundle_artifact(out_dir / "solver_policy_benchmark_summary.json", args.figures_dir)
        if args.secondary_figures_dir is not None:
            copy_bundle_artifact(
                out_dir / "solver_policy_benchmark_summary.json",
                args.secondary_figures_dir,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
