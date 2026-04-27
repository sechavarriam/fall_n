#!/usr/bin/env python3
"""
Benchmark OpenSees hi-fi solver-profile families on the reduced RC-column case.

This benchmark is intentionally narrow:
  - single external hi-fi comparator configuration
  - multiple declared OpenSees convergence-profile families
  - timing / completion / iteration metrics only

It exists to preserve the reasoning behind the selected external comparator
policy once the inelastic 200 mm benchmark becomes achievable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opensees_reduced_rc_column_reference import structural_convergence_profile_families
from python_launcher_utils import python_launcher_command


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


@dataclass(frozen=True)
class ProfileRow:
    solver_profile_family: str
    status: str
    return_code: int
    bundle_dir: str
    process_wall_seconds: float
    reported_total_wall_seconds: float
    reported_analysis_wall_seconds: float
    hysteresis_point_count: int
    max_newton_iterations: float
    avg_newton_iterations: float
    max_bisection_level: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run an OpenSees hi-fi solver-profile benchmark on the reduced RC column."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument(
        "--profiles",
        default=(
            "pure-newton-unbalance,pure-line-search-disp,pure-krylov-disp,"
            "pure-broyden-disp,pure-secant-disp,pure-modified-initial-energy,"
            "robust-large-amplitude"
        ),
    )
    parser.add_argument("--model-dimension", choices=("2d", "3d"), default="2d")
    parser.add_argument("--beam-element-family", choices=("disp", "force"), default="disp")
    parser.add_argument("--beam-integration", choices=("legendre", "lobatto"), default="legendre")
    parser.add_argument("--integration-points", type=int, default=5)
    parser.add_argument("--structural-element-count", type=int, default=20)
    parser.add_argument("--geom-transf", choices=("linear", "pdelta"), default="pdelta")
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
    parser.add_argument("--steps-per-segment", type=int, default=32)
    parser.add_argument("--reversal-substep-factor", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=10)
    parser.add_argument("--mapping-policy", default="cyclic-diagnostic")
    parser.add_argument("--steel-r0", type=float, default=20.0)
    parser.add_argument("--steel-cr1", type=float, default=0.925)
    parser.add_argument("--steel-cr2", type=float, default=0.15)
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_csv(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_control_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_command(args: argparse.Namespace, repo_root: Path, out_dir: Path, profile: str) -> list[str]:
    command = [
        *python_launcher_command(args.python_launcher),
        str(repo_root / "scripts" / "opensees_reduced_rc_column_hifi_reference.py"),
        "--output-dir",
        str(out_dir),
        "--model-dimension",
        args.model_dimension,
        "--analysis",
        "cyclic",
        "--material-mode",
        "nonlinear",
        "--beam-element-family",
        args.beam_element_family,
        "--beam-integration",
        args.beam_integration,
        "--integration-points",
        str(args.integration_points),
        "--structural-element-count",
        str(args.structural_element_count),
        "--geom-transf",
        args.geom_transf,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--reversal-substep-factor",
        str(args.reversal_substep_factor),
        "--max-bisections",
        str(args.max_bisections),
        "--mapping-policy",
        args.mapping_policy,
        "--solver-profile-family",
        profile,
        "--steel-r0",
        str(args.steel_r0),
        "--steel-cr1",
        str(args.steel_cr1),
        "--steel-cr2",
        str(args.steel_cr2),
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def write_rows_csv(path: Path, rows: list[ProfileRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def save(fig: plt.Figure, stem: str, figures_dir: Path, secondary_dir: Path | None) -> list[str]:
    outputs: list[str] = []
    for base in [figures_dir, *([secondary_dir] if secondary_dir else [])]:
        ensure_dir(base)
        for ext in ("png", "pdf"):
            path = base / f"{stem}.{ext}"
            fig.savefig(path)
            outputs.append(str(path))
    plt.close(fig)
    return outputs


def plot_rows(rows: list[ProfileRow], figures_dir: Path, secondary_dir: Path | None) -> list[str]:
    outputs: list[str] = []
    labels = [row.solver_profile_family for row in rows]
    times = [row.process_wall_seconds for row in rows]
    max_newton = [row.max_newton_iterations for row in rows]
    statuses = [1.0 if row.status == "completed" else 0.0 for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].bar(labels, times, color="#0b5fa5")
    axes[0].set_ylabel("Process wall time [s]")
    axes[0].set_title("OpenSees hi-fi profile timing")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8)
    axes[1].bar(labels, max_newton, color="#d97706")
    axes[1].set_ylabel("Max Newton/test iterations")
    axes[1].set_title("OpenSees hi-fi profile iteration demand")
    axes[1].tick_params(axis="x", rotation=45, labelsize=8)
    outputs += save(fig, "reduced_rc_opensees_hifi_solver_profile_timing", figures_dir, secondary_dir)

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    ax.bar(labels, statuses, color="#2f855a")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Completed")
    ax.set_title("OpenSees hi-fi profile robustness")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    outputs += save(fig, "reduced_rc_opensees_hifi_solver_profile_status", figures_dir, secondary_dir)

    return outputs


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    ensure_dir(root_dir)

    profiles = parse_csv(args.profiles)
    for profile in profiles:
        if profile not in structural_convergence_profile_families():
            raise ValueError(f"Unknown OpenSees solver profile family: {profile}")

    rows: list[ProfileRow] = []
    for profile in profiles:
        bundle_dir = root_dir / profile.replace(":", "_")
        ensure_dir(bundle_dir)
        command = build_command(args, repo_root, bundle_dir, profile)
        elapsed, proc = run_command(command, repo_root)
        (bundle_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (bundle_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        manifest = read_json(bundle_dir / "reference_manifest.json")
        control_rows = read_control_rows(bundle_dir / "control_state.csv") if (bundle_dir / "control_state.csv").exists() else []
        newton_values = [
            float(row["newton_iterations"])
            for row in control_rows
            if row.get("newton_iterations")
        ]
        rows.append(
            ProfileRow(
                solver_profile_family=profile,
                status=str(manifest.get("status", "failed")),
                return_code=proc.returncode,
                bundle_dir=str(bundle_dir),
                process_wall_seconds=elapsed,
                reported_total_wall_seconds=float((manifest.get("timing") or {}).get("total_wall_seconds", math.nan)),
                reported_analysis_wall_seconds=float((manifest.get("timing") or {}).get("analysis_wall_seconds", math.nan)),
                hysteresis_point_count=int(manifest.get("hysteresis_point_count", 0)),
                max_newton_iterations=max(newton_values) if newton_values else math.nan,
                avg_newton_iterations=(sum(newton_values) / len(newton_values)) if newton_values else math.nan,
                max_bisection_level=max((int(float(row["max_bisection_level"])) for row in control_rows), default=0),
            )
        )
        if args.print_progress:
            print(
                f"[{profile}] status={rows[-1].status} "
                f"t={rows[-1].process_wall_seconds:.3f}s "
                f"iters={rows[-1].max_newton_iterations}"
            )

    write_rows_csv(root_dir / "solver_profile_cases.csv", rows)
    figures = plot_rows(rows, args.figures_dir, args.secondary_figures_dir)

    completed = [row for row in rows if row.status == "completed"]
    summary = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_opensees_hifi_solver_profile_benchmark",
        "model_dimension": args.model_dimension,
        "beam_element_family": args.beam_element_family,
        "beam_integration": args.beam_integration,
        "integration_points": args.integration_points,
        "structural_element_count": args.structural_element_count,
        "cyclic_amplitudes_mm": [float(value) for value in parse_csv(args.amplitudes_mm)],
        "profiles": profiles,
        "case_count": len(rows),
        "completed_case_count": len(completed),
        "failed_case_count": len(rows) - len(completed),
        "fastest_completed_profile": asdict(min(completed, key=lambda row: row.process_wall_seconds)) if completed else None,
        "artifacts": {
            "cases_csv": str(root_dir / "solver_profile_cases.csv"),
            "figures": figures,
        },
    }
    write_json(root_dir / "solver_profile_benchmark_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
