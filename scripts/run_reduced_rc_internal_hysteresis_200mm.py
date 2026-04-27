#!/usr/bin/env python3
"""
Run the canonical internal fall_n reduced RC-column cyclic benchmark up to
200 mm and freeze its hysteresis/timing bundle.

This script intentionally stays inside fall_n. Its purpose is to make the
current internal structural response visible before the external large-
amplitude equivalence is fully closed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
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

BLUE = "#0b5fa5"
GREEN = "#2f855a"
ORANGE = "#d97706"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the internal reduced RC-column cyclic benchmark up to 200 mm "
            "and emit preliminary hysteresis figures."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_reference_benchmark.exe",
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


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = args.output_dir.resolve()
    ensure_dir(out_dir)

    command = [
        str(args.falln_exe),
        "--analysis",
        "cyclic",
        "--output-dir",
        str(out_dir),
        "--material-mode",
        "nonlinear",
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

    process_wall_seconds, proc = run_command(command, repo_root)
    (out_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (out_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")

    manifest_path = out_dir / "runtime_manifest.json"
    if not manifest_path.exists():
        write_json(
            out_dir / "internal_hysteresis_200mm_summary.json",
            {
                "status": "failed",
                "return_code": proc.returncode,
                "command": command,
                "failure_reason": "runtime_manifest_missing",
            },
        )
        return proc.returncode or 1

    manifest = read_json(manifest_path)
    hysteresis = read_csv_rows(out_dir / "comparison_hysteresis.csv")
    moment_curvature = read_csv_rows(out_dir / "comparison_moment_curvature_base.csv")
    control_state = read_csv_rows(out_dir / "control_state.csv")

    max_abs_tip_drift_mm = max(abs(1.0e3 * float(row["drift_m"])) for row in hysteresis)
    max_abs_base_shear_kn = max(abs(1.0e3 * float(row["base_shear_MN"])) for row in hysteresis)
    max_abs_base_moment_knm = max(
        abs(1.0e3 * float(row["moment_y_MNm"])) for row in moment_curvature
    )
    max_newton_iterations = max(float(row["newton_iterations"]) for row in control_state)
    max_bisection_level = max(int(float(row["max_bisection_level"])) for row in control_state)
    unique_profiles = sorted(
        {
            str(row["solver_profile_label"])
            for row in control_state
            if str(row["solver_profile_label"]).strip()
        }
    )

    summary = {
        "status": "completed" if proc.returncode == 0 else "failed",
        "return_code": proc.returncode,
        "command": command,
        "solver_policy": args.solver_policy,
        "reported_status": manifest.get("status", "unknown"),
        "reported_timing": manifest.get("timing", {}),
        "process_wall_seconds": process_wall_seconds,
        "max_abs_tip_drift_mm": max_abs_tip_drift_mm,
        "max_abs_base_shear_kn": max_abs_base_shear_kn,
        "max_abs_base_moment_knm": max_abs_base_moment_knm,
        "max_newton_iterations": max_newton_iterations,
        "max_bisection_level": max_bisection_level,
        "active_solver_profiles": unique_profiles,
        "hysteresis_point_count": len(hysteresis),
        "control_state_count": len(control_state),
    }
    write_json(out_dir / "internal_hysteresis_200mm_summary.json", summary)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(
        [1.0e3 * float(row["drift_m"]) for row in hysteresis],
        [1.0e3 * float(row["base_shear_MN"]) for row in hysteresis],
        color=BLUE,
        lw=1.5,
    )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title(
        "fall_n internal RC-column cyclic hysteresis\n"
        + rf"$u_{{\max}}={max_abs_tip_drift_mm:.1f}\,\mathrm{{mm}}$, "
        + rf"$V_{{\max}}={max_abs_base_shear_kn:.2f}\,\mathrm{{kN}}$"
    )
    save(
        fig,
        figure_paths(
            "reduced_rc_internal_hysteresis_200mm",
            args.figures_dir,
            args.secondary_figures_dir,
        ),
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(
        [float(row["curvature_y"]) for row in moment_curvature],
        [1.0e3 * float(row["moment_y_MNm"]) for row in moment_curvature],
        color=ORANGE,
        lw=1.5,
    )
    ax.set_xlabel(r"Base curvature $\kappa_y$ [1/m]")
    ax.set_ylabel(r"Base moment $M_y$ [kN m]")
    ax.set_title(
        "fall_n internal base moment-curvature\n"
        + rf"$M_{{\max}}={max_abs_base_moment_knm:.2f}\,\mathrm{{kN\,m}}$"
    )
    save(
        fig,
        figure_paths(
            "reduced_rc_internal_moment_curvature_200mm",
            args.figures_dir,
            args.secondary_figures_dir,
        ),
    )

    fig, axes = plt.subplots(3, 1, figsize=(5.8, 7.4), sharex=True)
    runtime_p = [float(row["runtime_p"]) for row in control_state]
    axes[0].plot(runtime_p, [1.0e3 * float(row["target_drift_m"]) for row in control_state], color=GREEN, lw=1.0, ls=":", label="Target")
    axes[0].plot(runtime_p, [1.0e3 * float(row["actual_tip_drift_m"]) for row in control_state], color=BLUE, lw=1.4, label="Actual")
    axes[0].set_ylabel("Tip drift [mm]")
    axes[0].legend(loc="best")
    axes[0].set_title("Internal control trace to 200 mm")

    axes[1].plot(runtime_p, [float(row["newton_iterations"]) for row in control_state], color=ORANGE, lw=1.3)
    axes[1].set_ylabel("Newton its.")
    axes[1].set_title(rf"$\max n_{{Newton}}={max_newton_iterations:.1f}$")

    axes[2].step(runtime_p, [float(row["max_bisection_level"]) for row in control_state], where="mid", color=BLUE, lw=1.3)
    axes[2].set_ylabel("Bisection lvl")
    axes[2].set_xlabel("Pseudo-time")
    axes[2].set_title(
        "Active solver profiles: " + (", ".join(unique_profiles) if unique_profiles else "n/a")
    )
    save(
        fig,
        figure_paths(
            "reduced_rc_internal_control_trace_200mm",
            args.figures_dir,
            args.secondary_figures_dir,
        ),
    )

    copy_bundle_artifact(out_dir / "internal_hysteresis_200mm_summary.json", args.figures_dir)
    if args.secondary_figures_dir is not None:
        copy_bundle_artifact(
            out_dir / "internal_hysteresis_200mm_summary.json",
            args.secondary_figures_dir,
        )

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
