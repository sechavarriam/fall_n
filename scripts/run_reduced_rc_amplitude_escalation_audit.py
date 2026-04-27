#!/usr/bin/env python3
"""
Run a compact amplitude-escalation audit over the reduced RC-column external
computational benchmarks.

The audit deliberately separates:
  1. section-level cyclic amplitude escalation, where the current nonlinear
     frontier is localized, and
  2. structural cyclic amplitude escalation over the declared displacement-
     based reference family.

This keeps the validation story honest: we do not promote larger-amplitude
structural claims unless the section-level bridge remains scientifically
interpretable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from python_launcher_utils import python_launcher_command


def parse_csv_floats(raw: str) -> list[float]:
    return [float(token) for token in raw.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced RC-column cyclic amplitude-escalation audit over "
            "the staged fall_n vs OpenSees benchmarks."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("section", "structural", "both"),
        default="both",
    )
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument(
        "--runner-launcher",
        default="py -3.11",
        help=(
            "Python launcher used to invoke the benchmark runners themselves. "
            "Kept separate from --python-launcher because the nested OpenSees "
            "bridge can require a different interpreter than the orchestration "
            "scripts."
        ),
    )
    parser.add_argument(
        "--section-amplitudes-curvature-y",
        default="0.01,0.015,0.02,0.025,0.03",
    )
    parser.add_argument(
        "--structural-amplitudes-mm",
        default="1.25,2.50,5.00,7.50,10.00,15.00",
    )
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--section-steps", type=int, default=120)
    parser.add_argument("--section-steps-per-segment", type=int, default=4)
    parser.add_argument("--structural-steps-per-segment", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--beam-nodes", type=int, default=4)
    parser.add_argument(
        "--beam-integration",
        choices=("legendre", "lobatto", "radau-left", "radau-right"),
        default="legendre",
    )
    parser.add_argument(
        "--continuation",
        choices=("reversal-guarded", "segmented", "monolithic", "arc-length"),
        default="reversal-guarded",
    )
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("doc/figures/validation_reboot"),
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=Path("PhD_Thesis/Figuras/validation_reboot"),
    )
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def dig(payload: dict[str, object], *keys: str) -> object:
    cursor: object = payload
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return math.nan
        cursor = cursor[key]
    return cursor


def sanitize_token(value: float, unit_label: str) -> str:
    token = f"{value:.4f}".replace("-", "m").replace(".", "p")
    return f"{unit_label}_{token}"


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


@dataclass(frozen=True)
class AuditRow:
    scope: str
    amplitude_value: float
    amplitude_unit: str
    bundle_dir: str
    status: str
    fall_n_completed: bool
    opensees_completed: bool
    failed_stage: str = ""
    return_code: int = 0
    failure_note: str = ""
    section_moment_max_rel_error: float = math.nan
    section_moment_rms_rel_error: float = math.nan
    section_tangent_max_rel_error: float = math.nan
    section_tangent_rms_rel_error: float = math.nan
    section_tangent_branch_max_rel_error: float = math.nan
    section_tangent_anchor_max_rel_error: float = math.nan
    hysteresis_max_rel_error: float = math.nan
    hysteresis_rms_rel_error: float = math.nan
    base_moment_max_rel_error: float = math.nan
    base_moment_rms_rel_error: float = math.nan
    structural_section_tangent_max_rel_error: float = math.nan
    structural_section_tangent_rms_rel_error: float = math.nan
    tip_drift_max_rel_error: float = math.nan
    preload_tip_axial_displacement_max_rel_error: float = math.nan
    fall_n_process_wall_seconds: float = math.nan
    opensees_process_wall_seconds: float = math.nan
    fall_n_reported_total_wall_seconds: float = math.nan
    opensees_reported_total_wall_seconds: float = math.nan
    process_wall_ratio_fall_n_over_opensees: float = math.nan
    reported_total_ratio_fall_n_over_opensees: float = math.nan


def completed_status_for(summary: dict[str, object], tool: str) -> bool:
    manifest_status = dig(summary, tool, "manifest", "status")
    if manifest_status == "completed":
        return True

    if tool == "fall_n":
        return summary.get("fall_n_status") == "completed"
    if tool == "opensees":
        return summary.get("opensees_status") == "completed"
    return False


def section_row_from_summary(amplitude: float, bundle_dir: Path, summary: dict[str, object]) -> AuditRow:
    return AuditRow(
        scope="section",
        amplitude_value=amplitude,
        amplitude_unit="1_over_m",
        bundle_dir=str(bundle_dir),
        status=str(summary.get("status", "unknown")),
        fall_n_completed=completed_status_for(summary, "fall_n"),
        opensees_completed=completed_status_for(summary, "opensees"),
        failed_stage=str(summary.get("failed_stage", "")),
        return_code=int(summary.get("return_code", 0) or 0),
        failure_note=str(
            dig(summary, "opensees", "manifest", "benchmark_note")
            if summary.get("status") == "completed"
            else ""
        ),
        section_moment_max_rel_error=safe_float(
            dig(summary, "comparison", "section_moment_curvature", "max_rel_moment_error")
        ),
        section_moment_rms_rel_error=safe_float(
            dig(summary, "comparison", "section_moment_curvature", "rms_rel_moment_error")
        ),
        section_tangent_max_rel_error=safe_float(
            dig(summary, "comparison", "section_tangent", "max_rel_tangent_error")
        ),
        section_tangent_rms_rel_error=safe_float(
            dig(summary, "comparison", "section_tangent", "rms_rel_tangent_error")
        ),
        section_tangent_branch_max_rel_error=safe_float(
            dig(summary, "comparison", "section_tangent_branch_only", "max_rel_tangent_error")
        ),
        section_tangent_anchor_max_rel_error=safe_float(
            dig(summary, "comparison", "section_tangent_zero_curvature_anchor_only", "max_rel_tangent_error")
        ),
        fall_n_process_wall_seconds=safe_float(dig(summary, "fall_n", "process_wall_seconds")),
        opensees_process_wall_seconds=safe_float(dig(summary, "opensees", "process_wall_seconds")),
        fall_n_reported_total_wall_seconds=safe_float(
            dig(summary, "fall_n", "manifest", "timing", "total_wall_seconds")
        ),
        opensees_reported_total_wall_seconds=safe_float(
            dig(summary, "opensees", "manifest", "timing", "total_wall_seconds")
        ),
        process_wall_ratio_fall_n_over_opensees=safe_float(
            dig(summary, "timing_comparison", "fall_n_over_opensees_process_ratio")
        ),
        reported_total_ratio_fall_n_over_opensees=safe_float(
            dig(summary, "timing_comparison", "fall_n_over_opensees_reported_total_ratio")
        ),
    )


def structural_row_from_summary(amplitude: float, bundle_dir: Path, summary: dict[str, object]) -> AuditRow:
    return AuditRow(
        scope="structural",
        amplitude_value=amplitude,
        amplitude_unit="mm",
        bundle_dir=str(bundle_dir),
        status=str(summary.get("status", "unknown")),
        fall_n_completed=completed_status_for(summary, "fall_n"),
        opensees_completed=completed_status_for(summary, "opensees"),
        failed_stage=str(summary.get("failed_stage", "")),
        return_code=int(summary.get("return_code", 0) or 0),
        failure_note=str(
            dig(summary, "fall_n", "manifest", "benchmark_note")
            if summary.get("status") == "completed"
            else ""
        ),
        hysteresis_max_rel_error=safe_float(
            dig(summary, "comparison", "hysteresis", "max_rel_base_shear_error")
        ),
        hysteresis_rms_rel_error=safe_float(
            dig(summary, "comparison", "hysteresis", "rms_rel_base_shear_error")
        ),
        base_moment_max_rel_error=safe_float(
            dig(summary, "comparison", "moment_curvature_base", "max_rel_moment_error")
        ),
        base_moment_rms_rel_error=safe_float(
            dig(summary, "comparison", "moment_curvature_base", "rms_rel_moment_error")
        ),
        structural_section_tangent_max_rel_error=safe_float(
            dig(summary, "comparison", "section_response_tangent", "max_rel_tangent_error")
        ),
        structural_section_tangent_rms_rel_error=safe_float(
            dig(summary, "comparison", "section_response_tangent", "rms_rel_tangent_error")
        ),
        tip_drift_max_rel_error=safe_float(
            dig(summary, "comparison", "control_state_tip_drift", "max_rel_tip_drift_error")
        ),
        preload_tip_axial_displacement_max_rel_error=safe_float(
            dig(summary, "comparison", "preload_state_tip_axial_displacement", "max_rel_tip_axial_displacement_error")
        ),
        fall_n_process_wall_seconds=safe_float(dig(summary, "fall_n", "process_wall_seconds")),
        opensees_process_wall_seconds=safe_float(dig(summary, "opensees", "process_wall_seconds")),
        fall_n_reported_total_wall_seconds=safe_float(
            dig(summary, "fall_n", "manifest", "timing", "total_wall_seconds")
        ),
        opensees_reported_total_wall_seconds=safe_float(
            dig(summary, "opensees", "manifest", "timing", "total_wall_seconds")
        ),
        process_wall_ratio_fall_n_over_opensees=safe_float(
            dig(summary, "timing_comparison", "fall_n_over_opensees_process_ratio")
        ),
        reported_total_ratio_fall_n_over_opensees=safe_float(
            dig(summary, "timing_comparison", "fall_n_over_opensees_reported_total_ratio")
        ),
    )


def write_rows_csv(path: Path, rows: list[AuditRow]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def finite_rows(rows: list[AuditRow], metric: str) -> list[AuditRow]:
    return [row for row in rows if math.isfinite(getattr(row, metric))]


def make_summary(rows: list[AuditRow], scope: str) -> dict[str, object]:
    summary: dict[str, object] = {
        "scope": scope,
        "case_count": len(rows),
        "combined_completed_case_count": sum(row.status == "completed" for row in rows),
        "fall_n_completed_case_count": sum(row.fall_n_completed for row in rows),
        "opensees_completed_case_count": sum(row.opensees_completed for row in rows),
        "highest_combined_completed_amplitude": math.nan,
        "highest_fall_n_completed_amplitude": math.nan,
        "highest_opensees_completed_amplitude": math.nan,
        "first_combined_failed_amplitude": math.nan,
        "first_fall_n_failed_amplitude": math.nan,
        "first_opensees_failed_amplitude": math.nan,
        "all_completed": all(row.status == "completed" for row in rows),
    }
    completed = [row for row in rows if row.status == "completed"]
    if completed:
        summary["highest_combined_completed_amplitude"] = max(
            row.amplitude_value for row in completed
        )
    fall_n_completed = [row for row in rows if row.fall_n_completed]
    if fall_n_completed:
        summary["highest_fall_n_completed_amplitude"] = max(
            row.amplitude_value for row in fall_n_completed
        )
    opensees_completed = [row for row in rows if row.opensees_completed]
    if opensees_completed:
        summary["highest_opensees_completed_amplitude"] = max(
            row.amplitude_value for row in opensees_completed
        )
    failed = [row for row in rows if row.status != "completed"]
    if failed:
        first_failed = min(failed, key=lambda row: row.amplitude_value)
        summary["first_combined_failed_amplitude"] = first_failed.amplitude_value
        summary["first_combined_failed_case"] = {
            "amplitude": first_failed.amplitude_value,
            "failed_stage": first_failed.failed_stage,
            "return_code": first_failed.return_code,
            "bundle_dir": first_failed.bundle_dir,
        }
    fall_n_failed = [row for row in rows if not row.fall_n_completed]
    if fall_n_failed:
        first_fall_n_failed = min(fall_n_failed, key=lambda row: row.amplitude_value)
        summary["first_fall_n_failed_amplitude"] = first_fall_n_failed.amplitude_value
        summary["first_fall_n_failed_case"] = {
            "amplitude": first_fall_n_failed.amplitude_value,
            "failed_stage": first_fall_n_failed.failed_stage,
            "return_code": first_fall_n_failed.return_code,
            "bundle_dir": first_fall_n_failed.bundle_dir,
        }
    opensees_failed = [row for row in rows if not row.opensees_completed]
    if opensees_failed:
        first_opensees_failed = min(opensees_failed, key=lambda row: row.amplitude_value)
        summary["first_opensees_failed_amplitude"] = first_opensees_failed.amplitude_value
        summary["first_opensees_failed_case"] = {
            "amplitude": first_opensees_failed.amplitude_value,
            "failed_stage": first_opensees_failed.failed_stage,
            "return_code": first_opensees_failed.return_code,
            "bundle_dir": first_opensees_failed.bundle_dir,
        }

    metric_labels = {
        "section": (
            "section_moment_max_rel_error",
            "section_tangent_max_rel_error",
            "section_tangent_branch_max_rel_error",
            "section_tangent_anchor_max_rel_error",
        ),
        "structural": (
            "hysteresis_max_rel_error",
            "base_moment_max_rel_error",
            "structural_section_tangent_max_rel_error",
            "tip_drift_max_rel_error",
        ),
    }
    for metric in metric_labels.get(scope, ()):
        metric_rows = finite_rows(rows, metric)
        if not metric_rows:
            continue
        worst = max(metric_rows, key=lambda row: getattr(row, metric))
        summary[f"worst_{metric}"] = {
            "amplitude": worst.amplitude_value,
            "value": getattr(worst, metric),
            "bundle_dir": worst.bundle_dir,
        }
    return summary


def maybe_plot_rows(
    rows: list[AuditRow],
    scope: str,
    figures_dir: Path,
    secondary_figures_dir: Path | None,
) -> list[str]:
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        return []

    if not rows:
        return []

    ensure_dir(figures_dir)
    if secondary_figures_dir is not None:
        ensure_dir(secondary_figures_dir)

    amplitudes = [row.amplitude_value for row in rows]
    outputs: list[str] = []

    def save(fig, stem: str) -> None:
        primary = figures_dir / f"{stem}.png"
        fig.tight_layout()
        fig.savefig(primary, dpi=180, bbox_inches="tight")
        outputs.append(str(primary))
        if secondary_figures_dir is not None:
            fig.savefig(secondary_figures_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    if scope == "section":
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        axes[0].plot(amplitudes, [row.section_moment_max_rel_error for row in rows], marker="o", label="moment max")
        axes[0].plot(amplitudes, [row.section_moment_rms_rel_error for row in rows], marker="s", label="moment rms")
        axes[0].set_xlabel(r"Curvature amplitude $\kappa_y$ [1/m]")
        axes[0].set_ylabel("Relative error")
        axes[0].set_title("Section moment-curvature error growth")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(amplitudes, [row.section_tangent_max_rel_error for row in rows], marker="o", label="tangent max")
        axes[1].plot(amplitudes, [row.section_tangent_branch_max_rel_error for row in rows], marker="s", label="branch-only max")
        axes[1].plot(amplitudes, [row.section_tangent_anchor_max_rel_error for row in rows], marker="^", label="anchor-only max")
        axes[1].set_xlabel(r"Curvature amplitude $\kappa_y$ [1/m]")
        axes[1].set_ylabel("Relative error")
        axes[1].set_title("Section tangent error growth")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        save(fig, "reduced_rc_section_amplitude_escalation_errors")

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        axes[0].plot(amplitudes, [row.fall_n_process_wall_seconds for row in rows], marker="o", label="fall_n process")
        axes[0].plot(amplitudes, [row.opensees_process_wall_seconds for row in rows], marker="s", label="OpenSees process")
        axes[0].set_xlabel(r"Curvature amplitude $\kappa_y$ [1/m]")
        axes[0].set_ylabel("Wall time [s]")
        axes[0].set_title("Section benchmark process time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(amplitudes, [row.process_wall_ratio_fall_n_over_opensees for row in rows], marker="o")
        axes[1].axhline(1.0, color="k", linestyle="--", linewidth=1.0)
        axes[1].set_xlabel(r"Curvature amplitude $\kappa_y$ [1/m]")
        axes[1].set_ylabel(r"$t_{fall_n} / t_{OpenSees}$")
        axes[1].set_title("Section process-time ratio")
        axes[1].grid(True, alpha=0.3)
        save(fig, "reduced_rc_section_amplitude_escalation_timing")
    elif scope == "structural":
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        axes[0].plot(amplitudes, [row.hysteresis_max_rel_error for row in rows], marker="o", label="hysteresis max")
        axes[0].plot(amplitudes, [row.base_moment_max_rel_error for row in rows], marker="s", label="base moment max")
        axes[0].set_xlabel("Tip amplitude [mm]")
        axes[0].set_ylabel("Relative error")
        axes[0].set_title("Structural error growth")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(amplitudes, [row.structural_section_tangent_max_rel_error for row in rows], marker="o", label="section tangent max")
        axes[1].plot(amplitudes, [row.tip_drift_max_rel_error for row in rows], marker="s", label="tip drift max")
        axes[1].set_xlabel("Tip amplitude [mm]")
        axes[1].set_ylabel("Relative error")
        axes[1].set_title("Structural tangent/control growth")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        save(fig, "reduced_rc_structural_amplitude_escalation_errors")

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        axes[0].plot(amplitudes, [row.fall_n_process_wall_seconds for row in rows], marker="o", label="fall_n process")
        axes[0].plot(amplitudes, [row.opensees_process_wall_seconds for row in rows], marker="s", label="OpenSees process")
        axes[0].set_xlabel("Tip amplitude [mm]")
        axes[0].set_ylabel("Wall time [s]")
        axes[0].set_title("Structural benchmark process time")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(amplitudes, [row.process_wall_ratio_fall_n_over_opensees for row in rows], marker="o")
        axes[1].axhline(1.0, color="k", linestyle="--", linewidth=1.0)
        axes[1].set_xlabel("Tip amplitude [mm]")
        axes[1].set_ylabel(r"$t_{fall_n} / t_{OpenSees}$")
        axes[1].set_title("Structural process-time ratio")
        axes[1].grid(True, alpha=0.3)
        save(fig, "reduced_rc_structural_amplitude_escalation_timing")

    return outputs


def run_section_cases(args: argparse.Namespace, repo_root: Path, root_dir: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    section_root = root_dir / "section"
    ensure_dir(section_root)
    launcher = python_launcher_command(args.runner_launcher)
    runner = repo_root / "scripts/run_reduced_rc_column_section_external_benchmark.py"

    for amplitude in parse_csv_floats(args.section_amplitudes_curvature_y):
        bundle_dir = section_root / sanitize_token(amplitude, "kappa")
        ensure_dir(bundle_dir)
        command = [
            *launcher,
            str(runner),
            "--output-dir",
            str(bundle_dir),
            "--python-launcher",
            args.python_launcher,
            "--analysis",
            "cyclic",
            "--material-mode",
            "nonlinear",
            "--mapping-policy",
            "cyclic-diagnostic",
            "--axial-compression-mn",
            str(args.axial_compression_mn),
            "--axial-preload-steps",
            str(args.axial_preload_steps),
            "--max-curvature-y",
            str(max(amplitude * 3.0, amplitude)),
            "--section-amplitudes-curvature-y",
            f"{amplitude}",
            "--section-steps",
            str(args.section_steps),
            "--steps-per-segment",
            str(args.section_steps_per_segment),
            "--reversal-substep-factor",
            str(args.continuation_segment_substep_factor),
            "--max-bisections",
            str(args.max_bisections),
        ]
        if args.print_progress:
            command.append("--print-progress")
        elapsed, proc = run_command(command, repo_root)
        (bundle_dir / "audit_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (bundle_dir / "audit_stderr.log").write_text(proc.stderr, encoding="utf-8")
        summary_path = bundle_dir / "benchmark_summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {"status": "failed"}
        row = section_row_from_summary(amplitude, bundle_dir, summary)
        rows.append(row if proc.returncode == 0 else AuditRow(**{**asdict(row), "status": "failed"}))
        if args.print_progress:
            print(
                f"[section] amplitude={amplitude:.4f} 1/m status={rows[-1].status} "
                f"moment_max={rows[-1].section_moment_max_rel_error:.3e} "
                f"tangent_max={rows[-1].section_tangent_max_rel_error:.3e} "
                f"elapsed={elapsed:.3f}s"
            )
    return rows


def run_structural_cases(args: argparse.Namespace, repo_root: Path, root_dir: Path) -> list[AuditRow]:
    rows: list[AuditRow] = []
    structural_root = root_dir / "structural"
    ensure_dir(structural_root)
    launcher = python_launcher_command(args.runner_launcher)
    runner = repo_root / "scripts/run_reduced_rc_column_external_benchmark.py"

    for amplitude in parse_csv_floats(args.structural_amplitudes_mm):
        bundle_dir = structural_root / sanitize_token(amplitude, "tip_mm")
        ensure_dir(bundle_dir)
        command = [
            *launcher,
            str(runner),
            "--output-dir",
            str(bundle_dir),
            "--python-launcher",
            args.python_launcher,
            "--analysis",
            "cyclic",
            "--material-mode",
            "nonlinear",
            "--beam-nodes",
            str(args.beam_nodes),
            "--beam-integration",
            args.beam_integration,
            "--beam-element-family",
            "disp",
            "--geom-transf",
            "linear",
            "--mapping-policy",
            "cyclic-diagnostic",
            "--axial-compression-mn",
            str(args.axial_compression_mn),
            "--axial-preload-steps",
            str(args.axial_preload_steps),
            "--continuation",
            args.continuation,
            "--continuation-segment-substep-factor",
            str(args.continuation_segment_substep_factor),
            "--amplitudes-mm",
            f"{amplitude}",
            "--steps-per-segment",
            str(args.structural_steps_per_segment),
            "--reversal-substep-factor",
            str(args.continuation_segment_substep_factor),
            "--max-bisections",
            str(args.max_bisections),
        ]
        if args.print_progress:
            command.append("--print-progress")
        elapsed, proc = run_command(command, repo_root)
        (bundle_dir / "audit_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (bundle_dir / "audit_stderr.log").write_text(proc.stderr, encoding="utf-8")
        summary_path = bundle_dir / "benchmark_summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {"status": "failed"}
        row = structural_row_from_summary(amplitude, bundle_dir, summary)
        rows.append(row if proc.returncode == 0 else AuditRow(**{**asdict(row), "status": "failed"}))
        if args.print_progress:
            print(
                f"[structural] amplitude={amplitude:.2f} mm status={rows[-1].status} "
                f"hyst_max={rows[-1].hysteresis_max_rel_error:.3e} "
                f"baseM_max={rows[-1].base_moment_max_rel_error:.3e} "
                f"elapsed={elapsed:.3f}s"
            )
    return rows


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    ensure_dir(root_dir)

    section_rows: list[AuditRow] = []
    structural_rows: list[AuditRow] = []
    if args.mode in ("section", "both"):
        section_rows = run_section_cases(args, repo_root, root_dir)
        write_rows_csv(root_dir / "section_amplitude_escalation_summary.csv", section_rows)
    if args.mode in ("structural", "both"):
        structural_rows = run_structural_cases(args, repo_root, root_dir)
        write_rows_csv(root_dir / "structural_amplitude_escalation_summary.csv", structural_rows)

    figure_outputs: dict[str, list[str]] = {}
    if section_rows:
        figure_outputs["section"] = maybe_plot_rows(
            section_rows,
            "section",
            args.figures_dir,
            args.secondary_figures_dir,
        )
    if structural_rows:
        figure_outputs["structural"] = maybe_plot_rows(
            structural_rows,
            "structural",
            args.figures_dir,
            args.secondary_figures_dir,
        )

    payload = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_column_cyclic_amplitude_escalation_audit",
        "mode": args.mode,
        "declared_reference_family": args.beam_integration,
        "declared_continuation": args.continuation,
        "runner_launcher": args.runner_launcher,
        "python_launcher": args.python_launcher,
        "section": make_summary(section_rows, "section") if section_rows else None,
        "structural": make_summary(structural_rows, "structural") if structural_rows else None,
        "figure_outputs": figure_outputs,
    }
    write_json(root_dir / "amplitude_escalation_summary.json", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
