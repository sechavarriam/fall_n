#!/usr/bin/env python3
"""
Audit the first failing cyclic section-reversal frontier of the external
OpenSees bridge under a small set of materially meaningful concrete profiles.

The goal is deliberately narrow: determine whether the first failing reversal
of the zeroLengthSection comparator at the canonical reduced-column section
amplitude is rescued by a scientifically plausible constitutive override, or
whether the frontier remains open beyond the fiber-level uniaxial mismatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrontierProfile:
    label: str
    description: str
    concrete_model: str = "Concrete02"
    concrete_lambda: float = 0.1
    concrete_ft_ratio: float = 0.02
    concrete_softening_multiplier: float = 0.5
    concrete_unconfined_residual_ratio: float = 0.2
    concrete_confined_residual_ratio: float = 0.2
    concrete_ultimate_strain: float = -0.006
    mapping_policy: str = "cyclic-diagnostic"


@dataclass(frozen=True)
class FrontierRow:
    label: str
    description: str
    status: str
    fall_n_completed: bool
    opensees_completed: bool
    failed_stage: str
    failure_step: int
    failure_target_curvature_y: float
    failure_trial_actual_curvature_y: float
    max_rel_moment_error: float
    rms_rel_moment_error: float
    max_rel_tangent_error: float
    rms_rel_tangent_error: float
    fall_n_process_wall_seconds: float
    opensees_process_wall_seconds: float
    fall_n_reported_total_wall_seconds: float
    opensees_reported_total_wall_seconds: float
    concrete_model: str
    concrete_lambda: float
    concrete_ft_ratio: float
    concrete_softening_multiplier: float
    concrete_unconfined_residual_ratio: float
    concrete_confined_residual_ratio: float
    concrete_ultimate_strain: float
    bundle_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced RC section reversal-frontier audit over the first "
            "externally failing cyclic amplitude."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--python-launcher",
        default="C:\\Users\\Sebastian\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_reduced_rc_column_section_external_benchmark.py"),
    )
    parser.add_argument("--amplitude-curvature-y", type=float, default=0.02)
    parser.add_argument("--max-curvature-y", type=float, default=0.03)
    parser.add_argument("--section-steps", type=int, default=120)
    parser.add_argument("--steps-per-segment", type=int, default=4)
    parser.add_argument("--reversal-substep-factor", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def profile_catalog() -> tuple[FrontierProfile, ...]:
    return (
        FrontierProfile(
            label="cyclic_diagnostic",
            description="Declared reduced-tension Concrete02 bridge currently used in the cyclic external benchmark.",
        ),
        FrontierProfile(
            label="concrete02_no_tension",
            description="Concrete02 with zero tensile branch; matches the problematic fiber replay much more closely.",
            concrete_ft_ratio=0.0,
        ),
        FrontierProfile(
            label="concrete02_lambda_0p5",
            description="Concrete02 with stronger unloading-memory ratio lambda=0.5 and reduced tension.",
            concrete_lambda=0.5,
        ),
        FrontierProfile(
            label="concrete01_ksp_like",
            description="Concrete01 compression-only bridge as a Kent-Park-like external control.",
            concrete_model="Concrete01",
            concrete_lambda=0.1,
            concrete_ft_ratio=0.1,
            concrete_softening_multiplier=0.1,
            concrete_unconfined_residual_ratio=0.2,
            concrete_confined_residual_ratio=0.2,
            concrete_ultimate_strain=-0.006,
        ),
    )


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_summary(
    profile: FrontierProfile,
    bundle_dir: Path,
    summary: dict[str, object],
) -> FrontierRow:
    falln_manifest_path = bundle_dir / "fall_n" / "runtime_manifest.json"
    opensees_manifest_path = bundle_dir / "opensees" / "reference_manifest.json"
    falln_manifest = (
        load_json(falln_manifest_path) if falln_manifest_path.exists() else {}
    )
    opensees_manifest = (
        load_json(opensees_manifest_path) if opensees_manifest_path.exists() else {}
    )
    return FrontierRow(
        label=profile.label,
        description=profile.description,
        status=str(summary.get("status", "unknown")),
        fall_n_completed=str(summary.get("fall_n_status", "")) == "completed"
        or str(dig(summary, "fall_n", "manifest", "status")) == "completed"
        or str(falln_manifest.get("status", "")) == "completed",
        opensees_completed=str(summary.get("opensees_status", "")) == "completed"
        or str(dig(summary, "opensees", "manifest", "status")) == "completed"
        or str(opensees_manifest.get("status", "")) == "completed",
        failed_stage=str(summary.get("failed_stage", "")),
        failure_step=int(safe_float(summary.get("failure_step", math.nan)))
        if math.isfinite(safe_float(summary.get("failure_step", math.nan)))
        else -1,
        failure_target_curvature_y=safe_float(summary.get("failure_target_curvature_y", math.nan)),
        failure_trial_actual_curvature_y=(
            safe_float(dig(summary, "opensees", "manifest", "failure_trial_actual_curvature_y"))
            if math.isfinite(
                safe_float(dig(summary, "opensees", "manifest", "failure_trial_actual_curvature_y"))
            )
            else safe_float(opensees_manifest.get("failure_trial_actual_curvature_y", math.nan))
        ),
        max_rel_moment_error=safe_float(
            dig(summary, "comparison", "section_moment_curvature", "max_rel_moment_error")
        ),
        rms_rel_moment_error=safe_float(
            dig(summary, "comparison", "section_moment_curvature", "rms_rel_moment_error")
        ),
        max_rel_tangent_error=safe_float(
            dig(summary, "comparison", "section_tangent", "max_rel_tangent_error")
        ),
        rms_rel_tangent_error=safe_float(
            dig(summary, "comparison", "section_tangent", "rms_rel_tangent_error")
        ),
        fall_n_process_wall_seconds=(
            safe_float(dig(summary, "fall_n", "process_wall_seconds"))
            if math.isfinite(safe_float(dig(summary, "fall_n", "process_wall_seconds")))
            else math.nan
        ),
        opensees_process_wall_seconds=(
            safe_float(dig(summary, "opensees", "process_wall_seconds"))
            if math.isfinite(safe_float(dig(summary, "opensees", "process_wall_seconds")))
            else math.nan
        ),
        fall_n_reported_total_wall_seconds=(
            safe_float(dig(summary, "fall_n", "manifest", "timing", "total_wall_seconds"))
            if math.isfinite(
                safe_float(dig(summary, "fall_n", "manifest", "timing", "total_wall_seconds"))
            )
            else safe_float(dig(falln_manifest, "timing", "total_wall_seconds"))
        ),
        opensees_reported_total_wall_seconds=(
            safe_float(dig(summary, "opensees", "manifest", "timing", "total_wall_seconds"))
            if math.isfinite(
                safe_float(dig(summary, "opensees", "manifest", "timing", "total_wall_seconds"))
            )
            else safe_float(dig(opensees_manifest, "timing", "total_wall_seconds"))
        ),
        concrete_model=profile.concrete_model,
        concrete_lambda=profile.concrete_lambda,
        concrete_ft_ratio=profile.concrete_ft_ratio,
        concrete_softening_multiplier=profile.concrete_softening_multiplier,
        concrete_unconfined_residual_ratio=profile.concrete_unconfined_residual_ratio,
        concrete_confined_residual_ratio=profile.concrete_confined_residual_ratio,
        concrete_ultimate_strain=profile.concrete_ultimate_strain,
        bundle_dir=str(bundle_dir),
    )


def write_rows_csv(path: Path, rows: list[FrontierRow]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def maybe_plot(rows: list[FrontierRow], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [row.label for row in rows]
    failure_targets = [
        row.failure_target_curvature_y
        if math.isfinite(row.failure_target_curvature_y)
        else 0.0
        for row in rows
    ]
    completed = [1 if row.status == "completed" else 0 for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), constrained_layout=True)

    axes[0].bar(labels, failure_targets, color="#C47F00")
    axes[0].set_ylabel("Failure target $\\kappa_y$ [1/m]")
    axes[0].set_title("Section reversal frontier by external concrete profile")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, completed, color=["#2A7F62" if value else "#B03A2E" for value in completed])
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_yticks([0, 1], labels=["failed", "completed"])
    axes[1].set_ylabel("Benchmark status")
    axes[1].tick_params(axis="x", rotation=20)

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    ensure_dir(root_dir)

    rows: list[FrontierRow] = []
    runner = (repo_root / args.runner).resolve()

    for profile in profile_catalog():
        bundle_dir = root_dir / profile.label
        ensure_dir(bundle_dir)
        command = [
            sys.executable,
            str(runner),
            "--output-dir",
            str(bundle_dir),
            "--analysis",
            "cyclic",
            "--material-mode",
            "nonlinear",
            "--python-launcher",
            args.python_launcher,
            "--axial-compression-mn",
            str(args.axial_compression_mn),
            "--axial-preload-steps",
            str(args.axial_preload_steps),
            "--max-curvature-y",
            str(args.max_curvature_y),
            "--section-amplitudes-curvature-y",
            str(args.amplitude_curvature_y),
            "--section-steps",
            str(args.section_steps),
            "--steps-per-segment",
            str(args.steps_per_segment),
            "--reversal-substep-factor",
            str(args.reversal_substep_factor),
            "--max-bisections",
            str(args.max_bisections),
            "--mapping-policy",
            profile.mapping_policy,
            "--concrete-model",
            profile.concrete_model,
            "--concrete-lambda",
            str(profile.concrete_lambda),
            "--concrete-ft-ratio",
            str(profile.concrete_ft_ratio),
            "--concrete-softening-multiplier",
            str(profile.concrete_softening_multiplier),
            "--concrete-unconfined-residual-ratio",
            str(profile.concrete_unconfined_residual_ratio),
            "--concrete-confined-residual-ratio",
            str(profile.concrete_confined_residual_ratio),
            "--concrete-ultimate-strain",
            str(profile.concrete_ultimate_strain),
        ]
        if args.print_progress:
            command.append("--print-progress")

        elapsed, proc = run_command(command, repo_root)
        (bundle_dir / "audit_stdout.log").write_text(proc.stdout, encoding="utf-8")
        (bundle_dir / "audit_stderr.log").write_text(proc.stderr, encoding="utf-8")
        summary_path = bundle_dir / "benchmark_summary.json"
        if not summary_path.exists():
            summary = {
                "status": "process_failed",
                "failed_stage": "runner",
                "failure_step": math.nan,
                "failure_target_curvature_y": math.nan,
                "runner_return_code": proc.returncode,
                "runner_wall_seconds": elapsed,
            }
        else:
            summary = load_json(summary_path)
            summary["runner_return_code"] = proc.returncode
            summary["runner_wall_seconds"] = elapsed
            write_json(summary_path, summary)

        rows.append(row_from_summary(profile, bundle_dir, summary))

    summary_payload = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_section_reversal_frontier_audit",
        "amplitude_curvature_y": args.amplitude_curvature_y,
        "profiles": [asdict(row) for row in rows],
        "rescued_profile_labels": [row.label for row in rows if row.status == "completed"],
        "all_profiles_failed": all(row.status != "completed" for row in rows),
        "best_failure_target_curvature_y": max(
            (
                row.failure_target_curvature_y
                for row in rows
                if math.isfinite(row.failure_target_curvature_y)
            ),
            default=math.nan,
        ),
    }
    write_json(root_dir / "reversal_frontier_summary.json", summary_payload)
    write_rows_csv(root_dir / "reversal_frontier_summary.csv", rows)
    maybe_plot(rows, root_dir / "reversal_frontier_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
