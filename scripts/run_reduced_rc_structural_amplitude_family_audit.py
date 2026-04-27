#!/usr/bin/env python3
"""
Compare the reduced RC-column structural amplitude frontier across OpenSees
beam-column families while keeping the fall_n slice fixed.

The audit is intentionally narrow:

  1. run the declared structural benchmark for each amplitude/family pair;
  2. record completion, mismatch, and timing;
  3. optionally probe one stronger fall_n continuation policy when the
     declared policy fails on the fall_n side.

This keeps the stage closure honest: larger-amplitude claims are not promoted
from a single benchmark trace.
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

FAMILY_STYLE = {
    "disp": {"label": "dispBeamColumn", "color": "#0b5fa5", "marker": "o"},
    "force": {"label": "forceBeamColumn", "color": "#d97706", "marker": "s"},
}


def parse_csv(raw: str, caster=float) -> list[float] | list[str]:
    return [caster(token.strip()) for token in raw.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Audit structural cyclic amplitude growth across OpenSees beam-column "
            "families for the reduced RC-column benchmark."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument("--runner-launcher", default="py -3.11")
    parser.add_argument(
        "--beam-element-families",
        default="disp,force",
        help="Comma-separated OpenSees beam-column families to compare.",
    )
    parser.add_argument(
        "--amplitudes-mm",
        default="1.25,2.50,5.00,7.50,10.00,12.50,15.00",
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
    parser.add_argument("--max-bisections", type=int, default=4)
    parser.add_argument(
        "--retry-continuation",
        choices=("none", "arc-length", "segmented"),
        default="arc-length",
    )
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


def safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


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
class StructuralFamilyRow:
    beam_element_family: str
    amplitude_mm: float
    status: str
    bundle_dir: str
    failed_stage: str = ""
    return_code: int = 0
    fall_n_completed: bool = False
    opensees_completed: bool = False
    hysteresis_max_rel_error: float = math.nan
    hysteresis_rms_rel_error: float = math.nan
    base_moment_max_rel_error: float = math.nan
    base_moment_rms_rel_error: float = math.nan
    section_tangent_max_rel_error: float = math.nan
    section_tangent_rms_rel_error: float = math.nan
    preload_tip_axial_displacement_max_rel_error: float = math.nan
    fall_n_process_wall_seconds: float = math.nan
    opensees_process_wall_seconds: float = math.nan
    fall_n_reported_total_wall_seconds: float = math.nan
    opensees_reported_total_wall_seconds: float = math.nan
    process_wall_ratio_fall_n_over_opensees: float = math.nan
    retry_attempted: bool = False
    retry_continuation: str = ""
    retry_status: str = ""
    retry_failed_stage: str = ""
    retry_bundle_dir: str = ""
    retry_hysteresis_max_rel_error: float = math.nan


def row_from_summary(
    family: str,
    amplitude_mm: float,
    bundle_dir: Path,
    summary: dict[str, object],
) -> StructuralFamilyRow:
    return StructuralFamilyRow(
        beam_element_family=family,
        amplitude_mm=amplitude_mm,
        status=str(summary.get("status", "unknown")),
        bundle_dir=str(bundle_dir),
        failed_stage=str(summary.get("failed_stage", "")),
        return_code=int(summary.get("return_code", 0) or 0),
        fall_n_completed=dig(summary, "fall_n", "manifest", "status") == "completed",
        opensees_completed=dig(summary, "opensees", "manifest", "status") == "completed",
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
        section_tangent_max_rel_error=safe_float(
            dig(summary, "comparison", "section_response_tangent", "max_rel_tangent_error")
        ),
        section_tangent_rms_rel_error=safe_float(
            dig(summary, "comparison", "section_response_tangent", "rms_rel_tangent_error")
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
    )


def build_command(
    args: argparse.Namespace,
    repo_root: Path,
    family: str,
    amplitude_mm: float,
    bundle_dir: Path,
    continuation: str,
) -> list[str]:
    runner = repo_root / "scripts" / "run_reduced_rc_column_external_benchmark.py"
    return [
        *shlex.split(args.runner_launcher),
        str(runner),
        "--output-dir",
        str(bundle_dir),
        "--analysis",
        "cyclic",
        "--material-mode",
        "nonlinear",
        "--python-launcher",
        args.python_launcher,
        "--beam-nodes",
        str(args.beam_nodes),
        "--beam-integration",
        args.beam_integration,
        "--beam-element-family",
        family,
        "--geom-transf",
        "linear",
        "--mapping-policy",
        "cyclic-diagnostic",
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--continuation",
        continuation,
        "--continuation-segment-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--amplitudes-mm",
        f"{amplitude_mm}",
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--reversal-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--max-bisections",
        str(args.max_bisections),
        *(["--print-progress"] if args.print_progress else []),
    ]


def maybe_retry(
    args: argparse.Namespace,
    repo_root: Path,
    base_row: StructuralFamilyRow,
    family_root: Path,
) -> StructuralFamilyRow:
    if (
        args.retry_continuation == "none"
        or base_row.status == "completed"
        or base_row.failed_stage != "fall_n"
    ):
        return base_row

    retry_bundle_dir = family_root / f"{sanitize_token(base_row.amplitude_mm, 'tip_mm')}_retry_{args.retry_continuation.replace('-', '_')}"
    command = build_command(
        args,
        repo_root,
        base_row.beam_element_family,
        base_row.amplitude_mm,
        retry_bundle_dir,
        args.retry_continuation,
    )
    _, proc = run_command(command, repo_root)
    (retry_bundle_dir / "audit_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (retry_bundle_dir / "audit_stderr.log").write_text(proc.stderr, encoding="utf-8")
    summary_path = retry_bundle_dir / "benchmark_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"status": "failed"}
    return StructuralFamilyRow(
        **{
            **asdict(base_row),
            "retry_attempted": True,
            "retry_continuation": args.retry_continuation,
            "retry_status": str(summary.get("status", "unknown")),
            "retry_failed_stage": str(summary.get("failed_stage", "")),
            "retry_bundle_dir": str(retry_bundle_dir),
            "retry_hysteresis_max_rel_error": safe_float(
                dig(summary, "comparison", "hysteresis", "max_rel_base_shear_error")
            ),
        }
    )


def write_rows_csv(path: Path, rows: list[StructuralFamilyRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def family_summary(rows: list[StructuralFamilyRow], family: str) -> dict[str, object]:
    scoped = [row for row in rows if row.beam_element_family == family]
    completed = [row for row in scoped if row.status == "completed"]
    failed = [row for row in scoped if row.status != "completed"]
    rescued = [row for row in scoped if row.retry_attempted and row.retry_status == "completed"]

    def worst(metric: str) -> dict[str, object] | None:
        finite_rows = [row for row in completed if math.isfinite(getattr(row, metric))]
        if not finite_rows:
            return None
        row = max(finite_rows, key=lambda item: getattr(item, metric))
        return {"amplitude_mm": row.amplitude_mm, "value": getattr(row, metric), "bundle_dir": row.bundle_dir}

    return {
        "beam_element_family": family,
        "case_count": len(scoped),
        "completed_case_count": len(completed),
        "highest_completed_amplitude_mm": max((row.amplitude_mm for row in completed), default=math.nan),
        "first_failed_case": (
            {
                "amplitude_mm": failed[0].amplitude_mm,
                "failed_stage": failed[0].failed_stage,
                "bundle_dir": failed[0].bundle_dir,
            }
            if failed
            else None
        ),
        "rescued_case_count": len(rescued),
        "highest_rescued_amplitude_mm": max((row.amplitude_mm for row in rescued), default=math.nan),
        "worst_hysteresis_max_rel_error": worst("hysteresis_max_rel_error"),
        "worst_base_moment_max_rel_error": worst("base_moment_max_rel_error"),
        "worst_section_tangent_max_rel_error": worst("section_tangent_max_rel_error"),
        "best_process_time_ratio": min(
            (row.process_wall_ratio_fall_n_over_opensees for row in completed if math.isfinite(row.process_wall_ratio_fall_n_over_opensees)),
            default=math.nan,
        ),
        "worst_process_time_ratio": max(
            (row.process_wall_ratio_fall_n_over_opensees for row in completed if math.isfinite(row.process_wall_ratio_fall_n_over_opensees)),
            default=math.nan,
        ),
    }


def plot_rows(
    rows: list[StructuralFamilyRow],
    figures_dir: Path,
    secondary_figures_dir: Path | None,
) -> list[str]:
    out_dirs = [figures_dir, *( [secondary_figures_dir] if secondary_figures_dir else [] )]
    for out_dir in out_dirs:
        ensure_dir(out_dir)
    outputs: list[str] = []

    def save(fig: plt.Figure, stem: str) -> None:
        primary = figures_dir / f"{stem}.png"
        fig.tight_layout()
        fig.savefig(primary, dpi=180, bbox_inches="tight")
        outputs.append(str(primary))
        for out_dir in out_dirs:
            if out_dir == figures_dir:
                continue
            fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    families = [family for family in FAMILY_STYLE if any(row.beam_element_family == family for row in rows)]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for family in families:
        scoped = [row for row in rows if row.beam_element_family == family and row.status == "completed"]
        if not scoped:
            continue
        xs = [row.amplitude_mm for row in scoped]
        style = FAMILY_STYLE[family]
        axes[0].plot(xs, [row.hysteresis_max_rel_error for row in scoped], color=style["color"], marker=style["marker"], label=style["label"])
        axes[0].plot(xs, [row.base_moment_max_rel_error for row in scoped], color=style["color"], linestyle="--", marker=style["marker"], alpha=0.8)
        axes[1].plot(xs, [row.section_tangent_max_rel_error for row in scoped], color=style["color"], marker=style["marker"], label=style["label"])
    axes[0].set_xlabel("Tip amplitude [mm]")
    axes[0].set_ylabel("Relative error")
    axes[0].set_title("Large-amplitude structural mismatch growth\nsolid: hysteresis, dashed: base moment")
    axes[0].legend()
    axes[1].set_xlabel("Tip amplitude [mm]")
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("Section tangent mismatch by family")
    axes[1].legend()
    save(fig, "reduced_rc_structural_amplitude_family_errors")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for family in families:
        scoped = [row for row in rows if row.beam_element_family == family and row.status == "completed"]
        if not scoped:
            continue
        xs = [row.amplitude_mm for row in scoped]
        style = FAMILY_STYLE[family]
        axes[0].plot(xs, [row.fall_n_process_wall_seconds for row in scoped], color=style["color"], marker=style["marker"], label=f"fall_n / {style['label']}")
        axes[0].plot(xs, [row.opensees_process_wall_seconds for row in scoped], color=style["color"], linestyle="--", marker=style["marker"], alpha=0.8)
        axes[1].plot(xs, [row.process_wall_ratio_fall_n_over_opensees for row in scoped], color=style["color"], marker=style["marker"], label=style["label"])
    axes[0].set_xlabel("Tip amplitude [mm]")
    axes[0].set_ylabel("Process wall time [s]")
    axes[0].set_title("Compute-time growth\nsolid: fall_n, dashed: OpenSees")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].axhline(1.0, color="k", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Tip amplitude [mm]")
    axes[1].set_ylabel(r"$t_{fall_n} / t_{OpenSees}$")
    axes[1].set_title("Process-time ratio by family")
    axes[1].legend()
    save(fig, "reduced_rc_structural_amplitude_family_timing")

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    y_positions = {family: idx for idx, family in enumerate(families)}
    for family in families:
        style = FAMILY_STYLE[family]
        family_rows = [row for row in rows if row.beam_element_family == family]
        for row in family_rows:
            marker = style["marker"] if row.status == "completed" else "x"
            face = style["color"] if row.status == "completed" else "none"
            ax.scatter(row.amplitude_mm, y_positions[family], s=70, marker=marker, color=style["color"], facecolors=face)
            if row.retry_attempted and row.retry_status == "completed":
                ax.scatter(row.amplitude_mm, y_positions[family] + 0.08, s=40, marker="^", color="#2f855a")
    ax.set_yticks(list(y_positions.values()), [FAMILY_STYLE[family]["label"] for family in families])
    ax.set_xlabel("Tip amplitude [mm]")
    ax.set_title("Completion frontier by family\nx: declared continuation failed, triangle: retry rescued")
    ax.grid(True, axis="x", alpha=0.3)
    save(fig, "reduced_rc_structural_amplitude_family_frontier")

    return outputs


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    ensure_dir(root_dir)

    amplitudes = parse_csv(args.amplitudes_mm)
    families = parse_csv(args.beam_element_families, caster=str)
    rows: list[StructuralFamilyRow] = []

    for family in families:
        family_root = root_dir / family
        ensure_dir(family_root)
        for amplitude in amplitudes:
            bundle_dir = family_root / sanitize_token(float(amplitude), "tip_mm")
            command = build_command(args, repo_root, family, float(amplitude), bundle_dir, args.continuation)
            elapsed, proc = run_command(command, repo_root)
            (bundle_dir / "audit_stdout.log").write_text(proc.stdout, encoding="utf-8")
            (bundle_dir / "audit_stderr.log").write_text(proc.stderr, encoding="utf-8")
            summary_path = bundle_dir / "benchmark_summary.json"
            summary = read_json(summary_path) if summary_path.exists() else {"status": "failed"}
            row = row_from_summary(family, float(amplitude), bundle_dir, summary)
            row = row if proc.returncode == 0 else StructuralFamilyRow(**{**asdict(row), "status": "failed"})
            row = maybe_retry(args, repo_root, row, family_root)
            rows.append(row)
            if args.print_progress:
                print(
                    f"[{family}] amplitude={float(amplitude):.2f} mm status={row.status} "
                    f"hyst_max={row.hysteresis_max_rel_error:.3e} elapsed={elapsed:.3f}s "
                    f"retry={row.retry_status or 'n/a'}"
                )

    write_rows_csv(root_dir / "structural_amplitude_family_audit.csv", rows)
    figures = plot_rows(rows, args.figures_dir, args.secondary_figures_dir)
    payload = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_structural_amplitude_family_audit",
        "declared_reference_family": args.beam_integration,
        "declared_continuation": args.continuation,
        "retry_continuation": args.retry_continuation,
        "families": {family: family_summary(rows, family) for family in families},
        "artifacts": {
            "csv": str(root_dir / "structural_amplitude_family_audit.csv"),
            "figures": figures,
        },
    }
    write_json(root_dir / "structural_amplitude_family_audit_summary.json", payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
