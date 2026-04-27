#!/usr/bin/env python3
"""
Run a high-resolution structural reference audit for the reduced RC column.

The audit deliberately separates two ideas that were previously conflated:

  * interpolation order inside one fall_n TimoshenkoBeamN element; and
  * physical resolution of the plastic hinge by using several structural
    elements along the column height.

The produced bundle contains one fall_n multi-element structural run, one
OpenSeesPy multi-element high-fidelity run, a small set of global/station
comparison metrics, and hysteresis plots that can be reused by the continuum
validation stage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from python_launcher_utils import python_launcher_command


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.bbox": "tight",
        "figure.dpi": 140,
        "savefig.dpi": 220,
    }
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run fall_n and OpenSeesPy high-resolution multi-element "
            "structural references for the reduced RC-column benchmark."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=repo_root
        / "build_stage8_validation"
        / "fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument("--analysis", choices=("monotonic", "cyclic"), default="cyclic")
    parser.add_argument("--material-mode", choices=("nonlinear", "elasticized"), default="nonlinear")
    parser.add_argument("--solver-policy", default="newton-l2-lu-symbolic-reuse-only")
    parser.add_argument("--beam-nodes", type=int, default=4)
    parser.add_argument("--structural-element-count", type=int, default=12)
    parser.add_argument("--beam-integration", choices=("legendre", "lobatto"), default="lobatto")
    parser.add_argument("--opensees-beam-element-family", choices=("disp", "force"), default="force")
    parser.add_argument("--opensees-model-dimension", choices=("2d", "3d"), default="3d")
    parser.add_argument("--opensees-solver-profile-family", default="robust-large-amplitude")
    parser.add_argument(
        "--opensees-lateral-control-mode",
        choices=("displacement-control", "sp-path"),
        default="displacement-control",
    )
    parser.add_argument("--opensees-element-local-iterations", type=int, default=10)
    parser.add_argument("--opensees-element-local-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--concrete-model", choices=("Elastic", "Concrete01", "Concrete02"), default=None)
    parser.add_argument("--concrete-lambda", type=float, default=None)
    parser.add_argument("--concrete-ft-ratio", type=float, default=None)
    parser.add_argument("--concrete-softening-multiplier", type=float, default=None)
    parser.add_argument("--concrete-unconfined-residual-ratio", type=float, default=None)
    parser.add_argument("--concrete-confined-residual-ratio", type=float, default=None)
    parser.add_argument("--concrete-ultimate-strain", type=float, default=None)
    parser.add_argument("--steel-r0", type=float, default=None)
    parser.add_argument("--steel-cr1", type=float, default=None)
    parser.add_argument("--steel-cr2", type=float, default=None)
    parser.add_argument("--steel-a1", type=float, default=None)
    parser.add_argument("--steel-a2", type=float, default=None)
    parser.add_argument("--steel-a3", type=float, default=None)
    parser.add_argument("--steel-a4", type=float, default=None)
    parser.add_argument("--geom-transf", choices=("linear", "pdelta"), default="linear")
    parser.add_argument("--integration-points", type=int)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--continuation", choices=("monolithic", "segmented", "reversal-guarded"), default="reversal-guarded")
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument("--monotonic-tip-mm", type=float, default=200.0)
    parser.add_argument("--monotonic-steps", type=int, default=24)
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--reversal-substep-factor", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def finite(value: Any, fallback: float = math.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if math.isfinite(parsed) else fallback


def relative_error(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(rhs), 1.0e-12)


def amplitude_suffix(args: argparse.Namespace) -> str:
    if args.analysis == "monotonic":
        peak_mm = float(args.monotonic_tip_mm)
    else:
        amplitudes = [
            abs(float(token.strip()))
            for token in args.amplitudes_mm.split(",")
            if token.strip()
        ]
        peak_mm = max(amplitudes) if amplitudes else 0.0
    label = f"{peak_mm:.3f}".rstrip("0").rstrip(".").replace(".", "p")
    concrete = args.concrete_model or "defaultConcrete"
    opensees_ip = args.integration_points or max(args.beam_nodes - 1, 2)
    parts = [
        args.analysis,
        f"{label}mm",
        f"falln_n{args.beam_nodes}_e{max(args.structural_element_count, 1)}",
        args.beam_integration,
        f"ops_{args.opensees_beam_element_family}{args.opensees_model_dimension}",
        f"ip{opensees_ip}",
        concrete,
    ]
    return "_".join(parts)


def compare_by_step(
    lhs_rows: list[dict[str, Any]],
    rhs_rows: list[dict[str, Any]],
    lhs_field: str,
    rhs_field: str,
    label: str,
) -> dict[str, Any]:
    lhs = {int(row["step"]): finite(row[lhs_field]) for row in lhs_rows}
    rhs = {int(row["step"]): finite(row[rhs_field]) for row in rhs_rows}
    steps = sorted(set(lhs) & set(rhs))
    if not steps:
        return {"shared_step_count": 0}
    peak = max(abs(rhs[step]) for step in steps)
    floor = max(0.05 * peak, 1.0e-12)
    active = [step for step in steps if abs(rhs[step]) >= floor] or steps
    rel = [relative_error(lhs[step], rhs[step]) for step in active]
    abs_errors = [abs(lhs[step] - rhs[step]) for step in steps]
    return {
        "shared_step_count": len(steps),
        "active_step_count": len(active),
        "activity_floor": floor,
        f"max_abs_{label}": max(abs_errors),
        f"max_rel_{label}": max(rel),
        f"rms_rel_{label}": math.sqrt(sum(v * v for v in rel) / len(rel)),
    }


def base_station_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for step in sorted(by_step):
        selected.append(
            min(
                by_step[step],
                key=lambda row: (finite(row.get("xi", 0.0)), finite(row.get("section_gp", 0.0))),
            )
        )
    return selected


def station_parity(lhs_path: Path, rhs_path: Path) -> dict[str, Any]:
    if not lhs_path.exists() or not rhs_path.exists():
        return {"status": "missing_station_layout"}
    lhs = read_csv_rows(lhs_path)
    rhs = read_csv_rows(rhs_path)
    count = min(len(lhs), len(rhs))
    diffs = [
        abs(finite(lhs[idx].get("xi")) - finite(rhs[idx].get("xi")))
        for idx in range(count)
    ]
    return {
        "fall_n_station_count": len(lhs),
        "opensees_station_count": len(rhs),
        "common_station_count": count,
        "max_abs_xi_mismatch": max(diffs) if diffs else math.nan,
        "same_station_count": len(lhs) == len(rhs),
    }


def build_falln_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    command = [
        str(args.falln_exe),
        "--analysis",
        args.analysis,
        "--output-dir",
        str(out_dir),
        "--material-mode",
        args.material_mode,
        "--solver-policy",
        args.solver_policy,
        "--beam-nodes",
        str(args.beam_nodes),
        "--structural-element-count",
        str(max(args.structural_element_count, 1)),
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
        "--monotonic-tip-mm",
        str(args.monotonic_tip_mm),
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
        "--lateral-control-mode",
        args.opensees_lateral_control_mode,
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def build_opensees_command(args: argparse.Namespace, out_dir: Path, repo_root: Path) -> list[str]:
    integration_points = args.integration_points or max(args.beam_nodes - 1, 2)
    command = [
        *python_launcher_command(args.python_launcher),
        str(repo_root / "scripts" / "opensees_reduced_rc_column_hifi_reference.py"),
        "--analysis",
        args.analysis,
        "--output-dir",
        str(out_dir),
        "--material-mode",
        args.material_mode,
        "--model-dimension",
        args.opensees_model_dimension,
        "--solver-profile-family",
        args.opensees_solver_profile_family,
        "--beam-element-family",
        args.opensees_beam_element_family,
        "--beam-integration",
        args.beam_integration,
        "--integration-points",
        str(integration_points),
        "--structural-element-count",
        str(max(args.structural_element_count, 1)),
        "--element-local-iterations",
        str(args.opensees_element_local_iterations),
        "--element-local-tolerance",
        str(args.opensees_element_local_tolerance),
        "--geom-transf",
        args.geom_transf,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        str(args.monotonic_tip_mm),
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--reversal-substep-factor",
        str(args.reversal_substep_factor),
        "--max-bisections",
        str(args.max_bisections),
    ]
    if args.print_progress:
        command.append("--print-progress")
    for flag, value in (
        ("--concrete-model", args.concrete_model),
        ("--concrete-lambda", args.concrete_lambda),
        ("--concrete-ft-ratio", args.concrete_ft_ratio),
        ("--concrete-softening-multiplier", args.concrete_softening_multiplier),
        ("--concrete-unconfined-residual-ratio", args.concrete_unconfined_residual_ratio),
        ("--concrete-confined-residual-ratio", args.concrete_confined_residual_ratio),
        ("--concrete-ultimate-strain", args.concrete_ultimate_strain),
        ("--steel-r0", args.steel_r0),
        ("--steel-cr1", args.steel_cr1),
        ("--steel-cr2", args.steel_cr2),
        ("--steel-a1", args.steel_a1),
        ("--steel-a2", args.steel_a2),
        ("--steel-a3", args.steel_a3),
        ("--steel-a4", args.steel_a4),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    return command


def extract_base_moment_curve(section_path: Path, out_path: Path) -> list[dict[str, Any]]:
    rows = base_station_rows(read_csv_rows(section_path))
    curve = [
        {
            "step": int(row["step"]),
            "p": finite(row["p"]),
            "drift_m": finite(row["drift_m"]),
            "section_gp": int(finite(row["section_gp"], 0.0)),
            "xi": finite(row["xi"]),
            "curvature_y": finite(row["curvature_y"]),
            "moment_y_MNm": finite(row["moment_y_MNm"]),
            "axial_force_MN": finite(row["axial_force_MN"]),
        }
        for row in rows
    ]
    write_csv(out_path, curve)
    return curve


def plot_overlay(
    out_stem: str,
    falln_hysteresis: list[dict[str, Any]],
    opensees_hysteresis: list[dict[str, Any]],
    falln_moment: list[dict[str, Any]],
    opensees_moment: list[dict[str, Any]],
    figures_dir: Path,
    secondary_figures_dir: Path,
) -> list[str]:
    ensure_dir(figures_dir)
    ensure_dir(secondary_figures_dir)
    outputs: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot(
        [1.0e3 * finite(row["drift_m"]) for row in falln_hysteresis],
        [1.0e3 * finite(row["base_shear_MN"]) for row in falln_hysteresis],
        color="#0b5fa5",
        label="fall_n multi-element",
    )
    axes[0].plot(
        [1.0e3 * finite(row["drift_m"]) for row in opensees_hysteresis],
        [1.0e3 * finite(row["base_shear_MN"]) for row in opensees_hysteresis],
        color="#d97706",
        linestyle="--",
        label="OpenSees hi-fi",
    )
    axes[0].set_xlabel("Tip drift [mm]")
    axes[0].set_ylabel("Base shear [kN]")
    axes[0].set_title("Global hysteresis")
    axes[0].legend()

    axes[1].plot(
        [finite(row["curvature_y"]) for row in falln_moment],
        [1.0e3 * finite(row["moment_y_MNm"]) for row in falln_moment],
        color="#0b5fa5",
        label="fall_n base station",
    )
    axes[1].plot(
        [finite(row["curvature_y"]) for row in opensees_moment],
        [1.0e3 * finite(row["moment_y_MNm"]) for row in opensees_moment],
        color="#d97706",
        linestyle="--",
        label="OpenSees base station",
    )
    axes[1].set_xlabel(r"Base curvature $\kappa_y$ [1/m]")
    axes[1].set_ylabel("Base moment [kNm]")
    axes[1].set_title("Base moment-curvature")
    axes[1].legend()

    for root in (figures_dir, secondary_figures_dir):
        path = root / f"{out_stem}.png"
        fig.savefig(path)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root = args.output_dir.resolve()
    falln_dir = root / "fall_n_multielement"
    opensees_dir = root / "opensees_hifi"
    ensure_dir(root)
    ensure_dir(falln_dir)
    ensure_dir(opensees_dir)

    falln_command = build_falln_command(args, falln_dir)
    falln_elapsed, falln_proc = run_command(falln_command, repo_root)
    (falln_dir / "stdout.log").write_text(falln_proc.stdout, encoding="utf-8")
    (falln_dir / "stderr.log").write_text(falln_proc.stderr, encoding="utf-8")
    if falln_proc.returncode != 0:
        write_json(
            root / "structural_multielement_hifi_audit_summary.json",
            {
                "status": "failed",
                "failed_stage": "fall_n",
                "return_code": falln_proc.returncode,
                "fall_n_command": falln_command,
            },
        )
        return falln_proc.returncode

    opensees_command = build_opensees_command(args, opensees_dir, repo_root)
    opensees_elapsed, opensees_proc = run_command(opensees_command, repo_root)
    (opensees_dir / "stdout.log").write_text(opensees_proc.stdout, encoding="utf-8")
    (opensees_dir / "stderr.log").write_text(opensees_proc.stderr, encoding="utf-8")
    if opensees_proc.returncode != 0:
        write_json(
            root / "structural_multielement_hifi_audit_summary.json",
            {
                "status": "failed",
                "failed_stage": "opensees",
                "return_code": opensees_proc.returncode,
                "fall_n_command": falln_command,
                "opensees_command": opensees_command,
            },
        )
        return opensees_proc.returncode

    falln_manifest = read_json(falln_dir / "runtime_manifest.json")
    opensees_manifest = read_json(opensees_dir / "reference_manifest.json")
    falln_hysteresis = read_csv_rows(falln_dir / "comparison_hysteresis.csv")
    opensees_hysteresis = read_csv_rows(opensees_dir / "hysteresis.csv")
    falln_moment = extract_base_moment_curve(
        falln_dir / "section_response.csv",
        root / "fall_n_base_moment_curvature.csv",
    )
    opensees_moment = extract_base_moment_curve(
        opensees_dir / "section_response.csv",
        root / "opensees_base_moment_curvature.csv",
    )

    figure_outputs = plot_overlay(
        f"reduced_rc_structural_multielement_hifi_overlay_{amplitude_suffix(args)}",
        falln_hysteresis,
        opensees_hysteresis,
        falln_moment,
        opensees_moment,
        args.figures_dir,
        args.secondary_figures_dir,
    )

    timing = {
        "fall_n_process_wall_seconds": falln_elapsed,
        "opensees_process_wall_seconds": opensees_elapsed,
        "fall_n_reported_total_wall_seconds": finite(
            dict(falln_manifest.get("timing", {})).get("total_wall_seconds")
        ),
        "opensees_reported_total_wall_seconds": finite(
            dict(opensees_manifest.get("timing", {})).get("total_wall_seconds")
        ),
    }
    ratio = (
        falln_elapsed / opensees_elapsed
        if abs(opensees_elapsed) > 1.0e-12
        else math.nan
    )
    payload = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_structural_multielement_hifi_reference",
        "interpretation": (
            "Multi-element structural references are promoted for global "
            "hysteresis and plastic-hinge envelope comparisons before using "
            "the continuum as a local multiscale model."
        ),
        "fall_n": {
            "dir": str(falln_dir),
            "command": falln_command,
            "manifest": falln_manifest,
        },
        "opensees": {
            "dir": str(opensees_dir),
            "command": opensees_command,
            "manifest": opensees_manifest,
        },
        "comparison": {
            "hysteresis": compare_by_step(
                opensees_hysteresis,
                falln_hysteresis,
                "base_shear_MN",
                "base_shear_MN",
                "base_shear_error",
            ),
            "base_moment": compare_by_step(
                opensees_moment,
                falln_moment,
                "moment_y_MNm",
                "moment_y_MNm",
                "base_moment_error",
            ),
            "base_curvature": compare_by_step(
                opensees_moment,
                falln_moment,
                "curvature_y",
                "curvature_y",
                "base_curvature_error",
            ),
            "station_parity": station_parity(
                falln_dir / "section_station_layout.csv",
                opensees_dir / "section_station_layout.csv",
            ),
        },
        "timing": {
            **timing,
            "fall_n_over_opensees_process_ratio": ratio,
        },
        "artifacts": {
            "fall_n_hysteresis": str(falln_dir / "comparison_hysteresis.csv"),
            "opensees_hysteresis": str(opensees_dir / "hysteresis.csv"),
            "fall_n_base_moment_curvature": str(root / "fall_n_base_moment_curvature.csv"),
            "opensees_base_moment_curvature": str(root / "opensees_base_moment_curvature.csv"),
            "figures": figure_outputs,
        },
    }
    write_json(root / "structural_multielement_hifi_audit_summary.json", payload)
    print(
        "Structural multi-element hi-fi audit completed:",
        f"fall_n={falln_elapsed:.3f}s",
        f"OpenSees={opensees_elapsed:.3f}s",
        f"ratio={ratio:.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
