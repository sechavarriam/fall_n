#!/usr/bin/env python3
"""Elastic stiffness control for the reduced RC structural/continuum bridge.

The control intentionally stays in a small elastic displacement range.  Its
purpose is diagnostic: before tuning nonlinear concrete or embedded steel, it
checks whether the structural beam and the 3D continuum share compatible
boundary kinematics and basic elastic stiffness.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CaseResult:
    key: str
    label: str
    output_dir: Path
    process_wall_seconds: float | None
    terminal_drift_mm: float
    terminal_base_shear_kn: float
    secant_stiffness_kn_per_mm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reduced RC elastic axis/BC stiffness control."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--structural-exe",
        default="build/fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument(
        "--continuum-exe",
        default="build/fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
    )
    parser.add_argument("--beam-nodes", type=int, default=10)
    parser.add_argument(
        "--beam-integration",
        choices=("legendre", "lobatto", "radau-left", "radau-right"),
        default="lobatto",
    )
    parser.add_argument(
        "--section-fiber-profile",
        choices=("coarse", "canonical", "fine", "ultra", "ultra-fine"),
        default="fine",
    )
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--longitudinal-bias-power", type=float, default=2.0)
    parser.add_argument("--monotonic-tip-mm", type=float, default=5.0)
    parser.add_argument("--monotonic-steps", type=int, default=5)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--figures-dir", default="doc/figures/validation_reboot")
    parser.add_argument(
        "--secondary-figures-dir",
        default="PhD_Thesis/Figuras/validation_reboot",
    )
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def run_case(command: list[str], output_dir: Path, timeout_seconds: int,
             force_rerun: bool) -> float | None:
    manifest = output_dir / "runtime_manifest.json"
    if manifest.exists() and not force_rerun:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    subprocess.run(command, check=True, timeout=timeout_seconds)
    return time.perf_counter() - start


def read_terminal_stiffness(output_dir: Path) -> tuple[float, float, float]:
    rows = list(csv.DictReader((output_dir / "hysteresis.csv").open()))
    if not rows:
        raise RuntimeError(f"Empty hysteresis.csv in {output_dir}")
    last = rows[-1]
    drift_mm = float(last["drift_m"]) * 1000.0
    shear_kn = float(last["base_shear_MN"]) * 1000.0
    stiffness = abs(shear_kn) / abs(drift_mm) if drift_mm else 0.0
    return drift_mm, shear_kn, stiffness


def structural_command(args: argparse.Namespace, output_dir: Path,
                       clamp_top_rotation: bool) -> list[str]:
    cmd = [
        args.structural_exe,
        "--output-dir", str(output_dir),
        "--analysis", "monotonic",
        "--material-mode", "elasticized",
        "--beam-nodes", str(args.beam_nodes),
        "--beam-integration", args.beam_integration,
        "--solver-policy", "newton-l2-only",
        "--axial-compression-mn", str(args.axial_compression_mn),
        "--axial-preload-steps", str(args.axial_preload_steps),
        "--monotonic-tip-mm", str(args.monotonic_tip_mm),
        "--monotonic-steps", str(args.monotonic_steps),
        "--max-bisections", "8",
        "--section-fiber-profile", args.section_fiber_profile,
    ]
    if clamp_top_rotation:
        cmd.append("--clamp-top-bending-rotation")
    return cmd


def continuum_command(args: argparse.Namespace, output_dir: Path,
                      top_cap_mode: str) -> list[str]:
    return [
        args.continuum_exe,
        "--output-dir", str(output_dir),
        "--analysis", "monotonic",
        "--material-mode", "elasticized",
        "--continuum-kinematics", "corotational",
        "--concrete-profile", "production-stabilized",
        "--reinforcement-mode", "embedded-longitudinal-bars",
        "--rebar-interpolation", "two-node-linear",
        "--rebar-layout", "structural-matched-eight-bar",
        "--host-concrete-zoning-mode", "cover-core-split",
        "--transverse-mesh-mode", "cover-aligned",
        "--hex-order", "hex8",
        "--nx", str(args.nx),
        "--ny", str(args.ny),
        "--nz", str(args.nz),
        "--longitudinal-bias-power", str(args.longitudinal_bias_power),
        "--embedded-boundary-mode", "full-penalty-coupling",
        "--axial-preload-transfer-mode", "host-surface-only",
        "--top-cap-mode", top_cap_mode,
        "--solver-policy", "newton-l2-only",
        "--axial-compression-mn", str(args.axial_compression_mn),
        "--axial-preload-steps", str(args.axial_preload_steps),
        "--monotonic-tip-mm", str(args.monotonic_tip_mm),
        "--monotonic-steps", str(args.monotonic_steps),
        "--max-bisections", "8",
        "--disable-crack-summary-csv",
    ]


def write_summary(output_dir: Path, results: Iterable[CaseResult]) -> dict:
    result_list = list(results)
    by_key = {r.key: r for r in result_list}
    free = by_key["structural_free"].secant_stiffness_kn_per_mm
    clamped = by_key["structural_clamped"].secant_stiffness_kn_per_mm
    cont = by_key["continuum_lateral_top_face"].secant_stiffness_kn_per_mm
    cont_guided = by_key[
        "continuum_uniform_axial_top_cap"].secant_stiffness_kn_per_mm
    summary = {
        "control": "reduced_rc_elastic_axis_stiffness",
        "diagnostic_conclusion": (
            "The current continuum top face prescribes lateral translation only; "
            "it is therefore kinematically comparable to the free-rotation "
            "structural beam.  The uniform-axial-penalty top cap is an explicit "
            "guided-cap audit branch that suppresses top-face axial warping "
            "without prescribing the mean axial shortening."
        ),
        "stiffness_ratios": {
            "continuum_over_structural_free": cont / free if free else None,
            "continuum_guided_cap_over_structural_free":
                cont_guided / free if free else None,
            "structural_clamped_over_structural_free": clamped / free if free else None,
            "continuum_over_structural_clamped": cont / clamped if clamped else None,
            "continuum_guided_cap_over_structural_clamped":
                cont_guided / clamped if clamped else None,
        },
        "cases": [
            {
                "key": r.key,
                "label": r.label,
                "output_dir": str(r.output_dir),
                "process_wall_seconds": r.process_wall_seconds,
                "terminal_drift_mm": r.terminal_drift_mm,
                "terminal_base_shear_kn": r.terminal_base_shear_kn,
                "secant_stiffness_kn_per_mm": r.secant_stiffness_kn_per_mm,
            }
            for r in result_list
        ],
    }
    (output_dir / "elastic_axis_stiffness_control_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    with (output_dir / "elastic_axis_stiffness_control_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "label",
                "terminal_drift_mm",
                "terminal_base_shear_kn",
                "secant_stiffness_kn_per_mm",
                "process_wall_seconds",
                "output_dir",
            ],
        )
        writer.writeheader()
        for r in result_list:
            writer.writerow({
                "key": r.key,
                "label": r.label,
                "terminal_drift_mm": r.terminal_drift_mm,
                "terminal_base_shear_kn": r.terminal_base_shear_kn,
                "secant_stiffness_kn_per_mm": r.secant_stiffness_kn_per_mm,
                "process_wall_seconds": r.process_wall_seconds,
                "output_dir": r.output_dir,
            })
    return summary


def export_figures(output_dir: Path, summary: dict, figure_dirs: list[Path]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for figure_dir in figure_dirs:
        figure_dir.mkdir(parents=True, exist_ok=True)

    labels = [case["label"] for case in summary["cases"]]
    stiffness = [case["secant_stiffness_kn_per_mm"] for case in summary["cases"]]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#2d6cdf", "#d47a22", "#555555", "#118a78"]
    ax.bar(labels, stiffness, color=colors[:len(labels)])
    ax.set_ylabel("Secant stiffness [kN/mm]")
    ax.set_title("Reduced RC elastic axis/BC stiffness control")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    primary = figure_dirs[0] / "reduced_rc_elastic_axis_stiffness_control.png"
    fig.savefig(primary, dpi=180)
    plt.close(fig)
    for figure_dir in figure_dirs[1:]:
        shutil.copy2(primary, figure_dir / primary.name)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_specs = [
        ("structural_free", "Structural free top rotation",
         structural_command(args, output_dir / "structural_free", False)),
        ("structural_clamped", "Structural clamped top rotation",
         structural_command(args, output_dir / "structural_clamped", True)),
        ("continuum_lateral_top_face", "Continuum lateral top face",
         continuum_command(
             args,
             output_dir / "continuum_lateral_top_face",
             "lateral-translation-only")),
        ("continuum_uniform_axial_top_cap", "Continuum guided axial top cap",
         continuum_command(
             args,
             output_dir / "continuum_uniform_axial_top_cap",
             "uniform-axial-penalty-cap")),
    ]

    results: list[CaseResult] = []
    for key, label, command in case_specs:
        case_output = output_dir / key
        wall = run_case(command, case_output, args.timeout_seconds, args.force_rerun)
        drift_mm, shear_kn, stiffness = read_terminal_stiffness(case_output)
        results.append(CaseResult(
            key=key,
            label=label,
            output_dir=case_output,
            process_wall_seconds=wall,
            terminal_drift_mm=drift_mm,
            terminal_base_shear_kn=shear_kn,
            secant_stiffness_kn_per_mm=stiffness,
        ))

    summary = write_summary(output_dir, results)
    if not args.skip_figures:
        export_figures(
            output_dir,
            summary,
            [Path(args.figures_dir), Path(args.secondary_figures_dir)],
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
