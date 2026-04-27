#!/usr/bin/env python3
"""
Run the first reduced-RC continuum bring-up matrix.

The goal of this audit is deliberately narrow:

1. prove that the same continuum driver can run a plain concrete slice and an
   embedded-bar slice without changing executable or solver surface;
2. freeze the correction that made the monotonic path genuinely monotonic
   instead of a disguised cyclic branch;
3. preserve early timing / peak-force evidence before the phase-4 continuum
   campaign grows to Hex20/Hex27 and richer material comparisons.
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Freeze the first reduced RC continuum bring-up matrix."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-exe",
        type=Path,
        default=repo_root
        / "build"
        / "fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
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


@dataclass(frozen=True)
class ContinuumBringupCase:
    key: str
    analysis: str
    reinforcement_mode: str
    material_mode: str
    hex_order: str
    nx: int
    ny: int
    nz: int
    solver_policy: str
    continuation: str
    continuation_segment_substep_factor: int
    monotonic_tip_mm: float = 0.0
    monotonic_steps: int = 0
    amplitudes_mm: tuple[float, ...] = ()
    steps_per_segment: int = 0
    max_bisections: int = 6
    embedded_boundary_mode: str = "dirichlet-rebar-endcap"


@dataclass(frozen=True)
class ContinuumBringupRow:
    key: str
    analysis: str
    reinforcement_mode: str
    material_mode: str
    hex_order: str
    mesh: str
    status: str
    completed_successfully: bool
    runtime_steps: int
    max_abs_base_shear_mn: float
    max_abs_top_rebar_face_gap_m: float
    solve_wall_seconds: float
    total_wall_seconds: float
    output_dir: str
    return_code: int


def build_command(exe: Path, case: ContinuumBringupCase, output_dir: Path) -> list[str]:
    command = [
        str(exe),
        "--output-dir",
        str(output_dir),
        "--analysis",
        case.analysis,
        "--reinforcement-mode",
        case.reinforcement_mode,
        "--material-mode",
        case.material_mode,
        "--hex-order",
        case.hex_order,
        "--nx",
        str(case.nx),
        "--ny",
        str(case.ny),
        "--nz",
        str(case.nz),
        "--embedded-boundary-mode",
        case.embedded_boundary_mode,
        "--solver-policy",
        case.solver_policy,
        "--continuation",
        case.continuation,
        "--continuation-segment-substep-factor",
        str(case.continuation_segment_substep_factor),
        "--max-bisections",
        str(case.max_bisections),
    ]
    if case.analysis == "monotonic":
        command.extend(
            [
                "--monotonic-tip-mm",
                f"{case.monotonic_tip_mm}",
                "--monotonic-steps",
                str(case.monotonic_steps),
            ]
        )
    else:
        command.extend(
            [
                "--amplitudes-mm",
                ",".join(f"{value:.6f}" for value in case.amplitudes_mm),
                "--steps-per-segment",
                str(case.steps_per_segment),
            ]
        )
    return command


def run_case(
    exe: Path,
    root: Path,
    case: ContinuumBringupCase,
    print_progress: bool,
) -> ContinuumBringupRow:
    output_dir = root / case.key
    ensure_dir(output_dir)
    command = build_command(exe, case, output_dir)
    if print_progress:
        command.append("--print-progress")

    start = time.perf_counter()
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - start

    stdout_path = output_dir / "runner_stdout.log"
    stderr_path = output_dir / "runner_stderr.log"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    manifest_path = output_dir / "runtime_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        timing = manifest.get("timing") or {}
        observables = manifest.get("observables") or {}
        return ContinuumBringupRow(
            key=case.key,
            analysis=case.analysis,
            reinforcement_mode=case.reinforcement_mode,
            material_mode=case.material_mode,
            hex_order=case.hex_order,
            mesh=f"{case.nx}x{case.ny}x{case.nz}",
            status="completed" if bool(manifest.get("completed_successfully")) else "aborted",
            completed_successfully=bool(manifest.get("completed_successfully")),
            runtime_steps=int(manifest.get("runtime_steps", 0) or 0),
            max_abs_base_shear_mn=safe_float(observables.get("max_abs_base_shear_mn")),
            max_abs_top_rebar_face_gap_m=safe_float(
                observables.get("max_abs_top_rebar_face_gap_m")
            ),
            solve_wall_seconds=safe_float(timing.get("solve_wall_seconds")),
            total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
            output_dir=str(output_dir),
            return_code=proc.returncode,
        )

    return ContinuumBringupRow(
        key=case.key,
        analysis=case.analysis,
        reinforcement_mode=case.reinforcement_mode,
        material_mode=case.material_mode,
        hex_order=case.hex_order,
        mesh=f"{case.nx}x{case.ny}x{case.nz}",
        status="runner_failed",
        completed_successfully=False,
        runtime_steps=0,
        max_abs_base_shear_mn=math.nan,
        max_abs_top_rebar_face_gap_m=math.nan,
        solve_wall_seconds=elapsed,
        total_wall_seconds=elapsed,
        output_dir=str(output_dir),
        return_code=proc.returncode,
    )


def write_csv(path: Path, rows: list[ContinuumBringupRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_figure(fig: plt.Figure, output_path: Path, secondary_output_path: Path | None) -> None:
    ensure_dir(output_path.parent)
    fig.savefig(output_path)
    if secondary_output_path is not None:
        ensure_dir(secondary_output_path.parent)
        fig.savefig(secondary_output_path)
    plt.close(fig)


def plot_timing(rows: list[ContinuumBringupRow], output_path: Path, secondary_output_path: Path | None) -> None:
    labels = [row.key for row in rows]
    values = [row.solve_wall_seconds for row in rows]
    colors = ["#0b5fa5" if row.completed_successfully else "#b91c1c" for row in rows]

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.bar(labels, values, color=colors, width=0.65)
    ax.set_ylabel("Solve wall time [s]")
    ax.set_title("Reduced RC continuum bring-up timing")
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(values):
        if math.isfinite(value):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    save_figure(fig, output_path, secondary_output_path)


def plot_base_shear(rows: list[ContinuumBringupRow], output_path: Path, secondary_output_path: Path | None) -> None:
    labels = [row.key for row in rows]
    values = [row.max_abs_base_shear_mn * 1.0e3 for row in rows]
    colors = ["#0b5fa5" if row.reinforcement_mode != "continuum-only" else "#2563eb" for row in rows]

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.bar(labels, values, color=colors, width=0.65)
    ax.set_ylabel(r"Peak $|V_\mathrm{base}|$ [kN]")
    ax.set_title("Reduced RC continuum bring-up peak base shear")
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(values):
        if math.isfinite(value):
            ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    save_figure(fig, output_path, secondary_output_path)


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)
    ensure_dir(args.secondary_figures_dir)

    cases = [
        ContinuumBringupCase(
            key="continuum_only_elastic_monotonic_hex8_2x2x4",
            analysis="monotonic",
            reinforcement_mode="continuum-only",
            material_mode="elasticized",
            hex_order="hex8",
            nx=2,
            ny=2,
            nz=4,
            solver_policy="newton-l2-only",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=0.5,
            monotonic_steps=2,
        ),
        ContinuumBringupCase(
            key="embedded_elastic_monotonic_hex8_2x2x4",
            analysis="monotonic",
            reinforcement_mode="embedded-longitudinal-bars",
            material_mode="elasticized",
            hex_order="hex8",
            nx=2,
            ny=2,
            nz=4,
            solver_policy="newton-l2-only",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=0.5,
            monotonic_steps=2,
        ),
        ContinuumBringupCase(
            key="continuum_only_nonlinear_monotonic_hex8_2x2x8",
            analysis="monotonic",
            reinforcement_mode="continuum-only",
            material_mode="nonlinear",
            hex_order="hex8",
            nx=2,
            ny=2,
            nz=8,
            solver_policy="canonical-cascade",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=2.5,
            monotonic_steps=8,
        ),
        ContinuumBringupCase(
            key="embedded_nonlinear_monotonic_hex8_2x2x8",
            analysis="monotonic",
            reinforcement_mode="embedded-longitudinal-bars",
            material_mode="nonlinear",
            hex_order="hex8",
            nx=2,
            ny=2,
            nz=8,
            solver_policy="canonical-cascade",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=2.5,
            monotonic_steps=8,
        ),
        ContinuumBringupCase(
            key="embedded_nonlinear_cyclic_hex8_2x2x8",
            analysis="cyclic",
            reinforcement_mode="embedded-longitudinal-bars",
            material_mode="nonlinear",
            hex_order="hex8",
            nx=2,
            ny=2,
            nz=8,
            solver_policy="canonical-cascade",
            continuation="reversal-guarded",
            continuation_segment_substep_factor=2,
            amplitudes_mm=(1.25, 2.50),
            steps_per_segment=2,
        ),
    ]

    rows = [
        run_case(args.benchmark_exe, args.output_dir, case, args.print_progress)
        for case in cases
    ]

    summary = {
        "completed_case_count": sum(1 for row in rows if row.completed_successfully),
        "total_case_count": len(rows),
        "all_cases_completed": all(row.completed_successfully for row in rows),
        "max_solve_wall_seconds": max(
            (row.solve_wall_seconds for row in rows if math.isfinite(row.solve_wall_seconds)),
            default=math.nan,
        ),
        "max_abs_base_shear_mn": max(
            (row.max_abs_base_shear_mn for row in rows if math.isfinite(row.max_abs_base_shear_mn)),
            default=math.nan,
        ),
        "continuum_only_vs_embedded_elastic_stiffness_ratio": math.nan,
        "continuum_only_vs_embedded_nonlinear_stiffness_ratio": math.nan,
        "rows": [asdict(row) for row in rows],
    }

    row_by_key = {row.key: row for row in rows}
    elastic_plain = row_by_key["continuum_only_elastic_monotonic_hex8_2x2x4"]
    elastic_embedded = row_by_key["embedded_elastic_monotonic_hex8_2x2x4"]
    nonlinear_plain = row_by_key["continuum_only_nonlinear_monotonic_hex8_2x2x8"]
    nonlinear_embedded = row_by_key["embedded_nonlinear_monotonic_hex8_2x2x8"]

    if (
        elastic_plain.max_abs_base_shear_mn > 0.0
        and math.isfinite(elastic_plain.max_abs_base_shear_mn)
        and math.isfinite(elastic_embedded.max_abs_base_shear_mn)
    ):
        summary["continuum_only_vs_embedded_elastic_stiffness_ratio"] = (
            elastic_embedded.max_abs_base_shear_mn / elastic_plain.max_abs_base_shear_mn
        )
    if (
        nonlinear_plain.max_abs_base_shear_mn > 0.0
        and math.isfinite(nonlinear_plain.max_abs_base_shear_mn)
        and math.isfinite(nonlinear_embedded.max_abs_base_shear_mn)
    ):
        summary["continuum_only_vs_embedded_nonlinear_stiffness_ratio"] = (
            nonlinear_embedded.max_abs_base_shear_mn / nonlinear_plain.max_abs_base_shear_mn
        )

    write_json(args.output_dir / "continuum_bringup_summary.json", summary)
    write_csv(args.output_dir / "continuum_bringup_cases.csv", rows)

    plot_timing(
        rows,
        args.figures_dir / "reduced_rc_continuum_bringup_timing.png",
        args.secondary_figures_dir / "reduced_rc_continuum_bringup_timing.png",
    )
    plot_base_shear(
        rows,
        args.figures_dir / "reduced_rc_continuum_bringup_base_shear.png",
        args.secondary_figures_dir / "reduced_rc_continuum_bringup_base_shear.png",
    )

    print(
        "Reduced RC continuum bring-up audit completed:"
        f" {summary['completed_case_count']}/{summary['total_case_count']} cases completed."
    )
    return 0 if summary["all_cases_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
