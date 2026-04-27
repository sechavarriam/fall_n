#!/usr/bin/env python3
"""
Freeze the first reduced-RC continuum hex-order extension matrix.

This audit deliberately separates two questions:

1. matched-topology coherence across Hex8/Hex20/Hex27 on short elastic and
   short nonlinear monotonic slices; and
2. cost-controlled cyclic viability across the same hex families with embedded
   bars, where Hex27 already requires a lighter mesh to stay inside a practical
   validation budget.

The goal is not to claim formal numerical convergence yet. The goal is to make
the transition to the continuum stage scientifically honest: which continuum
families are already stable and kinematically coherent, which ones remain too
expensive for the current baseline, and which bundles are ready for the first
structural-versus-continuum comparisons.
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
from matplotlib.axes import Axes


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
        description="Run the first reduced RC continuum hex-order extension matrix."
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
    parser.add_argument("--reuse-existing", action="store_true", default=True)
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
class ContinuumOrderMatrixCase:
    key: str
    group: str
    analysis: str
    hex_order: str
    nx: int
    ny: int
    nz: int
    material_mode: str
    solver_policy: str
    continuation: str
    continuation_segment_substep_factor: int
    timeout_seconds: int
    axial_compression_mn: float = 0.02
    axial_preload_steps: int = 4
    monotonic_tip_mm: float = 0.0
    monotonic_steps: int = 0
    amplitudes_mm: tuple[float, ...] = ()
    steps_per_segment: int = 0
    max_bisections: int = 6
    reinforcement_mode: str = "embedded-longitudinal-bars"
    embedded_boundary_mode: str = "dirichlet-rebar-endcap"


@dataclass(frozen=True)
class ContinuumOrderMatrixRow:
    key: str
    group: str
    analysis: str
    hex_order: str
    mesh: str
    material_mode: str
    status: str
    completed_successfully: bool
    timed_out: bool
    runtime_steps: int
    max_abs_base_shear_mn: float
    max_abs_top_rebar_face_gap_m: float
    solve_wall_seconds: float
    total_wall_seconds: float
    process_wall_seconds: float
    timeout_seconds: int
    output_dir: str
    return_code: int


STATUS_COLOR = {
    "completed": "#0b5fa5",
    "timeout_budget_exceeded": "#c2410c",
    "runner_failed": "#b91c1c",
    "aborted": "#d97706",
}

STATUS_LEVEL = {
    "runner_failed": 0.0,
    "aborted": 0.35,
    "timeout_budget_exceeded": 0.65,
    "completed": 1.0,
}


def build_command(
    exe: Path, case: ContinuumOrderMatrixCase, output_dir: Path
) -> list[str]:
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
        "--axial-compression-mn",
        f"{case.axial_compression_mn}",
        "--axial-preload-steps",
        str(case.axial_preload_steps),
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
    case: ContinuumOrderMatrixCase,
    reuse_existing: bool,
    print_progress: bool,
) -> ContinuumOrderMatrixRow:
    output_dir = root / case.key
    ensure_dir(output_dir)
    manifest_path = output_dir / "runtime_manifest.json"
    if reuse_existing and manifest_path.exists():
        manifest = read_json(manifest_path)
        timing = manifest.get("timing") or {}
        observables = manifest.get("observables") or {}
        return ContinuumOrderMatrixRow(
            key=case.key,
            group=case.group,
            analysis=case.analysis,
            hex_order=case.hex_order,
            mesh=f"{case.nx}x{case.ny}x{case.nz}",
            material_mode=case.material_mode,
            status="completed" if bool(manifest.get("completed_successfully")) else "aborted",
            completed_successfully=bool(manifest.get("completed_successfully")),
            timed_out=False,
            runtime_steps=int(manifest.get("runtime_steps", 0) or 0),
            max_abs_base_shear_mn=safe_float(observables.get("max_abs_base_shear_mn")),
            max_abs_top_rebar_face_gap_m=safe_float(
                observables.get("max_abs_top_rebar_face_gap_m")
            ),
            solve_wall_seconds=safe_float(timing.get("solve_wall_seconds")),
            total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
            process_wall_seconds=max(
                safe_float(timing.get("solve_wall_seconds")),
                safe_float(timing.get("total_wall_seconds")),
            ),
            timeout_seconds=case.timeout_seconds,
            output_dir=str(output_dir),
            return_code=0,
        )

    command = build_command(exe, case, output_dir)
    if print_progress:
        command.append("--print-progress")

    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=case.timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - start

    (output_dir / "runner_stdout.log").write_text(stdout or "", encoding="utf-8")
    (output_dir / "runner_stderr.log").write_text(stderr or "", encoding="utf-8")

    manifest_path = output_dir / "runtime_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        timing = manifest.get("timing") or {}
        observables = manifest.get("observables") or {}
        status = (
            "timeout_budget_exceeded"
            if timed_out
            else ("completed" if bool(manifest.get("completed_successfully")) else "aborted")
        )
        return ContinuumOrderMatrixRow(
            key=case.key,
            group=case.group,
            analysis=case.analysis,
            hex_order=case.hex_order,
            mesh=f"{case.nx}x{case.ny}x{case.nz}",
            material_mode=case.material_mode,
            status=status,
            completed_successfully=bool(manifest.get("completed_successfully")) and not timed_out,
            timed_out=timed_out,
            runtime_steps=int(manifest.get("runtime_steps", 0) or 0),
            max_abs_base_shear_mn=safe_float(observables.get("max_abs_base_shear_mn")),
            max_abs_top_rebar_face_gap_m=safe_float(
                observables.get("max_abs_top_rebar_face_gap_m")
            ),
            solve_wall_seconds=safe_float(timing.get("solve_wall_seconds")),
            total_wall_seconds=safe_float(timing.get("total_wall_seconds")),
            process_wall_seconds=elapsed,
            timeout_seconds=case.timeout_seconds,
            output_dir=str(output_dir),
            return_code=proc.returncode,
        )

    return ContinuumOrderMatrixRow(
        key=case.key,
        group=case.group,
        analysis=case.analysis,
        hex_order=case.hex_order,
        mesh=f"{case.nx}x{case.ny}x{case.nz}",
        material_mode=case.material_mode,
        status="timeout_budget_exceeded" if timed_out else "runner_failed",
        completed_successfully=False,
        timed_out=timed_out,
        runtime_steps=0,
        max_abs_base_shear_mn=math.nan,
        max_abs_top_rebar_face_gap_m=math.nan,
        solve_wall_seconds=math.nan,
        total_wall_seconds=math.nan,
        process_wall_seconds=elapsed,
        timeout_seconds=case.timeout_seconds,
        output_dir=str(output_dir),
        return_code=proc.returncode if proc.returncode is not None else -1,
    )


def write_csv(path: Path, rows: list[ContinuumOrderMatrixRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_figure(
    fig: plt.Figure, output_path: Path, secondary_output_path: Path | None
) -> None:
    ensure_dir(output_path.parent)
    fig.savefig(output_path)
    if secondary_output_path is not None:
        ensure_dir(secondary_output_path.parent)
        fig.savefig(secondary_output_path)
    plt.close(fig)


def grouped_rows(rows: list[ContinuumOrderMatrixRow]) -> dict[str, list[ContinuumOrderMatrixRow]]:
    groups: dict[str, list[ContinuumOrderMatrixRow]] = {}
    for row in rows:
        groups.setdefault(row.group, []).append(row)
    return groups


def plot_timing(
    rows: list[ContinuumOrderMatrixRow],
    output_path: Path,
    secondary_output_path: Path | None,
) -> None:
    groups = grouped_rows(rows)
    fig, axes = plt.subplots(len(groups), 1, figsize=(8.8, 6.8), sharex=False)
    if isinstance(axes, Axes):
        axes = [axes]
    else:
        axes = list(axes.flat)

    for ax, (group, group_rows) in zip(axes, groups.items()):
        labels = [f"{row.hex_order}\n{row.mesh}" for row in group_rows]
        values = [
            row.solve_wall_seconds
            if math.isfinite(row.solve_wall_seconds)
            else row.process_wall_seconds
            for row in group_rows
        ]
        colors = [STATUS_COLOR[row.status] for row in group_rows]
        ax.bar(labels, values, color=colors, width=0.65)
        ax.set_ylabel("Solve wall [s]")
        ax.set_title(group.replace("_", " "))
        for idx, value in enumerate(values):
            if math.isfinite(value):
                ax.text(idx, value, f"{value:.1f}", ha="center", va="bottom")

    fig.suptitle("Reduced RC continuum hex-order matrix timing", y=0.995)
    fig.tight_layout()
    save_figure(fig, output_path, secondary_output_path)


def plot_base_shear(
    rows: list[ContinuumOrderMatrixRow],
    output_path: Path,
    secondary_output_path: Path | None,
) -> None:
    groups = grouped_rows(rows)
    fig, axes = plt.subplots(len(groups), 1, figsize=(8.8, 6.8), sharex=False)
    if isinstance(axes, Axes):
        axes = [axes]
    else:
        axes = list(axes.flat)

    for ax, (group, group_rows) in zip(axes, groups.items()):
        labels = [f"{row.hex_order}\n{row.mesh}" for row in group_rows]
        values = [
            row.max_abs_base_shear_mn * 1.0e3 if math.isfinite(row.max_abs_base_shear_mn) else 0.0
            for row in group_rows
        ]
        colors = [STATUS_COLOR[row.status] for row in group_rows]
        ax.bar(labels, values, color=colors, width=0.65)
        ax.set_ylabel(r"Peak $|V_\mathrm{base}|$ [kN]")
        ax.set_title(group.replace("_", " "))
        for idx, value in enumerate(values):
            if value > 0.0:
                ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom")

    fig.suptitle("Reduced RC continuum hex-order matrix peak base shear", y=0.995)
    fig.tight_layout()
    save_figure(fig, output_path, secondary_output_path)


def plot_status(
    rows: list[ContinuumOrderMatrixRow],
    output_path: Path,
    secondary_output_path: Path | None,
) -> None:
    groups = list(grouped_rows(rows).keys())
    hex_orders = ["hex8", "hex20", "hex27"]
    values = []
    for group in groups:
        group_rows = [row for row in rows if row.group == group]
        group_map = {row.hex_order: row for row in group_rows}
        values.append(
            [STATUS_LEVEL[group_map[hex_order].status] for hex_order in hex_orders]
        )

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(hex_orders)), [order.upper() for order in hex_orders])
    ax.set_yticks(range(len(groups)), [group.replace("_", " ") for group in groups])
    ax.set_title("Reduced RC continuum hex-order matrix status")
    for row_idx, group in enumerate(groups):
        for col_idx, hex_order in enumerate(hex_orders):
            row = next(
                candidate
                for candidate in rows
                if candidate.group == group and candidate.hex_order == hex_order
            )
            label = "OK" if row.status == "completed" else (
                "TIME" if row.status == "timeout_budget_exceeded" else "FAIL"
            )
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.tight_layout()
    save_figure(fig, output_path, secondary_output_path)


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)
    ensure_dir(args.secondary_figures_dir)

    cases = [
        ContinuumOrderMatrixCase(
            key="embedded_elastic_monotonic_hex8_2x2x4",
            group="elastic_monotonic_matched_topology",
            analysis="monotonic",
            hex_order="hex8",
            nx=2,
            ny=2,
            nz=4,
            material_mode="elasticized",
            solver_policy="newton-l2-only",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=0.5,
            monotonic_steps=2,
            timeout_seconds=180,
        ),
        ContinuumOrderMatrixCase(
            key="embedded_elastic_monotonic_hex20_2x2x4",
            group="elastic_monotonic_matched_topology",
            analysis="monotonic",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=4,
            material_mode="elasticized",
            solver_policy="newton-l2-only",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=0.5,
            monotonic_steps=2,
            timeout_seconds=240,
        ),
        ContinuumOrderMatrixCase(
            key="embedded_elastic_monotonic_hex27_2x2x4",
            group="elastic_monotonic_matched_topology",
            analysis="monotonic",
            hex_order="hex27",
            nx=2,
            ny=2,
            nz=4,
            material_mode="elasticized",
            solver_policy="newton-l2-only",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=0.5,
            monotonic_steps=2,
            timeout_seconds=300,
        ),
        ContinuumOrderMatrixCase(
            key="embedded_nonlinear_monotonic_hex8_2x2x4",
            group="nonlinear_monotonic_matched_topology",
            analysis="monotonic",
            hex_order="hex8",
            nx=2,
            ny=2,
            nz=4,
            material_mode="nonlinear",
            solver_policy="canonical-cascade",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=2.5,
            monotonic_steps=4,
            timeout_seconds=300,
        ),
        ContinuumOrderMatrixCase(
            key="embedded_nonlinear_monotonic_hex20_2x2x4",
            group="nonlinear_monotonic_matched_topology",
            analysis="monotonic",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=4,
            material_mode="nonlinear",
            solver_policy="canonical-cascade",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=2.5,
            monotonic_steps=4,
            timeout_seconds=300,
        ),
        ContinuumOrderMatrixCase(
            key="embedded_nonlinear_monotonic_hex27_2x2x4",
            group="nonlinear_monotonic_matched_topology",
            analysis="monotonic",
            hex_order="hex27",
            nx=2,
            ny=2,
            nz=4,
            material_mode="nonlinear",
            solver_policy="canonical-cascade",
            continuation="monolithic",
            continuation_segment_substep_factor=1,
            monotonic_tip_mm=2.5,
            monotonic_steps=4,
            timeout_seconds=480,
        ),
        ContinuumOrderMatrixCase(
            key="embedded_nonlinear_cyclic_hex8_2x2x4",
            group="nonlinear_cyclic_cost_controlled",
            analysis="cyclic",
            hex_order="hex8",
            nx=2,
            ny=2,
            nz=4,
            material_mode="nonlinear",
            solver_policy="canonical-cascade",
            continuation="reversal-guarded",
            continuation_segment_substep_factor=2,
            amplitudes_mm=(1.25, 2.50),
            steps_per_segment=2,
            timeout_seconds=300,
        ),
        ContinuumOrderMatrixCase(
            key="embedded_nonlinear_cyclic_hex20_2x2x4",
            group="nonlinear_cyclic_cost_controlled",
            analysis="cyclic",
            hex_order="hex20",
            nx=2,
            ny=2,
            nz=4,
            material_mode="nonlinear",
            solver_policy="canonical-cascade",
            continuation="reversal-guarded",
            continuation_segment_substep_factor=2,
            amplitudes_mm=(1.25, 2.50),
            steps_per_segment=2,
            timeout_seconds=240,
        ),
        ContinuumOrderMatrixCase(
            key="embedded_nonlinear_cyclic_hex27_1x1x4",
            group="nonlinear_cyclic_cost_controlled",
            analysis="cyclic",
            hex_order="hex27",
            nx=1,
            ny=1,
            nz=4,
            material_mode="nonlinear",
            solver_policy="canonical-cascade",
            continuation="reversal-guarded",
            continuation_segment_substep_factor=2,
            amplitudes_mm=(1.25, 2.50),
            steps_per_segment=2,
            timeout_seconds=240,
        ),
    ]

    rows = [
        run_case(
            args.benchmark_exe,
            args.output_dir,
            case,
            args.reuse_existing,
            args.print_progress,
        )
        for case in cases
    ]

    elastic_rows = [row for row in rows if row.group == "elastic_monotonic_matched_topology"]
    nonlinear_matched_rows = [
        row for row in rows if row.group == "nonlinear_monotonic_matched_topology"
    ]
    nonlinear_cyclic_rows = [
        row for row in rows if row.group == "nonlinear_cyclic_cost_controlled"
    ]

    def completed_count(subrows: list[ContinuumOrderMatrixRow]) -> int:
        return sum(1 for row in subrows if row.completed_successfully)

    def spread_ratio(subrows: list[ContinuumOrderMatrixRow]) -> float:
        values = [
            row.max_abs_base_shear_mn
            for row in subrows
            if row.completed_successfully and math.isfinite(row.max_abs_base_shear_mn)
        ]
        if len(values) < 2:
            return math.nan
        return max(values) / min(values) - 1.0

    summary = {
        "completed_case_count": completed_count(rows),
        "total_case_count": len(rows),
        "group_completed_counts": {
            "elastic_monotonic_matched_topology": completed_count(elastic_rows),
            "nonlinear_monotonic_matched_topology": completed_count(nonlinear_matched_rows),
            "nonlinear_cyclic_cost_controlled": completed_count(nonlinear_cyclic_rows),
        },
        "elastic_monotonic_matched_topology_base_shear_spread_ratio": spread_ratio(
            elastic_rows
        ),
        "nonlinear_monotonic_matched_topology_base_shear_spread_ratio": spread_ratio(
            nonlinear_matched_rows
        ),
        "nonlinear_cyclic_cost_controlled_base_shear_spread_ratio": spread_ratio(
            nonlinear_cyclic_rows
        ),
        "max_solve_wall_seconds": max(
            (row.solve_wall_seconds for row in rows if math.isfinite(row.solve_wall_seconds)),
            default=math.nan,
        ),
        "max_process_wall_seconds": max(
            (row.process_wall_seconds for row in rows if math.isfinite(row.process_wall_seconds)),
            default=math.nan,
        ),
        "hex27_nonlinear_monotonic_matched_status": next(
            row.status
            for row in rows
            if row.key == "embedded_nonlinear_monotonic_hex27_2x2x4"
        ),
        "rows": [asdict(row) for row in rows],
    }

    write_json(args.output_dir / "continuum_order_matrix_summary.json", summary)
    write_csv(args.output_dir / "continuum_order_matrix_cases.csv", rows)

    plot_timing(
        rows,
        args.figures_dir / "reduced_rc_continuum_order_matrix_timing.png",
        args.secondary_figures_dir / "reduced_rc_continuum_order_matrix_timing.png",
    )
    plot_base_shear(
        rows,
        args.figures_dir / "reduced_rc_continuum_order_matrix_base_shear.png",
        args.secondary_figures_dir / "reduced_rc_continuum_order_matrix_base_shear.png",
    )
    plot_status(
        rows,
        args.figures_dir / "reduced_rc_continuum_order_matrix_status.png",
        args.secondary_figures_dir / "reduced_rc_continuum_order_matrix_status.png",
    )

    print(
        "Reduced RC continuum order matrix audit completed:"
        f" {summary['completed_case_count']}/{summary['total_case_count']} cases completed."
    )
    return 0 if completed_count(rows) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
