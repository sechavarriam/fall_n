#!/usr/bin/env python3
"""
Freeze the reduced RC steel-validation chain:

    direct Menegotto material
    -> standalone TrussElement<3,3>
    -> promoted structural steel fiber
    -> promoted continuum embedded bar

The structural/continuum traces come from the canonical bridge bundle. The
material and standalone-truss paths are then replayed on those exact strain
histories so we can separate:

1. constitutive effects,
2. axial carrier effects, and
3. structural-vs-continuum differences.
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
ORANGE = "#d97706"
GREEN = "#2f855a"
RED = "#c53030"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Freeze the reduced RC steel chain audit."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bridge-bundle-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_promoted_cyclic_50mm_audit"
        / "hex20",
    )
    parser.add_argument(
        "--material-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_material_reference_benchmark.exe",
    )
    parser.add_argument(
        "--truss-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_truss_reference_benchmark.exe",
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
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def resolve_existing_csv(directory: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = directory / candidate
        if path.exists():
            return path
    names = ", ".join(candidates)
    raise FileNotFoundError(f"None of the expected CSV files exist in {directory}: {names}")


def amplitude_suffix_from_protocol(protocol: dict[str, object]) -> str:
    amplitudes = protocol.get("amplitudes_mm")
    if isinstance(amplitudes, list) and amplitudes:
        peak_mm = max(float(value) for value in amplitudes)
    else:
        peak_mm = float(protocol.get("monotonic_tip_mm", 0.0))
    label = f"{peak_mm:.3f}".rstrip("0").rstrip(".")
    return label.replace(".", "p") + "mm"


def write_protocol_csv(
    path: Path,
    trace_rows: list[dict[str, object]],
    strain_key: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "strain"])
        for index, row in enumerate(trace_rows, start=1):
            writer.writerow([index, f"{float(row[strain_key]):.16e}"])


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def run_or_reuse(
    command: list[str],
    bundle_dir: Path,
    manifest_name: str = "runtime_manifest.json",
    reuse_existing: bool = True,
) -> tuple[float, dict[str, object]]:
    ensure_dir(bundle_dir)
    manifest_path = bundle_dir / manifest_name
    if reuse_existing and manifest_path.exists():
        return math.nan, read_json(manifest_path)

    elapsed, proc = run_command(command, bundle_dir.parent)
    (bundle_dir / "runner_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (bundle_dir / "runner_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit {proc.returncode}:\n{' '.join(command)}\n\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest after successful run: {manifest_path}")
    return elapsed, read_json(manifest_path)


def align_by_step(
    actual_rows: list[dict[str, object]],
    replay_rows: list[dict[str, object]],
    actual_stress_key: str,
    actual_strain_key: str,
) -> list[dict[str, float]]:
    replay_by_step = {int(float(row["step"])): row for row in replay_rows if "step" in row}
    replay_offset = 0
    if len(replay_rows) == len(actual_rows) + 1 and replay_rows:
        first_replay = replay_rows[0]
        first_replay_strain = abs(float(first_replay.get("strain", first_replay.get("axial_strain", 0.0))))
        first_replay_stress = abs(float(first_replay.get("stress_MPa", first_replay.get("axial_stress_MPa", 0.0))))
        if first_replay_strain < 1.0e-18 and first_replay_stress < 1.0e-9:
            replay_offset = 1
    aligned: list[dict[str, float]] = []
    for index, actual in enumerate(actual_rows):
        step = index + replay_offset
        replay = replay_by_step.get(step)
        if replay is None:
            continue
        aligned.append(
            {
                "step": float(step),
                "actual_step": float(actual.get("step", index)),
                "actual_strain": float(actual[actual_strain_key]),
                "actual_stress": float(actual[actual_stress_key]),
                "replay_strain": float(replay["strain"] if "strain" in replay else replay["axial_strain"]),
                "replay_stress": float(replay["stress_MPa"] if "stress_MPa" in replay else replay["axial_stress_MPa"]),
                "replay_tangent": float(replay["tangent_MPa"] if "tangent_MPa" in replay else replay["tangent_from_element_MPa"]),
            }
        )
    return aligned


def rms(values: list[float]) -> float:
    if not values:
        return math.nan
    return math.sqrt(sum(value * value for value in values) / len(values))


def max_abs(values: list[float]) -> float:
    return max((abs(value) for value in values), default=math.nan)


def active_relative_errors(
    actual: list[float],
    replay: list[float],
    activity_floor: float = 5.0,
) -> tuple[float, float]:
    rel_errors: list[float] = []
    for actual_value, replay_value in zip(actual, replay, strict=False):
        scale = max(abs(actual_value), abs(replay_value))
        if scale < activity_floor:
            continue
        rel_errors.append(abs(actual_value - replay_value) / scale)
    if not rel_errors:
        return math.nan, math.nan
    return max(rel_errors), rms(rel_errors)


def compute_chain_metrics(aligned_rows: list[dict[str, float]]) -> dict[str, float]:
    stress_errors = [
        row["actual_stress"] - row["replay_stress"] for row in aligned_rows
    ]
    actual_stress = [row["actual_stress"] for row in aligned_rows]
    replay_stress = [row["replay_stress"] for row in aligned_rows]
    max_rel, rms_rel = active_relative_errors(actual_stress, replay_stress)
    return {
        "max_abs_stress_error_mpa": max_abs(stress_errors),
        "rms_abs_stress_error_mpa": rms(stress_errors),
        "max_rel_stress_error_active": max_rel,
        "rms_rel_stress_error_active": rms_rel,
    }


def select_structural_site_trace(
    rows: list[dict[str, object]],
    fiber_index: int,
    section_gp: int,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if int(float(row["fiber_index"])) == fiber_index
        and int(float(row["section_gp"])) == section_gp
    ]
    return sorted(selected, key=lambda row: (float(row["step"]), float(row["p"])))


def build_interpolated_replay_trace(
    lower_rows: list[dict[str, object]],
    upper_rows: list[dict[str, object]],
    alpha: float,
) -> list[dict[str, float]]:
    count = min(len(lower_rows), len(upper_rows))
    interpolated: list[dict[str, float]] = []
    for index in range(count):
        lower = lower_rows[index]
        upper = upper_rows[index]

        strain_key = "strain" if "strain" in lower else "axial_strain"
        stress_key = "stress_MPa" if "stress_MPa" in lower else "axial_stress_MPa"
        tangent_key = (
            "tangent_MPa"
            if "tangent_MPa" in lower and "tangent_from_element_MPa" not in lower
            else "tangent_from_element_MPa"
        )

        def lerp(name: str) -> float:
            lhs = float(lower[name])
            rhs = float(upper[name])
            return lhs + alpha * (rhs - lhs)

        interpolated.append(
            {
                "step": float(index),
                "strain": lerp(strain_key),
                "stress_MPa": lerp(stress_key),
                "tangent_MPa": lerp(tangent_key),
            }
        )
    return interpolated


def figure_paths(stem: str, figures_dir: Path, secondary_dir: Path) -> list[Path]:
    return [
        figures_dir / f"{stem}.png",
        secondary_dir / f"{stem}.png",
    ]


def save(fig: plt.Figure, paths: list[Path]) -> None:
    for path in paths:
        ensure_dir(path.parent)
        fig.savefig(path)
    plt.close(fig)


def make_figures(
    structural_actual: list[dict[str, object]],
    continuum_actual: list[dict[str, object]],
    structural_material_aligned: list[dict[str, float]],
    structural_truss_aligned: list[dict[str, float]],
    continuum_material_aligned: list[dict[str, float]],
    continuum_truss_aligned: list[dict[str, float]],
    structural_lower_material_aligned: list[dict[str, float]],
    structural_upper_material_aligned: list[dict[str, float]],
    figures_dir: Path,
    secondary_dir: Path,
    suffix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax0, ax1 = axes

    ax0.plot(
        [row["actual_strain"] for row in structural_material_aligned],
        [row["actual_stress"] for row in structural_material_aligned],
        color=BLUE,
        linewidth=1.8,
        label="Structural steel fiber",
    )
    ax0.plot(
        [row["replay_strain"] for row in structural_material_aligned],
        [row["replay_stress"] for row in structural_material_aligned],
        color=GREEN,
        linestyle="--",
        linewidth=1.6,
        label="Direct Menegotto replay",
    )
    ax0.plot(
        [row["replay_strain"] for row in structural_truss_aligned],
        [row["replay_stress"] for row in structural_truss_aligned],
        color=ORANGE,
        linestyle=":",
        linewidth=1.8,
        label="Standalone Truss<3> replay",
    )
    ax0.set_title("Structural steel chain")
    ax0.set_xlabel("Steel strain")
    ax0.set_ylabel("Steel stress [MPa]")
    ax0.legend(frameon=False)

    ax1.plot(
        [row["actual_strain"] for row in continuum_material_aligned],
        [row["actual_stress"] for row in continuum_material_aligned],
        color=BLUE,
        linewidth=1.8,
        label="Continuum embedded bar",
    )
    ax1.plot(
        [row["replay_strain"] for row in continuum_material_aligned],
        [row["replay_stress"] for row in continuum_material_aligned],
        color=GREEN,
        linestyle="--",
        linewidth=1.6,
        label="Direct Menegotto replay",
    )
    ax1.plot(
        [row["replay_strain"] for row in continuum_truss_aligned],
        [row["replay_stress"] for row in continuum_truss_aligned],
        color=ORANGE,
        linestyle=":",
        linewidth=1.8,
        label="Standalone Truss<3> replay",
    )
    ax1.set_title("Continuum steel chain")
    ax1.set_xlabel("Steel strain")
    ax1.set_ylabel("Steel stress [MPa]")
    ax1.legend(frameon=False)
    save(
        fig,
        figure_paths(
            f"reduced_rc_steel_chain_replay_{suffix}",
            figures_dir,
            secondary_dir,
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax0, ax1 = axes
    structural_drift_mm = [1.0e3 * float(row["drift_m"]) for row in structural_actual]
    continuum_drift_mm = [1.0e3 * float(row["drift_m"]) for row in continuum_actual]

    ax0.plot(
        structural_drift_mm,
        [float(row["stress_xx_MPa"]) for row in structural_actual],
        color=BLUE,
        linewidth=1.7,
        label="Structural fiber",
    )
    ax0.plot(
        structural_drift_mm,
        [row["replay_stress"] for row in structural_material_aligned],
        color=GREEN,
        linestyle="--",
        linewidth=1.5,
        label="Direct replay on fiber strain",
    )
    ax0.plot(
        structural_drift_mm,
        [row["replay_stress"] for row in structural_truss_aligned],
        color=ORANGE,
        linestyle=":",
        linewidth=1.7,
        label="Truss replay on fiber strain",
    )
    ax0.set_title("Structural drift-stress chain")
    ax0.set_xlabel("Tip drift [mm]")
    ax0.set_ylabel("Steel stress [MPa]")
    ax0.legend(frameon=False)

    ax1.plot(
        continuum_drift_mm,
        [float(row["stress_xx_MPa"]) for row in continuum_actual],
        color=BLUE,
        linewidth=1.7,
        label="Continuum bar",
    )
    ax1.plot(
        continuum_drift_mm,
        [row["replay_stress"] for row in continuum_material_aligned],
        color=GREEN,
        linestyle="--",
        linewidth=1.5,
        label="Direct replay on bar strain",
    )
    ax1.plot(
        continuum_drift_mm,
        [row["replay_stress"] for row in continuum_truss_aligned],
        color=ORANGE,
        linestyle=":",
        linewidth=1.7,
        label="Truss replay on bar strain",
    )
    ax1.set_title("Continuum drift-stress chain")
    ax1.set_xlabel("Tip drift [mm]")
    ax1.set_ylabel("Steel stress [MPa]")
    ax1.legend(frameon=False)
    save(
        fig,
        figure_paths(
            f"reduced_rc_steel_chain_drift_{suffix}",
            figures_dir,
            secondary_dir,
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax0, ax1 = axes
    ax0.plot(
        [row["actual_strain"] for row in structural_material_aligned],
        [row["actual_stress"] for row in structural_material_aligned],
        color=BLUE,
        linewidth=1.8,
        label="Interpolated structural trace",
    )
    ax0.plot(
        [row["actual_strain"] for row in structural_lower_material_aligned],
        [row["actual_stress"] for row in structural_lower_material_aligned],
        color=GREEN,
        linestyle="--",
        linewidth=1.5,
        label="Lower structural GP trace",
    )
    ax0.plot(
        [row["actual_strain"] for row in structural_upper_material_aligned],
        [row["actual_stress"] for row in structural_upper_material_aligned],
        color=ORANGE,
        linestyle=":",
        linewidth=1.7,
        label="Upper structural GP trace",
    )
    ax0.set_title("Structural steel site traces")
    ax0.set_xlabel("Steel strain")
    ax0.set_ylabel("Steel stress [MPa]")
    ax0.legend(frameon=False)

    structural_steps = [row["actual_step"] for row in structural_material_aligned]
    ax1.plot(
        structural_steps,
        [row["actual_stress"] - row["replay_stress"] for row in structural_material_aligned],
        color=BLUE,
        linewidth=1.8,
        label="Interpolated trace - direct replay",
    )
    ax1.plot(
        [row["actual_step"] for row in structural_lower_material_aligned],
        [row["actual_stress"] - row["replay_stress"] for row in structural_lower_material_aligned],
        color=GREEN,
        linestyle="--",
        linewidth=1.5,
        label="Lower GP - direct replay",
    )
    ax1.plot(
        [row["actual_step"] for row in structural_upper_material_aligned],
        [row["actual_stress"] - row["replay_stress"] for row in structural_upper_material_aligned],
        color=ORANGE,
        linestyle=":",
        linewidth=1.7,
        label="Upper GP - direct replay",
    )
    ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax1.set_title("Structural replay residuals by trace type")
    ax1.set_xlabel("Protocol step")
    ax1.set_ylabel("Stress residual [MPa]")
    ax1.legend(frameon=False)
    save(
        fig,
        figure_paths(
            f"reduced_rc_steel_chain_structural_sites_{suffix}",
            figures_dir,
            secondary_dir,
        ),
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    bridge_bundle_dir = args.bridge_bundle_dir.resolve()
    ensure_dir(output_dir)

    structural_trace = read_csv_rows(bridge_bundle_dir / "matched_structural_steel_trace.csv")
    continuum_trace = read_csv_rows(bridge_bundle_dir / "selected_continuum_steel_trace.csv")
    bridge_summary = read_json(bridge_bundle_dir.parent / "structural_continuum_steel_hysteresis_summary.json")
    structural_bundle_dir = Path(
        str(bridge_summary.get("structural_reference", {}).get("bundle_dir", bridge_bundle_dir.parent / "structural"))
    )
    structural_section_rows = read_csv_rows(
        structural_bundle_dir / "section_fiber_state_history.csv"
    )
    structural_station = bridge_summary.get("continuum_cases", {}).get("hex20", {}).get(
        "matched_structural_station", {}
    )
    selected_structural = bridge_summary.get("selected_structural_fibers", {}).get("hex20", {})
    lower_section_gp = int(structural_station.get("lower_section_gp", 0))
    upper_section_gp = int(structural_station.get("upper_section_gp", lower_section_gp))
    interpolation_alpha = float(structural_station.get("interpolation_alpha", 0.0))
    fiber_index = int(selected_structural.get("fiber_index", -1))
    structural_lower_trace = select_structural_site_trace(
        structural_section_rows,
        fiber_index,
        lower_section_gp,
    )
    structural_upper_trace = select_structural_site_trace(
        structural_section_rows,
        fiber_index,
        upper_section_gp,
    )
    if not structural_lower_trace or not structural_upper_trace:
        raise RuntimeError(
            "Failed to recover the structural lower/upper Gauss-point steel traces "
            "from section_fiber_state_history.csv."
        )

    structural_protocol_csv = output_dir / "structural_protocol.csv"
    continuum_protocol_csv = output_dir / "continuum_protocol.csv"
    structural_lower_protocol_csv = output_dir / "structural_lower_protocol.csv"
    structural_upper_protocol_csv = output_dir / "structural_upper_protocol.csv"
    write_protocol_csv(structural_protocol_csv, structural_trace, "strain_xx")
    write_protocol_csv(continuum_protocol_csv, continuum_trace, "axial_strain")
    write_protocol_csv(structural_lower_protocol_csv, structural_lower_trace, "strain_xx")
    write_protocol_csv(structural_upper_protocol_csv, structural_upper_trace, "strain_xx")

    structural_material_dir = output_dir / "structural_material_replay"
    structural_truss_dir = output_dir / "structural_truss_replay"
    structural_lower_material_dir = output_dir / "structural_lower_material_replay"
    structural_lower_truss_dir = output_dir / "structural_lower_truss_replay"
    structural_upper_material_dir = output_dir / "structural_upper_material_replay"
    structural_upper_truss_dir = output_dir / "structural_upper_truss_replay"
    continuum_material_dir = output_dir / "continuum_material_replay"
    continuum_truss_dir = output_dir / "continuum_truss_replay"

    structural_material_elapsed, structural_material_manifest = run_or_reuse(
        [
            str(args.material_exe.resolve()),
            "--output-dir",
            str(structural_material_dir),
            "--material",
            "steel",
            "--protocol",
            "cyclic",
            "--protocol-csv",
            str(structural_protocol_csv),
            "--steps-per-branch",
            "20",
        ],
        structural_material_dir,
        reuse_existing=args.reuse_existing,
    )
    structural_truss_elapsed, structural_truss_manifest = run_or_reuse(
        [
            str(args.truss_exe.resolve()),
            "--output-dir",
            str(structural_truss_dir),
            "--protocol",
            "cyclic_compression_return",
            "--protocol-csv",
            str(structural_protocol_csv),
            "--steps-per-branch",
            "20",
        ],
        structural_truss_dir,
        reuse_existing=args.reuse_existing,
    )
    continuum_material_elapsed, continuum_material_manifest = run_or_reuse(
        [
            str(args.material_exe.resolve()),
            "--output-dir",
            str(continuum_material_dir),
            "--material",
            "steel",
            "--protocol",
            "cyclic",
            "--protocol-csv",
            str(continuum_protocol_csv),
            "--steps-per-branch",
            "20",
        ],
        continuum_material_dir,
        reuse_existing=args.reuse_existing,
    )
    continuum_truss_elapsed, continuum_truss_manifest = run_or_reuse(
        [
            str(args.truss_exe.resolve()),
            "--output-dir",
            str(continuum_truss_dir),
            "--protocol",
            "cyclic_compression_return",
            "--protocol-csv",
            str(continuum_protocol_csv),
            "--steps-per-branch",
            "20",
        ],
        continuum_truss_dir,
        reuse_existing=args.reuse_existing,
    )
    structural_lower_material_elapsed, structural_lower_material_manifest = run_or_reuse(
        [
            str(args.material_exe.resolve()),
            "--output-dir",
            str(structural_lower_material_dir),
            "--material",
            "steel",
            "--protocol",
            "cyclic",
            "--protocol-csv",
            str(structural_lower_protocol_csv),
            "--steps-per-branch",
            "20",
        ],
        structural_lower_material_dir,
        reuse_existing=args.reuse_existing,
    )
    structural_lower_truss_elapsed, structural_lower_truss_manifest = run_or_reuse(
        [
            str(args.truss_exe.resolve()),
            "--output-dir",
            str(structural_lower_truss_dir),
            "--protocol",
            "cyclic_compression_return",
            "--protocol-csv",
            str(structural_lower_protocol_csv),
            "--steps-per-branch",
            "20",
        ],
        structural_lower_truss_dir,
        reuse_existing=args.reuse_existing,
    )
    structural_upper_material_elapsed, structural_upper_material_manifest = run_or_reuse(
        [
            str(args.material_exe.resolve()),
            "--output-dir",
            str(structural_upper_material_dir),
            "--material",
            "steel",
            "--protocol",
            "cyclic",
            "--protocol-csv",
            str(structural_upper_protocol_csv),
            "--steps-per-branch",
            "20",
        ],
        structural_upper_material_dir,
        reuse_existing=args.reuse_existing,
    )
    structural_upper_truss_elapsed, structural_upper_truss_manifest = run_or_reuse(
        [
            str(args.truss_exe.resolve()),
            "--output-dir",
            str(structural_upper_truss_dir),
            "--protocol",
            "cyclic_compression_return",
            "--protocol-csv",
            str(structural_upper_protocol_csv),
            "--steps-per-branch",
            "20",
        ],
        structural_upper_truss_dir,
        reuse_existing=args.reuse_existing,
    )

    structural_material_rows = read_csv_rows(
        resolve_existing_csv(structural_material_dir, ["material_response.csv", "uniaxial_response.csv"])
    )
    structural_truss_rows = read_csv_rows(structural_truss_dir / "truss_response.csv")
    structural_lower_material_rows = read_csv_rows(
        resolve_existing_csv(structural_lower_material_dir, ["material_response.csv", "uniaxial_response.csv"])
    )
    structural_lower_truss_rows = read_csv_rows(structural_lower_truss_dir / "truss_response.csv")
    structural_upper_material_rows = read_csv_rows(
        resolve_existing_csv(structural_upper_material_dir, ["material_response.csv", "uniaxial_response.csv"])
    )
    structural_upper_truss_rows = read_csv_rows(structural_upper_truss_dir / "truss_response.csv")
    continuum_material_rows = read_csv_rows(
        resolve_existing_csv(continuum_material_dir, ["material_response.csv", "uniaxial_response.csv"])
    )
    continuum_truss_rows = read_csv_rows(continuum_truss_dir / "truss_response.csv")

    structural_material_aligned = align_by_step(
        structural_trace,
        structural_material_rows,
        "stress_xx_MPa",
        "strain_xx",
    )
    structural_truss_aligned = align_by_step(
        structural_trace,
        structural_truss_rows,
        "stress_xx_MPa",
        "strain_xx",
    )
    structural_lower_material_aligned = align_by_step(
        structural_lower_trace,
        structural_lower_material_rows,
        "stress_xx_MPa",
        "strain_xx",
    )
    structural_lower_truss_aligned = align_by_step(
        structural_lower_trace,
        structural_lower_truss_rows,
        "stress_xx_MPa",
        "strain_xx",
    )
    structural_upper_material_aligned = align_by_step(
        structural_upper_trace,
        structural_upper_material_rows,
        "stress_xx_MPa",
        "strain_xx",
    )
    structural_upper_truss_aligned = align_by_step(
        structural_upper_trace,
        structural_upper_truss_rows,
        "stress_xx_MPa",
        "strain_xx",
    )
    continuum_material_aligned = align_by_step(
        continuum_trace,
        continuum_material_rows,
        "stress_xx_MPa",
        "axial_strain",
    )
    continuum_truss_aligned = align_by_step(
        continuum_trace,
        continuum_truss_rows,
        "stress_xx_MPa",
        "axial_strain",
    )
    structural_site_interpolated_material = build_interpolated_replay_trace(
        structural_lower_material_rows,
        structural_upper_material_rows,
        interpolation_alpha,
    )
    structural_site_interpolated_truss = build_interpolated_replay_trace(
        structural_lower_truss_rows,
        structural_upper_truss_rows,
        interpolation_alpha,
    )
    structural_site_interpolated_material_aligned = align_by_step(
        structural_trace,
        structural_site_interpolated_material,
        "stress_xx_MPa",
        "strain_xx",
    )
    structural_site_interpolated_truss_aligned = align_by_step(
        structural_trace,
        structural_site_interpolated_truss,
        "stress_xx_MPa",
        "strain_xx",
    )

    summary = {
        "bridge_bundle_dir": str(bridge_bundle_dir.parent),
        "bridge_scope": bridge_summary.get("protocol", {}),
        "structural_chain": {
            "fiber_identity": bridge_summary.get("selected_structural_fibers", {}).get("hex20", {}),
            "material_replay": compute_chain_metrics(structural_material_aligned),
            "truss_replay": compute_chain_metrics(structural_truss_aligned),
            "lower_site_material_replay": compute_chain_metrics(structural_lower_material_aligned),
            "lower_site_truss_replay": compute_chain_metrics(structural_lower_truss_aligned),
            "upper_site_material_replay": compute_chain_metrics(structural_upper_material_aligned),
            "upper_site_truss_replay": compute_chain_metrics(structural_upper_truss_aligned),
            "site_interpolated_material_replay": compute_chain_metrics(structural_site_interpolated_material_aligned),
            "site_interpolated_truss_replay": compute_chain_metrics(structural_site_interpolated_truss_aligned),
            "structural_station": structural_station,
            "material_runtime_manifest": structural_material_manifest,
            "truss_runtime_manifest": structural_truss_manifest,
            "lower_site_material_runtime_manifest": structural_lower_material_manifest,
            "lower_site_truss_runtime_manifest": structural_lower_truss_manifest,
            "upper_site_material_runtime_manifest": structural_upper_material_manifest,
            "upper_site_truss_runtime_manifest": structural_upper_truss_manifest,
            "material_process_wall_seconds": structural_material_elapsed,
            "truss_process_wall_seconds": structural_truss_elapsed,
            "lower_site_material_process_wall_seconds": structural_lower_material_elapsed,
            "lower_site_truss_process_wall_seconds": structural_lower_truss_elapsed,
            "upper_site_material_process_wall_seconds": structural_upper_material_elapsed,
            "upper_site_truss_process_wall_seconds": structural_upper_truss_elapsed,
        },
        "continuum_chain": {
            "bar_identity": bridge_summary.get("continuum_cases", {}).get("hex20", {}).get("selected_bar", {}),
            "material_replay": compute_chain_metrics(continuum_material_aligned),
            "truss_replay": compute_chain_metrics(continuum_truss_aligned),
            "material_runtime_manifest": continuum_material_manifest,
            "truss_runtime_manifest": continuum_truss_manifest,
            "material_process_wall_seconds": continuum_material_elapsed,
            "truss_process_wall_seconds": continuum_truss_elapsed,
        },
        "artifacts": {
            "structural_protocol_csv": str(structural_protocol_csv),
            "continuum_protocol_csv": str(continuum_protocol_csv),
            "structural_material_dir": str(structural_material_dir),
            "structural_truss_dir": str(structural_truss_dir),
            "structural_lower_protocol_csv": str(structural_lower_protocol_csv),
            "structural_upper_protocol_csv": str(structural_upper_protocol_csv),
            "structural_lower_material_dir": str(structural_lower_material_dir),
            "structural_lower_truss_dir": str(structural_lower_truss_dir),
            "structural_upper_material_dir": str(structural_upper_material_dir),
            "structural_upper_truss_dir": str(structural_upper_truss_dir),
            "continuum_material_dir": str(continuum_material_dir),
            "continuum_truss_dir": str(continuum_truss_dir),
        },
    }

    suffix = amplitude_suffix_from_protocol(
        bridge_summary.get("protocol", {}) if isinstance(bridge_summary, dict) else {}
    )
    replay_paths = figure_paths(
        f"reduced_rc_steel_chain_replay_{suffix}",
        args.figures_dir.resolve(),
        args.secondary_figures_dir.resolve(),
    )
    drift_paths = figure_paths(
        f"reduced_rc_steel_chain_drift_{suffix}",
        args.figures_dir.resolve(),
        args.secondary_figures_dir.resolve(),
    )
    site_paths = figure_paths(
        f"reduced_rc_steel_chain_structural_sites_{suffix}",
        args.figures_dir.resolve(),
        args.secondary_figures_dir.resolve(),
    )
    summary["artifacts"].update(
        {
            "steel_chain_replay_png": [str(path) for path in replay_paths],
            "steel_chain_drift_png": [str(path) for path in drift_paths],
            "steel_chain_structural_sites_png": [str(path) for path in site_paths],
        }
    )

    write_json(output_dir / "steel_chain_summary.json", summary)
    make_figures(
        structural_trace,
        continuum_trace,
        structural_material_aligned,
        structural_truss_aligned,
        continuum_material_aligned,
        continuum_truss_aligned,
        structural_lower_material_aligned,
        structural_upper_material_aligned,
        args.figures_dir.resolve(),
        args.secondary_figures_dir.resolve(),
        suffix,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
