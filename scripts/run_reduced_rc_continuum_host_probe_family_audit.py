#!/usr/bin/env python3
"""
Audit the same host-concrete neighborhood with and without embedded steel.

The promoted continuum branch now closes to larger amplitudes and the embedded
host↔bar kinematic tie already closes essentially to machine precision. The
remaining question is more local:

  * what concrete history does the host carry at the physical steel location
    when there is no steel at all?
  * how does that same host neighborhood change when the steel is present
    inside the section?
  * how different is the cheaper boundary-bar control branch at that same
    physical coordinate?

This script reruns the three canonical cover/core branches with an explicit
host probe placed at the promoted interior steel coordinate recovered from the
structural↔continuum bridge summary. It therefore compares the *same host
position* across branches, rather than relying on the selected embedded bar
trace alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
PURPLE = "#7c3aed"


@dataclass(frozen=True)
class HostProbeCase:
    key: str
    label: str
    reinforcement_mode: str
    rebar_layout: str
    timeout_seconds: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the canonical continuum family branches with an explicit "
            "host probe at the promoted steel location."
        )
    )
    parser.add_argument(
        "--promoted-bridge-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_promoted_cyclic_75mm_audit",
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
    parser.add_argument("--analysis", choices=("cyclic", "monotonic"), default="cyclic")
    parser.add_argument("--amplitudes-mm", default=None)
    parser.add_argument("--steps-per-segment", type=int, default=3)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--plain-timeout-seconds", type=int, default=4200)
    parser.add_argument("--interior-timeout-seconds", type=int, default=7200)
    parser.add_argument("--boundary-timeout-seconds", type=int, default=6000)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def amplitude_suffix_from_summary(summary: dict[str, Any]) -> str:
    protocol = summary.get("protocol", {})
    amplitudes = protocol.get("amplitudes_mm")
    if isinstance(amplitudes, list) and amplitudes:
        peak_mm = max(float(value) for value in amplitudes)
    else:
        peak_mm = float(protocol.get("monotonic_tip_mm", 0.0))
    label = f"{peak_mm:.3f}".rstrip("0").rstrip(".")
    return label.replace(".", "p") + "mm"


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def max_or_nan(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return max(clean) if clean else math.nan


def rms(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return math.nan
    return math.sqrt(sum(v * v for v in clean) / len(clean))


def clean_optional(value: float) -> float | None:
    return value if math.isfinite(value) else None


def run_command(command: list[str], timeout_seconds: int, print_progress: bool) -> tuple[bool, float, str]:
    start = time.perf_counter()
    if print_progress:
        print(" ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
        success = process.returncode == 0
        return success, time.perf_counter() - start, stdout
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            stdout, _ = process.communicate()
        return False, time.perf_counter() - start, stdout


def resolve_amplitudes(summary: dict[str, Any], raw_override: str | None) -> str:
    if raw_override:
        return raw_override
    amplitudes = summary.get("protocol", {}).get("amplitudes_mm", [])
    return ",".join(str(float(value)).rstrip("0").rstrip(".") for value in amplitudes)


def case_command(
    args: argparse.Namespace,
    case: HostProbeCase,
    case_dir: Path,
    amplitudes_mm: str,
    probe_label: str,
    probe_x: float,
    probe_y: float,
    probe_z: float,
) -> list[str]:
    return [
        str(args.benchmark_exe),
        "--output-dir",
        str(case_dir),
        "--analysis",
        args.analysis,
        "--material-mode",
        "nonlinear",
        "--concrete-profile",
        "production-stabilized",
        "--concrete-tangent-mode",
        "fracture-secant",
        "--concrete-characteristic-length-mode",
        "mean-longitudinal-host-edge-mm",
        "--reinforcement-mode",
        case.reinforcement_mode,
        "--rebar-layout",
        case.rebar_layout,
        "--rebar-interpolation",
        "automatic",
        "--host-concrete-zoning-mode",
        "cover-core-split",
        "--transverse-mesh-mode",
        "cover-aligned",
        "--hex-order",
        "hex20",
        "--nx",
        "4",
        "--ny",
        "4",
        "--nz",
        "2",
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--solver-policy",
        "newton-l2-only",
        "--predictor-policy",
        "hybrid-secant-linearized",
        "--continuation",
        "reversal-guarded",
        "--continuation-segment-substep-factor",
        "2",
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
        "--amplitudes-mm",
        amplitudes_mm,
        "--host-probe",
        f"{probe_label}:{probe_x}:{probe_y}:{probe_z}",
    ] + (["--print-progress"] if args.print_progress else [])


def load_host_probe_rows(case_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(case_dir / "host_probe_history.csv")
    return [row for row in rows if str(row["probe_label"]) == "steel_path_probe"]


def rows_by_runtime_step(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["runtime_step"]): row for row in rows}


def relative_metrics(
    lhs_rows: list[dict[str, Any]],
    rhs_rows: list[dict[str, Any]],
    field: str,
    *,
    active_floor: float | None = None,
) -> dict[str, float]:
    rhs_by_step = rows_by_runtime_step(rhs_rows)
    rel_errors: list[float] = []
    active_errors: list[float] = []
    for row in lhs_rows:
        rhs = rhs_by_step.get(int(row["runtime_step"]))
        if rhs is None:
            continue
        lhs_value = safe_float(row[field])
        rhs_value = safe_float(rhs[field])
        if not (math.isfinite(lhs_value) and math.isfinite(rhs_value)):
            continue
        rel = abs(lhs_value - rhs_value) / max(abs(rhs_value), 1.0e-12)
        rel_errors.append(rel)
        if active_floor is not None and abs(rhs_value) >= active_floor:
            active_errors.append(rel)
    return {
        "max_rel_error": max_or_nan(rel_errors),
        "rms_rel_error": rms(rel_errors),
        "max_rel_error_active": max_or_nan(active_errors),
        "rms_rel_error_active": rms(active_errors),
    }


def first_active_runtime_step(rows: list[dict[str, Any]], field: str, threshold: float) -> int:
    for row in rows:
        if abs(safe_float(row[field])) >= threshold:
            return int(row["runtime_step"])
    return -1


def save_figure(fig: plt.Figure, primary: Path, secondary: Path | None) -> None:
    ensure_dir(primary.parent)
    fig.savefig(primary)
    if secondary is not None:
        ensure_dir(secondary.parent)
        fig.savefig(secondary)
    plt.close(fig)


def plot_overlay(
    rows_by_case: dict[str, list[dict[str, Any]]],
    field: str,
    ylabel: str,
    title: str,
    stem: str,
    suffix: str,
    out_dirs: list[Path],
) -> list[str]:
    colors = {
        "plain": ORANGE,
        "embedded_interior": GREEN,
        "embedded_boundary": PURPLE,
    }
    labels = {
        "plain": "Plain",
        "embedded_interior": "Embedded interior",
        "embedded_boundary": "Boundary bars",
    }
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for key, rows in rows_by_case.items():
        ax.plot(
            [1.0e3 * safe_float(row["drift_m"]) for row in rows],
            [safe_float(row[field]) for row in rows],
            color=colors[key],
            linewidth=1.5,
            label=labels[key],
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} ({suffix})")
    ax.legend(frameon=False)
    fig.tight_layout()
    primary = out_dirs[0] / f"{stem}_{suffix}.png"
    secondary = out_dirs[1] / primary.name if len(out_dirs) > 1 else None
    save_figure(fig, primary, secondary)
    return [str(primary)] + ([str(secondary)] if secondary else [])


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    promoted_summary = read_json(
        args.promoted_bridge_dir / "structural_continuum_steel_hysteresis_summary.json"
    )
    selected_bar = promoted_summary.get("continuum_cases", {}).get("hex20", {}).get("selected_bar", {})
    probe_x = float(selected_bar["bar_y"])
    probe_y = float(selected_bar["bar_z"])
    probe_z = float(selected_bar["position_z_m"])
    probe_label = "steel_path_probe"
    suffix = amplitude_suffix_from_summary(promoted_summary)
    amplitudes_mm = resolve_amplitudes(promoted_summary, args.amplitudes_mm)

    cases = [
        HostProbeCase(
            key="plain",
            label="Plain cover/core",
            reinforcement_mode="continuum-only",
            rebar_layout="structural-matched-eight-bar",
            timeout_seconds=args.plain_timeout_seconds,
        ),
        HostProbeCase(
            key="embedded_interior",
            label="Embedded interior",
            reinforcement_mode="embedded-longitudinal-bars",
            rebar_layout="structural-matched-eight-bar",
            timeout_seconds=args.interior_timeout_seconds,
        ),
        HostProbeCase(
            key="embedded_boundary",
            label="Boundary bars",
            reinforcement_mode="embedded-longitudinal-bars",
            rebar_layout="boundary-matched-eight-bar",
            timeout_seconds=args.boundary_timeout_seconds,
        ),
    ]

    summary_rows: list[dict[str, Any]] = []
    host_rows_by_case: dict[str, list[dict[str, Any]]] = {}
    runtime_manifests: dict[str, dict[str, Any]] = {}

    for case in cases:
        case_dir = args.output_dir / case.key
        manifest_path = case_dir / "runtime_manifest.json"
        host_probe_path = case_dir / "host_probe_history.csv"
        should_reuse = (
            args.reuse_existing
            and manifest_path.exists()
            and host_probe_path.exists()
            and read_json(manifest_path).get("host_probe_count", 0) > 0
        )
        if not should_reuse:
            ensure_dir(case_dir)
            command = case_command(
                args,
                case,
                case_dir,
                amplitudes_mm,
                probe_label,
                probe_x,
                probe_y,
                probe_z,
            )
            ok, elapsed, stdout = run_command(command, case.timeout_seconds, args.print_progress)
            (case_dir / "host_probe_stdout.log").write_text(stdout, encoding="utf-8")
            if not ok:
                raise RuntimeError(
                    f"Host-probe family audit failed for {case.key} after {elapsed:.1f}s. "
                    f"See {case_dir / 'host_probe_stdout.log'}."
                )

        manifest = read_json(manifest_path)
        rows = load_host_probe_rows(case_dir)
        runtime_manifests[case.key] = manifest
        host_rows_by_case[case.key] = rows

    interior_rows = host_rows_by_case["embedded_interior"]
    boundary_rows = host_rows_by_case["embedded_boundary"]
    plain_rows = host_rows_by_case["plain"]
    interior_peak_stress = max(abs(safe_float(row["nearest_host_axial_stress_MPa"])) for row in interior_rows)
    interior_peak_strain = max(abs(safe_float(row["nearest_host_axial_strain"])) for row in interior_rows)
    stress_active_floor = 0.05 * interior_peak_stress
    strain_active_floor = 0.05 * interior_peak_strain

    # Validate the new generic probe against the same-run embedded-bar host read,
    # not against an older promoted bridge with a different continuum slice.
    selected_bar_trace = [
        row
        for row in read_csv_rows(args.output_dir / "embedded_interior" / "rebar_history.csv")
        if int(safe_float(row["bar_index"], -1)) == int(selected_bar["bar_index"])
        and int(safe_float(row["bar_element_layer"], -1))
            == int(selected_bar["bar_element_layer"])
        and int(safe_float(row["gp_index"], -1)) == int(selected_bar["gp_index"])
    ]
    selected_bar_by_step = rows_by_runtime_step(selected_bar_trace)
    probe_vs_bar_host_stress_gap: list[float] = []
    probe_vs_bar_host_strain_gap: list[float] = []
    for row in interior_rows:
        ref = selected_bar_by_step.get(int(row["runtime_step"]))
        if ref is None:
            continue
        probe_vs_bar_host_stress_gap.append(
            safe_float(row["nearest_host_axial_stress_MPa"])
            - safe_float(ref["nearest_host_axial_stress_MPa"])
        )
        probe_vs_bar_host_strain_gap.append(
            safe_float(row["nearest_host_axial_strain"])
            - safe_float(ref["nearest_host_axial_strain"])
        )

    for case in cases:
        rows = host_rows_by_case[case.key]
        manifest = runtime_manifests[case.key]
        comparison_to_interior_stress = (
            relative_metrics(rows, interior_rows, "nearest_host_axial_stress_MPa", active_floor=stress_active_floor)
            if case.key != "embedded_interior"
            else {}
        )
        comparison_to_interior_strain = (
            relative_metrics(rows, interior_rows, "nearest_host_axial_strain", active_floor=strain_active_floor)
            if case.key != "embedded_interior"
            else {}
        )
        summary_rows.append(
            {
                "key": case.key,
                "label": case.label,
                "completed_successfully": bool(manifest.get("completed_successfully", False)),
                "process_wall_seconds": safe_float(manifest.get("timing", {}).get("total_wall_seconds")),
                "solve_wall_seconds": safe_float(manifest.get("timing", {}).get("solve_wall_seconds")),
                "max_abs_base_shear_mn": safe_float(manifest.get("observables", {}).get("max_abs_base_shear_mn")),
                "max_host_probe_gp_distance_m": max_or_nan(
                    [safe_float(row["nearest_host_gp_distance_m"]) for row in rows]
                ),
                "rms_host_probe_gp_distance_m": rms(
                    [safe_float(row["nearest_host_gp_distance_m"]) for row in rows]
                ),
                "first_host_probe_stress_runtime_step": first_active_runtime_step(
                    rows, "nearest_host_axial_stress_MPa", stress_active_floor
                ),
                "first_host_probe_strain_runtime_step": first_active_runtime_step(
                    rows, "nearest_host_axial_strain", strain_active_floor
                ),
                "max_abs_host_probe_axial_stress_mpa": max_or_nan(
                    [abs(safe_float(row["nearest_host_axial_stress_MPa"])) for row in rows]
                ),
                "max_abs_host_probe_axial_strain": max_or_nan(
                    [abs(safe_float(row["nearest_host_axial_strain"])) for row in rows]
                ),
                "max_host_probe_crack_opening_m": max_or_nan(
                    [safe_float(row["nearest_host_max_crack_opening"]) for row in rows]
                ),
                "peak_host_probe_crack_count": max(
                    int(safe_float(row["nearest_host_num_cracks"], 0.0)) for row in rows
                ),
                "max_host_probe_damage": max_or_nan(
                    [
                        safe_float(row["nearest_host_damage"])
                        for row in rows
                        if int(safe_float(row["nearest_host_damage_available"], 0.0)) > 0
                    ]
                ),
                "max_rel_stress_vs_interior": clean_optional(
                    comparison_to_interior_stress.get("max_rel_error", math.nan)
                ),
                "rms_rel_stress_vs_interior": clean_optional(
                    comparison_to_interior_stress.get("rms_rel_error", math.nan)
                ),
                "max_rel_stress_vs_interior_active": clean_optional(
                    comparison_to_interior_stress.get("max_rel_error_active", math.nan)
                ),
                "rms_rel_stress_vs_interior_active": clean_optional(
                    comparison_to_interior_stress.get("rms_rel_error_active", math.nan)
                ),
                "max_rel_strain_vs_interior": clean_optional(
                    comparison_to_interior_strain.get("max_rel_error", math.nan)
                ),
                "rms_rel_strain_vs_interior": clean_optional(
                    comparison_to_interior_strain.get("rms_rel_error", math.nan)
                ),
                "max_rel_strain_vs_interior_active": clean_optional(
                    comparison_to_interior_strain.get("max_rel_error_active", math.nan)
                ),
                "rms_rel_strain_vs_interior_active": clean_optional(
                    comparison_to_interior_strain.get("rms_rel_error_active", math.nan)
                ),
                "host_probe_csv": str(args.output_dir / case.key / "host_probe_history.csv"),
            }
        )

    figure_dirs = [args.figures_dir, args.secondary_figures_dir]
    figure_artifacts = {
        "host_probe_stress_overlay_png": plot_overlay(
            host_rows_by_case,
            "nearest_host_axial_stress_MPa",
            "Host axial stress [MPa]",
            "Host probe axial stress",
            "reduced_rc_continuum_host_probe_stress",
            suffix,
            figure_dirs,
        ),
        "host_probe_strain_overlay_png": plot_overlay(
            host_rows_by_case,
            "nearest_host_axial_strain",
            "Host axial strain",
            "Host probe axial strain",
            "reduced_rc_continuum_host_probe_strain",
            suffix,
            figure_dirs,
        ),
        "host_probe_crack_overlay_png": plot_overlay(
            host_rows_by_case,
            "nearest_host_max_crack_opening",
            "Host crack opening [m]",
            "Host probe crack opening",
            "reduced_rc_continuum_host_probe_crack_opening",
            suffix,
            figure_dirs,
        ),
    }

    payload = {
        "promoted_bridge_dir": str(args.promoted_bridge_dir),
        "benchmark_exe": str(args.benchmark_exe),
        "probe": {
            "label": probe_label,
            "target_x_m": probe_x,
            "target_y_m": probe_y,
            "target_z_m": probe_z,
            "source_selected_bar": selected_bar,
        },
        "protocol": promoted_summary.get("protocol", {}),
        "cases": summary_rows,
        "interior_probe_vs_same_run_selected_bar_host_trace": {
            "max_abs_stress_gap_mpa": clean_optional(max_or_nan([abs(v) for v in probe_vs_bar_host_stress_gap])),
            "rms_abs_stress_gap_mpa": clean_optional(rms(probe_vs_bar_host_stress_gap)),
            "max_abs_strain_gap": clean_optional(max_or_nan([abs(v) for v in probe_vs_bar_host_strain_gap])),
            "rms_abs_strain_gap": clean_optional(rms(probe_vs_bar_host_strain_gap)),
        },
        "artifacts": {
            "summary_csv": str(args.output_dir / "continuum_host_probe_family_cases.csv"),
            **figure_artifacts,
        },
    }

    write_csv(args.output_dir / "continuum_host_probe_family_cases.csv", summary_rows)
    write_json(args.output_dir / "continuum_host_probe_family_summary.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
