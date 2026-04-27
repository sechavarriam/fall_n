#!/usr/bin/env python3
"""Compare the promoted interior and boundary rebar continuum branches locally.

This audit is intentionally narrower than the global family-vs-structural
comparison. It focuses on the selected steel path and its nearest host
neighbourhood so we can tell whether the cheaper `boundary bars` branch is
globally closer for the right physical reason or only because it changes the
local damage story around the reinforcement.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
class StructuralSteelIdentity:
    fiber_index: int
    y: float
    z: float
    area: float
    zone: str
    material_role: str


def amplitude_suffix_from_protocol(protocol: dict[str, Any]) -> str:
    amplitudes = protocol.get("amplitudes_mm")
    if isinstance(amplitudes, list) and amplitudes:
        peak_mm = max(float(value) for value in amplitudes)
    else:
        peak_mm = float(protocol.get("monotonic_tip_mm", 0.0))
    label = f"{peak_mm:.3f}".rstrip("0").rstrip(".")
    return label.replace(".", "p") + "mm"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compare the promoted interior and boundary continuum branches on the "
            "same local steel/host neighbourhood read."
        )
    )
    parser.add_argument(
        "--family-comparison-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_family_comparison_30mm",
    )
    parser.add_argument(
        "--promoted-bridge-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_promoted_cyclic_30mm_audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_branch_locality_comparison_30mm",
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
    parser.add_argument("--column-height-m", type=float, default=3.2)
    parser.add_argument("--skip-figure-export", action="store_true")
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
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def rms(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return math.nan
    return math.sqrt(sum(v * v for v in clean) / len(clean))


def max_or_nan(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return max(clean) if clean else math.nan


def clean_optional(value: float) -> float | None:
    return value if math.isfinite(value) else None


def branchwise_relative_metrics(
    lhs_rows: list[dict[str, Any]],
    rhs_rows: list[dict[str, Any]],
    x_field: str,
    y_field: str,
    *,
    active_floor: float | None = None,
) -> dict[str, float]:
    if not lhs_rows or not rhs_rows:
        return {
            "max_rel_error": math.nan,
            "rms_rel_error": math.nan,
            "max_rel_error_active": math.nan,
            "rms_rel_error_active": math.nan,
        }

    rhs_by_x = {safe_float(row[x_field]): safe_float(row[y_field]) for row in rhs_rows}
    rel_errors: list[float] = []
    active_errors: list[float] = []
    for row in lhs_rows:
        x = safe_float(row[x_field])
        lhs = safe_float(row[y_field])
        rhs = rhs_by_x.get(x, math.nan)
        if not (math.isfinite(lhs) and math.isfinite(rhs)):
            continue
        denom = max(abs(rhs), 1.0e-12)
        rel = abs(lhs - rhs) / denom
        rel_errors.append(rel)
        if active_floor is not None and abs(rhs) >= active_floor:
            active_errors.append(rel)
    return {
        "max_rel_error": max_or_nan(rel_errors),
        "rms_rel_error": rms(rel_errors),
        "max_rel_error_active": max_or_nan(active_errors),
        "rms_rel_error_active": rms(active_errors),
    }


def cyclic_loop_work(rows: list[dict[str, Any]], x_field: str, y_field: str) -> float:
    if len(rows) < 2:
        return math.nan
    total = 0.0
    for prev, curr in zip(rows[:-1], rows[1:]):
        dx = safe_float(curr[x_field]) - safe_float(prev[x_field])
        y_avg = 0.5 * (safe_float(curr[y_field]) + safe_float(prev[y_field]))
        if math.isfinite(dx) and math.isfinite(y_avg):
            total += y_avg * dx
    return total


def select_structural_steel_trace(
    rows: list[dict[str, Any]],
    identity: StructuralSteelIdentity,
    target_position_z_m: float,
    column_height_m: float,
) -> list[dict[str, Any]]:
    matching_rows = [
        row
        for row in rows
        if int(row["fiber_index"]) == identity.fiber_index
        and abs(safe_float(row["y"]) - identity.y) < 1.0e-12
        and abs(safe_float(row["z"]) - identity.z) < 1.0e-12
    ]
    if not matching_rows:
        raise RuntimeError("Failed to recover the structural steel trace for the selected fiber.")

    def axial_position(row: dict[str, Any]) -> float:
        xi = safe_float(row["xi"])
        return 0.5 * column_height_m * (xi + 1.0)

    def bracket(stations: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], float]:
        ordered = sorted(stations, key=axial_position)
        if len(ordered) == 1:
            return ordered[0], ordered[0], 0.0
        first = ordered[0]
        last = ordered[-1]
        if target_position_z_m <= axial_position(first):
            return first, first, 0.0
        if target_position_z_m >= axial_position(last):
            return last, last, 0.0
        for left, right in zip(ordered[:-1], ordered[1:]):
            z0 = axial_position(left)
            z1 = axial_position(right)
            if z0 <= target_position_z_m <= z1:
                if abs(z1 - z0) <= 1.0e-14:
                    return left, right, 0.0
                alpha = (target_position_z_m - z0) / (z1 - z0)
                return left, right, alpha
        return last, last, 0.0

    grouped_by_step: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in matching_rows:
        grouped_by_step.setdefault((safe_float(row["step"]), safe_float(row["p"])), []).append(row)

    interpolated_trace: list[dict[str, Any]] = []
    for _, stations in sorted(grouped_by_step.items(), key=lambda item: item[0]):
        lower, upper, alpha = bracket(stations)

        def lerp(field: str) -> float:
            lhs = safe_float(lower[field])
            rhs = safe_float(upper[field])
            return lhs + alpha * (rhs - lhs)

        interpolated_trace.append(
            {
                "step": safe_float(lower["step"]),
                "p": safe_float(lower["p"]),
                "drift_m": safe_float(lower["drift_m"]),
                "position_z_m": axial_position(lower)
                + alpha * (axial_position(upper) - axial_position(lower)),
                "strain_xx": lerp("strain_xx"),
                "stress_xx_MPa": lerp("stress_xx_MPa"),
                "tangent_xx_MPa": lerp("tangent_xx_MPa"),
                "fiber_index": identity.fiber_index,
                "y": identity.y,
                "z": identity.z,
            }
        )
    return interpolated_trace


def save_figure(fig: plt.Figure, out_dirs: list[Path], name: str) -> list[str]:
    saved: list[str] = []
    for directory in out_dirs:
        ensure_dir(directory)
        target = directory / f"{name}.png"
        fig.savefig(target)
        saved.append(str(target))
    plt.close(fig)
    return saved


def plot_branch_locality_overlay(
    structural_rows: list[dict[str, Any]],
    interior_rows: list[dict[str, Any]] | None,
    boundary_rows: list[dict[str, Any]] | None,
    out_dirs: list[Path],
    suffix: str,
) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))

    axes[0].plot(
        [safe_float(row["drift_m"]) * 1.0e3 for row in structural_rows],
        [safe_float(row["stress_xx_MPa"]) for row in structural_rows],
        color=BLUE,
        linewidth=2.0,
        label="Structural steel fiber",
    )
    if interior_rows:
        axes[0].plot(
            [safe_float(row["drift_m"]) * 1.0e3 for row in interior_rows],
            [safe_float(row["stress_xx_MPa"]) for row in interior_rows],
            color=GREEN,
            linewidth=1.8,
            label="Continuum interior bar",
        )
    if boundary_rows:
        axes[0].plot(
            [safe_float(row["drift_m"]) * 1.0e3 for row in boundary_rows],
            [safe_float(row["stress_xx_MPa"]) for row in boundary_rows],
            color=ORANGE,
            linewidth=1.8,
            linestyle="--",
            label="Continuum boundary bar",
        )
    axes[0].set_xlabel("Tip drift [mm]")
    axes[0].set_ylabel("Steel stress [MPa]")
    axes[0].set_title("Selected steel path")
    axes[0].legend(frameon=True, fontsize=8)

    if interior_rows:
        axes[1].plot(
            [safe_float(row["drift_m"]) * 1.0e3 for row in interior_rows],
            [safe_float(row["nearest_host_max_crack_opening"]) * 1.0e3 for row in interior_rows],
            color=GREEN,
            linewidth=1.8,
            label="Interior nearest host",
        )
    if boundary_rows:
        axes[1].plot(
            [safe_float(row["drift_m"]) * 1.0e3 for row in boundary_rows],
            [safe_float(row["nearest_host_max_crack_opening"]) * 1.0e3 for row in boundary_rows],
            color=ORANGE,
            linewidth=1.8,
            linestyle="--",
            label="Boundary nearest host",
        )
    axes[1].set_xlabel("Tip drift [mm]")
    axes[1].set_ylabel("Nearest-host crack opening [mm]")
    axes[1].set_title("Host neighborhood cracking")
    axes[1].legend(frameon=True, fontsize=8)

    if interior_rows:
        axes[2].plot(
            [safe_float(row["drift_m"]) * 1.0e3 for row in interior_rows],
            [
                (safe_float(row["axial_strain"]) - safe_float(row["nearest_host_axial_strain"])) * 1.0e6
                for row in interior_rows
            ],
            color=GREEN,
            linewidth=1.8,
            label="Interior bar - nearest host",
        )
    if boundary_rows:
        axes[2].plot(
            [safe_float(row["drift_m"]) * 1.0e3 for row in boundary_rows],
            [
                (safe_float(row["axial_strain"]) - safe_float(row["nearest_host_axial_strain"])) * 1.0e6
                for row in boundary_rows
            ],
            color=ORANGE,
            linewidth=1.8,
            linestyle="--",
            label="Boundary bar - nearest host",
        )
    axes[2].set_xlabel("Tip drift [mm]")
    axes[2].set_ylabel("Local strain gap [με]")
    axes[2].set_title("Bar vs nearest-host strain")
    axes[2].legend(frameon=True, fontsize=8)

    fig.suptitle(f"Continuum branch locality comparison at {suffix} cyclic window")
    return save_figure(fig, out_dirs, f"reduced_rc_structural_continuum_branch_locality_overlay_{suffix}")


def plot_branch_locality_overview(rows: list[dict[str, Any]], out_dirs: list[Path], suffix: str) -> list[str]:
    labels = [str(row["label"]) for row in rows]
    global_error = [safe_float(row["global_rms_active"]) for row in rows]
    steel_error = [safe_float(row["steel_rms_active"]) for row in rows]
    peak_crack_mm = [safe_float(row["peak_host_crack_opening_mm"]) for row in rows]
    wall_time = [safe_float(row["process_wall_seconds"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    x = list(range(len(rows)))

    axes[0, 0].bar(x, global_error, color=[GREEN, ORANGE])
    axes[0, 0].set_title("Active RMS base-shear error")
    axes[0, 0].set_xticks(x, labels, rotation=12, ha="right")

    axes[0, 1].bar(x, steel_error, color=[GREEN, ORANGE])
    axes[0, 1].set_title("Active RMS steel-stress error")
    axes[0, 1].set_xticks(x, labels, rotation=12, ha="right")

    axes[1, 0].bar(x, peak_crack_mm, color=[GREEN, ORANGE])
    axes[1, 0].set_title("Peak nearest-host crack opening [mm]")
    axes[1, 0].set_xticks(x, labels, rotation=12, ha="right")

    axes[1, 1].bar(x, wall_time, color=[GREEN, ORANGE])
    axes[1, 1].set_title("Process wall time [s]")
    axes[1, 1].set_xticks(x, labels, rotation=12, ha="right")

    fig.suptitle(f"Continuum branch locality overview ({suffix})")
    fig.tight_layout()
    return save_figure(fig, out_dirs, f"reduced_rc_structural_continuum_branch_locality_overview_{suffix}")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    family_summary = read_json(
        args.family_comparison_dir / "structural_continuum_family_comparison_summary.json"
    )
    figure_suffix = amplitude_suffix_from_protocol(family_summary.get("protocol", {}))
    bridge_dir = Path(family_summary.get("promoted_bridge_dir", args.promoted_bridge_dir))
    structural_identity_payload = family_summary["selected_structural_fiber"]
    structural_identity = StructuralSteelIdentity(
        fiber_index=int(structural_identity_payload["fiber_index"]),
        y=safe_float(structural_identity_payload["y"]),
        z=safe_float(structural_identity_payload["z"]),
        area=safe_float(structural_identity_payload["area"]),
        zone=str(structural_identity_payload["zone"]),
        material_role=str(structural_identity_payload["material_role"]),
    )
    structural_rows = read_csv_rows(bridge_dir / "structural" / "section_fiber_state_history.csv")

    embedded_cases = [
        case
        for case in family_summary["cases"]
        if str(case.get("reinforcement_mode", "")).startswith("embedded")
    ]
    if not embedded_cases:
        raise RuntimeError("Expected at least one embedded continuum family case.")

    summary_rows: list[dict[str, Any]] = []
    plotted_rows: dict[str, list[dict[str, Any]]] = {}
    aligned_exports: list[tuple[str, list[dict[str, Any]]]] = []
    interior_key: str | None = None
    boundary_key: str | None = None

    for case in embedded_cases:
        key = str(case["key"])
        label = "Interior bars" if "interior" in key else "Boundary bars"
        selected_bar = case["selected_bar"]
        selected_trace = read_csv_rows(args.family_comparison_dir / f"{key}_selected_bar_trace.csv")
        structural_trace = select_structural_steel_trace(
            structural_rows,
            structural_identity,
            safe_float(selected_bar["position_z_m"]),
            args.column_height_m,
        )
        steel_floor = (
            0.05
            * max(
                abs(safe_float(row["stress_xx_MPa"])) for row in structural_trace
            )
            if structural_trace
            else None
        )
        steel_metrics = branchwise_relative_metrics(
            selected_trace,
            structural_trace,
            "drift_m",
            "stress_xx_MPa",
            active_floor=steel_floor,
        )
        strain_metrics = branchwise_relative_metrics(
            [
                {
                    "drift_m": row["drift_m"],
                    "value": row["axial_strain"],
                }
                for row in selected_trace
            ],
            [
                {
                    "drift_m": row["drift_m"],
                    "value": row["strain_xx"],
                }
                for row in structural_trace
            ],
            "drift_m",
            "value",
            active_floor=(
                0.05 * max(abs(safe_float(row["strain_xx"])) for row in structural_trace)
                if structural_trace
                else None
            ),
        )
        host_bar_strain_gap = [
            abs(safe_float(row["axial_strain"]) - safe_float(row["nearest_host_axial_strain"]))
            for row in selected_trace
        ]
        projected_vs_nearest_gap = [
            abs(
                safe_float(row["host_projected_axial_strain"])
                - safe_float(row["nearest_host_axial_strain"])
            )
            for row in selected_trace
        ]
        peak_host_crack = max_or_nan(
            [safe_float(row["nearest_host_max_crack_opening"]) for row in selected_trace]
        )
        peak_host_crack_count = max_or_nan(
            [safe_float(row["nearest_host_num_cracks"]) for row in selected_trace]
        )
        peak_host_damage = max_or_nan(
            [safe_float(row["nearest_host_damage"]) for row in selected_trace]
        )
        peak_host_distance = max_or_nan(
            [safe_float(row["nearest_host_gp_distance_m"]) for row in selected_trace]
        )
        peak_bar_stress = max_or_nan([abs(safe_float(row["stress_xx_MPa"])) for row in selected_trace])
        peak_bar_strain = max_or_nan([abs(safe_float(row["axial_strain"])) for row in selected_trace])
        peak_host_strain = max_or_nan(
            [abs(safe_float(row["nearest_host_axial_strain"])) for row in selected_trace]
        )
        peak_host_stress = max_or_nan(
            [abs(safe_float(row["nearest_host_axial_stress_MPa"])) for row in selected_trace]
        )
        structural_target_y = abs(safe_float(structural_identity.y))
        structural_target_z = abs(safe_float(structural_identity.z))
        aligned_rows: list[dict[str, Any]] = []
        structural_by_drift = {
            safe_float(row["drift_m"]): row for row in structural_trace
        }
        for row in selected_trace:
            drift = safe_float(row["drift_m"])
            structural_row = structural_by_drift.get(drift)
            aligned_rows.append(
                {
                    "drift_m": drift,
                    "structural_steel_stress_MPa": (
                        safe_float(structural_row["stress_xx_MPa"]) if structural_row else math.nan
                    ),
                    "continuum_bar_stress_MPa": safe_float(row["stress_xx_MPa"]),
                    "structural_steel_strain": (
                        safe_float(structural_row["strain_xx"]) if structural_row else math.nan
                    ),
                    "continuum_bar_strain": safe_float(row["axial_strain"]),
                    "nearest_host_axial_strain": safe_float(row["nearest_host_axial_strain"]),
                    "nearest_host_axial_stress_MPa": safe_float(row["nearest_host_axial_stress_MPa"]),
                    "nearest_host_max_crack_opening": safe_float(row["nearest_host_max_crack_opening"]),
                    "nearest_host_num_cracks": safe_float(row["nearest_host_num_cracks"]),
                    "bar_minus_host_strain": safe_float(row["axial_strain"])
                    - safe_float(row["nearest_host_axial_strain"]),
                }
            )

        aligned_exports.append((key, aligned_rows))
        plotted_rows[key] = selected_trace
        if "interior" in key:
            interior_key = key
        elif "boundary" in key:
            boundary_key = key

        summary_rows.append(
            {
                "key": key,
                "label": label,
                "bar_y_m": clean_optional(safe_float(selected_bar["bar_y"])),
                "bar_z_m": clean_optional(safe_float(selected_bar["bar_z"])),
                "position_z_m": clean_optional(safe_float(selected_bar["position_z_m"])),
                "offset_from_structural_target_y_mm": clean_optional(
                    1.0e3 * (abs(safe_float(selected_bar["bar_y"])) - structural_target_y)
                ),
                "offset_from_structural_target_z_mm": clean_optional(
                    1.0e3 * (abs(safe_float(selected_bar["bar_z"])) - structural_target_z)
                ),
                "process_wall_seconds": clean_optional(safe_float(case["process_wall_seconds"])),
                "reported_total_wall_seconds": clean_optional(
                    safe_float(case["reported_total_wall_seconds"])
                ),
                "global_rms_active": clean_optional(
                    safe_float(case["global_metrics"]["rms_rel_error_active"])
                ),
                "steel_rms_active": clean_optional(safe_float(steel_metrics["rms_rel_error_active"])),
                "steel_strain_rms_active": clean_optional(
                    safe_float(strain_metrics["rms_rel_error_active"])
                ),
                "max_abs_bar_minus_nearest_host_strain": clean_optional(max_or_nan(host_bar_strain_gap)),
                "rms_bar_minus_nearest_host_strain": clean_optional(rms(host_bar_strain_gap)),
                "max_abs_projected_minus_nearest_host_strain": clean_optional(
                    max_or_nan(projected_vs_nearest_gap)
                ),
                "rms_projected_minus_nearest_host_strain": clean_optional(
                    rms(projected_vs_nearest_gap)
                ),
                "peak_host_crack_opening_mm": clean_optional(1.0e3 * peak_host_crack),
                "peak_host_crack_count": clean_optional(peak_host_crack_count),
                "peak_host_damage": clean_optional(peak_host_damage),
                "nearest_host_gp_distance_mm": clean_optional(1.0e3 * peak_host_distance),
                "peak_bar_stress_MPa": clean_optional(peak_bar_stress),
                "peak_bar_strain": clean_optional(peak_bar_strain),
                "peak_host_axial_strain": clean_optional(peak_host_strain),
                "peak_host_axial_stress_MPa": clean_optional(peak_host_stress),
                "bar_loop_work_MPa": clean_optional(
                    cyclic_loop_work(selected_trace, "axial_strain", "stress_xx_MPa")
                ),
            }
        )

    write_csv(args.output_dir / "branch_locality_comparison_cases.csv", summary_rows)
    for key, rows in aligned_exports:
        write_csv(args.output_dir / f"{key}_aligned_locality_trace.csv", rows)

    summary_by_key = {row["key"]: row for row in summary_rows}
    if interior_key is not None and boundary_key is not None:
        interior_row = summary_by_key[interior_key]
        boundary_row = summary_by_key[boundary_key]
        interior_global = safe_float(interior_row.get("global_rms_active"))
        boundary_global = safe_float(boundary_row.get("global_rms_active"))
        interior_steel = safe_float(interior_row.get("steel_rms_active"))
        boundary_steel = safe_float(boundary_row.get("steel_rms_active"))
        boundary_cheaper = safe_float(boundary_row.get("process_wall_seconds")) < safe_float(
            interior_row.get("process_wall_seconds")
        )
        if math.isfinite(boundary_global) and math.isfinite(interior_global):
            if boundary_global < interior_global:
                reason = (
                    "The boundary branch remains cheaper and globally closer to the clamped "
                    "structural hysteresis, but it does so with steel paths shifted 30 mm "
                    "outward in each section axis and with a more strongly cracked host "
                    "neighborhood around the selected bar."
                )
            else:
                steel_clause = ""
                if math.isfinite(boundary_steel) and math.isfinite(interior_steel):
                    steel_clause = (
                        f" It is also locally worse on the steel path "
                        f"({boundary_steel:.3g} vs {interior_steel:.3g} active RMS)."
                    )
                cost_clause = (
                    " It remains cheaper operationally."
                    if boundary_cheaper
                    else " It is no longer cheaper either."
                )
                reason = (
                    "At this amplitude the promoted interior branch is now both more honest "
                    "geometrically and better aligned with the clamped structural hysteresis "
                    f"than the boundary control ({interior_global:.3g} vs {boundary_global:.3g} "
                    f"active global RMS).{steel_clause}{cost_clause} The boundary branch still "
                    "changes the physics by moving the steel paths 30 mm outward in each "
                    "section axis and by driving a more strongly cracked host neighborhood."
                )
        else:
            reason = (
                "Both interior and boundary branches are available, but the comparison metrics "
                "are incomplete. The interior branch remains promoted because it preserves the "
                "intended steel geometry, while the boundary branch still shifts the bars 30 mm "
                "outward in each section axis."
            )
        interpretation = {
            "promoted_local_baseline_remains": "embedded interior",
            "reason": reason,
            "structural_target_bar_m": {
                "abs_y": abs(structural_identity.y),
                "abs_z": abs(structural_identity.z),
            },
        }
    elif boundary_key is not None:
        interpretation = {
            "promoted_local_baseline_remains": "embedded interior below the current operational frontier",
            "reason": (
                "At this amplitude the boundary branch still completes, but the interior branch "
                "does not finish within the current budget. The boundary branch is therefore a "
                "useful frontier control, not a promoted replacement for the microscale-oriented "
                "interior baseline."
            ),
            "structural_target_bar_m": {
                "abs_y": abs(structural_identity.y),
                "abs_z": abs(structural_identity.z),
            },
        }
    else:
        interpretation = {
            "promoted_local_baseline_remains": "embedded interior",
            "reason": (
                "Only the interior branch is available in this comparison window, so it remains "
                "the reference local branch by default."
            ),
            "structural_target_bar_m": {
                "abs_y": abs(structural_identity.y),
                "abs_z": abs(structural_identity.z),
            },
        }
    payload = {
        "family_comparison_dir": str(args.family_comparison_dir),
        "promoted_bridge_dir": str(bridge_dir),
        "selected_structural_fiber": {
            "fiber_index": structural_identity.fiber_index,
            "y": structural_identity.y,
            "z": structural_identity.z,
            "area": structural_identity.area,
            "zone": structural_identity.zone,
            "material_role": structural_identity.material_role,
        },
        "cases": summary_rows,
        "interpretation": interpretation,
    }
    write_json(args.output_dir / "branch_locality_comparison_summary.json", payload)

    if not args.skip_figure_export:
        out_dirs = [args.figures_dir, args.secondary_figures_dir]
        artifacts = {
            "branch_locality_overlay_png": plot_branch_locality_overlay(
                select_structural_steel_trace(
                    structural_rows,
                    structural_identity,
                    safe_float(
                        next(
                            case["selected_bar"]["position_z_m"]
                            for case in embedded_cases
                            if case["key"] == (interior_key or boundary_key)
                        )
                    ),
                    args.column_height_m,
                ),
                plotted_rows.get(interior_key) if interior_key is not None else None,
                plotted_rows.get(boundary_key) if boundary_key is not None else None,
                out_dirs,
                figure_suffix,
            ),
            "branch_locality_overview_png": plot_branch_locality_overview(
                summary_rows, out_dirs, figure_suffix
            ),
        }
        payload["artifacts"] = artifacts
        write_json(args.output_dir / "branch_locality_comparison_summary.json", payload)


if __name__ == "__main__":
    main()
