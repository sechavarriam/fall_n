#!/usr/bin/env python3
"""
Compare the promoted structural reference against the three cover/core-aware
continuum families already frozen at 15 mm:

  * plain cover/core concrete,
  * interior embedded bars, and
  * boundary bars.

This pass intentionally reuses existing canonical bundles instead of rerunning
the expensive continuum solves. The goal is to separate:

  1. the global hysteresis gap,
  2. the local steel gap on embedded branches, and
  3. the local host-vs-bar kinematic closure.

That lets us decide whether the promoted local continuum model is really the
interior embedded branch, or whether a seemingly cheaper alternative is only
looking attractive because it is changing the physics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
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
class SelectedBarIdentity:
    bar_index: int
    bar_element_layer: int
    gp_index: int
    bar_y: float
    bar_z: float
    position_z_m: float


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
            "Compare the promoted structural reference against the existing "
            "cover/core-aware continuum family bundles."
        )
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
        "--continuum-family-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_continuum_cover_core_cyclic_audit_30mm_branches",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_family_comparison_30mm",
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
    parser.add_argument("--skip-figure-export", action="store_true")
    parser.add_argument("--column-height-m", type=float, default=3.2)
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


def max_or_nan(values: list[float]) -> float:
    return max(values) if values else math.nan


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else math.nan


def monotone_branches(rows: list[dict[str, Any]], x_field: str) -> list[list[dict[str, Any]]]:
    if len(rows) <= 1:
        return [rows] if rows else []
    tol = 1.0e-12
    branches: list[list[dict[str, Any]]] = []
    current = [rows[0]]
    current_direction = 0
    previous_x = float(rows[0][x_field])
    for row in rows[1:]:
        x = float(row[x_field])
        delta = x - previous_x
        direction = 0 if abs(delta) <= tol else (1 if delta > 0.0 else -1)
        if current_direction == 0 and direction != 0:
            current_direction = direction
        elif direction != 0 and current_direction != 0 and direction != current_direction:
            branches.append(current)
            current = [current[-1], row]
            current_direction = direction
        else:
            current.append(row)
        previous_x = x
    if current:
        branches.append(current)
    return branches


def interpolate(rows: list[dict[str, Any]], x_field: str, y_field: str, target_x: float) -> float:
    if not rows:
        return math.nan
    xs = [float(row[x_field]) for row in rows]
    ys = [float(row[y_field]) for row in rows]
    if len(rows) == 1:
        return ys[0]
    lo = min(xs)
    hi = max(xs)
    tol = 1.0e-12
    if target_x < lo - tol or target_x > hi + tol:
        return math.nan
    for left, right in zip(range(len(rows) - 1), range(1, len(rows))):
        x0 = xs[left]
        x1 = xs[right]
        y0 = ys[left]
        y1 = ys[right]
        if abs(target_x - x0) <= tol:
            return y0
        if abs(target_x - x1) <= tol:
            return y1
        if (x0 <= target_x <= x1) or (x1 <= target_x <= x0):
            if abs(x1 - x0) <= tol:
                return 0.5 * (y0 + y1)
            alpha = (target_x - x0) / (x1 - x0)
            return y0 + alpha * (y1 - y0)
    return math.nan


def branchwise_relative_metrics(
    lhs_rows: list[dict[str, Any]],
    rhs_rows: list[dict[str, Any]],
    x_field: str,
    lhs_y_field: str,
    rhs_y_field: str,
    *,
    active_floor: float | None = None,
) -> dict[str, float]:
    lhs_branches = monotone_branches(lhs_rows, x_field)
    rhs_branches = monotone_branches(rhs_rows, x_field)
    rel_errors: list[float] = []
    active_errors: list[float] = []
    for lhs_branch, rhs_branch in zip(lhs_branches, rhs_branches):
        for row in lhs_branch:
            lhs = float(row[lhs_y_field])
            rhs = interpolate(rhs_branch, x_field, rhs_y_field, float(row[x_field]))
            if not math.isfinite(rhs):
                continue
            rel = abs(lhs - rhs) / max(abs(rhs), 1.0e-12)
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
        dx = float(curr[x_field]) - float(prev[x_field])
        y_avg = 0.5 * (float(curr[y_field]) + float(prev[y_field]))
        total += y_avg * dx
    return total


def clean_optional(value: float) -> float | None:
    return value if math.isfinite(value) else None


def select_bar_identity(
    rows: list[dict[str, Any]],
    target_y: float,
    target_z: float,
    target_position_z_m: float,
) -> SelectedBarIdentity:
    unique_bars: dict[int, tuple[float, float]] = {}
    for row in rows:
        bar_index = int(row["bar_index"])
        unique_bars.setdefault(bar_index, (float(row["bar_y"]), float(row["bar_z"])))
    chosen_bar_index, (bar_y, bar_z) = min(
        unique_bars.items(),
        key=lambda item: abs(item[1][0] - target_y) + abs(item[1][1] - target_z),
    )
    base_candidates = [
        row
        for row in rows
        if int(row["bar_index"]) == chosen_bar_index
        and int(row["bar_element_layer"]) == 0
    ]
    if not base_candidates:
        raise RuntimeError("Continuum benchmark has no base-layer rebar candidates.")
    chosen_gp_row = min(
        base_candidates,
        key=lambda row: abs(float(row["position_z_m"]) - target_position_z_m),
    )
    return SelectedBarIdentity(
        bar_index=chosen_bar_index,
        bar_element_layer=int(chosen_gp_row["bar_element_layer"]),
        gp_index=int(chosen_gp_row["gp_index"]),
        bar_y=bar_y,
        bar_z=bar_z,
        position_z_m=float(chosen_gp_row["position_z_m"]),
    )


def select_bar_trace(rows: list[dict[str, Any]], identity: SelectedBarIdentity) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if int(row["bar_index"]) == identity.bar_index
        and int(row["bar_element_layer"]) == identity.bar_element_layer
        and int(row["gp_index"]) == identity.gp_index
    ]
    return sorted(selected, key=lambda row: (float(row["runtime_p"]), float(row["runtime_step"])))


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
        and abs(float(row["y"]) - identity.y) < 1.0e-12
        and abs(float(row["z"]) - identity.z) < 1.0e-12
    ]
    if not matching_rows:
        raise RuntimeError("Failed to recover the structural steel trace for the selected fiber.")

    def axial_position(row: dict[str, Any]) -> float:
        xi = float(row["xi"])
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
        grouped_by_step.setdefault((float(row["step"]), float(row["p"])), []).append(row)

    interpolated_trace: list[dict[str, Any]] = []
    for _, stations in sorted(grouped_by_step.items(), key=lambda item: item[0]):
        lower, upper, alpha = bracket(stations)

        def lerp(field: str) -> float:
            lhs = float(lower[field])
            rhs = float(upper[field])
            return lhs + alpha * (rhs - lhs)

        interpolated_trace.append(
            {
                "step": float(lower["step"]),
                "p": float(lower["p"]),
                "drift_m": float(lower["drift_m"]),
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


def save_figure(fig: plt.Figure, path: Path, secondary: Path | None) -> None:
    ensure_dir(path.parent)
    fig.savefig(path)
    if secondary is not None:
        ensure_dir(secondary.parent)
        fig.savefig(secondary)
    plt.close(fig)


def plot_hysteresis_overlay(
    structural_rows: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    out_dirs: list[Path],
    suffix: str,
) -> None:
    colors = {
        "plain": ORANGE,
        "interior": GREEN,
        "boundary": PURPLE,
    }
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(
        [1.0e3 * float(row["drift_m"]) for row in structural_rows],
        [1.0e3 * float(row["base_shear_MN"]) for row in structural_rows],
        color=BLUE,
        linewidth=1.6,
        label="Structural Timoshenko",
    )
    for case in cases:
        rows = case["hysteresis_rows"]
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [1.0e3 * float(row["base_shear_MN"]) for row in rows],
            color=(
                colors["plain"]
                if "plain" in case["key"]
                else colors["boundary"]
                if "boundary" in case["key"]
                else colors["interior"]
            ),
            linewidth=1.3,
            linestyle="--",
            label=case["legend_label"],
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title(f"Structural vs continuum family audit ({suffix} window)")
    ax.legend(frameon=False)
    fig.tight_layout()
    primary = out_dirs[0] / f"reduced_rc_structural_continuum_family_hysteresis_{suffix}.png"
    secondary = out_dirs[1] / primary.name if len(out_dirs) > 1 else None
    save_figure(fig, primary, secondary)


def plot_error_timing(cases: list[dict[str, Any]], out_dirs: list[Path], suffix: str) -> None:
    labels = [case["short_label"] for case in cases]
    global_rms = [case["global_metrics"]["rms_rel_error_active"] for case in cases]
    timings = [case["process_wall_seconds"] for case in cases]
    x = list(range(len(cases)))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].bar(x, global_rms, color=[ORANGE, GREEN, PURPLE][: len(cases)])
    axes[0].set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set_ylabel("Active RMS base-shear error")
    axes[0].set_title("Global bridge error")

    axes[1].bar(x, timings, color=[ORANGE, GREEN, PURPLE][: len(cases)])
    axes[1].set_xticks(x, labels, rotation=15, ha="right")
    axes[1].set_ylabel("Process wall time [s]")
    axes[1].set_title("Continuum family timing")
    fig.tight_layout()

    primary = out_dirs[0] / f"reduced_rc_structural_continuum_family_error_timing_{suffix}.png"
    secondary = out_dirs[1] / primary.name if len(out_dirs) > 1 else None
    save_figure(fig, primary, secondary)


def plot_steel_overlay(cases: list[dict[str, Any]], out_dirs: list[Path], suffix: str) -> None:
    embedded_cases = [case for case in cases if case.get("steel_trace_rows")]
    if not embedded_cases:
        return

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for case in embedded_cases:
        structural_steel_rows = case["matched_structural_steel_rows"]
        rows = case["steel_trace_rows"]
        color = PURPLE if "boundary" in case["key"] else GREEN
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in structural_steel_rows],
            [float(row["stress_xx_MPa"]) for row in structural_steel_rows],
            color=color,
            linewidth=1.6,
            label=f"Structural fiber ({case['short_label']})",
        )
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [float(row["stress_xx_MPa"]) for row in rows],
            color=color,
            linewidth=1.3,
            linestyle="--",
            label=case["legend_label"],
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Steel stress [MPa]")
    ax.set_title("Structural fiber vs continuum bar stress histories")
    ax.legend(frameon=False)
    fig.tight_layout()
    primary = out_dirs[0] / f"reduced_rc_structural_continuum_family_steel_{suffix}.png"
    secondary = out_dirs[1] / primary.name if len(out_dirs) > 1 else None
    save_figure(fig, primary, secondary)


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    promoted_summary = read_json(
        args.promoted_bridge_dir / "structural_continuum_steel_hysteresis_summary.json"
    )
    family_summary = read_json(args.continuum_family_dir / "continuum_cover_core_cyclic_summary.json")

    structural_rows = read_csv_rows(args.promoted_bridge_dir / "structural" / "hysteresis.csv")
    structural_fiber_rows = read_csv_rows(
        args.promoted_bridge_dir / "structural" / "section_fiber_state_history.csv"
    )
    selected_bar_payload = (
        promoted_summary.get("continuum_cases", {}).get("hex20", {}).get("selected_bar", {})
    )
    target_bar_y = float(selected_bar_payload.get("bar_y", 0.095))
    target_bar_z = float(selected_bar_payload.get("bar_z", 0.095))
    target_position_z_m = float(selected_bar_payload.get("position_z_m", 0.0))
    selected_structural_payload = (
        promoted_summary.get("selected_structural_fibers", {}).get("hex20", {})
    )
    structural_identity = StructuralSteelIdentity(
        fiber_index=int(selected_structural_payload.get("fiber_index", 85)),
        y=float(selected_structural_payload.get("y", 0.095)),
        z=float(selected_structural_payload.get("z", -0.095)),
        area=float(selected_structural_payload.get("area", 0.0)),
        zone=str(selected_structural_payload.get("zone", "longitudinal_steel")),
        material_role=str(selected_structural_payload.get("material_role", "reinforcing_steel")),
    )

    peak_structural_base_shear = max(abs(float(row["base_shear_MN"])) for row in structural_rows)
    base_shear_active_floor = 0.05 * peak_structural_base_shear

    case_payloads: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    figure_suffix = amplitude_suffix_from_protocol(promoted_summary.get("protocol", {}))

    for case_row in family_summary.get("cases", []):
        if not bool(case_row.get("completed_successfully", False)):
            continue
        case_key = str(case_row["key"])
        case_dir = Path(str(case_row["output_dir"]))
        hysteresis_rows = read_csv_rows(case_dir / "hysteresis.csv")
        global_metrics = branchwise_relative_metrics(
            hysteresis_rows,
            structural_rows,
            "drift_m",
            "base_shear_MN",
            "base_shear_MN",
            active_floor=base_shear_active_floor,
        )
        payload: dict[str, Any] = {
            "key": case_key,
            "label": str(case_row["label"]),
            "short_label": (
                "plain"
                if "plain" in case_key
                else "embedded interior"
                if "interior" in case_key
                else "boundary bars"
            ),
            "legend_label": (
                "Continuum plain"
                if "plain" in case_key
                else "Continuum embedded interior"
                if "interior" in case_key
                else "Continuum boundary bars"
            ),
            "reinforcement_mode": str(case_row["reinforcement_mode"]),
            "rebar_layout": str(case_row["rebar_layout"]),
            "process_wall_seconds": float(case_row["process_wall_seconds"]),
            "reported_total_wall_seconds": float(case_row["reported_total_wall_seconds"]),
            "hysteresis_rows": hysteresis_rows,
            "global_metrics": {
                **global_metrics,
                "structural_loop_work_mn_m": cyclic_loop_work(structural_rows, "drift_m", "base_shear_MN"),
                "continuum_loop_work_mn_m": cyclic_loop_work(hysteresis_rows, "drift_m", "base_shear_MN"),
            },
        }

        steel_trace_rows: list[dict[str, Any]] | None = None
        if str(case_row["reinforcement_mode"]) == "embedded-longitudinal-bars":
            rebar_rows = read_csv_rows(case_dir / "rebar_history.csv")
            selected_bar = select_bar_identity(
                rebar_rows,
                target_bar_y,
                target_bar_z,
                target_position_z_m,
            )
            steel_trace_rows = select_bar_trace(rebar_rows, selected_bar)
            structural_steel_rows = select_structural_steel_trace(
                structural_fiber_rows,
                structural_identity,
                selected_bar.position_z_m,
                args.column_height_m,
            )
            peak_structural_steel_stress = max(
                abs(float(row["stress_xx_MPa"])) for row in structural_steel_rows
            )
            steel_active_floor = 0.05 * peak_structural_steel_stress
            peak_structural_steel_strain = max(
                abs(float(row["strain_xx"])) for row in structural_steel_rows
            )
            payload["selected_bar"] = asdict(selected_bar)
            payload["matched_structural_steel_rows"] = structural_steel_rows
            payload["steel_trace_rows"] = steel_trace_rows
            payload["steel_metrics"] = {
                **branchwise_relative_metrics(
                    steel_trace_rows,
                    structural_steel_rows,
                    "drift_m",
                    "stress_xx_MPa",
                    "stress_xx_MPa",
                    active_floor=steel_active_floor,
                ),
                "max_rel_strain_error": clean_optional(
                    branchwise_relative_metrics(
                        steel_trace_rows,
                        structural_steel_rows,
                        "drift_m",
                        "axial_strain",
                        "strain_xx",
                        active_floor=0.05 * peak_structural_steel_strain,
                    )["max_rel_error"]
                ),
                "rms_rel_strain_error": clean_optional(
                    branchwise_relative_metrics(
                        steel_trace_rows,
                        structural_steel_rows,
                        "drift_m",
                        "axial_strain",
                        "strain_xx",
                        active_floor=0.05 * peak_structural_steel_strain,
                    )["rms_rel_error"]
                ),
                "structural_loop_work_mpa": cyclic_loop_work(
                    structural_steel_rows, "strain_xx", "stress_xx_MPa"
                ),
                "continuum_loop_work_mpa": cyclic_loop_work(
                    steel_trace_rows, "axial_strain", "stress_xx_MPa"
                ),
            }
            payload["selected_bar_locality"] = {
                "max_abs_host_bar_axial_strain_gap": max(
                    abs(float(row["projected_axial_strain_gap"])) for row in steel_trace_rows
                ),
                "rms_abs_host_bar_axial_strain_gap": rms(
                    [float(row["projected_axial_strain_gap"]) for row in steel_trace_rows]
                ),
                "max_abs_projected_gap_norm_m": max(
                    abs(float(row["projected_gap_norm_m"])) for row in steel_trace_rows
                ),
                "max_abs_host_crack_opening": max(
                    abs(float(row["nearest_host_max_crack_opening"])) for row in steel_trace_rows
                ),
                "peak_host_crack_count": max(
                    int(float(row["nearest_host_num_cracks"])) for row in steel_trace_rows
                ),
            }
            write_csv(
                args.output_dir / f"{case_key}_selected_bar_trace.csv",
                steel_trace_rows,
            )
        case_payloads.append(payload)

        csv_rows.append(
            {
                "key": case_key,
                "label": payload["label"],
                "reinforcement_mode": payload["reinforcement_mode"],
                "rebar_layout": payload["rebar_layout"],
                "process_wall_seconds": payload["process_wall_seconds"],
                "reported_total_wall_seconds": payload["reported_total_wall_seconds"],
                "max_rel_base_shear_error": clean_optional(global_metrics["max_rel_error"]),
                "rms_rel_base_shear_error": clean_optional(global_metrics["rms_rel_error"]),
                "max_rel_base_shear_error_active": clean_optional(global_metrics["max_rel_error_active"]),
                "rms_rel_base_shear_error_active": clean_optional(global_metrics["rms_rel_error_active"]),
                "steel_rms_rel_stress_error_active": clean_optional(
                    payload.get("steel_metrics", {}).get("rms_rel_error_active", math.nan)
                ),
                "max_abs_host_bar_axial_strain_gap": clean_optional(
                    payload.get("selected_bar_locality", {}).get(
                        "max_abs_host_bar_axial_strain_gap", math.nan
                    )
                ),
                "peak_host_crack_count": payload.get("selected_bar_locality", {}).get(
                    "peak_host_crack_count", None
                ),
            }
        )

    figure_dirs = (
        []
        if args.skip_figure_export
        else [args.figures_dir, args.secondary_figures_dir]
    )
    if figure_dirs:
        plot_hysteresis_overlay(structural_rows, case_payloads, figure_dirs, figure_suffix)
        plot_error_timing(case_payloads, figure_dirs, figure_suffix)
        plot_steel_overlay(case_payloads, figure_dirs, figure_suffix)

    summary = {
        "promoted_bridge_dir": str(args.promoted_bridge_dir),
        "continuum_family_dir": str(args.continuum_family_dir),
        "protocol": promoted_summary.get("protocol", {}),
        "structural_reference": promoted_summary.get("structural_reference", {}),
        "selected_structural_fiber": promoted_summary.get("selected_structural_fibers", {}).get("hex20", {}),
        "target_continuum_bar": {
            "bar_y": target_bar_y,
            "bar_z": target_bar_z,
            "position_z_m": target_position_z_m,
        },
        "cases": [
            {
                "key": case["key"],
                "label": case["label"],
                "reinforcement_mode": case["reinforcement_mode"],
                "rebar_layout": case["rebar_layout"],
                "process_wall_seconds": case["process_wall_seconds"],
                "reported_total_wall_seconds": case["reported_total_wall_seconds"],
                "global_metrics": {
                    key: clean_optional(value)
                    for key, value in case["global_metrics"].items()
                },
                "selected_bar": case.get("selected_bar", {}),
                "steel_metrics": {
                    key: clean_optional(value)
                    for key, value in case.get("steel_metrics", {}).items()
                },
                "selected_bar_locality": {
                    key: clean_optional(value) if isinstance(value, float) else value
                    for key, value in case.get("selected_bar_locality", {}).items()
                },
            }
            for case in case_payloads
        ],
        "artifacts": {
            "cases_csv": str(args.output_dir / "structural_continuum_family_comparison_cases.csv"),
            "hysteresis_figure": str(
                args.figures_dir / f"reduced_rc_structural_continuum_family_hysteresis_{figure_suffix}.png"
            ),
            "error_timing_figure": str(
                args.figures_dir / f"reduced_rc_structural_continuum_family_error_timing_{figure_suffix}.png"
            ),
            "steel_figure": str(
                args.figures_dir / f"reduced_rc_structural_continuum_family_steel_{figure_suffix}.png"
            ),
        },
    }

    write_csv(args.output_dir / "structural_continuum_family_comparison_cases.csv", csv_rows)
    write_json(
        args.output_dir / "structural_continuum_family_comparison_summary.json",
        summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
