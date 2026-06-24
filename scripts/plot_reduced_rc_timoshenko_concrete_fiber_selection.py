#!/usr/bin/env python3
"""Extreme concrete-fiber hysteresis selection plots for the reduced RC matrix."""

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

from plot_reduced_rc_timoshenko_steel_hysteresis_selection import (
    AXIAL_CONTRACT_TOLERANCE_MN,
    DEFAULT_EXPECTED_AXIAL_MN,
    QUAD_COLOR,
    QUAD_LABEL,
    axial_rejection_by_step,
    expected_axial_mn,
    finite,
    fiber_history_path,
    load_case_rows,
    loop_area,
    read_csv,
    rel_area_error,
    rms_rel_at_p,
    save_figure,
    unique_by_p,
    write_csv,
)


ROLE_SPECS = {
    "unconfined_concrete": {
        "label": "extreme unconfined concrete",
        "short_label": "Unconfined concrete",
        "stem": "extreme_unconfined_concrete",
    },
    "confined_concrete": {
        "label": "extreme confined concrete",
        "short_label": "Confined concrete",
        "stem": "extreme_confined_concrete",
    },
}


@dataclass(frozen=True)
class ConcreteMetric:
    material_role: str
    beam_nodes: int
    beam_integration: str
    status: str
    bundle_dir: str
    wall_seconds: float
    hysteresis_rms_rel: float
    work_rel_error: float
    fiber_stress_rms_rel: float
    fiber_strain_rms_rel: float
    fiber_loop_work_rel_error: float
    fiber_outlier_count: int
    section_gp: int
    fiber_index: int
    fiber_y: float
    fiber_z: float
    fiber_score: float
    production_score: float


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
        "timoshenko_matrix_reproduced_historical_closure_20260520",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo / "doc/figures/validation_reboot",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=repo / "PhD_Thesis/Figuras/validation_reboot",
    )
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=10)
    return parser.parse_args()


def base_station_id(rows: list[dict[str, Any]]) -> int:
    candidates: dict[int, float] = {}
    for row in rows:
        gp = int(finite(row.get("section_gp"), 0))
        xi = finite(row.get("xi"))
        z_m = finite(row.get("z_m"))
        coordinate = z_m if math.isfinite(z_m) else xi
        if gp not in candidates or coordinate < candidates[gp]:
            candidates[gp] = coordinate
    if not candidates:
        return 0
    return min(candidates.items(), key=lambda item: item[1])[0]


def choose_extreme_fiber(rows: list[dict[str, Any]], section_gp: int, role: str) -> dict[str, Any]:
    first_step = min(
        (
            int(finite(row.get("step"), 0))
            for row in rows
            if int(finite(row.get("section_gp"), -1)) == section_gp
        ),
        default=0,
    )
    candidates = [
        row
        for row in rows
        if int(finite(row.get("section_gp"), -1)) == section_gp
        and int(finite(row.get("step"), -1)) == first_step
        and str(row.get("material_role", "")) == role
    ]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: (
            round(abs(finite(row.get("z"))), 10),
            round(abs(finite(row.get("y"))), 10),
            1 if finite(row.get("z")) < 0.0 else 0,
            1 if finite(row.get("y")) < 0.0 else 0,
            -finite(row.get("fiber_index"), 1.0e12),
        ),
    )


def fiber_curve(
    bundle_dir: Path,
    case_name: str,
    role: str,
) -> tuple[list[dict[str, float]], dict[str, Any], list[dict[str, Any]]]:
    fibers = read_csv(fiber_history_path(bundle_dir))
    section_gp = base_station_id(fibers)
    selected = choose_extreme_fiber(fibers, section_gp, role)
    if not selected:
        return [], {}, []
    fiber_index = int(finite(selected.get("fiber_index"), -1))
    rejected_steps, rejected_rows = axial_rejection_by_step(bundle_dir, case_name)
    curve: list[dict[str, float]] = []
    for row in fibers:
        if int(finite(row.get("section_gp"), -1)) != section_gp:
            continue
        if int(finite(row.get("fiber_index"), -2)) != fiber_index:
            continue
        step = int(finite(row.get("step"), -1))
        if step in rejected_steps:
            continue
        p = finite(row.get("p"))
        strain = finite(row.get("strain_xx"))
        stress = finite(row.get("stress_xx_MPa"))
        if not (math.isfinite(p) and math.isfinite(strain) and math.isfinite(stress)):
            rejected_rows.append(
                {
                    "case": case_name,
                    "material_role": role,
                    "step": step,
                    "p": p,
                    "section_gp": section_gp,
                    "fiber_index": fiber_index,
                    "strain_xx": strain,
                    "stress_xx_MPa": stress,
                    "reason": "nonfinite_concrete_fiber_row",
                }
            )
            continue
        curve.append({"p": p, "strain": strain, "stress": stress})
    info = {
        "section_gp": section_gp,
        "fiber_index": fiber_index,
        "y": finite(selected.get("y")),
        "z": finite(selected.get("z")),
        "zone": selected.get("zone", ""),
    }
    return sorted(curve, key=lambda row: (row["p"], row["strain"])), info, rejected_rows


def compute_role_metrics(
    args: argparse.Namespace,
    role: str,
) -> tuple[list[ConcreteMetric], list[dict[str, float]], dict[tuple[int, str], list[dict[str, float]]], list[dict[str, Any]], dict[str, Any]]:
    case_rows = load_case_rows(args.matrix_dir)
    ref_dir = args.matrix_dir / "opensees_hifi_reference"
    ref_curve_raw, ref_info, ref_outliers = fiber_curve(ref_dir, "opensees_hifi_reference", role)
    ref_curve = unique_by_p(ref_curve_raw)
    metrics: list[ConcreteMetric] = []
    series_cache: dict[tuple[int, str], list[dict[str, float]]] = {}
    outliers: list[dict[str, Any]] = ref_outliers
    for (n, q), case in sorted(case_rows.items()):
        if n < args.min_n or n > args.max_n:
            continue
        status = str(case.get("status", ""))
        bundle = Path(str(case.get("bundle_dir", "")))
        if status != "completed" or not bundle.exists():
            metrics.append(
                ConcreteMetric(
                    role,
                    n,
                    q,
                    status,
                    str(bundle),
                    finite(case.get("process_wall_seconds")),
                    finite(case.get("hifi_hysteresis_rms_rel_error")),
                    finite(case.get("hifi_total_work_rel_error")),
                    math.nan,
                    math.nan,
                    math.nan,
                    0,
                    -1,
                    -1,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                )
            )
            continue
        case_name = f"n{n:02d}_{q}"
        curve_raw, info, case_outliers = fiber_curve(bundle, case_name, role)
        for item in case_outliers:
            item.setdefault("material_role", role)
        outliers.extend(case_outliers)
        curve = unique_by_p(curve_raw)
        series_cache[(n, q)] = curve
        stress_rms = rms_rel_at_p(curve, ref_curve, "stress")
        strain_rms = rms_rel_at_p(curve, ref_curve, "strain")
        area = rel_area_error(curve, ref_curve)
        hys = finite(case.get("hifi_hysteresis_rms_rel_error"))
        work = finite(case.get("hifi_total_work_rel_error"))
        wall = finite(case.get("process_wall_seconds"))
        fiber_score = (
            0.45 * stress_rms + 0.35 * strain_rms + 0.20 * area
            if all(math.isfinite(v) for v in (stress_rms, strain_rms, area))
            else math.nan
        )
        production_score = (
            0.20 * hys
            + 0.15 * work
            + 0.30 * stress_rms
            + 0.20 * strain_rms
            + 0.15 * area
            + 0.03 * math.log10(max(wall, 1.0))
            if all(math.isfinite(v) for v in (hys, work, stress_rms, strain_rms, area, wall))
            else math.nan
        )
        metrics.append(
            ConcreteMetric(
                role,
                n,
                q,
                status,
                str(bundle),
                wall,
                hys,
                work,
                stress_rms,
                strain_rms,
                area,
                len(case_outliers),
                int(info.get("section_gp", -1)),
                int(info.get("fiber_index", -1)),
                finite(info.get("y")),
                finite(info.get("z")),
                fiber_score,
                production_score,
            )
        )
    return metrics, ref_curve, series_cache, outliers, ref_info


def plot_role_metrics(
    role: str,
    metrics: list[ConcreteMetric],
    figures_dir: Path,
    secondary_dir: Path | None,
) -> list[str]:
    spec = ROLE_SPECS[role]
    completed = [m for m in metrics if m.status == "completed"]
    quadratures = [q for q in QUAD_LABEL if any(m.beam_integration == q for m in completed)]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=True)
    for q in quadratures:
        scoped = sorted([m for m in completed if m.beam_integration == q], key=lambda m: m.beam_nodes)
        x = [m.beam_nodes for m in scoped]
        color = QUAD_COLOR[q]
        axes[0].plot(x, [m.fiber_stress_rms_rel for m in scoped], marker="o", color=color, label=QUAD_LABEL[q])
        axes[1].plot(x, [m.fiber_strain_rms_rel for m in scoped], marker="o", color=color, label=QUAD_LABEL[q])
        axes[2].plot(x, [m.production_score for m in scoped], marker="o", color=color, label=QUAD_LABEL[q])
    axes[0].set_ylabel("RMS relative error")
    axes[0].set_title(f"{spec['short_label']} stress")
    axes[1].set_ylabel("RMS relative error")
    axes[1].set_title(f"{spec['short_label']} strain")
    axes[2].set_ylabel("Weighted score")
    axes[2].set_title("Production score")
    for ax in axes:
        ax.set_xlabel("Beam nodes N")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    return save_figure(
        fig,
        f"reduced_rc_timoshenko_matrix_{spec['stem']}_selection",
        figures_dir,
        secondary_dir,
    )


def plot_role_overlays(
    role: str,
    metrics: list[ConcreteMetric],
    ref_curve: list[dict[str, float]],
    series_cache: dict[tuple[int, str], list[dict[str, float]]],
    figures_dir: Path,
    secondary_dir: Path | None,
) -> list[str]:
    spec = ROLE_SPECS[role]
    completed = [m for m in metrics if m.status == "completed"]
    quadratures = [q for q in QUAD_LABEL if any(m.beam_integration == q for m in completed)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, q in zip(axes.flat, quadratures, strict=False):
        ax.plot(
            [row["strain"] for row in ref_curve],
            [row["stress"] for row in ref_curve],
            color="black",
            lw=1.7,
            label="OpenSees hi-fi",
        )
        scoped = sorted([m for m in completed if m.beam_integration == q], key=lambda m: m.beam_nodes)
        for m in scoped:
            rows = series_cache.get((m.beam_nodes, m.beam_integration), [])
            color = cmap((m.beam_nodes - 3) / max(10 - 3, 1))
            lw = 1.5 if m.beam_nodes in (4, 10) else 0.9
            alpha = 0.95 if m.beam_nodes in (4, 10) else 0.45
            label = f"N={m.beam_nodes}" if m.beam_nodes in (3, 4, 6, 8, 10) else None
            ax.plot(
                [row["strain"] for row in rows],
                [row["stress"] for row in rows],
                color=color,
                lw=lw,
                alpha=alpha,
                label=label,
            )
        ax.set_xlim(right=0.01)
        ax.set_title(QUAD_LABEL[q])
        ax.set_xlabel(f"{spec['short_label']} strain")
        ax.set_ylabel(f"{spec['short_label']} stress [MPa]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    return save_figure(
        fig,
        f"reduced_rc_timoshenko_matrix_{spec['stem']}_overlays",
        figures_dir,
        secondary_dir,
    )


def best_payload(role: str, metrics: list[ConcreteMetric], ref_info: dict[str, Any]) -> dict[str, Any]:
    completed = [m for m in metrics if m.status == "completed"]
    by_fiber = min(completed, key=lambda m: m.fiber_score if math.isfinite(m.fiber_score) else math.inf)
    by_production = min(completed, key=lambda m: m.production_score if math.isfinite(m.production_score) else math.inf)
    lobatto = [m for m in completed if m.beam_integration == "lobatto"]
    by_lobatto_fiber = min(
        lobatto,
        key=lambda m: m.fiber_score if math.isfinite(m.fiber_score) else math.inf,
    ) if lobatto else None
    by_lobatto_production = min(
        lobatto,
        key=lambda m: m.production_score if math.isfinite(m.production_score) else math.inf,
    ) if lobatto else None
    return {
        "material_role": role,
        "reference_fiber": ref_info,
        "fiber_selection_rule": (
            "Base station; selected material_role; maximum |z|, then maximum |y|, "
            "then lowest fiber_index."
        ),
        "best_fiber": asdict(by_fiber),
        "best_production": asdict(by_production),
        "best_lobatto_fiber": asdict(by_lobatto_fiber) if by_lobatto_fiber else None,
        "best_lobatto_production": asdict(by_lobatto_production) if by_lobatto_production else None,
    }


def main() -> int:
    args = parse_args()
    all_metrics: list[dict[str, Any]] = []
    all_outliers: list[dict[str, Any]] = []
    figures: list[str] = []
    role_payloads: dict[str, Any] = {}
    for role in ROLE_SPECS:
        metrics, ref_curve, series_cache, outliers, ref_info = compute_role_metrics(args, role)
        all_metrics.extend(asdict(m) for m in metrics)
        all_outliers.extend(outliers)
        figures.extend(plot_role_metrics(role, metrics, args.figures_dir, args.secondary_figures_dir))
        figures.extend(plot_role_overlays(role, metrics, ref_curve, series_cache, args.figures_dir, args.secondary_figures_dir))
        role_payloads[role] = best_payload(role, metrics, ref_info)

    metrics_csv = args.figures_dir / "reduced_rc_timoshenko_matrix_extreme_concrete_selection.csv"
    outlier_csv = args.figures_dir / "reduced_rc_timoshenko_matrix_extreme_concrete_outliers.csv"
    write_csv(metrics_csv, all_metrics)
    write_csv(outlier_csv, all_outliers)
    if args.secondary_figures_dir:
        write_csv(
            args.secondary_figures_dir / "reduced_rc_timoshenko_matrix_extreme_concrete_selection.csv",
            all_metrics,
        )
        write_csv(
            args.secondary_figures_dir / "reduced_rc_timoshenko_matrix_extreme_concrete_outliers.csv",
            all_outliers,
        )
    payload = {
        "matrix_dir": str(args.matrix_dir),
        "metrics_csv": str(metrics_csv),
        "outlier_csv": str(outlier_csv),
        "outlier_filter": {
            "contract": "Extreme concrete selection excludes steps whose base-section axial force violates the imposed axial compression.",
            "expected_abs_axial_MN_default": DEFAULT_EXPECTED_AXIAL_MN,
            "axial_tolerance_MN": AXIAL_CONTRACT_TOLERANCE_MN,
            "outlier_count": len(all_outliers),
        },
        "figures": figures,
        "roles": role_payloads,
    }
    json_path = args.figures_dir / "reduced_rc_timoshenko_matrix_extreme_concrete_selection.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.secondary_figures_dir:
        (
            args.secondary_figures_dir / "reduced_rc_timoshenko_matrix_extreme_concrete_selection.json"
        ).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
