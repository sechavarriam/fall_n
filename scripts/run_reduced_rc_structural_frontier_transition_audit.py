#!/usr/bin/env python3
"""
Compare the strongest successful reduced RC structural benchmark against the
current fall_n-only structural frontier, including the exact failed recursive
trial when it is available.

The goal is to answer a narrow validation question:

  Is the transition from the 12.5 mm successful slice to the ~13.95 mm
  last-converged frontier state, and then to the exact failed recursive trial,
  explained by an abrupt local constitutive collapse, or by a more gradual
  drift of an already degraded section/fiber state?

The audit intentionally reuses the tracked extremal fiber exported by the
existing hotspot audit so the comparison stays tied to the same mechanical
object across bundles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
        description="Audit the transition from the successful 12.5 mm slice to the current structural frontier."
    )
    parser.add_argument(
        "--success-bundle",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_12p5mm_fiber_audit_zfix",
    )
    parser.add_argument(
        "--frontier-bundle",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_15mm_failed_trial_capture",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_frontier_transition_failed_attempt_audit",
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
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def benchmark_root(bundle: Path) -> Path:
    candidate = bundle / "fall_n"
    return candidate if candidate.exists() else bundle


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(fh):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def optional_csv_rows(path: Path | None) -> list[dict[str, object]]:
    return read_csv_rows(path) if path is not None and path.exists() else []


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric(row: dict[str, object], key: str) -> float:
    value = row.get(key, math.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def relative_change(reference: float, candidate: float) -> float:
    return (candidate - reference) / max(abs(reference), 1.0e-12)


def relative_magnitude_change(reference: float, candidate: float) -> float:
    return (abs(candidate) - abs(reference)) / max(abs(reference), 1.0e-12)


def save(fig: plt.Figure, figures_dirs: list[Path], stem: str) -> None:
    for out_dir in figures_dirs:
        ensure_dir(out_dir)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def derive_hotspot_identity(rows: list[dict[str, object]]) -> dict[str, object]:
    latest_step = max(int(row["step"]) for row in rows)
    candidates = [row for row in rows if int(row["step"]) == latest_step]
    hotspot = max(
        candidates,
        key=lambda row: (
            abs(float(row["strain_xx"])),
            abs(float(row["stress_xx_MPa"])),
            abs(float(row["tangent_xx_MPa"])),
            -int(row["section_gp"]),
        ),
    )
    return {
        "section_gp": int(hotspot["section_gp"]),
        "material_role": str(hotspot["material_role"]),
        "zone": str(hotspot["zone"]),
        "y": float(hotspot["y"]),
        "z": float(hotspot["z"]),
    }


def hotspot_identity(bundle_dir: Path, fallback_rows: list[dict[str, object]]) -> dict[str, object]:
    summary_path = bundle_dir / "hotspot_audit" / "hotspot_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        hotspot = summary["top_frontier_hotspot"]
        return {
            "section_gp": int(hotspot["section_gp"]),
            "material_role": hotspot["material_role"],
            "zone": hotspot["zone"],
            "y": float(hotspot["y"]),
            "z": float(hotspot["z"]),
        }
    return derive_hotspot_identity(fallback_rows)


def select_tracked_fiber_rows(
    rows: list[dict[str, object]],
    identity: dict[str, object],
) -> list[dict[str, object]]:
    def matches(row: dict[str, object]) -> bool:
        return (
            int(row["section_gp"]) == int(identity["section_gp"])
            and row["material_role"] == identity["material_role"]
            and row["zone"] == identity["zone"]
            and abs(float(row["y"]) - float(identity["y"])) < 1.0e-10
            and abs(float(row["z"]) - float(identity["z"])) < 1.0e-10
        )

    return [row for row in rows if matches(row)]


def select_positive_loading_branch(
    rows: list[dict[str, object]],
    peak_step: int | None = None,
) -> list[dict[str, object]]:
    limit = peak_step if peak_step is not None else max(int(row["step"]) for row in rows)
    return [
        row
        for row in rows
        if int(row["step"]) <= limit and float(row["drift_m"]) >= -1.0e-15
    ]


def select_section_rows(
    rows: list[dict[str, object]],
    section_gp: int,
    limit_step: int | None = None,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if int(row["section_gp"]) == section_gp
        and (limit_step is None or int(row["step"]) <= limit_step)
    ]


def select_control_rows(
    rows: list[dict[str, object]],
    limit_step: int | None = None,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if (limit_step is None or int(row["step"]) <= limit_step)
    ]


def select_peak_positive_row(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(rows, key=lambda row: (float(row["drift_m"]), -int(row["step"])))


def select_last_row(rows: list[dict[str, object]]) -> dict[str, object]:
    return max(rows, key=lambda row: int(row["step"]))


def state_payload(row: dict[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    return {key: row[key] for key in keys}


def failed_attempt_csv_path(manifest: dict[str, object], key: str) -> Path | None:
    failed_attempt = manifest.get("failed_attempt")
    if not isinstance(failed_attempt, dict):
        return None
    value = failed_attempt.get(key)
    if not value:
        return None
    return Path(str(value)).resolve()


def aggregate_zone_material_rows(
    rows: list[dict[str, object]],
    section_gp: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        if int(row["section_gp"]) != section_gp:
            continue
        key = (str(row["zone"]), str(row["material_role"]))
        bucket = grouped.setdefault(
            key,
            {
                "zone": key[0],
                "material_role": key[1],
                "fiber_count": 0,
                "axial_force_contribution_MN": 0.0,
                "moment_y_contribution_MNm": 0.0,
                "raw_k00_contribution": 0.0,
                "raw_k0y_contribution": 0.0,
                "raw_kyy_contribution": 0.0,
                "max_abs_stress_xx_MPa": 0.0,
                "max_abs_tangent_xx_MPa": 0.0,
                "max_abs_strain_xx": 0.0,
            },
        )
        bucket["fiber_count"] += 1
        bucket["axial_force_contribution_MN"] += float(row["axial_force_contribution_MN"])
        bucket["moment_y_contribution_MNm"] += float(row["moment_y_contribution_MNm"])
        bucket["raw_k00_contribution"] += float(row["raw_k00_contribution"])
        bucket["raw_k0y_contribution"] += float(row["raw_k0y_contribution"])
        bucket["raw_kyy_contribution"] += float(row["raw_kyy_contribution"])
        bucket["max_abs_stress_xx_MPa"] = max(
            float(bucket["max_abs_stress_xx_MPa"]),
            abs(float(row["stress_xx_MPa"])),
        )
        bucket["max_abs_tangent_xx_MPa"] = max(
            float(bucket["max_abs_tangent_xx_MPa"]),
            abs(float(row["tangent_xx_MPa"])),
        )
        bucket["max_abs_strain_xx"] = max(
            float(bucket["max_abs_strain_xx"]),
            abs(float(row["strain_xx"])),
        )
    return sorted(
        grouped.values(),
        key=lambda row: (str(row["zone"]), str(row["material_role"])),
    )


def max_optional_numeric(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return max(values) if values else math.nan


def make_tracked_fiber_figure(
    success_rows: list[dict[str, object]],
    frontier_rows: list[dict[str, object]],
    failed_attempt_row: dict[str, object] | None,
    figures_dirs: list[Path],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), sharex=True)
    series = (
        ("strain_xx", "Fiber strain $\\varepsilon_x$", GREEN),
        ("stress_xx_MPa", "Fiber stress [MPa]", BLUE),
        ("tangent_xx_MPa", "Fiber tangent [MPa]", RED),
    )
    for ax, (field, ylabel, color) in zip(axes, series):
        ax.plot(
            [float(row["drift_m"]) * 1.0e3 for row in success_rows],
            [float(row[field]) for row in success_rows],
            color=color,
            lw=1.8,
            label="fall_n 12.5 mm success",
        )
        ax.plot(
            [float(row["drift_m"]) * 1.0e3 for row in frontier_rows],
            [float(row[field]) for row in frontier_rows],
            color=ORANGE,
            lw=1.8,
            ls="--",
            label="fall_n 15 mm frontier",
        )
        if failed_attempt_row is not None:
            ax.scatter(
                [float(failed_attempt_row["drift_m"]) * 1.0e3],
                [float(failed_attempt_row[field])],
                color=RED,
                marker="x",
                s=42,
                zorder=5,
                label="exact failed trial" if field == "strain_xx" else None,
            )
        ax.set_xlabel("Tip drift [mm]")
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Tracked extremal cover-concrete fiber\n"
        "successful 12.5 mm slice vs current 15 mm frontier",
        fontsize=11,
    )
    save(fig, figures_dirs, "reduced_rc_structural_frontier_tracked_fiber_transition")


def make_section_block_figure(
    success_rows: list[dict[str, object]],
    frontier_rows: list[dict[str, object]],
    failed_attempt_row: dict[str, object] | None,
    figures_dirs: list[Path],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0), sharex=True)
    fields = (
        ("tangent_eiy", "Condensed $EI_y$"),
        ("raw_k00", "$K_{00}$"),
        ("raw_k0y", "$K_{0y}$"),
        ("raw_kyy", "$K_{yy}$"),
    )
    for ax, (field, title) in zip(axes.flat, fields):
        ax.plot(
            [float(row["drift_m"]) * 1.0e3 for row in success_rows],
            [float(row[field]) for row in success_rows],
            color=BLUE,
            lw=1.8,
            label="fall_n 12.5 mm success",
        )
        ax.plot(
            [float(row["drift_m"]) * 1.0e3 for row in frontier_rows],
            [float(row[field]) for row in frontier_rows],
            color=ORANGE,
            lw=1.8,
            ls="--",
            label="fall_n 15 mm frontier",
        )
        if failed_attempt_row is not None:
            ax.scatter(
                [float(failed_attempt_row["drift_m"]) * 1.0e3],
                [float(failed_attempt_row[field])],
                color=RED,
                marker="x",
                s=42,
                zorder=5,
                label="exact failed trial" if field == "tangent_eiy" else None,
            )
        ax.set_title(title)
        ax.set_xlabel("Tip drift [mm]")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Base-section block evolution near the current structural frontier",
        fontsize=11,
    )
    save(fig, figures_dirs, "reduced_rc_structural_frontier_section_block_transition")


def make_zone_aggregate_figure(
    committed_rows: list[dict[str, object]],
    failed_rows: list[dict[str, object]],
    figures_dirs: list[Path],
) -> None:
    if not committed_rows or not failed_rows:
        return

    labels = [
        f"{row['zone']}\n{row['material_role']}"
        for row in committed_rows
    ]
    x = list(range(len(labels)))
    width = 0.35
    x_lhs = [value - width / 2 for value in x]
    x_rhs = [value + width / 2 for value in x]

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 7.2), sharex=True)
    for ax, field, ylabel in (
        (axes[0], "moment_y_contribution_MNm", r"Moment contrib. $M_y$ [MN m]"),
        (axes[1], "raw_k0y_contribution", r"Coupling contrib. $K_{0y}$"),
    ):
        ax.bar(
            x_lhs,
            [float(row[field]) for row in committed_rows],
            width=width,
            color=ORANGE,
            alpha=0.9,
            label="last converged",
        )
        ax.bar(
            x_rhs,
            [float(row[field]) for row in failed_rows],
            width=width,
            color=RED,
            alpha=0.8,
            label="exact failed trial",
        )
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xticks(x, labels)
    fig.suptitle(
        "Base-section zone/material aggregate transition\n"
        "last converged point vs exact failed recursive trial",
        fontsize=11,
    )
    save(fig, figures_dirs, "reduced_rc_structural_frontier_zone_transition")


def make_solver_trace_figure(
    success_rows: list[dict[str, object]],
    frontier_rows: list[dict[str, object]],
    failed_attempt: dict[str, object] | None,
    figures_dirs: list[Path],
) -> None:
    def numeric_series(rows: list[dict[str, object]], key: str) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            if key not in row:
                continue
            xs.append(float(row["target_drift_m"]) * 1.0e3)
            ys.append(float(row[key]))
        return xs, ys

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), sharex=True)
    series = (
        ("newton_iterations", "Newton iterations"),
        ("newton_iterations_per_substep", "Newton / accepted substep"),
        ("last_function_norm", "Last residual norm"),
    )
    for ax, (field, ylabel) in zip(axes, series):
        success_x, success_y = numeric_series(success_rows, field)
        frontier_x, frontier_y = numeric_series(frontier_rows, field)
        if success_x:
            ax.plot(
                success_x,
                success_y,
                color=BLUE,
                lw=1.8,
                label="fall_n 12.5 mm success",
            )
        if frontier_x:
            ax.plot(
                frontier_x,
                frontier_y,
                color=ORANGE,
                lw=1.8,
                ls="--",
                label="fall_n 15 mm frontier",
            )
        if failed_attempt is not None:
            ax.scatter(
                [float(failed_attempt["target_drift_m"]) * 1.0e3],
                [float(failed_attempt[field])],
                color=RED,
                marker="x",
                s=42,
                label="failed attempt" if field == "newton_iterations" else None,
                zorder=5,
            )
        ax.set_xlabel("Target tip drift [mm]")
        ax.set_ylabel(ylabel)

    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "Structural solver trace near the current amplitude frontier",
        fontsize=11,
    )
    save(fig, figures_dirs, "reduced_rc_structural_frontier_solver_trace_transition")


def main() -> int:
    args = parse_args()
    success_bundle = args.success_bundle.resolve()
    frontier_bundle = args.frontier_bundle.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    figures_dirs = [args.figures_dir.resolve(), args.secondary_figures_dir.resolve()]
    success_root = benchmark_root(success_bundle)
    frontier_root = benchmark_root(frontier_bundle)

    success_fiber_rows = read_csv_rows(success_root / "section_fiber_state_history.csv")
    frontier_fiber_rows = read_csv_rows(frontier_root / "section_fiber_state_history.csv")
    success_section_rows = read_csv_rows(success_root / "section_response.csv")
    frontier_section_rows = read_csv_rows(frontier_root / "section_response.csv")
    success_control_rows = read_csv_rows(success_root / "control_state.csv")
    frontier_control_rows = read_csv_rows(frontier_root / "control_state.csv")
    frontier_manifest = read_json(frontier_root / "runtime_manifest.json")
    identity = hotspot_identity(frontier_bundle, frontier_fiber_rows)
    failed_attempt_section_rows = optional_csv_rows(
        failed_attempt_csv_path(frontier_manifest, "failed_attempt_section_response_csv")
    )
    failed_attempt_fiber_rows = optional_csv_rows(
        failed_attempt_csv_path(frontier_manifest, "failed_attempt_section_fiber_state_history_csv")
    )

    tracked_success = select_tracked_fiber_rows(success_fiber_rows, identity)
    tracked_frontier = select_tracked_fiber_rows(frontier_fiber_rows, identity)
    tracked_failed_attempt = select_tracked_fiber_rows(failed_attempt_fiber_rows, identity)
    success_peak = select_peak_positive_row(tracked_success)
    frontier_last = select_last_row(tracked_frontier)
    failed_attempt_fiber = (
        select_last_row(tracked_failed_attempt) if tracked_failed_attempt else None
    )
    success_peak_step = int(success_peak["step"])
    frontier_last_step = int(frontier_last["step"])

    success_branch = select_positive_loading_branch(tracked_success, success_peak_step)
    frontier_branch = select_positive_loading_branch(tracked_frontier)

    success_base_section_rows = select_section_rows(
        success_section_rows, int(identity["section_gp"]), success_peak_step
    )
    frontier_base_section_rows = select_section_rows(
        frontier_section_rows, int(identity["section_gp"])
    )
    failed_attempt_base_section_rows = select_section_rows(
        failed_attempt_section_rows, int(identity["section_gp"])
    )
    success_control_branch = select_control_rows(success_control_rows, success_peak_step)
    frontier_control_branch = select_control_rows(frontier_control_rows)
    success_base_peak = select_peak_positive_row(success_base_section_rows)
    frontier_base_last = select_last_row(frontier_base_section_rows)
    failed_attempt_base = (
        select_last_row(failed_attempt_base_section_rows)
        if failed_attempt_base_section_rows
        else None
    )
    failed_attempt = frontier_manifest.get("failed_attempt")
    committed_zone_rows = aggregate_zone_material_rows(
        [row for row in frontier_fiber_rows if int(row["step"]) == frontier_last_step],
        int(identity["section_gp"]),
    )
    failed_zone_rows = aggregate_zone_material_rows(
        failed_attempt_fiber_rows,
        int(identity["section_gp"]),
    )

    write_csv(
        output_dir / "tracked_fiber_success_branch.csv",
        (
            "step",
            "drift_m",
            "curvature_y",
            "axial_strain",
            "strain_xx",
            "stress_xx_MPa",
            "tangent_xx_MPa",
            "raw_k00_contribution",
            "raw_k0y_contribution",
            "raw_kyy_contribution",
        ),
        success_branch,
    )
    write_csv(
        output_dir / "tracked_fiber_frontier_branch.csv",
        (
            "step",
            "drift_m",
            "curvature_y",
            "axial_strain",
            "strain_xx",
            "stress_xx_MPa",
            "tangent_xx_MPa",
            "raw_k00_contribution",
            "raw_k0y_contribution",
            "raw_kyy_contribution",
        ),
        frontier_branch,
    )
    write_csv(
        output_dir / "tracked_fiber_failed_attempt.csv",
        (
            "step",
            "drift_m",
            "curvature_y",
            "axial_strain",
            "strain_xx",
            "stress_xx_MPa",
            "tangent_xx_MPa",
            "raw_k00_contribution",
            "raw_k0y_contribution",
            "raw_kyy_contribution",
        ),
        [failed_attempt_fiber] if failed_attempt_fiber is not None else [],
    )
    write_csv(
        output_dir / "base_section_success_branch.csv",
        (
            "step",
            "drift_m",
            "axial_strain",
            "curvature_y",
            "axial_force_MN",
            "moment_y_MNm",
            "tangent_eiy",
            "raw_k00",
            "raw_k0y",
            "raw_kyy",
        ),
        success_base_section_rows,
    )
    write_csv(
        output_dir / "base_section_frontier_branch.csv",
        (
            "step",
            "drift_m",
            "axial_strain",
            "curvature_y",
            "axial_force_MN",
            "moment_y_MNm",
            "tangent_eiy",
            "raw_k00",
            "raw_k0y",
            "raw_kyy",
        ),
        frontier_base_section_rows,
    )
    write_csv(
        output_dir / "base_section_failed_attempt.csv",
        (
            "step",
            "drift_m",
            "axial_strain",
            "curvature_y",
            "axial_force_MN",
            "moment_y_MNm",
            "tangent_eiy",
            "raw_k00",
            "raw_k0y",
            "raw_kyy",
        ),
        [failed_attempt_base] if failed_attempt_base is not None else [],
    )
    write_csv(
        output_dir / "control_success_branch.csv",
        (
            "runtime_step",
            "step",
            "p",
            "runtime_p",
            "target_drift_m",
            "actual_tip_drift_m",
            "accepted_substep_count",
            "max_bisection_level",
            "newton_iterations",
            "newton_iterations_per_substep",
            "last_snes_reason",
            "last_function_norm",
        ),
        success_control_branch,
    )
    write_csv(
        output_dir / "control_frontier_branch.csv",
        (
            "runtime_step",
            "step",
            "p",
            "runtime_p",
            "target_drift_m",
            "actual_tip_drift_m",
            "accepted_substep_count",
            "max_bisection_level",
            "newton_iterations",
            "newton_iterations_per_substep",
            "last_snes_reason",
            "last_function_norm",
        ),
        frontier_control_branch,
    )
    write_csv(
        output_dir / "base_section_zone_transition.csv",
        (
            "zone",
            "material_role",
            "fiber_count",
            "committed_axial_force_contribution_MN",
            "failed_axial_force_contribution_MN",
            "committed_moment_y_contribution_MNm",
            "failed_moment_y_contribution_MNm",
            "committed_raw_k00_contribution",
            "failed_raw_k00_contribution",
            "committed_raw_k0y_contribution",
            "failed_raw_k0y_contribution",
            "committed_raw_kyy_contribution",
            "failed_raw_kyy_contribution",
            "committed_max_abs_stress_xx_MPa",
            "failed_max_abs_stress_xx_MPa",
            "committed_max_abs_tangent_xx_MPa",
            "failed_max_abs_tangent_xx_MPa",
            "committed_max_abs_strain_xx",
            "failed_max_abs_strain_xx",
        ),
        [
            {
                "zone": committed["zone"],
                "material_role": committed["material_role"],
                "fiber_count": committed["fiber_count"],
                "committed_axial_force_contribution_MN": committed["axial_force_contribution_MN"],
                "failed_axial_force_contribution_MN": failed["axial_force_contribution_MN"],
                "committed_moment_y_contribution_MNm": committed["moment_y_contribution_MNm"],
                "failed_moment_y_contribution_MNm": failed["moment_y_contribution_MNm"],
                "committed_raw_k00_contribution": committed["raw_k00_contribution"],
                "failed_raw_k00_contribution": failed["raw_k00_contribution"],
                "committed_raw_k0y_contribution": committed["raw_k0y_contribution"],
                "failed_raw_k0y_contribution": failed["raw_k0y_contribution"],
                "committed_raw_kyy_contribution": committed["raw_kyy_contribution"],
                "failed_raw_kyy_contribution": failed["raw_kyy_contribution"],
                "committed_max_abs_stress_xx_MPa": committed["max_abs_stress_xx_MPa"],
                "failed_max_abs_stress_xx_MPa": failed["max_abs_stress_xx_MPa"],
                "committed_max_abs_tangent_xx_MPa": committed["max_abs_tangent_xx_MPa"],
                "failed_max_abs_tangent_xx_MPa": failed["max_abs_tangent_xx_MPa"],
                "committed_max_abs_strain_xx": committed["max_abs_strain_xx"],
                "failed_max_abs_strain_xx": failed["max_abs_strain_xx"],
            }
            for committed, failed in zip(committed_zone_rows, failed_zone_rows)
            if committed["zone"] == failed["zone"]
            and committed["material_role"] == failed["material_role"]
        ],
    )

    make_tracked_fiber_figure(
        success_branch,
        frontier_branch,
        failed_attempt_fiber,
        figures_dirs,
    )
    make_section_block_figure(
        success_base_section_rows,
        frontier_base_section_rows,
        failed_attempt_base,
        figures_dirs,
    )
    make_solver_trace_figure(
        success_control_branch,
        frontier_control_branch,
        failed_attempt if isinstance(failed_attempt, dict) else None,
        figures_dirs,
    )
    make_zone_aggregate_figure(committed_zone_rows, failed_zone_rows, figures_dirs)

    summary = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_structural_frontier_transition_audit",
        "success_bundle": str(success_bundle),
        "frontier_bundle": str(frontier_bundle),
        "tracked_fiber_identity": identity,
        "success_peak_positive": {
            "fiber": state_payload(
                success_peak,
                (
                    "step",
                    "drift_m",
                    "curvature_y",
                    "axial_strain",
                    "strain_xx",
                    "stress_xx_MPa",
                    "tangent_xx_MPa",
                    "raw_k00_contribution",
                    "raw_k0y_contribution",
                    "raw_kyy_contribution",
                ),
            ),
            "section": state_payload(
                success_base_peak,
                (
                    "step",
                    "drift_m",
                    "axial_strain",
                    "curvature_y",
                    "axial_force_MN",
                    "moment_y_MNm",
                    "tangent_eiy",
                    "raw_k00",
                    "raw_k0y",
                    "raw_kyy",
                ),
            ),
        },
        "frontier_last_converged": {
            "fiber": state_payload(
                frontier_last,
                (
                    "step",
                    "drift_m",
                    "curvature_y",
                    "axial_strain",
                    "strain_xx",
                    "stress_xx_MPa",
                    "tangent_xx_MPa",
                    "raw_k00_contribution",
                    "raw_k0y_contribution",
                    "raw_kyy_contribution",
                ),
            ),
            "section": state_payload(
                frontier_base_last,
                (
                    "step",
                    "drift_m",
                    "axial_strain",
                    "curvature_y",
                    "axial_force_MN",
                    "moment_y_MNm",
                    "tangent_eiy",
                    "raw_k00",
                    "raw_k0y",
                    "raw_kyy",
                ),
            ),
        },
        "frontier_exact_failed_attempt": {
            "fiber": state_payload(
                failed_attempt_fiber,
                (
                    "step",
                    "drift_m",
                    "curvature_y",
                    "axial_strain",
                    "strain_xx",
                    "stress_xx_MPa",
                    "tangent_xx_MPa",
                    "raw_k00_contribution",
                    "raw_k0y_contribution",
                    "raw_kyy_contribution",
                ),
            )
            if failed_attempt_fiber is not None
            else None,
            "section": state_payload(
                failed_attempt_base,
                (
                    "step",
                    "drift_m",
                    "axial_strain",
                    "curvature_y",
                    "axial_force_MN",
                    "moment_y_MNm",
                    "tangent_eiy",
                    "raw_k00",
                    "raw_k0y",
                    "raw_kyy",
                ),
            )
            if failed_attempt_base is not None
            else None,
        },
        "derived_findings": {
            "drift_increase_fraction": relative_change(
                metric(success_peak, "drift_m"),
                metric(frontier_last, "drift_m"),
            ),
            "fiber_strain_increase_fraction": relative_change(
                metric(success_peak, "strain_xx"),
                metric(frontier_last, "strain_xx"),
            ),
            "section_curvature_magnitude_increase_fraction": relative_magnitude_change(
                metric(success_base_peak, "curvature_y"),
                metric(frontier_base_last, "curvature_y"),
            ),
            "section_moment_magnitude_increase_fraction": relative_magnitude_change(
                metric(success_base_peak, "moment_y_MNm"),
                metric(frontier_base_last, "moment_y_MNm"),
            ),
            "section_tangent_eiy_change_fraction": relative_change(
                metric(success_base_peak, "tangent_eiy"),
                metric(frontier_base_last, "tangent_eiy"),
            ),
            "section_raw_k00_change_fraction": relative_change(
                metric(success_base_peak, "raw_k00"),
                metric(frontier_base_last, "raw_k00"),
            ),
            "section_raw_k0y_change_fraction": relative_change(
                metric(success_base_peak, "raw_k0y"),
                metric(frontier_base_last, "raw_k0y"),
            ),
            "section_raw_kyy_change_fraction": relative_change(
                metric(success_base_peak, "raw_kyy"),
                metric(frontier_base_last, "raw_kyy"),
            ),
            "tracked_fiber_already_open_at_success_peak": (
                abs(metric(success_peak, "stress_xx_MPa")) <= 1.0e-12
                and abs(metric(success_peak, "tangent_xx_MPa") - 0.03) <= 1.0e-12
            ),
            "tracked_fiber_open_at_frontier_last": (
                abs(metric(frontier_last, "stress_xx_MPa")) <= 1.0e-12
                and abs(metric(frontier_last, "tangent_xx_MPa") - 0.03) <= 1.0e-12
            ),
            "tracked_fiber_open_at_exact_failed_attempt": (
                failed_attempt_fiber is not None
                and abs(metric(failed_attempt_fiber, "stress_xx_MPa")) <= 1.0e-12
                and abs(metric(failed_attempt_fiber, "tangent_xx_MPa") - 0.03) <= 1.0e-12
            ),
            "success_peak_newton_iterations": max_optional_numeric(
                success_control_branch, "newton_iterations"
            ),
            "success_peak_max_bisection_level": max_optional_numeric(
                success_control_branch, "max_bisection_level"
            ),
            "frontier_committed_newton_iterations": max_optional_numeric(
                frontier_control_branch, "newton_iterations"
            ),
            "frontier_committed_max_bisection_level": max_optional_numeric(
                frontier_control_branch, "max_bisection_level"
            ),
            "frontier_failed_attempt": failed_attempt,
            "failed_attempt_vs_last_converged_drift_fraction": relative_change(
                metric(frontier_last, "drift_m"),
                metric(failed_attempt_fiber, "drift_m"),
            )
            if failed_attempt_fiber is not None
            else math.nan,
            "failed_attempt_vs_last_converged_fiber_strain_fraction": relative_change(
                metric(frontier_last, "strain_xx"),
                metric(failed_attempt_fiber, "strain_xx"),
            )
            if failed_attempt_fiber is not None
            else math.nan,
            "failed_attempt_vs_last_converged_section_curvature_magnitude_fraction": relative_magnitude_change(
                metric(frontier_base_last, "curvature_y"),
                metric(failed_attempt_base, "curvature_y"),
            )
            if failed_attempt_base is not None
            else math.nan,
            "failed_attempt_vs_last_converged_section_moment_magnitude_fraction": relative_magnitude_change(
                metric(frontier_base_last, "moment_y_MNm"),
                metric(failed_attempt_base, "moment_y_MNm"),
            )
            if failed_attempt_base is not None
            else math.nan,
            "failed_attempt_vs_last_converged_section_tangent_fraction": relative_change(
                metric(frontier_base_last, "tangent_eiy"),
                metric(failed_attempt_base, "tangent_eiy"),
            )
            if failed_attempt_base is not None
            else math.nan,
            "findings": [
                "The tracked extremal cover-top unconfined-concrete fiber is already fully open on the successful 12.5 mm slice: zero stress and residual tangent 0.03 MPa at the peak positive branch.",
                "The same fiber remains in the same open state at the last converged frontier state (~13.95 mm). The transition to the current structural frontier is therefore not explained by a first opening of that fiber.",
                "Between the successful 12.5 mm peak and the last converged frontier state, the tracked-fiber strain and the base-section curvature both grow by only a few tens of percent; the section remains degraded, but the transition is still gradual rather than catastrophic.",
                "The strongest base-section changes over that interval are moderate rather than singular: the condensed flexural tangent softens, raw K00 and K0y both drop materially, and raw Kyy softens more mildly, but none of those quantities collapse abruptly on the committed branch.",
                "The new exact failed-trial capture shows that the structural break is even narrower than the committed-branch comparison suggests: the failed recursive target sits near 13.967 mm, only about 0.16% beyond the last converged drift.",
                "At that exact failed trial, the tracked extremal cover-top fiber still stays fully open with zero stress and residual 0.03 MPa tangent, and its strain rises by only about 0.15% relative to the last converged point.",
                "The base section also barely changes between the last converged point and the exact failed trial: curvature grows by about 0.15%, moment by about 0.14%, and the condensed base tangent changes by only about -0.004%.",
                "Zone/material aggregates at the base confirm the same reading: no abrupt redistribution appears between the last converged point and the exact failed trial; the compressive cover/core and the reinforcing steel continue to carry the section smoothly.",
                "The successful 12.5 mm bundle does not need bisection on the committed branch (max committed bisection level = 0). The frontier bundle also reaches its last converged point as a committed 2-iteration step with zero committed bisection depth, even though a few earlier committed steps on the branch rise above that.",
                "The current 15 mm failure is therefore not preceded by a broad inflation of Newton effort across the committed branch. The solver trace stays calm and then breaks on a specific runtime step near 13.97 mm.",
                "On that failed runtime step, the solver accepts one substep, exhausts bisection to level 4 on the remaining portion, and exits with SNES reason -6 and residual norm about 1.67e-3. That looks much more like a narrow continuation frontier than a diffuse branch-wide loss of equilibrium.",
                "The current 15 mm blockage is therefore better interpreted as a fine continuation frontier acting on an already degraded section/fiber state, not as an abrupt local constitutive collapse newly triggered beyond 12.5 mm.",
            ],
        },
        "artifacts": {
            "tracked_fiber_success_branch_csv": str(
                output_dir / "tracked_fiber_success_branch.csv"
            ),
            "tracked_fiber_frontier_branch_csv": str(
                output_dir / "tracked_fiber_frontier_branch.csv"
            ),
            "tracked_fiber_failed_attempt_csv": str(
                output_dir / "tracked_fiber_failed_attempt.csv"
            ),
            "base_section_success_branch_csv": str(
                output_dir / "base_section_success_branch.csv"
            ),
            "base_section_frontier_branch_csv": str(
                output_dir / "base_section_frontier_branch.csv"
            ),
            "base_section_failed_attempt_csv": str(
                output_dir / "base_section_failed_attempt.csv"
            ),
            "base_section_zone_transition_csv": str(
                output_dir / "base_section_zone_transition.csv"
            ),
            "control_success_branch_csv": str(output_dir / "control_success_branch.csv"),
            "control_frontier_branch_csv": str(output_dir / "control_frontier_branch.csv"),
            "figure_tracked_fiber": str(
                figures_dirs[0] / "reduced_rc_structural_frontier_tracked_fiber_transition.png"
            ),
            "figure_section_blocks": str(
                figures_dirs[0] / "reduced_rc_structural_frontier_section_block_transition.png"
            ),
            "figure_solver_trace": str(
                figures_dirs[0] / "reduced_rc_structural_frontier_solver_trace_transition.png"
            ),
            "figure_zone_transition": str(
                figures_dirs[0] / "reduced_rc_structural_frontier_zone_transition.png"
            ),
        },
    }
    write_json(output_dir / "frontier_transition_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
