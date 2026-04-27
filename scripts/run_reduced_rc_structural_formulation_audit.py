#!/usr/bin/env python3
"""
Summarize the reduced RC-column structural formulation audit.

This script does not rerun the full benchmark matrix. It freezes the current
five structural parity bundles:

  - elasticized + ElasticTimoshenkoBeam
  - elasticized + dispBeamColumn
  - elasticized + forceBeamColumn
  - nonlinear  + dispBeamColumn
  - nonlinear  + forceBeamColumn

and emits one compact machine-readable summary plus two figures:

  1. error audit across global/local observables;
  2. reported wall-time comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

BLUE = "#0b5fa5"
ORANGE = "#d97706"
GREEN = "#2f855a"
RED = "#c53030"
PURPLE = "#6b46c1"


@dataclass(frozen=True)
class AuditCase:
    key: str
    label: str
    material_mode: str
    beam_element_family: str
    bundle_name: str


CASES = (
    AuditCase(
        "elasticized_timoshenko",
        "Elasticized timo",
        "elasticized",
        "elastic-timoshenko",
        "reboot_external_benchmark_structural_elasticized_formulation_audit_timoshenko",
    ),
    AuditCase(
        "elasticized_disp",
        "Elasticized disp",
        "elasticized",
        "disp",
        "reboot_external_benchmark_structural_elasticized_formulation_audit_disp",
    ),
    AuditCase(
        "elasticized_force",
        "Elasticized force",
        "elasticized",
        "force",
        "reboot_external_benchmark_structural_elasticized_formulation_audit_force",
    ),
    AuditCase(
        "nonlinear_disp",
        "Nonlinear disp",
        "nonlinear",
        "disp",
        "reboot_external_benchmark_structural_nonlinear_formulation_audit_disp",
    ),
    AuditCase(
        "nonlinear_force",
        "Nonlinear force",
        "nonlinear",
        "force",
        "reboot_external_benchmark_structural_nonlinear_formulation_audit_force",
    ),
)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Freeze the structural formulation audit for the reduced RC column."
    )
    parser.add_argument(
        "--bundles-root",
        type=Path,
        default=repo_root / "data" / "output" / "cyclic_validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_formulation_audit_summary",
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


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric(payload: dict[str, object], *keys: str, default: float = math.nan) -> float:
    node: object = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    try:
        value = float(node)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def relative_error(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(rhs), 1.0e-12)


def select_peak_positive_step(hysteresis_rows: list[dict[str, object]]) -> int:
    if not hysteresis_rows:
        return 0
    peak = max(
        hysteresis_rows,
        key=lambda row: (float(row["drift_m"]), -int(row["step"])),
    )
    return int(peak["step"])


def profile_rows_for_step(
    rows: list[dict[str, object]],
    step: int,
) -> list[dict[str, object]]:
    return sorted(
        (row for row in rows if int(row["step"]) == step),
        key=lambda row: int(row["section_gp"]),
    )


def field_profile(rows: list[dict[str, object]], field: str) -> list[float]:
    return [float(row[field]) for row in rows]


def max_profile_relative_error(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
    field: str,
) -> float:
    errors = [
        relative_error(float(lhs[field]), float(rhs[field]))
        for lhs, rhs in zip(lhs_rows, rhs_rows)
    ]
    return max(errors) if errors else math.nan


def profile_spread(values: list[float]) -> float:
    return max(values) - min(values) if values else math.nan


def save(fig: plt.Figure, figures_dirs: list[Path], stem: str) -> None:
    for out_dir in figures_dirs:
        ensure_dir(out_dir)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def collect_case(bundle_root: Path, case: AuditCase) -> dict[str, object]:
    bundle_dir = bundle_root / case.bundle_name
    summary = read_json(bundle_dir / "benchmark_summary.json")
    manifest_falln = summary["fall_n"]["manifest"]
    manifest_ops = summary["opensees"]["manifest"]
    comparison = summary["comparison"]
    falln_hysteresis = read_csv_rows(bundle_dir / "fall_n" / "comparison_hysteresis.csv")
    falln_sections = read_csv_rows(bundle_dir / "fall_n" / "section_response.csv")
    opensees_sections = read_csv_rows(bundle_dir / "opensees" / "section_response.csv")
    peak_positive_step = select_peak_positive_step(falln_hysteresis)
    falln_peak_profile = profile_rows_for_step(falln_sections, peak_positive_step)
    opensees_peak_profile = profile_rows_for_step(opensees_sections, peak_positive_step)
    falln_beam_theory_note = (
        "The current reduced structural benchmark uses a mixed-interpolation Timoshenko beam with explicit shear strains and direct global assembly; no element-level static condensation is active on this slice."
    )
    opensees_beam_theory_note = (
        "OpenSees uses dispBeamColumn or forceBeamColumn with a fiber section plus an uncoupled SectionAggregator(Vy,Vz); this is a close structural comparator, but not the explicit ElasticTimoshenkoBeam family."
    )

    return {
        "key": case.key,
        "label": case.label,
        "material_mode": case.material_mode,
        "beam_element_family": case.beam_element_family,
        "bundle_name": case.bundle_name,
        "falln_element_formulation": manifest_falln["element_formulation"],
        "falln_element_assembly_policy": manifest_falln["element_assembly_policy"],
        "falln_beam_theory_note": manifest_falln.get(
            "beam_theory_note", falln_beam_theory_note
        )
        or falln_beam_theory_note,
        "opensees_beam_formulation_note": manifest_ops.get("beam_element_family", ""),
        "opensees_beam_note": manifest_ops["equivalence_scope"]["beam_formulation_note"],
        "opensees_beam_theory_note": manifest_ops["equivalence_scope"].get(
            "beam_theory_note", opensees_beam_theory_note
        )
        or opensees_beam_theory_note,
        "global_base_shear_error": metric(
            comparison, "hysteresis", "max_rel_base_shear_error"
        ),
        "global_base_moment_error": metric(
            comparison, "moment_curvature_base", "max_rel_moment_error"
        ),
        "station_section_moment_error": metric(
            comparison, "section_response_moment", "max_rel_moment_error"
        ),
        "station_section_curvature_error": metric(
            comparison, "section_response_curvature", "max_rel_curvature_error"
        ),
        "station_section_axial_force_error": metric(
            comparison, "section_response_axial_force", "max_rel_axial_force_error"
        ),
        "station_section_tangent_error": metric(
            comparison, "section_response_tangent", "max_rel_tangent_error"
        ),
        "station_raw_k00_error": metric(
            comparison, "section_response_raw_k00", "max_rel_raw_k00_error"
        ),
        "station_raw_k0y_error": metric(
            comparison, "section_response_raw_k0y", "max_rel_raw_k0y_error"
        ),
        "control_tip_drift_error": metric(
            comparison, "control_state_tip_drift", "max_rel_tip_drift_error"
        ),
        "control_tip_axial_displacement_error": metric(
            comparison,
            "control_state_tip_axial_displacement",
            "max_rel_tip_axial_displacement_error",
        ),
        "control_base_axial_reaction_error": metric(
            comparison,
            "control_state_base_axial_reaction",
            "max_rel_base_axial_reaction_error",
        ),
        "peak_positive_step": peak_positive_step,
        "peak_positive_drift_m": max(
            (float(row["drift_m"]) for row in falln_hysteresis),
            default=math.nan,
        ),
        "peak_positive_curvature_profile_error": max_profile_relative_error(
            falln_peak_profile, opensees_peak_profile, "curvature_y"
        ),
        "peak_positive_moment_profile_error": max_profile_relative_error(
            falln_peak_profile, opensees_peak_profile, "moment_y_MNm"
        ),
        "peak_positive_axial_force_profile_error": max_profile_relative_error(
            falln_peak_profile, opensees_peak_profile, "axial_force_MN"
        ),
        "peak_positive_falln_axial_force_spread_mn": profile_spread(
            field_profile(falln_peak_profile, "axial_force_MN")
        ),
        "peak_positive_opensees_axial_force_spread_mn": profile_spread(
            field_profile(opensees_peak_profile, "axial_force_MN")
        ),
        "peak_positive_profile": {
            "xi": field_profile(falln_peak_profile, "xi"),
            "falln_curvature_y": field_profile(falln_peak_profile, "curvature_y"),
            "opensees_curvature_y": field_profile(opensees_peak_profile, "curvature_y"),
            "falln_axial_force_mn": field_profile(falln_peak_profile, "axial_force_MN"),
            "opensees_axial_force_mn": field_profile(opensees_peak_profile, "axial_force_MN"),
            "falln_moment_y_mnm": field_profile(falln_peak_profile, "moment_y_MNm"),
            "opensees_moment_y_mnm": field_profile(opensees_peak_profile, "moment_y_MNm"),
        },
        "falln_total_wall_seconds": metric(
            summary, "fall_n", "manifest", "timing", "total_wall_seconds"
        ),
        "opensees_total_wall_seconds": metric(
            summary, "opensees", "manifest", "timing", "total_wall_seconds"
        ),
        "falln_process_wall_seconds": metric(
            summary, "fall_n", "process_wall_seconds"
        ),
        "opensees_process_wall_seconds": metric(
            summary, "opensees", "process_wall_seconds"
        ),
    }


def make_error_plot(rows: list[dict[str, object]], figures_dirs: list[Path]) -> None:
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.16
    metrics = (
        ("global_base_shear_error", "Base shear", BLUE),
        ("station_section_axial_force_error", "Section axial", GREEN),
        ("station_section_tangent_error", "Section tangent", ORANGE),
        ("control_tip_axial_displacement_error", "Tip axial disp.", RED),
        ("control_base_axial_reaction_error", "Base axial reaction", PURPLE),
    )

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for idx, (field, label, color) in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2.0) * width
        ax.bar(
            [value + offset for value in x],
            [row[field] for row in rows],
            width=width,
            color=color,
            alpha=0.9,
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_yscale("log")
    ax.set_ylabel("Max relative error [-]")
    ax.set_title("Structural formulation audit\nerror surface by comparator slice")
    ax.legend(ncol=2, fontsize=8)
    save(fig, figures_dirs, "reduced_rc_structural_formulation_audit_errors")


def make_timing_plot(rows: list[dict[str, object]], figures_dirs: list[Path]) -> None:
    labels = [row["label"] for row in rows]
    x = list(range(len(rows)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(
        [value - width / 2.0 for value in x],
        [row["falln_total_wall_seconds"] for row in rows],
        width=width,
        color=BLUE,
        alpha=0.9,
        label="fall_n",
    )
    ax.bar(
        [value + width / 2.0 for value in x],
        [row["opensees_total_wall_seconds"] for row in rows],
        width=width,
        color=ORANGE,
        alpha=0.9,
        label="OpenSeesPy",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Reported total wall time [s]")
    ax.set_title("Structural formulation audit\ntiming surface by comparator slice")
    ax.legend()
    save(fig, figures_dirs, "reduced_rc_structural_formulation_audit_timing")


def make_peak_profile_plot(rows: list[dict[str, object]], figures_dirs: list[Path]) -> None:
    structural_rows = [
        row for row in rows if row["key"] in ("nonlinear_disp", "nonlinear_force")
    ]
    if not structural_rows:
        return

    fig, axes = plt.subplots(
        len(structural_rows),
        3,
        figsize=(10.6, 3.6 * len(structural_rows)),
        sharex=True,
    )
    if len(structural_rows) == 1:
        axes = [axes]

    panel_defs = (
        ("curvature_y", r"$\kappa_y$ [1/m]", "peak_positive_curvature_profile_error"),
        ("axial_force_mn", r"$N$ [MN]", "peak_positive_axial_force_profile_error"),
        ("moment_y_mnm", r"$M_y$ [MN m]", "peak_positive_moment_profile_error"),
    )

    for ax_row, row in zip(axes, structural_rows):
        profile = row["peak_positive_profile"]
        xi = profile["xi"]
        for ax, (field_suffix, ylabel, metric_field) in zip(ax_row, panel_defs):
            ax.plot(
                xi,
                profile[f"falln_{field_suffix}"],
                color=BLUE,
                lw=1.4,
                marker="o",
                label="fall_n",
            )
            ax.plot(
                xi,
                profile[f"opensees_{field_suffix}"],
                color=ORANGE,
                lw=1.2,
                ls="--",
                marker="s",
                label="OpenSeesPy",
            )
            ax.set_ylabel(ylabel)
            ax.set_title(
                f"{row['label']} at step={row['peak_positive_step']}, "
                + rf"$\max \varepsilon={row[metric_field]:.2e}$"
            )

        axial_spread_note = (
            rf"$\Delta N_{{\mathrm{{fall\_n}}}}={row['peak_positive_falln_axial_force_spread_mn']:.2e}$ MN, "
            + rf"$\Delta N_{{\mathrm{{OpenSees}}}}={row['peak_positive_opensees_axial_force_spread_mn']:.2e}$ MN"
        )
        ax_row[1].text(
            0.03,
            0.04,
            axial_spread_note,
            transform=ax_row[1].transAxes,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85},
        )
        ax_row[0].legend(loc="best")

    for ax in axes[-1]:
        ax.set_xlabel(r"Station coordinate $\xi$")

    fig.suptitle(
        "Peak-positive station-profile audit\n"
        "forceBeamColumn restores axial equilibrium strongly, but the distributed curvature profile still differs from fall_n",
        y=0.995,
    )
    save(fig, figures_dirs, "reduced_rc_structural_formulation_audit_peak_profile")


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    figures_dirs = [args.figures_dir]
    if args.secondary_figures_dir:
        figures_dirs.append(args.secondary_figures_dir)

    rows = [collect_case(args.bundles_root, case) for case in CASES]
    rows_by_key = {row["key"]: row for row in rows}

    elasticized_timoshenko = rows_by_key["elasticized_timoshenko"]
    nonlinear_disp = rows_by_key["nonlinear_disp"]
    nonlinear_force = rows_by_key["nonlinear_force"]
    elasticized_disp = rows_by_key["elasticized_disp"]
    elasticized_force = rows_by_key["elasticized_force"]

    findings = [
        "The current fall_n structural slice is a mixed-interpolation Timoshenko beam with direct global assembly; no element-level static condensation is active in the reduced RC-column benchmark.",
        "The OpenSees structural comparators are beam-column elements (dispBeamColumn/forceBeamColumn) plus a SectionAggregator(Vy,Vz), not the explicit ElasticTimoshenkoBeam family.",
        "OpenSees does provide an ElasticTimoshenkoBeam element for elastic shear-flexible controls, but the current nonlinear RC benchmark still has to pass through dispBeamColumn/forceBeamColumn because those are the documented nonlinear beam-column routes tied to fiber sections and beam integration.",
        "A stiffness-equivalent ElasticTimoshenkoBeam control is now frozen as an auxiliary elastic parity slice. It is useful for beam-theory and runtime comparison, but it does not replace the nonlinear fiber-section comparators.",
        "Elasticized parity closes tightly for all three OpenSees control slices, and the stiffness-equivalent ElasticTimoshenkoBeam control now closes essentially to machine precision in section tangents and raw K00. That makes beam-axis orientation, fiber cloud, station layout, and basic axial-load enforcement even less credible as primary suspects.",
        "Under nonlinear materials, forceBeamColumn improves station-wise axial-force consistency dramatically but does not close the global hysteresis or the section tangent mismatch.",
        "For the declared displacement-based parity anchor, station-wise section force/tangent traces must now be read as family-aware diagnostics rather than unconditional acceptance gates, because dispBeamColumn is weak-equilibrium at station level.",
        "At the peak positive drift of the 12.5 mm nonlinear benchmark, dispBeamColumn shows an OpenSees axial-force spread of about 8.80e-2 MN across the three section stations while fall_n remains effectively constant at 0.02 MN; forceBeamColumn collapses the OpenSees axial-force spread back to machine precision, but the intermediate-station curvature profile still differs materially from fall_n.",
        "The remaining structural gap is therefore localized mainly in nonlinear preload plus reversal response, consistent with the problematic extremal cover-concrete fiber tracked at step 8.",
    ]

    summary = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_structural_formulation_audit",
        "cases": rows,
        "derived_findings": {
            "elasticized_timoshenko_global_base_shear_error": elasticized_timoshenko[
                "global_base_shear_error"
            ],
            "elasticized_timoshenko_section_tangent_error": elasticized_timoshenko[
                "station_section_tangent_error"
            ],
            "elasticized_disp_global_base_shear_error": elasticized_disp[
                "global_base_shear_error"
            ],
            "elasticized_force_global_base_shear_error": elasticized_force[
                "global_base_shear_error"
            ],
            "nonlinear_disp_global_base_shear_error": nonlinear_disp[
                "global_base_shear_error"
            ],
            "nonlinear_force_global_base_shear_error": nonlinear_force[
                "global_base_shear_error"
            ],
            "nonlinear_axial_force_improvement_factor_force_over_disp": nonlinear_disp[
                "station_section_axial_force_error"
            ]
            / max(nonlinear_force["station_section_axial_force_error"], 1.0e-18),
            "nonlinear_force_minus_disp_global_base_shear_error": nonlinear_force[
                "global_base_shear_error"
            ]
            - nonlinear_disp["global_base_shear_error"],
            "nonlinear_force_minus_disp_section_tangent_error": nonlinear_force[
                "station_section_tangent_error"
            ]
            - nonlinear_disp["station_section_tangent_error"],
            "nonlinear_disp_peak_positive_curvature_profile_error": nonlinear_disp[
                "peak_positive_curvature_profile_error"
            ],
            "nonlinear_force_peak_positive_curvature_profile_error": nonlinear_force[
                "peak_positive_curvature_profile_error"
            ],
            "nonlinear_disp_peak_positive_opensees_axial_force_spread_mn": nonlinear_disp[
                "peak_positive_opensees_axial_force_spread_mn"
            ],
            "nonlinear_force_peak_positive_opensees_axial_force_spread_mn": nonlinear_force[
                "peak_positive_opensees_axial_force_spread_mn"
            ],
            "findings": findings,
        },
        "artifacts": {
            "summary_csv": str(args.output_dir / "structural_formulation_audit_summary.csv"),
            "summary_json": str(args.output_dir / "structural_formulation_audit_summary.json"),
            "figure_errors": str(
                args.figures_dir / "reduced_rc_structural_formulation_audit_errors.png"
            ),
            "figure_timing": str(
                args.figures_dir / "reduced_rc_structural_formulation_audit_timing.png"
            ),
            "figure_peak_profile": str(
                args.figures_dir / "reduced_rc_structural_formulation_audit_peak_profile.png"
            ),
        },
    }

    write_csv(
        args.output_dir / "structural_formulation_audit_summary.csv",
        (
            "key",
            "label",
            "material_mode",
            "beam_element_family",
            "global_base_shear_error",
            "global_base_moment_error",
            "station_section_moment_error",
            "station_section_curvature_error",
            "station_section_axial_force_error",
            "station_section_tangent_error",
            "station_raw_k00_error",
            "station_raw_k0y_error",
            "control_tip_drift_error",
            "control_tip_axial_displacement_error",
            "control_base_axial_reaction_error",
            "peak_positive_step",
            "peak_positive_drift_m",
            "peak_positive_curvature_profile_error",
            "peak_positive_moment_profile_error",
            "peak_positive_axial_force_profile_error",
            "peak_positive_falln_axial_force_spread_mn",
            "peak_positive_opensees_axial_force_spread_mn",
            "falln_total_wall_seconds",
            "opensees_total_wall_seconds",
        ),
        rows,
    )
    write_json(
        args.output_dir / "structural_formulation_audit_summary.json", summary
    )
    make_error_plot(rows, figures_dirs)
    make_timing_plot(rows, figures_dirs)
    make_peak_profile_plot(rows, figures_dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
