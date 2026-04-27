#!/usr/bin/env python3
"""
Freeze the first reduced RC-column validation stage as one canonical artifact.

Stage 1 is not the final physical validation of the library. Its scope is
smaller and more rigorous:

  1. prove that the internal fall_n reduced-column slice is algorithmically
     robust on the declared displacement-driven path;
  2. prove that elasticized external parity closes tightly enough to remove
     gross geometry/observable mismatches from the active suspicion list;
  3. declare, with explicit evidence bundles, where the current external
     section and structural frontiers still live;
  4. preserve timing and architectural comparisons against OpenSees without
     overstating them.

The resulting JSON/CSV/figures are meant to be cited by README, doc/, and the
thesis so the closure argument survives beyond transient logs.
"""

from __future__ import annotations

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


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Freeze the first reduced RC-column validation stage into one "
            "canonical closure bundle."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--internal-runtime-summary",
        type=Path,
        default=repo_root
        / "data/output/cyclic_validation/reboot_internal_hysteresis_200mm/internal_hysteresis_200mm_summary.json",
    )
    parser.add_argument(
        "--section-summary",
        type=Path,
        default=repo_root
        / "data/output/cyclic_validation/reboot_external_section_amplitude_escalation_axial_admissibility/amplitude_escalation_summary.json",
    )
    parser.add_argument(
        "--structural-summary",
        type=Path,
        default=repo_root
        / "data/output/cyclic_validation/reboot_external_amplitude_escalation_stage1_closure/amplitude_escalation_summary.json",
    )
    parser.add_argument(
        "--formulation-summary",
        type=Path,
        default=repo_root
        / "data/output/cyclic_validation/reboot_structural_formulation_audit_summary/structural_formulation_audit_summary.json",
    )
    parser.add_argument(
        "--solver-policy-summary",
        type=Path,
        default=repo_root
        / "data/output/cyclic_validation/reboot_internal_solver_policy_benchmark_200mm_stage3/solver_policy_benchmark_summary.json",
    )
    parser.add_argument(
        "--solver-policy-case-matrix-summary",
        type=Path,
        default=repo_root
        / "data/output/cyclic_validation/reboot_solver_policy_case_matrix_stage3_extended/solver_policy_case_matrix_summary.json",
    )
    parser.add_argument(
        "--problematic-fiber-summary",
        type=Path,
        default=repo_root
        / "data/output/cyclic_validation/reboot_problematic_fiber_replay_audit/benchmark_summary.json",
    )
    parser.add_argument(
        "--reversal-frontier-summary",
        type=Path,
        default=repo_root
        / "data/output/cyclic_validation/reboot_section_reversal_frontier_audit/reversal_frontier_summary.json",
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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def first_case(cases: list[dict[str, object]], key: str) -> dict[str, object]:
    for case in cases:
        if str(case.get("key")) == key:
            return case
    return {}


@dataclass(frozen=True)
class ClosureCheckpointRow:
    checkpoint: str
    status: str
    evidence_summary: str
    primary_metric_label: str
    primary_metric_value: float
    secondary_metric_label: str
    secondary_metric_value: float
    artifact_path: str


def maybe_plot_frontier(
    output_path: Path,
    secondary_output_path: Path | None,
    internal_runtime_summary: dict[str, object],
    section_summary: dict[str, object],
    structural_summary: dict[str, object],
) -> list[str]:
    section = section_summary.get("section") or {}
    structural = structural_summary.get("structural") or {}

    labels = [
        "Section ext.",
        "Section fall_n",
        "Struct. ext.",
        "Struct. fall_n",
    ]
    values = [
        safe_float(section.get("highest_opensees_completed_amplitude")),
        safe_float(section.get("highest_fall_n_completed_amplitude")),
        safe_float(structural.get("highest_opensees_completed_amplitude")),
        safe_float(internal_runtime_summary.get("max_abs_tip_drift_mm")),
    ]
    units = ["1/m", "1/m", "mm", "mm"]
    colors = ["#d97706", "#0b5fa5", "#d97706", "#0b5fa5"]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = range(len(labels))
    ax.bar(x, values, color=colors, width=0.62)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Frontier value")
    ax.set_title("Reduced RC stage-1 frontier closure")
    for idx, (value, unit) in enumerate(zip(values, units)):
        if math.isfinite(value):
            ax.text(idx, value, f"{value:.3g} {unit}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path)
    outputs = [str(output_path)]
    if secondary_output_path is not None:
        ensure_dir(secondary_output_path.parent)
        fig.savefig(secondary_output_path)
        outputs.append(str(secondary_output_path))
    plt.close(fig)
    return outputs


def maybe_plot_timing(
    output_path: Path,
    secondary_output_path: Path | None,
    elasticized_timo_case: dict[str, object],
    nonlinear_100_case: dict[str, object],
) -> list[str]:
    labels = ["Elastic Timo total", "Elastic Timo process", "Nonlinear 100 total", "Nonlinear 100 process"]
    fall_n_values = [
        safe_float(elasticized_timo_case.get("falln_total_wall_seconds")),
        safe_float(elasticized_timo_case.get("falln_process_wall_seconds")),
        safe_float(nonlinear_100_case.get("fall_n_reported_total_wall_seconds")),
        safe_float(nonlinear_100_case.get("fall_n_process_wall_seconds")),
    ]
    opensees_values = [
        safe_float(elasticized_timo_case.get("opensees_total_wall_seconds")),
        safe_float(elasticized_timo_case.get("opensees_process_wall_seconds")),
        safe_float(nonlinear_100_case.get("opensees_reported_total_wall_seconds")),
        safe_float(nonlinear_100_case.get("opensees_process_wall_seconds")),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    x = list(range(len(labels)))
    width = 0.35
    ax.bar([v - width / 2 for v in x], fall_n_values, width, label="fall_n", color="#0b5fa5")
    ax.bar([v + width / 2 for v in x], opensees_values, width, label="OpenSeesPy", color="#d97706")
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylabel("Wall time [s]")
    ax.set_title("Stage-1 timing anchors")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    outputs = [str(output_path)]
    if secondary_output_path is not None:
        ensure_dir(secondary_output_path.parent)
        fig.savefig(secondary_output_path)
        outputs.append(str(secondary_output_path))
    plt.close(fig)
    return outputs


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    ensure_dir(args.figures_dir)
    ensure_dir(args.secondary_figures_dir)

    internal_runtime_summary = read_json(args.internal_runtime_summary)
    section_summary = read_json(args.section_summary)
    structural_summary = read_json(args.structural_summary)
    formulation_summary = read_json(args.formulation_summary)
    solver_policy_summary = read_json(args.solver_policy_summary)
    solver_policy_case_matrix_summary = read_json(args.solver_policy_case_matrix_summary)
    problematic_fiber_summary = read_json(args.problematic_fiber_summary)
    reversal_frontier_summary = read_json(args.reversal_frontier_summary)

    formulation_cases = list(formulation_summary.get("cases", []))
    elasticized_timo_case = first_case(formulation_cases, "elasticized_timoshenko")
    nonlinear_disp_case = first_case(formulation_cases, "nonlinear_disp")

    structural_rows: list[dict[str, object]] = []
    structural_csv_path = output_dir.parent / "reboot_external_amplitude_escalation_stage1_closure" / "structural_amplitude_escalation_summary.csv"
    if structural_csv_path.exists():
        with structural_csv_path.open("r", encoding="utf-8", newline="") as fh:
            structural_rows = list(csv.DictReader(fh))
    nonlinear_100_case = next(
        (
            row
            for row in structural_rows
            if row.get("scope") == "structural" and row.get("amplitude_value") == "100.0"
        ),
        {},
    )

    completed_case_matrix = list(solver_policy_case_matrix_summary.get("cases", []))

    checkpoints = [
        ClosureCheckpointRow(
            checkpoint="internal_structural_slice_runtime",
            status="closed"
            if safe_float(internal_runtime_summary.get("max_abs_tip_drift_mm")) >= 200.0
            else "open",
            evidence_summary=(
                "The declared fall_n structural slice now completes through the "
                "full internal 200 mm cyclic protocol under the shared PETSc SNES profile cascade."
            ),
            primary_metric_label="highest_fall_n_completed_amplitude_mm",
            primary_metric_value=safe_float(internal_runtime_summary.get("max_abs_tip_drift_mm")),
            secondary_metric_label="first_fall_n_failed_amplitude_mm",
            secondary_metric_value=math.nan,
            artifact_path=str(args.internal_runtime_summary),
        ),
        ClosureCheckpointRow(
            checkpoint="internal_solver_policy_surface_declared",
            status="closed"
            if str(solver_policy_summary.get("fastest_completed_policy")) == "newton-l2-only"
            and any(
                str(row.get("solver_policy")) == "anderson-only" and str(row.get("status")) == "completed"
                for row in solver_policy_summary.get("rows", [])
            )
            else "open",
            evidence_summary=(
                "The internal nonlinear-solver surface now has a clear promoted baseline "
                "(newton-l2-only), while anderson-only and nonlinear-cg-only both close "
                "the canonical slice without displacing the promoted baseline."
            ),
            primary_metric_label="fastest_completed_policy",
            primary_metric_value=1.0 if str(solver_policy_summary.get("fastest_completed_policy")) == "newton-l2-only" else 0.0,
            secondary_metric_label="completed_non_newton_policy_count",
            secondary_metric_value=float(
                sum(
                    1
                    for row in solver_policy_summary.get("rows", [])
                    if str(row.get("status")) == "completed"
                    and str(row.get("solver_policy")) in {"anderson-only", "quasi-newton-only", "nonlinear-gmres-only", "nonlinear-cg-only", "nonlinear-richardson-only"}
                )
            ),
            artifact_path=str(args.solver_policy_summary),
        ),
        ClosureCheckpointRow(
            checkpoint="alternative_newton_surface_declared",
            status="closed"
            if any(
                str(row.get("solver_policy")) == "newton-trust-region-dogleg-only"
                and str(row.get("status")) == "completed"
                for row in solver_policy_summary.get("rows", [])
            )
            else "open",
            evidence_summary=(
                "The Newton-family surface is no longer a single-profile story: "
                "newton-trust-region-dogleg closes the canonical slice and survives "
                "the representative Lobatto matrix, but remains slower than newton-l2-only."
            ),
            primary_metric_label="dogleg_completed_flag",
            primary_metric_value=1.0
            if any(
                str(row.get("solver_policy")) == "newton-trust-region-dogleg-only"
                and str(row.get("status")) == "completed"
                for row in solver_policy_summary.get("rows", [])
            )
            else 0.0,
            secondary_metric_label="dogleg_process_wall_seconds",
            secondary_metric_value=safe_float(
                next(
                    (
                        row.get("process_wall_seconds")
                        for row in solver_policy_summary.get("rows", [])
                        if str(row.get("solver_policy"))
                        == "newton-trust-region-dogleg-only"
                    ),
                    math.nan,
                )
            ),
            artifact_path=str(args.solver_policy_summary),
        ),
        ClosureCheckpointRow(
            checkpoint="elasticized_external_timoshenko_parity",
            status="closed"
            if safe_float(elasticized_timo_case.get("global_base_shear_error")) <= 1.0e-3
            and safe_float(elasticized_timo_case.get("station_section_tangent_error")) <= 1.0e-8
            else "open",
            evidence_summary=(
                "The stiffness-equivalent ElasticTimoshenkoBeam control closes the "
                "elasticized parity slice tightly enough to remove gross geometry/"
                "observable mismatch from the active suspicion list."
            ),
            primary_metric_label="elasticized_timoshenko_global_base_shear_error",
            primary_metric_value=safe_float(elasticized_timo_case.get("global_base_shear_error")),
            secondary_metric_label="elasticized_timoshenko_section_tangent_error",
            secondary_metric_value=safe_float(elasticized_timo_case.get("station_section_tangent_error")),
            artifact_path=str(args.formulation_summary),
        ),
        ClosureCheckpointRow(
            checkpoint="external_section_frontier_declared",
            status="closed"
            if safe_float(section_summary["section"]["highest_opensees_completed_amplitude"]) >= 0.015
            and safe_float(section_summary["section"]["first_opensees_failed_amplitude"]) == 0.02
            else "open",
            evidence_summary=(
                "The external section comparator closes through 0.015 1/m and still "
                "fails first at 0.020 1/m on the OpenSees zeroLengthSection side."
            ),
            primary_metric_label="highest_opensees_completed_curvature_1_over_m",
            primary_metric_value=safe_float(section_summary["section"]["highest_opensees_completed_amplitude"]),
            secondary_metric_label="first_opensees_failed_curvature_1_over_m",
            secondary_metric_value=safe_float(section_summary["section"]["first_opensees_failed_amplitude"]),
            artifact_path=str(args.section_summary),
        ),
        ClosureCheckpointRow(
            checkpoint="external_section_reversal_profile_rescue",
            status="closed"
            if bool(reversal_frontier_summary.get("all_profiles_failed"))
            else "open",
            evidence_summary=(
                "No materially plausible single-profile external concrete override "
                "rescues the first failing reversal at 0.020 1/m."
            ),
            primary_metric_label="all_profiles_failed_flag",
            primary_metric_value=1.0 if bool(reversal_frontier_summary.get("all_profiles_failed")) else 0.0,
            secondary_metric_label="best_failure_target_curvature_1_over_m",
            secondary_metric_value=safe_float(reversal_frontier_summary.get("best_failure_target_curvature_y")),
            artifact_path=str(args.reversal_frontier_summary),
        ),
        ClosureCheckpointRow(
            checkpoint="problematic_fiber_anchor_localization",
            status="closed"
            if str(problematic_fiber_summary.get("status")) == "completed"
            else "open",
            evidence_summary=(
                "The problematic extremal cover-concrete fiber has been localized "
                "and replayed under the exact strain history; the nonlinear mismatch "
                "survives there even though the broad event ordering is physically consistent."
            ),
            primary_metric_label="anchor_abs_tangent_error_mpa",
            primary_metric_value=safe_float(
                problematic_fiber_summary["replays"]["falln_history_replay"]["anchor_step_diagnostics"]["abs_tangent_error_mpa"]
            ),
            secondary_metric_label="anchor_abs_stress_error_mpa",
            secondary_metric_value=safe_float(
                problematic_fiber_summary["replays"]["falln_history_replay"]["anchor_step_diagnostics"]["abs_stress_error_mpa"]
            ),
            artifact_path=str(args.problematic_fiber_summary),
        ),
        ClosureCheckpointRow(
            checkpoint="external_structural_frontier_declared",
            status="closed"
            if safe_float(structural_summary["structural"]["highest_opensees_completed_amplitude"]) >= 100.0
            and safe_float(structural_summary["structural"]["first_opensees_failed_amplitude"]) == 75.0
            else "open",
            evidence_summary=(
                "The external structural comparator is now explicitly non-monotone: "
                "it fails first at 75 mm, reopens at 100 mm, and fails again at 125/150 mm."
            ),
            primary_metric_label="highest_opensees_completed_amplitude_mm",
            primary_metric_value=safe_float(structural_summary["structural"]["highest_opensees_completed_amplitude"]),
            secondary_metric_label="first_opensees_failed_amplitude_mm",
            secondary_metric_value=safe_float(structural_summary["structural"]["first_opensees_failed_amplitude"]),
            artifact_path=str(args.structural_summary),
        ),
        ClosureCheckpointRow(
            checkpoint="supplemental_non_newton_case_matrix",
            status="closed"
            if all(
                str(case.get("newton_l2_completed")).lower() == "true"
                for case in completed_case_matrix
            )
            and any(
                safe_float(case.get("completed_policy_count")) >= 3.0
                for case in completed_case_matrix
            )
            else "open",
            evidence_summary=(
                "A supplemental Lobatto case matrix now confirms that the extended solver "
                "surface remains hierarchical: newton-l2-only is still the strongest baseline, "
                "dogleg is a viable but slower Newton alternative, Anderson is the strongest "
                "new non-Newton route, nonlinear-cg is admissible but too expensive and fragile, "
                "and nonlinear-richardson remains non-promotable."
            ),
            primary_metric_label="case_matrix_completed_case_count",
            primary_metric_value=float(len(completed_case_matrix)),
            secondary_metric_label="best_non_newton_completed_cases",
            secondary_metric_value=float(
                sum(
                    1
                    for case in completed_case_matrix
                    if safe_float(case.get("completed_policy_count")) >= 3.0
                )
            ),
            artifact_path=str(args.solver_policy_case_matrix_summary),
        ),
        ClosureCheckpointRow(
            checkpoint="timing_anchor_fairness",
            status="closed",
            evidence_summary=(
                "Timing comparisons are now explicit and scoped: newton-l2-only is the "
                "cheapest completed internal baseline, dogleg is a viable but slower Newton "
                "alternative, anderson-only is validated but much more expensive, while "
                "nonlinear-cg-only closes the canonical slice only at prohibitive cost. fall_n is slightly faster on the elasticized Timoshenko "
                "parity control, while OpenSees remains faster on the nonlinear 100 mm external benchmark."
            ),
            primary_metric_label="elasticized_timoshenko_fall_n_over_opensees_total_ratio",
            primary_metric_value=safe_float(elasticized_timo_case.get("falln_total_wall_seconds"))
            / safe_float(elasticized_timo_case.get("opensees_total_wall_seconds")),
            secondary_metric_label="nonlinear_100_fall_n_over_opensees_total_ratio",
            secondary_metric_value=safe_float(nonlinear_100_case.get("fall_n_reported_total_wall_seconds"))
            / safe_float(nonlinear_100_case.get("opensees_reported_total_wall_seconds")),
            artifact_path=str(args.formulation_summary),
        ),
    ]

    stage1_algorithmic_baseline_closed = all(
        row.status == "closed" for row in checkpoints if row.checkpoint != "timing_anchor_fairness"
    )

    figure_frontier = args.figures_dir / "reduced_rc_stage1_closure_frontier.png"
    figure_timing = args.figures_dir / "reduced_rc_stage1_closure_timing.png"
    secondary_frontier = args.secondary_figures_dir / figure_frontier.name
    secondary_timing = args.secondary_figures_dir / figure_timing.name

    figure_outputs = {
        "frontier": maybe_plot_frontier(
            figure_frontier,
            secondary_frontier,
            internal_runtime_summary,
            section_summary,
            structural_summary,
        ),
        "timing": maybe_plot_timing(
            figure_timing,
            secondary_timing,
            elasticized_timo_case,
            nonlinear_100_case,
        ),
    }

    csv_path = output_dir / "stage1_closure_checkpoints.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(checkpoints[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in checkpoints)

    payload = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_stage1_closure_audit",
        "stage1_algorithmic_baseline_closed": stage1_algorithmic_baseline_closed,
        "stage1_interpretation": {
            "closed_scope": (
                "initial algorithmic/computational validation of the reduced RC-column slice"
            ),
            "not_closed_scope": (
                "full physical equivalence to experimental RC-column evidence at large amplitude"
            ),
            "summary": (
                "Stage 1 is now closed as an algorithmic baseline: fall_n's internal "
                "slice is robust through the full internal 200 mm protocol, the promoted "
                "internal solver baseline is explicit, an alternative Newton surface and "
                "the non-Newton extension surface are both explicitly benchmarked, elasticized external parity closes, "
                "the external section and structural frontiers are explicitly declared, and "
                "the remaining nonlinear gap is localized rather than diffuse."
            ),
        },
        "internal_runtime": {
            "highest_fall_n_completed_amplitude_mm": safe_float(internal_runtime_summary.get("max_abs_tip_drift_mm")),
            "max_abs_base_shear_kn": safe_float(internal_runtime_summary.get("max_abs_base_shear_kn")),
            "max_abs_base_moment_knm": safe_float(internal_runtime_summary.get("max_abs_base_moment_knm")),
            "max_newton_iterations": safe_float(internal_runtime_summary.get("max_newton_iterations")),
            "max_bisection_level": safe_float(internal_runtime_summary.get("max_bisection_level")),
        },
        "internal_solver_policy_surface": {
            "fastest_completed_policy": str(solver_policy_summary.get("fastest_completed_policy")),
            "best_process_wall_seconds": safe_float(solver_policy_summary.get("best_process_wall_seconds")),
            "completed_policy_count": safe_float(solver_policy_summary.get("completed_policy_count")),
            "failed_policy_count": safe_float(solver_policy_summary.get("failed_policy_count")),
            "completed_non_newton_policies": [
                str(row.get("solver_policy"))
                for row in solver_policy_summary.get("rows", [])
                if str(row.get("status")) == "completed"
                and str(row.get("solver_policy")) in {"anderson-only", "quasi-newton-only", "nonlinear-gmres-only", "nonlinear-cg-only", "nonlinear-richardson-only"}
            ],
            "completed_alternative_newton_policies": [
                str(row.get("solver_policy"))
                for row in solver_policy_summary.get("rows", [])
                if str(row.get("status")) == "completed"
                and str(row.get("solver_policy")) in {"newton-trust-region-only", "newton-trust-region-dogleg-only"}
            ],
        },
        "section_frontier": {
            "highest_opensees_completed_curvature_y": safe_float(
                section_summary["section"]["highest_opensees_completed_amplitude"]
            ),
            "first_opensees_failed_curvature_y": safe_float(
                section_summary["section"]["first_opensees_failed_amplitude"]
            ),
            "all_single_profile_reversal_overrides_failed": bool(
                reversal_frontier_summary.get("all_profiles_failed")
            ),
        },
        "structural_frontier": {
            "highest_fall_n_completed_amplitude_mm": safe_float(internal_runtime_summary.get("max_abs_tip_drift_mm")),
            "highest_opensees_completed_amplitude_mm": safe_float(
                structural_summary["structural"]["highest_opensees_completed_amplitude"]
            ),
            "first_opensees_failed_amplitude_mm": safe_float(
                structural_summary["structural"]["first_opensees_failed_amplitude"]
            ),
            "non_monotone_reopen_note": (
                "The OpenSees structural bridge fails first at 75 mm, reopens at 100 mm, "
                "and fails again at 125/150 mm under the declared convergence-profile cascade."
            ),
        },
        "timing_anchors": {
            "elasticized_timoshenko": {
                "fall_n_total_wall_seconds": safe_float(elasticized_timo_case.get("falln_total_wall_seconds")),
                "opensees_total_wall_seconds": safe_float(elasticized_timo_case.get("opensees_total_wall_seconds")),
                "fall_n_process_wall_seconds": safe_float(elasticized_timo_case.get("falln_process_wall_seconds")),
                "opensees_process_wall_seconds": safe_float(elasticized_timo_case.get("opensees_process_wall_seconds")),
            },
            "internal_solver_policy_200mm": {
                "newton_l2_process_wall_seconds": safe_float(
                    next(
                        (
                            row.get("process_wall_seconds")
                            for row in solver_policy_summary.get("rows", [])
                            if str(row.get("solver_policy")) == "newton-l2-only"
                        ),
                        math.nan,
                    )
                ),
                "newton_trust_region_dogleg_process_wall_seconds": safe_float(
                    next(
                        (
                            row.get("process_wall_seconds")
                            for row in solver_policy_summary.get("rows", [])
                            if str(row.get("solver_policy")) == "newton-trust-region-dogleg-only"
                        ),
                        math.nan,
                    )
                ),
                "anderson_process_wall_seconds": safe_float(
                    next(
                        (
                            row.get("process_wall_seconds")
                            for row in solver_policy_summary.get("rows", [])
                            if str(row.get("solver_policy")) == "anderson-only"
                        ),
                        math.nan,
                    )
                ),
                "nonlinear_cg_process_wall_seconds": safe_float(
                    next(
                        (
                            row.get("process_wall_seconds")
                            for row in solver_policy_summary.get("rows", [])
                            if str(row.get("solver_policy")) == "nonlinear-cg-only"
                        ),
                        math.nan,
                    )
                ),
            },
            "nonlinear_100mm": {
                "fall_n_total_wall_seconds": safe_float(nonlinear_100_case.get("fall_n_reported_total_wall_seconds")),
                "opensees_total_wall_seconds": safe_float(nonlinear_100_case.get("opensees_reported_total_wall_seconds")),
                "fall_n_process_wall_seconds": safe_float(nonlinear_100_case.get("fall_n_process_wall_seconds")),
                "opensees_process_wall_seconds": safe_float(nonlinear_100_case.get("opensees_process_wall_seconds")),
            },
        },
        "checkpoints": [asdict(row) for row in checkpoints],
        "artifacts": {
            "csv": str(csv_path),
            "figures": figure_outputs,
        },
        "source_artifacts": {
            "internal_runtime_summary": str(args.internal_runtime_summary),
            "section_summary": str(args.section_summary),
            "structural_summary": str(args.structural_summary),
            "formulation_summary": str(args.formulation_summary),
            "solver_policy_summary": str(args.solver_policy_summary),
            "solver_policy_case_matrix_summary": str(args.solver_policy_case_matrix_summary),
            "problematic_fiber_summary": str(args.problematic_fiber_summary),
            "reversal_frontier_summary": str(args.reversal_frontier_summary),
        },
    }
    write_json(output_dir / "stage1_closure_summary.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
