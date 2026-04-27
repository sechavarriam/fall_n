#!/usr/bin/env python3
"""
Plot canonical fall_n vs OpenSees reduced RC-column benchmark bundles.

This script is intentionally narrow: it reads one benchmark bundle and emits
compact overlay figures for the observables that are already part of the
validation contract, plus a small compute-time comparison plot.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    parser = argparse.ArgumentParser(
        description="Plot fall_n vs OpenSees reduced RC-column benchmark bundles."
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        help="Optional second output directory where the same figures are copied.",
    )
    parser.add_argument(
        "--stem-suffix",
        default="",
        help="Optional suffix appended to each output figure stem.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows = []
        for row in csv.DictReader(fh):
            converted: dict[str, object] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
        return rows


def save(fig: plt.Figure, out_dirs: list[Path], stem: str) -> None:
    for out_dir in out_dirs:
        ensure_dir(out_dir)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def suffixed(stem: str, suffix: str) -> str:
    return f"{stem}{suffix}" if suffix else stem


def timing_plot(summary: dict[str, object], out_dirs: list[Path], stem: str, suffix: str) -> None:
    falln = summary["fall_n"]["manifest"]["timing"]["total_wall_seconds"]
    ops = summary["opensees"]["manifest"]["timing"]["total_wall_seconds"]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(["fall_n", "OpenSeesPy"], [falln, ops], color=[BLUE, ORANGE], alpha=0.9)
    ax.set_ylabel("Reported total wall time [s]")
    ax.set_title("Compute-time comparison")
    for idx, value in enumerate((falln, ops)):
        ax.text(idx, value, f"{value:.3e}", ha="center", va="bottom", fontsize=8)
    save(fig, out_dirs, suffixed(stem, suffix))


def structural_plots(bundle: Path, summary: dict[str, object], out_dirs: list[Path], suffix: str) -> None:
    falln_h = read_csv_rows(bundle / "fall_n" / "comparison_hysteresis.csv")
    ops_h = read_csv_rows(bundle / "opensees" / "hysteresis.csv")
    falln_mk = read_csv_rows(bundle / "fall_n" / "comparison_moment_curvature_base.csv")
    ops_mk = read_csv_rows(bundle / "opensees" / "moment_curvature_base.csv")
    falln_control = read_csv_rows(bundle / "fall_n" / "control_state.csv")
    ops_control = read_csv_rows(bundle / "opensees" / "control_state.csv")
    comp = summary["comparison"]

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.plot(
        [1.0e3 * row["drift_m"] for row in falln_h],
        [1.0e3 * row["base_shear_MN"] for row in falln_h],
        color=BLUE,
        lw=1.4,
        label="fall_n",
    )
    ax.plot(
        [1.0e3 * row["drift_m"] for row in ops_h],
        [1.0e3 * row["base_shear_MN"] for row in ops_h],
        color=ORANGE,
        lw=1.2,
        ls="--",
        label="OpenSeesPy",
    )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title(
        "Reduced RC column cyclic hysteresis\n"
        + rf"$\max \varepsilon_V={comp['hysteresis']['max_rel_base_shear_error']:.2e}$, "
        + rf"$\mathrm{{rms}}\varepsilon_V={comp['hysteresis']['rms_rel_base_shear_error']:.2e}$"
    )
    ax.legend()
    save(fig, out_dirs, suffixed("reduced_rc_external_structural_hysteresis", suffix))

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.plot(
        [row["curvature_y"] for row in falln_mk],
        [1.0e3 * row["moment_y_MNm"] for row in falln_mk],
        color=BLUE,
        lw=1.4,
        label="fall_n",
    )
    ax.plot(
        [row["curvature_y"] for row in ops_mk],
        [1.0e3 * row["moment_y_MNm"] for row in ops_mk],
        color=ORANGE,
        lw=1.2,
        ls="--",
        label="OpenSeesPy",
    )
    ax.set_xlabel(r"Base curvature $\kappa_y$ [1/m]")
    ax.set_ylabel(r"Base moment $M_y$ [kN m]")
    ax.set_title(
        "Base-side moment-curvature comparison\n"
        + rf"$\max \varepsilon_M={comp['moment_curvature_base']['max_rel_moment_error']:.2e}$, "
        + rf"$\mathrm{{rms}}\varepsilon_M={comp['moment_curvature_base']['rms_rel_moment_error']:.2e}$"
    )
    ax.legend()
    save(fig, out_dirs, suffixed("reduced_rc_external_structural_moment_curvature", suffix))

    falln_section_path = bundle / "fall_n" / "section_response.csv"
    ops_section_path = bundle / "opensees" / "section_response.csv"
    if falln_section_path.exists() and ops_section_path.exists():
        falln_sections = read_csv_rows(falln_section_path)
        ops_sections = read_csv_rows(ops_section_path)
        falln_gps = {int(row["section_gp"]) for row in falln_sections}
        ops_gps = {int(row["section_gp"]) for row in ops_sections}
        gps = sorted(falln_gps & ops_gps)
        if gps:
            fig, axes = plt.subplots(
                len(gps),
                1,
                figsize=(5.4, 2.6 * len(gps)),
                sharex=True,
            )
            if len(gps) == 1:
                axes = [axes]
            by_gp = lambda rows, gp: [row for row in rows if int(row["section_gp"]) == gp]
            moment_comp = comp.get("section_response_moment", {})
            station_metrics = moment_comp.get("by_station", {})
            for ax, gp in zip(axes, gps):
                lhs = by_gp(falln_sections, gp)
                rhs = by_gp(ops_sections, gp)
                ax.plot(
                    [row["curvature_y"] for row in lhs],
                    [1.0e3 * row["moment_y_MNm"] for row in lhs],
                    color=BLUE,
                    lw=1.3,
                    label="fall_n",
                )
                ax.plot(
                    [row["curvature_y"] for row in rhs],
                    [1.0e3 * row["moment_y_MNm"] for row in rhs],
                    color=ORANGE,
                    lw=1.1,
                    ls="--",
                    label="OpenSeesPy",
                )
                metrics = station_metrics.get(str(gp), {})
                subtitle = f"gp={gp}"
                if metrics:
                    subtitle += (
                        rf", $\max \varepsilon_M={metrics.get('max_rel_moment_error', float('nan')):.2e}$"
                        + rf", $\mathrm{{rms}}\varepsilon_M={metrics.get('rms_rel_moment_error', float('nan')):.2e}$"
                    )
                ax.set_ylabel(r"$M_y$ [kN m]")
                ax.set_title(subtitle)
                ax.legend(loc="best")
            axes[-1].set_xlabel(r"Section curvature $\kappa_y$ [1/m]")
            fig.suptitle("Structural section-path parity by station", y=0.995)
            save(
                fig,
                out_dirs,
                suffixed("reduced_rc_external_structural_section_path_by_station", suffix),
            )

            if all("tangent_eiy_direct_raw" in row for row in falln_sections) and all(
                "tangent_eiy_direct_raw" in row for row in ops_sections
            ):
                base_gp = min(gps)
                lhs = by_gp(falln_sections, base_gp)
                rhs = by_gp(ops_sections, base_gp)
                fig, ax = plt.subplots(figsize=(5.4, 4.2))
                for rows, color, label in (
                    (lhs, BLUE, "fall_n"),
                    (rhs, ORANGE, "OpenSeesPy"),
                ):
                    ax.plot(
                        [row["curvature_y"] for row in rows],
                        [row["tangent_eiy_direct_raw"] for row in rows],
                        color=color,
                        lw=1.1,
                        alpha=0.75,
                        label=f"{label} raw",
                    )
                    ax.plot(
                        [row["curvature_y"] for row in rows],
                        [row["tangent_eiy"] for row in rows],
                        color=color,
                        lw=1.3,
                        ls="--",
                        label=f"{label} condensed",
                    )
                ax.set_xlabel(r"Base-side curvature $\kappa_y$ [1/m]")
                ax.set_ylabel(r"Tangent $EI_y$")
                ax.set_title(
                    "Structural base-station tangent audit\n"
                    + rf"$\max \varepsilon_{{K_{{raw}}}}={comp.get('section_response_tangent_direct_raw', {}).get('max_rel_tangent_direct_raw_error', float('nan')):.2e}$, "
                    + rf"$\max \varepsilon_{{\Delta K}}={comp.get('section_response_tangent_condensation_gap', {}).get('max_rel_tangent_condensation_gap_error', float('nan')):.2e}$"
                )
                ax.legend(loc="best", ncol=2)
                save(
                    fig,
                    out_dirs,
                    suffixed("reduced_rc_external_structural_base_station_tangent_modes", suffix),
                )

    if falln_control and ops_control:
        fig, axes = plt.subplots(3, 1, figsize=(5.4, 7.2), sharex=True)
        lhs_p = [row["p"] for row in falln_control]
        rhs_p = [row["p"] for row in ops_control]

        axes[0].plot(
            lhs_p,
            [1.0e3 * row["target_drift_m"] for row in falln_control],
            color=GREEN,
            lw=1.0,
            ls=":",
            label="Target",
        )
        axes[0].plot(
            lhs_p,
            [1.0e3 * row["actual_tip_drift_m"] for row in falln_control],
            color=BLUE,
            lw=1.3,
            label="fall_n",
        )
        axes[0].plot(
            rhs_p,
            [1.0e3 * row["actual_tip_drift_m"] for row in ops_control],
            color=ORANGE,
            lw=1.1,
            ls="--",
            label="OpenSeesPy",
        )
        axes[0].set_ylabel("Tip drift [mm]")
        axes[0].legend(loc="best")
        drift_metrics = comp.get("control_state_tip_drift", {})
        axes[0].set_title(
            "Boundary-control trace\n"
            + rf"$\max \varepsilon_u={drift_metrics.get('max_rel_tip_drift_error', float('nan')):.2e}$, "
            + rf"$\mathrm{{rms}}\varepsilon_u={drift_metrics.get('rms_rel_tip_drift_error', float('nan')):.2e}$"
        )

        axes[1].plot(
            lhs_p,
            [1.0e3 * row["top_axial_displacement_m"] for row in falln_control],
            color=BLUE,
            lw=1.3,
            label="fall_n",
        )
        axes[1].plot(
            rhs_p,
            [1.0e3 * row["top_axial_displacement_m"] for row in ops_control],
            color=ORANGE,
            lw=1.1,
            ls="--",
            label="OpenSeesPy",
        )
        axial_disp_metrics = comp.get("control_state_tip_axial_displacement", {})
        axes[1].set_ylabel("Top axial disp. [mm]")
        axes[1].set_title(
            rf"$\max \varepsilon_{{u_z}}={axial_disp_metrics.get('max_rel_tip_axial_displacement_error', float('nan')):.2e}$, "
            + rf"$\mathrm{{rms}}\varepsilon_{{u_z}}={axial_disp_metrics.get('rms_rel_tip_axial_displacement_error', float('nan')):.2e}$"
        )
        axes[1].legend(loc="best")

        axes[2].plot(
            lhs_p,
            [1.0e3 * row["base_axial_reaction_MN"] for row in falln_control],
            color=BLUE,
            lw=1.3,
            label="fall_n",
        )
        axes[2].plot(
            rhs_p,
            [1.0e3 * row["base_axial_reaction_MN"] for row in ops_control],
            color=ORANGE,
            lw=1.1,
            ls="--",
            label="OpenSeesPy",
        )
        axial_reaction_metrics = comp.get("control_state_base_axial_reaction", {})
        axes[2].set_ylabel("Base axial react. [kN]")
        axes[2].set_xlabel("Protocol fraction $p$")
        axes[2].set_title(
            rf"$\max \varepsilon_{{R_z}}={axial_reaction_metrics.get('max_rel_base_axial_reaction_error', float('nan')):.2e}$, "
            + rf"$\mathrm{{rms}}\varepsilon_{{R_z}}={axial_reaction_metrics.get('rms_rel_base_axial_reaction_error', float('nan')):.2e}$"
        )
        axes[2].legend(loc="best")
        save(
            fig,
            out_dirs,
            suffixed("reduced_rc_external_structural_control_trace", suffix),
        )

    timing_plot(summary, out_dirs, "reduced_rc_external_structural_timing", suffix)


def section_plots(bundle: Path, summary: dict[str, object], out_dirs: list[Path], suffix: str) -> None:
    falln = read_csv_rows(bundle / "fall_n" / "section_moment_curvature_baseline.csv")
    ops = read_csv_rows(bundle / "opensees" / "section_moment_curvature_baseline.csv")
    falln_control_path = bundle / "fall_n" / "section_control_trace.csv"
    ops_control_path = bundle / "opensees" / "section_control_trace.csv"
    falln_diag_path = bundle / "fall_n" / "section_tangent_diagnostics.csv"
    ops_diag_path = bundle / "opensees" / "section_tangent_diagnostics.csv"
    anchor_zone_path = bundle / "opensees" / "section_fiber_anchor_zone_summary.csv"
    anchor_total_path = bundle / "opensees" / "section_fiber_anchor_total_summary.csv"
    falln_diag = read_csv_rows(falln_diag_path) if falln_diag_path.exists() else []
    ops_diag = read_csv_rows(ops_diag_path) if ops_diag_path.exists() else []
    falln_control = read_csv_rows(falln_control_path) if falln_control_path.exists() else []
    ops_control = read_csv_rows(ops_control_path) if ops_control_path.exists() else []
    anchor_zone_rows = read_csv_rows(anchor_zone_path) if anchor_zone_path.exists() else []
    anchor_total_rows = read_csv_rows(anchor_total_path) if anchor_total_path.exists() else []
    comp = summary["comparison"]
    analysis = summary.get("analysis", "monotonic")

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.plot(
        [row["curvature_y"] for row in falln],
        [1.0e3 * row["moment_y_MNm"] for row in falln],
        color=BLUE,
        lw=1.4,
        label="fall_n",
    )
    ax.plot(
        [row["curvature_y"] for row in ops],
        [1.0e3 * row["moment_y_MNm"] for row in ops],
        color=ORANGE,
        lw=1.2,
        ls="--",
        label="OpenSeesPy",
    )
    ax.set_xlabel(r"Section curvature $\kappa_y$ [1/m]")
    ax.set_ylabel(r"Section moment $M_y$ [kN m]")
    ax.set_title(
        f"Section {analysis} moment-curvature comparison\n"
        + rf"$\max \varepsilon_M={comp['section_moment_curvature']['max_rel_moment_error']:.2e}$, "
        + rf"$\mathrm{{rms}}\varepsilon_M={comp['section_moment_curvature']['rms_rel_moment_error']:.2e}$"
    )
    ax.legend()
    save(
        fig,
        out_dirs,
        suffixed(f"reduced_rc_external_section_{analysis}_moment_curvature", suffix),
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.plot(
        [row["curvature_y"] for row in falln],
        [row["tangent_eiy"] for row in falln],
        color=BLUE,
        lw=1.3,
        label="fall_n",
    )
    ax.plot(
        [row["curvature_y"] for row in ops],
        [row["tangent_eiy"] for row in ops],
        color=ORANGE,
        lw=1.1,
        ls="--",
        label="OpenSeesPy",
    )
    ax.set_xlabel(r"Section curvature $\kappa_y$ [1/m]")
    ax.set_ylabel(r"Tangent $EI_y$")
    ax.set_title(
        f"Section {analysis} tangent comparison\n"
        + rf"$\max \varepsilon_{{K_t}}={comp['section_tangent']['max_rel_tangent_error']:.2e}$, "
        + rf"$\mathrm{{rms}}\varepsilon_{{K_t}}={comp['section_tangent']['rms_rel_tangent_error']:.2e}$"
    )
    ax.legend()
    save(fig, out_dirs, suffixed(f"reduced_rc_external_section_{analysis}_tangent", suffix))

    if falln_diag and ops_diag:
        fig, axes = plt.subplots(2, 1, figsize=(5.4, 6.6), sharex=True)
        for ax, rows, title, color in (
            (axes[0], falln_diag, "fall_n condensed vs numerical tangent", BLUE),
            (axes[1], ops_diag, "OpenSeesPy condensed vs numerical tangent", ORANGE),
        ):
            kappas = [row["curvature_y"] for row in rows]
            condensed = [
                row.get("tangent_eiy_condensed", row.get("tangent_eiy"))
                for row in rows
            ]
            direct_raw = [
                row.get("tangent_eiy_direct_raw", row.get("tangent_eiy_direct"))
                for row in rows
            ]
            numerical = [row["tangent_eiy_numerical"] for row in rows]
            left = [row["tangent_eiy_left"] for row in rows]
            right = [row["tangent_eiy_right"] for row in rows]
            ax.plot(kappas, condensed, color=color, lw=1.4, label="Condensed")
            ax.plot(kappas, direct_raw, color=color, lw=0.9, ls="-.", alpha=0.8, label="Direct raw")
            ax.plot(kappas, numerical, color=GREEN, lw=1.1, ls="--", label="Numerical")
            ax.plot(kappas, left, color=RED, lw=0.9, ls=":", label="Left slope")
            ax.plot(kappas, right, color="#6b46c1", lw=0.9, ls="-.", label="Right slope")
            zero_rows = [row for row in rows if int(row["zero_curvature_anchor"]) == 1]
            if zero_rows:
                ax.scatter(
                    [row["curvature_y"] for row in zero_rows],
                    [row["tangent_eiy_condensed"] for row in zero_rows],
                    color=color,
                    s=18,
                    zorder=3,
                )
            ax.set_ylabel(r"Tangent $EI_y$")
            ax.set_title(title)
            ax.legend(loc="best", ncol=2)
        axes[-1].set_xlabel(r"Section curvature $\kappa_y$ [1/m]")
        fig.suptitle(
            "Section tangent consistency audit\n"
            + rf"branch $\max \varepsilon_{{K_t}}={comp.get('section_tangent_branch_only', {}).get('max_rel_tangent_error', float('nan')):.2e}$, "
            + rf"zero-anchor $\max \varepsilon_{{K_t}}={comp.get('section_tangent_zero_curvature_anchor_only', {}).get('max_rel_tangent_error', float('nan')):.2e}$",
            y=0.995,
        )
        save(
            fig,
            out_dirs,
            suffixed(f"reduced_rc_external_section_{analysis}_tangent_consistency", suffix),
        )

        fig, ax = plt.subplots(figsize=(5.3, 4.2))
        for rows, color, label in (
            (falln_diag, BLUE, "fall_n"),
            (ops_diag, ORANGE, "OpenSeesPy"),
        ):
            kappas = [row["curvature_y"] for row in rows]
            direct_raw = [
                row.get("tangent_eiy_direct_raw", row.get("tangent_eiy_direct"))
                for row in rows
            ]
            condensed = [
                row.get("tangent_eiy_condensed", row.get("tangent_eiy"))
                for row in rows
            ]
            ax.plot(kappas, direct_raw, color=color, lw=1.2, alpha=0.75, label=f"{label} raw")
            ax.plot(kappas, condensed, color=color, lw=1.4, ls="--", label=f"{label} condensed")
        ax.set_xlabel(r"Section curvature $\kappa_y$ [1/m]")
        ax.set_ylabel(r"Tangent $EI_y$")
        ax.set_title(
            "Section raw-vs-condensed tangent audit\n"
            + rf"$\max \varepsilon_{{K_{{raw}}}}={comp.get('section_tangent_direct_raw', {}).get('max_rel_tangent_direct_raw_error', float('nan')):.2e}$, "
            + rf"$\max \varepsilon_{{\Delta K}}={comp.get('section_tangent_condensation_gap', {}).get('max_rel_tangent_condensation_gap_error', float('nan')):.2e}$"
        )
        ax.legend(loc="best", ncol=2)
        save(
            fig,
            out_dirs,
            suffixed(f"reduced_rc_external_section_{analysis}_tangent_modes", suffix),
        )

    if falln_control and ops_control:
        fig, axes = plt.subplots(3, 1, figsize=(5.8, 8.8), sharex=False)
        lhs_p = [row["pseudo_time_after"] for row in falln_control]
        rhs_p = [row["pseudo_time_after"] for row in ops_control]

        axes[0].plot(
            lhs_p,
            [row["target_curvature_y"] for row in falln_control],
            color=GREEN,
            lw=1.0,
            ls=":",
            label="Target",
        )
        axes[0].plot(
            lhs_p,
            [row["actual_curvature_y"] for row in falln_control],
            color=BLUE,
            lw=1.3,
            label="fall_n",
        )
        axes[0].plot(
            rhs_p,
            [row["actual_curvature_y"] for row in ops_control],
            color=ORANGE,
            lw=1.1,
            ls="--",
            label="OpenSeesPy",
        )
        control_metrics = comp.get("section_control_actual_curvature", {})
        axes[0].set_ylabel(r"Actual $\kappa_y$ [1/m]")
        axes[0].set_title(
            "Section control-trace parity\n"
            + rf"$\max \varepsilon_{{\kappa}}={control_metrics.get('max_rel_actual_curvature_error', float('nan')):.2e}$, "
            + rf"$\mathrm{{rms}}\varepsilon_{{\kappa}}={control_metrics.get('rms_rel_actual_curvature_error', float('nan')):.2e}$"
        )
        axes[0].legend(loc="best")

        iter_metrics = comp.get("section_control_newton_iterations", {})
        effort_metrics = comp.get("section_control_newton_iterations_per_substep", {})
        accepted_metrics = comp.get("section_control_accepted_substeps", {})
        bisection_metrics = comp.get("section_control_bisection_level", {})
        branch_metrics = comp.get("section_control_protocol_branch_id", {})
        reversal_metrics = comp.get("section_control_reversal_index", {})
        axes[1].step(
            lhs_p,
            [row["newton_iterations"] for row in falln_control],
            where="post",
            color=BLUE,
            lw=1.3,
            label="fall_n Newton iters",
        )
        axes[1].step(
            rhs_p,
            [row["newton_iterations"] for row in ops_control],
            where="post",
            color=ORANGE,
            lw=1.1,
            ls="--",
            label="OpenSeesPy Newton iters",
        )
        axes[1].set_ylabel("Newton iters")
        axes[1].set_title(
            "Iterative effort parity\n"
            + rf"$\max |\Delta i|={iter_metrics.get('max_abs_newton_iteration_error', float('nan')):.0f}$, "
            + f"mismatch steps={iter_metrics.get('mismatched_newton_iteration_error_step_count', 0)}, "
            + rf"$\max \varepsilon_{{i/n_s}}={effort_metrics.get('max_rel_newton_iterations_per_substep_error', float('nan')):.2e}$, "
            + rf"$\max |\Delta n_s|={accepted_metrics.get('max_abs_accepted_substep_error', float('nan')):.0f}$, "
            + rf"$\max |\Delta b|={bisection_metrics.get('max_abs_bisection_level_error', float('nan')):.0f}$"
            + "\n"
            + f"branch match={branch_metrics.get('exact_match_protocol_branch_id_error', False)}, "
            + f"reversal match={reversal_metrics.get('exact_match_reversal_index_error', False)}"
        )
        axes[1].legend(loc="best")

        axes[2].plot(
            rhs_p,
            [row["domain_time_after"] for row in ops_control],
            color=ORANGE,
            lw=1.1,
            ls="--",
            label="OpenSees domain time",
        )
        axes[2].plot(
            rhs_p,
            rhs_p,
            color=GREEN,
            lw=1.0,
            ls=":",
            label="Pseudo-time",
        )
        domain_metrics = comp.get("section_control_domain_time_opensees", {})
        axes[2].set_xlabel("Pseudo-time")
        axes[2].set_ylabel("OpenSees time")
        axes[2].set_title(
            "OpenSees domain-time audit\n"
            + f"monotone={domain_metrics.get('domain_time_monotone', False)}, "
            + rf"$\max |\Delta t - \Delta p|={domain_metrics.get('max_abs_increment_vs_pseudo_increment', float('nan')):.2e}$"
        )
        axes[2].legend(loc="best")
        save(
            fig,
            out_dirs,
            suffixed(f"reduced_rc_external_section_{analysis}_control_trace", suffix),
        )

    if anchor_zone_rows and anchor_total_rows:
        worst_anchor_row = max(
            anchor_total_rows,
            key=lambda row: float(row.get("rel_error_condensed_tangent", "nan")),
        )
        target_step = int(worst_anchor_row["step"])
        zone_rows = [
            row for row in anchor_zone_rows if int(row["step"]) == target_step
        ]
        zone_rows.sort(key=lambda row: row["zone"])
        if zone_rows:
            labels = [row["zone"].replace("_", "\n") for row in zone_rows]
            x = list(range(len(labels)))
            width = 0.35
            x_lhs = [value - width / 2 for value in x]
            x_rhs = [value + width / 2 for value in x]

            fig, axes = plt.subplots(2, 1, figsize=(6.2, 6.8), sharex=True)

            lhs_moment = [1.0e3 * row["lhs_moment_y_contribution_MNm"] for row in zone_rows]
            rhs_moment = [1.0e3 * row["rhs_moment_y_contribution_MNm"] for row in zone_rows]
            axes[0].bar(x_lhs, lhs_moment, width=width, color=BLUE, label="fall_n")
            axes[0].bar(x_rhs, rhs_moment, width=width, color=ORANGE, alpha=0.8, label="OpenSeesPy")
            axes[0].set_ylabel(r"$M_y$ contrib. [kN m]")
            axes[0].set_title(
                "Anchor-zone contribution audit\n"
                + f"step={target_step}, "
                + rf"$\varepsilon_{{K_t}}={float(worst_anchor_row['rel_error_condensed_tangent']):.2e}$"
            )
            axes[0].legend(loc="best")

            lhs_k0y = [row["lhs_raw_k0y_contribution"] for row in zone_rows]
            rhs_k0y = [row["rhs_raw_k0y_contribution"] for row in zone_rows]
            axes[1].bar(x_lhs, lhs_k0y, width=width, color=BLUE, label="fall_n")
            axes[1].bar(x_rhs, rhs_k0y, width=width, color=ORANGE, alpha=0.8, label="OpenSeesPy")
            axes[1].set_ylabel(r"$K_{0y}$ contrib.")
            axes[1].set_xticks(x, labels)
            axes[1].legend(loc="best")

            save(
                fig,
                out_dirs,
                suffixed(f"reduced_rc_external_section_{analysis}_anchor_zone_contributions", suffix),
            )

    timing_plot(summary, out_dirs, f"reduced_rc_external_section_{analysis}_timing", suffix)


def main() -> int:
    args = parse_args()
    summary = read_json(args.bundle / "benchmark_summary.json")
    out_dirs = [args.figures_dir]
    if args.secondary_figures_dir:
        out_dirs.append(args.secondary_figures_dir)

    scope = str(summary.get("benchmark_scope", ""))
    if "section" in scope:
        section_plots(args.bundle, summary, out_dirs, args.stem_suffix)
    else:
        structural_plots(args.bundle, summary, out_dirs, args.stem_suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
