#!/usr/bin/env python3
"""Plot the physical scale=1 activation gate and FE2/XFEM one-step smoke."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALARM = ROOT / "data" / "output" / "lshaped_16storey_physical_scale1_linear_steel_yield_20260509"
DEFAULT_FE2 = ROOT / "data" / "output" / "lshaped_16storey_physical_scale1_xfem_linear_steel_alarm_onestep_20260509"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean_component(rows: list[dict[str, str]], suffix: str) -> list[float]:
    cols = [name for name in rows[0] if name != "time" and name.endswith(suffix)]
    if not cols:
        raise ValueError(f"No columns ending with {suffix}")
    return [sum(float(row[col]) for col in cols) / len(cols) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-alarm-dir", type=Path, default=DEFAULT_ALARM)
    parser.add_argument("--fe2-dir", type=Path, default=DEFAULT_FE2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc" / "figures" / "validation_reboot")
    parser.add_argument("--prefix", default="lshaped_16_physical_scale1_fe2_activation_gate")
    args = parser.parse_args()

    linear_summary_path = args.linear_alarm_dir / "recorders" / "newmark_linear_reference_summary.json"
    fe2_summary_path = args.fe2_dir / "recorders" / "seismic_fe2_one_way_summary.json"
    vtk_audit_path = args.fe2_dir / "recorders" / "publication_vtk_audit.json"
    linear_summary = json.loads(linear_summary_path.read_text(encoding="utf-8"))
    fe2_summary = json.loads(fe2_summary_path.read_text(encoding="utf-8"))
    vtk_audit = json.loads(vtk_audit_path.read_text(encoding="utf-8"))
    alarm_t = float(linear_summary["damage_alarm_time_s"])

    roof_rows = read_csv(args.linear_alarm_dir / "recorders" / "roof_displacement_newmark_linear_reference.csv")
    t_roof = [float(row["time"]) for row in roof_rows]
    ux = mean_component(roof_rows, "_dof0")
    uy = mean_component(roof_rows, "_dof1")
    uz = mean_component(roof_rows, "_dof2")

    scan_rows = read_csv(args.linear_alarm_dir / "recorders" / "linear_first_alarm_scan.csv")
    t_scan = [float(row["time"]) for row in scan_rows]
    damage = [float(row["peak_damage"]) for row in scan_rows]

    cracks_rows = read_csv(args.fe2_dir / "recorders" / "crack_evolution.csv")
    t_cracks = [float(row["time"]) for row in cracks_rows]
    total_cracks = [float(row["total_cracks"]) for row in cracks_rows]
    max_opening = [float(row["max_opening"]) * 1e3 for row in cracks_rows]

    site_rows = read_csv(args.fe2_dir / "recorders" / "local_macro_inferred_sites.csv")
    labels = [f"e{row['macro_element_id']} z/L={float(row['crack_z_over_l']):.2f}" for row in site_rows]
    scores = [float(row["candidate_score"]) for row in site_rows]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    ax = axes[0, 0]
    ax.plot(t_roof, ux, label=r"$\bar{u}_x$", lw=0.8)
    ax.plot(t_roof, uy, label=r"$\bar{u}_y$", lw=0.8)
    ax.plot(t_roof, uz, label=r"$\bar{u}_z$", lw=0.8)
    ax.axvline(alarm_t, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Roof displacement (m)")
    ax.set_title("Linear steel-yield alarm, physical scale=1")
    ax.grid(True, alpha=0.35, lw=0.3)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_scan, damage, color="#8c2d04", lw=0.8)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.axvline(alarm_t, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Steel-yield demand index")
    ax.set_title("Activation gate")
    ax.grid(True, alpha=0.35, lw=0.3)

    ax = axes[1, 0]
    ax.plot(t_cracks, total_cracks, "o-", lw=0.8, ms=3, color="#1f77b4", label="cracks")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Total cracks")
    ax2 = ax.twinx()
    ax2.plot(t_cracks, max_opening, "s-", lw=0.8, ms=3, color="#d62728", label="max opening")
    ax2.set_ylabel("Max opening (mm)")
    ax.set_title("One-way XFEM local response")
    ax.grid(True, alpha=0.35, lw=0.3)

    ax = axes[1, 1]
    ax.barh(range(len(scores)), scores, color="#4c78a8")
    ax.set_yticks(range(len(scores)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Macro-inferred site score")
    ax.set_title("Activated local sites")
    ax.grid(True, axis="x", alpha=0.35, lw=0.3)

    fig.suptitle("Physical scale=1 FE2/XFEM activation smoke from MYG004 steel-yield alarm", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.output_dir / f"{args.prefix}.pdf"
    png = args.output_dir / f"{args.prefix}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    summary = {
        "schema": "fall_n_lshaped_physical_scale1_fe2_activation_gate_plot_v1",
        "linear_alarm_dir": str(args.linear_alarm_dir.relative_to(ROOT)),
        "fe2_dir": str(args.fe2_dir.relative_to(ROOT)),
        "eq_scale": linear_summary["eq_scale"],
        "alarm_time_s": linear_summary["damage_alarm_time_s"],
        "alarm_element": linear_summary["damage_alarm_element"],
        "peak_linear_roof_component_m": linear_summary["peak_abs_roof_component_m"],
        "peak_linear_damage_index": linear_summary["peak_damage_index"],
        "fe2_transition_time_s": fe2_summary["transition_time_s"],
        "fe2_selected_site_count": fe2_summary["selected_site_count"],
        "fe2_completed_site_count": fe2_summary["completed_site_count"],
        "fe2_overall_pass": fe2_summary["overall_pass"],
        "vtk_issue_count": len(vtk_audit["issues"]),
        "vtk_visible_crack_cells": vtk_audit["visible_crack_cells"],
        "vtk_gauss_points": vtk_audit["gauss_points"],
        "vtk_endpoint_max_gap_m": vtk_audit["local_global_endpoint_max_gap_m"],
        "figures": [str(pdf.relative_to(ROOT)), str(png.relative_to(ROOT))],
    }
    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
