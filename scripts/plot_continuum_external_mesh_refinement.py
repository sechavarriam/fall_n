#!/usr/bin/env python3
"""Summarize OpenSees continuum mesh refinement against fall_n.

This report is intentionally narrower than the global external benchmark
matrix.  It focuses on the elastic-with-steel continuum control, where a mesh
refinement study is physically meaningful and not entangled with the calibration
of external concrete damage laws.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/output/cyclic_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("doc/figures/validation_reboot"))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows: list[dict[str, float]] = []
        for row in csv.DictReader(f):
            rows.append({key: float(value) for key, value in row.items()})
        return rows


def compare_at_drift(
    falln_rows: list[dict[str, float]],
    opensees_rows: list[dict[str, float]],
    key: str,
) -> dict[str, float]:
    unused = set(range(len(opensees_rows)))
    errors: list[float] = []
    axis_errors: list[float] = []
    for lhs in falln_rows:
        best_idx = None
        best_axis_error = math.inf
        for idx in unused:
            axis_error = abs(lhs["drift_m"] - opensees_rows[idx]["drift_m"])
            if axis_error < best_axis_error:
                best_idx = idx
                best_axis_error = axis_error
        if best_idx is None or best_axis_error > 1.0e-9:
            continue
        unused.remove(best_idx)
        errors.append(lhs[key] - opensees_rows[best_idx][key])
        axis_errors.append(best_axis_error)
    return {
        "matched_count": len(errors),
        "rms_abs_error": math.sqrt(sum(e * e for e in errors) / len(errors))
        if errors
        else math.nan,
        "max_abs_error": max((abs(e) for e in errors), default=math.nan),
        "max_axis_error": max(axis_errors, default=math.nan),
    }


def summary_row_from_paired_bundle(bundle: Path, label: str) -> dict[str, Any]:
    summary = json.loads((bundle / "continuum_external_benchmark_summary.json").read_text())
    ops_manifest = summary["opensees"]["manifest"]
    comparison = summary["comparison"]
    return {
        "label": label,
        "bundle": bundle.name,
        "status": summary["status"],
        "solid_element": ops_manifest["solid_element"],
        "amplitude_mm": max(float(v) for v in ops_manifest["cyclic_amplitudes_mm"]),
        "mesh_request": (
            f'{ops_manifest["mesh_request"]["nx"]}x'
            f'{ops_manifest["mesh_request"]["ny"]}x'
            f'{ops_manifest["mesh_request"]["nz"]}'
        ),
        "opensees_node_count": ops_manifest["mesh_actual"]["node_count"],
        "opensees_solid_element_count": ops_manifest["mesh_actual"]["solid_element_count"],
        "falln_seconds": summary["fall_n"]["manifest"]["timing"]["total_wall_seconds"],
        "opensees_seconds": ops_manifest["timing"]["total_wall_seconds"],
        "base_shear_rms_error_MN": comparison["hysteresis_base_shear_at_matched_drift"]["rms_abs_error"],
        "base_shear_max_error_MN": comparison["hysteresis_base_shear_at_matched_drift"]["max_abs_error"],
        "steel_stress_rms_error_MPa": comparison["steel_max_abs_stress_at_matched_drift"]["rms_abs_error"],
        "steel_stress_max_error_MPa": comparison["steel_max_abs_stress_at_matched_drift"]["max_abs_error"],
        "note": "paired fall_n/OpenSees",
    }


def summary_row_from_direct_opensees(
    *,
    falln_bundle: Path,
    opensees_bundle: Path,
    label: str,
    amplitude_mm: float,
) -> dict[str, Any]:
    falln_summary = json.loads(
        (falln_bundle / "continuum_external_benchmark_summary.json").read_text()
    )
    falln_rows = read_csv(falln_bundle / "fall_n" / "hysteresis.csv")
    falln_steel = read_csv(falln_bundle / "steel_envelope_fall_n.csv")
    ops_rows = read_csv(opensees_bundle / "hysteresis.csv")
    ops_steel_source = read_csv(opensees_bundle / "steel_bar_response.csv")
    by_step: dict[int, list[dict[str, float]]] = {}
    for row in ops_steel_source:
        by_step.setdefault(int(row["step"]), []).append(row)
    ops_steel = []
    for step, rows in sorted(by_step.items()):
        ops_steel.append(
            {
                "step": float(step),
                "drift_m": sum(r["drift_m"] for r in rows) / len(rows),
                "max_abs_stress_MPa": max(abs(r["axial_stress_MPa"]) for r in rows),
            }
        )
    ops_manifest = json.loads((opensees_bundle / "reference_manifest.json").read_text())
    base = compare_at_drift(falln_rows, ops_rows, "base_shear_MN")
    steel = compare_at_drift(falln_steel, ops_steel, "max_abs_stress_MPa")
    return {
        "label": label,
        "bundle": opensees_bundle.name,
        "status": ops_manifest.get("status", ""),
        "solid_element": ops_manifest["solid_element"],
        "amplitude_mm": amplitude_mm,
        "mesh_request": (
            f'{ops_manifest["mesh_request"]["nx"]}x'
            f'{ops_manifest["mesh_request"]["ny"]}x'
            f'{ops_manifest["mesh_request"]["nz"]}'
        ),
        "opensees_node_count": ops_manifest["mesh_actual"]["node_count"],
        "opensees_solid_element_count": ops_manifest["mesh_actual"]["solid_element_count"],
        "falln_seconds": falln_summary["fall_n"]["manifest"]["timing"]["total_wall_seconds"],
        "opensees_seconds": ops_manifest["timing"]["total_wall_seconds"],
        "base_shear_rms_error_MN": base["rms_abs_error"],
        "base_shear_max_error_MN": base["max_abs_error"],
        "steel_stress_rms_error_MPa": steel["rms_abs_error"],
        "steel_stress_max_error_MPa": steel["max_abs_error"],
        "note": "direct OpenSees compared against paired fall_n mesh",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = [row for row in rows if row["status"] == "completed"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, ylabel, scale in [
        (axes[0], "base_shear_rms_error_MN", "base-shear RMS error [kN]", 1000.0),
        (axes[1], "opensees_seconds", "OpenSees wall time [s]", 1.0),
    ]:
        for row in completed:
            x = float(row["opensees_solid_element_count"])
            y = float(row[key]) * scale
            marker = "o" if row["solid_element"] == "bbarBrick" else "s"
            ax.scatter(x, y, s=72, marker=marker, label=f'{row["label"]} {row["solid_element"]}')
            ax.annotate(
                str(row["label"]),
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("OpenSees solid elements")
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=8)
    fig.suptitle("Elastic continuum external mesh-refinement control")
    fig.savefig(path, bbox_inches="tight", dpi=250)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.root
    rows = [
        summary_row_from_paired_bundle(
            root / "continuum_external_pair_cyclic_50mm_elastic_control_bbarBrick",
            "coarse-50mm",
        ),
        summary_row_from_paired_bundle(
            root / "continuum_external_elastic_with_steel_50mm_bbarBrick_refined_6x6x12",
            "6x6x12-50mm",
        ),
        summary_row_from_direct_opensees(
            falln_bundle=root / "continuum_external_elastic_with_steel_50mm_bbarBrick_refined_6x6x12",
            opensees_bundle=root / "opensees_continuum_elastic_stdBrick_6x6x12_50mm",
            label="6x6x12-50mm",
            amplitude_mm=50.0,
        ),
        summary_row_from_paired_bundle(
            root / "continuum_external_elastic_with_steel_200mm_bbarBrick_v1",
            "coarse-200mm",
        ),
        summary_row_from_paired_bundle(
            root / "continuum_external_elastic_with_steel_200mm_bbarBrick_refined_6x6x12",
            "6x6x12-200mm",
        ),
        {
            "label": "8x8x16-50mm",
            "bundle": "opensees_continuum_elastic_bbarBrick_8x8x16_50mm",
            "status": "timeout",
            "solid_element": "bbarBrick",
            "amplitude_mm": 50.0,
            "mesh_request": "8x8x16",
            "opensees_node_count": "",
            "opensees_solid_element_count": "",
            "falln_seconds": "",
            "opensees_seconds": 1200.0,
            "base_shear_rms_error_MN": "",
            "base_shear_max_error_MN": "",
            "steel_stress_rms_error_MPa": "",
            "steel_stress_max_error_MPa": "",
            "note": "OpenSees-only run exceeded 20 minute diagnostic budget",
        },
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "continuum_external_mesh_refinement_summary.csv"
    fig_path = args.output_dir / "continuum_external_mesh_refinement_summary.png"
    write_csv(csv_path, rows)
    plot(fig_path, rows)
    summary = {
        "status": "completed",
        "csv": str(csv_path),
        "figure": str(fig_path),
        "rows": rows,
    }
    summary_path = args.output_dir / "continuum_external_mesh_refinement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
