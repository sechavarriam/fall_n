#!/usr/bin/env python3
"""Summarize crack-transfer models considered for the continuum RC gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("data/output/cyclic_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create crack-transfer model selection artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/figures/validation_reboot"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict[str, Any], group: str, name: str) -> float:
    try:
        return float(summary[group]["hex8"][name])
    except (KeyError, TypeError, ValueError):
        return math.nan


def structural_continuum_row(
    key: str,
    label: str,
    bundle: Path,
    notes: str,
) -> dict[str, Any]:
    summary = read_json(bundle / "structural_continuum_steel_hysteresis_summary.json")
    spec = summary["continuum_reference_spec"]
    continuum = summary["continuum_cases"]["hex8"]
    return {
        "key": key,
        "label": label,
        "family": "smeared_crack_band",
        "direct_cyclic_gate": True,
        "status": "completed",
        "material_mode": spec.get("material_mode", ""),
        "crack_transfer_law": spec.get(
            "concrete_crack_band_shear_transfer_law",
            "opening-exponential",
        ),
        "amplitude_mm": float(summary["protocol"]["amplitudes_mm"][-1]),
        "global_rms_base_shear_error": metric(
            summary,
            "global_comparison",
            "peak_normalized_rms_base_shear_error",
        ),
        "steel_hinge_work_ratio": metric(
            summary,
            "steel_hinge_band_comparison",
            "continuum_to_structural_loop_work_ratio",
        ),
        "total_wall_seconds": float(
            continuum.get("reported_total_wall_seconds", math.nan)
        ),
        "first_crack_drift_mm": math.nan,
        "peak_base_shear_kn": math.nan,
        "notes": notes,
    }


def fixed_timeout_row() -> dict[str, Any]:
    return {
        "key": "fixed_crackband_compression_gated_timeout",
        "label": "fixed + gated",
        "family": "fixed_crack_band",
        "direct_cyclic_gate": True,
        "status": "timeout",
        "material_mode": "fixed-crack-band-concrete",
        "crack_transfer_law": "compression-gated-opening",
        "amplitude_mm": 50.0,
        "global_rms_base_shear_error": math.nan,
        "steel_hinge_work_ratio": math.nan,
        "total_wall_seconds": 900.0,
        "first_crack_drift_mm": math.nan,
        "peak_base_shear_kn": math.nan,
        "notes": (
            "The richer fixed-crack basis did not complete the 50 mm cyclic "
            "gate within 900 s; this branch needs tangent/stabilization work "
            "before promotion."
        ),
    }


def kobathe_rows() -> list[dict[str, Any]]:
    summary = read_json(
        ROOT
        / "reboot_continuum_ko_bathe_crack_audit"
        / "continuum_crack_audit_summary.json"
    )
    rows: list[dict[str, Any]] = []
    for row in summary["rows"]:
        rows.append(
            {
                "key": f"kobathe_{row['hex_order']}",
                "label": f"Ko-Bathe {row['hex_order']}",
                "family": "ko_bathe_3d",
                "direct_cyclic_gate": False,
                "status": "completed" if row["completed_successfully"] else "failed",
                "material_mode": "nonlinear",
                "crack_transfer_law": "Ko-Bathe internal plasticity-fracture",
                "amplitude_mm": 20.0,
                "global_rms_base_shear_error": math.nan,
                "steel_hinge_work_ratio": math.nan,
                "total_wall_seconds": float(row["total_wall_seconds"]),
                "first_crack_drift_mm": float(row["first_crack_drift_mm"]),
                "peak_base_shear_kn": 1000.0 * float(row["peak_base_shear_mn"]),
                "notes": (
                    "Existing monotonic Ko-Bathe crack audit; included as "
                    "the heavier physics reference, not as the direct cyclic "
                    "100 mm closure metric."
                ),
            }
        )
    return rows


def xfem_row(output_dir: Path) -> dict[str, Any]:
    summary_path = output_dir / "xfem_column_base_crack_candidate_summary.json"
    summary = read_json(summary_path)
    return {
        "key": "xfem_mask_contract",
        "label": "XFEM mask",
        "family": "xfem_candidate",
        "direct_cyclic_gate": False,
        "status": summary["status"],
        "material_mode": "cohesive-interface-candidate",
        "crack_transfer_law": "shared cohesive crack shear-transfer law",
        "amplitude_mm": math.nan,
        "global_rms_base_shear_error": math.nan,
        "steel_hinge_work_ratio": math.nan,
        "total_wall_seconds": math.nan,
        "first_crack_drift_mm": math.nan,
        "peak_base_shear_kn": math.nan,
        "notes": (
            f"XFEM benchmark seam: {summary['cut_element_count']} cut element "
            f"and {summary['enriched_node_count']} enriched nodes in the "
            "base-crack mask contract."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    direct = [row for row in rows if row["direct_cyclic_gate"] and row["status"] == "completed"]
    labels = [row["label"] for row in direct]
    global_error = [
        100.0 * float(row["global_rms_base_shear_error"]) for row in direct
    ]
    steel_work = [float(row["steel_hinge_work_ratio"]) for row in direct]
    time_s = [float(row["total_wall_seconds"]) for row in direct]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    axes[0].bar(labels, global_error, color="#2563eb")
    axes[0].set_ylabel("RMS dV / peak structural [%]")
    axes[0].set_title("Direct cyclic gate")

    axes[1].bar(labels, steel_work, color="#16a34a")
    axes[1].axhline(1.0, color="#6b7280", linestyle="--", linewidth=0.9)
    axes[1].set_ylabel("continuum / structural")
    axes[1].set_title("Steel hinge work")

    axes[2].bar(labels, time_s, color="#7c3aed")
    axes[2].set_ylabel("wall time [s]")
    axes[2].set_title("Cost")

    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Crack-transfer law candidates for the RC continuum gate")
    fig.tight_layout()
    outputs: dict[str, str] = {}
    for ext in ("png", "pdf"):
        path = out_dir / f"structural_continuum_crack_transfer_model_audit.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=250 if ext == "png" else None)
        outputs[ext] = str(path)
    plt.close(fig)
    return outputs


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        structural_continuum_row(
            "cyclic_opening_exponential",
            "opening exp.",
            ROOT
            / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_100mm_both_ends_bias2_shear010",
            "Promoted lightweight smeared crack-band branch with opening-dependent exponential shear floor.",
        ),
        structural_continuum_row(
            "cyclic_compression_gated",
            "compression gated",
            ROOT
            / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_100mm_both_ends_bias2_shear010_compression_gated",
            "Same branch with compression-gated crack closure; response is nearly identical in this scalar crack state.",
        ),
        fixed_timeout_row(),
        *kobathe_rows(),
    ]

    # Keep the XFEM branch in the same benchmark report by regenerating its
    # lightweight mask artifact before reading its summary.
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/plot_xfem_column_enrichment_candidate.py",
            "--output-dir",
            str(args.output_dir),
        ],
        check=True,
    )
    rows.append(xfem_row(args.output_dir))

    csv_path = args.output_dir / "structural_continuum_crack_transfer_model_audit.csv"
    write_csv(csv_path, rows)
    figures = plot(rows, args.output_dir)
    summary = {
        "status": "completed",
        "csv": str(csv_path),
        "figures": figures,
        "cases": rows,
        "diagnosis": (
            "Three opening-aware transfer families were selected: smeared "
            "opening-exponential retention, compression/closure-gated aggregate "
            "interlock proxy, and MCFT/Ko-Bathe-style richer concrete physics. "
            "The first two are implemented through the shared "
            "CrackShearTransferLaw seam and are now consumable from both the "
            "smeared crack-band concrete and the XFEM cohesive interface. In the "
            "current scalar cyclic crack-band gate, compression-gating is almost "
            "inactive; the richer fixed-crack branch timed out, and Ko-Bathe "
            "remains the heavier monotonic crack reference rather than the cheap "
            "cyclic local-model candidate."
        ),
    }
    summary_path = args.output_dir / "structural_continuum_crack_transfer_model_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
