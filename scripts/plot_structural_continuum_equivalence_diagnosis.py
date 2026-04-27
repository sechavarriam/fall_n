#!/usr/bin/env python3
"""Summarize the structural-vs-continuum equivalence diagnosis.

This dashboard is intentionally small. It records the current lesson from the
RC-column validation: before tuning concrete degradation or chasing external
solvers, the continuum gate must preserve the imposed axial resultant and the
embedded-bar kinematic transfer. Cases that fail that balance are not valid
physical-equivalence evidence against the structural beam.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create structural-continuum equivalence diagnosis artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/figures/validation_reboot"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows: list[dict[str, float]] = []
        for row in csv.DictReader(fh):
            out: dict[str, float] = {}
            for key, value in row.items():
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    out[key] = math.nan
            rows.append(out)
        return rows


def metric(summary: dict[str, Any], group: str, name: str) -> float:
    try:
        value = summary[group]["hex8"][name]
        out = float(value)
    except (KeyError, TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def first_metric(
    summary: dict[str, Any],
    candidates: tuple[tuple[str, str], ...],
) -> float:
    for group, name in candidates:
        value = metric(summary, group, name)
        if math.isfinite(value):
            return value
    return math.nan


def axial_metrics(control_state: Path, target_mn: float = 0.02) -> dict[str, float]:
    rows = read_csv_rows(control_state)
    values = [row["base_axial_reaction_MN"] for row in rows]
    deviations = [abs(value - target_mn) for value in values]
    return {
        "record_count": float(len(values)),
        "min_base_axial_reaction_MN": min(values) if values else math.nan,
        "max_base_axial_reaction_MN": max(values) if values else math.nan,
        "max_axial_deviation_MN": max(deviations) if deviations else math.nan,
        "rms_axial_deviation_MN": math.sqrt(
            sum(value * value for value in deviations) / len(deviations)
        )
        if deviations
        else math.nan,
    }


def case_row(
    *,
    key: str,
    label: str,
    bundle: Path,
    notes: str,
) -> dict[str, Any]:
    summary = read_json(bundle / "structural_continuum_steel_hysteresis_summary.json")
    continuum = summary["continuum_cases"]["hex8"]
    runtime = read_json(Path(continuum["bundle_dir"]) / "runtime_manifest.json")
    control_state = Path(continuum["bundle_dir"]) / "control_state.csv"
    row: dict[str, Any] = {
        "key": key,
        "label": label,
        "bundle": str(bundle),
        "completed_successfully": bool(continuum.get("completed_successfully", False)),
        "embedded_boundary_mode": runtime.get("embedded_boundary_mode", ""),
        "axial_preload_transfer_mode": runtime.get("axial_preload_transfer_mode", ""),
        "support_reaction_node_count": runtime.get("discretization", {}).get(
            "support_reaction_node_count", math.nan
        ),
        "base_shear_peak_normalized_rms_error": metric(
            summary,
            "global_comparison",
            "peak_normalized_rms_base_shear_error",
        ),
        "steel_hinge_loop_work_ratio": first_metric(
            summary,
            (
                (
                    "steel_hinge_band_comparison",
                    "continuum_to_structural_loop_work_ratio",
                ),
                (
                    "steel_local_comparison",
                    "continuum_to_structural_loop_work_ratio",
                ),
            ),
        ),
        "host_bar_rms_strain_gap": metric(
            summary,
            "embedded_transfer_comparison",
            "rms_abs_host_bar_axial_strain_gap",
        ),
        "reported_total_wall_seconds": continuum.get("reported_total_wall_seconds", math.nan),
        "notes": notes,
    }
    row.update(axial_metrics(control_state))
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = tuple(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["label"]) for row in rows]
    axial = [1000.0 * float(row["max_axial_deviation_MN"]) for row in rows]
    shear = [
        100.0 * float(row["base_shear_peak_normalized_rms_error"]) for row in rows
    ]
    work = [float(row["steel_hinge_loop_work_ratio"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    colors = ["#dc2626" if value > 1.0 else "#16a34a" for value in axial]

    axes[0].bar(labels, axial, color=colors)
    axes[0].set_ylabel("max |Nbase - Ntarget| [kN]")
    axes[0].set_title("Axial-resultant gate")
    axes[0].axhline(1.0, color="#6b7280", linestyle="--", linewidth=0.9)

    axes[1].bar(labels, shear, color="#2563eb")
    axes[1].set_ylabel("RMS dV / peak structural [%]")
    axes[1].set_title("Global hysteresis gap")

    axes[2].bar(labels, work, color="#7c3aed")
    axes[2].axhline(1.0, color="#6b7280", linestyle="--", linewidth=0.9)
    axes[2].set_ylabel("continuum / structural")
    axes[2].set_title("Steel hinge work")

    for ax in axes:
        ax.tick_params(axis="x", rotation=18)
        ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Structural-continuum equivalence diagnosis")
    fig.tight_layout()
    outputs: dict[str, str] = {}
    for ext in ("png", "pdf"):
        path = out_dir / f"structural_continuum_equivalence_diagnosis.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=250 if ext == "png" else None)
        outputs[ext] = str(path)
    plt.close(fig)
    return outputs


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path("data/output/cyclic_validation")
    cases = [
        case_row(
            key="legacy_full_penalty",
            label="full penalty",
            bundle=root
            / "reboot_structural_continuum_multielement_cyclic_crack_band_4x4x4_50mm_free_audit",
            notes=(
                "Historical structural-continuum gate; fails axial-resultant "
                "balance under lateral cycling."
            ),
        ),
        case_row(
            key="dirichlet_composite",
            label="Dirichlet + split",
            bundle=root
            / "reboot_structural_continuum_cyclic_crack_band_4x4x4_50mm_clamped_dirichlet_composite_audit",
            notes=(
                "Current equivalence gate candidate; preserves axial resultant "
                "and embedded-bar kinematics."
            ),
        ),
    ]

    csv_path = args.output_dir / "structural_continuum_equivalence_diagnosis.csv"
    write_csv(csv_path, cases)
    figures = plot(cases, args.output_dir)
    summary = {
        "status": "completed",
        "csv": str(csv_path),
        "figures": figures,
        "cases": cases,
        "diagnosis": (
            "The old full-penalty coupling path is not valid equivalence "
            "evidence because it violates the imposed axial resultant. The "
            "Dirichlet endcap plus composite preload split restores axial "
            "balance; remaining discrepancies should be treated as material/"
            "kinematic localization gaps, not load-application artifacts."
        ),
    }
    summary_path = args.output_dir / "structural_continuum_equivalence_diagnosis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
