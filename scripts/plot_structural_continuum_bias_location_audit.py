#!/usr/bin/env python3
"""Summarize the longitudinal-bias sensitivity of the RC continuum gate.

The continuum column is intentionally biased in the longitudinal direction to
resolve the fixed-end plastic hinge. The loaded face also carries imposed
Dirichlet motion and the axial preload transfer, so this audit makes the bias
location explicit instead of treating the bias power as a scalar mesh detail.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path("data/output/cyclic_validation")


CASES = (
    (
        "50_fixed",
        "50 fixed",
        ROOT / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_50mm",
        "Implicit fixed-end bias from the original gate.",
    ),
    (
        "50_loaded",
        "50 loaded",
        ROOT / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_50mm_loaded_end_bias2",
        "Loaded-face-only refinement; intentionally tests the user's tip-bias hypothesis.",
    ),
    (
        "50_both",
        "50 both",
        ROOT / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_50mm_both_ends_bias2",
        "Symmetric endpoint refinement: fixed hinge plus loaded face.",
    ),
    (
        "100_fixed",
        "100 fixed",
        ROOT / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_100mm",
        "Implicit fixed-end bias extended to 100 mm.",
    ),
    (
        "100_both",
        "100 both",
        ROOT / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_100mm_both_ends_bias2",
        "Symmetric endpoint refinement extended to 100 mm.",
    ),
    (
        "50_refined_proxy",
        "50 6x6x12 proxy",
        ROOT / "structural_continuum_valid_refined_proxy_r0p5_6x6x12_50mm",
        "Refined fixed-end mesh with the cheaper tensile crack-band proxy.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create longitudinal-bias audit artifacts for the RC column gate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/figures/validation_reboot"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_numeric_csv(path: Path) -> list[dict[str, float]]:
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


def get_metric(summary: dict[str, Any], group: str, name: str) -> float:
    try:
        value = float(summary[group]["hex8"][name])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def first_metric(summary: dict[str, Any], candidates: tuple[tuple[str, str], ...]) -> float:
    for group, name in candidates:
        value = get_metric(summary, group, name)
        if math.isfinite(value):
            return value
    return math.nan


def case_row(key: str, label: str, bundle: Path, notes: str) -> dict[str, Any]:
    summary_path = bundle / "structural_continuum_steel_hysteresis_summary.json"
    summary = read_json(summary_path)
    spec = summary["continuum_reference_spec"]
    continuum = summary["continuum_cases"]["hex8"]
    row: dict[str, Any] = {
        "key": key,
        "label": label,
        "bundle": str(bundle),
        "amplitude_mm": float(summary["protocol"]["amplitudes_mm"][-1]),
        "nx": int(spec["nx"]),
        "ny": int(spec["ny"]),
        "nz": int(spec["nz"]),
        "bias_power": float(spec.get("longitudinal_bias_power", math.nan)),
        "bias_location": spec.get("longitudinal_bias_location", "fixed-end-implicit"),
        "material_mode": spec.get("material_mode", ""),
        "completed_successfully": bool(continuum.get("completed_successfully", False)),
        "continuum_total_wall_seconds": float(
            continuum.get("reported_total_wall_seconds", math.nan)
        ),
        "global_rms_base_shear_error": get_metric(
            summary,
            "global_comparison",
            "peak_normalized_rms_base_shear_error",
        ),
        "global_max_base_shear_error": get_metric(
            summary,
            "global_comparison",
            "peak_normalized_max_base_shear_error",
        ),
        "steel_local_work_ratio": first_metric(
            summary,
            (("steel_local_comparison", "continuum_to_structural_loop_work_ratio"),),
        ),
        "steel_hinge_work_ratio": first_metric(
            summary,
            (("steel_hinge_band_comparison", "continuum_to_structural_loop_work_ratio"),),
        ),
        "host_bar_rms_strain_gap": get_metric(
            summary,
            "embedded_transfer_comparison",
            "rms_abs_host_bar_axial_strain_gap",
        ),
        "notes": notes,
    }
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def hysteresis_series(bundle: Path) -> tuple[list[float], list[float], list[float], list[float]]:
    summary = read_json(bundle / "structural_continuum_steel_hysteresis_summary.json")
    structural = Path(summary["structural_reference"]["bundle_dir"]) / "hysteresis.csv"
    continuum = Path(summary["continuum_cases"]["hex8"]["bundle_dir"]) / "hysteresis.csv"
    s_rows = read_numeric_csv(structural)
    c_rows = read_numeric_csv(continuum)
    return (
        [1000.0 * row["drift_m"] for row in s_rows],
        [1000.0 * row["base_shear_MN"] for row in s_rows],
        [1000.0 * row["drift_m"] for row in c_rows],
        [1000.0 * row["base_shear_MN"] for row in c_rows],
    )


def plot(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["label"]) for row in rows]
    shear = [100.0 * float(row["global_rms_base_shear_error"]) for row in rows]
    hinge = [float(row["steel_hinge_work_ratio"]) for row in rows]
    times = [float(row["continuum_total_wall_seconds"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))

    axes[0, 0].bar(labels, shear, color="#2563eb")
    axes[0, 0].set_ylabel("RMS dV / peak structural [%]")
    axes[0, 0].set_title("Global hysteresis gap")
    axes[0, 0].grid(True, axis="y", alpha=0.25)

    colors = ["#16a34a" if abs(value - 1.0) <= 0.35 else "#f97316" for value in hinge]
    axes[0, 1].bar(labels, hinge, color=colors)
    axes[0, 1].axhline(1.0, color="#6b7280", linestyle="--", linewidth=0.9)
    axes[0, 1].set_ylabel("continuum / structural")
    axes[0, 1].set_title("Steel hinge loop work")
    axes[0, 1].grid(True, axis="y", alpha=0.25)

    for axis in axes[0, :]:
        axis.tick_params(axis="x", rotation=22)

    hyst_cases = (
        (
            "50 mm bias-location hysteresis",
            CASES[0][2],
            CASES[1][2],
            CASES[2][2],
            ("fixed", "loaded", "both"),
        ),
        (
            "100 mm endpoint-bias check",
            CASES[3][2],
            CASES[4][2],
            None,
            ("fixed", "both", ""),
        ),
    )
    palette = {"structural": "#111827", "fixed": "#2563eb", "loaded": "#dc2626", "both": "#16a34a"}

    for axis, (title, first, second, third, case_labels) in zip(axes[1, :], hyst_cases):
        sx, sy, cx, cy = hysteresis_series(first)
        axis.plot(sx, sy, color=palette["structural"], linewidth=1.6, label="structural")
        axis.plot(cx, cy, color=palette[case_labels[0]], linewidth=1.2, label=case_labels[0])
        _, _, cx, cy = hysteresis_series(second)
        axis.plot(cx, cy, color=palette[case_labels[1]], linewidth=1.2, label=case_labels[1])
        if third is not None:
            _, _, cx, cy = hysteresis_series(third)
            axis.plot(cx, cy, color=palette[case_labels[2]], linewidth=1.2, label=case_labels[2])
        axis.set_xlabel("tip drift [mm]")
        axis.set_ylabel("base shear [kN]")
        axis.set_title(title)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=True, fontsize=8)

    fig.suptitle("Longitudinal-bias audit for the structural-continuum RC gate")
    fig.tight_layout()

    outputs: dict[str, str] = {}
    for ext in ("png", "pdf"):
        path = out_dir / f"structural_continuum_bias_location_audit.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=250 if ext == "png" else None)
        outputs[ext] = str(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.bar(labels, times, color="#0f766e")
    ax.set_ylabel("continuum wall time [s]")
    ax.set_title("Cost of the accepted continuum slices")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"structural_continuum_bias_location_timing.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=250 if ext == "png" else None)
        outputs[f"timing_{ext}"] = str(path)
    plt.close(fig)

    return outputs


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [case_row(*case) for case in CASES]

    csv_path = args.output_dir / "structural_continuum_bias_location_audit.csv"
    write_csv(csv_path, rows)
    figures = plot(rows, args.output_dir)
    summary = {
        "status": "completed",
        "csv": str(csv_path),
        "figures": figures,
        "cases": rows,
        "diagnosis": (
            "Loaded-end-only bias is not acceptable for this cantilever RC column: "
            "it removes resolution from the fixed-end hinge and collapses the steel "
            "work transfer. Biasing both ends is the first mesh policy that improves "
            "the 50 mm global gate while preserving the base hinge; at 100 mm it "
            "also brings the steel hinge work close to unity, but the remaining "
            "global shear gap points back to the host concrete crack/shear-transfer "
            "law rather than to bar kinematics or axial load application."
        ),
    }
    summary_path = args.output_dir / "structural_continuum_bias_location_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
