#!/usr/bin/env python3
"""Plot the crack-band shear-retention sensitivity at the 100 mm gate."""

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
        "shear005",
        "0.05",
        ROOT / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_100mm_both_ends_bias2_shear005",
    ),
    (
        "shear010",
        "0.10",
        ROOT / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_100mm_both_ends_bias2_shear010",
    ),
    (
        "shear020",
        "0.20",
        ROOT / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_100mm_both_ends_bias2",
    ),
    (
        "shear040",
        "0.40",
        ROOT / "kinematic_gate_lateral_only_struct_ratio0p5_4x4x4_100mm_both_ends_bias2_shear04",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the crack-band shear-retention audit artifacts."
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
            converted: dict[str, float] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = math.nan
            rows.append(converted)
        return rows


def metric(summary: dict[str, Any], group: str, name: str) -> float:
    try:
        value = float(summary[group]["hex8"][name])
    except (KeyError, TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def case_row(key: str, label: str, bundle: Path) -> dict[str, Any]:
    summary = read_json(bundle / "structural_continuum_steel_hysteresis_summary.json")
    spec = summary["continuum_reference_spec"]
    continuum = summary["continuum_cases"]["hex8"]
    return {
        "key": key,
        "label": label,
        "bundle": str(bundle),
        "residual_shear_ratio": float(
            spec["concrete_crack_band_residual_shear_ratio"]
        ),
        "global_rms_base_shear_error": metric(
            summary,
            "global_comparison",
            "peak_normalized_rms_base_shear_error",
        ),
        "global_max_base_shear_error": metric(
            summary,
            "global_comparison",
            "peak_normalized_max_base_shear_error",
        ),
        "steel_hinge_work_ratio": metric(
            summary,
            "steel_hinge_band_comparison",
            "continuum_to_structural_loop_work_ratio",
        ),
        "steel_local_work_ratio": metric(
            summary,
            "steel_local_comparison",
            "continuum_to_structural_loop_work_ratio",
        ),
        "host_bar_rms_strain_gap": metric(
            summary,
            "embedded_transfer_comparison",
            "rms_abs_host_bar_axial_strain_gap",
        ),
        "continuum_total_wall_seconds": float(
            continuum.get("reported_total_wall_seconds", math.nan)
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def hysteresis(bundle: Path) -> tuple[list[float], list[float], list[float], list[float]]:
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

    ratios = [float(row["residual_shear_ratio"]) for row in rows]
    shear = [100.0 * float(row["global_rms_base_shear_error"]) for row in rows]
    work = [float(row["steel_hinge_work_ratio"]) for row in rows]
    times = [float(row["continuum_total_wall_seconds"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    axes[0].plot(ratios, shear, marker="o", color="#2563eb")
    axes[0].set_xlabel("residual shear ratio")
    axes[0].set_ylabel("RMS dV / peak structural [%]")
    axes[0].set_title("Global gap")

    axes[1].plot(ratios, work, marker="o", color="#16a34a")
    axes[1].axhline(1.0, color="#6b7280", linestyle="--", linewidth=0.9)
    axes[1].set_xlabel("residual shear ratio")
    axes[1].set_ylabel("continuum / structural")
    axes[1].set_title("Steel hinge work")

    axes[2].plot(ratios, times, marker="o", color="#7c3aed")
    axes[2].set_xlabel("residual shear ratio")
    axes[2].set_ylabel("continuum wall time [s]")
    axes[2].set_title("Cost")

    for axis in axes:
        axis.grid(True, alpha=0.25)

    fig.suptitle("Crack-band residual shear sensitivity at 100 mm")
    fig.tight_layout()
    outputs: dict[str, str] = {}
    for ext in ("png", "pdf"):
        path = out_dir / f"structural_continuum_crack_shear_retention_audit.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=250 if ext == "png" else None)
        outputs[ext] = str(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sx, sy, _, _ = hysteresis(CASES[1][2])
    ax.plot(sx, sy, color="#111827", linewidth=1.8, label="structural")
    colors = ("#0ea5e9", "#16a34a", "#f97316", "#dc2626")
    for (_, label, bundle), color in zip(CASES, colors):
        _, _, cx, cy = hysteresis(bundle)
        ax.plot(cx, cy, color=color, linewidth=1.2, label=f"shear {label}")
    ax.set_xlabel("tip drift [mm]")
    ax.set_ylabel("base shear [kN]")
    ax.set_title("100 mm hysteresis under residual shear sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"structural_continuum_crack_shear_retention_hysteresis.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=250 if ext == "png" else None)
        outputs[f"hysteresis_{ext}"] = str(path)
    plt.close(fig)
    return outputs


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [case_row(*case) for case in CASES]
    csv_path = args.output_dir / "structural_continuum_crack_shear_retention_audit.csv"
    write_csv(csv_path, rows)
    figures = plot(rows, args.output_dir)
    best_global = min(rows, key=lambda row: row["global_rms_base_shear_error"])
    summary = {
        "status": "completed",
        "csv": str(csv_path),
        "figures": figures,
        "cases": rows,
        "best_global_case": best_global,
        "diagnosis": (
            "A constant residual shear ratio is influential but not a complete "
            "closure mechanism. The 0.10 case gives the best global 100 mm "
            "match in this matrix while keeping the steel hinge work near unity. "
            "Larger residual shear over-stiffens the global loop, and very low "
            "residual shear becomes expensive and under-dissipates steel work. "
            "The next constitutive improvement should make shear transfer depend "
            "on crack opening and normal compression instead of using a single "
            "constant residual factor."
        ),
    }
    summary_path = args.output_dir / "structural_continuum_crack_shear_retention_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
