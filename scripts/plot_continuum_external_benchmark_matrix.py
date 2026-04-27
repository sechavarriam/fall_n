#!/usr/bin/env python3
"""Summarize paired fall_n/OpenSees continuum benchmark bundles.

The script intentionally reads only the public JSON/CSV artifacts emitted by
``run_reduced_rc_continuum_external_benchmark.py``.  It is therefore usable as
a lightweight wrapper-side dashboard without importing fall_n or OpenSeesPy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV and plots for continuum external benchmark bundles."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/output/cyclic_validation"),
        help="Directory containing continuum_external_* bundles.",
    )
    parser.add_argument(
        "--pattern",
        default="continuum_external_*",
        help="Glob pattern under --root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/figures/validation_reboot"),
    )
    return parser.parse_args()


def finite(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def metric(summary: dict[str, object], name: str, field: str) -> float:
    comparison = summary.get("comparison", {})
    if not isinstance(comparison, dict):
        return math.nan
    payload = comparison.get(name, {})
    return finite(payload.get(field)) if isinstance(payload, dict) else math.nan


def nested_str(payload: object, keys: tuple[str, ...], default: str = "") -> str:
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return str(current) if current not in (None, "") else default


def classify_case(summary: dict[str, object], ops_manifest: dict[str, object]) -> str:
    falln_rebar = nested_str(
        summary,
        ("model_controls", "fall_n", "reinforcement_mode"),
        nested_str(summary.get("fall_n", {}).get("manifest", {}), ("reinforcement_mode",)),
    ).replace("_", "-")
    ops_rebar = nested_str(
        summary,
        ("model_controls", "OpenSeesPy", "reinforcement_mode"),
        nested_str(ops_manifest, ("reinforcement_mode",)),
    )
    concrete = str(ops_manifest.get("concrete_model", ""))
    steel = str(ops_manifest.get("steel_model", ""))
    if falln_rebar == "continuum-only" and ops_rebar == "none":
        return "host-only"
    if concrete == "elastic-isotropic" and steel == "Elastic":
        return "elastic-with-steel"
    if concrete == "ASDConcrete3D":
        return "nonlinear-ASDConcrete3D"
    if concrete == "Damage2p":
        return "nonlinear-Damage2p"
    return "other"


def load_rows(root: Path, pattern: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bundle in sorted(root.glob(pattern)):
        summary_path = bundle / "continuum_external_benchmark_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        falln = summary.get("fall_n", {})
        ops = summary.get("opensees", {})
        falln_manifest = falln.get("manifest", {}) if isinstance(falln, dict) else {}
        ops_manifest = ops.get("manifest", {}) if isinstance(ops, dict) else {}
        if not falln_manifest or not ops_manifest:
            continue
        ops_mesh = ops_manifest.get("mesh_actual", {}) if isinstance(ops_manifest, dict) else {}
        amplitudes = ops_manifest.get("cyclic_amplitudes_mm", []) if isinstance(ops_manifest, dict) else []
        amplitude = max([finite(value) for value in amplitudes], default=math.nan)
        row = {
            "bundle": bundle.name,
            "status": summary.get("status", ""),
            "case_family": classify_case(summary, ops_manifest),
            "solid_element": ops_manifest.get("solid_element", ""),
            "concrete_model": ops_manifest.get("concrete_model", ""),
            "steel_model": ops_manifest.get("steel_model", ""),
            "falln_material_mode": nested_str(falln_manifest, ("material_mode",)),
            "falln_kinematics": nested_str(falln_manifest, ("continuum_kinematics",)),
            "falln_reinforcement_mode": nested_str(
                summary,
                ("model_controls", "fall_n", "reinforcement_mode"),
                nested_str(falln_manifest, ("reinforcement_mode",)),
            ),
            "opensees_reinforcement_mode": nested_str(
                summary,
                ("model_controls", "OpenSeesPy", "reinforcement_mode"),
                nested_str(ops_manifest, ("reinforcement_mode",)),
            ),
            "falln_host_concrete_zoning_mode": nested_str(
                summary,
                ("model_controls", "fall_n", "host_concrete_zoning_mode"),
                nested_str(falln_manifest, ("host_concrete_zoning_mode",)),
            ),
            "opensees_host_concrete_zoning_mode": nested_str(
                summary,
                ("model_controls", "OpenSeesPy", "host_concrete_zoning_mode"),
                nested_str(ops_manifest.get("mesh_request", {}), ("host_concrete_zoning_mode",)),
            ),
            "amplitude_mm": amplitude,
            "failure_step": ops_manifest.get("failure_step", ""),
            "failure_target_drift_m": ops_manifest.get("failure_target_drift_m", ""),
            "falln_seconds": finite(
                (falln_manifest.get("timing", {}) if isinstance(falln_manifest, dict) else {}).get("total_wall_seconds")
            ),
            "opensees_seconds": finite(
                (ops_manifest.get("timing", {}) if isinstance(ops_manifest, dict) else {}).get("total_wall_seconds")
            ),
            "opensees_node_count": ops_mesh.get("node_count", "") if isinstance(ops_mesh, dict) else "",
            "opensees_solid_element_count": ops_mesh.get("solid_element_count", "") if isinstance(ops_mesh, dict) else "",
            "base_shear_rms_error_MN": metric(
                summary,
                "hysteresis_base_shear_at_matched_drift",
                "rms_abs_error",
            ),
            "base_shear_max_error_MN": metric(
                summary,
                "hysteresis_base_shear_at_matched_drift",
                "max_abs_error",
            ),
            "axial_reaction_rms_error_MN": metric(
                summary,
                "control_base_axial_reaction_at_matched_drift",
                "rms_abs_error",
            ),
            "steel_stress_rms_error_MPa": metric(
                summary,
                "steel_max_abs_stress_at_matched_drift",
                "rms_abs_error",
            ),
            "steel_stress_max_error_MPa": metric(
                summary,
                "steel_max_abs_stress_at_matched_drift",
                "max_abs_error",
            ),
        }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_matrix(out_dir: Path, rows: list[dict[str, object]]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {"SSPbrick": "#1f77b4", "stdBrick": "#2ca02c", "bbarBrick": "#d62728"}
    markers = {"completed": "o", "failed_with_partial_artifacts": "x", "failed": "x"}

    def scatter_metric(metric_key: str, ylabel: str, stem: str) -> str:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for row in rows:
            x = finite(row["amplitude_mm"])
            y = finite(row[metric_key])
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            solid = str(row["solid_element"])
            status = str(row["status"])
            ax.scatter(
                x,
                y,
                color=colors.get(solid, "#7f7f7f"),
                marker=markers.get(status, "o"),
                s=68,
                label=f"{solid} / {status}",
            )
            label = str(row.get("case_family", "case")).replace("nonlinear-", "")
            ax.annotate(
                f"{solid.replace('Brick', '')}/{label}",
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
        ax.set_xlabel("Cyclic amplitude [mm]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.set_title(stem.replace("_", " "))
        path = out_dir / f"{stem}.png"
        fig.savefig(path, bbox_inches="tight", dpi=250)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def timing_plot() -> str:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for row in rows:
            x = finite(row["amplitude_mm"])
            falln = finite(row["falln_seconds"])
            ops = finite(row["opensees_seconds"])
            solid = str(row["solid_element"])
            if math.isfinite(x) and math.isfinite(falln):
                ax.scatter(x, falln, color=colors.get(solid, "#7f7f7f"), marker="o", s=68)
            if math.isfinite(x) and math.isfinite(ops):
                ax.scatter(x, ops, color=colors.get(solid, "#7f7f7f"), marker="^", s=68)
        ax.set_xlabel("Cyclic amplitude [mm]")
        ax.set_ylabel("reported wall time [s]")
        ax.grid(True, alpha=0.25)
        ax.set_title("continuum external benchmark timing")
        ax.scatter([], [], color="#4b5563", marker="o", label="fall_n")
        ax.scatter([], [], color="#4b5563", marker="^", label="OpenSeesPy")
        ax.legend(fontsize=8)
        path = out_dir / "continuum_external_benchmark_timing.png"
        fig.savefig(path, bbox_inches="tight", dpi=250)
        fig.savefig(out_dir / "continuum_external_benchmark_timing.pdf", bbox_inches="tight")
        plt.close(fig)
        return str(path)

    return {
        "base_shear": scatter_metric(
            "base_shear_rms_error_MN",
            "RMS base-shear difference [MN]",
            "continuum_external_benchmark_base_shear_error",
        ),
        "steel_stress": scatter_metric(
            "steel_stress_rms_error_MPa",
            "RMS steel-stress envelope difference [MPa]",
            "continuum_external_benchmark_steel_error",
        ),
        "timing": timing_plot(),
    }


def main() -> int:
    args = parse_args()
    rows = load_rows(args.root, args.pattern)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "continuum_external_benchmark_matrix.csv"
    write_csv(csv_path, rows)
    figures = plot_matrix(args.output_dir, rows) if rows else {}
    summary = {
        "status": "completed",
        "root": str(args.root),
        "pattern": args.pattern,
        "case_count": len(rows),
        "csv": str(csv_path),
        "figures": figures,
    }
    (args.output_dir / "continuum_external_benchmark_matrix_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
