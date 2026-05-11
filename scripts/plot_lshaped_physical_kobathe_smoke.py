#!/usr/bin/env python3
"""Plot the physical scale=1 Ko-Bathe Hex27 FE2 one-step smoke."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "data" / "output" / "lshaped_16storey_physical_scale1_kobathe_linear_steel_alarm_onestep_20260509"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_log(path: Path) -> dict[str, float | int | str | None]:
    out: dict[str, float | int | str | None] = {
        "eq_scale": None,
        "first_yield_time_s": None,
        "first_yield_element": None,
        "submodels": None,
        "evolution_steps": None,
        "peak_damage": None,
        "active_cracks": None,
    }
    if not path.exists():
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "eq_scale": r"scale=([0-9.]+)",
        "first_yield_time_s": r"First yield:\s+t\s+=\s+([0-9.eE+-]+)",
        "first_yield_element": r"First yield:\s+t\s+=\s+[0-9.eE+-]+\s+s\s+\(element\s+([0-9]+)\)",
        "submodels": r"Sub-models:\s+([0-9]+)",
        "evolution_steps": r"Evolution:\s+([0-9]+)\s+steps",
        "peak_damage": r"Peak damage:\s+([0-9.eE+-]+)",
        "active_cracks": r"Active cracks:\s+([0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            out[key] = float(value) if key not in {"first_yield_element", "submodels", "evolution_steps", "active_cracks"} else int(value)
    return out


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def plot_log_value(value: float | int | None) -> float:
    if value is None:
        return 1.0e-12
    return max(float(value), 1.0e-12)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc" / "figures" / "validation_reboot")
    parser.add_argument("--prefix", default="lshaped_16_physical_scale1_kobathe_hex27_smoke")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    audit_path = first_existing([
        run_dir / "recorders" / "publication_vtk_audit.json",
        run_dir / "recorders" / "vtk_audit_summary.json",
        run_dir / "recorders" / "vtk_audit_summary_partial.json",
    ])
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    sites = read_csv(run_dir / "recorders" / "local_macro_inferred_sites.csv")
    cracks = read_csv(run_dir / "recorders" / "crack_evolution.csv")
    log_path = first_existing([
        run_dir / "run_stdout.log",
        run_dir.parent / "kobathe_smoke_stdout.log",
    ])
    log_summary = parse_log(log_path)

    site_labels = [f"site {row['local_site_index']}\nz/L={float(row['crack_z_over_l']):.2f}" for row in sites]
    site_scores = [float(row["candidate_score"]) for row in sites]
    crack_row = cracks[-1]
    crack_count = int(float(crack_row["total_cracks"]))
    max_opening_mm = float(crack_row["max_opening"]) * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8))

    ax = axes[0, 0]
    ax.bar(site_labels, site_scores, color="#4c78a8")
    ax.set_ylabel("Macro-inferred score")
    ax.set_title("Selected physical sites")
    ax.grid(True, axis="y", alpha=0.35, lw=0.3)

    ax = axes[0, 1]
    ax.bar(["cracks", "visible cells", "Gauss / 10"], [crack_count, audit["visible_crack_cells"], audit["gauss_points"] / 10.0],
           color=["#8c2d04", "#d62728", "#54a24b"])
    ax.set_title("Local observables")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.35, lw=0.3)

    ax = axes[1, 0]
    file_counts = audit["files"]
    ax.bar(file_counts.keys(), file_counts.values(), color="#7f7f7f")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("VTU files")
    ax.set_title("Publication VTK bundle")
    ax.grid(True, axis="y", alpha=0.35, lw=0.3)

    ax = axes[1, 1]
    gap = audit.get("local_global_endpoint_max_gap_m")
    warp_gap = audit.get("local_global_reference_warp_endpoint_max_gap_m")
    slip = audit.get("max_bond_slip_m")
    ax.bar(["endpoint gap", "warp gap", "bond slip"],
           [plot_log_value(gap), plot_log_value(warp_gap), plot_log_value(slip)],
           color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.set_yscale("log")
    ax.set_ylabel("m")
    ax.set_title(f"Placement/slip checks; max opening={max_opening_mm:.3f} mm")
    ax.grid(True, axis="y", alpha=0.35, lw=0.3)

    fig.suptitle("Physical scale=1 FE2/Ko-Bathe Hex27 smoke", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{args.prefix}.pdf"
    png = output_dir / f"{args.prefix}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    summary = {
        "schema": "fall_n_lshaped_physical_scale1_kobathe_hex27_smoke_plot_v1",
        "run_dir": str(run_dir.relative_to(ROOT)),
        "audit": str(audit_path.relative_to(ROOT)),
        "log": str(log_path.relative_to(ROOT)),
        **log_summary,
        "selected_site_count": len(sites),
        "vtk_issue_count": len(audit["issues"]),
        "vtk_total_vtu_files": audit["total_vtu_files"],
        "vtk_visible_crack_cells": audit["visible_crack_cells"],
        "vtk_gauss_points": audit["gauss_points"],
        "kobathe_crack_families_active": audit["kobathe_crack_families_active"],
        "min_visible_crack_opening_m": audit["min_visible_crack_opening_m"],
        "max_crack_opening_m": float(crack_row["max_opening"]),
        "total_cracks": crack_count,
        "local_global_endpoint_max_gap_m": audit["local_global_endpoint_max_gap_m"],
        "local_global_reference_warp_endpoint_max_gap_m": audit["local_global_reference_warp_endpoint_max_gap_m"],
        "max_bond_slip_m": audit["max_bond_slip_m"],
        "figures": [str(pdf.relative_to(ROOT)), str(png.relative_to(ROOT))],
    }
    out_json = output_dir / f"{args.prefix}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
