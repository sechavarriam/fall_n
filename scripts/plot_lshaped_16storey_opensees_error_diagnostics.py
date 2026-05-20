#!/usr/bin/env python3
"""Plot component-wise OpenSees versus fall_n roof-node residuals.

This is intentionally a small publication helper. It compares one roof point
only, using the same observable used by the dynamic OpenSees convergence audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_FALLN_NODE_ID = 329


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot OpenSees minus fall_n roof displacement residuals."
    )
    parser.add_argument(
        "--falln-reference",
        type=Path,
        default=repo
        / "data/output/lshaped_16storey_global_only_primary_nodal_fixeddt_10s_20260518/recorders/roof_displacement_global_reference.csv",
    )
    parser.add_argument(
        "--opensees-roof",
        type=Path,
        default=repo
        / "data/output/opensees_lshaped_16storey_nonlinear_convergence_materialmapped_10s_20260518/fs_ets008_shear050_pdelta_materialmapped_deepcutback_10s_publication/roof_displacement.csv",
    )
    parser.add_argument(
        "--falln-roof-node-id",
        type=int,
        default=DEFAULT_FALLN_NODE_ID,
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=repo
        / "doc/figures/validation_reboot/lshaped_16_opensees_nonlinear_convergence_materialmapped_10s_nodal_publication_error_diagnostics",
    )
    return parser.parse_args()


def read_falln(path: Path, node_id: int) -> dict[str, list[float]]:
    fields = [f"node{node_id}_dof{i}" for i in range(3)]
    out = {"time": [], "ux": [], "uy": [], "uz": []}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out["time"].append(float(row["time"]))
            out["ux"].append(float(row[fields[0]]))
            out["uy"].append(float(row[fields[1]]))
            out["uz"].append(float(row[fields[2]]))
    return out


def read_opensees(path: Path) -> dict[str, list[float]]:
    out = {"time": [], "ux": [], "uy": [], "uz": []}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out["time"].append(float(row["time"]))
            out["ux"].append(float(row["ux"]))
            out["uy"].append(float(row["uy"]))
            out["uz"].append(float(row["uz"]))
    return out


def interp(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    span = xs[hi] - xs[lo]
    if span <= 0.0:
        return ys[lo]
    a = (x - xs[lo]) / span
    return ys[lo] * (1.0 - a) + ys[hi] * a


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / max(1, len(values)))


def main() -> None:
    args = parse_args()
    falln = read_falln(args.falln_reference, args.falln_roof_node_id)
    opensees = read_opensees(args.opensees_roof)

    common_times = [
        t
        for t in opensees["time"]
        if falln["time"][0] <= t <= falln["time"][-1]
    ]
    components = ["ux", "uy", "uz"]
    labels = {"ux": "$u_x$", "uy": "$u_y$", "uz": "$u_z$"}
    colors = {"ux": "#1f77b4", "uy": "#d62728", "uz": "#2ca02c"}

    rows: list[dict[str, float]] = []
    stats: dict[str, dict[str, float]] = {}
    for t in common_times:
        row = {"time": t}
        for comp in components:
            f_val = interp(falln["time"], falln[comp], t)
            o_val = interp(opensees["time"], opensees[comp], t)
            row[f"falln_{comp}_m"] = f_val
            row[f"opensees_{comp}_m"] = o_val
            row[f"error_{comp}_m"] = o_val - f_val
        rows.append(row)

    for comp in components:
        errors = [row[f"error_{comp}_m"] for row in rows]
        falln_peak = max(abs(row[f"falln_{comp}_m"]) for row in rows)
        opensees_peak = max(abs(row[f"opensees_{comp}_m"]) for row in rows)
        stats[comp] = {
            "rms_error_m": rms(errors),
            "max_abs_error_m": max(abs(v) for v in errors),
            "falln_peak_abs_m": falln_peak,
            "opensees_peak_abs_m": opensees_peak,
            "normalized_rms_by_falln_peak": rms(errors) / falln_peak
            if falln_peak > 0.0
            else float("nan"),
        }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_prefix.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "schema": "lshaped_16_opensees_error_diagnostics_v1",
                "falln_reference": str(args.falln_reference),
                "opensees_roof": str(args.opensees_roof),
                "falln_roof_node_id": args.falln_roof_node_id,
                "comparison_samples": len(rows),
                "comparison_start_s": rows[0]["time"],
                "comparison_end_s": rows[-1]["time"],
                "stats": stats,
                "csv": str(csv_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.1), constrained_layout=True)
    time = [row["time"] for row in rows]
    for ax, comp in zip(axes.flat[:3], components):
        err_mm = [1000.0 * row[f"error_{comp}_m"] for row in rows]
        ax.plot(time, err_mm, color=colors[comp], linewidth=1.2)
        ax.axhline(0.0, color="0.25", linewidth=0.7)
        ax.set_title(f"Residual OpenSees - fall_n, {labels[comp]}")
        ax.set_xlabel("time in observation window [s]")
        ax.set_ylabel("residual [mm]")
        ax.text(
            0.02,
            0.94,
            (
                f"RMS = {1000.0 * stats[comp]['rms_error_m']:.1f} mm\n"
                f"max |e| = {1000.0 * stats[comp]['max_abs_error_m']:.1f} mm"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.85},
        )

    ax = axes.flat[3]
    x = range(len(components))
    ax.bar(
        [i - 0.18 for i in x],
        [stats[c]["normalized_rms_by_falln_peak"] for c in components],
        width=0.36,
        color=[colors[c] for c in components],
        alpha=0.82,
        label="RMS / fall_n peak",
    )
    ax.bar(
        [i + 0.18 for i in x],
        [
            stats[c]["opensees_peak_abs_m"] / stats[c]["falln_peak_abs_m"]
            for c in components
        ],
        width=0.36,
        color="0.55",
        alpha=0.55,
        label="OpenSees peak / fall_n peak",
    )
    ax.set_xticks(list(x), [labels[c] for c in components])
    ax.set_ylabel("dimensionless ratio")
    ax.set_title("Component metrics")
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle("Roof-node dynamic comparator diagnostics")

    for suffix in (".pdf", ".png"):
        fig.savefig(args.output_prefix.with_suffix(suffix), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
