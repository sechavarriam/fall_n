#!/usr/bin/env python3
"""Plot the L-shaped one-way XFEM macro-inferred cell-audit run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "data"
    / "output"
    / "lshaped_16storey_xfem_macro_inferred_cell_audit_10s_20260515"
)
DEFAULT_REFERENCE = (
    ROOT / "data" / "output" / "lshaped_16storey_global_only_timegrid_10s_20260511"
)
DEFAULT_OUT = ROOT / "doc" / "figures" / "validation_reboot"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def interpolate(times: list[float], values: list[float], query: float) -> float | None:
    if not times or query < times[0] or query > times[-1]:
        return None
    pos = bisect_right(times, query)
    if pos == 0:
        return values[0]
    if pos >= len(times):
        return values[-1]
    t0, t1 = times[pos - 1], times[pos]
    y0, y1 = values[pos - 1], values[pos]
    if abs(t1 - t0) <= 1.0e-15:
        return y0
    alpha = (query - t0) / (t1 - t0)
    return y0 + alpha * (y1 - y0)


def roof_envelope(rows: list[dict[str, str]]) -> tuple[list[float], list[float]]:
    if not rows:
        return [], []
    channels = [name for name in rows[0] if name != "time"]
    times: list[float] = []
    values: list[float] = []
    for row in rows:
        times.append(as_float(row, "time"))
        values.append(max(abs(as_float(row, name)) for name in channels))
    return times, values


def roof_error_series(
    run_rows: list[dict[str, str]],
    ref_rows: list[dict[str, str]],
) -> tuple[list[float], list[float]]:
    if not run_rows or not ref_rows:
        return [], []
    channels = [name for name in run_rows[0] if name != "time" and name in ref_rows[0]]
    ref_times = [as_float(row, "time") for row in ref_rows]
    ref_values = {
        name: [as_float(row, name) for row in ref_rows]
        for name in channels
    }
    times: list[float] = []
    errors: list[float] = []
    for row in run_rows:
        t = as_float(row, "time")
        max_error = 0.0
        usable = False
        for name in channels:
            ref = interpolate(ref_times, ref_values[name], t)
            if ref is None:
                continue
            usable = True
            max_error = max(max_error, abs(as_float(row, name) - ref))
        if usable:
            times.append(t)
            errors.append(max_error)
    return times, errors


def decimate(x: list[float], y: list[float], max_points: int = 2500) -> tuple[list[float], list[float]]:
    if len(x) <= max_points:
        return x, y
    stride = max(1, math.ceil(len(x) / max_points))
    return x[::stride], y[::stride]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--prefix", default="lshaped_16_xfem_macro_inferred_cell_audit_10s"
    )
    args = parser.parse_args()

    run = args.run_dir.resolve()
    ref = args.reference_dir.resolve()
    rec = run / "recorders"
    history = read_rows(rec / "global_history.csv")
    cracks = read_rows(rec / "crack_evolution.csv")
    cells = read_rows(rec / "xfem_enriched_cell_integration" / "site_0000.csv")
    sites = read_rows(rec / "local_macro_inferred_sites.csv")
    planes = read_rows(rec / "xfem_crack_plane_sequence.csv")
    vtk_audit_path = rec / "publication_vtk_audit.json"
    vtk_audit = (
        json.loads(vtk_audit_path.read_text(encoding="utf-8"))
        if vtk_audit_path.exists()
        else {}
    )
    run_roof = read_rows(rec / "roof_displacement.csv")
    ref_roof = read_rows(ref / "recorders" / "roof_displacement.csv")

    t_env, u_env = roof_envelope(run_roof)
    t_ref, u_ref = roof_envelope(ref_roof)
    t_err, roof_err = roof_error_series(run_roof, ref_roof)
    t_env, u_env = decimate(t_env, u_env)
    t_ref, u_ref = decimate(t_ref, u_ref)
    t_err, roof_err = decimate(t_err, roof_err)

    t_crack = [as_float(row, "time") for row in cracks]
    crack_count = [as_int(row, "total_cracks") for row in cracks]
    max_opening_mm = [1000.0 * as_float(row, "max_opening") for row in cracks]
    t_crack, crack_count = decimate(t_crack, crack_count)
    _, max_opening_mm = decimate([as_float(row, "time") for row in cracks], max_opening_mm)

    cell_id = [as_int(row, "element_id") for row in cells]
    pos_fraction = [
        as_float(row, "gauss_positive_volume")
        / max(as_float(row, "total_volume"), 1.0e-30)
        for row in cells
    ]
    interface_area = [as_float(row, "interface_area") for row in cells]
    status = [row.get("status", "") for row in cells]
    colors = ["#d62728" if s == "cut" else "#4c78a8" for s in status]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))

    ax = axes[0, 0]
    ax.plot(t_ref, u_ref, color="#888888", lw=0.9, label="global-only anterior")
    ax.plot(t_env, u_env, color="#1f77b4", lw=0.8, label="FE2 one-way XFEM")
    ax.set_title("Respuesta de techo")
    ax.set_xlabel("tiempo fisico (s)")
    ax.set_ylabel(r"$\max |u_{techo}|$ (m)")
    ax.grid(True, alpha=0.35, lw=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_err, [1000.0 * value for value in roof_err], color="#8c2d04", lw=0.8)
    ax.axhline(0.5, color="black", lw=0.7, ls="--", label="guarda 0.5 mm")
    ax.set_title("Contraste macro contra referencia antigua")
    ax.set_xlabel("tiempo fisico (s)")
    ax.set_ylabel(r"$\max |\Delta u_{techo}|$ (mm)")
    ax.grid(True, alpha=0.35, lw=0.3)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    ax.plot(t_crack, crack_count, color="#4c78a8", lw=0.8, label="fisuras")
    ax2 = ax.twinx()
    ax2.plot(t_crack, max_opening_mm, color="#d62728", lw=0.8, label="apertura")
    ax.set_title("Evolucion local XFEM")
    ax.set_xlabel("tiempo fisico (s)")
    ax.set_ylabel("registros de fisura")
    ax2.set_ylabel("apertura maxima (mm)")
    ax.grid(True, alpha=0.35, lw=0.3)

    ax = axes[1, 1]
    ax.bar(cell_id, pos_fraction, color=colors, width=0.8, alpha=0.75, label="fraccion positiva")
    ax2 = ax.twinx()
    ax2.plot(cell_id, interface_area, color="#111111", lw=0.9, marker=".", ms=3, label="area interfaz")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Auditoria de celdas enriquecidas")
    ax.set_xlabel("elemento local")
    ax.set_ylabel("fraccion de volumen positivo")
    ax2.set_ylabel(r"area de interfaz (m$^2$)")
    ax.grid(True, alpha=0.25, lw=0.3)

    fig.suptitle("FE2 one-way XFEM macro-inferred, 10 s, auditoria de celdas", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{args.prefix}.pdf"
    png = output_dir / f"{args.prefix}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    max_error_m = max(roof_err) if roof_err else 0.0
    max_opening = max(as_float(row, "max_opening") for row in cracks)
    max_opening_row = max(cracks, key=lambda row: as_float(row, "max_opening"))
    max_u_row = max(history, key=lambda row: as_float(row, "u_inf"))
    rel_errors = [
        abs(as_float(row, "positive_volume_relative_error"))
        for row in cells
    ]
    summary = {
        "schema": "fall_n_lshaped_xfem_macro_inferred_cell_audit_plot_v1",
        "run_dir": str(run.relative_to(ROOT) if run.is_relative_to(ROOT) else run),
        "reference_dir": str(ref.relative_to(ROOT) if ref.is_relative_to(ROOT) else ref),
        "figure_pdf": str(pdf.relative_to(ROOT)),
        "figure_png": str(png.relative_to(ROOT)),
        "t_final_s": as_float(history[-1], "time"),
        "accepted_steps": len(history) - 1,
        "u_inf_max_m": as_float(max_u_row, "u_inf"),
        "u_inf_max_time_s": as_float(max_u_row, "time"),
        "roof_error_vs_reference_max_m": max_error_m,
        "roof_error_vs_reference_rms_m": math.sqrt(
            sum(value * value for value in roof_err) / len(roof_err)
        )
        if roof_err
        else 0.0,
        "cracks_final": as_int(cracks[-1], "total_cracks"),
        "max_opening_m": max_opening,
        "max_opening_time_s": as_float(max_opening_row, "time"),
        "cell_audit": {
            "rows": len(cells),
            "cut_cells": sum(1 for row in cells if row.get("status") == "cut"),
            "uncut_cells": sum(1 for row in cells if row.get("status") == "uncut"),
            "max_positive_volume_relative_error": max(rel_errors) if rel_errors else 0.0,
            "interface_area_total": sum(interface_area),
        },
        "selected_site": sites[0] if sites else {},
        "plane_sequence": planes,
        "vtk_audit": {
            "issue_count": len(vtk_audit.get("issues", [])),
            "total_vtu_files": vtk_audit.get("total_vtu_files"),
            "visible_crack_cells": vtk_audit.get("visible_crack_cells"),
            "min_visible_crack_opening_m": vtk_audit.get("min_visible_crack_opening_m"),
            "local_global_endpoint_max_gap_m": vtk_audit.get(
                "local_global_endpoint_max_gap_m"
            ),
        },
    }
    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
