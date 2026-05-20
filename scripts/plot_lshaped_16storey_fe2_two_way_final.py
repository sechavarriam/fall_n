#!/usr/bin/env python3
"""Publication plots for the final 10 s L-shaped FE2 two-way XFEM run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "data"
    / "output"
    / "lshaped_16storey_two_way_xfem_deep_regularized_hybrid_diag_1site_10s_20260515"
)
DEFAULT_REF = ROOT / "data" / "output" / "lshaped_16storey_global_only_timegrid_10s_20260511"
DEFAULT_OUT = ROOT / "doc" / "figures" / "validation_reboot"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8", errors="ignore") as fh:
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


def decimate_pairs(
    x: list[float],
    ys: Iterable[list[float]],
    max_points: int = 2600,
) -> tuple[list[float], list[list[float]]]:
    ys_list = [list(y) for y in ys]
    if len(x) <= max_points:
        return x, ys_list
    stride = max(1, math.ceil(len(x) / max_points))
    return x[::stride], [y[::stride] for y in ys_list]


def interp(times: list[float], values: list[float], query: float) -> float | None:
    if not times or query < times[0] or query > times[-1]:
        return None
    pos = bisect_right(times, query)
    if pos <= 0:
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
    channels = [name for name in rows[0] if name != "time"] if rows else []
    times: list[float] = []
    values: list[float] = []
    for row in rows:
        times.append(as_float(row, "time"))
        values.append(max(abs(as_float(row, name)) for name in channels))
    return times, values


def node_series(rows: list[dict[str, str]], node: str) -> dict[str, list[float]]:
    cols = [f"{node}_dof{i}" for i in range(3)]
    if rows and not all(col in rows[0] for col in cols):
        raise KeyError(f"missing roof node columns for {node}")
    return {
        "time": [as_float(row, "time") for row in rows],
        "ux": [as_float(row, cols[0]) for row in rows],
        "uy": [as_float(row, cols[1]) for row in rows],
        "uz": [as_float(row, cols[2]) for row in rows],
    }


def numeric_series(rows: list[dict[str, str]], key: str) -> list[float]:
    return [as_float(row, key) for row in rows]


def max_abs(values: Iterable[float]) -> float:
    return max((abs(v) for v in values), default=0.0)


def rms(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def compare_series(
    query_rows: list[dict[str, str]],
    ref_rows: list[dict[str, str]],
    keys: list[str],
) -> dict[str, dict[str, float]]:
    ref_times = [as_float(row, "time") for row in ref_rows]
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        ref_values = [as_float(row, key) for row in ref_rows]
        diffs: list[float] = []
        refs: list[float] = []
        for row in query_rows:
            t = as_float(row, "time")
            ref = interp(ref_times, ref_values, t)
            if ref is None:
                continue
            diffs.append(as_float(row, key) - ref)
            refs.append(ref)
        peak_ref = max_abs(refs)
        out[key] = {
            "max_abs": max_abs(diffs),
            "rms": rms(diffs),
            "relative_rms_to_ref_peak": rms(diffs) / peak_ref if peak_ref else math.nan,
            "final_difference": diffs[-1] if diffs else math.nan,
        }
    return out


def regime_spans(rows: list[dict[str, str]]) -> list[tuple[float, float, str]]:
    if not rows:
        return []
    spans: list[tuple[float, float, str]] = []
    start = as_float(rows[0], "time")
    last_t = start
    current = rows[0].get("coupling_regime", "")
    for row in rows[1:]:
        t = as_float(row, "time")
        regime = row.get("coupling_regime", "")
        if regime != current:
            spans.append((start, last_t, current))
            start = t
            current = regime
        last_t = t
    spans.append((start, last_t, current))
    return spans


def terminal_strict_window(rows: list[dict[str, str]]) -> tuple[float | None, float | None, int]:
    if not rows:
        return None, None, 0
    count = 0
    start: float | None = None
    end = as_float(rows[-1], "time")
    for row in reversed(rows):
        if row.get("coupling_regime", "") != "strict_two_way":
            break
        start = as_float(row, "time")
        count += 1
    return start, end, count


def work_proxy(x: list[float], force: list[float]) -> float:
    total = 0.0
    for i in range(1, min(len(x), len(force))):
        total += 0.5 * (force[i] + force[i - 1]) * (x[i] - x[i - 1])
    return total


def plot_regime_background(ax, spans: list[tuple[float, float, str]]) -> None:
    colors = {
        "strict_two_way": "#2f855a",
        "hybrid_observation_window": "#f2c94c",
    }
    labels_seen: set[str] = set()
    for start, end, regime in spans:
        color = colors.get(regime)
        if not color:
            continue
        label = regime.replace("_", " ") if regime not in labels_seen else None
        ax.axvspan(start, end, color=color, alpha=0.13, lw=0, label=label)
        labels_seen.add(regime)


def save(fig, out_dir: Path, name: str) -> list[str]:
    paths: list[str] = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{name}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=180)
        else:
            fig.savefig(path)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prefix", default="lshaped_16_fe2_two_way_deep_regularized_final_10s")
    parser.add_argument("--roof-node", default="node335")
    args = parser.parse_args()

    run = args.run_dir.resolve()
    ref = args.reference_dir.resolve()
    rec = run / "recorders"
    ref_rec = ref / "recorders"
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    history = read_rows(rec / "global_history.csv")
    roof = read_rows(rec / "roof_displacement.csv")
    force = read_rows(rec / "selected_element_0_global_force.csv")
    coupling = read_rows(rec / "fe2_two_way_coupling_audit.csv")
    site = read_rows(rec / "fe2_two_way_site_response_audit.csv")
    cracks = read_rows(rec / "crack_evolution.csv")
    cells = read_rows(rec / "xfem_enriched_cell_integration" / "site_0000.csv")
    planes = read_rows(rec / "xfem_crack_plane_sequence.csv")
    vtk_path = rec / "publication_vtk_audit.json"
    vtk = json.loads(vtk_path.read_text(encoding="utf-8")) if vtk_path.exists() else {}
    ref_roof = read_rows(ref_rec / "roof_displacement.csv")
    ref_force = read_rows(ref_rec / "selected_element_0_global_force.csv")

    plt.rcParams.update({
        "font.family": "serif",
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.26,
        "grid.linewidth": 0.35,
    })

    t_env, u_env = roof_envelope(roof)
    tr_env, ur_env = roof_envelope(ref_roof)
    node = node_series(roof, args.roof_node)
    ref_node = node_series(ref_roof, args.roof_node)
    spans = regime_spans(coupling)

    # Common decimated traces.
    t_env_d, (u_env_d,) = decimate_pairs(t_env, [u_env])
    tr_env_d, (ur_env_d,) = decimate_pairs(tr_env, [ur_env])
    t_node_d, (ux_d, uy_d, uz_d) = decimate_pairs(
        node["time"], [node["ux"], node["uy"], node["uz"]]
    )
    tr_node_d, (rux_d, ruy_d, ruz_d) = decimate_pairs(
        ref_node["time"], [ref_node["ux"], ref_node["uy"], ref_node["uz"]]
    )
    t_c = numeric_series(coupling, "time")
    rf = numeric_series(coupling, "max_force_residual_rel")
    rfc = numeric_series(coupling, "max_force_component_residual_rel")
    rt = numeric_series(coupling, "max_tangent_residual_rel")
    rtc = numeric_series(coupling, "max_tangent_column_residual_rel")
    wg = numeric_series(coupling, "work_gap")
    t_c_d, (rf_d, rfc_d, rt_d, rtc_d, wg_d) = decimate_pairs(
        t_c, [rf, rfc, rt, rtc, wg]
    )
    t_cr = numeric_series(cracks, "time")
    crack_count = numeric_series(cracks, "total_cracks")
    opening_mm = [1000.0 * as_float(row, "max_opening") for row in cracks]
    damage = numeric_series(cracks, "max_damage_scalar")
    t_cr_d, (crack_count_d, opening_mm_d, damage_d) = decimate_pairs(
        t_cr, [crack_count, opening_mm, damage]
    )

    # Hysteresis proxy uses monitored element component f2 against roof y.
    force_times = numeric_series(force, "time")
    f2 = numeric_series(force, "f2")
    f4 = numeric_series(force, "f4")
    uy_at_force = [
        interp(node["time"], node["uy"], t) or node["uy"][0]
        for t in force_times
    ]
    ux_at_force = [
        interp(node["time"], node["ux"], t) or node["ux"][0]
        for t in force_times
    ]
    ref_force_times = numeric_series(ref_force, "time")
    ref_f2 = numeric_series(ref_force, "f2")
    ref_f4 = numeric_series(ref_force, "f4")
    ref_uy_at_force = [
        interp(ref_node["time"], ref_node["uy"], t) or ref_node["uy"][0]
        for t in ref_force_times
    ]
    ref_ux_at_force = [
        interp(ref_node["time"], ref_node["ux"], t) or ref_node["ux"][0]
        for t in ref_force_times
    ]
    uy_dh, (f2_d, ) = decimate_pairs(uy_at_force, [f2])
    ruy_dh, (rf2_d, ) = decimate_pairs(ref_uy_at_force, [ref_f2])
    ux_dh, (f4_d, ) = decimate_pairs(ux_at_force, [f4])
    rux_dh, (rf4_d, ) = decimate_pairs(ref_ux_at_force, [ref_f4])

    figures: list[str] = []

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.8))
    ax = axes[0, 0]
    ax.plot(tr_env_d, ur_env_d, color="#747474", lw=0.95, label="global-only")
    ax.plot(t_env_d, u_env_d, color="#1f5a99", lw=0.9, label="FE2 two-way XFEM")
    ax.set_title("Envolvente de techo")
    ax.set_xlabel("tiempo [s]")
    ax.set_ylabel(r"$\max |u_{techo}|$ [m]")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.plot(rux_d, ruy_d, color="#747474", lw=0.9, label="global-only")
    ax.plot(ux_d, uy_d, color="#b84a39", lw=0.9, label="FE2 two-way")
    ax.set_title("Orbita de techo")
    ax.set_xlabel(r"$u_x$ [m]")
    ax.set_ylabel(r"$u_y$ [m]")
    ax.axis("equal")

    ax = axes[0, 2]
    ax.plot(ruy_dh, rf2_d, color="#747474", lw=0.9, label="global-only")
    ax.plot(uy_dh, f2_d, color="#7a3b86", lw=0.9, label="FE2 two-way")
    ax.set_title("Lazo monitorizado")
    ax.set_xlabel(r"$u_y$ techo [m]")
    ax.set_ylabel(r"$f_2$ elemento 0")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    plot_regime_background(ax, spans)
    ax.semilogy(t_c_d, [max(v, 1e-12) for v in rf_d], color="#7a3b86", lw=0.8, label=r"$r_F$")
    ax.semilogy(t_c_d, [max(v, 1e-12) for v in rfc_d], color="#1f5a99", lw=0.8, label=r"$r_{F,c}$")
    ax.semilogy(t_c_d, [max(v, 1e-12) for v in rt_d], color="#c1762d", lw=0.8, label=r"$r_D$")
    ax.semilogy(t_c_d, [max(v, 1e-12) for v in rtc_d], color="#2e7d59", lw=0.8, label=r"$r_{D,c}$")
    ax.axhline(5e-2, color="#333333", ls="--", lw=0.8, label="0.05 ref.")
    ax.axhline(5.5e-1, color="#777777", ls=":", lw=0.7, label="0.55 col.")
    ax.set_title("Compuertas de acople")
    ax.set_xlabel("tiempo [s]")
    ax.set_ylabel("residuo relativo")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1, 1]
    ax.plot(t_cr_d, opening_mm_d, color="#b84a39", lw=0.9, label="apertura max.")
    ax2 = ax.twinx()
    ax2.plot(t_cr_d, crack_count_d, color="#1f5a99", lw=0.8, label="fisuras")
    ax.set_title("Respuesta local XFEM")
    ax.set_xlabel("tiempo [s]")
    ax.set_ylabel("apertura [mm]", color="#b84a39")
    ax2.set_ylabel("registros de fisura", color="#1f5a99")

    ax = axes[1, 2]
    plot_regime_background(ax, spans)
    ax.plot(t_c_d, wg_d, color="#5b4b8a", lw=0.9, label="work gap")
    ax.axhline(5e-2, color="#333333", ls="--", lw=0.8, label="0.05 ref.")
    ax.axhline(3e-1, color="#777777", ls=":", lw=0.7, label="0.30 fuerza")
    ax.set_title("Regimen y brecha de trabajo")
    ax.set_xlabel("tiempo [s]")
    ax.set_ylabel("brecha [-]")
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("FE2 two-way XFEM final, 10 s, edificio L de 16 pisos", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figures += save(fig, out_dir, f"{args.prefix}_summary")

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.2), sharex=True)
    comps = [("ux", r"$u_x$ [m]"), ("uy", r"$u_y$ [m]"), ("uz", r"$u_z$ [m]")]
    ref_dec = {"ux": rux_d, "uy": ruy_d, "uz": ruz_d}
    fe2_dec = {"ux": ux_d, "uy": uy_d, "uz": uz_d}
    for ax, (comp, label) in zip(axes, comps):
        ax.plot(tr_node_d, ref_dec[comp], color="#747474", lw=0.9, label="global-only")
        ax.plot(t_node_d, fe2_dec[comp], color="#1f5a99", lw=0.9, label="FE2 two-way")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("tiempo [s]")
    fig.suptitle(f"Componentes de techo, {args.roof_node}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figures += save(fig, out_dir, f"{args.prefix}_roof_components")

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot(rux_d, ruy_d, color="#747474", lw=0.9, label="global-only")
    ax.plot(ux_d, uy_d, color="#b84a39", lw=0.9, label="FE2 two-way")
    ax.scatter([ux_d[0], ux_d[-1]], [uy_d[0], uy_d[-1]], color="#b84a39", s=24)
    ax.set_xlabel(r"$u_x$ [m]")
    ax.set_ylabel(r"$u_y$ [m]")
    ax.set_title(f"Orbita de techo, {args.roof_node}")
    ax.axis("equal")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    figures += save(fig, out_dir, f"{args.prefix}_roof_orbit")

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    axes[0].plot(ruy_dh, rf2_d, color="#747474", lw=0.9, label="global-only")
    axes[0].plot(uy_dh, f2_d, color="#7a3b86", lw=0.9, label="FE2 two-way")
    axes[0].set_xlabel(r"$u_y$ techo [m]")
    axes[0].set_ylabel(r"$f_2$ elemento 0")
    axes[0].set_title(r"Lazo \(f_2-u_y\)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(rux_dh, rf4_d, color="#747474", lw=0.9, label="global-only")
    axes[1].plot(ux_dh, f4_d, color="#2e7d59", lw=0.9, label="FE2 two-way")
    axes[1].set_xlabel(r"$u_x$ techo [m]")
    axes[1].set_ylabel(r"$f_4$ elemento 0")
    axes[1].set_title(r"Lazo \(f_4-u_x\)")
    fig.suptitle("Histeresis monitorizada con fuerzas internas crudas")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figures += save(fig, out_dir, f"{args.prefix}_monitored_hysteresis")

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    plot_regime_background(ax, spans)
    ax.semilogy(t_c_d, [max(v, 1e-12) for v in rf_d], color="#7a3b86", lw=0.85, label=r"$r_F$")
    ax.semilogy(t_c_d, [max(v, 1e-12) for v in rfc_d], color="#1f5a99", lw=0.85, label=r"$r_{F,c}$")
    ax.semilogy(t_c_d, [max(v, 1e-12) for v in rt_d], color="#c1762d", lw=0.85, label=r"$r_D$")
    ax.semilogy(t_c_d, [max(v, 1e-12) for v in rtc_d], color="#2e7d59", lw=0.85, label=r"$r_{D,c}$")
    ax.semilogy(t_c_d, [max(v, 1e-12) for v in wg_d], color="#5b4b8a", lw=0.75, label="work gap")
    ax.axhline(5e-2, color="#333333", ls="--", lw=0.8, label="0.05 ref.")
    ax.axhline(5.5e-1, color="#777777", ls=":", lw=0.7, label="0.55 col.")
    ax.set_xlabel("tiempo [s]")
    ax.set_ylabel("residuo relativo")
    ax.set_title("Auditoria de acople y retorno a strict two-way")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    figures += save(fig, out_dir, f"{args.prefix}_coupling_gates")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    axes[0].plot(t_cr_d, opening_mm_d, color="#b84a39", lw=0.9, label="apertura max.")
    axes[0].set_xlabel("tiempo [s]")
    axes[0].set_ylabel("apertura [mm]")
    axes[0].set_title("Apertura XFEM")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(t_cr_d, crack_count_d, color="#1f5a99", lw=0.8, label="fisuras")
    axes[1].plot(t_cr_d, damage_d, color="#b84a39", lw=0.8, label="dano max.")
    axes[1].set_xlabel("tiempo [s]")
    axes[1].set_ylabel("conteo / dano [-]")
    axes[1].set_title("Fisuracion y dano")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Evolucion local XFEM del sitio promovido")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figures += save(fig, out_dir, f"{args.prefix}_local_xfem_evolution")

    last_history = history[-1]
    max_u_row = max(history, key=lambda r: as_float(r, "u_inf"))
    last_coupling = coupling[-1]
    strict_start, strict_end, strict_count = terminal_strict_window(coupling)
    regime_counts: dict[str, int] = {}
    for row in coupling:
        regime = row.get("coupling_regime", "")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
    cut_cells = sum(1 for row in cells if row.get("status") == "cut")
    max_cell_error = max(
        (as_float(row, "positive_volume_relative_error") for row in cells),
        default=0.0,
    )
    interface_total = sum(as_float(row, "interface_area") for row in cells)
    last_crack = cracks[-1]
    max_opening_row = max(cracks, key=lambda r: as_float(r, "max_opening"))
    max_force_residual = max(as_float(row, "max_force_residual_rel") for row in coupling)
    max_force_component_residual = max(as_float(row, "max_force_component_residual_rel") for row in coupling)
    max_tangent_residual = max(as_float(row, "max_tangent_residual_rel") for row in coupling)
    max_tangent_column_residual = max(as_float(row, "max_tangent_column_residual_rel") for row in coupling)
    strict_rows = [row for row in coupling if row.get("coupling_regime") == "strict_two_way"]
    terminal_strict_rows = coupling[-strict_count:] if strict_count else []
    max_strict_force_residual = max((as_float(row, "max_force_residual_rel") for row in strict_rows), default=0.0)
    max_strict_component_residual = max((as_float(row, "max_force_component_residual_rel") for row in strict_rows), default=0.0)
    max_strict_tangent_residual = max((as_float(row, "max_tangent_residual_rel") for row in strict_rows), default=0.0)
    max_strict_tangent_column_residual = max((as_float(row, "max_tangent_column_residual_rel") for row in strict_rows), default=0.0)
    max_terminal_force_residual = max(
        (as_float(row, "max_force_residual_rel") for row in terminal_strict_rows),
        default=0.0,
    )
    max_terminal_component_residual = max(
        (as_float(row, "max_force_component_residual_rel") for row in terminal_strict_rows),
        default=0.0,
    )
    max_terminal_tangent_residual = max(
        (as_float(row, "max_tangent_residual_rel") for row in terminal_strict_rows),
        default=0.0,
    )
    max_terminal_tangent_column_residual = max(
        (as_float(row, "max_tangent_column_residual_rel") for row in terminal_strict_rows),
        default=0.0,
    )
    force_keys = [f"f{i}" for i in range(6)]
    roof_keys = [f"{args.roof_node}_dof{i}" for i in range(3)]

    summary = {
        "schema": "fall_n_lshaped_16_fe2_two_way_final_v1",
        "run_dir": str(run),
        "reference_dir": str(ref),
        "output_prefix": args.prefix,
        "figures": figures,
        "global": {
            "t_final_s": as_float(last_history, "time"),
            "accepted_steps": as_int(last_history, "step"),
            "u_inf_final_m": as_float(last_history, "u_inf"),
            "u_inf_max_m": as_float(max_u_row, "u_inf"),
            "u_inf_max_time_s": as_float(max_u_row, "time"),
            "peak_damage_final": as_float(last_history, "peak_damage"),
        },
        "roof_node": {
            "node": args.roof_node,
            "comparison_to_global_only": compare_series(roof, ref_roof, roof_keys),
            "peak_abs_fe2_m": {
                "ux": max_abs(node["ux"]),
                "uy": max_abs(node["uy"]),
                "uz": max_abs(node["uz"]),
            },
            "final_fe2_m": {
                "ux": node["ux"][-1],
                "uy": node["uy"][-1],
                "uz": node["uz"][-1],
            },
        },
        "monitored_element": {
            "force_comparison_to_global_only": compare_series(force, ref_force, force_keys),
            "work_proxy_f2_uy_fe2": work_proxy(uy_at_force, f2),
            "work_proxy_f2_uy_global_only": work_proxy(ref_uy_at_force, ref_f2),
            "work_proxy_f4_ux_fe2": work_proxy(ux_at_force, f4),
            "work_proxy_f4_ux_global_only": work_proxy(ref_ux_at_force, ref_f4),
        },
        "coupling": {
            "launch_tolerances_declared": {
                "force": 0.30,
                "force_component": 0.30,
                "tangent": 0.07,
                "tangent_column": 0.55,
            },
            "rows": len(coupling),
            "regime_counts": regime_counts,
            "terminal_strict_start_s": strict_start,
            "terminal_strict_end_s": strict_end,
            "terminal_strict_rows": strict_count,
            "terminal_strict_duration_s": (
                strict_end - strict_start if strict_start is not None and strict_end is not None else None
            ),
            "last_regime": last_coupling.get("coupling_regime", ""),
            "last_return_gate_passed": as_int(last_coupling, "return_gate_passed"),
            "last_work_gap": as_float(last_coupling, "work_gap"),
            "last_force_residual_rel": as_float(last_coupling, "max_force_residual_rel"),
            "last_force_component_residual_rel": as_float(last_coupling, "max_force_component_residual_rel"),
            "last_tangent_residual_rel": as_float(last_coupling, "max_tangent_residual_rel"),
            "last_tangent_column_residual_rel": as_float(last_coupling, "max_tangent_column_residual_rel"),
            "max_force_residual_rel_all": max_force_residual,
            "max_force_component_residual_rel_all": max_force_component_residual,
            "max_tangent_residual_rel_all": max_tangent_residual,
            "max_tangent_column_residual_rel_all": max_tangent_column_residual,
            "max_force_residual_rel_strict": max_strict_force_residual,
            "max_force_component_residual_rel_strict": max_strict_component_residual,
            "max_tangent_residual_rel_strict": max_strict_tangent_residual,
            "max_tangent_column_residual_rel_strict": max_strict_tangent_column_residual,
            "max_force_residual_rel_terminal_strict": max_terminal_force_residual,
            "max_force_component_residual_rel_terminal_strict": max_terminal_component_residual,
            "max_tangent_residual_rel_terminal_strict": max_terminal_tangent_residual,
            "max_tangent_column_residual_rel_terminal_strict": max_terminal_tangent_column_residual,
        },
        "xfem": {
            "final_cracks": as_int(last_crack, "total_cracks"),
            "final_cracked_gps": as_int(last_crack, "total_cracked_gps"),
            "final_max_opening_m": as_float(last_crack, "max_opening"),
            "max_opening_m": as_float(max_opening_row, "max_opening"),
            "max_opening_time_s": as_float(max_opening_row, "time"),
            "final_damage_scalar": as_float(last_crack, "max_damage_scalar"),
            "plane_sequence": planes,
            "cell_audit": {
                "rows": len(cells),
                "cut_cells": cut_cells,
                "uncut_cells": len(cells) - cut_cells,
                "interface_area_total_m2": interface_total,
                "max_relative_volume_error": max_cell_error,
            },
        },
        "vtk_audit": {
            "issue_count": len(vtk.get("issues", [])),
            "total_vtu_files": vtk.get("total_vtu_files"),
            "visible_crack_cells": vtk.get("visible_crack_cells"),
            "min_visible_crack_opening_m": vtk.get("min_visible_crack_opening_m"),
            "local_global_endpoint_max_gap_m": vtk.get("local_global_endpoint_max_gap_m"),
        },
    }
    summary_path = out_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
