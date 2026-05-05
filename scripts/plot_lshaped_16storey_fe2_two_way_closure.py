#!/usr/bin/env python3
"""Compare the stabilized FE2 two-way gate against the single-scale structure.

The FE2 run currently closes a short post-activation window.  This script keeps
the comparison on that common interval and records both visual evidence and
simple interpolation-based error metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_table(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        rows: list[dict[str, float]] = []
        for row in csv.DictReader(f):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key is None or value in (None, ""):
                    continue
                try:
                    parsed[key.strip()] = float(value)
                except ValueError:
                    pass
            if "time" in parsed:
                rows.append(parsed)
    if not rows:
        raise ValueError(f"no valid numeric rows in {path}")
    rows.sort(key=lambda r: r["time"])
    return rows


def read_raw_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def window(rows: list[dict[str, float]], t0: float, t1: float) -> list[dict[str, float]]:
    return [r for r in rows if t0 - 1.0e-12 <= r["time"] <= t1 + 1.0e-12]


def interp(rows: list[dict[str, float]], key: str, t: float) -> float:
    if t <= rows[0]["time"]:
        return rows[0][key]
    if t >= rows[-1]["time"]:
        return rows[-1][key]
    lo = 0
    hi = len(rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rows[mid]["time"] <= t:
            lo = mid
        else:
            hi = mid
    t0 = rows[lo]["time"]
    t1 = rows[hi]["time"]
    y0 = rows[lo][key]
    y1 = rows[hi][key]
    if abs(t1 - t0) < 1.0e-15:
        return y0
    alpha = (t - t0) / (t1 - t0)
    return (1.0 - alpha) * y0 + alpha * y1


def detect_roof_node(rows: list[dict[str, float]], requested: str) -> str:
    keys = set(rows[0].keys())
    if requested != "auto":
        required = {f"{requested}_dof0", f"{requested}_dof1", f"{requested}_dof2"}
        missing = required - keys
        if missing:
            raise KeyError(f"missing requested roof node columns: {sorted(missing)}")
        return requested

    best_node = ""
    best_norm = -1.0
    last = rows[-1]
    for key in keys:
        if not key.endswith("_dof0"):
            continue
        node = key[:-5]
        kx, ky, kz = f"{node}_dof0", f"{node}_dof1", f"{node}_dof2"
        if {kx, ky, kz}.issubset(keys):
            norm = math.sqrt(last[kx] ** 2 + last[ky] ** 2 + last[kz] ** 2)
            if norm > best_norm:
                best_norm = norm
                best_node = node
    if not best_node:
        raise KeyError("could not infer a roof node with dof0/dof1/dof2 columns")
    return best_node


def extract_node(rows: list[dict[str, float]], node: str) -> dict[str, list[float]]:
    return {
        "time": [r["time"] for r in rows],
        "ux": [r[f"{node}_dof0"] for r in rows],
        "uy": [r[f"{node}_dof1"] for r in rows],
        "uz": [r[f"{node}_dof2"] for r in rows],
    }


def peak_abs(values: list[float]) -> float:
    return max((abs(v) for v in values), default=0.0)


def metrics_at_fe2_times(
    structural_rows: list[dict[str, float]],
    fe2_rows: list[dict[str, float]],
    keys: list[str],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        diffs: list[float] = []
        refs: list[float] = []
        for row in fe2_rows:
            t = row["time"]
            ref = interp(structural_rows, key, t)
            diffs.append(row[key] - ref)
            refs.append(ref)
        rms = math.sqrt(sum(d * d for d in diffs) / max(len(diffs), 1))
        ref_peak = max((abs(v) for v in refs), default=0.0)
        out[key] = {
            "rms_abs": rms,
            "max_abs": max((abs(d) for d in diffs), default=0.0),
            "relative_to_reference_peak": rms / ref_peak if ref_peak > 0 else math.nan,
            "final_difference": diffs[-1] if diffs else math.nan,
        }
    return out


def plot_roof_components(
    structural: dict[str, list[float]],
    fe2: dict[str, list[float]],
    out_dir: Path,
    prefix: str,
    node: str,
) -> list[str]:
    figures: list[str] = []
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.0), sharex=True)
    labels = {
        "ux": r"$u_x$ [m]",
        "uy": r"$u_y$ [m]",
        "uz": r"$u_z$ [m]",
    }
    for ax, comp in zip(axes, ("ux", "uy", "uz")):
        ax.plot(
            structural["time"],
            structural[comp],
            color="#1f5a99",
            linewidth=1.35,
            label="fall_n estructural puro",
        )
        ax.plot(
            fe2["time"],
            fe2[comp],
            color="#b84a39",
            linewidth=1.45,
            marker="o",
            markersize=3.0,
            label="fall_n FE2 two-way",
        )
        ax.set_ylabel(labels[comp])
    axes[-1].set_xlabel("tiempo [s]")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Respuesta de cubierta: {node}")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"{prefix}_roof_components.{ext}"
        fig.savefig(path)
        figures.append(str(path))
    plt.close(fig)
    return figures


def plot_roof_orbit(
    structural: dict[str, list[float]],
    fe2: dict[str, list[float]],
    out_dir: Path,
    prefix: str,
    node: str,
) -> list[str]:
    figures: list[str] = []
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot(
        structural["ux"],
        structural["uy"],
        color="#1f5a99",
        linewidth=1.35,
        label="fall_n estructural puro",
    )
    ax.plot(
        fe2["ux"],
        fe2["uy"],
        color="#b84a39",
        linewidth=1.45,
        marker="o",
        markersize=3.0,
        label="fall_n FE2 two-way",
    )
    ax.scatter([structural["ux"][0]], [structural["uy"][0]], color="#1f5a99", s=20)
    ax.scatter([fe2["ux"][0]], [fe2["uy"][0]], color="#b84a39", s=20)
    ax.scatter([structural["ux"][-1]], [structural["uy"][-1]], color="#1f5a99", marker="x", s=30)
    ax.scatter([fe2["ux"][-1]], [fe2["uy"][-1]], color="#b84a39", marker="x", s=30)
    ax.set_xlabel(r"$u_x$ [m]")
    ax.set_ylabel(r"$u_y$ [m]")
    ax.set_title(f"Orbita de cubierta en planta: {node}")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"{prefix}_roof_plan_orbit.{ext}"
        fig.savefig(path)
        figures.append(str(path))
    plt.close(fig)
    return figures


def plot_force_components(
    structural_rows: list[dict[str, float]],
    fe2_rows: list[dict[str, float]],
    out_dir: Path,
    prefix: str,
    components: list[str],
) -> list[str]:
    figures: list[str] = []
    n = len(components)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.2, 2.35 * nrows), sharex=True)
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    for ax, comp in zip(axes_flat, components):
        ax.plot(
            [r["time"] for r in structural_rows],
            [r[comp] for r in structural_rows],
            color="#1f5a99",
            linewidth=1.25,
            label="fall_n estructural puro",
        )
        ax.plot(
            [r["time"] for r in fe2_rows],
            [r[comp] for r in fe2_rows],
            color="#b84a39",
            linewidth=1.35,
            marker="o",
            markersize=2.7,
            label="fall_n FE2 two-way",
        )
        ax.set_ylabel(comp)
    for ax in axes_flat[n:]:
        ax.axis("off")
    axes_flat[0].legend(loc="best", fontsize=8)
    for ax in axes_flat[-ncols:]:
        if ax.has_data():
            ax.set_xlabel("tiempo [s]")
    fig.suptitle("Elemento monitorizado 0: componentes de fuerza interna")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"{prefix}_selected_element_forces.{ext}"
        fig.savefig(path)
        figures.append(str(path))
    plt.close(fig)
    return figures


def plot_coupling_residuals(
    audit_rows: list[dict[str, float]],
    out_dir: Path,
    prefix: str,
) -> list[str]:
    if not audit_rows:
        return []
    figures: list[str] = []
    times = [r["time"] for r in audit_rows]
    series = [
        ("max_force_residual_rel", "residuo fuerza"),
        ("max_tangent_residual_rel", "residuo tangente"),
        ("max_tangent_column_residual_rel", "residuo columna tangente"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#7a3b86", "#c1762d", "#2e7d59"]
    for (key, label), color in zip(series, colors):
        if key not in audit_rows[0]:
            continue
        values = [max(r.get(key, 0.0), 1.0e-12) for r in audit_rows]
        ax.semilogy(times, values, color=color, linewidth=1.35, marker="o", markersize=2.4, label=label)
    ax.axhline(5.0e-2, color="#444444", linestyle="--", linewidth=0.9, label="umbral 5 %")
    ax.set_xlabel("tiempo [s]")
    ax.set_ylabel("residuo relativo [-]")
    ax.set_title("Cierre iterativo FE2 two-way")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"{prefix}_coupling_residuals.{ext}"
        fig.savefig(path)
        figures.append(str(path))
    plt.close(fig)
    return figures


def plot_crack_evolution(
    crack_rows: list[dict[str, float]],
    out_dir: Path,
    prefix: str,
) -> list[str]:
    if not crack_rows:
        return []
    figures: list[str] = []
    times = [r["time"] for r in crack_rows]
    if "max_damage_scalar" in crack_rows[0]:
        damage_key = "max_damage_scalar"
    elif "max_damage" in crack_rows[0]:
        damage_key = "max_damage"
    else:
        damage_key = "damage_max"
    if "total_cracks" in crack_rows[0]:
        cracked_key = "total_cracks"
    elif "active_cracks" in crack_rows[0]:
        cracked_key = "active_cracks"
    else:
        cracked_key = "cracked_points"
    fig, ax1 = plt.subplots(figsize=(8.2, 4.4))
    ax1.plot(
        times,
        [r.get(damage_key, 0.0) for r in crack_rows],
        color="#b84a39",
        linewidth=1.45,
        marker="o",
        markersize=2.4,
        label="dano maximo local",
    )
    ax1.set_xlabel("tiempo [s]")
    ax1.set_ylabel("dano maximo [-]", color="#b84a39")
    ax1.tick_params(axis="y", labelcolor="#b84a39")
    ax2 = ax1.twinx()
    ax2.plot(
        times,
        [r.get(cracked_key, 0.0) for r in crack_rows],
        color="#1f5a99",
        linewidth=1.25,
        marker="s",
        markersize=2.2,
        label="puntos fisurados",
    )
    ax2.set_ylabel("puntos/superficies fisuradas [-]", color="#1f5a99")
    ax2.tick_params(axis="y", labelcolor="#1f5a99")
    ax1.set_title("Evolucion local XFEM: dano y fisuracion")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"{prefix}_local_crack_evolution.{ext}"
        fig.savefig(path)
        figures.append(str(path))
    plt.close(fig)
    return figures


def summarize_vtk_index(raw_rows: list[dict[str, str]]) -> dict[str, object]:
    if not raw_rows:
        return {}
    role_counts: dict[str, int] = {}
    local_sites: set[str] = set()
    global_frames = 0
    local_vtk_files = 0
    global_vtk_files = 0
    t_min = math.inf
    t_max = -math.inf
    for row in raw_rows:
        role = row.get("role", "")
        role_counts[role] = role_counts.get(role, 0) + 1
        try:
            t = float(row.get("physical_time", "nan"))
            if not math.isnan(t):
                t_min = min(t_min, t)
                t_max = max(t_max, t)
        except ValueError:
            pass
        if row.get("local_site_index", ""):
            local_sites.add(row["local_site_index"])
        if row.get("global_vtk_path", ""):
            global_vtk_files += 1
        if row.get("local_vtk_path", ""):
            local_vtk_files += 1
        if role == "global_frame":
            global_frames += 1
    return {
        "rows": len(raw_rows),
        "role_counts": role_counts,
        "local_sites": sorted(local_sites),
        "global_frames": global_frames,
        "global_vtk_entries": global_vtk_files,
        "local_vtk_entries": local_vtk_files,
        "time_window_s": [t_min if t_min < math.inf else None, t_max if t_max > -math.inf else None],
    }


def max_series(rows: list[dict[str, float]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        values = [abs(r.get(key, 0.0)) for r in rows]
        out[key] = max(values, default=math.nan)
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--structural-roof",
        default=str(root / "data/output/stage_c_16storey/falln_n4_nonlinear_restart_alarm_to_5p4s_roof_displacement.csv"),
    )
    parser.add_argument(
        "--fe2-roof",
        default=str(root / "data/output/lshaped_multiscale_16/recorders/roof_displacement.csv"),
    )
    parser.add_argument(
        "--structural-force",
        default=str(root / "data/output/stage_c_16storey/falln_n4_nonlinear_restart_alarm_to_5p4s_selected_element_0_global_force.csv"),
    )
    parser.add_argument(
        "--fe2-force",
        default=str(root / "data/output/lshaped_multiscale_16/recorders/selected_element_0_global_force.csv"),
    )
    parser.add_argument(
        "--coupling-audit",
        default=str(root / "data/output/lshaped_multiscale_16/recorders/fe2_two_way_coupling_audit.csv"),
    )
    parser.add_argument(
        "--crack-evolution",
        default=str(root / "data/output/lshaped_multiscale_16/recorders/crack_evolution.csv"),
    )
    parser.add_argument(
        "--time-index",
        default=str(root / "data/output/lshaped_multiscale_16/recorders/multiscale_time_index.csv"),
    )
    parser.add_argument("--roof-node", default="node335")
    parser.add_argument(
        "--out-dir",
        default=str(root / "doc/figures/validation_reboot"),
    )
    parser.add_argument("--prefix", default="lshaped_16_fe2_two_way_closure")
    args = parser.parse_args()

    structural_roof_all = read_table(Path(args.structural_roof))
    fe2_roof = read_table(Path(args.fe2_roof))
    structural_force_all = read_table(Path(args.structural_force))
    fe2_force = read_table(Path(args.fe2_force))
    coupling_audit = read_table(Path(args.coupling_audit)) if Path(args.coupling_audit).exists() else []
    crack_evolution = read_table(Path(args.crack_evolution)) if Path(args.crack_evolution).exists() else []
    vtk_index_rows = read_raw_csv(Path(args.time_index))

    t0 = max(fe2_roof[0]["time"], fe2_force[0]["time"])
    t1 = min(fe2_roof[-1]["time"], fe2_force[-1]["time"])
    structural_roof = window(structural_roof_all, t0, t1)
    structural_force = window(structural_force_all, t0, t1)
    if not structural_roof or not structural_force:
        raise ValueError("single-scale structural reference does not overlap FE2 window")

    node = detect_roof_node(fe2_roof, args.roof_node)
    # Ensure the same node exists in the reference.
    detect_roof_node(structural_roof, node)

    roof_keys = [f"{node}_dof{i}" for i in range(3)]
    force_components = [f"f{i}" for i in range(6)]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    structural_node = extract_node(structural_roof, node)
    fe2_node = extract_node(fe2_roof, node)
    figures: list[str] = []
    figures += plot_roof_components(structural_node, fe2_node, out_dir, args.prefix, node)
    figures += plot_roof_orbit(structural_node, fe2_node, out_dir, args.prefix, node)
    figures += plot_force_components(
        structural_force,
        fe2_force,
        out_dir,
        args.prefix,
        force_components,
    )
    figures += plot_coupling_residuals(window(coupling_audit, t0, t1), out_dir, args.prefix)
    figures += plot_crack_evolution(window(crack_evolution, t0, t1), out_dir, args.prefix)

    summary = {
        "schema": "lshaped_16_fe2_two_way_closure_v1",
        "window_s": [t0, t1],
        "roof_node": node,
        "structural_roof_csv": str(Path(args.structural_roof)),
        "fe2_roof_csv": str(Path(args.fe2_roof)),
        "structural_force_csv": str(Path(args.structural_force)),
        "fe2_force_csv": str(Path(args.fe2_force)),
        "coupling_audit_csv": str(Path(args.coupling_audit)),
        "crack_evolution_csv": str(Path(args.crack_evolution)),
        "time_index_csv": str(Path(args.time_index)),
        "structural_roof_samples_in_window": len(structural_roof),
        "fe2_roof_samples": len(fe2_roof),
        "structural_force_samples_in_window": len(structural_force),
        "fe2_force_samples": len(fe2_force),
        "roof_metrics": metrics_at_fe2_times(structural_roof, fe2_roof, roof_keys),
        "force_metrics": metrics_at_fe2_times(structural_force, fe2_force, force_components),
        "coupling_audit_metrics": max_series(
            window(coupling_audit, t0, t1),
            [
                "max_force_residual_rel",
                "max_tangent_residual_rel",
                "max_tangent_column_residual_rel",
                "local_failed_solve_attempts",
            ],
        ) if coupling_audit else {},
        "vtk_index_summary": summarize_vtk_index(vtk_index_rows),
        "peaks": {
            "structural_roof": {k: peak_abs(structural_node[k]) for k in ("ux", "uy", "uz")},
            "fe2_roof": {k: peak_abs(fe2_node[k]) for k in ("ux", "uy", "uz")},
            "structural_force": {k: peak_abs([r[k] for r in structural_force]) for k in force_components},
            "fe2_force": {k: peak_abs([r[k] for r in fe2_force]) for k in force_components},
        },
        "figures": figures,
        "notes": [
            "Comparison is restricted to the current stabilized FE2 two-way window.",
            "Element force components are raw internal-force vector entries for monitored macro element 0; they are a parity diagnostic, not a final physical-resultant convention.",
        ],
    }
    summary_path = out_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
