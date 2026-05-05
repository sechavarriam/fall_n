#!/usr/bin/env python3
"""Plot promoted elastic/nonlinear single-scale comparators over the 10 s window.

The script is intentionally tolerant of long-running cases: missing or partial
CSV files are reported in the JSON summary instead of aborting the whole plot.
This lets us update the same figure set as OpenSees/fall_n long runs finish.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_roof(path: Path, label: str, *, time_limit: float) -> tuple[dict | None, str | None]:
    if not path.exists():
        return None, f"missing: {path}"
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, f"empty: {path}"

    columns = list(rows[0].keys())
    if {"ux", "uy", "uz"}.issubset(columns):
        comp_cols = ("ux", "uy", "uz")
    else:
        comp_cols = tuple(columns[-3:])

    data_rows = []
    for row in rows:
        try:
            t = float(row["time"])
            if t > time_limit + 1.0e-12:
                continue
            data_rows.append({
                "time": t,
                "ux": float(row[comp_cols[0]]),
                "uy": float(row[comp_cols[1]]),
                "uz": float(row[comp_cols[2]]),
            })
        except (KeyError, ValueError):
            continue
    if not data_rows:
        return None, f"no valid rows: {path}"

    return {
        "label": label,
        "path": str(path),
        "time": [r["time"] for r in data_rows],
        "ux": [r["ux"] for r in data_rows],
        "uy": [r["uy"] for r in data_rows],
        "uz": [r["uz"] for r in data_rows],
        "component_columns": comp_cols,
    }, None


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def summarize(case: dict, manifest: dict) -> dict:
    row = {
        "label": case["label"],
        "path": case["path"],
        "samples": len(case["time"]),
        "t_start_s": case["time"][0],
        "t_end_s": case["time"][-1],
        "component_columns": list(case["component_columns"]),
    }
    for comp in ("ux", "uy", "uz"):
        vals = case[comp]
        row[f"peak_abs_{comp}_m"] = max(abs(v) for v in vals)
        row[f"final_{comp}_m"] = vals[-1]
    for key in (
        "status",
        "accepted_steps",
        "requested_steps",
        "beam_element_family",
        "concrete_model",
        "case_kind",
        "macro_element_family",
        "structural_mass_policy",
        "duration_s",
        "first_yield_time_s",
        "peak_damage_index",
        "total_wall_seconds",
    ):
        if key in manifest:
            row[key] = manifest[key]
    return row


def plot_cases(cases: list[dict], out_dir: Path, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    figures: list[str] = []

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.6), sharex=True)
    for ax, comp, ylabel in zip(
        axes,
        ("ux", "uy", "uz"),
        (r"$u_x$ [m]", r"$u_y$ [m]", r"$u_z$ [m]"),
    ):
        for case in cases:
            ax.plot(case["time"], case[comp], linewidth=1.25, label=case["label"])
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("tiempo relativo en la ventana [s]")
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle("Casos promovidos elastico/inelastico: respuesta de cubierta")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"{prefix}_time_components.{ext}"
        fig.savefig(path)
        figures.append(str(path))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for case in cases:
        ax.plot(case["ux"], case["uy"], linewidth=1.25, label=case["label"])
        ax.scatter(case["ux"][0], case["uy"][0], s=14)
        ax.scatter(case["ux"][-1], case["uy"][-1], s=22, marker="x")
    ax.set_xlabel(r"$u_x$ [m]")
    ax.set_ylabel(r"$u_y$ [m]")
    ax.set_title("Orbita de cubierta en planta")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"{prefix}_plan_orbit.{ext}"
        fig.savefig(path)
        figures.append(str(path))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    labels = [case["label"] for case in cases]
    x = range(len(cases))
    width = 0.25
    for offset, comp in zip((-width, 0.0, width), ("ux", "uy", "uz")):
        peaks = [max(abs(v) for v in case[comp]) for case in cases]
        ax.bar([i + offset for i in x], peaks, width=width, label=comp)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("pico absoluto [m]")
    ax.set_title("Picos de desplazamiento de cubierta")
    ax.legend(loc="best")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = out_dir / f"{prefix}_peak_components.{ext}"
        fig.savefig(path)
        figures.append(str(path))
    plt.close(fig)
    return figures


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--out-dir", default=str(root / "doc/figures/validation_reboot"))
    parser.add_argument("--prefix", default="lshaped_16_promoted_elastic_inelastic_10s")
    parser.add_argument(
        "--falln-elastic",
        default=str(root / "data/output/stage_c_16storey/falln_n4_newmark_linear_primary_nodal_10s_roof_displacement.csv"),
    )
    parser.add_argument(
        "--falln-elastic-manifest",
        default=str(root / "data/output/stage_c_16storey/falln_n4_newmark_linear_primary_nodal_10s_summary.json"),
    )
    parser.add_argument(
        "--opensees-elastic",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_elastic_timoshenko_sub1_nodalmass_current/roof_displacement.csv"),
    )
    parser.add_argument(
        "--opensees-elastic-manifest",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_elastic_timoshenko_sub1_nodalmass_current/opensees_lshaped_16storey_manifest.json"),
    )
    parser.add_argument(
        "--falln-nonlinear-stepped",
        default=str(root / "data/output/stage_c_16storey/falln_n4_nonlinear_restart_alarm_to_10s_roof_displacement.csv"),
    )
    parser.add_argument(
        "--falln-nonlinear-stepped-manifest",
        default=str(root / "data/output/stage_c_16storey/falln_n4_nonlinear_restart_alarm_to_10s_summary.json"),
    )
    parser.add_argument(
        "--falln-nonlinear-full",
        default=str(root / "data/output/stage_c_16storey/falln_n4_full_nonlinear_primary_nodal_10s_roof_displacement.csv"),
    )
    parser.add_argument(
        "--falln-nonlinear-full-manifest",
        default=str(root / "data/output/stage_c_16storey/falln_n4_full_nonlinear_primary_nodal_10s_summary.json"),
    )
    parser.add_argument(
        "--opensees-nonlinear",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_disp_falln_proxy_axisfixed/roof_displacement.csv"),
    )
    parser.add_argument(
        "--opensees-nonlinear-manifest",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_disp_falln_proxy_axisfixed/opensees_lshaped_16storey_manifest.json"),
    )
    parser.add_argument(
        "--opensees-force-regularized",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_force_shear_proxy_ets005_axisfixed/roof_displacement.csv"),
    )
    parser.add_argument(
        "--opensees-force-regularized-manifest",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_force_shear_proxy_ets005_axisfixed/opensees_lshaped_16storey_manifest.json"),
    )
    args = parser.parse_args()

    specs = [
        ("fall_n elastico promovido", Path(args.falln_elastic), Path(args.falln_elastic_manifest)),
        ("OpenSees ElasticTimoshenko", Path(args.opensees_elastic), Path(args.opensees_elastic_manifest)),
        ("fall_n inelastico con atajo", Path(args.falln_nonlinear_stepped), Path(args.falln_nonlinear_stepped_manifest)),
        ("fall_n inelastico completo", Path(args.falln_nonlinear_full), Path(args.falln_nonlinear_full_manifest)),
        ("OpenSees dispBeam Concrete02 proxy", Path(args.opensees_nonlinear), Path(args.opensees_nonlinear_manifest)),
        ("OpenSees forceBeam Concrete02 Ets=0.05", Path(args.opensees_force_regularized), Path(args.opensees_force_regularized_manifest)),
    ]
    cases = []
    missing = []
    rows = []
    for label, csv_path, manifest_path in specs:
        case, error = read_roof(csv_path, label, time_limit=args.time_limit)
        if case is None:
            missing.append({"label": label, "path": str(csv_path), "reason": error})
            continue
        manifest = read_manifest(manifest_path)
        cases.append(case)
        rows.append(summarize(case, manifest))

    if len(cases) < 2:
        raise SystemExit("Need at least two available cases to plot.")

    out_dir = Path(args.out_dir)
    figures = plot_cases(cases, out_dir, args.prefix)
    summary = {
        "schema": "lshaped_16_promoted_elastic_inelastic_10s_v1",
        "time_limit_s": args.time_limit,
        "available_case_count": len(cases),
        "missing": missing,
        "rows": rows,
        "figures": figures,
        "notes": [
            "The script can be rerun while long nonlinear OpenSees/fall_n cases finish; missing cases are reported instead of aborting.",
            "The promoted OpenSees nonlinear comparator is dispBeamColumn with the fall_n Kent-Park Concrete02 proxy.",
        ],
    }
    path = out_dir / f"{args.prefix}_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
