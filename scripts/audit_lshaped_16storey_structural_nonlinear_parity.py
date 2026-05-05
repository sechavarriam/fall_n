#!/usr/bin/env python3
"""Audit single-scale nonlinear structural comparators for the L-shaped frame.

The purpose is deliberately narrow: compare fall_n's currently promoted
single-scale structural response against OpenSees variants after closing the
basic material/section parity issues.  OpenSees cases are comparators, not an
oracle; the summary therefore records model assumptions and convergence status.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_roof_csv(path: Path, label: str, *, time_limit: float | None = None) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")

    columns = list(rows[0].keys())
    if {"ux", "uy", "uz"}.issubset(columns):
        ux_col, uy_col, uz_col = "ux", "uy", "uz"
    else:
        ux_col, uy_col, uz_col = columns[-3:]

    out_rows = []
    for row in rows:
        t = float(row["time"])
        if time_limit is not None and t > time_limit + 1.0e-12:
            continue
        out_rows.append({
            "time": t,
            "ux": float(row[ux_col]),
            "uy": float(row[uy_col]),
            "uz": float(row[uz_col]),
        })
    if not out_rows:
        raise ValueError(f"No rows within time limit in {path}")
    return {
        "label": label,
        "path": str(path),
        "time": [r["time"] for r in out_rows],
        "ux": [r["ux"] for r in out_rows],
        "uy": [r["uy"] for r in out_rows],
        "uz": [r["uz"] for r in out_rows],
    }


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(case: dict, manifest: dict | None = None) -> dict:
    manifest = manifest or {}
    row = {
        "label": case["label"],
        "samples": len(case["time"]),
        "t_start_s": case["time"][0],
        "t_end_s": case["time"][-1],
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
        "elasticized_fiber_sections",
        "section_axis_convention",
        "total_wall_seconds",
    ):
        if key in manifest:
            row[key] = manifest[key]
    return row


def plot(cases: list[dict], out_dir: Path, prefix: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    figures: list[str] = []

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 8.5), sharex=True)
    for ax, comp, ylabel in zip(
        axes,
        ("ux", "uy", "uz"),
        (r"$u_x$ [m]", r"$u_y$ [m]", r"$u_z$ [m]"),
    ):
        for case in cases:
            ax.plot(case["time"], case[comp], linewidth=1.35, label=case["label"])
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("tiempo relativo [s]")
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle("Auditoria no lineal de escala simple: cubierta")
    fig.tight_layout()
    pdf = out_dir / f"{prefix}_time_components.pdf"
    png = out_dir / f"{prefix}_time_components.png"
    fig.savefig(pdf)
    fig.savefig(png)
    figures.extend([str(pdf), str(png)])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for case in cases:
        ax.plot(case["ux"], case["uy"], linewidth=1.35, label=case["label"])
        ax.scatter(case["ux"][0], case["uy"][0], s=16)
        ax.scatter(case["ux"][-1], case["uy"][-1], s=22, marker="x")
    ax.set_xlabel(r"$u_x$ [m]")
    ax.set_ylabel(r"$u_y$ [m]")
    ax.set_title("Orbita de cubierta en planta")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    pdf = out_dir / f"{prefix}_plan_orbit.pdf"
    png = out_dir / f"{prefix}_plan_orbit.png"
    fig.savefig(pdf)
    fig.savefig(png)
    figures.extend([str(pdf), str(png)])
    plt.close(fig)
    return figures


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--out-dir", default=str(root / "doc/figures/validation_reboot"))
    parser.add_argument("--prefix", default="lshaped_16_structural_single_scale_nonlinear_parity_1s")
    parser.add_argument(
        "--falln-linear",
        default=str(root / "data/output/stage_c_16storey/falln_n4_newmark_linear_primary_nodal_2s_roof_displacement.csv"),
    )
    parser.add_argument(
        "--opensees-elastic",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_2s_elastic_timoshenko_nodalmass_massmatched/roof_displacement.csv"),
    )
    parser.add_argument(
        "--opensees-force-concrete01",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_1s_force_shear_concrete01_axisfixed/roof_displacement.csv"),
    )
    parser.add_argument(
        "--opensees-disp-proxy",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_1s_disp_falln_proxy_axisfixed/roof_displacement.csv"),
    )
    args = parser.parse_args()

    entries = [
        ("fall_n Newmark linear parity", Path(args.falln_linear), None),
        ("OpenSees ElasticTimoshenko", Path(args.opensees_elastic), Path(args.opensees_elastic).with_name("opensees_lshaped_16storey_manifest.json")),
        ("OpenSees forceBeam+Vy/Vz Concrete01", Path(args.opensees_force_concrete01), Path(args.opensees_force_concrete01).with_name("opensees_lshaped_16storey_manifest.json")),
        ("OpenSees dispBeam Concrete02 proxy", Path(args.opensees_disp_proxy), Path(args.opensees_disp_proxy).with_name("opensees_lshaped_16storey_manifest.json")),
    ]
    cases = []
    summaries = []
    for label, csv_path, manifest_path in entries:
        if not csv_path.exists():
            continue
        case = read_roof_csv(csv_path, label, time_limit=args.time_limit)
        manifest = read_manifest(manifest_path) if manifest_path else {}
        cases.append(case)
        summaries.append(summarize(case, manifest))

    if len(cases) < 2:
        raise SystemExit("Need at least two cases to audit.")

    out_dir = Path(args.out_dir)
    figures = plot(cases, out_dir, args.prefix)
    summary = {
        "schema": "lshaped_16_structural_single_scale_nonlinear_parity_v1",
        "time_limit_s": args.time_limit,
        "rows": summaries,
        "notes": [
            "The promoted elastic/dynamic parity remains fall_n Newmark vs OpenSees ElasticTimoshenko.",
            "OpenSees Concrete01 is intentionally retained as a diagnostic: without tensile concrete it softens immediately under flexure.",
            "OpenSees dispBeamColumn with Concrete02 fall_n proxy is the closest stable nonlinear comparator found in this pass.",
            "OpenSees forceBeamColumn with the same tensile proxy assembled and matched modal stiffness but did not complete the transient because element compatibility failed during crack/tension transitions.",
        ],
        "figures": figures,
    }
    summary_path = out_dir / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
