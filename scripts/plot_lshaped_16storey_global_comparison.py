#!/usr/bin/env python3
"""Plot roof response overlays for the 16-storey L-shaped seismic benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def read_opensees(path: Path, label: str) -> dict | None:
    csv_path = path / "roof_displacement.csv"
    manifest_path = path / "opensees_lshaped_16storey_manifest.json"
    if not csv_path.exists():
        return None
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "time": [r["time"] for r in rows],
        "ux": [r["ux"] for r in rows],
        "uy": [r["uy"] for r in rows],
        "uz": [r["uz"] for r in rows],
        "manifest": manifest,
    }


def read_falln(path: Path, label: str) -> dict | None:
    csv_path = path / "roof_displacement_global_reference.csv"
    manifest_path = path / "global_reference_summary.json"
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = [row for row in reader]
    triplets = []
    for i in range(1, len(columns), 3):
        if i + 2 < len(columns):
            triplets.append((columns[i], columns[i + 1], columns[i + 2]))
    if not triplets:
        return None
    ux_col, uy_col, uz_col = triplets[-1]
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "time": [float(r["time"]) for r in rows],
        "ux": [float(r[ux_col]) for r in rows],
        "uy": [float(r[uy_col]) for r in rows],
        "uz": [float(r[uz_col]) for r in rows],
        "manifest": manifest,
        "falln_columns": [ux_col, uy_col, uz_col],
    }


def summarize_case(case: dict) -> dict:
    out = {"label": case["label"], "samples": len(case["time"])}
    for comp in ("ux", "uy", "uz"):
        vals = case[comp]
        out[f"peak_abs_{comp}_m"] = max((abs(v) for v in vals), default=0.0)
        out[f"final_{comp}_m"] = vals[-1] if vals else 0.0
    return out


def plot(cases: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "figure.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True)
    for ax, comp, ylabel in zip(
        axes,
        ("ux", "uy", "uz"),
        (r"$u_x$ [m]", r"$u_y$ [m]", r"$u_z$ [m]"),
    ):
        for case in cases:
            ax.plot(case["time"], case[comp], linewidth=1.4, label=case["label"])
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("tiempo relativo en la ventana [s]")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Respuesta de cubierta: componentes temporales")
    fig.tight_layout()
    fig.savefig(out_dir / "lshaped_16_roof_time_components.pdf")
    fig.savefig(out_dir / "lshaped_16_roof_time_components.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_subplot(111)
    for case in cases:
        ax.plot(case["ux"], case["uy"], linewidth=1.4, label=case["label"])
        if case["ux"] and case["uy"]:
            ax.scatter(case["ux"][0], case["uy"][0], s=18, marker="o")
            ax.scatter(case["ux"][-1], case["uy"][-1], s=22, marker="x")
    ax.set_xlabel(r"$u_x$ [m]")
    ax.set_ylabel(r"$u_y$ [m]")
    ax.set_title("Orbita espacial de cubierta en planta")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "lshaped_16_roof_plan_orbit.pdf")
    fig.savefig(out_dir / "lshaped_16_roof_plan_orbit.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for case in cases:
        ax.plot(case["ux"], case["uy"], case["uz"], linewidth=1.2, label=case["label"])
    ax.set_xlabel(r"$u_x$ [m]")
    ax.set_ylabel(r"$u_y$ [m]")
    ax.set_zlabel(r"$u_z$ [m]")
    ax.set_title("Trayectoria espacial 3D de cubierta")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "lshaped_16_roof_3d_orbit.pdf")
    fig.savefig(out_dir / "lshaped_16_roof_3d_orbit.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--out-dir", default=str(root / "doc/figures/validation_reboot"))
    parser.add_argument(
        "--opensees-elastic",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_elastic_linear"),
    )
    parser.add_argument(
        "--opensees-disp",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_disp_linear"),
    )
    parser.add_argument(
        "--falln",
        default=str(root / "data/output/lshaped_multiscale_16/recorders"),
    )
    args = parser.parse_args()

    cases = [
        read_opensees(Path(args.opensees_elastic), "OpenSees elastic"),
        read_opensees(Path(args.opensees_disp), "OpenSees dispBeamColumn"),
        read_falln(Path(args.falln), "fall_n TimoshenkoN4"),
    ]
    cases = [c for c in cases if c is not None]
    if not cases:
        raise SystemExit("No response histories found.")
    out_dir = Path(args.out_dir)
    plot(cases, out_dir)
    summary = {
        "schema": "lshaped_16_global_roof_comparison_v1",
        "case_count": len(cases),
        "cases": [summarize_case(c) for c in cases],
        "figures": [
            str(out_dir / "lshaped_16_roof_time_components.pdf"),
            str(out_dir / "lshaped_16_roof_plan_orbit.pdf"),
            str(out_dir / "lshaped_16_roof_3d_orbit.pdf"),
        ],
    }
    (out_dir / "lshaped_16_roof_global_comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
