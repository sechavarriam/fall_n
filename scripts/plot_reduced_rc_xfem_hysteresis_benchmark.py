#!/usr/bin/env python3
"""Plot XFEM benchmark branches for the reduced RC column.

The main branch remains the local XFEM cohesive-hinge surrogate with steel
crossing the crack.  When available, the plot also overlays the full PETSc/SNES
shifted-Heaviside solid solve with truss reinforcement coupled through the
enriched host interpolation.  The host concrete may be elastic or nonlinear,
as declared by the bundle manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot XFEM local cohesive-hinge hysteresis benchmark."
    )
    parser.add_argument(
        "--flexural-bundle",
        type=Path,
        default=Path("data/output/cyclic_validation/xfem_local_cohesive_hinge_200mm_flexural"),
    )
    parser.add_argument(
        "--shear-transfer-bundle",
        type=Path,
        default=Path("data/output/cyclic_validation/xfem_local_cohesive_hinge_200mm"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("doc/figures/validation_reboot"),
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=Path("PhD_Thesis/Figuras/validation_reboot"),
    )
    parser.add_argument(
        "--basename",
        default="xfem_local_cohesive_hinge_200mm",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def loop_work(rows: list[dict[str, float]]) -> float:
    work = 0.0
    for a, b in zip(rows, rows[1:]):
        dx = b["drift_mm"] - a["drift_mm"]
        avg_v = 0.5 * (a["base_shear_MN"] + b["base_shear_MN"])
        work += avg_v * dx
    return work


def peak_abs(rows: list[dict[str, float]], key: str) -> float:
    return max(abs(row[key]) for row in rows) if rows else math.nan


def save_all(fig: Any, figures_dir: Path, secondary_dir: Path, stem: str) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": figures_dir / f"{stem}.png",
        "pdf": figures_dir / f"{stem}.pdf",
    }
    fig.savefig(paths["png"], bbox_inches="tight", dpi=300)
    fig.savefig(paths["pdf"], bbox_inches="tight")
    copied: dict[str, str] = {}
    for kind, path in paths.items():
        target = secondary_dir / path.name
        shutil.copy2(path, target)
        copied[kind] = str(path)
        copied[f"secondary_{kind}"] = str(target)
    return copied


def main() -> int:
    args = parse_args()
    flex_manifest = read_json(args.flexural_bundle / "runtime_manifest.json")
    flex_rows = read_rows(args.flexural_bundle / "hysteresis.csv")
    global_rows: list[dict[str, float]] = []
    global_manifest: dict[str, Any] | None = None
    global_csv = args.flexural_bundle / "global_xfem_newton_hysteresis.csv"
    global_manifest_path = args.flexural_bundle / "global_xfem_newton_manifest.json"
    if global_csv.exists() and global_manifest_path.exists():
        global_rows = read_rows(global_csv)
        global_manifest = read_json(global_manifest_path)

    shear_rows: list[dict[str, float]] = []
    shear_manifest: dict[str, Any] | None = None
    if (args.shear_transfer_bundle / "hysteresis.csv").exists():
        shear_manifest = read_json(args.shear_transfer_bundle / "runtime_manifest.json")
        shear_rows = read_rows(args.shear_transfer_bundle / "hysteresis.csv")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    ax = axes[0]
    ax.plot(
        [row["drift_mm"] for row in flex_rows],
        [1000.0 * row["base_shear_MN"] for row in flex_rows],
        color="#0b5fa5",
        lw=1.8,
        label="XFEM flexural crack",
    )
    if shear_rows:
        ax.plot(
            [row["drift_mm"] for row in shear_rows],
            [1000.0 * row["base_shear_MN"] for row in shear_rows],
            color="#d97706",
            lw=1.2,
            ls="--",
            label="XFEM + direct shear-transfer sensitivity",
        )
    if global_rows:
        ax.plot(
            [row["drift_mm"] for row in global_rows],
            [1000.0 * row["base_shear_MN"] for row in global_rows],
            color="#111827",
            lw=1.2,
            ls="-.",
            label="Global XFEM/SNES concrete trial",
        )
    ax.set_title("XFEM hysteresis branches")
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Equivalent base shear [kN]")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(
        [row["drift_mm"] for row in flex_rows],
        [1000.0 * row["flexural_shear_MN"] for row in flex_rows],
        color="#2563eb",
        label="flexural resultant",
    )
    ax.plot(
        [row["drift_mm"] for row in flex_rows],
        [1000.0 * row["steel_shear_MN"] for row in flex_rows],
        color="#16a34a",
        label="steel contribution",
    )
    ax.plot(
        [row["drift_mm"] for row in flex_rows],
        [row["max_damage"] for row in flex_rows],
        color="#991b1b",
        label="max cohesive damage",
    )
    ax.set_title("Promoted XFEM-flexural branch components")
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("kN or damage")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Reduced RC column XFEM local benchmark up to 200 mm",
        y=1.03,
        fontsize=12,
    )
    artifacts = save_all(
        fig,
        args.figures_dir,
        args.secondary_figures_dir,
        f"{args.basename}_hysteresis",
    )
    plt.close(fig)

    summary = {
        "benchmark_scope": "reduced_rc_xfem_local_cohesive_hinge_200mm",
        "status": "completed",
        "interpretation": (
            "First XFEM-in-benchmark branch. It uses a local cohesive base crack "
            "and Menegotto-Pinto steel crossing the crack. Runtime manifests now "
            "also audit that shifted-Heaviside host-node DOFs are real PETSc "
            "section DOFs and that the shifted-Heaviside solid element assembles "
            "nonzero volumetric plus cohesive residual/tangent contributions. "
            "When present, the global branch is a full PETSc/SNES solve of the "
            "shifted-Heaviside solid with concrete, a cohesive crack plane, "
            "and Menegotto-Pinto truss bars coupled through the enriched host "
            "interpolation. The manifest declares whether the host concrete "
            "is the elastic proxy or a nonlinear validation material."
        ),
        "flexural_branch": {
            "bundle": str(args.flexural_bundle),
            "peak_abs_base_shear_mn": peak_abs(flex_rows, "base_shear_MN"),
            "loop_work_mn_mm": loop_work(flex_rows),
            "peak_abs_steel_stress_mpa": peak_abs(flex_rows, "max_abs_steel_stress_MPa"),
            "max_damage": max(row["max_damage"] for row in flex_rows),
            "manifest": flex_manifest,
        },
        "global_xfem_newton_trial": None,
        "direct_shear_transfer_sensitivity": None,
        "artifacts": artifacts,
    }
    if global_rows and global_manifest is not None:
        summary["global_xfem_newton_trial"] = {
            "bundle": str(args.flexural_bundle),
            "peak_abs_base_shear_mn": peak_abs(global_rows, "base_shear_MN"),
            "loop_work_mn_mm": loop_work(global_rows),
            "peak_abs_steel_stress_mpa": peak_abs(
                global_rows, "max_abs_steel_stress_MPa"
            ),
            "manifest": global_manifest,
            "diagnostic_note": (
                "This is the full global shifted-Heaviside solid solve. Steel "
                "is coupled through the enriched host interpolation; the host "
                "concrete law and solver policy are declared in the manifest."
            ),
        }
    if shear_rows and shear_manifest is not None:
        summary["direct_shear_transfer_sensitivity"] = {
            "bundle": str(args.shear_transfer_bundle),
            "peak_abs_base_shear_mn": peak_abs(shear_rows, "base_shear_MN"),
            "loop_work_mn_mm": loop_work(shear_rows),
            "peak_abs_steel_stress_mpa": peak_abs(shear_rows, "max_abs_steel_stress_MPa"),
            "max_damage": max(row["max_damage"] for row in shear_rows),
            "manifest": shear_manifest,
            "diagnostic_note": (
                "This branch is intentionally not promoted: direct crack-plane "
                "shear transfer saturates the current cap and dominates the "
                "base shear. It is retained as a sensitivity guardrail."
            ),
        }

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.figures_dir / f"{args.basename}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, args.secondary_figures_dir / summary_path.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
