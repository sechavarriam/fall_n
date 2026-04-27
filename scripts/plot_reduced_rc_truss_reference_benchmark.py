#!/usr/bin/env python3
"""
Emit canonical figures for the standalone TrussElement<3,3> benchmark.

The benchmark compares:

    direct Menegotto-Pinto material path
    vs
    standalone quadratic truss path

under the same cyclic compression-return protocol. The purpose is to make the
isolated axial-bar carrier visible before embedding it into the promoted
continuum RC local model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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
        "figure.dpi": 140,
        "savefig.dpi": 300,
    }
)

BLUE = "#0b5fa5"
ORANGE = "#d97706"
GREEN = "#2f855a"
RED = "#c53030"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Plot the canonical standalone Truss<3> cyclic benchmark."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_truss3_cyclic_compression_baseline",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "doc" / "figures" / "validation_reboot",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=repo_root / "PhD_Thesis" / "Figuras" / "validation_reboot",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def figure_paths(stem: str, figures_dir: Path, secondary_dir: Path) -> list[Path]:
    return [
        figures_dir / f"{stem}.png",
        secondary_dir / f"{stem}.png",
    ]


def save(fig: plt.Figure, paths: list[Path]) -> None:
    for path in paths:
        ensure_dir(path.parent)
        fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    secondary_dir = args.secondary_figures_dir.resolve()

    manifest = read_json(bundle_dir / "runtime_manifest.json")
    material_rows = read_csv_rows(bundle_dir / "material_response.csv")
    truss_rows = read_csv_rows(bundle_dir / "truss_response.csv")

    strain = [row["strain"] for row in material_rows]
    material_stress = [row["stress_MPa"] for row in material_rows]
    material_tangent = [row["tangent_MPa"] for row in material_rows]
    truss_strain = [row["axial_strain"] for row in truss_rows]
    truss_stress = [row["axial_stress_MPa"] for row in truss_rows]
    truss_tangent = [row["tangent_MPa"] for row in truss_rows]
    truss_projected_tangent = [row["tangent_from_element_MPa"] for row in truss_rows]
    end_disp_mm = [1000.0 * row["end_displacement_m"] for row in truss_rows]
    axial_force_kn = [1000.0 * row["axial_force_MN"] for row in truss_rows]
    gp_stress_spread = [row["gp_stress_spread_MPa"] for row in truss_rows]

    comparison = manifest.get("comparison", {})
    max_stress_error = float(comparison.get("max_abs_stress_error_mpa", 0.0))
    max_tangent_error = float(comparison.get("max_abs_element_tangent_error_mpa", 0.0))

    zero_return_rows = [
        row
        for row in truss_rows
        if abs(row["axial_strain"]) < 1.0e-12 and row["step"] > 0
    ]
    final_zero_return_stress = zero_return_rows[-1]["axial_stress_MPa"] if zero_return_rows else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    ax0, ax1 = axes
    ax0.plot(strain, material_stress, color=BLUE, linewidth=2.0, label="direct Menegotto")
    ax0.plot(
        truss_strain,
        truss_stress,
        color=ORANGE,
        linestyle="--",
        linewidth=1.8,
        label="Truss<3>",
    )
    ax0.set_title(
        "Quadratic truss cyclic steel equivalence\n"
        f"max|Δσ|={max_stress_error:.2e} MPa"
    )
    ax0.set_xlabel("Axial strain")
    ax0.set_ylabel("Axial stress [MPa]")
    ax0.legend(loc="best")

    ax1.plot(strain, material_tangent, color=BLUE, linewidth=2.0, label="direct Menegotto")
    ax1.plot(
        truss_strain,
        truss_tangent,
        color=GREEN,
        linestyle="--",
        linewidth=1.6,
        label="Truss<3> GP tangent",
    )
    ax1.plot(
        truss_strain,
        truss_projected_tangent,
        color=RED,
        linestyle=":",
        linewidth=1.8,
        label="Truss<3> projected tangent",
    )
    ax1.set_title(
        "Tangent comparison on the same protocol\n"
        f"max|ΔE_t|={max_tangent_error:.2e} MPa"
    )
    ax1.set_xlabel("Axial strain")
    ax1.set_ylabel("Axial tangent [MPa]")
    ax1.legend(loc="best")
    save(
        fig,
        figure_paths(
            "reduced_rc_truss3_cyclic_menegotto_equivalence",
            figures_dir,
            secondary_dir,
        ),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    ax0, ax1 = axes
    ax0.plot(end_disp_mm, axial_force_kn, color=ORANGE, linewidth=2.0)
    ax0.set_title("Quadratic truss force-displacement loop")
    ax0.set_xlabel("End displacement [mm]")
    ax0.set_ylabel("Axial force [kN]")

    ax1.plot([row["step"] for row in truss_rows], gp_stress_spread, color=GREEN, linewidth=1.8)
    ax1.set_title(
        "Gauss-point stress spread under affine control\n"
        f"σ(ε=0 final)={final_zero_return_stress:.2f} MPa"
    )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("GP stress spread [MPa]")
    save(
        fig,
        figure_paths(
            "reduced_rc_truss3_cyclic_force_displacement",
            figures_dir,
            secondary_dir,
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
