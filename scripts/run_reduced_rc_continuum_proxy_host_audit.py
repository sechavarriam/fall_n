#!/usr/bin/env python3
"""
Summarize the low-cost continuum proxy-host branch.

This audit intentionally stays narrow:
  * same Hex20 4x4x2 cover/core-aware host mesh,
  * same embedded or boundary longitudinal bars,
  * same nonlinear Menegotto steel,
  * cheap orthotropic-bimodular concrete proxy host, and
  * amplitudes pushed until the steel is close to yield.

The goal is not to promote this proxy above the Ko-Bathe baseline. The goal is
to freeze a fast control branch that can expose approximate steel hysteresis
while the full nonlinear host remains the promoted physical reference.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
PURPLE = "#7c3aed"
GRAY = "#4b5563"


@dataclass(frozen=True)
class ProxyBundle:
    key: str
    label: str
    bundle_dir: Path
    color: str


@dataclass(frozen=True)
class ProxyRow:
    key: str
    label: str
    bundle_dir: str
    solve_wall_seconds: float
    max_abs_base_shear_kn: float
    max_rebar_stress_mpa: float
    max_rebar_strain: float
    max_abs_host_rebar_axial_strain_gap: float
    max_embedding_gap_norm_m: float
    tension_stiffness_ratio: float
    peak_cracked_gauss_points: int
    max_crack_opening_m: float
    completed_successfully: bool


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Freeze the reduced RC continuum proxy-host audit."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument(
        "--embedded-30mm-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_continuum_bimodular_proxy_cyclic_30mm",
    )
    parser.add_argument(
        "--embedded-50mm-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_continuum_bimodular_proxy_embedded_cyclic_50mm",
    )
    parser.add_argument(
        "--embedded-75mm-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_continuum_bimodular_proxy_embedded_cyclic_75mm",
    )
    parser.add_argument(
        "--boundary-75mm-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_continuum_bimodular_proxy_boundary_cyclic_75mm",
    )
    parser.add_argument(
        "--embedded-50mm-tr002-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_continuum_bimodular_proxy_cyclic_50mm_tr002",
    )
    parser.add_argument(
        "--embedded-50mm-tr001-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_continuum_bimodular_proxy_cyclic_50mm_tr001",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, out_dirs: list[Path], stem: str) -> list[str]:
    paths: list[str] = []
    for out_dir in out_dirs:
        ensure_dir(out_dir)
        target = out_dir / f"{stem}.png"
        fig.savefig(target)
        paths.append(str(target))
    plt.close(fig)
    return paths


def collect_row(bundle: ProxyBundle) -> tuple[ProxyRow, list[dict[str, str]], list[dict[str, str]]]:
    manifest = read_json(bundle.bundle_dir / "runtime_manifest.json")
    hysteresis_rows = load_csv(bundle.bundle_dir / "hysteresis.csv")
    rebar_rows = load_csv(bundle.bundle_dir / "rebar_history.csv")
    observables = manifest["observables"]

    return (
        ProxyRow(
            key=bundle.key,
            label=bundle.label,
            bundle_dir=str(bundle.bundle_dir),
            solve_wall_seconds=float(manifest["timing"]["solve_wall_seconds"]),
            max_abs_base_shear_kn=1.0e3 * float(observables["max_abs_base_shear_mn"]),
            max_rebar_stress_mpa=float(observables["max_abs_rebar_stress_mpa"]),
            max_rebar_strain=float(observables["max_abs_rebar_strain"]),
            max_abs_host_rebar_axial_strain_gap=float(
                observables["max_abs_host_rebar_axial_strain_gap"]
            ),
            max_embedding_gap_norm_m=float(observables["max_embedding_gap_norm_m"]),
            tension_stiffness_ratio=float(manifest["concrete_tension_stiffness_ratio"]),
            peak_cracked_gauss_points=int(observables["peak_cracked_gauss_points"]),
            max_crack_opening_m=float(observables["max_crack_opening"]),
            completed_successfully=bool(manifest["completed_successfully"]),
        ),
        hysteresis_rows,
        rebar_rows,
    )


def plot_proxy_hysteresis(
    bundles: list[ProxyBundle],
    hysteresis_by_key: dict[str, list[dict[str, str]]],
    out_dirs: list[Path],
) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for bundle in bundles:
        rows = hysteresis_by_key[bundle.key]
        drift_mm = [1.0e3 * float(row["drift_m"]) for row in rows]
        shear_kn = [1.0e3 * float(row["base_shear_MN"]) for row in rows]
        ax.plot(drift_mm, shear_kn, linewidth=1.35, color=bundle.color, label=bundle.label)
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Reduced RC continuum proxy-host hysteresis")
    ax.legend(loc="best", fontsize=8)
    return save(fig, out_dirs, "reduced_rc_continuum_proxy_host_hysteresis_overlay")


def plot_proxy_rebar_stress(
    bundles: list[ProxyBundle],
    rebar_by_key: dict[str, list[dict[str, str]]],
    out_dirs: list[Path],
) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for bundle in bundles:
        rows = rebar_by_key[bundle.key]
        drift_mm = [1.0e3 * float(row["drift_m"]) for row in rows]
        stress_mpa = [float(row["stress_xx_MPa"]) for row in rows]
        ax.plot(drift_mm, stress_mpa, linewidth=1.2, color=bundle.color, label=bundle.label)
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Steel stress [MPa]")
    ax.set_title("Reduced RC continuum proxy-host steel hysteresis")
    ax.legend(loc="best", fontsize=8)
    return save(fig, out_dirs, "reduced_rc_continuum_proxy_host_rebar_stress")


def plot_proxy_timing(rows: list[ProxyRow], out_dirs: list[Path]) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    y = list(range(len(rows)))
    ax.barh(
        y,
        [row.solve_wall_seconds for row in rows],
        color=[GRAY, BLUE, ORANGE, GREEN, RED, PURPLE][: len(rows)],
        alpha=0.9,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([row.label for row in rows], fontsize=8)
    ax.set_xlabel("Solve wall time [s]")
    ax.set_title("Reduced RC continuum proxy-host timing")
    return save(fig, out_dirs, "reduced_rc_continuum_proxy_host_timing")


def plot_proxy_frontier(rows: list[ProxyRow], out_dirs: list[Path]) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    x = range(len(rows))
    ax.plot(
        x,
        [row.max_rebar_stress_mpa for row in rows],
        marker="o",
        linewidth=1.25,
        color=BLUE,
        label="Peak steel stress",
    )
    ax.axhline(420.0, color=RED, linestyle="--", linewidth=1.0, label="Reference fy")
    ax.set_xticks(list(x))
    ax.set_xticklabels([row.label for row in rows], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Peak steel stress [MPa]")
    ax.set_title("Reduced RC continuum proxy-host steel frontier")
    ax.legend(loc="best", fontsize=8)
    return save(fig, out_dirs, "reduced_rc_continuum_proxy_host_steel_frontier")


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    bundles = [
        ProxyBundle(
            key="embedded_30mm_tr010",
            label="Embedded interior, 30 mm, tr=0.10",
            bundle_dir=args.embedded_30mm_dir,
            color=BLUE,
        ),
        ProxyBundle(
            key="embedded_50mm_tr010",
            label="Embedded interior, 50 mm, tr=0.10",
            bundle_dir=args.embedded_50mm_dir,
            color=ORANGE,
        ),
        ProxyBundle(
            key="embedded_75mm_tr010",
            label="Embedded interior, 75 mm, tr=0.10",
            bundle_dir=args.embedded_75mm_dir,
            color=GREEN,
        ),
        ProxyBundle(
            key="boundary_75mm_tr010",
            label="Boundary bars, 75 mm, tr=0.10",
            bundle_dir=args.boundary_75mm_dir,
            color=RED,
        ),
        ProxyBundle(
            key="embedded_50mm_tr002",
            label="Embedded interior, 50 mm, tr=0.02",
            bundle_dir=args.embedded_50mm_tr002_dir,
            color=PURPLE,
        ),
        ProxyBundle(
            key="embedded_50mm_tr001",
            label="Embedded interior, 50 mm, tr=0.01",
            bundle_dir=args.embedded_50mm_tr001_dir,
            color=GRAY,
        ),
    ]

    rows: list[ProxyRow] = []
    hysteresis_by_key: dict[str, list[dict[str, str]]] = {}
    rebar_by_key: dict[str, list[dict[str, str]]] = {}
    for bundle in bundles:
        row, hysteresis_rows, rebar_rows = collect_row(bundle)
        rows.append(row)
        hysteresis_by_key[bundle.key] = hysteresis_rows
        rebar_by_key[bundle.key] = rebar_rows

    out_dirs = [args.figures_dir, args.secondary_figures_dir]
    figures = {
        "hysteresis_overlay_png": plot_proxy_hysteresis(
            bundles, hysteresis_by_key, out_dirs
        )[0],
        "rebar_stress_png": plot_proxy_rebar_stress(
            bundles, rebar_by_key, out_dirs
        )[0],
        "timing_png": plot_proxy_timing(rows, out_dirs)[0],
        "steel_frontier_png": plot_proxy_frontier(rows, out_dirs)[0],
    }

    rows_dict = [asdict(row) for row in rows]
    summary = {
        "status": "completed",
        "case_count": len(rows_dict),
        "cases": rows_dict,
        "key_findings": {
            "support_resultant_fix_required_rebar_endcap_nodes": True,
            "cheap_proxy_host_reaches_near_yield_only_at_large_amplitude": True,
            "proxy_host_remains_uncracked_by_design": True,
            "embedded_interior_75mm_peak_steel_stress_mpa": rows_dict[2][
                "max_rebar_stress_mpa"
            ],
            "boundary_75mm_peak_steel_stress_mpa": rows_dict[3][
                "max_rebar_stress_mpa"
            ],
        },
        "artifacts": {
            "cases_csv": str(args.output_dir / "continuum_proxy_host_cases.csv"),
            "summary_json": str(args.output_dir / "continuum_proxy_host_summary.json"),
            **figures,
        },
    }

    write_csv(args.output_dir / "continuum_proxy_host_cases.csv", rows_dict)
    write_json(args.output_dir / "continuum_proxy_host_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
