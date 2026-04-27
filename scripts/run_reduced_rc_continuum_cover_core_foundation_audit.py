#!/usr/bin/env python3
"""
Freeze a continuum-foundation audit for the reduced RC column with explicit
cross-section physics:

* uniform host vs cover/core host zoning
* continuum-only vs embedded longitudinal steel
* interior bars vs boundary bars
* biased longitudinal mesh with characteristic-length sensitivity

The aim is not to promote the finest mesh blindly. The audit tries to keep the
chain interpretable enough to answer three questions:

1. Does an explicit cover/core split materially change the monotonic response?
2. Do boundary bars behave differently from the promoted interior-bar branch in a meaningful way?
3. When the mesh is biased toward the fixed end, does the crack opening stay
   physically reasonable under the chosen characteristic-length policy?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
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
PURPLE = "#6b46c1"


@dataclass(frozen=True)
class ContinuumCase:
    key: str
    monotonic_tip_mm: float
    monotonic_steps: int
    timeout_seconds: int
    material_mode: str = "nonlinear"
    reinforcement_mode: str = "continuum-only"
    rebar_layout: str = "structural-matched-eight-bar"
    host_concrete_zoning_mode: str = "uniform-reference"
    transverse_mesh_mode: str = "uniform"
    hex_order: str = "hex20"
    nx: int = 4
    ny: int = 4
    nz: int = 2
    longitudinal_bias_power: float = 1.0
    concrete_characteristic_length_mode: str = "mean-longitudinal-host-edge-mm"
    solver_policy: str = "newton-l2-only"
    predictor_policy: str = "hybrid-secant-linearized"
    axial_compression_mn: float = 0.02
    axial_preload_steps: int = 4


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the reduced RC continuum cover/core foundation audit."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--continuum-exe",
        type=Path,
        default=repo_root
        / "build"
        / "fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
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
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def run_command(
    command: list[str], cwd: Path, timeout_seconds: int
) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return time.perf_counter() - start, proc


def continuum_command(exe: Path, out_dir: Path, case: ContinuumCase) -> list[str]:
    return [
        str(exe),
        "--analysis",
        "monotonic",
        "--output-dir",
        str(out_dir),
        "--material-mode",
        case.material_mode,
        "--reinforcement-mode",
        case.reinforcement_mode,
        "--rebar-layout",
        case.rebar_layout,
        "--host-concrete-zoning-mode",
        case.host_concrete_zoning_mode,
        "--transverse-mesh-mode",
        case.transverse_mesh_mode,
        "--hex-order",
        case.hex_order,
        "--nx",
        str(case.nx),
        "--ny",
        str(case.ny),
        "--nz",
        str(case.nz),
        "--longitudinal-bias-power",
        f"{case.longitudinal_bias_power}",
        "--concrete-characteristic-length-mode",
        case.concrete_characteristic_length_mode,
        "--solver-policy",
        case.solver_policy,
        "--predictor-policy",
        case.predictor_policy,
        "--axial-compression-mn",
        f"{case.axial_compression_mn}",
        "--axial-preload-steps",
        str(case.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{case.monotonic_tip_mm}",
        "--monotonic-steps",
        str(case.monotonic_steps),
    ]


def run_case(
    command: list[str],
    output_dir: Path,
    reuse_existing: bool,
    timeout_seconds: int,
) -> tuple[float, dict[str, Any]]:
    ensure_dir(output_dir)
    manifest_path = output_dir / "runtime_manifest.json"
    if reuse_existing and manifest_path.exists():
        return math.nan, read_json(manifest_path)

    elapsed, proc = run_command(command, output_dir.parent, timeout_seconds)
    (output_dir / "runner_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "runner_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit {proc.returncode}:\n{' '.join(command)}\n\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if not manifest_path.exists():
        raise RuntimeError(f"Missing runtime manifest after successful run: {manifest_path}")
    return elapsed, read_json(manifest_path)


def clean_optional_number(value: float) -> float | None:
    return None if not math.isfinite(value) else value


def hysteresis_rows(bundle_dir: Path) -> list[dict[str, Any]]:
    return read_csv_rows(bundle_dir / "hysteresis.csv")


def save(fig: plt.Figure, out_dirs: list[Path], stem: str) -> None:
    for out_dir in out_dirs:
        ensure_dir(out_dir)
        fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


def plot_20mm_overlay(
    histories: dict[str, list[dict[str, Any]]], out_dirs: list[Path]
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    styles = {
        "plain_uniform_hex20_4x4x2_20mm": (BLUE, "-"),
        "plain_covercore_hex20_4x4x2_20mm": (GREEN, "--"),
        "embedded_covercore_interior_hex20_4x4x2_20mm": (ORANGE, "-."),
        "embedded_covercore_boundary_hex20_4x4x2_20mm": (RED, ":"),
    }
    labels = {
        "plain_uniform_hex20_4x4x2_20mm": "Host uniform, no steel",
        "plain_covercore_hex20_4x4x2_20mm": "Host cover/core, no steel",
        "embedded_covercore_interior_hex20_4x4x2_20mm": "Host cover/core + interior bars",
        "embedded_covercore_boundary_hex20_4x4x2_20mm": "Host cover/core + boundary bars",
    }
    for key, rows in histories.items():
        color, linestyle = styles[key]
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [1.0e3 * float(row["base_shear_MN"]) for row in rows],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=labels[key],
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Continuum monotonic overlay at 20 mm")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out_dirs, "reduced_rc_continuum_cover_core_monotonic_overlay")


def plot_characteristic_length_sensitivity(
    histories: dict[str, list[dict[str, Any]]], out_dirs: list[Path]
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    styles = {
        "plain_covercore_hex20_4x4x4_bias3_10mm": (PURPLE, "--"),
        "plain_covercore_hex20_4x4x4_bias3_10mm_fixedend_lb": (GREEN, "-"),
        "embedded_covercore_interior_hex20_4x4x4_bias3_10mm_fixedend_lb": (ORANGE, "-."),
    }
    labels = {
        "plain_covercore_hex20_4x4x4_bias3_10mm": "Host cover/core, mean lb",
        "plain_covercore_hex20_4x4x4_bias3_10mm_fixedend_lb": "Host cover/core, fixed-end lb",
        "embedded_covercore_interior_hex20_4x4x4_bias3_10mm_fixedend_lb": "Host cover/core + interior bars, fixed-end lb",
    }
    for key, rows in histories.items():
        color, linestyle = styles[key]
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [1.0e3 * float(row["base_shear_MN"]) for row in rows],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=labels[key],
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Biased longitudinal mesh and characteristic length")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(
        fig,
        out_dirs,
        "reduced_rc_continuum_characteristic_length_sensitivity",
    )


def plot_timing(rows: list[dict[str, Any]], out_dirs: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    labels = [row["label"] for row in rows]
    x = range(len(rows))
    ax.bar(x, [row["solve_wall_seconds"] for row in rows], color=BLUE)
    ax.set_xticks(list(x), labels, rotation=20, ha="right")
    ax.set_ylabel("Solve wall time [s]")
    ax.set_title("Continuum foundation timing")
    fig.tight_layout()
    save(fig, out_dirs, "reduced_rc_continuum_cover_core_timing")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    out_dirs = [args.figures_dir.resolve(), args.secondary_figures_dir.resolve()]

    cases = [
        ContinuumCase(
            key="plain_uniform_hex20_4x4x2_20mm",
            reinforcement_mode="continuum-only",
            host_concrete_zoning_mode="uniform-reference",
            transverse_mesh_mode="uniform",
            nx=4,
            ny=4,
            nz=2,
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=1800,
        ),
        ContinuumCase(
            key="plain_covercore_hex20_4x4x2_20mm",
            reinforcement_mode="continuum-only",
            host_concrete_zoning_mode="cover-core-split",
            transverse_mesh_mode="cover-aligned",
            nx=4,
            ny=4,
            nz=2,
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=1800,
        ),
        ContinuumCase(
            key="embedded_covercore_interior_hex20_4x4x2_20mm",
            reinforcement_mode="embedded-longitudinal-bars",
            rebar_layout="structural-matched-eight-bar",
            host_concrete_zoning_mode="cover-core-split",
            transverse_mesh_mode="cover-aligned",
            nx=4,
            ny=4,
            nz=2,
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=2400,
        ),
        ContinuumCase(
            key="embedded_covercore_boundary_hex20_4x4x2_20mm",
            reinforcement_mode="embedded-longitudinal-bars",
            rebar_layout="boundary-matched-eight-bar",
            host_concrete_zoning_mode="cover-core-split",
            transverse_mesh_mode="cover-aligned",
            nx=4,
            ny=4,
            nz=2,
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=2400,
        ),
        ContinuumCase(
            key="plain_covercore_hex20_4x4x4_bias3_10mm",
            reinforcement_mode="continuum-only",
            host_concrete_zoning_mode="cover-core-split",
            transverse_mesh_mode="cover-aligned",
            nx=4,
            ny=4,
            nz=4,
            longitudinal_bias_power=3.0,
            monotonic_tip_mm=10.0,
            monotonic_steps=6,
            timeout_seconds=3600,
        ),
        ContinuumCase(
            key="plain_covercore_hex20_4x4x4_bias3_10mm_fixedend_lb",
            reinforcement_mode="continuum-only",
            host_concrete_zoning_mode="cover-core-split",
            transverse_mesh_mode="cover-aligned",
            nx=4,
            ny=4,
            nz=4,
            longitudinal_bias_power=3.0,
            concrete_characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
            monotonic_tip_mm=10.0,
            monotonic_steps=6,
            timeout_seconds=3600,
        ),
        ContinuumCase(
            key="embedded_covercore_interior_hex20_4x4x4_bias3_10mm_fixedend_lb",
            reinforcement_mode="embedded-longitudinal-bars",
            rebar_layout="structural-matched-eight-bar",
            host_concrete_zoning_mode="cover-core-split",
            transverse_mesh_mode="cover-aligned",
            nx=4,
            ny=4,
            nz=4,
            longitudinal_bias_power=3.0,
            concrete_characteristic_length_mode="fixed-end-longitudinal-host-edge-mm",
            monotonic_tip_mm=10.0,
            monotonic_steps=6,
            timeout_seconds=4800,
        ),
    ]

    manifests: dict[str, dict[str, Any]] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_dir = output_dir / case.key
        elapsed, manifest = run_case(
            continuum_command(args.continuum_exe.resolve(), case_dir, case),
            case_dir,
            args.reuse_existing,
            case.timeout_seconds,
        )
        manifests[case.key] = manifest
        histories[case.key] = hysteresis_rows(case_dir)
        observables = manifest.get("observables") or {}
        timing = manifest.get("timing") or {}
        rows.append(
            {
                "key": case.key,
                "label": case.key.replace("_", " "),
                "reinforcement_mode": case.reinforcement_mode,
                "rebar_layout": case.rebar_layout,
                "host_concrete_zoning_mode": case.host_concrete_zoning_mode,
                "transverse_mesh_mode": case.transverse_mesh_mode,
                "mesh": f"{case.nx}x{case.ny}x{case.nz}",
                "longitudinal_bias_power": case.longitudinal_bias_power,
                "characteristic_length_mode": case.concrete_characteristic_length_mode,
                "characteristic_length_mm": float(
                    (manifest.get("concrete_profile_details") or {}).get(
                        "characteristic_length_mm", math.nan
                    )
                ),
                "peak_base_shear_kn": 1.0e3
                * float(observables.get("max_abs_base_shear_mn", math.nan)),
                "peak_cracked_gauss_points": int(
                    observables.get("peak_cracked_gauss_points", 0) or 0
                ),
                "max_crack_opening": float(
                    observables.get("max_crack_opening", 0.0)
                ),
                "peak_embedding_gap_mm": 1.0e3
                * float(observables.get("max_embedding_gap_norm_m", 0.0)),
                "solve_wall_seconds": clean_optional_number(
                    float(timing.get("solve_wall_seconds", math.nan))
                ),
                "process_wall_seconds": clean_optional_number(elapsed),
                "output_dir": str(case_dir),
            }
        )

    plot_20mm_overlay(
        {
            key: histories[key]
            for key in [
                "plain_uniform_hex20_4x4x2_20mm",
                "plain_covercore_hex20_4x4x2_20mm",
                "embedded_covercore_interior_hex20_4x4x2_20mm",
                "embedded_covercore_boundary_hex20_4x4x2_20mm",
            ]
        },
        out_dirs,
    )
    plot_characteristic_length_sensitivity(
        {
            key: histories[key]
            for key in [
                "plain_covercore_hex20_4x4x4_bias3_10mm",
                "plain_covercore_hex20_4x4x4_bias3_10mm_fixedend_lb",
                "embedded_covercore_interior_hex20_4x4x4_bias3_10mm_fixedend_lb",
            ]
        },
        out_dirs,
    )
    plot_timing(rows, out_dirs)

    summary = {
        "status": "completed",
        "cases": rows,
        "key_findings": {
            "cover_core_split_note": (
                "Promoting an explicit cover/core split in the host barely moves "
                "the 4x4x2 monotonic response without steel, but it gives a more "
                "defensible physical separation between unconfined cover and "
                "confined core."
            ),
            "interface_equivalence_note": (
                "An explicit interface-bar probe collapses to the same response "
                "as the promoted structural matched eight-bar branch for the "
                "current reduced-column geometry, because the canonical steel "
                "positions already lie on the cover-core interfaces."
            ),
            "boundary_bar_note": (
                "Boundary bars remain kinematically admissible on the host "
                "boundary. In the 4x4x2 monotonic window they produce a stronger "
                "response than the interior-bar layout, so they must remain an "
                "explicit comparison branch rather than being mixed into the "
                "promoted local baseline."
            ),
            "characteristic_length_note": (
                "With a longitudinally biased 4x4x4 mesh, the mean-host-edge "
                "characteristic length yields an unphysical crack opening even "
                "though the global base shear stays almost unchanged. The fixed-end "
                "longitudinal host edge restores physically plausible crack "
                "openings without changing the monotonic force trace materially."
            ),
            "refinement_frontier_note": (
                "The 4x4x10 cover/core probes remain outside the current practical "
                "budget, so they are treated as an operational frontier rather than "
                "a promoted local-model baseline."
            ),
        },
        "artifacts": {
            "monotonic_overlay_figure": str(
                args.figures_dir / "reduced_rc_continuum_cover_core_monotonic_overlay.png"
            ),
            "characteristic_length_figure": str(
                args.figures_dir
                / "reduced_rc_continuum_characteristic_length_sensitivity.png"
            ),
            "timing_figure": str(
                args.figures_dir / "reduced_rc_continuum_cover_core_timing.png"
            ),
        },
    }

    write_json(output_dir / "continuum_cover_core_foundation_summary.json", summary)
    write_csv(output_dir / "continuum_cover_core_foundation_cases.csv", rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
