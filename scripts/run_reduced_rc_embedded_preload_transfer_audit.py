#!/usr/bin/env python3
"""
Audit how the reduced RC continuum introduces axial preload into the embedded
reinforcement chain.

The key question is whether the currently promoted preload path is physically
and numerically coherent for embedded bars:

  * current path: host surface traction only
  * audit path: split the preload explicitly between host and top rebar nodes

The script intentionally stays on the smallest falsifiable slice:

  * elasticized host
  * preload-only monotonic path
  * Hex20 host
  * 2-node and 3-node embedded bars
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
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

GREEN = "#2f855a"
RED = "#b91c1c"
BLUE = "#0b5fa5"
ORANGE = "#d97706"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Audit axial preload transfer modes for embedded rebar in the reduced RC continuum."
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=repo_root
        / "build"
        / "fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_embedded_preload_transfer_audit",
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
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--skip-figure-export", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        proc.kill()


def run_case(
    exe: Path,
    case_dir: Path,
    *,
    hex_order: str,
    nx: int,
    ny: int,
    nz: int,
    longitudinal_bias_power: float,
    rebar_interpolation: str,
    embedded_boundary_mode: str,
    axial_preload_transfer_mode: str,
    axial_compression_mn: float,
    solver_policy: str,
    snes_divergence_tolerance: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    ensure_dir(case_dir)
    command = [
        str(exe),
        "--output-dir",
        str(case_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        "elasticized",
        "--reinforcement-mode",
        "embedded-longitudinal-bars",
        "--rebar-interpolation",
        rebar_interpolation,
        "--rebar-layout",
        "structural-matched-eight-bar",
        "--hex-order",
        hex_order,
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--nz",
        str(nz),
        "--longitudinal-bias-power",
        f"{longitudinal_bias_power}",
        "--concrete-profile",
        "production-stabilized",
        "--concrete-tangent-mode",
        "fracture-secant",
        "--concrete-characteristic-length-mode",
        "fixed-end-longitudinal-host-edge-mm",
        "--solver-policy",
        solver_policy,
        "--snes-divergence-tolerance",
        snes_divergence_tolerance,
        "--continuation",
        "monolithic",
        "--embedded-boundary-mode",
        embedded_boundary_mode,
        "--axial-preload-transfer-mode",
        axial_preload_transfer_mode,
        "--penalty-alpha-scale-over-ec",
        "1e4",
        "--monotonic-tip-mm",
        "0",
        "--monotonic-steps",
        "1",
        "--axial-compression-mn",
        f"{axial_compression_mn}",
        "--axial-preload-steps",
        "4",
    ]
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=exe.parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        terminate_process_tree(proc)
        stdout, _ = proc.communicate()
        return {
            "completed_successfully": False,
            "termination_reason": "timeout",
            "timing": {"total_wall_seconds": time.perf_counter() - start},
            "solve_summary": {},
            "observables": {},
            "stdout_tail": stdout[-4000:],
        }

    manifest_path = case_dir / "runtime_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = {
            "completed_successfully": False,
            "termination_reason": "missing_runtime_manifest",
            "timing": {"total_wall_seconds": time.perf_counter() - start},
            "solve_summary": {},
            "observables": {},
        }
    manifest["stdout_tail"] = stdout[-4000:]
    return manifest


def save_figure(fig: plt.Figure, path: Path, secondary: Path | None) -> None:
    ensure_dir(path.parent)
    fig.savefig(path)
    if secondary is not None:
        ensure_dir(secondary.parent)
        fig.savefig(secondary)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    cases = [
        {
            "name": "truss2_host_surface_dirichlet",
            "label": "Truss2 host-only",
            "hex_order": "hex20",
            "nx": 2,
            "ny": 2,
            "nz": 2,
            "longitudinal_bias_power": 3.0,
            "rebar_interpolation": "two-node-linear",
            "embedded_boundary_mode": "dirichlet-rebar-endcap",
            "axial_preload_transfer_mode": "host-surface-only",
            "axial_compression_mn": 0.02,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "default",
        },
        {
            "name": "truss2_split_dirichlet",
            "label": "Truss2 split preload",
            "hex_order": "hex20",
            "nx": 2,
            "ny": 2,
            "nz": 2,
            "longitudinal_bias_power": 3.0,
            "rebar_interpolation": "two-node-linear",
            "embedded_boundary_mode": "dirichlet-rebar-endcap",
            "axial_preload_transfer_mode": "composite-section-force-split",
            "axial_compression_mn": 0.02,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "default",
        },
        {
            "name": "truss3_host_surface_dirichlet",
            "label": "Truss3 host-only",
            "hex_order": "hex20",
            "nx": 2,
            "ny": 2,
            "nz": 2,
            "longitudinal_bias_power": 3.0,
            "rebar_interpolation": "three-node-quadratic",
            "embedded_boundary_mode": "dirichlet-rebar-endcap",
            "axial_preload_transfer_mode": "host-surface-only",
            "axial_compression_mn": 0.02,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "default",
        },
        {
            "name": "truss3_split_dirichlet",
            "label": "Truss3 split preload",
            "hex_order": "hex20",
            "nx": 2,
            "ny": 2,
            "nz": 2,
            "longitudinal_bias_power": 3.0,
            "rebar_interpolation": "three-node-quadratic",
            "embedded_boundary_mode": "dirichlet-rebar-endcap",
            "axial_preload_transfer_mode": "composite-section-force-split",
            "axial_compression_mn": 0.02,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "default",
        },
        {
            "name": "truss2_host_surface_fullpenalty",
            "label": "Truss2 full-penalty",
            "hex_order": "hex20",
            "nx": 2,
            "ny": 2,
            "nz": 2,
            "longitudinal_bias_power": 3.0,
            "rebar_interpolation": "two-node-linear",
            "embedded_boundary_mode": "full-penalty-coupling",
            "axial_preload_transfer_mode": "host-surface-only",
            "axial_compression_mn": 0.02,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "default",
        },
        {
            "name": "truss3_host_hex20_1x1x1",
            "label": "Truss3 Hex20 1x1x1",
            "hex_order": "hex20",
            "nx": 1,
            "ny": 1,
            "nz": 1,
            "longitudinal_bias_power": 1.0,
            "rebar_interpolation": "three-node-quadratic",
            "embedded_boundary_mode": "dirichlet-rebar-endcap",
            "axial_preload_transfer_mode": "host-surface-only",
            "axial_compression_mn": 0.02,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "default",
        },
        {
            "name": "truss3_host_hex20_1x1x1_unlimited_dtol",
            "label": "Truss3 Hex20 unlimited dtol",
            "hex_order": "hex20",
            "nx": 1,
            "ny": 1,
            "nz": 1,
            "longitudinal_bias_power": 1.0,
            "rebar_interpolation": "three-node-quadratic",
            "embedded_boundary_mode": "dirichlet-rebar-endcap",
            "axial_preload_transfer_mode": "host-surface-only",
            "axial_compression_mn": 0.02,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "unlimited",
        },
        {
            "name": "truss3_split_hex20_1x1x1_unlimited_dtol",
            "label": "Truss3 split unlimited dtol",
            "hex_order": "hex20",
            "nx": 1,
            "ny": 1,
            "nz": 1,
            "longitudinal_bias_power": 1.0,
            "rebar_interpolation": "three-node-quadratic",
            "embedded_boundary_mode": "dirichlet-rebar-endcap",
            "axial_preload_transfer_mode": "composite-section-force-split",
            "axial_compression_mn": 0.02,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "unlimited",
        },
        {
            "name": "truss3_host_hex27_1x1x1",
            "label": "Truss3 Hex27 1x1x1",
            "hex_order": "hex27",
            "nx": 1,
            "ny": 1,
            "nz": 1,
            "longitudinal_bias_power": 1.0,
            "rebar_interpolation": "automatic",
            "embedded_boundary_mode": "dirichlet-rebar-endcap",
            "axial_preload_transfer_mode": "host-surface-only",
            "axial_compression_mn": 0.02,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "default",
        },
        {
            "name": "truss3_zero_load",
            "label": "Truss3 zero-load",
            "hex_order": "hex20",
            "nx": 2,
            "ny": 2,
            "nz": 2,
            "longitudinal_bias_power": 3.0,
            "rebar_interpolation": "three-node-quadratic",
            "embedded_boundary_mode": "dirichlet-rebar-endcap",
            "axial_preload_transfer_mode": "host-surface-only",
            "axial_compression_mn": 0.0,
            "solver_policy": "newton-l2-only",
            "snes_divergence_tolerance": "default",
        },
    ]

    rows: list[dict[str, Any]] = []
    for case in cases:
        manifest = run_case(
            args.exe,
            args.output_dir / case["name"],
            hex_order=case["hex_order"],
            nx=case["nx"],
            ny=case["ny"],
            nz=case["nz"],
            longitudinal_bias_power=case["longitudinal_bias_power"],
            rebar_interpolation=case["rebar_interpolation"],
            embedded_boundary_mode=case["embedded_boundary_mode"],
            axial_preload_transfer_mode=case["axial_preload_transfer_mode"],
            axial_compression_mn=case["axial_compression_mn"],
            solver_policy=case["solver_policy"],
            snes_divergence_tolerance=case["snes_divergence_tolerance"],
            timeout_seconds=args.timeout_seconds,
        )
        solve_summary = manifest.get("solve_summary", {})
        observables = manifest.get("observables", {})
        rows.append(
            {
                **case,
                "completed_successfully": bool(
                    manifest.get("completed_successfully", False)
                ),
                "termination_reason": str(
                    manifest.get("termination_reason", "unknown")
                ),
                "accepted_runtime_steps": int(
                    solve_summary.get("accepted_runtime_steps", 0)
                ),
                "last_snes_reason": solve_summary.get("last_snes_reason", 0),
                "last_function_norm": float(
                    solve_summary.get("last_function_norm", 0.0)
                ),
                "last_attempt_p_target": float(
                    solve_summary.get("last_attempt_p_target", 0.0)
                ),
                "total_wall_seconds": float(
                    manifest.get("timing", {}).get("total_wall_seconds", 0.0)
                ),
                "solver_policy": case["solver_policy"],
                "snes_divergence_tolerance": case["snes_divergence_tolerance"],
                "max_abs_top_rebar_face_axial_gap_m": float(
                    observables.get("max_abs_top_rebar_face_axial_gap_m", 0.0)
                ),
                "max_embedding_gap_norm_m": float(
                    observables.get("max_embedding_gap_norm_m", 0.0)
                ),
            }
        )

    payload = {"cases": rows}
    write_json(args.output_dir / "embedded_preload_transfer_summary.json", payload)

    if not args.skip_figure_export:
        labels = [row["label"] for row in rows]
        colors = [GREEN if row["completed_successfully"] else RED for row in rows]
        accepted = [row["accepted_runtime_steps"] for row in rows]
        axial_gap_mm = [1.0e3 * row["max_abs_top_rebar_face_axial_gap_m"] for row in rows]
        wall_times = [row["total_wall_seconds"] for row in rows]

        fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.4), sharex=True)
        axes[0].bar(labels, accepted, color=colors)
        axes[0].set_ylabel("Accepted steps")
        axes[0].set_title("Embedded preload transfer audit")

        axes[1].bar(labels, axial_gap_mm, color=colors)
        axes[1].set_ylabel("Top bar-face axial gap [mm]")

        axes[2].bar(labels, wall_times, color=colors)
        axes[2].set_ylabel("Wall time [s]")
        axes[2].tick_params(axis="x", rotation=15)

        primary = (
            args.figures_dir
            / "reduced_rc_embedded_preload_transfer_frontier.png"
        )
        secondary = (
            args.secondary_figures_dir / primary.name
            if args.secondary_figures_dir
            else None
        )
        save_figure(fig, primary, secondary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
