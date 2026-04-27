#!/usr/bin/env python3
"""
Audit the quadratic embedded-bar preload frontier on the reduced RC continuum.

The goal is narrow and falsifiable:

  * keep the continuum host elasticized;
  * remove cyclic reversal entirely;
  * drive only the axial preload stage;
  * compare 2-node and 3-node embedded-bar interpolation under the same host.

If the 3-node route still fails before the first accepted step, then the open
frontier belongs to the embedded quadratic-bar chain itself rather than to
Ko-Bathe cracking, cyclic continuation, or OpenSees comparison logic.
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

BLUE = "#0b5fa5"
ORANGE = "#d97706"
GREEN = "#2f855a"
RED = "#b91c1c"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Audit the preload-only frontier of the quadratic embedded-bar path."
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
        / "reboot_quadratic_embedded_preload_audit",
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    rebar_interpolation: str,
    axial_compression_mn: float,
    penalty_alpha_scale_over_ec: float,
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
        "hex20",
        "--nx",
        "2",
        "--ny",
        "2",
        "--nz",
        "2",
        "--longitudinal-bias-power",
        "3",
        "--concrete-profile",
        "production-stabilized",
        "--concrete-tangent-mode",
        "fracture-secant",
        "--concrete-characteristic-length-mode",
        "fixed-end-longitudinal-host-edge-mm",
        "--solver-policy",
        "newton-l2-only",
        "--continuation",
        "monolithic",
        "--embedded-boundary-mode",
        "dirichlet-rebar-endcap",
        "--penalty-alpha-scale-over-ec",
        f"{penalty_alpha_scale_over_ec}",
        "--monotonic-tip-mm",
        "0.0",
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
            "name": "truss2_preload_only",
            "label": "Truss2 preload",
            "rebar_interpolation": "two-node-linear",
            "axial_compression_mn": 0.02,
            "penalty_alpha_scale_over_ec": 1.0e4,
        },
        {
            "name": "truss3_preload_only",
            "label": "Truss3 preload",
            "rebar_interpolation": "three-node-quadratic",
            "axial_compression_mn": 0.02,
            "penalty_alpha_scale_over_ec": 1.0e4,
        },
        {
            "name": "truss3_preload_alpha1e2",
            "label": "Truss3 preload alpha=1e2",
            "rebar_interpolation": "three-node-quadratic",
            "axial_compression_mn": 0.02,
            "penalty_alpha_scale_over_ec": 1.0e2,
        },
        {
            "name": "truss3_zero_load",
            "label": "Truss3 zero-load control",
            "rebar_interpolation": "three-node-quadratic",
            "axial_compression_mn": 0.0,
            "penalty_alpha_scale_over_ec": 1.0e4,
        },
    ]

    results: list[dict[str, Any]] = []
    for case in cases:
        manifest = run_case(
            args.exe,
            args.output_dir / case["name"],
            rebar_interpolation=case["rebar_interpolation"],
            axial_compression_mn=case["axial_compression_mn"],
            penalty_alpha_scale_over_ec=case["penalty_alpha_scale_over_ec"],
            timeout_seconds=args.timeout_seconds,
        )
        solve_summary = manifest.get("solve_summary", {})
        results.append(
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
                "last_function_norm": solve_summary.get("last_function_norm", 0.0),
                "last_attempt_p_target": solve_summary.get(
                    "last_attempt_p_target", 0.0
                ),
                "total_wall_seconds": float(
                    manifest.get("timing", {}).get("total_wall_seconds", 0.0)
                ),
            }
        )

    payload = {"cases": results}
    write_json(args.output_dir / "quadratic_embedded_preload_summary.json", payload)

    if not args.skip_figure_export:
        labels = [row["label"] for row in results]
        times = [row["total_wall_seconds"] for row in results]
        accepted = [row["accepted_runtime_steps"] for row in results]
        colors = [
            GREEN if row["completed_successfully"] else RED for row in results
        ]

        fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)
        axes[0].bar(labels, times, color=colors)
        axes[0].set_ylabel("Wall time [s]")
        axes[0].set_title("Quadratic embedded-bar preload frontier")

        axes[1].bar(labels, accepted, color=colors)
        axes[1].set_ylabel("Accepted runtime steps")
        axes[1].tick_params(axis="x", rotation=15)

        primary = (
            args.figures_dir
            / "reduced_rc_quadratic_embedded_preload_frontier.png"
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
