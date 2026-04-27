#!/usr/bin/env python3
"""Sweep the structural top-section rotation against a continuum reference.

The continuum benchmark can prescribe a uniform lateral drift over the whole
top face while leaving the axial top-face kinematics free.  That creates an
effective section rotation that is neither exactly the classic free-end beam
condition nor the fully guided theta_y = 0 condition.  This helper keeps the
continuum fixed and sweeps structural controls of the form

    theta_y = ratio * drift / L

so the validation reboot can separate kinematic equivalence from material and
mesh effects before spending minutes per continuum run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuum-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--structural-exe",
        type=Path,
        default=repo / "build_stage8_validation" / "fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument("--ratios", default="0,0.5,1,1.3,1.6,2")
    parser.add_argument("--beam-nodes", type=int, default=4)
    parser.add_argument("--structural-element-count", type=int, default=6)
    parser.add_argument("--beam-integration", default="lobatto")
    parser.add_argument("--solver-policy", default="newton-l2-lu-symbolic-reuse-only")
    parser.add_argument("--amplitudes-mm", default="50")
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--continuation", default="reversal-guarded")
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--section-fiber-profile", default="canonical")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def read_hysteresis(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [
            {
                "step": float(row["step"]),
                "p": float(row["p"]),
                "drift_m": float(row["drift_m"]),
                "base_shear_MN": float(row["base_shear_MN"]),
            }
            for row in csv.DictReader(f)
        ]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ratio_label(ratio: float) -> str:
    return f"{ratio:.3f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def run_structural_case(
    args: argparse.Namespace, ratio: float, out_dir: Path
) -> tuple[bool, float, str]:
    command = [
        str(args.structural_exe),
        "--output-dir",
        str(out_dir),
        "--analysis",
        "cyclic",
        "--material-mode",
        "nonlinear",
        "--beam-nodes",
        str(args.beam_nodes),
        "--structural-element-count",
        str(args.structural_element_count),
        "--beam-integration",
        args.beam_integration,
        "--top-bending-rotation-drift-ratio",
        f"{ratio}",
        "--solver-policy",
        args.solver_policy,
        "--continuation",
        args.continuation,
        "--continuation-segment-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
        "--section-fiber-profile",
        args.section_fiber_profile,
    ]
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        return (
            False,
            elapsed,
            f"returncode={completed.returncode}; stdout_tail={completed.stdout[-500:]}",
        )
    return True, elapsed, ""


def compare(reference: list[dict[str, float]], candidate: list[dict[str, float]]) -> dict[str, float]:
    pairs = list(zip(reference, candidate, strict=False))
    if not pairs:
        return {
            "peak_normalized_rms_base_shear_error": math.nan,
            "peak_normalized_max_base_shear_error": math.nan,
            "candidate_peak_base_shear_mn": math.nan,
        }
    errors = [c["base_shear_MN"] - r["base_shear_MN"] for r, c in pairs]
    peak = max(
        max(abs(r["base_shear_MN"]) for r, _ in pairs),
        max(abs(c["base_shear_MN"]) for _, c in pairs),
        1.0e-12,
    )
    return {
        "peak_normalized_rms_base_shear_error": math.sqrt(
            sum(e * e for e in errors) / len(errors)
        )
        / peak,
        "peak_normalized_max_base_shear_error": max(abs(e) for e in errors) / peak,
        "candidate_peak_base_shear_mn": max(abs(c["base_shear_MN"]) for _, c in pairs),
    }


def plot_overlay(
    path: Path,
    continuum: list[dict[str, float]],
    cases: list[tuple[float, list[dict[str, float]]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.0, 5.0))
    plt.plot(
        [1000.0 * r["drift_m"] for r in continuum],
        [1000.0 * r["base_shear_MN"] for r in continuum],
        color="black",
        linewidth=2.0,
        label="continuum reference",
    )
    colors = plt.cm.viridis([i / max(len(cases) - 1, 1) for i in range(len(cases))])
    for color, (ratio, rows) in zip(colors, cases, strict=True):
        plt.plot(
            [1000.0 * r["drift_m"] for r in rows],
            [1000.0 * r["base_shear_MN"] for r in rows],
            color=color,
            linewidth=1.25,
            linestyle="--",
            label=f"struct theta={ratio:g} drift/L",
        )
    plt.xlabel("Tip drift [mm]")
    plt.ylabel("Base shear [kN]")
    plt.title("Top-rotation kinematic sweep against continuum")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    continuum = read_hysteresis(args.continuum_dir / "hysteresis.csv")
    ratios = [float(token) for token in args.ratios.split(",") if token.strip()]

    rows: list[dict[str, Any]] = []
    cases: list[tuple[float, list[dict[str, float]]]] = []
    for ratio in ratios:
        case_dir = args.output_dir / f"struct_ratio_{ratio_label(ratio)}"
        completed, elapsed, failure_reason = run_structural_case(
            args, ratio, case_dir
        )
        history_path = case_dir / "hysteresis.csv"
        control_path = case_dir / "control_state.csv"
        history = read_hysteresis(history_path) if history_path.exists() else []
        control_rows = read_csv_dicts(control_path) if control_path.exists() else []
        metrics = (
            compare(continuum, history)
            if completed
            else {
                "peak_normalized_rms_base_shear_error": math.nan,
                "peak_normalized_max_base_shear_error": math.nan,
                "candidate_peak_base_shear_mn": max(
                    [abs(r["base_shear_MN"]) for r in history],
                    default=math.nan,
                ),
            }
        )
        metrics.update(
            {
                "ratio": ratio,
                "case_dir": str(case_dir),
                "completed_successfully": completed,
                "failure_reason": failure_reason,
                "process_wall_seconds": elapsed,
                "max_abs_prescribed_top_rotation_rad": max(
                    abs(float(r.get("prescribed_top_bending_rotation_rad", 0.0)))
                    for r in control_rows
                )
                if control_rows
                else math.nan,
                "max_abs_actual_top_rotation_rad": max(
                    abs(float(r.get("actual_top_bending_rotation_rad", 0.0)))
                    for r in control_rows
                )
                if control_rows
                else math.nan,
            }
        )
        rows.append(metrics)
        if completed and history:
            cases.append((ratio, history))

    rows.sort(
        key=lambda row: (
            not bool(row.get("completed_successfully", False)),
            row["peak_normalized_rms_base_shear_error"]
            if math.isfinite(row["peak_normalized_rms_base_shear_error"])
            else math.inf,
        )
    )
    write_csv(args.output_dir / "top_rotation_ratio_sweep.csv", rows)
    figure = args.output_dir / "top_rotation_ratio_sweep_hysteresis.png"
    plot_overlay(figure, continuum, cases)

    summary = {
        "continuum_dir": str(args.continuum_dir),
        "best_by_peak_normalized_rms": rows[0] if rows else None,
        "rows": rows,
        "artifacts": {
            "csv": str(args.output_dir / "top_rotation_ratio_sweep.csv"),
            "figure": str(figure),
        },
    }
    (args.output_dir / "top_rotation_ratio_sweep_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
