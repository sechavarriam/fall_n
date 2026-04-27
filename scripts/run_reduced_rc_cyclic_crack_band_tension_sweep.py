#!/usr/bin/env python3
"""Sweep the residual tension-stiffening floor of cyclic crack-band concrete.

The continuum/structural equivalence gap is currently controlled mostly by the
concrete host after tensile cracking.  This runner keeps the structural
reference, mesh, embedded-bar policy, continuation and solver fixed while
varying only the residual post-crack tension-stiffening floor of
``cyclic-crack-band-concrete``.

The output is intentionally compact:
  * one audit bundle per ratio;
  * a machine-readable sweep CSV/JSON;
  * an overlay of global hysteresis and steel hysteresis across the sweep;
  * a metric plot that makes non-monotone parameter effects visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_structural = (
        repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_componentwise_fullpenalty_hex8_2x2x4_uniform_200mm"
        / "structural"
    )
    parser = argparse.ArgumentParser(
        description="Run the reduced RC cyclic crack-band tension-stiffening sweep."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_cyclic_crack_band_tension_sweep_200mm",
    )
    parser.add_argument(
        "--ratios",
        default="0,0.005,0.01,0.02",
        help="Comma-separated residual tension-stiffening ratios.",
    )
    parser.add_argument(
        "--audit-script",
        type=Path,
        default=repo_root
        / "scripts"
        / "run_reduced_rc_structural_continuum_steel_hysteresis_audit.py",
    )
    parser.add_argument(
        "--structural-bundle-dir",
        type=Path,
        default=default_structural if default_structural.exists() else None,
    )
    parser.add_argument(
        "--continuum-exe",
        type=Path,
        default=repo_root
        / "build"
        / "fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
    )
    parser.add_argument(
        "--structural-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
    parser.add_argument("--nx", type=int, default=2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--longitudinal-bias-power", type=float, default=2.0)
    parser.add_argument("--hex-orders", default="hex8")
    parser.add_argument("--solver-policy", default="newton-l2-only")
    parser.add_argument("--continuation", default="reversal-guarded")
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--continuum-timeout-seconds", type=int, default=2400)
    parser.add_argument("--structural-timeout-seconds", type=int, default=900)
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
        "--force-rerun",
        action="store_true",
        help="Forward --force-rerun to each individual audit case.",
    )
    return parser.parse_args()


def parse_ratios(raw: str) -> list[float]:
    ratios = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if not ratios:
        raise ValueError("At least one tension-stiffening ratio is required.")
    return ratios


def ratio_label(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return "ts" + text.replace(".", "p").replace("-", "m")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scalar(payload: dict[str, Any], *path: str) -> float:
    node: Any = payload
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return math.nan
        node = node[key]
    return float(node) if isinstance(node, (int, float)) else math.nan


def case_command(args: argparse.Namespace, ratio: float, case_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(args.audit_script.resolve()),
        "--output-dir",
        str(case_dir.resolve()),
        "--structural-exe",
        str(args.structural_exe.resolve()),
        "--continuum-exe",
        str(args.continuum_exe.resolve()),
        "--analysis",
        "cyclic",
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--continuum-material-mode",
        "cyclic-crack-band-concrete",
        "--concrete-tension-stiffness-ratio",
        f"{ratio}",
        "--continuum-kinematics",
        "corotational",
        "--continuum-concrete-profile",
        "production-stabilized",
        "--continuum-characteristic-length-mode",
        "fixed-end-longitudinal-host-edge-mm",
        "--embedded-boundary-mode",
        "full-penalty-coupling",
        "--axial-preload-transfer-mode",
        "host-surface-only",
        "--host-concrete-zoning-mode",
        "uniform-reference",
        "--transverse-mesh-mode",
        "uniform",
        "--continuum-rebar-layout",
        "structural-matched-eight-bar",
        "--continuum-rebar-interpolation",
        "automatic",
        "--hex-orders",
        args.hex_orders,
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
        "--nz",
        str(args.nz),
        "--longitudinal-bias-power",
        f"{args.longitudinal_bias_power}",
        "--solver-policy",
        args.solver_policy,
        "--continuation",
        args.continuation,
        "--continuation-segment-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--continuum-timeout-seconds",
        str(args.continuum_timeout_seconds),
        "--structural-timeout-seconds",
        str(args.structural_timeout_seconds),
        "--figures-dir",
        str(args.figures_dir.resolve()),
        "--secondary-figures-dir",
        str(args.secondary_figures_dir.resolve()),
        "--skip-figure-export",
    ]
    if args.structural_bundle_dir is not None:
        cmd.extend(["--structural-bundle-dir", str(args.structural_bundle_dir.resolve())])
    if args.force_rerun:
        cmd.append("--force-rerun")
    return cmd


def run_case(args: argparse.Namespace, ratio: float, case_dir: Path) -> dict[str, Any]:
    summary_path = case_dir / "structural_continuum_steel_hysteresis_summary.json"
    if summary_path.exists() and not args.force_rerun:
        return read_json(summary_path)

    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = case_command(args, ratio, case_dir)
    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parent.parent),
        text=True,
        capture_output=True,
        timeout=args.continuum_timeout_seconds + args.structural_timeout_seconds + 120,
        check=False,
    )
    (case_dir / "sweep_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "sweep_stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Sweep case {ratio_label(ratio)} failed with {completed.returncode}.\n"
            f"stderr:\n{completed.stderr}"
        )
    return read_json(summary_path)


def summarize_case(ratio: float, case_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    hex_key = next(iter(summary.get("continuum_cases", {"hex8": {}})), "hex8")
    loop_ratio = scalar(
        summary,
        "steel_local_comparison",
        hex_key,
        "continuum_to_structural_loop_work_ratio",
    )
    base_rms = scalar(
        summary,
        "global_comparison",
        hex_key,
        "peak_normalized_rms_base_shear_error",
    )
    stress_rms = scalar(
        summary,
        "steel_local_comparison",
        hex_key,
        "peak_normalized_rms_stress_vs_drift_error",
    )
    strain_rms = scalar(
        summary,
        "steel_local_comparison",
        hex_key,
        "peak_normalized_rms_strain_vs_drift_error",
    )
    loop_gap = abs(loop_ratio - 1.0) if math.isfinite(loop_ratio) else math.nan
    terms = [
        value
        for value in (base_rms, stress_rms, strain_rms, loop_gap)
        if math.isfinite(value)
    ]
    score = math.sqrt(sum(value * value for value in terms) / len(terms)) if terms else math.nan
    return {
        "ratio": ratio,
        "label": ratio_label(ratio),
        "case_dir": str(case_dir),
        "completed": bool(
            summary.get("continuum_cases", {}).get(hex_key, {}).get(
                "completed_successfully", False
            )
        ),
        "solve_wall_seconds": scalar(
            summary,
            "continuum_cases",
            hex_key,
            "reported_solve_wall_seconds",
        ),
        "base_shear_peak_rms": base_rms,
        "steel_stress_peak_rms": stress_rms,
        "steel_strain_peak_rms": strain_rms,
        "steel_loop_ratio": loop_ratio,
        "steel_loop_gap": loop_gap,
        "host_bar_strain_gap_max": scalar(
            summary,
            "embedded_transfer_comparison",
            hex_key,
            "max_abs_host_bar_axial_strain_gap",
        ),
        "equivalence_score": score,
    }


def save(fig: plt.Figure, out_dirs: list[Path], stem: str) -> None:
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def plot_metrics(rows: list[dict[str, Any]], out_dirs: list[Path], stem: str) -> None:
    xs = [float(row["ratio"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    ax.plot(xs, [float(row["base_shear_peak_rms"]) for row in rows], "o-", label="Base shear RMS")
    ax.plot(xs, [float(row["steel_stress_peak_rms"]) for row in rows], "s-", label="Steel stress RMS")
    ax.plot(xs, [float(row["steel_strain_peak_rms"]) for row in rows], "^-", label="Steel strain RMS")
    ax.plot(xs, [float(row["steel_loop_gap"]) for row in rows], "d-", label="|steel loop ratio - 1|")
    ax.set_xlabel("Residual tension-stiffening ratio")
    ax.set_ylabel("Peak-normalized error / loop gap")
    ax.set_title("Cyclic crack-band concrete equivalence sweep")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dirs, stem + "_metrics")

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    ax.plot(xs, [float(row["solve_wall_seconds"]) for row in rows], "o-", color="#805ad5")
    ax.set_xlabel("Residual tension-stiffening ratio")
    ax.set_ylabel("Solve wall time [s]")
    ax.set_title("Cost sensitivity of cyclic crack-band sweep")
    fig.tight_layout()
    save(fig, out_dirs, stem + "_cost")


def plot_hysteresis_overlays(
    rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    out_dirs: list[Path],
    stem: str,
) -> None:
    if not rows:
        return

    first_summary = summaries[str(rows[0]["ratio"])]
    structural_dir = Path(first_summary["structural_reference"]["bundle_dir"])
    structural_hist = read_csv_rows(structural_dir / "hysteresis.csv")
    structural_steel = read_csv_rows(
        Path(first_summary["artifacts"]["hex8_matched_structural_trace_csv"])
    )

    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(
        [1.0e3 * float(row["drift_m"]) for row in structural_hist],
        [1.0e3 * float(row["base_shear_MN"]) for row in structural_hist],
        color="#0b5fa5",
        linewidth=1.8,
        label="Structural",
    )
    for i, row in enumerate(rows):
        ratio = float(row["ratio"])
        color = cmap(i / max(len(rows) - 1, 1))
        case_dir = Path(str(row["case_dir"]))
        continuum_hist = read_csv_rows(case_dir / "hex8" / "hysteresis.csv")
        ax.plot(
            [1.0e3 * float(item["drift_m"]) for item in continuum_hist],
            [1.0e3 * float(item["base_shear_MN"]) for item in continuum_hist],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=ratio_label(ratio),
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Structural vs continuum base-shear sweep")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, out_dirs, stem + "_base_shear_overlay")

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(
        [float(row["strain_xx"]) for row in structural_steel],
        [float(row["stress_xx_MPa"]) for row in structural_steel],
        color="#0b5fa5",
        linewidth=1.8,
        label="Structural steel",
    )
    for i, row in enumerate(rows):
        ratio = float(row["ratio"])
        color = cmap(i / max(len(rows) - 1, 1))
        case_dir = Path(str(row["case_dir"]))
        continuum_steel = read_csv_rows(case_dir / "hex8" / "selected_continuum_steel_trace.csv")
        ax.plot(
            [float(item["axial_strain"]) for item in continuum_steel],
            [float(item["stress_xx_MPa"]) for item in continuum_steel],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=ratio_label(ratio),
        )
    ax.set_xlabel(r"Steel strain $\varepsilon$")
    ax.set_ylabel(r"Steel stress $\sigma$ [MPa]")
    ax.set_title("Structural fiber vs embedded steel sweep")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, out_dirs, stem + "_steel_overlay")


def main() -> int:
    args = parse_args()
    ratios = parse_ratios(args.ratios)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        label = ratio_label(ratio)
        case_dir = args.output_root / label
        print(f"[sweep] running {label} in {case_dir}", flush=True)
        summary = run_case(args, ratio, case_dir)
        summaries[str(ratio)] = summary
        row = summarize_case(ratio, case_dir, summary)
        rows.append(row)
        print(
            "[sweep] {label}: base={base:.3f}, steel={steel:.3f}, "
            "loop={loop:.3f}, time={time:.1f}s".format(
                label=label,
                base=float(row["base_shear_peak_rms"]),
                steel=float(row["steel_stress_peak_rms"]),
                loop=float(row["steel_loop_ratio"]),
                time=float(row["solve_wall_seconds"]),
            ),
            flush=True,
        )

    rows.sort(key=lambda row: float(row["ratio"]))
    write_csv(args.output_root / "cyclic_crack_band_tension_sweep_summary.csv", rows)
    payload = {
        "description": (
            "Residual tension-stiffening sweep for the reduced RC structural/"
            "continuum equivalence benchmark."
        ),
        "ratios": ratios,
        "cases": rows,
        "best_by_equivalence_score": min(
            rows,
            key=lambda row: (
                float(row["equivalence_score"])
                if math.isfinite(float(row["equivalence_score"]))
                else math.inf
            ),
        )
        if rows
        else None,
    }
    (args.output_root / "cyclic_crack_band_tension_sweep_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    out_dirs = [args.figures_dir.resolve(), args.secondary_figures_dir.resolve()]
    stem = "reduced_rc_cyclic_crack_band_tension_sweep_200mm"
    plot_metrics(rows, out_dirs, stem)
    plot_hysteresis_overlays(rows, summaries, out_dirs, stem)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
