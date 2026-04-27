#!/usr/bin/env python3
"""Run and compare global XFEM mesh-refinement branches against the structural reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class XFEMCase:
    name: str
    nx: int
    ny: int
    nz: int
    bias_power: float
    bias_location: str


DEFAULT_CASES = (
    XFEMCase("nx1_ny1_nz2_uniform", 1, 1, 2, 1.0, "fixed-end"),
    XFEMCase("nx1_ny1_nz4_uniform", 1, 1, 4, 1.0, "fixed-end"),
    XFEMCase("nx2_ny2_nz2_uniform", 2, 2, 2, 1.0, "fixed-end"),
    XFEMCase("nx1_ny1_nz2_fixed_bias2", 1, 1, 2, 2.0, "fixed-end"),
    XFEMCase("nx1_ny1_nz4_fixed_bias2", 1, 1, 4, 2.0, "fixed-end"),
    XFEMCase("nx2_ny2_nz2_fixed_bias2", 2, 2, 2, 2.0, "fixed-end"),
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run a global XFEM secant mesh-refinement matrix."
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=repo / "build/fall_n_reduced_rc_xfem_reference_benchmark.exe",
    )
    parser.add_argument(
        "--structural-dir",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
        "reboot_structural_reference_n10_lobatto_200mm_for_xfem_secant_compare",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
        "xfem_structural_refinement_matrix_200mm",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo / "doc/figures/validation_reboot",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=repo / "PhD_Thesis/Figuras/validation_reboot",
    )
    parser.add_argument("--steps-per-segment", type=int, default=8)
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=900.0,
        help="Maximum wall time per XFEM case before marking it as timed out.",
    )
    parser.add_argument(
        "--crack-z-m",
        type=float,
        default=0.6,
        help=(
            "Physical XFEM crack-plane height. Keeping this fixed separates "
            "mesh convergence from moving-hinge artifacts."
        ),
    )
    parser.add_argument(
        "--auto-crack-z",
        action="store_true",
        help="Use the executable default: crack at the first host element midpoint.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only-case",
        action="append",
        default=[],
        help="Run only matching case names; may be passed more than once.",
    )
    parser.add_argument(
        "--include-2x2x4",
        action="store_true",
        help="Also run a heavier 2x2x4 fixed-end biased case.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compact_case_label(name: str) -> str:
    return (
        name.replace("nx", "")
        .replace("_ny", "x")
        .replace("_nz", "x")
        .replace("_fixed_bias2", " bias2")
        .replace("_uniform", " uniform")
    )


def interpolate_by_p(rows: list[dict[str, float]], p: float, key: str) -> float:
    ps = [row["p"] for row in rows]
    index = bisect_left(ps, p)
    if index <= 0:
        return rows[0][key]
    if index >= len(rows):
        return rows[-1][key]
    lo = rows[index - 1]
    hi = rows[index]
    span = hi["p"] - lo["p"]
    if abs(span) < 1.0e-15:
        return hi[key]
    t = (p - lo["p"]) / span
    return (1.0 - t) * lo[key] + t * hi[key]


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def peak_abs(values: list[float]) -> float:
    return max(abs(value) for value in values) if values else math.nan


def loop_work(rows: list[dict[str, float]], shear_key: str, drift_key: str) -> float:
    work = 0.0
    for previous, current in zip(rows, rows[1:]):
        work += (
            0.5
            * (previous[shear_key] + current[shear_key])
            * (current[drift_key] - previous[drift_key])
        )
    return work


def run_case(args: argparse.Namespace, case: XFEMCase) -> Path:
    out_dir = args.output_root / case.name
    manifest = out_dir / "global_xfem_newton_manifest.json"
    failure_marker = out_dir / "xfem_case_failure.json"
    if manifest.exists() and not args.force:
        data = read_json(manifest)
        physics = data.get("physics", {})
        crack_matches = True
        if not args.auto_crack_z:
            crack_matches = (
                abs(float(physics.get("crack_z_m", math.nan)) - args.crack_z_m)
                <= 1.0e-10
                and physics.get("crack_z_source")
                == "user_prescribed_physical_position"
            )
        if data.get("completed_successfully") and crack_matches:
            print(f"[skip] {case.name}: completed artifact exists")
            return out_dir
    if failure_marker.exists() and not args.force:
        failure = read_json(failure_marker)
        print(f"[skip] {case.name}: previous {failure.get('status', 'failure')}")
        return out_dir

    cmd = [
        str(args.exe),
        "--output-dir",
        str(out_dir),
        "--amplitudes-mm",
        "50,100,150,200",
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--section-cells-x",
        "2",
        "--section-cells-y",
        "2",
        "--tangential-slip-drift-ratio",
        "0.0",
        "--global-xfem-concrete-material",
        "cyclic-crack-band",
        "--global-xfem-crack-band-tangent",
        "secant",
        "--global-xfem-solver-max-iterations",
        "120",
        "--global-xfem-solver-cascade",
        "--global-xfem-adaptive-increments",
        "--global-xfem-nx",
        str(case.nx),
        "--global-xfem-ny",
        str(case.ny),
        "--global-xfem-nz",
        str(case.nz),
        "--global-xfem-bias-power",
        f"{case.bias_power:g}",
        "--global-xfem-bias-location",
        case.bias_location,
    ]
    if not args.auto_crack_z:
        cmd += ["--global-xfem-crack-z-m", f"{args.crack_z_m:g}"]
    print(f"[run] {case.name}: {' '.join(cmd)}")
    tic = time.perf_counter()
    try:
        subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            timeout=args.case_timeout_seconds,
        )
        print(f"[done] {case.name}: {time.perf_counter() - tic:.1f} s")
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - tic
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "case": case.name,
            "status": "timeout",
            "timeout_seconds": args.case_timeout_seconds,
            "elapsed_seconds": elapsed,
            "command": cmd,
        }
        (out_dir / "xfem_case_failure.json").write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        print(f"[timeout] {case.name}: {elapsed:.1f} s")
    except subprocess.CalledProcessError as exc:
        elapsed = time.perf_counter() - tic
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "case": case.name,
            "status": "failed",
            "returncode": exc.returncode,
            "elapsed_seconds": elapsed,
            "command": cmd,
        }
        (out_dir / "xfem_case_failure.json").write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        print(f"[failed] {case.name}: rc={exc.returncode}, {elapsed:.1f} s")
    return out_dir


def compare_case(
    case: XFEMCase,
    out_dir: Path,
    structural_rows: list[dict[str, float]],
) -> dict[str, Any]:
    xfem_rows = read_csv(out_dir / "global_xfem_newton_hysteresis.csv")
    manifest = read_json(out_dir / "global_xfem_newton_manifest.json")
    structural_for_compare = [
        {
            "p": row["p"],
            "drift_mm": 1000.0 * interpolate_by_p(structural_rows, row["p"], "drift_m"),
            "base_shear_MN": interpolate_by_p(
                structural_rows,
                row["p"],
                "base_shear_MN",
            ),
        }
        for row in xfem_rows
    ]
    xfem_shear = [row["base_shear_MN"] for row in xfem_rows]
    structural_shear = [row["base_shear_MN"] for row in structural_for_compare]
    raw_errors = [candidate - reference for candidate, reference in zip(xfem_shear, structural_shear)]
    flipped_errors = [candidate + reference for candidate, reference in zip(xfem_shear, structural_shear)]
    raw_rms = rms(raw_errors)
    flipped_rms = rms(flipped_errors)
    sign_factor = -1.0 if flipped_rms < raw_rms else 1.0
    aligned_structural = [sign_factor * value for value in structural_shear]
    aligned_errors = [
        candidate - reference for candidate, reference in zip(xfem_shear, aligned_structural)
    ]
    peak_xfem = peak_abs(xfem_shear)
    peak_structural = peak_abs(aligned_structural)
    normalization = max(peak_xfem, peak_structural, 1.0e-12)
    aligned_structural_rows = [
        {
            "drift_mm": row["drift_mm"],
            "base_shear_MN": sign_factor * row["base_shear_MN"],
        }
        for row in structural_for_compare
    ]
    xfem_work_rows = [
        {"drift_mm": row["drift_mm"], "base_shear_MN": row["base_shear_MN"]}
        for row in xfem_rows
    ]
    return {
        "case": case.name,
        "output_dir": str(out_dir),
        "completed_successfully": bool(manifest.get("completed_successfully")),
        "nx": case.nx,
        "ny": case.ny,
        "nz": case.nz,
        "bias_power": case.bias_power,
        "bias_location": case.bias_location,
        "crack_z_m": manifest["physics"].get("crack_z_m", math.nan),
        "crack_z_source": manifest["physics"].get("crack_z_source", "unknown"),
        "solver_global_dofs": manifest["dofs"]["solver_global_dofs"],
        "host_element_count": manifest["mesh"]["element_count"],
        "enriched_node_count": manifest["mesh"]["enriched_node_count"],
        "point_count": manifest["protocol"]["point_count"],
        "wall_seconds": manifest["timing"]["total_wall_seconds"],
        "peak_abs_structural_base_shear_MN": peak_structural,
        "peak_abs_xfem_base_shear_MN": peak_xfem,
        "xfem_to_structural_peak_base_shear_ratio": peak_xfem
        / max(peak_structural, 1.0e-12),
        "raw_rms_error_MN": raw_rms,
        "sign_factor_applied_to_structural": sign_factor,
        "rms_base_shear_error_MN": rms(aligned_errors),
        "max_abs_base_shear_error_MN": peak_abs(aligned_errors),
        "peak_normalized_rms_base_shear_error": rms(aligned_errors) / normalization,
        "peak_normalized_max_base_shear_error": peak_abs(aligned_errors) / normalization,
        "structural_loop_work_MN_mm": loop_work(
            aligned_structural_rows,
            "base_shear_MN",
            "drift_mm",
        ),
        "xfem_loop_work_MN_mm": loop_work(
            xfem_work_rows,
            "base_shear_MN",
            "drift_mm",
        ),
        "peak_abs_steel_stress_MPa": manifest["observables"]["peak_abs_steel_stress_mpa"],
        "max_host_damage": manifest["observables"]["max_host_damage"],
        "max_damaged_host_points": manifest["observables"]["max_damaged_host_points"],
    }


def save_figure(fig: Any, figures_dir: Path, secondary_dir: Path, stem: str) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for extension in ("png", "pdf"):
        path = figures_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        shutil.copy2(path, secondary_dir / path.name)
        artifacts[extension] = str(path)
        artifacts[f"secondary_{extension}"] = str(secondary_dir / path.name)
    return artifacts


def plot_matrix(
    args: argparse.Namespace,
    structural_rows: list[dict[str, float]],
    summaries: list[dict[str, Any]],
) -> dict[str, str]:
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
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    sign = summaries[0]["sign_factor_applied_to_structural"] if summaries else -1.0
    axes[0].plot(
        [1000.0 * row["drift_m"] for row in structural_rows],
        [1000.0 * sign * row["base_shear_MN"] for row in structural_rows],
        color="#111827",
        lw=1.8,
        label="Structural N=10 Lobatto",
    )
    colors = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#4d7c0f"]
    for index, summary in enumerate(summaries):
        rows = read_csv(Path(summary["output_dir"]) / "global_xfem_newton_hysteresis.csv")
        axes[0].plot(
            [row["drift_mm"] for row in rows],
            [1000.0 * row["base_shear_MN"] for row in rows],
            lw=1.0,
            color=colors[index % len(colors)],
            label=compact_case_label(summary["case"]),
        )
    axes[0].set_title("Hysteresis overlays")
    axes[0].set_xlabel("Tip drift [mm]")
    axes[0].set_ylabel("Base shear [kN]")
    axes[0].legend(fontsize=6)

    axes[1].scatter(
        [summary["solver_global_dofs"] for summary in summaries],
        [summary["peak_normalized_rms_base_shear_error"] for summary in summaries],
        c=[summary["wall_seconds"] for summary in summaries],
        cmap="viridis",
        s=70,
        edgecolor="#111827",
    )
    for summary in summaries:
        axes[1].annotate(
            compact_case_label(summary["case"]).replace(" uniform", " uni"),
            (
                summary["solver_global_dofs"],
                summary["peak_normalized_rms_base_shear_error"],
            ),
            fontsize=6,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[1].set_title("Error vs global DOFs")
    axes[1].set_xlabel("PETSc reduced DOFs")
    axes[1].set_ylabel("Peak-normalized RMS error")
    fig.suptitle(
        "Global XFEM secant mesh-refinement matrix vs structural reference",
        y=1.04,
        fontsize=12,
    )
    return save_figure(
        fig,
        args.figures_dir,
        args.secondary_figures_dir,
        "xfem_global_secant_structural_refinement_matrix_200mm",
    )


def main() -> int:
    args = parse_args()
    cases = list(DEFAULT_CASES)
    if args.include_2x2x4:
        cases.append(XFEMCase("nx2_ny2_nz4_fixed_bias2", 2, 2, 4, 2.0, "fixed-end"))
    if args.only_case:
        selected = set(args.only_case)
        cases = [case for case in cases if case.name in selected]
    args.output_root.mkdir(parents=True, exist_ok=True)

    structural_rows = read_csv(args.structural_dir / "comparison_hysteresis.csv")
    summaries: list[dict[str, Any]] = []
    for case in cases:
        out_dir = run_case(args, case)
        try:
            summaries.append(compare_case(case, out_dir, structural_rows))
        except FileNotFoundError:
            print(f"[warn] {case.name}: missing completed comparison artifacts")

    summaries.sort(key=lambda row: (row["nx"] * row["ny"] * row["nz"], row["case"]))
    write_csv(args.output_root / "xfem_structural_refinement_matrix.csv", summaries)
    artifacts = plot_matrix(args, structural_rows, summaries)
    failures = [
        read_json(failure)
        for failure in sorted(args.output_root.glob("*/xfem_case_failure.json"))
    ]

    best = min(
        summaries,
        key=lambda row: row["peak_normalized_rms_base_shear_error"],
        default=None,
    )
    summary = {
        "scope": "global_xfem_secant_structural_refinement_matrix_200mm",
        "status": "completed",
        "structural_bundle": str(args.structural_dir),
        "output_root": str(args.output_root),
        "steps_per_segment": args.steps_per_segment,
        "crack_z_m": None if args.auto_crack_z else args.crack_z_m,
        "crack_z_policy": "auto_first_element_midpoint"
        if args.auto_crack_z
        else "fixed_physical_position",
        "case_count": len(summaries),
        "failure_count": len(failures),
        "best_case_by_peak_normalized_rms": best,
        "cases": summaries,
        "failures": failures,
        "interpretation": (
            "Uniform and fixed-end-biased XFEM meshes are compared against the "
            "same N=10 Lobatto structural reference using protocol-time "
            "alignment and explicit base-reaction sign alignment."
        ),
        "artifacts": artifacts,
    }
    summary_path = args.figures_dir / "xfem_global_secant_structural_refinement_matrix_200mm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, args.secondary_figures_dir / summary_path.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
