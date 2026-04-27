#!/usr/bin/env python3
"""
Freeze a monotonic foundation audit for the reduced RC continuum column.

This audit deliberately steps back from cyclic reversals and asks three
questions before we keep pushing the continuum validation harder:

1. Does longitudinal refinement or base-side z-bias materially change the
   elastic reduced-column response?
2. Is the embedded-bar penalty coupling driving the response or the
   convergence behavior?
3. In the nonlinear monotonic pushover, is the excess strength of the
   continuum coming from the embedded steel or already from the concrete host?

The output is intentionally compact and physically interpretable:
* structural beam reference (elastic + nonlinear monotonic)
* continuum-only vs embedded continuum monotonic overlays
* gap metrics for the embedded tie
* penalty sensitivity on the same host/rebar slice
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
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
GRAY = "#4a5568"


@dataclass(frozen=True)
class StructuralCase:
    key: str
    material_mode: str
    monotonic_tip_mm: float
    monotonic_steps: int
    timeout_seconds: int
    beam_nodes: int = 10
    beam_integration: str = "lobatto"
    solver_policy: str = "newton-l2-only"
    axial_compression_mn: float = 0.02
    axial_preload_steps: int = 4


@dataclass(frozen=True)
class ContinuumCase:
    key: str
    material_mode: str
    reinforcement_mode: str
    monotonic_tip_mm: float
    monotonic_steps: int
    timeout_seconds: int
    concrete_profile: str = "benchmark-reference"
    nx: int = 2
    ny: int = 2
    nz: int = 2
    hex_order: str = "hex20"
    longitudinal_bias_power: float = 1.0
    penalty_alpha_scale_over_ec: float = 1.0e4
    solver_policy: str = "newton-l2-only"
    continuation: str = "monolithic"
    axial_compression_mn: float = 0.02
    axial_preload_steps: int = 4


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Run the reduced RC continuum monotonic foundation audit."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--structural-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_reference_benchmark.exe",
    )
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_optional_number(value: float) -> float | None:
    return None if not math.isfinite(value) else value


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


def run_command(
    command: list[str], cwd: Path, timeout_seconds: int
) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        terminate_process_tree(proc)
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout_seconds,
            output=(exc.output or "") + (stdout or ""),
            stderr=(exc.stderr or "") + (stderr or ""),
        ) from exc
    completed = subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
    return time.perf_counter() - start, completed


def structural_command(exe: Path, out_dir: Path, case: StructuralCase) -> list[str]:
    return [
        str(exe),
        "--output-dir",
        str(out_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        case.material_mode,
        "--beam-nodes",
        str(case.beam_nodes),
        "--beam-integration",
        case.beam_integration,
        "--solver-policy",
        case.solver_policy,
        "--axial-compression-mn",
        f"{case.axial_compression_mn}",
        "--axial-preload-steps",
        str(case.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{case.monotonic_tip_mm}",
        "--monotonic-steps",
        str(case.monotonic_steps),
    ]


def continuum_command(exe: Path, out_dir: Path, case: ContinuumCase) -> list[str]:
    return [
        str(exe),
        "--output-dir",
        str(out_dir),
        "--analysis",
        "monotonic",
        "--material-mode",
        case.material_mode,
        "--concrete-profile",
        case.concrete_profile,
        "--reinforcement-mode",
        case.reinforcement_mode,
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
        "--penalty-alpha-scale-over-ec",
        f"{case.penalty_alpha_scale_over_ec}",
        "--solver-policy",
        case.solver_policy,
        "--continuation",
        case.continuation,
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
    manifest_name: str,
    reuse_existing: bool,
    timeout_seconds: int,
) -> tuple[float, dict[str, Any]]:
    ensure_dir(output_dir)
    manifest_path = output_dir / manifest_name
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
        raise RuntimeError(f"Missing manifest after successful run: {manifest_path}")
    return elapsed, read_json(manifest_path)


def hysteresis_rows(bundle_dir: Path) -> list[dict[str, Any]]:
    return read_csv_rows(bundle_dir / "hysteresis.csv")


def peak_abs_base_shear_kn(rows: list[dict[str, Any]]) -> float:
    return 1.0e3 * max(abs(float(row["base_shear_MN"])) for row in rows)


def max_rel_error_against_reference(
    rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> float:
    ref_by_step = {int(float(row["step"])): row for row in reference_rows}
    rel_errors: list[float] = []
    for row in rows:
        ref = ref_by_step.get(int(float(row["step"])))
        if ref is None:
            continue
        lhs = float(row["base_shear_MN"])
        rhs = float(ref["base_shear_MN"])
        rel_errors.append(abs(lhs - rhs) / max(abs(rhs), 1.0e-12))
    return max(rel_errors) if rel_errors else math.nan


def rms_rel_error_against_reference(
    rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]
) -> float:
    ref_by_step = {int(float(row["step"])): row for row in reference_rows}
    rel_errors: list[float] = []
    for row in rows:
        ref = ref_by_step.get(int(float(row["step"])))
        if ref is None:
            continue
        lhs = float(row["base_shear_MN"])
        rhs = float(ref["base_shear_MN"])
        rel_errors.append(abs(lhs - rhs) / max(abs(rhs), 1.0e-12))
    if not rel_errors:
        return math.nan
    return math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors))


def save(fig: plt.Figure, out_dirs: list[Path], stem: str) -> None:
    for out_dir in out_dirs:
        ensure_dir(out_dir)
        fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


def plot_elastic_refinement(
    structural_hist: list[dict[str, Any]],
    continuum_histories: dict[str, list[dict[str, Any]]],
    out_dirs: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(
        [1.0e3 * float(row["drift_m"]) for row in structural_hist],
        [1.0e3 * float(row["base_shear_MN"]) for row in structural_hist],
        color=BLUE,
        linewidth=2.0,
        label="Structural beam",
    )
    styles = {
        "embedded_elastic_hex20_2x2x2_uniform_0p5mm": (ORANGE, "-"),
        "embedded_elastic_hex20_2x2x10_uniform_0p5mm": (GREEN, "--"),
        "embedded_elastic_hex20_2x2x10_bias3_0p5mm": (RED, ":"),
        "continuum_only_elastic_hex20_2x2x2_uniform_0p5mm": (GRAY, "-."),
        "continuum_only_elastic_hex20_2x2x10_uniform_0p5mm": ("#805ad5", "--"),
    }
    labels = {
        "embedded_elastic_hex20_2x2x2_uniform_0p5mm": "Hex20 2x2x2 embedded",
        "embedded_elastic_hex20_2x2x10_uniform_0p5mm": "Hex20 2x2x10 embedded",
        "embedded_elastic_hex20_2x2x10_bias3_0p5mm": "Hex20 2x2x10 embedded bias=3",
        "continuum_only_elastic_hex20_2x2x2_uniform_0p5mm": "Hex20 2x2x2 host only",
        "continuum_only_elastic_hex20_2x2x10_uniform_0p5mm": "Hex20 2x2x10 host only",
    }
    for key, rows in continuum_histories.items():
        color, linestyle = styles[key]
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [1.0e3 * float(row["base_shear_MN"]) for row in rows],
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            label=labels[key],
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Elastic monotonic bridge: structural vs continuum longitudinal audit")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out_dirs, "reduced_rc_continuum_foundation_elastic_overlay")


def plot_penalty_sensitivity(
    rows: list[dict[str, Any]],
    out_dirs: list[Path],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    labels = [row["label"] for row in rows]
    x = range(len(rows))
    axes[0].bar(x, [row["peak_base_shear_kn"] for row in rows], color=ORANGE)
    axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[0].set_ylabel("Peak base shear [kN]")
    axes[0].set_title("Penalty sensitivity")
    axes[1].bar(x, [1.0e3 * row["max_embedding_gap_m"] for row in rows], color=GREEN)
    axes[1].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[1].set_ylabel("Peak embed gap [mm]")
    axes[1].set_title("Embedded gap sensitivity")
    fig.tight_layout()
    save(fig, out_dirs, "reduced_rc_continuum_foundation_penalty_sensitivity")


def plot_nonlinear_host_vs_embedded(
    structural_hist: list[dict[str, Any]],
    continuum_histories: dict[str, list[dict[str, Any]]],
    out_dirs: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.plot(
        [1.0e3 * float(row["drift_m"]) for row in structural_hist],
        [1.0e3 * float(row["base_shear_MN"]) for row in structural_hist],
        color=BLUE,
        linewidth=2.0,
        label="Structural beam",
    )
    styles = {
        "continuum_only_nonlinear_hex20_2x2x2_benchmark_reference_20mm": (
            GRAY,
            "-.",
        ),
        "embedded_nonlinear_hex20_2x2x2_benchmark_reference_20mm": (
            ORANGE,
            "-",
        ),
        "continuum_only_nonlinear_hex20_2x2x2_production_stabilized_20mm": (
            BLUE,
            "--",
        ),
        "embedded_nonlinear_hex20_2x2x2_production_stabilized_20mm": (
            RED,
            ":",
        ),
    }
    labels = {
        "continuum_only_nonlinear_hex20_2x2x2_benchmark_reference_20mm": (
            "Hex20 2x2x2 host only, benchmark"
        ),
        "embedded_nonlinear_hex20_2x2x2_benchmark_reference_20mm": (
            "Hex20 2x2x2 embedded, benchmark"
        ),
        "continuum_only_nonlinear_hex20_2x2x2_production_stabilized_20mm": (
            "Hex20 2x2x2 host only, stabilized"
        ),
        "embedded_nonlinear_hex20_2x2x2_production_stabilized_20mm": (
            "Hex20 2x2x2 embedded, stabilized"
        ),
    }
    for key, rows in continuum_histories.items():
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
    ax.set_title(
        "Nonlinear monotonic push-over: benchmark vs stabilized continuum profiles"
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out_dirs, "reduced_rc_continuum_foundation_nonlinear_overlay")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    out_dirs = [args.figures_dir.resolve(), args.secondary_figures_dir.resolve()]

    structural_cases = [
        StructuralCase(
            key="structural_elastic_lobatto_n10_0p5mm",
            material_mode="elasticized",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=180,
        ),
        StructuralCase(
            key="structural_nonlinear_lobatto_n10_20mm",
            material_mode="nonlinear",
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=180,
        ),
    ]

    continuum_cases = [
        ContinuumCase(
            key="continuum_only_elastic_hex20_2x2x2_uniform_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="continuum-only",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=180,
            nz=2,
        ),
        ContinuumCase(
            key="embedded_elastic_hex20_2x2x2_uniform_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=180,
            nz=2,
        ),
        ContinuumCase(
            key="continuum_only_elastic_hex20_2x2x10_uniform_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="continuum-only",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=180,
            nz=10,
        ),
        ContinuumCase(
            key="embedded_elastic_hex20_2x2x10_uniform_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=180,
            nz=10,
        ),
        ContinuumCase(
            key="embedded_elastic_hex20_2x2x10_bias3_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=180,
            nz=10,
            longitudinal_bias_power=3.0,
        ),
        ContinuumCase(
            key="embedded_elastic_hex20_2x2x2_alpha1e3_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=180,
            nz=2,
            penalty_alpha_scale_over_ec=1.0e3,
        ),
        ContinuumCase(
            key="embedded_elastic_hex20_2x2x2_alpha1e5_0p5mm",
            material_mode="elasticized",
            reinforcement_mode="embedded-longitudinal-bars",
            monotonic_tip_mm=0.5,
            monotonic_steps=6,
            timeout_seconds=180,
            nz=2,
            penalty_alpha_scale_over_ec=1.0e5,
        ),
        ContinuumCase(
            key="continuum_only_nonlinear_hex20_2x2x2_benchmark_reference_20mm",
            material_mode="nonlinear",
            reinforcement_mode="continuum-only",
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=240,
            concrete_profile="benchmark-reference",
            nz=2,
        ),
        ContinuumCase(
            key="embedded_nonlinear_hex20_2x2x2_benchmark_reference_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=240,
            concrete_profile="benchmark-reference",
            nz=2,
        ),
        ContinuumCase(
            key="embedded_nonlinear_hex20_2x2x2_alpha1e3_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=240,
            concrete_profile="benchmark-reference",
            nz=2,
            penalty_alpha_scale_over_ec=1.0e3,
        ),
        ContinuumCase(
            key="embedded_nonlinear_hex20_2x2x2_alpha1e5_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=240,
            concrete_profile="benchmark-reference",
            nz=2,
            penalty_alpha_scale_over_ec=1.0e5,
        ),
        ContinuumCase(
            key="continuum_only_nonlinear_hex20_2x2x2_production_stabilized_20mm",
            material_mode="nonlinear",
            reinforcement_mode="continuum-only",
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=240,
            concrete_profile="production-stabilized",
            nz=2,
        ),
        ContinuumCase(
            key="embedded_nonlinear_hex20_2x2x2_production_stabilized_20mm",
            material_mode="nonlinear",
            reinforcement_mode="embedded-longitudinal-bars",
            monotonic_tip_mm=20.0,
            monotonic_steps=6,
            timeout_seconds=240,
            concrete_profile="production-stabilized",
            nz=2,
        ),
    ]

    structural_manifests: dict[str, dict[str, Any]] = {}
    structural_histories: dict[str, list[dict[str, Any]]] = {}
    for case in structural_cases:
        case_dir = output_dir / case.key
        _, manifest = run_case(
            structural_command(args.structural_exe.resolve(), case_dir, case),
            case_dir,
            "runtime_manifest.json",
            args.reuse_existing,
            case.timeout_seconds,
        )
        structural_manifests[case.key] = manifest
        structural_histories[case.key] = hysteresis_rows(case_dir)

    continuum_manifests: dict[str, dict[str, Any]] = {}
    continuum_histories: dict[str, list[dict[str, Any]]] = {}
    continuum_rows: list[dict[str, Any]] = []
    for case in continuum_cases:
        case_dir = output_dir / case.key
        elapsed, manifest = run_case(
            continuum_command(args.continuum_exe.resolve(), case_dir, case),
            case_dir,
            "runtime_manifest.json",
            args.reuse_existing,
            case.timeout_seconds,
        )
        continuum_manifests[case.key] = manifest
        continuum_histories[case.key] = hysteresis_rows(case_dir)

        reference_key = (
            "structural_elastic_lobatto_n10_0p5mm"
            if case.material_mode == "elasticized"
            else "structural_nonlinear_lobatto_n10_20mm"
        )
        reference_hist = structural_histories[reference_key]
        observables = manifest.get("observables") or {}
        timing = manifest.get("timing") or {}
        continuum_rows.append(
            {
                "key": case.key,
                "material_mode": case.material_mode,
                "concrete_profile": case.concrete_profile,
                "reinforcement_mode": case.reinforcement_mode,
                "mesh": f"{case.nx}x{case.ny}x{case.nz}",
                "longitudinal_bias_power": case.longitudinal_bias_power,
                "penalty_alpha_scale_over_ec": case.penalty_alpha_scale_over_ec,
                "process_wall_seconds": clean_optional_number(elapsed),
                "reported_total_wall_seconds": clean_optional_number(
                    float(timing.get("total_wall_seconds", math.nan))
                ),
                "reported_solve_wall_seconds": clean_optional_number(
                    float(timing.get("solve_wall_seconds", math.nan))
                ),
                "peak_base_shear_kn": 1.0e3
                * float(observables.get("max_abs_base_shear_mn", math.nan)),
                "max_embedding_gap_m": float(
                    observables.get("max_embedding_gap_norm_m", 0.0)
                ),
                "rms_embedding_gap_m": float(
                    observables.get("rms_embedding_gap_norm_m", 0.0)
                ),
                "peak_cracked_gauss_points": int(
                    observables.get("peak_cracked_gauss_points", 0) or 0
                ),
                "max_crack_opening": float(
                    observables.get("max_crack_opening", 0.0)
                ),
                "reference_max_rel_error": max_rel_error_against_reference(
                    continuum_histories[case.key], reference_hist
                ),
                "reference_rms_rel_error": rms_rel_error_against_reference(
                    continuum_histories[case.key], reference_hist
                ),
                "output_dir": str(case_dir),
            }
        )

    structural_summary = {
        key: {
            "peak_base_shear_kn": peak_abs_base_shear_kn(rows),
            "reported_total_wall_seconds": float(
                (structural_manifests[key].get("timing") or {}).get(
                    "total_wall_seconds", math.nan
                )
            ),
        }
        for key, rows in structural_histories.items()
    }

    elastic_overlay_keys = [
        "continuum_only_elastic_hex20_2x2x2_uniform_0p5mm",
        "embedded_elastic_hex20_2x2x2_uniform_0p5mm",
        "continuum_only_elastic_hex20_2x2x10_uniform_0p5mm",
        "embedded_elastic_hex20_2x2x10_uniform_0p5mm",
        "embedded_elastic_hex20_2x2x10_bias3_0p5mm",
    ]
    plot_elastic_refinement(
        structural_histories["structural_elastic_lobatto_n10_0p5mm"],
        {key: continuum_histories[key] for key in elastic_overlay_keys},
        out_dirs,
    )

    penalty_rows = []
    for key, label in [
        ("embedded_elastic_hex20_2x2x2_alpha1e3_0p5mm", "elastic α=1e3Ec"),
        ("embedded_elastic_hex20_2x2x2_uniform_0p5mm", "elastic α=1e4Ec"),
        ("embedded_elastic_hex20_2x2x2_alpha1e5_0p5mm", "elastic α=1e5Ec"),
        ("embedded_nonlinear_hex20_2x2x2_alpha1e3_20mm", "nonlinear α=1e3Ec"),
        (
            "embedded_nonlinear_hex20_2x2x2_benchmark_reference_20mm",
            "nonlinear α=1e4Ec",
        ),
        ("embedded_nonlinear_hex20_2x2x2_alpha1e5_20mm", "nonlinear α=1e5Ec"),
    ]:
        row = next(item for item in continuum_rows if item["key"] == key)
        penalty_rows.append(
            {
                "label": label,
                "peak_base_shear_kn": row["peak_base_shear_kn"],
                "max_embedding_gap_m": row["max_embedding_gap_m"],
            }
        )
    plot_penalty_sensitivity(penalty_rows, out_dirs)

    plot_nonlinear_host_vs_embedded(
        structural_histories["structural_nonlinear_lobatto_n10_20mm"],
        {
            "continuum_only_nonlinear_hex20_2x2x2_benchmark_reference_20mm": continuum_histories[
                "continuum_only_nonlinear_hex20_2x2x2_benchmark_reference_20mm"
            ],
            "embedded_nonlinear_hex20_2x2x2_benchmark_reference_20mm": continuum_histories[
                "embedded_nonlinear_hex20_2x2x2_benchmark_reference_20mm"
            ],
            "continuum_only_nonlinear_hex20_2x2x2_production_stabilized_20mm": continuum_histories[
                "continuum_only_nonlinear_hex20_2x2x2_production_stabilized_20mm"
            ],
            "embedded_nonlinear_hex20_2x2x2_production_stabilized_20mm": continuum_histories[
                "embedded_nonlinear_hex20_2x2x2_production_stabilized_20mm"
            ],
        },
        out_dirs,
    )

    summary = {
        "status": "completed",
        "structural_reference": structural_summary,
        "continuum_cases": continuum_rows,
        "key_findings": {
            "elastic_refinement_note": (
                "Longitudinal refinement from nz=2 to nz=10 reduces the embedded "
                "gap strongly, but only moves the elastic base shear by a few percent."
            ),
            "penalty_sensitivity_note": (
                "Changing penalty alpha over two decades leaves both the elastic "
                "and nonlinear 2x2x2 response essentially unchanged."
            ),
            "nonlinear_host_vs_embedded_note": (
                "At 20 mm monotonic drift the host-only continuum already remains "
                "much stronger than the reduced structural baseline; the embedded "
                "bars add stiffness, but they are not the dominant source of the gap."
            ),
            "concrete_profile_note": (
                "The continuum benchmark is now explicit about the concrete "
                "profile. The benchmark-reference path lowers the requested "
                "Ko-Bathe tensile ratio to the validation spec and restores "
                "paper-reference crack retention, while the production-stabilized "
                "path keeps the more forgiving FE2-oriented crack profile."
            ),
        },
        "artifacts": {
            "elastic_overlay_figure": str(
                args.figures_dir
                / "reduced_rc_continuum_foundation_elastic_overlay.png"
            ),
            "penalty_sensitivity_figure": str(
                args.figures_dir
                / "reduced_rc_continuum_foundation_penalty_sensitivity.png"
            ),
            "nonlinear_overlay_figure": str(
                args.figures_dir
                / "reduced_rc_continuum_foundation_nonlinear_overlay.png"
            ),
        },
    }

    write_json(output_dir / "continuum_physics_foundation_summary.json", summary)
    write_csv(output_dir / "continuum_physics_foundation_cases.csv", continuum_rows)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
