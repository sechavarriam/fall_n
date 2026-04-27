#!/usr/bin/env python3
"""
Audit a diagonal host-concrete strip around the promoted steel path.

This pass extends the single-point host probe into a small physically
interpretable strip along the corner-diagonal of the RC section:

  outer cover  -> cover/core interface (steel path) -> inner core

The goal is to test whether the remaining structural-vs-continuum gap is
really concentrated at the immediate steel location, or whether it is better
read as a more distributed host-field difference through the surrounding
concrete band.
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
PURPLE = "#7c3aed"
RED = "#c53030"


@dataclass(frozen=True)
class HostStripProbe:
    key: str
    label: str
    x_m: float
    y_m: float
    distance_from_interface_mm: float


@dataclass(frozen=True)
class HostStripCase:
    key: str
    label: str
    reinforcement_mode: str
    rebar_layout: str
    timeout_seconds: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Rerun the canonical continuum family with a diagonal host strip "
            "around the promoted steel path."
        )
    )
    parser.add_argument(
        "--promoted-bridge-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_promoted_cyclic_75mm_audit",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark-exe",
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
    parser.add_argument(
        "--case-filter",
        action="append",
        default=[],
        help="Only run cases whose key contains one of these substrings.",
    )
    parser.add_argument("--analysis", choices=("cyclic", "monotonic"), default="cyclic")
    parser.add_argument("--amplitudes-mm", default=None)
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--plain-timeout-seconds", type=int, default=4200)
    parser.add_argument("--interior-timeout-seconds", type=int, default=7200)
    parser.add_argument("--boundary-timeout-seconds", type=int, default=6000)
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
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def max_or_nan(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return max(clean) if clean else math.nan


def rms(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return math.nan
    return math.sqrt(sum(v * v for v in clean) / len(clean))


def clean_optional(value: float) -> float | None:
    return value if math.isfinite(value) else None


def run_command(
    command: list[str], timeout_seconds: int, print_progress: bool
) -> tuple[bool, float, str]:
    start = time.perf_counter()
    if print_progress:
        print(" ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
        return process.returncode == 0, time.perf_counter() - start, stdout
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            stdout, _ = process.communicate()
        return False, time.perf_counter() - start, stdout


def resolve_amplitudes(summary: dict[str, Any], raw_override: str | None) -> str:
    if raw_override:
        return raw_override
    amplitudes = summary.get("protocol", {}).get("amplitudes_mm", [])
    return ",".join(str(float(value)).rstrip("0").rstrip(".") for value in amplitudes)


def default_probes() -> list[HostStripProbe]:
    return [
        HostStripProbe(
            key="outer_cover",
            label="Outer cover",
            x_m=0.110,
            y_m=0.110,
            distance_from_interface_mm=+15.0,
        ),
        HostStripProbe(
            key="steel_interface",
            label="Steel path / interface",
            x_m=0.095,
            y_m=0.095,
            distance_from_interface_mm=0.0,
        ),
        HostStripProbe(
            key="near_core",
            label="Near core",
            x_m=0.080,
            y_m=0.080,
            distance_from_interface_mm=-15.0,
        ),
        HostStripProbe(
            key="deep_core",
            label="Deep core",
            x_m=0.050,
            y_m=0.050,
            distance_from_interface_mm=-45.0,
        ),
    ]


def all_cases(args: argparse.Namespace) -> list[HostStripCase]:
    cases = [
        HostStripCase(
            key="plain",
            label="Plain cover/core",
            reinforcement_mode="plain_concrete",
            rebar_layout="structural_matched_eight_bar",
            timeout_seconds=args.plain_timeout_seconds,
        ),
        HostStripCase(
            key="embedded_interior",
            label="Embedded interior",
            reinforcement_mode="embedded_truss",
            rebar_layout="structural_matched_eight_bar",
            timeout_seconds=args.interior_timeout_seconds,
        ),
        HostStripCase(
            key="embedded_boundary",
            label="Boundary bars",
            reinforcement_mode="embedded_truss",
            rebar_layout="boundary_matched_eight_bar",
            timeout_seconds=args.boundary_timeout_seconds,
        ),
    ]
    if not args.case_filter:
        return cases
    lowered = [token.lower() for token in args.case_filter]
    return [case for case in cases if any(token in case.key.lower() for token in lowered)]


def case_command(
    args: argparse.Namespace,
    case: HostStripCase,
    case_dir: Path,
    amplitudes_mm: str,
    probes: list[HostStripProbe],
    probe_z: float,
) -> list[str]:
    command = [
        str(args.benchmark_exe),
        "--output-dir",
        str(case_dir),
        "--analysis",
        args.analysis,
        "--material-mode",
        "nonlinear",
        "--concrete-profile",
        "production-stabilized",
        "--concrete-tangent-mode",
        "fracture-secant",
        "--concrete-characteristic-length-mode",
        "mean-longitudinal-host-edge-mm",
        "--reinforcement-mode",
        case.reinforcement_mode,
        "--rebar-layout",
        case.rebar_layout,
        "--rebar-interpolation",
        "automatic",
        "--host-concrete-zoning-mode",
        "cover-core-split",
        "--transverse-mesh-mode",
        "cover-aligned",
        "--hex-order",
        "hex20",
        "--nx",
        "4",
        "--ny",
        "4",
        "--nz",
        "2",
        "--longitudinal-bias-power",
        "1.0",
        "--solver-policy",
        "newton-l2-only",
        "--predictor-policy",
        "hybrid-secant-linearized",
        "--continuation",
        "reversal-guarded",
        "--continuation-segment-substep-factor",
        "2",
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
    ]
    if args.analysis == "cyclic":
        command.extend(["--amplitudes-mm", amplitudes_mm])
    else:
        command.extend(["--monotonic-tip-mm", amplitudes_mm, "--monotonic-steps", "12"])
    for probe in probes:
        command.extend(
            [
                "--host-probe",
                f"{probe.key}:{probe.x_m}:{probe.y_m}:{probe_z}",
            ]
        )
    return command


def profile_rows(
    case: HostStripCase,
    case_dir: Path,
    probes: list[HostStripProbe],
) -> list[dict[str, Any]]:
    manifest = read_json(case_dir / "runtime_manifest.json")
    probe_rows = read_csv_rows(case_dir / "host_probe_history.csv")
    probe_by_key = {probe.key: probe for probe in probes}
    rows: list[dict[str, Any]] = []
    for probe in probes:
        samples = [
            row for row in probe_rows if row.get("probe_label") == probe.key
        ]
        rows.append(
            {
                "case_key": case.key,
                "case_label": case.label,
                "probe_key": probe.key,
                "probe_label": probe.label,
                "distance_from_interface_mm": probe.distance_from_interface_mm,
                "target_x_m": probe.x_m,
                "target_y_m": probe.y_m,
                "target_z_m": safe_float(
                    samples[0].get("target_z_m") if samples else math.nan
                ),
                "completed_successfully": bool(
                    manifest.get("completed_successfully", False)
                ),
                "process_wall_seconds": safe_float(
                    manifest.get("timings", {}).get("process_wall_seconds")
                ),
                "solve_wall_seconds": safe_float(
                    manifest.get("timings", {}).get("analysis_solve_wall_seconds")
                ),
                "max_abs_axial_stress_mpa": max_or_nan(
                    [abs(safe_float(row.get("nearest_host_axial_stress_MPa"))) for row in samples]
                ),
                "max_abs_axial_strain": max_or_nan(
                    [abs(safe_float(row.get("nearest_host_axial_strain"))) for row in samples]
                ),
                "max_crack_opening_m": max_or_nan(
                    [safe_float(row.get("nearest_host_max_crack_opening")) for row in samples]
                ),
                "peak_crack_count": max_or_nan(
                    [safe_float(row.get("nearest_host_num_cracks")) for row in samples]
                ),
                "mean_gp_distance_m": rms(
                    [safe_float(row.get("nearest_host_gp_distance_m")) for row in samples]
                ),
                "host_probe_csv": str((case_dir / "host_probe_history.csv").relative_to(Path.cwd())),
            }
        )
    return rows


def plot_profile_metric(
    profile_rows_data: list[dict[str, Any]],
    probes: list[HostStripProbe],
    metric_key: str,
    ylabel: str,
    title: str,
    output_paths: list[Path],
) -> None:
    case_colors = {
        "plain": ORANGE,
        "embedded_interior": BLUE,
        "embedded_boundary": GREEN,
    }
    case_labels = {
        "plain": "Plain cover/core",
        "embedded_interior": "Embedded interior",
        "embedded_boundary": "Boundary bars",
    }
    probe_order = {probe.key: probe.distance_from_interface_mm for probe in probes}

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for case_key in ("plain", "embedded_interior", "embedded_boundary"):
        case_rows = [
            row for row in profile_rows_data if row["case_key"] == case_key
        ]
        if not case_rows:
            continue
        case_rows.sort(key=lambda row: probe_order[row["probe_key"]], reverse=True)
        xs = [float(row["distance_from_interface_mm"]) for row in case_rows]
        ys = [safe_float(row.get(metric_key)) for row in case_rows]
        clean_points = [
            (x, y) for x, y in zip(xs, ys) if y is not None and math.isfinite(y)
        ]
        if not clean_points:
            continue
        ax.plot(
            [point[0] for point in clean_points],
            [point[1] for point in clean_points],
            marker="o",
            linewidth=1.6,
            color=case_colors[case_key],
            label=case_labels[case_key],
        )

    ax.set_xlabel("Distance from steel/interface along diagonal [mm]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axvline(0.0, color=PURPLE, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.legend(loc="best", frameon=True)

    for output_path in output_paths:
        ensure_dir(output_path.parent)
        fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.figures_dir)
    ensure_dir(args.secondary_figures_dir)

    promoted_summary = read_json(
        args.promoted_bridge_dir / "structural_continuum_steel_hysteresis_summary.json"
    )
    promoted_case = promoted_summary["continuum_cases"]["hex20"]
    probe_z = safe_float(promoted_case["selected_bar"]["position_z_m"])
    amplitudes_mm = resolve_amplitudes(promoted_summary, args.amplitudes_mm)
    probes = default_probes()

    cases = all_cases(args)
    profile_summary_rows: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []

    for case in cases:
        case_dir = args.output_dir / case.key
        manifest_path = case_dir / "runtime_manifest.json"
        if args.reuse_existing and manifest_path.exists():
            manifest = read_json(manifest_path)
        else:
            ensure_dir(case_dir)
            command = case_command(args, case, case_dir, amplitudes_mm, probes, probe_z)
            success, elapsed, stdout = run_command(
                command, case.timeout_seconds, args.print_progress
            )
            (case_dir / "runner_stdout.log").write_text(stdout, encoding="utf-8")
            if not success or not manifest_path.exists():
                raise RuntimeError(
                    f"Continuum host-strip case failed: {case.key}\n\n"
                    f"command: {' '.join(command)}\n"
                )
            manifest = read_json(manifest_path)
            manifest.setdefault("timings", {})
            manifest["timings"].setdefault("process_wall_seconds", elapsed)
            write_json(manifest_path, manifest)

        case_profile_rows = profile_rows(case, case_dir, probes)
        profile_summary_rows.extend(case_profile_rows)
        case_records.append(
            {
                "key": case.key,
                "label": case.label,
                "completed_successfully": bool(
                    manifest.get("completed_successfully", False)
                ),
                "process_wall_seconds": safe_float(
                    manifest.get("timings", {}).get("process_wall_seconds")
                ),
                "solve_wall_seconds": safe_float(
                    manifest.get("timings", {}).get("analysis_solve_wall_seconds")
                ),
                "output_dir": str(case_dir.relative_to(Path.cwd())),
            }
        )

    write_csv(args.output_dir / "continuum_host_strip_profile.csv", profile_summary_rows)

    profile_summary = {
        "promoted_bridge_dir": str(args.promoted_bridge_dir.resolve()),
        "benchmark_exe": str(args.benchmark_exe.resolve()),
        "analysis": args.analysis,
        "protocol": promoted_summary.get("protocol", {}),
        "probe_z_m": probe_z,
        "probes": [
            {
                "key": probe.key,
                "label": probe.label,
                "x_m": probe.x_m,
                "y_m": probe.y_m,
                "distance_from_interface_mm": probe.distance_from_interface_mm,
            }
            for probe in probes
        ],
        "cases": case_records,
        "artifacts": {
            "profile_csv": str(
                (args.output_dir / "continuum_host_strip_profile.csv").relative_to(Path.cwd())
            ),
        },
    }

    suffix = "75mm"
    stress_paths = [
        args.figures_dir / f"reduced_rc_continuum_host_strip_peak_stress_{suffix}.png",
        args.secondary_figures_dir
        / f"reduced_rc_continuum_host_strip_peak_stress_{suffix}.png",
    ]
    crack_paths = [
        args.figures_dir
        / f"reduced_rc_continuum_host_strip_peak_crack_opening_{suffix}.png",
        args.secondary_figures_dir
        / f"reduced_rc_continuum_host_strip_peak_crack_opening_{suffix}.png",
    ]
    strain_paths = [
        args.figures_dir / f"reduced_rc_continuum_host_strip_peak_strain_{suffix}.png",
        args.secondary_figures_dir
        / f"reduced_rc_continuum_host_strip_peak_strain_{suffix}.png",
    ]

    plot_profile_metric(
        profile_summary_rows,
        probes,
        "max_abs_axial_stress_mpa",
        "Peak |host axial stress| [MPa]",
        "Continuum host-strip peak stress profile",
        stress_paths,
    )
    plot_profile_metric(
        profile_summary_rows,
        probes,
        "max_crack_opening_m",
        "Peak host crack opening [m]",
        "Continuum host-strip peak crack opening profile",
        crack_paths,
    )
    plot_profile_metric(
        profile_summary_rows,
        probes,
        "max_abs_axial_strain",
        "Peak |host axial strain| [-]",
        "Continuum host-strip peak strain profile",
        strain_paths,
    )

    profile_summary["artifacts"].update(
        {
            "peak_stress_profile_png": [str(path.resolve()) for path in stress_paths],
            "peak_crack_profile_png": [str(path.resolve()) for path in crack_paths],
            "peak_strain_profile_png": [str(path.resolve()) for path in strain_paths],
        }
    )
    write_json(
        args.output_dir / "continuum_host_strip_summary.json", profile_summary
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
