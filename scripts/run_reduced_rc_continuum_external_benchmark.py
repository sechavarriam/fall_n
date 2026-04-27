#!/usr/bin/env python3
"""
Run a paired fall_n-continuum vs OpenSeesPy-continuum reduced-column benchmark.

The comparison is deliberately macro/local rather than pointwise Gauss-point
parity.  OpenSees uses conforming truss bars sharing nodes with the solid mesh;
fall_n may use embedded bars.  Therefore the useful question at this stage is
whether both continuum models show compatible global hysteresis, axial reaction
and steel-stress envelopes under the same declared protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from python_launcher_utils import python_launcher_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fall_n continuum and conforming OpenSees continuum side by side."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=Path("build_stage8_validation/fall_n_reduced_rc_column_continuum_reference_benchmark.exe"),
    )
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument("--analysis", choices=("monotonic", "cyclic"), default="cyclic")
    parser.add_argument(
        "--falln-material-mode",
        choices=(
            "elasticized",
            "orthotropic-bimodular-proxy",
            "tensile-crack-band-damage-proxy",
            "cyclic-crack-band-concrete",
            "fixed-crack-band-concrete",
            "componentwise-kent-park-concrete",
        ),
        default="cyclic-crack-band-concrete",
    )
    parser.add_argument(
        "--opensees-concrete-model",
        choices=("elastic-isotropic", "ASDConcrete3D", "Damage2p"),
        default="ASDConcrete3D",
    )
    parser.add_argument("--opensees-steel-model", choices=("Steel02", "Elastic"), default="Steel02")
    parser.add_argument("--solid-element", choices=("stdBrick", "bbarBrick", "SSPbrick"), default="SSPbrick")
    parser.add_argument("--falln-hex-order", choices=("hex8", "hex20", "hex27"), default="hex8")
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument("--longitudinal-bias-power", type=float, default=1.5)
    parser.add_argument(
        "--longitudinal-bias-location",
        choices=("fixed-end", "loaded-end", "both-ends"),
        default="fixed-end",
    )
    parser.add_argument(
        "--falln-reinforcement-mode",
        choices=("continuum-only", "embedded-longitudinal-bars"),
        default="embedded-longitudinal-bars",
        help="fall_n longitudinal steel route. Use continuum-only for host stiffness controls.",
    )
    parser.add_argument(
        "--falln-transverse-reinforcement-mode",
        choices=("none", "embedded-stirrup-loops"),
        default="none",
    )
    parser.add_argument(
        "--falln-rebar-interpolation",
        choices=("automatic", "two-node-linear", "three-node-quadratic"),
        default="two-node-linear",
    )
    parser.add_argument(
        "--falln-rebar-layout",
        choices=("structural-matched-eight-bar", "boundary-matched-eight-bar", "enriched-twelve-bar"),
        default="structural-matched-eight-bar",
    )
    parser.add_argument(
        "--falln-host-concrete-zoning-mode",
        choices=("uniform-reference", "cover-core-split"),
        default="cover-core-split",
    )
    parser.add_argument(
        "--falln-transverse-mesh-mode",
        choices=("uniform", "cover-aligned"),
        default="cover-aligned",
    )
    parser.add_argument(
        "--falln-top-cap-mode",
        choices=("lateral-translation-only", "uniform-axial-penalty-cap"),
        default="lateral-translation-only",
    )
    parser.add_argument(
        "--falln-axial-preload-transfer-mode",
        choices=("host-surface-only", "composite-section-force-split"),
        default="composite-section-force-split",
    )
    parser.add_argument(
        "--opensees-reinforcement-mode",
        choices=("none", "conforming-eight-bar"),
        default="conforming-eight-bar",
    )
    parser.add_argument(
        "--opensees-host-concrete-zoning-mode",
        choices=("uniform", "cover-core-split"),
        default="cover-core-split",
    )
    parser.add_argument(
        "--opensees-lateral-control-mode",
        choices=("sp-path", "single-node-displacement-control"),
        default="sp-path",
    )
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--monotonic-tip-mm", type=float, default=50.0)
    parser.add_argument("--monotonic-steps", type=int, default=8)
    parser.add_argument("--amplitudes-mm", default="50")
    parser.add_argument("--steps-per-segment", type=int, default=1)
    parser.add_argument("--reversal-substep-factor", type=int, default=1)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument(
        "--falln-solver-policy",
        default="newton-l2-lu-symbolic-reuse-only",
        help="Solver policy passed to the fall_n continuum executable.",
    )
    parser.add_argument(
        "--falln-continuum-kinematics",
        choices=("small-strain", "total-lagrangian", "updated-lagrangian", "corotational"),
        default="corotational",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing subdirectories if their manifest reports completed.",
    )
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(fh):
            out: dict[str, object] = {}
            for key, value in row.items():
                try:
                    out[key] = float(value)
                except (TypeError, ValueError):
                    out[key] = value
            rows.append(out)
        return rows


def read_csv_rows_optional(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def completed_manifest(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        manifest = read_json(path)
    except Exception:
        return False
    return manifest.get("status") == "completed" or manifest.get("completed_successfully") is True


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def falln_command(args: argparse.Namespace, out_dir: Path) -> list[str]:
    continuation = "monolithic" if args.analysis == "monotonic" else "reversal-guarded"
    command = [
        str(args.falln_exe),
        "--output-dir",
        str(out_dir),
        "--analysis",
        args.analysis,
        "--continuum-kinematics",
        args.falln_continuum_kinematics,
        "--material-mode",
        args.falln_material_mode,
        "--hex-order",
        args.falln_hex_order,
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
        "--nz",
        str(args.nz),
        "--longitudinal-bias-power",
        str(args.longitudinal_bias_power),
        "--longitudinal-bias-location",
        args.longitudinal_bias_location,
        "--reinforcement-mode",
        args.falln_reinforcement_mode,
        "--transverse-reinforcement-mode",
        args.falln_transverse_reinforcement_mode,
        "--rebar-interpolation",
        args.falln_rebar_interpolation,
        "--rebar-layout",
        args.falln_rebar_layout,
        "--host-concrete-zoning-mode",
        args.falln_host_concrete_zoning_mode,
        "--transverse-mesh-mode",
        args.falln_transverse_mesh_mode,
        "--top-cap-mode",
        args.falln_top_cap_mode,
        "--axial-preload-transfer-mode",
        args.falln_axial_preload_transfer_mode,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--solver-policy",
        args.falln_solver_policy,
        "--continuation",
        continuation,
        "--monotonic-tip-mm",
        str(args.monotonic_tip_mm),
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--continuation-segment-substep-factor",
        str(args.reversal_substep_factor),
        "--max-bisections",
        str(args.max_bisections),
        "--disable-crack-summary-csv",
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def opensees_command(args: argparse.Namespace, out_dir: Path, repo_root: Path) -> list[str]:
    command = [
        *python_launcher_command(args.python_launcher),
        str(repo_root / "scripts" / "opensees_reduced_rc_column_continuum_reference.py"),
        "--output-dir",
        str(out_dir),
        "--analysis",
        args.analysis,
        "--concrete-model",
        args.opensees_concrete_model,
        "--steel-model",
        args.opensees_steel_model,
        "--solid-element",
        args.solid_element,
        "--reinforcement-mode",
        args.opensees_reinforcement_mode,
        "--host-concrete-zoning-mode",
        args.opensees_host_concrete_zoning_mode,
        "--lateral-control-mode",
        args.opensees_lateral_control_mode,
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
        "--nz",
        str(args.nz),
        "--longitudinal-bias-power",
        str(args.longitudinal_bias_power),
        "--longitudinal-bias-location",
        args.longitudinal_bias_location,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        str(args.monotonic_tip_mm),
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--reversal-substep-factor",
        str(args.reversal_substep_factor),
        "--max-bisections",
        str(args.max_bisections),
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def compare_series(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
    lhs_key: str,
    rhs_key: str | None = None,
) -> dict[str, object]:
    rhs_key = rhs_key or lhs_key
    rhs_by_step = {int(row["step"]): row for row in rhs_rows}
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    for lhs in lhs_rows:
        step = int(lhs["step"])
        rhs = rhs_by_step.get(step)
        if rhs is None:
            continue
        if lhs_key not in lhs or rhs_key not in rhs:
            continue
        lhs_value = float(lhs[lhs_key])
        rhs_value = float(rhs[rhs_key])
        if not (math.isfinite(lhs_value) and math.isfinite(rhs_value)):
            continue
        err = lhs_value - rhs_value
        abs_errors.append(abs(err))
        scale = max(abs(rhs_value), 1.0e-12)
        rel_errors.append(abs(err) / scale)
    return {
        "matched_count": len(abs_errors),
        "lhs_key": lhs_key,
        "rhs_key": rhs_key,
        "max_abs_error": max(abs_errors) if abs_errors else math.nan,
        "rms_abs_error": math.sqrt(sum(e * e for e in abs_errors) / len(abs_errors))
        if abs_errors
        else math.nan,
        "max_rel_error": max(rel_errors) if rel_errors else math.nan,
        "rms_rel_error": math.sqrt(sum(e * e for e in rel_errors) / len(rel_errors))
        if rel_errors
        else math.nan,
    }


def compare_series_at_matching_axis(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
    lhs_value_key: str,
    rhs_value_key: str | None = None,
    *,
    lhs_axis_key: str = "drift_m",
    rhs_axis_key: str | None = None,
    axis_abs_tol: float = 1.0e-9,
) -> dict[str, object]:
    rhs_value_key = rhs_value_key or lhs_value_key
    rhs_axis_key = rhs_axis_key or lhs_axis_key
    rhs_unused = set(range(len(rhs_rows)))
    rhs_reference_values = [
        abs(float(row[rhs_value_key]))
        for row in rhs_rows
        if rhs_value_key in row and math.isfinite(float(row[rhs_value_key]))
    ]
    relative_scale_floor = max(1.0e-12, 1.0e-3 * max(rhs_reference_values, default=0.0))
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    axis_errors: list[float] = []

    for lhs in lhs_rows:
        if lhs_axis_key not in lhs or lhs_value_key not in lhs:
            continue
        lhs_axis = float(lhs[lhs_axis_key])
        best_idx = None
        best_axis_error = math.inf
        for idx in rhs_unused:
            rhs = rhs_rows[idx]
            if rhs_axis_key not in rhs or rhs_value_key not in rhs:
                continue
            axis_error = abs(lhs_axis - float(rhs[rhs_axis_key]))
            if axis_error < best_axis_error:
                best_axis_error = axis_error
                best_idx = idx
        if best_idx is None or best_axis_error > axis_abs_tol:
            continue
        rhs_unused.remove(best_idx)
        rhs = rhs_rows[best_idx]
        lhs_value = float(lhs[lhs_value_key])
        rhs_value = float(rhs[rhs_value_key])
        if not (math.isfinite(lhs_value) and math.isfinite(rhs_value)):
            continue
        err = lhs_value - rhs_value
        abs_errors.append(abs(err))
        rel_errors.append(abs(err) / max(abs(rhs_value), relative_scale_floor))
        axis_errors.append(best_axis_error)

    return {
        "matched_count": len(abs_errors),
        "lhs_value_key": lhs_value_key,
        "rhs_value_key": rhs_value_key,
        "lhs_axis_key": lhs_axis_key,
        "rhs_axis_key": rhs_axis_key,
        "axis_abs_tol": axis_abs_tol,
        "relative_scale_floor": relative_scale_floor,
        "unmatched_lhs_count": max(0, len(lhs_rows) - len(abs_errors)),
        "unmatched_rhs_count": len(rhs_unused),
        "max_axis_abs_error": max(axis_errors) if axis_errors else math.nan,
        "max_abs_error": max(abs_errors) if abs_errors else math.nan,
        "rms_abs_error": math.sqrt(sum(e * e for e in abs_errors) / len(abs_errors))
        if abs_errors
        else math.nan,
        "max_rel_error": max(rel_errors) if rel_errors else math.nan,
        "rms_rel_error": math.sqrt(sum(e * e for e in rel_errors) / len(rel_errors))
        if rel_errors
        else math.nan,
    }


def steel_envelope(rows: list[dict[str, object]], stress_key: str, strain_key: str) -> list[dict[str, object]]:
    by_step: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    out: list[dict[str, object]] = []
    for step, step_rows in sorted(by_step.items()):
        stresses = [float(row[stress_key]) for row in step_rows if math.isfinite(float(row[stress_key]))]
        strains = [float(row[strain_key]) for row in step_rows if math.isfinite(float(row[strain_key]))]
        if not stresses or not strains:
            continue
        drift_values = [float(row["drift_m"]) for row in step_rows if "drift_m" in row]
        out.append(
            {
                "step": step,
                "drift_m": sum(drift_values) / len(drift_values) if drift_values else math.nan,
                "max_tension_stress_MPa": max(stresses),
                "max_compression_stress_MPa": min(stresses),
                "max_abs_stress_MPa": max(abs(value) for value in stresses),
                "max_tension_strain": max(strains),
                "max_compression_strain": min(strains),
                "max_abs_strain": max(abs(value) for value in strains),
            }
        )
    return out


def hysteresis_points(rows: list[dict[str, object]]) -> list[tuple[float, float]]:
    return [
        (1000.0 * float(row["drift_m"]), 1000.0 * float(row["base_shear_MN"]))
        for row in rows
        if math.isfinite(float(row["drift_m"])) and math.isfinite(float(row["base_shear_MN"]))
    ]


def padded_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1.0e-14:
        pad = max(abs(hi), 1.0) * 0.1
        return lo - pad, hi + pad
    pad = 0.08 * (hi - lo)
    return lo - pad, hi + pad


def polyline_svg(points: list[tuple[float, float]], x_map, y_map) -> str:
    if not points:
        return ""
    return " ".join(f"{x_map(x):.2f},{y_map(y):.2f}" for x, y in points)


def write_hysteresis_svg(
    path: Path,
    falln_rows: list[dict[str, object]],
    opensees_rows: list[dict[str, object]],
) -> None:
    falln = hysteresis_points(falln_rows)
    opensees = hysteresis_points(opensees_rows)
    x_values = [x for x, _ in [*falln, *opensees]]
    y_values = [y for _, y in [*falln, *opensees]]
    x_min, x_max = padded_bounds([*x_values, 0.0])
    y_min, y_max = padded_bounds([*y_values, 0.0])
    width = 760
    height = 520
    left = 82
    right = 24
    top = 40
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_map(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def y_map(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    grid = []
    for i in range(6):
        tx = x_min + (x_max - x_min) * i / 5.0
        x = x_map(tx)
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#e5e7eb"/>')
        grid.append(
            f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" '
            f'font-size="12" fill="#374151">{tx:.3g}</text>'
        )
    for i in range(6):
        ty = y_min + (y_max - y_min) * i / 5.0
        y = y_map(ty)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        grid.append(
            f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#374151">{ty:.3g}</text>'
        )

    axes = [
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
    ]
    if x_min <= 0.0 <= x_max:
        axes.append(f'<line x1="{x_map(0):.2f}" y1="{top}" x2="{x_map(0):.2f}" y2="{height-bottom}" stroke="#9ca3af"/>')
    if y_min <= 0.0 <= y_max:
        axes.append(f'<line x1="{left}" y1="{y_map(0):.2f}" x2="{width-right}" y2="{y_map(0):.2f}" stroke="#9ca3af"/>')

    falln_poly = polyline_svg(falln, x_map, y_map)
    ops_poly = polyline_svg(opensees, x_map, y_map)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width/2:.1f}" y="24" text-anchor="middle" font-family="serif" font-size="20">Continuum external benchmark</text>
  <g font-family="Arial, sans-serif">
    {''.join(grid)}
    {''.join(axes)}
    <polyline points="{falln_poly}" fill="none" stroke="#1f77b4" stroke-width="2.2"/>
    <polyline points="{ops_poly}" fill="none" stroke="#ff7f0e" stroke-width="2.0" stroke-dasharray="8 5"/>
    <text x="{width/2:.1f}" y="{height-20}" text-anchor="middle" font-size="14">Tip drift [mm]</text>
    <text x="20" y="{height/2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {height/2:.1f})">Base shear [kN]</text>
    <rect x="{width-220}" y="54" width="182" height="58" fill="white" stroke="#d1d5db"/>
    <line x1="{width-204}" y1="74" x2="{width-164}" y2="74" stroke="#1f77b4" stroke-width="2.2"/>
    <text x="{width-154}" y="78" font-size="13">fall_n continuum</text>
    <line x1="{width-204}" y1="96" x2="{width-164}" y2="96" stroke="#ff7f0e" stroke-width="2.0" stroke-dasharray="8 5"/>
    <text x="{width-154}" y="100" font-size="13">OpenSees continuum</text>
  </g>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_metric_svg(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    falln: list[tuple[float, float]],
    opensees: list[tuple[float, float]],
) -> None:
    x_values = [x for x, _ in [*falln, *opensees]]
    y_values = [y for _, y in [*falln, *opensees]]
    x_min, x_max = padded_bounds([*x_values, 0.0])
    y_min, y_max = padded_bounds([*y_values, 0.0])
    width = 760
    height = 520
    left = 92
    right = 24
    top = 40
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_map(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def y_map(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    grid: list[str] = []
    for i in range(6):
        tx = x_min + (x_max - x_min) * i / 5.0
        x = x_map(tx)
        grid.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#e5e7eb"/>')
        grid.append(
            f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" '
            f'font-size="12" fill="#374151">{tx:.3g}</text>'
        )
    for i in range(6):
        ty = y_min + (y_max - y_min) * i / 5.0
        y = y_map(ty)
        grid.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        grid.append(
            f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#374151">{ty:.3g}</text>'
        )

    axes = [
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
    ]
    if x_min <= 0.0 <= x_max:
        axes.append(f'<line x1="{x_map(0):.2f}" y1="{top}" x2="{x_map(0):.2f}" y2="{height-bottom}" stroke="#9ca3af"/>')
    if y_min <= 0.0 <= y_max:
        axes.append(f'<line x1="{left}" y1="{y_map(0):.2f}" x2="{width-right}" y2="{y_map(0):.2f}" stroke="#9ca3af"/>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="{width/2:.1f}" y="24" text-anchor="middle" font-family="serif" font-size="20">{title}</text>
  <g font-family="Arial, sans-serif">
    {''.join(grid)}
    {''.join(axes)}
    <polyline points="{polyline_svg(falln, x_map, y_map)}" fill="none" stroke="#1f77b4" stroke-width="2.2"/>
    <polyline points="{polyline_svg(opensees, x_map, y_map)}" fill="none" stroke="#ff7f0e" stroke-width="2.0" stroke-dasharray="8 5"/>
    <text x="{width/2:.1f}" y="{height-20}" text-anchor="middle" font-size="14">{x_label}</text>
    <text x="22" y="{height/2:.1f}" text-anchor="middle" font-size="14" transform="rotate(-90 22 {height/2:.1f})">{y_label}</text>
    <rect x="{width-220}" y="54" width="182" height="58" fill="white" stroke="#d1d5db"/>
    <line x1="{width-204}" y1="74" x2="{width-164}" y2="74" stroke="#1f77b4" stroke-width="2.2"/>
    <text x="{width-154}" y="78" font-size="13">fall_n continuum</text>
    <line x1="{width-204}" y1="96" x2="{width-164}" y2="96" stroke="#ff7f0e" stroke-width="2.0" stroke-dasharray="8 5"/>
    <text x="{width-154}" y="100" font-size="13">OpenSees continuum</text>
  </g>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def steel_metric_points(
    rows: list[dict[str, object]],
    value_key: str,
) -> list[tuple[float, float]]:
    return [
        (1000.0 * float(row["drift_m"]), float(row[value_key]))
        for row in rows
        if "drift_m" in row
        and value_key in row
        and math.isfinite(float(row["drift_m"]))
        and math.isfinite(float(row[value_key]))
    ]


def write_steel_envelope_svgs(bundle_dir: Path, fig_dir: Path) -> dict[str, str]:
    falln = read_csv_rows(bundle_dir / "steel_envelope_fall_n.csv")
    ops = read_csv_rows(bundle_dir / "steel_envelope_opensees.csv")
    stress_path = fig_dir / "continuum_external_steel_stress_envelope.svg"
    strain_path = fig_dir / "continuum_external_steel_strain_envelope.svg"
    write_metric_svg(
        stress_path,
        title="Continuum steel stress envelope",
        x_label="Tip drift [mm]",
        y_label="max |steel stress| [MPa]",
        falln=steel_metric_points(falln, "max_abs_stress_MPa"),
        opensees=steel_metric_points(ops, "max_abs_stress_MPa"),
    )
    write_metric_svg(
        strain_path,
        title="Continuum steel strain envelope",
        x_label="Tip drift [mm]",
        y_label="max |steel strain| [-]",
        falln=steel_metric_points(falln, "max_abs_strain"),
        opensees=steel_metric_points(ops, "max_abs_strain"),
    )
    return {
        "steel_stress_svg": str(stress_path),
        "steel_strain_svg": str(strain_path),
    }


def plot_steel_envelope_matplotlib(bundle_dir: Path, fig_dir: Path) -> dict[str, str]:
    import matplotlib.pyplot as plt

    falln = read_csv_rows(bundle_dir / "steel_envelope_fall_n.csv")
    ops = read_csv_rows(bundle_dir / "steel_envelope_opensees.csv")
    outputs: dict[str, str] = {}
    specs = (
        (
            "max_abs_stress_MPa",
            "max |steel stress| [MPa]",
            "Continuum steel stress envelope",
            "continuum_external_steel_stress_envelope",
            "steel_stress",
        ),
        (
            "max_abs_strain",
            "max |steel strain| [-]",
            "Continuum steel strain envelope",
            "continuum_external_steel_strain_envelope",
            "steel_strain",
        ),
    )
    for key, ylabel, title, stem, artifact_key in specs:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.plot(
            [1000.0 * float(row["drift_m"]) for row in falln],
            [float(row[key]) for row in falln],
            label="fall_n continuum",
            linewidth=1.4,
        )
        ax.plot(
            [1000.0 * float(row["drift_m"]) for row in ops],
            [float(row[key]) for row in ops],
            label="OpenSees continuum",
            linestyle="--",
            linewidth=1.2,
        )
        ax.set_xlabel("Tip drift [mm]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title(title)
        for ext in ("png", "pdf"):
            out = fig_dir / f"{stem}.{ext}"
            fig.savefig(out, bbox_inches="tight", dpi=250)
            outputs[f"{artifact_key}_{ext}"] = str(out)
        plt.close(fig)
    return outputs


def maybe_plot(bundle_dir: Path, summary: dict[str, object]) -> None:
    falln = read_csv_rows(bundle_dir / "fall_n" / "hysteresis.csv")
    ops = read_csv_rows(bundle_dir / "opensees" / "hysteresis.csv")
    fig_dir = bundle_dir / "figures"
    ensure_dir(fig_dir)
    svg_path = fig_dir / "continuum_external_hysteresis.svg"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        write_hysteresis_svg(svg_path, falln, ops)
        steel_figures = write_steel_envelope_svgs(bundle_dir, fig_dir)
        summary["plot_status"] = f"svg_fallback: {exc}"
        summary["figures"] = {"hysteresis_svg": str(svg_path), **steel_figures}
        return

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(
        [1000.0 * float(row["drift_m"]) for row in falln],
        [1000.0 * float(row["base_shear_MN"]) for row in falln],
        label="fall_n continuum",
        linewidth=1.4,
    )
    ax.plot(
        [1000.0 * float(row["drift_m"]) for row in ops],
        [1000.0 * float(row["base_shear_MN"]) for row in ops],
        label="OpenSees continuum",
        linestyle="--",
        linewidth=1.2,
    )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_title("Continuum external benchmark")
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"continuum_external_hysteresis.{ext}", bbox_inches="tight", dpi=250)
    plt.close(fig)
    write_hysteresis_svg(svg_path, falln, ops)
    steel_figures = write_steel_envelope_svgs(bundle_dir, fig_dir)
    steel_raster_figures = plot_steel_envelope_matplotlib(bundle_dir, fig_dir)
    summary["plot_status"] = "completed"
    summary["figures"] = {
        "hysteresis_png": str(fig_dir / "continuum_external_hysteresis.png"),
        "hysteresis_pdf": str(fig_dir / "continuum_external_hysteresis.pdf"),
        "hysteresis_svg": str(svg_path),
        **steel_figures,
        **steel_raster_figures,
    }


def try_write_partial_bundle(
    root: Path,
    falln_dir: Path,
    opensees_dir: Path,
    *,
    status: str,
    failed_stage: str,
    stages: list[dict[str, object]],
    falln_elapsed: float,
    opensees_elapsed: float,
) -> bool:
    required = (
        falln_dir / "runtime_manifest.json",
        opensees_dir / "reference_manifest.json",
        falln_dir / "hysteresis.csv",
        opensees_dir / "hysteresis.csv",
        falln_dir / "control_state.csv",
        opensees_dir / "control_state.csv",
    )
    if any(not path.exists() for path in required):
        return False

    try:
        falln_manifest = read_json(falln_dir / "runtime_manifest.json")
        opensees_manifest = read_json(opensees_dir / "reference_manifest.json")
        falln_h = read_csv_rows(falln_dir / "hysteresis.csv")
        ops_h = read_csv_rows(opensees_dir / "hysteresis.csv")
        falln_control = read_csv_rows(falln_dir / "control_state.csv")
        ops_control = read_csv_rows(opensees_dir / "control_state.csv")
        falln_steel = steel_envelope(
            read_csv_rows_optional(falln_dir / "rebar_history.csv"),
            "stress_xx_MPa",
            "axial_strain",
        )
        ops_steel = steel_envelope(
            read_csv_rows_optional(opensees_dir / "steel_bar_response.csv"),
            "axial_stress_MPa",
            "axial_strain",
        )
        write_csv(
            root / "steel_envelope_fall_n.csv",
            tuple(falln_steel[0].keys()) if falln_steel else ("step",),
            falln_steel,
        )
        write_csv(
            root / "steel_envelope_opensees.csv",
            tuple(ops_steel[0].keys()) if ops_steel else ("step",),
            ops_steel,
        )
        timing_rows = [
            {
                "tool": "fall_n",
                "process_wall_seconds": falln_elapsed,
                "reported_total_wall_seconds": dict(falln_manifest.get("timing", {})).get("total_wall_seconds", math.nan),
            },
            {
                "tool": "OpenSeesPy",
                "process_wall_seconds": opensees_elapsed,
                "reported_total_wall_seconds": dict(opensees_manifest.get("timing", {})).get("total_wall_seconds", math.nan),
            },
        ]
        write_csv(root / "timing_summary.csv", ("tool", "process_wall_seconds", "reported_total_wall_seconds"), timing_rows)
        summary: dict[str, object] = {
            "status": status,
            "failed_stage": failed_stage,
            "benchmark_scope": "continuum_external_computational_reference",
            "fall_n": {"dir": str(falln_dir), "manifest": falln_manifest, "process_wall_seconds": falln_elapsed},
            "opensees": {"dir": str(opensees_dir), "manifest": opensees_manifest, "process_wall_seconds": opensees_elapsed},
            "comparison": {
                "alignment": (
                    "Partial failed bundles are still matched by actual imposed "
                    "tip drift, not by step index."
                ),
                "hysteresis_base_shear_at_matched_drift": compare_series_at_matching_axis(
                    falln_h,
                    ops_h,
                    "base_shear_MN",
                ),
                "control_base_axial_reaction_at_matched_drift": compare_series_at_matching_axis(
                    falln_control,
                    ops_control,
                    "base_axial_reaction_MN",
                    lhs_axis_key="avg_top_face_total_dx_m",
                    rhs_axis_key="actual_tip_drift_m",
                ),
                "steel_max_abs_stress_at_matched_drift": compare_series_at_matching_axis(
                    falln_steel,
                    ops_steel,
                    "max_abs_stress_MPa",
                ),
                "steel_max_abs_strain_at_matched_drift": compare_series_at_matching_axis(
                    falln_steel,
                    ops_steel,
                    "max_abs_strain",
                ),
            },
            "artifacts": {
                "fall_n_hysteresis": str(falln_dir / "hysteresis.csv"),
                "opensees_hysteresis": str(opensees_dir / "hysteresis.csv"),
                "fall_n_steel_envelope": str(root / "steel_envelope_fall_n.csv"),
                "opensees_steel_envelope": str(root / "steel_envelope_opensees.csv"),
                "timing_summary": str(root / "timing_summary.csv"),
            },
            "stages": stages,
        }
        maybe_plot(root, summary)
        write_json(root / "continuum_external_benchmark_summary.json", summary)
    except Exception as exc:
        write_json(
            root / "continuum_external_benchmark_summary.json",
            {
                "status": "failed",
                "failed_stage": failed_stage,
                "partial_postprocess_error": repr(exc),
                "stages": stages,
            },
        )
        return False
    return True


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root = args.output_dir.resolve()
    falln_dir = root / "fall_n"
    opensees_dir = root / "opensees"
    ensure_dir(root)
    ensure_dir(falln_dir)
    ensure_dir(opensees_dir)

    stages: list[dict[str, object]] = []

    falln_manifest_path = falln_dir / "runtime_manifest.json"
    if args.reuse and completed_manifest(falln_manifest_path):
        falln_elapsed = 0.0
        falln_return = 0
    else:
        cmd = falln_command(args, falln_dir)
        falln_elapsed, proc = run_command(cmd, repo_root)
        (falln_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (falln_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        falln_return = proc.returncode
        stages.append({"stage": "fall_n", "command": cmd, "return_code": proc.returncode})
    if falln_return != 0:
        write_json(root / "continuum_external_benchmark_summary.json", {"status": "failed", "failed_stage": "fall_n", "stages": stages})
        return falln_return

    opensees_manifest_path = opensees_dir / "reference_manifest.json"
    if args.reuse and completed_manifest(opensees_manifest_path):
        opensees_elapsed = 0.0
        opensees_return = 0
    else:
        cmd = opensees_command(args, opensees_dir, repo_root)
        opensees_elapsed, proc = run_command(cmd, repo_root)
        (opensees_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
        (opensees_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        opensees_return = proc.returncode
        stages.append({"stage": "opensees", "command": cmd, "return_code": proc.returncode})
    if opensees_return != 0:
        if try_write_partial_bundle(
            root,
            falln_dir,
            opensees_dir,
            status="failed_with_partial_artifacts",
            failed_stage="opensees",
            stages=stages,
            falln_elapsed=falln_elapsed,
            opensees_elapsed=opensees_elapsed,
        ):
            return opensees_return
        write_json(root / "continuum_external_benchmark_summary.json", {"status": "failed", "failed_stage": "opensees", "stages": stages})
        return opensees_return

    falln_manifest = read_json(falln_manifest_path)
    opensees_manifest = read_json(opensees_manifest_path)
    falln_h = read_csv_rows(falln_dir / "hysteresis.csv")
    ops_h = read_csv_rows(opensees_dir / "hysteresis.csv")
    falln_control = read_csv_rows(falln_dir / "control_state.csv")
    ops_control = read_csv_rows(opensees_dir / "control_state.csv")
    falln_steel = steel_envelope(
        read_csv_rows_optional(falln_dir / "rebar_history.csv"),
        "stress_xx_MPa",
        "axial_strain",
    )
    ops_steel = steel_envelope(
        read_csv_rows_optional(opensees_dir / "steel_bar_response.csv"),
        "axial_stress_MPa",
        "axial_strain",
    )
    write_csv(
        root / "steel_envelope_fall_n.csv",
        tuple(falln_steel[0].keys()) if falln_steel else ("step",),
        falln_steel,
    )
    write_csv(
        root / "steel_envelope_opensees.csv",
        tuple(ops_steel[0].keys()) if ops_steel else ("step",),
        ops_steel,
    )
    timing_rows = [
        {
            "tool": "fall_n",
            "process_wall_seconds": falln_elapsed,
            "reported_total_wall_seconds": dict(falln_manifest.get("timing", {})).get("total_wall_seconds", math.nan),
        },
        {
            "tool": "OpenSeesPy",
            "process_wall_seconds": opensees_elapsed,
            "reported_total_wall_seconds": dict(opensees_manifest.get("timing", {})).get("total_wall_seconds", math.nan),
        },
    ]
    write_csv(root / "timing_summary.csv", ("tool", "process_wall_seconds", "reported_total_wall_seconds"), timing_rows)

    summary: dict[str, object] = {
        "status": "completed",
        "benchmark_scope": "continuum_external_computational_reference",
        "analysis": args.analysis,
        "mesh": {
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "bias": args.longitudinal_bias_power,
            "bias_location": args.longitudinal_bias_location,
            "falln_hex_order": args.falln_hex_order,
            "opensees_solid_element": args.solid_element,
        },
        "model_controls": {
            "fall_n": {
                "reinforcement_mode": args.falln_reinforcement_mode,
                "transverse_reinforcement_mode": args.falln_transverse_reinforcement_mode,
                "rebar_interpolation": args.falln_rebar_interpolation,
                "rebar_layout": args.falln_rebar_layout,
                "host_concrete_zoning_mode": args.falln_host_concrete_zoning_mode,
                "transverse_mesh_mode": args.falln_transverse_mesh_mode,
                "top_cap_mode": args.falln_top_cap_mode,
                "axial_preload_transfer_mode": args.falln_axial_preload_transfer_mode,
            },
            "OpenSeesPy": {
                "reinforcement_mode": args.opensees_reinforcement_mode,
                "host_concrete_zoning_mode": args.opensees_host_concrete_zoning_mode,
                "lateral_control_mode": args.opensees_lateral_control_mode,
            },
        },
        "fall_n": {"dir": str(falln_dir), "manifest": falln_manifest, "process_wall_seconds": falln_elapsed},
        "opensees": {"dir": str(opensees_dir), "manifest": opensees_manifest, "process_wall_seconds": opensees_elapsed},
        "comparison": {
            "alignment": (
                "Series are matched by actual imposed tip drift, not by step "
                "index, because fall_n may persist accepted continuation "
                "substeps while OpenSeesPy stores only the requested protocol "
                "points."
            ),
            "hysteresis_base_shear_at_matched_drift": compare_series_at_matching_axis(
                falln_h,
                ops_h,
                "base_shear_MN",
            ),
            "control_base_axial_reaction_at_matched_drift": compare_series_at_matching_axis(
                falln_control,
                ops_control,
                "base_axial_reaction_MN",
                lhs_axis_key="avg_top_face_total_dx_m",
                rhs_axis_key="actual_tip_drift_m",
            ),
            "steel_max_abs_stress_at_matched_drift": compare_series_at_matching_axis(
                falln_steel,
                ops_steel,
                "max_abs_stress_MPa",
            ),
            "steel_max_abs_strain_at_matched_drift": compare_series_at_matching_axis(
                falln_steel,
                ops_steel,
                "max_abs_strain",
            ),
        },
        "artifacts": {
            "fall_n_hysteresis": str(falln_dir / "hysteresis.csv"),
            "opensees_hysteresis": str(opensees_dir / "hysteresis.csv"),
            "fall_n_steel_envelope": str(root / "steel_envelope_fall_n.csv"),
            "opensees_steel_envelope": str(root / "steel_envelope_opensees.csv"),
            "timing_summary": str(root / "timing_summary.csv"),
        },
        "stages": stages,
    }
    maybe_plot(root, summary)
    write_json(root / "continuum_external_benchmark_summary.json", summary)
    print(
        "Continuum external benchmark completed:",
        f"fall_n={timing_rows[0]['reported_total_wall_seconds']}",
        f"OpenSees={timing_rows[1]['reported_total_wall_seconds']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
