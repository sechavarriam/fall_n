#!/usr/bin/env python3
"""Audit section and fiber observable identity for the reduced RC cyclic matrix.

The promoted cyclic comparison uses global hysteresis and loop work.  This
script documents why direct moment-curvature and single-fiber overlays against
the 2D OpenSees hi-fi run are diagnostic unless section-axis and fiber identity
are explicitly matched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROLES = ("reinforcing_steel", "unconfined_concrete", "confined_concrete")


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    default_matrix = (
        repo
        / "data/output/cyclic_validation/"
        "timoshenko_matrix_reproduced_historical_closure_20260520"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=default_matrix)
    parser.add_argument(
        "--falln-cases",
        default="n04_lobatto,n10_lobatto",
        help="Comma-separated fall_n bundles under fall_n_matrix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo
        / "doc/figures/validation_reboot/"
        "reduced_rc_section_fiber_observable_audit.json",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def value_range(values: list[float]) -> dict[str, Any]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return {"min": math.nan, "max": math.nan, "count": 0}
    return {"min": min(clean), "max": max(clean), "count": len(clean)}


def base_section_gp(rows: list[dict[str, Any]]) -> str:
    return min({row["section_gp"] for row in rows}, key=lambda x: float(x))


def solve_3x3(a: list[list[float]], b: list[float]) -> list[float]:
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1.0e-30:
            return [math.nan, math.nan, math.nan]
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        scale = m[col][col]
        for j in range(col, 4):
            m[col][j] /= scale
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, 4):
                m[r][j] -= factor * m[col][j]
    return [m[i][3] for i in range(3)]


def fit_affine_strain(rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    base_gp = base_section_gp(rows)
    selected = [
        row
        for row in rows
        if row.get("section_gp") == base_gp
        and int(finite_float(row.get("step"), -1)) == step
        and math.isfinite(finite_float(row.get("strain_xx")))
    ]
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    max_abs_strain = 0.0
    for row in selected:
        x = [1.0, finite_float(row.get("y")), finite_float(row.get("z"))]
        eps = finite_float(row.get("strain_xx"))
        max_abs_strain = max(max_abs_strain, abs(eps))
        for i in range(3):
            atb[i] += x[i] * eps
            for j in range(3):
                ata[i][j] += x[i] * x[j]
    coeff = solve_3x3(ata, atb)
    max_error = 0.0
    for row in selected:
        x = [1.0, finite_float(row.get("y")), finite_float(row.get("z"))]
        eps = finite_float(row.get("strain_xx"))
        pred = coeff[0] + coeff[1] * x[1] + coeff[2] * x[2]
        max_error = max(max_error, abs(pred - eps))
    sample = selected[0] if selected else {}
    curvature_y = finite_float(sample.get("curvature_y"))
    y_residual = abs(coeff[1] + curvature_y) if math.isfinite(curvature_y) else math.nan
    z_residual = abs(coeff[2] + curvature_y) if math.isfinite(curvature_y) else math.nan
    if math.isfinite(y_residual) and math.isfinite(z_residual):
        active_axis = "y" if y_residual < z_residual else "z"
    else:
        active_axis = "unknown"
    return {
        "step": step,
        "p": finite_float(sample.get("p")),
        "drift_mm": 1000.0 * finite_float(sample.get("drift_m")),
        "base_section_gp": base_gp,
        "fiber_count": len(selected),
        "csv_axial_strain": finite_float(sample.get("axial_strain")),
        "csv_curvature_y": curvature_y,
        "fit_axial_strain": coeff[0],
        "fit_coeff_y": coeff[1],
        "fit_coeff_z": coeff[2],
        "fit_max_abs_error": max_error,
        "fit_max_abs_strain": max_abs_strain,
        "axis_inferred_from_fiber_strains": active_axis,
    }


def representative_fiber(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
    base_gp = base_section_gp(rows)
    base = [row for row in rows if row.get("section_gp") == base_gp]
    first_step = min(int(finite_float(row.get("step"), 0)) for row in base)
    candidates = [
        row
        for row in base
        if row.get("material_role") == role
        and int(finite_float(row.get("step"), -1)) == first_step
    ]
    if not candidates:
        return {}
    picked = max(
        candidates,
        key=lambda row: (
            abs(finite_float(row.get("z"))),
            abs(finite_float(row.get("y"))),
            -finite_float(row.get("fiber_index"), 1.0e9),
        ),
    )
    fiber_index = picked.get("fiber_index")
    history = [
        row
        for row in base
        if row.get("material_role") == role and row.get("fiber_index") == fiber_index
    ]
    return {
        "fiber_index": int(finite_float(fiber_index, -1)),
        "y": finite_float(picked.get("y")),
        "z": finite_float(picked.get("z")),
        "zone": picked.get("zone", ""),
        "strain_range": value_range([finite_float(row.get("strain_xx")) for row in history]),
        "stress_MPa_range": value_range(
            [finite_float(row.get("stress_xx_MPa")) for row in history]
        ),
    }


def summarize_case(name: str, fiber_csv: Path, moment_csv: Path | None) -> dict[str, Any]:
    fibers = read_csv(fiber_csv)
    if moment_csv and not moment_csv.exists():
        fallback = moment_csv.parent / "section_response.csv"
        moment_csv = fallback if fallback.exists() else moment_csv
    steps = sorted({int(finite_float(row.get("step"), 0)) for row in fibers})
    probe_steps = sorted({steps[0], steps[len(steps) // 2], steps[-1]})
    moment_rows = read_csv(moment_csv) if moment_csv and moment_csv.exists() else []
    base_gp = base_section_gp(fibers)
    moment_base_rows = [
        row for row in moment_rows if str(row.get("section_gp", base_gp)) == str(base_gp)
    ]
    return {
        "name": name,
        "fiber_csv": str(fiber_csv),
        "moment_curvature_csv": str(moment_csv) if moment_csv else "",
        "state_count": len(steps),
        "base_section_gp": base_gp,
        "curvature_y_range": value_range(
            [finite_float(row.get("curvature_y")) for row in moment_base_rows]
        ),
        "moment_y_MNm_range": value_range(
            [finite_float(row.get("moment_y_MNm")) for row in moment_base_rows]
        ),
        "affine_strain_fits": [fit_affine_strain(fibers, step) for step in probe_steps],
        "representative_fibers": {
            role: representative_fiber(fibers, role) for role in ROLES
        },
    }


def main() -> int:
    args = parse_args()
    falln_root = args.matrix_dir / "fall_n_matrix"
    cases = []
    for name in [part.strip() for part in args.falln_cases.split(",") if part.strip()]:
        bundle = falln_root / name
        cases.append(
            summarize_case(
                name,
                bundle / "comparison_section_fiber_state_history.csv",
                bundle / "comparison_moment_curvature_base.csv",
            )
        )
    opensees = args.matrix_dir / "opensees_hifi_reference"
    cases.append(
        summarize_case(
            "opensees_hifi_reference",
            opensees / "section_fiber_state_history.csv",
            opensees / "moment_curvature_base.csv",
        )
    )
    payload = {
        "matrix_dir": str(args.matrix_dir),
        "mechanical_vtk_interference_check": (
            "The cyclic matrix VTK files are generated by postprocess from "
            "already committed CSV rows, so they do not enter Newton residuals, "
            "tangents, bisections, or material commits."
        ),
        "cases": cases,
        "diagnosis": [
            "fall_n fiber strains fit eps0 - z*kappa_y, matching the FiberSection convention.",
            "The 2D OpenSees hi-fi reference fiber strains fit eps0 - y*kappa, so its section curve is a 2D projection.",
            "Direct fiber overlays are diagnostic unless the section axis, station, fiber coordinate, and commit history are all matched.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
