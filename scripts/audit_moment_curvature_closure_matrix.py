"""
Plan v2 §Fase 0.6 — emit `audit_phase0_moment_curvature_closure_matrix.json`

Reads the matrix-summary CSV produced by
`fall_n_reduced_rc_column_moment_curvature_closure_matrix_test` (which already
sweeps the full N x quadrature_family product) and serialises a per-cell
coverage report linking each (N, quadrature) combination to its observed
errors.

This audit answers plan v2 §Fase 0.6: "ejecutar
`ReducedRCColumnMomentCurvatureClosureMatrix` para todos los
`(N in {2..10}, q in {GL, Lobatto, GRleft, GRright})` y registrar `eta_M` por
celda." It does not redefine the gate — that is owned by the test itself
(via `representative_closure_passes()`).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_MATRIX_DIR = (
    _REPO_ROOT / "data" / "output" / "cyclic_validation"
    / "reboot_moment_curvature_closure_matrix_full")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    cases_csv = _MATRIX_DIR / "moment_curvature_closure_matrix_cases.csv"
    summary_node_csv = (
        _MATRIX_DIR / "moment_curvature_closure_node_spread.csv")
    summary_quad_csv = (
        _MATRIX_DIR / "moment_curvature_closure_quadrature_spread.csv")
    overall_csv = (
        _MATRIX_DIR / "moment_curvature_closure_matrix_summary.csv")

    if not cases_csv.exists():
        # Fallback to whichever .csv files the matrix run actually produced.
        candidates = list(_MATRIX_DIR.glob("*.csv"))
        print(f"[audit] {cases_csv.name} not found; available CSVs in matrix_full:")
        for c in candidates:
            print(f"    {c.name}")
        return 2

    rows = _read_csv(cases_csv)
    summary_overall = _read_csv(overall_csv) if overall_csv.exists() else []

    expected_N = list(range(2, 11))
    expected_q = ["gauss_legendre", "gauss_lobatto",
                  "gauss_radau_left", "gauss_radau_right"]

    cell_index = {(int(r["beam_nodes"]),
                   r["beam_axis_quadrature_family"]): r for r in rows}

    cells = []
    missing = []
    for N in expected_N:
        for q in expected_q:
            r = cell_index.get((N, q))
            if r is None:
                missing.append({"beam_nodes": N, "quadrature_family": q})
                continue
            cells.append({
                "beam_nodes": N,
                "quadrature_family": q,
                "case_id": r.get("case_id"),
                "execution_ok": r.get("execution_ok") in ("1", "true"),
                "max_rel_moment_error": float(r.get("max_rel_moment_error", "nan")),
                "rms_rel_moment_error":
                    float(r["rms_rel_moment_error"])
                    if r.get("rms_rel_moment_error") else None,
                "max_rel_tangent_error": float(r.get("max_rel_tangent_error", "nan")),
                "max_rel_secant_error": float(r.get("max_rel_secant_error", "nan")),
                "max_rel_axial_force_error":
                    float(r.get("max_rel_axial_force_error", "nan")),
                "representative_closure_passes":
                    r.get("representative_closure_passes") in ("1", "true"),
            })

    n_pass = sum(1 for c in cells if c["representative_closure_passes"])

    audit = {
        "schema_version": 1,
        "phase_label": "phase0_moment_curvature_closure_matrix_coverage",
        "expected_beam_nodes": expected_N,
        "expected_quadrature_families": expected_q,
        "expected_cell_count": len(expected_N) * len(expected_q),
        "observed_cell_count": len(cells),
        "missing_cells": missing,
        "all_expected_cells_present": len(missing) == 0,
        "representative_pass_count": n_pass,
        "overall_summary":
            summary_overall[0] if summary_overall else None,
        "cells": cells,
    }

    out = (_REPO_ROOT / "data" / "output" / "validation_reboot"
           / "audit_phase0_moment_curvature_closure_matrix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[audit] wrote {out}")
    print(f"  expected {audit['expected_cell_count']} cells, "
          f"observed {audit['observed_cell_count']}, "
          f"missing {len(missing)}")
    print(f"  representative_pass_count: {n_pass}/{len(cells)}")
    return 0 if audit["all_expected_cells_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
