#!/usr/bin/env python3
"""Modal parity audit for the fall_n 16-storey L-shaped seismic benchmark.

The C++ driver exports the assembled tangent stiffness together with two mass
models: the native consistent element mass and a diagnostic primary-nodal mass
that preserves the same total translational inertia.  This script solves the
lowest generalized modes after statically eliminating massless degrees of
freedom.  The audit is intentionally diagnostic: it separates element stiffness
from inertial representation before drawing conclusions from time histories.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh, splu


ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "data/output/lshaped_multiscale_16/recorders"
FIG = ROOT / "doc/figures/validation_reboot"


@dataclass(frozen=True)
class ModalCase:
    label: str
    mass_file: Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def symmetrize(a: csr_matrix) -> csr_matrix:
    return (0.5 * (a + a.T)).tocsr()


def positive_mass_dofs(m: csr_matrix, rel_tol: float = 1.0e-11) -> np.ndarray:
    diag = np.asarray(m.diagonal(), dtype=float)
    max_diag = float(np.max(np.abs(diag))) if diag.size else 0.0
    tol = max(1.0e-18, rel_tol * max_diag)
    return np.flatnonzero(diag > tol)


def solve_condensed_modes(
    k_full: csr_matrix,
    m_full: csr_matrix,
    n_modes: int = 8,
) -> dict:
    """Solve K phi = lambda M phi after eliminating massless dofs.

    A full sparse factorization of K provides the shift-invert inverse of the
    condensed operator.  A smaller K_rr factorization supplies the direct Schur
    complement matvec used by ARPACK for consistency checks.
    """

    t0 = time.perf_counter()
    n = k_full.shape[0]
    mass = positive_mass_dofs(m_full)
    if mass.size == 0:
        raise RuntimeError("Mass matrix has no positive-mass degrees of freedom.")
    mass_set = np.zeros(n, dtype=bool)
    mass_set[mass] = True
    massless = np.flatnonzero(~mass_set)

    k_tt = k_full[mass][:, mass].tocsr()
    k_tr = k_full[mass][:, massless].tocsr()
    k_rt = k_full[massless][:, mass].tocsr()
    k_rr = k_full[massless][:, massless].tocsc()
    m_tt = m_full[mass][:, mass].tocsr()

    regularization = 0.0
    try:
        lu_rr = splu(k_rr)
    except RuntimeError:
        # This should not be the default path.  It is a diagnostic fallback for
        # nearly-mechanism-like audits, and the regularization is recorded.
        regularization = max(1.0e-12, 1.0e-12 * float(np.max(np.abs(k_rr.diagonal()))))
        lu_rr = splu((k_rr + regularization * eye(k_rr.shape[0], format="csc")).tocsc())

    k_factor_time = time.perf_counter()
    lu_full = splu(k_full.tocsc())
    full_factor_time = time.perf_counter()

    def condensed_matvec(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v)
        return k_tt @ v - k_tr @ lu_rr.solve(k_rt @ v)

    def condensed_inverse(v: np.ndarray) -> np.ndarray:
        rhs = np.zeros(n, dtype=float)
        rhs[mass] = v
        sol = lu_full.solve(rhs)
        return sol[mass]

    a_op = LinearOperator((mass.size, mass.size), matvec=condensed_matvec, dtype=float)
    inv_op = LinearOperator((mass.size, mass.size), matvec=condensed_inverse, dtype=float)

    eig_start = time.perf_counter()
    try:
        vals, vecs = eigsh(
            a_op,
            k=min(n_modes + 4, mass.size - 2),
            M=m_tt,
            sigma=0.0,
            which="LM",
            OPinv=inv_op,
            tol=1.0e-8,
            maxiter=2000,
        )
        converged = True
    except ArpackNoConvergence as exc:
        vals = exc.eigenvalues
        vecs = exc.eigenvectors
        converged = False
    eig_time = time.perf_counter()

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 1.0e-12)]
    vals.sort()
    vals = vals[:n_modes]
    periods = [2.0 * math.pi / math.sqrt(v) for v in vals]

    return {
        "converged": converged,
        "matrix_size": n,
        "positive_mass_dofs": int(mass.size),
        "massless_condensed_dofs": int(massless.size),
        "regularization": regularization,
        "periods_s": periods,
        "eigenvalues": vals.tolist(),
        "factor_krr_seconds": k_factor_time - t0,
        "factor_full_k_seconds": full_factor_time - k_factor_time,
        "eigsh_seconds": eig_time - eig_start,
        "total_seconds": eig_time - t0,
    }


def read_opensees_periods() -> dict[str, list[float]]:
    base = ROOT / "data/output/opensees_lshaped_16storey"
    manifests = {
        "OpenSees nodal mass": base
        / "eigen_elastic_timoshenko_sub1_nodal/opensees_lshaped_16storey_manifest.json",
        "OpenSees element mass": base
        / "eigen_elastic_timoshenko_sub1_element/opensees_lshaped_16storey_manifest.json",
        "OpenSees element mass sub3": base
        / "eigen_elastic_timoshenko_sub3_element/opensees_lshaped_16storey_manifest.json",
    }
    out: dict[str, list[float]] = {}
    for label, path in manifests.items():
        data = load_json(path)
        if "eigen_periods_s" in data:
            out[label] = data["eigen_periods_s"][:6]
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["case", "mode", "period_s"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_periods(period_sets: dict[str, list[float]], path_pdf: Path, path_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    modes = np.arange(1, 7)
    for label, periods in period_sets.items():
        y = np.asarray(periods[:6], dtype=float)
        if y.size == 0:
            continue
        ax.plot(modes[: y.size], y, marker="o", linewidth=1.6, label=label)
    ax.set_xlabel("Mode")
    ax.set_ylabel("Period [s]")
    ax.set_title("16-storey L-shaped modal audit: fall_n vs OpenSees")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=180)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    summary = load_json(REC / "falln_modal_matrix_export_summary.json")
    k = symmetrize(mmread(REC / summary["stiffness_matrix_market"]).tocsr())
    element_policy = summary.get("active_element_mass_policy", "consistent")
    element_label = f"fall_n {element_policy} element mass"
    cases = [
        ModalCase(element_label, REC / summary["element_mass_matrix_market"]),
        ModalCase("fall_n primary-nodal mass", REC / summary["primary_nodal_mass_matrix_market"]),
    ]

    result: dict[str, object] = {
        "schema": "falln_lshaped_modal_parity_audit_v1",
        "matrix_export_summary": str((REC / "falln_modal_matrix_export_summary.json").relative_to(ROOT)),
        "cases": {},
    }
    period_sets: dict[str, list[float]] = {}
    rows: list[dict] = []

    for case in cases:
        m = symmetrize(mmread(case.mass_file).tocsr())
        solved = solve_condensed_modes(k, m, n_modes=8)
        result["cases"][case.label] = solved
        period_sets[case.label] = solved["periods_s"][:6]
        for i, period in enumerate(solved["periods_s"][:6], start=1):
            rows.append({"case": case.label, "mode": i, "period_s": f"{period:.12e}"})

    opensees = read_opensees_periods()
    result["opensees_reference_periods_s"] = opensees
    for label, periods in opensees.items():
        period_sets[label] = periods[:6]
        for i, period in enumerate(periods[:6], start=1):
            rows.append({"case": label, "mode": i, "period_s": f"{period:.12e}"})

    if "fall_n primary-nodal mass" in period_sets and "OpenSees nodal mass" in period_sets:
        fn = np.asarray(period_sets["fall_n primary-nodal mass"][:3], dtype=float)
        os = np.asarray(period_sets["OpenSees nodal mass"][:3], dtype=float)
        result["first_three_period_relative_error_vs_opensees_nodal"] = (
            np.abs(fn - os) / np.maximum(np.abs(os), 1.0e-14)
        ).tolist()
    if element_label in period_sets and "OpenSees element mass" in period_sets:
        fn = np.asarray(period_sets[element_label][:3], dtype=float)
        os = np.asarray(period_sets["OpenSees element mass"][:3], dtype=float)
        result["first_three_period_relative_error_vs_opensees_element"] = (
            np.abs(fn - os) / np.maximum(np.abs(os), 1.0e-14)
        ).tolist()

    summary_path = FIG / "lshaped_16_falln_modal_condensation_summary.json"
    csv_path = FIG / "lshaped_16_falln_modal_periods.csv"
    pdf_path = FIG / "lshaped_16_modal_period_comparison.pdf"
    png_path = FIG / "lshaped_16_modal_period_comparison.png"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    plot_periods(period_sets, pdf_path, png_path)

    print(f"Wrote {summary_path.relative_to(ROOT)}")
    print(f"Wrote {csv_path.relative_to(ROOT)}")
    print(f"Wrote {pdf_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
