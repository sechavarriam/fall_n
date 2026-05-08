#!/usr/bin/env python3
"""Build the canonical cyclic-validation evidence inventory for Chapter 9.

The inventory is intentionally curated. It separates figures/results that can
support publication text from artifacts that are useful only as diagnostics.
The script depends only on the standard library so it can run on a clean
checkout before any plotting stack is available.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "doc" / "figures" / "validation_reboot"
OUT = FIG


def rel(path: Path | str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def metric(path: str, extractor: Callable[[dict[str, Any]], str]) -> str:
    data = read_json(path)
    if not data:
        return "backing JSON missing"
    try:
        return extractor(data)
    except (KeyError, TypeError, ValueError):
        return "metric unavailable in backing JSON"


@dataclass(frozen=True)
class Evidence:
    artifact: str
    label: str
    chapter9_slot: str
    state: str
    audit_level: str
    gates: str
    evidence_role: str
    backing_artifact: str
    metric_summary: str
    not_claimed: str

    def row(self) -> dict[str, str]:
        exists = (ROOT / self.artifact).exists()
        backing_exists = not self.backing_artifact or (ROOT / self.backing_artifact).exists()
        state = self.state if exists else "missing"
        return {
            "artifact": self.artifact,
            "label": self.label,
            "chapter9_slot": self.chapter9_slot,
            "state": state,
            "audit_level": self.audit_level,
            "gates": self.gates,
            "evidence_role": self.evidence_role,
            "backing_artifact": self.backing_artifact,
            "metric_summary": self.metric_summary,
            "not_claimed": self.not_claimed,
            "artifact_exists": str(exists).lower(),
            "backing_exists": str(backing_exists).lower(),
        }


def evidence_entries() -> list[Evidence]:
    return [
        Evidence(
            artifact="doc/figures/validation_reboot/reduced_rc_cyclic_load_protocol.pdf",
            label="fig:validacion-cyclic-load-protocol",
            chapter9_slot="cyclic load protocol",
            state="promovido",
            audit_level="protocolo/cinematica",
            gates="M2-M4",
            evidence_role="Applied cyclic drift protocol before local/global model comparison.",
            backing_artifact="data/output/cyclic_validation/protocol.pdf",
            metric_summary="protocol figure copied from the audited cyclic_validation bundle",
            not_claimed="Does not by itself validate any structural or local response.",
        ),
        Evidence(
            artifact="doc/figures/validation_reboot/reduced_rc_internal_hysteresis_200mm.pdf",
            label="fig:validacion-internal-hysteresis-200mm",
            chapter9_slot="structural hysteresis",
            state="promovido",
            audit_level="cinematica/energetica/algoritmica",
            gates="M3-M8",
            evidence_role="N=10 Lobatto structural reference under the 200 mm cyclic column protocol.",
            backing_artifact="doc/figures/validation_reboot/internal_hysteresis_200mm_summary.json",
            metric_summary=metric(
                "doc/figures/validation_reboot/internal_hysteresis_200mm_summary.json",
                lambda d: (
                    f"status={d.get('status')}; drift={d.get('max_abs_tip_drift_mm')} mm; "
                    f"points={d.get('hysteresis_point_count')}; "
                    f"wall={d.get('process_wall_seconds'):.1f} s"
                ),
            ),
            not_claimed="Does not prove local crack topology; it is the structural reference curve.",
        ),
        Evidence(
            artifact="doc/figures/validation_reboot/reduced_rc_external_structural_hysteresis.pdf",
            label="fig:validacion-external-structural-hysteresis",
            chapter9_slot="external comparator hysteresis",
            state="restringido",
            audit_level="cinematica/frontera",
            gates="M3-M8",
            evidence_role="External comparator check for the structural cyclic response.",
            backing_artifact="doc/figures/validation_reboot/continuum_external_hysteresis_200mm_panel_summary.json",
            metric_summary=metric(
                "doc/figures/validation_reboot/continuum_external_hysteresis_200mm_panel_summary.json",
                lambda d: f"status={d.get('status', 'completed')}; cases={len(d.get('cases', []))}",
            ),
            not_claimed="Does not make the external solver an absolute truth model.",
        ),
        Evidence(
            artifact=(
                "doc/figures/validation_reboot/"
                "xfem_ccb_bounded_dowelx_nz4_cap0p020_fy0p00190_vs_structural_200mm_hysteresis.pdf"
            ),
            label="fig:validacion-xfem-bounded-dowel-200mm",
            chapter9_slot="XFEM local hysteresis",
            state="promovido",
            audit_level="energetica_material/cinematica",
            gates="M4-M8",
            evidence_role="Promoted XFEM local baseline against the N=10 Lobatto structural reference.",
            backing_artifact=(
                "doc/figures/validation_reboot/"
                "xfem_ccb_bounded_dowelx_nz4_cap0p020_fy0p00190_vs_structural_200mm_summary.json"
            ),
            metric_summary=metric(
                "doc/figures/validation_reboot/"
                "xfem_ccb_bounded_dowelx_nz4_cap0p020_fy0p00190_vs_structural_200mm_summary.json",
                lambda d: (
                    f"gate={d['promotion_gate']['status']}; "
                    f"rms_error={d['metrics']['peak_normalized_rms_base_shear_error']:.3f}; "
                    f"peak_ratio={d['metrics']['xfem_to_structural_peak_base_shear_ratio']:.3f}; "
                    f"steel_peak={d['promotion_gate']['peak_abs_steel_stress_MPa']:.1f} MPa"
                ),
            ),
            not_claimed="Does not extrapolate outside the audited 200 mm cyclic column protocol.",
        ),
        Evidence(
            artifact="doc/figures/validation_reboot/structural_continuum_crack_gate_200mm.pdf",
            label="fig:validacion-continuum-crack-gate-200mm",
            chapter9_slot="crack opening and cracked Gauss evolution",
            state="diagnostico_util",
            audit_level="material/localizacion",
            gates="M4-M8",
            evidence_role="Continuum crack/opening gate including cracked/open material-point fractions.",
            backing_artifact="doc/figures/validation_reboot/structural_continuum_crack_gate_200mm_summary.json",
            metric_summary=metric(
                "doc/figures/validation_reboot/structural_continuum_crack_gate_200mm_summary.json",
                lambda d: (
                    f"records={d['record_count']}; "
                    f"max_opening={d['max_crack_opening_mm']:.3f} mm; "
                    f"first_full_open={d['first_full_open_cracking_equivalent_drift_mm']:.3f} mm"
                ),
            ),
            not_claimed="Does not promote the smeared continuum crack field as objective localization.",
        ),
        Evidence(
            artifact=(
                "doc/figures/validation_reboot/"
                "reduced_rc_structural_continuum_steel_stress_vs_drift_"
                "tensile_crack_band_damage_proxy_et0p1_200mm.pdf"
            ),
            label="fig:validacion-continuum-steel-stress-200mm",
            chapter9_slot="steel stress and yield ratio",
            state="diagnostico_util",
            audit_level="material/acero",
            gates="M4-M8",
            evidence_role="Steel stress-vs-drift diagnostic for the continuum cyclic column branch.",
            backing_artifact="doc/figures/validation_reboot/structural_continuum_cyclic_crack_band_200mm_axial_gate_summary.json",
            metric_summary=metric(
                "doc/figures/validation_reboot/structural_continuum_cyclic_crack_band_200mm_axial_gate_summary.json",
                lambda d: f"status={d.get('status', 'completed')}; cases={len(d.get('continuum_cases', []))}",
            ),
            not_claimed="Does not imply that the continuum branch is the promoted two-way local model.",
        ),
        Evidence(
            artifact="doc/figures/validation_reboot/reduced_rc_solver_policy_case_matrix_timing.pdf",
            label="fig:validacion-solver-policy-cost",
            chapter9_slot="solver diagnostics and cost",
            state="diagnostico_util",
            audit_level="algoritmica/coste",
            gates="M7-M8",
            evidence_role="Solver policy matrix used to separate physical failure from nonlinear-solver debt.",
            backing_artifact="doc/figures/validation_reboot/solver_policy_benchmark_summary.json",
            metric_summary=metric(
                "doc/figures/validation_reboot/solver_policy_benchmark_summary.json",
                lambda d: (
                    f"completed={d['completed_policy_count']}; failed={d['failed_policy_count']}; "
                    f"fastest={d['fastest_completed_policy']}; "
                    f"best_wall={d['best_process_wall_seconds']:.1f} s"
                ),
            ),
            not_claimed="Does not promote a solver policy without physical response gates.",
        ),
        Evidence(
            artifact="doc/figures/validation_reboot/fe2_one_way_managed_xfem_n10_lobatto_200mm_3x3x6_summary_panel.pdf",
            label="fig:validacion-fe2-one-way-managed-xfem-n10-200mm",
            chapter9_slot="FE2 one-way column replay",
            state="promovido",
            audit_level="acoplamiento/algoritmica",
            gates="M5-M8",
            evidence_role="Managed XFEM one-way replay using persistent local sites and the N=10 reference history.",
            backing_artifact="doc/figures/validation_reboot/fe2_one_way_managed_xfem_n10_lobatto_200mm_3x3x6_summary_panel.json",
            metric_summary=metric(
                "doc/figures/validation_reboot/fe2_one_way_managed_xfem_n10_lobatto_200mm_3x3x6_summary_panel.json",
                lambda d: (
                    f"pass={d['overall_pass']}; sites={d['selected_site_count']}; "
                    f"max_moment_error={d['max_relative_moment_error']:.4f}; "
                    f"max_snes={d['max_snes_iters']:.0f}"
                ),
            ),
            not_claimed="Does not alter the macro response; it validates replay and observation only.",
        ),
        Evidence(
            artifact="doc/figures/validation_reboot/kobathe_production_hex8_1x1x2_rebar_cyclic_50mm_fixedend_bias2.pdf",
            label="fig:validacion-kobathe-cyclic-bias-probe",
            chapter9_slot="Ko-Bathe comparator probe",
            state="diagnostico_util",
            audit_level="material/coste",
            gates="M4-M8",
            evidence_role="Small Ko-Bathe cyclic probe with embedded bars, used to audit crack activation and cost.",
            backing_artifact="data/output/fe2_validation/kobathe_cyclic_bias_probe_summary.json",
            metric_summary=metric(
                "data/output/fe2_validation/kobathe_cyclic_bias_probe_summary.json",
                lambda d: (
                    f"completed={d.get('completed')}; steps={d.get('runtime_steps')}; "
                    f"peak_cracked_gp={d.get('peak_cracked_gauss_points')}; "
                    f"max_opening={1000.0 * d.get('max_crack_opening_m', 0.0):.3f} mm; "
                    f"wall={d.get('solve_wall_seconds'):.1f} s"
                ),
            ),
            not_claimed="Does not promote Ko-Bathe as the main two-way local model.",
        ),
    ]


def required_figure_rows(entries: list[Evidence]) -> list[dict[str, str]]:
    rows = []
    for entry in entries:
        if entry.chapter9_slot in {
            "cyclic load protocol",
            "structural hysteresis",
            "XFEM local hysteresis",
            "crack opening and cracked Gauss evolution",
            "steel stress and yield ratio",
            "solver diagnostics and cost",
            "FE2 one-way column replay",
        }:
            row = entry.row()
            rows.append(
                {
                    "chapter9_slot": entry.chapter9_slot,
                    "artifact": entry.artifact,
                    "label": entry.label,
                    "state": row["state"],
                    "artifact_exists": row["artifact_exists"],
                    "backing_artifact": entry.backing_artifact,
                    "backing_exists": row["backing_exists"],
                    "metric_summary": entry.metric_summary,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    entries = evidence_entries()
    rows = [entry.row() for entry in entries]
    required_rows = required_figure_rows(entries)
    missing = [
        row["artifact"]
        for row in rows
        if row["artifact_exists"] != "true" or row["backing_exists"] != "true"
    ]

    inventory_csv = OUT / "cyclic_publication_evidence_inventory.csv"
    required_csv = OUT / "cyclic_publication_required_figures.csv"
    summary_json = OUT / "cyclic_publication_evidence_inventory.json"

    write_csv(inventory_csv, rows)
    write_csv(required_csv, required_rows)

    summary = {
        "schema": "fall_n_cyclic_publication_evidence_inventory_v1",
        "purpose": "canonical audited evidence for the Chapter 9 cyclic column validation before the full building case",
        "inventory_csv": rel(inventory_csv),
        "required_figures_csv": rel(required_csv),
        "entry_count": len(rows),
        "required_figure_count": len(required_rows),
        "missing_count": len(missing),
        "missing": missing,
        "state_counts": {},
        "entries": rows,
    }
    for row in rows:
        summary["state_counts"][row["state"]] = summary["state_counts"].get(row["state"], 0) + 1
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("inventory_csv", "required_figures_csv", "entry_count", "missing_count", "missing")}, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
