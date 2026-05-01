"""
Plan v2 §Fase 0.3 — emit `data/output/validation_reboot/audit_phase{N}.json`

For a given phase label (`phase0`..`phase6`), runs the corresponding ctest
label, parses the result, and serialises a summary linking each test to the
canonical workstream rows in
`src/validation/ValidationCampaignCatalog.hh::canonical_validation_reboot_workstream_table()`.

Usage
-----
    python scripts/run_validation_gate.py phase0
    python scripts/run_validation_gate.py phase3 --build-dir build --timeout 1800

Exits 0 iff every test in the label passes.

Design notes
------------
This audit is intentionally *evidence-aggregating*, not gate-redefining: the
gate semantics (RMS tolerances, representative pass counts, etc.) live in the
C++ catalogs (`ReducedRCColumnEvidenceClosureCatalog.hh`,
`ReducedRCMultiscaleReadinessGate.hh`, etc.) and in the test bodies
themselves. This script's job is to (a) run the labelled subset, (b) collect
PASS/FAIL with elapsed time, and (c) annotate each test with its workstream
metadata so the resulting JSON is self-describing for the audit trail.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WorkstreamRow:
    """Mirror of `ValidationCampaignWorkstreamRow` (subset relevant to audit)."""

    row_label: str
    module_label: str
    objective_label: str
    phase_kind: str
    priority_kind: str
    legacy_surface_disposition: str
    required_for_reference_structural_column: bool
    required_for_reference_continuum_column: bool
    required_for_full_structure_escalation: bool
    requires_new_implementation: bool
    uses_legacy_surfaces_only_as_input: bool


_PHASE_KIND_TO_LABEL = {
    "phase0_governance_and_legacy_quarantine": "phase0",
    "phase1_formulation_and_solver_audit":     "phase1",
    "phase2_material_and_section_baseline":    "phase2",
    "phase3_reduced_order_rc_column":          "phase3",
    "phase4_continuum_rc_column":              "phase4",
    "phase5_cross_model_equivalence":          "phase5",
    "phase6_full_structure_escalation":        "phase6",
}


def parse_workstream_table(catalog_path: Path) -> list[WorkstreamRow]:
    """Hand-rolled parser of the catalog's `make_validation_campaign_workstream_row`
    calls. Robust to whitespace; expects the canonical formatting used by the
    repository (which is the only form we need to support here)."""

    text = catalog_path.read_text(encoding="utf-8")
    # Carve out canonical_validation_reboot_workstream_table()
    fn_start = text.find("canonical_validation_reboot_workstream_table() noexcept")
    if fn_start < 0:
        raise RuntimeError(f"workstream table not found in {catalog_path}")
    fn_body = text[fn_start:]

    # Each row is a make_validation_campaign_workstream_row(...) call. Use a
    # state machine that respects parenthesis depth and string literals.
    rows: list[WorkstreamRow] = []
    i = 0
    n = len(fn_body)
    needle = "make_validation_campaign_workstream_row("
    while True:
        j = fn_body.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        depth = 1
        in_str = False
        esc = False
        start = k
        while k < n and depth > 0:
            c = fn_body[k]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
            k += 1
        body = fn_body[start:k - 1]
        rows.append(_parse_row_body(body))
        i = k
    return rows


_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _split_top_level_args(body: str) -> list[str]:
    """Split a function-call body on top-level commas (ignoring those inside
    parens or string literals)."""
    args: list[str] = []
    depth = 0
    in_str = False
    esc = False
    cur = []
    for c in body:
        if in_str:
            cur.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            cur.append(c)
            continue
        if c == '(':
            depth += 1
            cur.append(c)
            continue
        if c == ')':
            depth -= 1
            cur.append(c)
            continue
        if c == ',' and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        cur.append(c)
    if cur:
        args.append("".join(cur).strip())
    return args


def _join_string_concatenation(arg: str) -> str:
    """Concatenate adjacent C++ string literals: `"a" "b"` → `ab`."""
    pieces = _STRING_RE.findall(arg)
    if pieces:
        return "".join(pieces)
    return arg.strip()


def _parse_bool(arg: str) -> bool:
    s = arg.strip()
    if s == "true":
        return True
    if s == "false":
        return False
    raise ValueError(f"expected bool, got {arg!r}")


def _parse_enum(arg: str) -> str:
    """Strip `EnumName::` prefix from `ScopedEnumKind::value`."""
    s = arg.strip()
    return s.split("::")[-1]


def _parse_row_body(body: str) -> WorkstreamRow:
    args = _split_top_level_args(body)
    if len(args) != 14:
        raise RuntimeError(
            f"workstream row has {len(args)} args, expected 14: {body[:80]!r}…")
    return WorkstreamRow(
        row_label                                = _join_string_concatenation(args[0]),
        module_label                             = _join_string_concatenation(args[1]),
        objective_label                          = _join_string_concatenation(args[2]),
        # args[3] theory_anchor_label, args[4] planned_artifact_label,
        # args[5] legacy_input_surface_label intentionally captured below
        phase_kind                               = _parse_enum(args[6]),
        priority_kind                            = _parse_enum(args[7]),
        legacy_surface_disposition               = _parse_enum(args[8]),
        required_for_reference_structural_column = _parse_bool(args[9]),
        required_for_reference_continuum_column  = _parse_bool(args[10]),
        required_for_full_structure_escalation   = _parse_bool(args[11]),
        requires_new_implementation              = _parse_bool(args[12]),
        uses_legacy_surfaces_only_as_input       = _parse_bool(args[13]),
    )


@dataclass
class CtestResult:
    name: str
    status: str
    elapsed_seconds: Optional[float] = None


def _run_ctest_label(
    build_dir: Path, label: str, timeout: int, jobs: int) -> tuple[list[CtestResult], int]:
    """Run `ctest -L <label>` in `build_dir`; return (results, exit_code)."""

    cmd = [
        "ctest",
        "-L", label,
        "-j", str(jobs),
        "--timeout", str(timeout),
        "--no-compress-output",
        "--output-on-failure",
    ]
    print(f"[run_validation_gate] $ {shlex.join(cmd)}  (cwd={build_dir})",
          flush=True)
    proc = subprocess.run(
        cmd, cwd=build_dir, capture_output=True, text=True, errors="replace")
    out = proc.stdout + proc.stderr

    # CTest prints lines like:
    #   "        Start  74: reduced_rc_column_material_baseline"
    #   " 1/26 Test #74: reduced_rc_column_material_baseline ......   Passed    1.05 sec"
    # We extract the second form.
    line_re = re.compile(
        r"Test\s+#\d+:\s+(\S+)\s+\.+\s*\*{0,3}\s*"
        r"(Passed|Failed|Timeout|Skipped|Not Run|\*\*\*\S+)\s+([\d.]+)\s+sec")
    results: list[CtestResult] = []
    for m in line_re.finditer(out):
        results.append(CtestResult(
            name=m.group(1), status=m.group(2), elapsed_seconds=float(m.group(3))))

    if not results:
        # Fallback: list the tests we *would* have run, marking unknown.
        list_proc = subprocess.run(
            ["ctest", "-L", label, "-N"],
            cwd=build_dir, capture_output=True, text=True, errors="replace")
        for m in re.finditer(r"Test\s+#\d+:\s+(\S+)", list_proc.stdout):
            results.append(CtestResult(name=m.group(1), status="not_executed"))

    return results, proc.returncode


# Static mapping (test name → workstream row_label). The audit trail benefits
# from explicit anchors here rather than parsing free-form objective_label
# heuristics. This list is short, hand-curated, and easy to extend.
_TEST_TO_WORKSTREAM = {
    # phase0
    "validation_campaign_catalog":
        "governance_reset_and_evidence_protocol",
    # phase1 — solver / formulation audit lives in the catalogs, surfaced via
    # the catalog tests below; no dedicated single test, so we leave it empty
    # and document GAP=none-yet at the audit-emission stage.
    # phase2 — material & section baseline
    "reduced_rc_column_material_baseline":
        "uniaxial_material_and_section_baseline",
    "reduced_rc_column_section_baseline":
        "uniaxial_material_and_section_baseline",
    # phase3 — reduced-order RC column (matrix + sweeps)
    "reduced_rc_column_structural_matrix":
        "reduced_order_rc_column_matrix",
    "reduced_rc_column_moment_curvature_closure_matrix":
        "reduced_order_rc_column_matrix",
    "reduced_rc_column_node_refinement_study":
        "reduced_order_rc_column_matrix",
    "reduced_rc_column_cyclic_node_refinement_study":
        "reduced_order_rc_column_matrix",
    "reduced_rc_column_cyclic_continuation_sensitivity_study":
        "reduced_order_rc_column_matrix",
    "reduced_rc_column_quadrature_sensitivity_study":
        "beam_integration_family_extension",
    "reduced_rc_column_cyclic_quadrature_sensitivity_study":
        "beam_integration_family_extension",
    # phase4 — continuum RC column
    "reduced_rc_column_continuum_baseline":
        "continuum_rc_column_matrix",
    "reduced_rc_column_truss_baseline":
        "reinforcement_discretization_extension",
    "phase4_hysteretic_cycles":
        "continuum_rc_column_matrix",
    # phase5 — cross-model equivalence
    "reduced_rc_column_moment_curvature_closure":
        "reduced_vs_continuum_equivalence",
    "phase5_dynamic_verification":
        "reduced_vs_continuum_equivalence",
    # phase6 — full-structure escalation
    "phase6_mpi_scalability":
        "full_structure_escalation_gate",
}


def emit_audit(
    phase_label: str,
    rows: list[WorkstreamRow],
    test_results: list[CtestResult],
    output_path: Path) -> dict:
    by_label = {r.row_label: r for r in rows}
    phase_rows = [r for r in rows
                  if _PHASE_KIND_TO_LABEL.get(r.phase_kind) == phase_label]

    test_entries = []
    for tr in test_results:
        ws_label = _TEST_TO_WORKSTREAM.get(tr.name)
        ws = by_label.get(ws_label) if ws_label else None
        test_entries.append({
            "test_name": tr.name,
            "status": tr.status,
            "elapsed_seconds": tr.elapsed_seconds,
            "workstream_row_label": ws_label,
            "workstream_module_label": ws.module_label if ws else None,
            "workstream_priority_kind": ws.priority_kind if ws else None,
            "workstream_required_for_reference_structural_column":
                ws.required_for_reference_structural_column if ws else None,
            "workstream_required_for_reference_continuum_column":
                ws.required_for_reference_continuum_column if ws else None,
        })

    # GAP detection: workstreams in this phase with no test mapped.
    mapped = {e["workstream_row_label"] for e in test_entries
              if e["workstream_row_label"]}
    gaps = []
    for r in phase_rows:
        if r.row_label not in mapped:
            gaps.append({
                "workstream_row_label": r.row_label,
                "module_label": r.module_label,
                "priority_kind": r.priority_kind,
                "reason":
                    "no ctest in this label is currently mapped to this "
                    "workstream; review _TEST_TO_WORKSTREAM in "
                    "scripts/run_validation_gate.py and/or add a dedicated "
                    "test under tests/ that gates this row.",
            })

    n_pass = sum(1 for tr in test_results if tr.status == "Passed")
    n_total = len(test_results)
    audit = {
        "schema_version": 1,
        "generated_by": "scripts/run_validation_gate.py",
        "phase_label": phase_label,
        "workstream_rows_in_phase": [asdict(r) for r in phase_rows],
        "tests_executed": test_entries,
        "tests_passed_count": n_pass,
        "tests_total_count": n_total,
        "all_tests_passed":
            n_total > 0 and n_pass == n_total
            and all(e["status"] == "Passed" for e in test_entries),
        "workstream_gap_count": len(gaps),
        "workstream_gaps": gaps,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=sorted(set(_PHASE_KIND_TO_LABEL.values())))
    p.add_argument("--build-dir", default=str(_REPO_ROOT / "build"),
                   help="ctest build directory (default: %(default)s)")
    p.add_argument("--timeout", type=int, default=1800,
                   help="per-test timeout in seconds (default: %(default)s)")
    p.add_argument("--jobs", type=int, default=4,
                   help="ctest parallelism (default: %(default)s)")
    p.add_argument("--output", default=None,
                   help="output JSON path (default: "
                        "data/output/validation_reboot/audit_<phase>.json)")
    args = p.parse_args(argv)

    catalog_path = _REPO_ROOT / "src" / "validation" / "ValidationCampaignCatalog.hh"
    rows = parse_workstream_table(catalog_path)

    build_dir = Path(args.build_dir)
    if not build_dir.is_dir():
        print(f"[run_validation_gate] build dir not found: {build_dir}",
              file=sys.stderr)
        return 2

    test_results, _ = _run_ctest_label(
        build_dir, args.phase, args.timeout, args.jobs)

    out = Path(args.output) if args.output else (
        _REPO_ROOT / "data" / "output" / "validation_reboot"
        / f"audit_{args.phase}.json")
    audit = emit_audit(args.phase, rows, test_results, out)

    print(f"[run_validation_gate] wrote {out}")
    print(f"  tests: {audit['tests_passed_count']}/{audit['tests_total_count']} pass"
          f"   workstream_gaps: {audit['workstream_gap_count']}")
    if audit["workstream_gaps"]:
        for g in audit["workstream_gaps"]:
            print(f"   GAP {g['workstream_row_label']} "
                  f"({g['priority_kind']}) — {g['module_label']}")

    return 0 if audit["all_tests_passed"] and audit["workstream_gap_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
