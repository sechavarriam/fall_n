"""
Plan v2 §Fase 0.4 — Legacy quarantine audit.

Verifies that the validation surfaces marked
`quarantine_to_old_when_replacement_exists` or
`exploratory_only_do_not_anchor_claims` in
`canonical_validation_reboot_workstream_table()` are not anchoring any active
test in the ctest harness.

Anchoring criterion (conservative): a file is "anchored" iff it appears in the
SOURCE list of any `fall_n_add_test(...)` call in `CMakeLists.txt`. A file that
is merely linked into a hand-rolled `main_*.cpp` executable does NOT count as
anchoring an automated claim — those drivers are exploratory by definition,
not part of the gate suite. The audit also accepts files that compile into
intermediate libraries with no test dependency.

Output: data/output/validation_reboot/audit_phase0_legacy_quarantine.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Files cited by the catalog as legacy surfaces, tagged with the disposition
# the catalog assigns to them. The audit fails iff a surface marked
# `quarantine_to_old_when_replacement_exists` or
# `exploratory_only_do_not_anchor_claims` is anchoring an active ctest target.
# Surfaces marked `keep_only_as_audited_input` are intentionally allowed to
# remain as tests (they are validated inputs, not legacy claims).
_LEGACY_SURFACES = [
    # Quarantine candidates (must NOT anchor active ctest claims):
    ("ComprehensiveCyclicValidation.cpp",
        "quarantine_to_old_when_replacement_exists"),
    ("TableCyclicValidationStructural.cpp",
        "quarantine_to_old_when_replacement_exists"),
    ("ch82_cyclic_validation.tex",
        "quarantine_to_old_when_replacement_exists"),
    ("ch82_cyclic_validation.tex.old",
        "quarantine_to_old_when_replacement_exists"),
    ("ch85_comprehensive_cyclic_multiscale_validation.tex",
        "quarantine_to_old_when_replacement_exists"),
    # Kept as audited inputs (allowed to remain as ctest targets):
    ("test_uniaxial_fiber.cpp",                    "keep_only_as_audited_input"),
    ("test_rc_fiber_frame_nonlinear.cpp",          "keep_only_as_audited_input"),
    ("test_timoshenko_cantilever_benchmark.cpp",   "keep_only_as_audited_input"),
    ("KoBatheConcrete3D",                          "keep_only_as_audited_input"),
    ("KentParkConcrete.hh",
        "exploratory_only_do_not_anchor_claims"),
    ("KoBatheConcrete.hh",                         "keep_only_as_audited_input"),
]

_FORBIDDEN_DISPOSITIONS = {
    "quarantine_to_old_when_replacement_exists",
    "exploratory_only_do_not_anchor_claims",
    "retire_after_replacement_campaign",
}

_TEST_BLOCK_RE = re.compile(
    r"fall_n_add_test\s*\((?P<body>[^)]*)\)", re.DOTALL)
_NAME_RE   = re.compile(r"\bNAME\s+(\S+)")
_SOURCE_RE = re.compile(r"\bSOURCE\s+(\S+)")
_TARGET_RE = re.compile(r"\bTARGET\s+(\S+)")


def _scan_test_anchors(cmake_path: Path) -> list[dict]:
    text = cmake_path.read_text(encoding="utf-8")
    out = []
    for m in _TEST_BLOCK_RE.finditer(text):
        body = m.group("body")
        name = _NAME_RE.search(body)
        src  = _SOURCE_RE.search(body)
        tgt  = _TARGET_RE.search(body)
        if name and src:
            out.append({
                "test_name": name.group(1),
                "test_source": src.group(1),
                "test_target": tgt.group(1) if tgt else None,
            })
    return out


def _scan_target_sources_blocks(cmake_path: Path) -> dict[str, list[str]]:
    """Map test target → list of additional sources injected via
    `target_sources(<target> PRIVATE ...)`."""
    text = cmake_path.read_text(encoding="utf-8")
    rgx = re.compile(
        r"target_sources\s*\(\s*(?P<tgt>\S+)\s+PRIVATE\s+(?P<srcs>[^)]+)\)",
        re.DOTALL)
    result: dict[str, list[str]] = {}
    for m in rgx.finditer(text):
        tgt = m.group("tgt")
        raw = m.group("srcs")
        sources = [s.strip() for s in raw.split() if s.strip()]
        result.setdefault(tgt, []).extend(sources)
    return result


def main() -> int:
    cmake_path = _REPO_ROOT / "CMakeLists.txt"
    test_anchors = _scan_test_anchors(cmake_path)
    target_extra_sources = _scan_target_sources_blocks(cmake_path)

    # For each legacy surface, find which test (if any) anchors it.
    findings = []
    for surface, disposition in _LEGACY_SURFACES:
        anchored_by = []
        for t in test_anchors:
            tgt = t["test_target"]
            sources = [t["test_source"]] + target_extra_sources.get(tgt, [])
            if any(surface in s for s in sources):
                anchored_by.append(t["test_name"])
        # Also check whether the surface even exists in the repo (.cpp/.hh only)
        suffix = Path(surface).suffix
        if suffix in (".cpp", ".hh", ".h"):
            candidates = list(_REPO_ROOT.rglob(surface))
            present_paths = [str(p.relative_to(_REPO_ROOT)).replace("\\", "/")
                             for p in candidates if "build" not in p.parts]
        else:
            present_paths = []
        anchors_active_test = bool(anchored_by)
        violates_disposition = anchors_active_test and (
            disposition in _FORBIDDEN_DISPOSITIONS)
        findings.append({
            "legacy_surface": surface,
            "catalog_disposition": disposition,
            "present_in_repo_paths": present_paths,
            "anchors_test_targets": anchored_by,
            "anchors_active_test_claim": anchors_active_test,
            "violates_disposition": violates_disposition,
        })

    audit = {
        "schema_version": 1,
        "phase_label": "phase0_legacy_quarantine",
        "criterion":
            "A legacy surface is considered to anchor an active claim iff a "
            "ctest target (registered via fall_n_add_test) lists it in its "
            "SOURCE or in target_sources(<target> PRIVATE ...). Hand-rolled "
            "main_*.cpp drivers and library compilation are not gates and do "
            "not count as anchoring claims. The audit fails iff a surface "
            "with a forbidden catalog disposition (quarantine, retire, or "
            "exploratory) anchors a ctest target.",
        "forbidden_dispositions": sorted(_FORBIDDEN_DISPOSITIONS),
        "scanned_test_targets_count": len(test_anchors),
        "findings": findings,
        "any_legacy_violates_disposition":
            any(f["violates_disposition"] for f in findings),
    }

    out = _REPO_ROOT / "data" / "output" / "validation_reboot" \
        / "audit_phase0_legacy_quarantine.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[audit] wrote {out}")
    print(f"  scanned {audit['scanned_test_targets_count']} ctest targets")
    n_violate = sum(1 for f in findings if f["violates_disposition"])
    n_anchor = sum(1 for f in findings if f["anchors_active_test_claim"])
    print(f"  legacy surfaces anchoring active ctest claims: {n_anchor}/{len(findings)}")
    print(f"  disposition violations: {n_violate}")
    if n_violate:
        for f in findings:
            if f["violates_disposition"]:
                print(f"   VIOLATION {f['legacy_surface']} "
                      f"({f['catalog_disposition']}) -> "
                      f"{f['anchors_test_targets']}")
    return 0 if not audit["any_legacy_violates_disposition"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
