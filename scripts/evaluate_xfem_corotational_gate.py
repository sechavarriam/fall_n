#!/usr/bin/env python
"""Plan v2 Fase 3.1 -- XFEM 200mm corotational gate evaluation.

Compares the corotational XFEM run against the small-strain baseline at
identical mesh (NZ=4) and protocol (50,100,150,200 mm cyclic). Computes:

  - peak_normalized_rms_base_shear_error
  - peak_normalized_max_base_shear_error
  - peak_base_shear_ratio (corot peak / baseline peak)

against the canonical gate from
ReducedRCLocalModelPromotionCatalog::xfem_global_secant_200mm_primary_candidate:
  - max_peak_normalized_rms_base_shear_error = 0.10
  - max_peak_normalized_max_base_shear_error = 0.30
  - min_peak_base_shear_ratio = 0.90
  - max_peak_base_shear_ratio = 1.15

Emits data/output/validation_reboot/audit_phase3_xfem_corotational_gate.json.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


def read_progress(p: Path) -> tuple[list[float], list[float], list[float]]:
    drift, shear, ptime = [], [], []
    with p.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            drift.append(float(row["drift_mm"]))
            shear.append(float(row["base_shear_MN"]))
            ptime.append(float(row["p"]))
    return ptime, drift, shear


def main() -> int:
    base_dir = Path("data/output/cyclic_validation")
    corot = base_dir / "xfem_corotational_200mm_v2" / "global_xfem_newton_progress.csv"
    base = base_dir / "xfem_small_strain_200mm_v2" / "global_xfem_newton_progress.csv"

    out_dir = Path("data/output/validation_reboot")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_phase3_xfem_corotational_gate.json"

    if not corot.exists() or not base.exists():
        print(f"[gate] inputs missing: corot={corot.exists()} base={base.exists()}",
              file=sys.stderr)
        out_path.write_text(json.dumps({
            "schema_version": 1,
            "phase_label": "phase3_xfem_corotational_gate",
            "status": "input_missing",
            "corotational_run_present": corot.exists(),
            "baseline_run_present": base.exists(),
        }, indent=2))
        return 1

    pc, dc, sc = read_progress(corot)
    pb, db, sb = read_progress(base)

    # Reduce to common pseudo-time grid by index intersection (both runs
    # use the same protocol so step counts match if both reached p=1).
    n = min(len(sc), len(sb))
    sc = sc[:n]
    sb = sb[:n]
    pc_eff = pc[:n]

    peak_base = max(abs(x) for x in sb) if sb else 0.0
    peak_corot = max(abs(x) for x in sc) if sc else 0.0
    if peak_base == 0.0:
        print("[gate] baseline peak is zero", file=sys.stderr)
        return 2

    diff = [c - b for c, b in zip(sc, sb)]
    rms = math.sqrt(sum(d * d for d in diff) / len(diff))
    max_abs = max(abs(d) for d in diff)

    rms_norm = rms / peak_base
    max_norm = max_abs / peak_base
    peak_ratio = peak_corot / peak_base

    # Catalog gate values.
    max_rms = 0.10
    max_max = 0.30
    min_ratio = 0.90
    max_ratio = 1.15

    pass_rms = rms_norm <= max_rms
    pass_max = max_norm <= max_max
    pass_ratio = (min_ratio <= peak_ratio <= max_ratio)
    overall = pass_rms and pass_max and pass_ratio

    # Both runs must have completed (reached p ~ 1).
    completed_corot = pc_eff and pc_eff[-1] >= 0.999
    completed_base = pb and pb[-1] >= 0.999

    payload = {
        "schema_version": 1,
        "phase_label": "phase3_xfem_corotational_gate",
        "comparison": {
            "corotational_run": str(corot),
            "baseline_run": str(base),
            "n_compared_steps": n,
            "corotational_completed": completed_corot,
            "baseline_completed": completed_base,
            "corotational_final_p": pc_eff[-1] if pc_eff else None,
            "baseline_final_p": pb[-1] if pb else None,
        },
        "metrics": {
            "peak_baseline_base_shear_mn": peak_base,
            "peak_corotational_base_shear_mn": peak_corot,
            "peak_base_shear_ratio": peak_ratio,
            "peak_normalized_rms_base_shear_error": rms_norm,
            "peak_normalized_max_base_shear_error": max_norm,
        },
        "catalog_gate": {
            "max_peak_normalized_rms_base_shear_error": max_rms,
            "max_peak_normalized_max_base_shear_error": max_max,
            "min_peak_base_shear_ratio": min_ratio,
            "max_peak_base_shear_ratio": max_ratio,
        },
        "verdict": {
            "rms_pass": pass_rms,
            "max_pass": pass_max,
            "ratio_pass": pass_ratio,
            "overall_pass": overall,
            "completed": completed_corot and completed_base,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[gate] rms_norm={rms_norm:.4f} (gate<={max_rms})")
    print(f"[gate] max_norm={max_norm:.4f} (gate<={max_max})")
    print(f"[gate] peak_ratio={peak_ratio:.4f} (gate in [{min_ratio},{max_ratio}])")
    print(f"[gate] overall_pass={overall} completed={completed_corot and completed_base}")
    print(f"[gate] wrote {out_path}")
    return 0 if overall else 3


if __name__ == "__main__":
    sys.exit(main())
