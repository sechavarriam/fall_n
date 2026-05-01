#!/usr/bin/env python
"""Plan v2 Fase 3.1/3.2/3.3 -- XFEM finite-kinematics 200mm gate evaluation.

Generalises evaluate_xfem_corotational_gate.py to all three finite-kinematics
formulations (corotational, total-lagrangian, updated-lagrangian) against the
small-strain baseline at NZ=4, cyclic 50/100/150/200 mm.

Emits data/output/validation_reboot/audit_phase3_xfem_finite_kinematics_gate.json.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path


def read_progress(p: Path) -> tuple[list[float], list[float]]:
    shear, ptime = [], []
    with p.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            shear.append(float(row["base_shear_MN"]))
            ptime.append(float(row["p"]))
    return ptime, shear


def evaluate(label: str, run_csv: Path, base_csv: Path) -> dict:
    if not run_csv.exists():
        return {"label": label, "status": "input_missing", "run_csv": str(run_csv)}
    pr, sr = read_progress(run_csv)
    pb, sb = read_progress(base_csv)
    n = min(len(sr), len(sb))
    sr = sr[:n]; sb = sb[:n]
    peak_base = max(abs(x) for x in sb)
    peak_run = max(abs(x) for x in sr)
    if peak_base == 0.0:
        return {"label": label, "status": "baseline_zero"}
    diff = [c - b for c, b in zip(sr, sb)]
    rms = math.sqrt(sum(d * d for d in diff) / len(diff))
    max_abs = max(abs(d) for d in diff)
    rms_norm = rms / peak_base
    max_norm = max_abs / peak_base
    peak_ratio = peak_run / peak_base

    pass_rms = rms_norm <= 0.10
    pass_max = max_norm <= 0.30
    pass_ratio = 0.90 <= peak_ratio <= 1.15
    overall = pass_rms and pass_max and pass_ratio
    completed = pr and pr[-1] >= 0.999

    return {
        "label": label,
        "status": "evaluated",
        "n_compared_steps": n,
        "completed": completed,
        "final_p": pr[-1] if pr else None,
        "peak_base_shear_mn": peak_run,
        "peak_baseline_base_shear_mn": peak_base,
        "peak_normalized_rms_base_shear_error": rms_norm,
        "peak_normalized_max_base_shear_error": max_norm,
        "peak_base_shear_ratio": peak_ratio,
        "rms_pass": pass_rms,
        "max_pass": pass_max,
        "ratio_pass": pass_ratio,
        "overall_pass": overall,
    }


def main() -> int:
    base_dir = Path("data/output/cyclic_validation")
    base_csv = base_dir / "xfem_small_strain_200mm_v2" / "global_xfem_newton_progress.csv"
    if not base_csv.exists():
        print(f"[gate] baseline missing: {base_csv}", file=sys.stderr)
        return 1

    runs = [
        ("corotational",      base_dir / "xfem_corotational_200mm_v2"      / "global_xfem_newton_progress.csv"),
        ("total-lagrangian",  base_dir / "xfem_total_lagrangian_200mm_v2"  / "global_xfem_newton_progress.csv"),
        ("updated-lagrangian", base_dir / "xfem_updated_lagrangian_200mm_v2" / "global_xfem_newton_progress.csv"),
    ]

    results = [evaluate(lbl, p, base_csv) for lbl, p in runs]

    out_dir = Path("data/output/validation_reboot")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "audit_phase3_xfem_finite_kinematics_gate.json"

    payload = {
        "schema_version": 1,
        "phase_label": "phase3_xfem_finite_kinematics_gate",
        "baseline_run": str(base_csv),
        "catalog_gate": {
            "max_peak_normalized_rms_base_shear_error": 0.10,
            "max_peak_normalized_max_base_shear_error": 0.30,
            "min_peak_base_shear_ratio": 0.90,
            "max_peak_base_shear_ratio": 1.15,
        },
        "formulations": results,
        "summary": {
            "evaluated": [r["label"] for r in results if r.get("status") == "evaluated"],
            "passing": [r["label"] for r in results
                        if r.get("status") == "evaluated" and r.get("overall_pass")],
            "missing": [r["label"] for r in results if r.get("status") == "input_missing"],
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    for r in results:
        if r.get("status") != "evaluated":
            print(f"[gate] {r['label']}: {r.get('status')}")
            continue
        print(f"[gate] {r['label']:>20s}: rms={r['peak_normalized_rms_base_shear_error']:.4f}"
              f"  max={r['peak_normalized_max_base_shear_error']:.4f}"
              f"  ratio={r['peak_base_shear_ratio']:.4f}"
              f"  pass={r['overall_pass']}  completed={r['completed']}")
    print(f"[gate] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
