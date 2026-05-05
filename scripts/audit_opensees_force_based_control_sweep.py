#!/usr/bin/env python3
"""Summarize OpenSees forceBeamColumn local-iteration control attempts.

This audit records the last "force-based" promotion attempt for the L-shaped
16-storey benchmark.  It intentionally separates two questions:

1. Can forceBeamColumn run with the legacy no-tension Concrete01 model?
2. Can forceBeamColumn remain robust when the concrete law is moved closer to
   fall_n's Kent-Park tension/confined-core behaviour?

The runs are expensive, so this script summarizes existing output folders
instead of relaunching OpenSees.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/output/opensees_lshaped_16storey"
OUT = ROOT / "doc/figures/validation_reboot"


CASES = [
    {
        "label": "Concrete01 axis-fixed reference",
        "folder": "scale1p0_window87p65_1s_force_shear_concrete01_axisfixed",
        "concrete": "Concrete01",
        "element_iterations": 20,
        "element_tolerance": 1.0e-12,
        "wall_limit_s": None,
        "interpretation": (
            "Completes, but remains physically less comparable because "
            "Concrete01 has no tensile branch."
        ),
    },
    {
        "label": "Concrete02 proxy default local iteration",
        "folder": "scale1p0_window87p65_1s_force_shear_falln_proxy_axisfixed",
        "concrete": "Concrete02 fall_n Kent-Park proxy",
        "element_iterations": 20,
        "element_tolerance": 1.0e-12,
        "wall_limit_s": None,
        "interpretation": (
            "Assembles and starts, but element compatibility fails around "
            "the first crack/tension transition."
        ),
    },
    {
        "label": "Concrete02 proxy without local -iter",
        "folder": "scale1p0_window87p65_0p5s_force_shear_proxy_noiter_axisfixed",
        "concrete": "Concrete02 fall_n Kent-Park proxy",
        "element_iterations": 0,
        "element_tolerance": None,
        "wall_limit_s": None,
        "interpretation": (
            "Removing element-level iteration worsens robustness; the run "
            "accepts fewer global steps."
        ),
    },
    {
        "label": "Concrete02 proxy iter100 tol1e-6",
        "folder": "scale1p0_window87p65_0p5s_force_shear_proxy_iter100_tol1e6_axisfixed",
        "concrete": "Concrete02 fall_n Kent-Park proxy",
        "element_iterations": 100,
        "element_tolerance": 1.0e-6,
        "wall_limit_s": 900,
        "interpretation": (
            "Did not produce a manifest within the wall limit; the element "
            "state determination is not viable for campaign use."
        ),
    },
    {
        "label": "Concrete02 proxy iter40 tol1e-4",
        "folder": "scale1p0_window87p65_0p5s_force_shear_proxy_iter40_tol1e4_axisfixed",
        "concrete": "Concrete02 fall_n Kent-Park proxy",
        "element_iterations": 40,
        "element_tolerance": 1.0e-4,
        "wall_limit_s": 300,
        "interpretation": (
            "Looser element compatibility still stalls before a useful "
            "window; the issue is not resolved by tolerance tuning."
        ),
    },
    {
        "label": "Concrete02 proxy Ets/Ec=0.05, 1s",
        "folder": "scale1p0_window87p65_1s_force_shear_proxy_ets005_axisfixed",
        "concrete": "Concrete02 fall_n Kent-Park proxy, regularized tension",
        "element_iterations": 40,
        "element_tolerance": 1.0e-8,
        "wall_limit_s": None,
        "interpretation": (
            "Completes the first second quickly; the earlier force-based "
            "failure was strongly coupled to an overly brittle tensile proxy."
        ),
    },
    {
        "label": "Concrete02 proxy Ets/Ec=0.05, 10s partial",
        "folder": "scale1p0_window87p65_10s_force_shear_proxy_ets005_axisfixed",
        "concrete": "Concrete02 fall_n Kent-Park proxy, regularized tension",
        "element_iterations": 40,
        "element_tolerance": 1.0e-8,
        "wall_limit_s": 2400,
        "interpretation": (
            "Advances much farther than the displacement-based proxy run, "
            "but remains cost-limited before the full 10 s window."
        ),
    },
]


def read_manifest(folder: Path) -> dict:
    path = folder / "opensees_lshaped_16storey_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def recorder_status(folder: Path) -> dict:
    csv_path = folder / "roof_displacement.csv"
    out_path = folder / "roof_displacement.out"
    rows: list = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        t_last = float(rows[-1]["time"]) if rows else None
    elif out_path.exists():
        rows = [
            line.split()
            for line in out_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.split()
        ]
        t_last = float(rows[-1][0]) if rows else None
    else:
        t_last = None
    return {
        "roof_samples": len(rows),
        "last_recorded_time_s": t_last,
        "roof_csv": str(csv_path) if csv_path.exists() else "",
        "roof_out": str(out_path) if out_path.exists() else "",
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        folder = BASE / case["folder"]
        manifest = read_manifest(folder)
        row = {
            "label": case["label"],
            "folder": str(folder),
            "concrete": case["concrete"],
            "element_iterations": case["element_iterations"],
            "element_tolerance": case["element_tolerance"],
            "wall_limit_s": case["wall_limit_s"],
            "status": manifest.get("status", "no_manifest"),
            "accepted_steps": manifest.get("accepted_steps"),
            "requested_steps": manifest.get("requested_steps"),
            "total_wall_seconds": manifest.get("total_wall_seconds"),
            "interpretation": case["interpretation"],
        }
        row.update(recorder_status(folder))
        rows.append(row)

    summary = {
        "schema": "opensees_force_based_control_sweep_v1",
        "objective": (
            "Last promotion attempt for forceBeamColumn as the nonlinear "
            "external structural comparator after material/section parity fixes."
        ),
        "rows": rows,
        "decision": (
            "Do not yet promote forceBeamColumn as the primary long-window "
            "reference, but keep it as an active candidate.  Regularizing the "
            "Concrete02 tensile softening with Ets/Ec=0.05 rescues robustness "
            "relative to the original brittle proxy; remaining blocker is "
            "cost, not immediate element incompatibility."
        ),
        "primary_source": (
            "OpenSeesPy forceBeamColumn documentation: optional '-iter "
            "maxIter tol' controls element-level compatibility iterations."
        ),
    }
    path = OUT / "opensees_force_based_control_sweep_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
