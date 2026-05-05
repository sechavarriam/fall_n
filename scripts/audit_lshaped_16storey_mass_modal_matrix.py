#!/usr/bin/env python3
"""Collect mass and modal diagnostics for the L-shaped benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_case(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.name, path
    label, path = text.split("=", 1)
    return label, Path(path)


def opensees_row(label: str, path: Path) -> dict:
    manifest = read_json(path / "opensees_lshaped_16storey_manifest.json")
    periods = manifest.get("eigen_periods_s", [])
    row = {
        "label": label,
        "kind": "opensees",
        "beam_element_family": manifest.get("beam_element_family", ""),
        "mass_model": manifest.get("mass_model", ""),
        "element_mass_form": manifest.get("element_mass_form", ""),
        "member_subdivisions": manifest.get("member_subdivisions", ""),
        "include_vertical": manifest.get("include_vertical", ""),
        "global_rows": "",
        "sum_M_ones_all_dofs": "",
        "translational_mass_x": (
            manifest.get("nodal_mass_total_per_direction")
            if manifest.get("mass_model") == "nodal"
            else manifest.get("element_mass_total_per_direction", "")
        ),
        "translational_mass_y": (
            manifest.get("nodal_mass_total_per_direction")
            if manifest.get("mass_model") == "nodal"
            else manifest.get("element_mass_total_per_direction", "")
        ),
        "translational_mass_z": (
            manifest.get("nodal_mass_total_per_direction")
            if manifest.get("mass_model") == "nodal"
            else manifest.get("element_mass_total_per_direction", "")
        ),
        "rotational_mass_sum": "",
        "wall_seconds": manifest.get("total_wall_seconds", ""),
    }
    for i in range(6):
        row[f"T{i + 1}_s"] = periods[i] if i < len(periods) else ""
    return row


def falln_row(label: str, path: Path) -> dict:
    summary = read_json(path)
    row_sums = summary.get("row_sum_by_local_dof", [0.0] * 6)
    return {
        "label": label,
        "kind": "fall_n",
        "beam_element_family": "TimoshenkoBeamN<4>",
        "mass_model": "consistent element assembly",
        "element_mass_form": "translational-only",
        "member_subdivisions": "N=4 geometry nodes",
        "include_vertical": "True",
        "global_rows": summary.get("global_rows", ""),
        "sum_M_ones_all_dofs": summary.get("sum_M_ones_all_dofs", ""),
        "translational_mass_x": row_sums[0] if len(row_sums) > 0 else "",
        "translational_mass_y": row_sums[1] if len(row_sums) > 1 else "",
        "translational_mass_z": row_sums[2] if len(row_sums) > 2 else "",
        "rotational_mass_sum": sum(row_sums[3:]) if len(row_sums) >= 6 else "",
        "wall_seconds": "",
        **{f"T{i + 1}_s": "" for i in range(6)},
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(root / "doc/figures/validation_reboot"))
    parser.add_argument(
        "--falln-mass",
        default=str(root / "data/output/lshaped_multiscale_16/recorders/falln_mass_matrix_audit_summary.json"),
    )
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()

    defaults = [
        "OpenSees ElasticTimoshenko sub1 nodal=data/output/opensees_lshaped_16storey/eigen_elastic_timoshenko_sub1_nodal",
        "OpenSees ElasticTimoshenko sub1 element=data/output/opensees_lshaped_16storey/eigen_elastic_timoshenko_sub1_element",
        "OpenSees ElasticTimoshenko sub3 element=data/output/opensees_lshaped_16storey/eigen_elastic_timoshenko_sub3_element",
    ]
    rows = [falln_row("fall_n mass audit", Path(args.falln_mass))]
    for item in args.case or defaults:
        label, path = parse_case(item)
        rows.append(opensees_row(label, path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lshaped_16_mass_modal_audit_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "lshaped_16_mass_modal_audit_matrix_v1",
        "rows": rows,
        "csv": str(csv_path),
        "notes": [
            "fall_n modal periods are intentionally blank until a PETSc/SLEPc or reduced modal extraction path is added.",
            "fall_n row-sum mass confirms translational mass per direction from the assembled matrix.",
            "OpenSees periods are computed after gravity/loadConst for the stated mass policy.",
        ],
    }
    json_path = out_dir / "lshaped_16_mass_modal_audit_matrix_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
