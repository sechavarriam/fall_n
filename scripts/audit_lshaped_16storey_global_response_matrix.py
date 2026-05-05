#!/usr/bin/env python3
"""Build a global roof-response audit matrix for the L-shaped benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_opensees(path: Path, label: str) -> dict | None:
    csv_path = path / "roof_displacement.csv"
    manifest_path = path / "opensees_lshaped_16storey_manifest.json"
    if not csv_path.exists():
        return None
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "label": label,
        "kind": "opensees",
        "path": str(path),
        "time": [r["time"] for r in rows],
        "ux": [r["ux"] for r in rows],
        "uy": [r["uy"] for r in rows],
        "uz": [r["uz"] for r in rows],
        "manifest": manifest,
    }


def read_falln(path: Path, label: str) -> dict | None:
    csv_path = path / "roof_displacement_global_reference.csv"
    manifest_path = path / "global_reference_summary.json"
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = [row for row in reader]
    triplets = []
    for i in range(1, len(columns), 3):
        if i + 2 < len(columns):
            triplets.append((columns[i], columns[i + 1], columns[i + 2]))
    if not triplets:
        return None
    ux_col, uy_col, uz_col = triplets[-1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "label": label,
        "kind": "fall_n",
        "path": str(path),
        "time": [float(r["time"]) for r in rows],
        "ux": [float(r[ux_col]) for r in rows],
        "uy": [float(r[uy_col]) for r in rows],
        "uz": [float(r[uz_col]) for r in rows],
        "manifest": manifest,
        "columns": [ux_col, uy_col, uz_col],
    }


def interp(time: list[float], values: list[float], target: list[float]) -> list[float]:
    if not time:
        return []
    out = []
    j = 0
    for t in target:
        while j + 1 < len(time) and time[j + 1] < t:
            j += 1
        if t < time[0] or j + 1 >= len(time) or t > time[-1]:
            out.append(math.nan)
            continue
        t0, t1 = time[j], time[j + 1]
        v0, v1 = values[j], values[j + 1]
        if t1 == t0:
            out.append(v0)
        else:
            a = (t - t0) / (t1 - t0)
            out.append((1.0 - a) * v0 + a * v1)
    return out


def rms_error(case: dict, ref: dict, comp: str) -> float:
    common = [t for t in ref["time"] if case["time"][0] <= t <= case["time"][-1]]
    if not common:
        return math.nan
    a = interp(case["time"], case[comp], common)
    b = interp(ref["time"], ref[comp], common)
    vals = [(x - y) ** 2 for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    return math.sqrt(sum(vals) / len(vals)) if vals else math.nan


def peak(values: list[float]) -> float:
    return max((abs(v) for v in values), default=0.0)


def summarize(case: dict, ref: dict | None) -> dict:
    manifest = case.get("manifest") or {}
    out = {
        "label": case["label"],
        "kind": case["kind"],
        "samples": len(case["time"]),
        "path": case["path"],
        "status": manifest.get("status", ""),
        "beam_element_family": manifest.get("beam_element_family", ""),
        "mass_model": manifest.get("mass_model", ""),
        "element_mass_form": manifest.get("element_mass_form", ""),
        "member_subdivisions": manifest.get("member_subdivisions", ""),
        "include_vertical": manifest.get("include_vertical", ""),
        "accepted_steps": manifest.get("accepted_steps", ""),
        "requested_steps": manifest.get("requested_steps", ""),
        "wall_seconds": manifest.get("total_wall_seconds", ""),
    }
    for comp in ("ux", "uy", "uz"):
        out[f"peak_abs_{comp}_m"] = peak(case[comp])
        out[f"final_{comp}_m"] = case[comp][-1] if case[comp] else 0.0
        out[f"rms_error_vs_falln_{comp}_m"] = rms_error(case, ref, comp) if ref and case is not ref else 0.0
    return out


def parse_case(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.name, path
    label, path = text.split("=", 1)
    return label, Path(path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(root / "doc/figures/validation_reboot"))
    parser.add_argument(
        "--falln",
        default=str(root / "data/output/lshaped_multiscale_16/recorders"),
    )
    parser.add_argument("--falln-label", default="fall_n elasticized TimoshenkoN4")
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args()

    falln = read_falln(Path(args.falln), args.falln_label)
    if falln is None:
        raise SystemExit("Could not read fall_n roof response.")

    defaults = [
        "OpenSees ElasticTimoshenko sub1=data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_elastic_timoshenko_transformed_clean",
        "OpenSees ElasticTimoshenko sub3=data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_elastic_timoshenko_sub3_elementmass_clean",
        "OpenSees forceBeamColumn+Vy/Vz=data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_force_shear_sub1_nodalmass",
        "OpenSees dispBeamColumn legacy=data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_disp_linear_topology",
    ]
    cases = [falln]
    for item in args.case or defaults:
        label, path = parse_case(item)
        case = read_opensees(Path(path), label)
        if case is not None:
            cases.append(case)

    rows = [summarize(case, falln) for case in cases]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "lshaped_16_global_response_audit_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "lshaped_16_global_response_audit_matrix_v1",
        "reference": falln["label"],
        "rows": rows,
        "csv": str(csv_path),
        "notes": [
            "RMS errors use fall_n roof response as the current internal reference, not as a final truth model.",
            "Rows with different include_vertical, mass_model or member_subdivisions are not strictly equivalent; they are shown to expose sensitivity.",
        ],
    }
    json_path = out_dir / "lshaped_16_global_response_audit_matrix_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
