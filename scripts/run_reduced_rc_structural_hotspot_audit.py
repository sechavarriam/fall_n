#!/usr/bin/env python3
"""
Rank structural fiber-level hotspots inside a reduced RC-column benchmark bundle.

The audit is intentionally lightweight:
  - it reads an existing benchmark bundle;
  - compares fall_n and OpenSees fiber histories when both exist;
  - ranks the worst step x station x fiber mismatches;
  - and, independently, reports the most strained/stressed/tangent-critical
    fibers on the fall_n side near the structural frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit structural fiber hotspots from an existing reduced RC-column benchmark bundle."
    )
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_error(lhs: float, rhs: float) -> float:
    scale = max(abs(rhs), 1.0e-12)
    return abs(lhs - rhs) / scale


def fiber_key(row: dict[str, str]) -> tuple[object, ...]:
    return (
        int(row["step"]),
        int(row["section_gp"]),
        row["material_role"],
        row["zone"],
        round(float(row["y"]), 10),
        round(float(row["z"]), 10),
        round(float(row["area"]), 12),
    )


def hotspot_rows(
    lhs_rows: list[dict[str, str]],
    rhs_rows: list[dict[str, str]],
    lhs_field: str,
    rhs_field: str,
    label: str,
    top_k: int,
) -> list[dict[str, object]]:
    lhs_by_key = {fiber_key(row): row for row in lhs_rows}
    rhs_by_key = {fiber_key(row): row for row in rhs_rows}
    common_keys = sorted(set(lhs_by_key) & set(rhs_by_key))
    ranked: list[dict[str, object]] = []
    for key in common_keys:
        lhs = lhs_by_key[key]
        rhs = rhs_by_key[key]
        lhs_value = float(lhs[lhs_field])
        rhs_value = float(rhs[rhs_field])
        ranked.append(
            {
                "step": key[0],
                "section_gp": key[1],
                "material_role": key[2],
                "zone": key[3],
                "y": key[4],
                "z": key[5],
                "area": key[6],
                f"fall_n_{label}": lhs_value,
                f"opensees_{label}": rhs_value,
                f"abs_{label}_error": abs(lhs_value - rhs_value),
                f"rel_{label}_error": relative_error(lhs_value, rhs_value),
                "curvature_y": float(lhs["curvature_y"]),
                "drift_m": float(lhs["drift_m"]),
                "zero_curvature_anchor": int(lhs["zero_curvature_anchor"]),
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row[f"rel_{label}_error"]),
            -float(row[f"abs_{label}_error"]),
        )
    )
    return ranked[:top_k]


def falln_frontier_rows(rows: list[dict[str, str]], top_k: int) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for row in rows:
        ranked.append(
            {
                "step": int(row["step"]),
                "section_gp": int(row["section_gp"]),
                "material_role": row["material_role"],
                "zone": row["zone"],
                "y": float(row["y"]),
                "z": float(row["z"]),
                "area": float(row["area"]),
                "curvature_y": float(row["curvature_y"]),
                "drift_m": float(row["drift_m"]),
                "stress_xx_MPa": float(row["stress_xx_MPa"]),
                "tangent_xx_MPa": float(row["tangent_xx_MPa"]),
                "strain_xx": float(row["strain_xx"]),
                "abs_stress_xx_MPa": abs(float(row["stress_xx_MPa"])),
                "abs_tangent_xx_MPa": abs(float(row["tangent_xx_MPa"])),
                "abs_strain_xx": abs(float(row["strain_xx"])),
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["step"],
            -row["abs_strain_xx"],
            -row["abs_stress_xx_MPa"],
            -row["abs_tangent_xx_MPa"],
        )
    )
    return ranked[:top_k]


def main() -> int:
    args = parse_args()
    bundle_dir = args.bundle_dir.resolve()
    out_dir = bundle_dir / "hotspot_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    falln_root = bundle_dir / "fall_n" if (bundle_dir / "fall_n").exists() else bundle_dir
    opensees_root = bundle_dir / "opensees" if (bundle_dir / "opensees").exists() else bundle_dir

    falln_rows = read_rows(falln_root / "comparison_section_fiber_state_history.csv")
    if not falln_rows:
        falln_rows = read_rows(falln_root / "section_fiber_state_history.csv")
    opensees_rows = (
        read_rows(opensees_root / "section_fiber_state_history.csv")
        if (bundle_dir / "opensees").exists()
        else []
    )

    stress_hotspots = (
        hotspot_rows(
            falln_rows,
            opensees_rows,
            "stress_xx_MPa",
            "stress_xx_MPa",
            "stress_xx_MPa",
            args.top_k,
        )
        if falln_rows and opensees_rows
        else []
    )
    tangent_hotspots = (
        hotspot_rows(
            falln_rows,
            opensees_rows,
            "tangent_xx_MPa",
            "tangent_xx_MPa",
            "tangent_xx_MPa",
            args.top_k,
        )
        if falln_rows and opensees_rows
        else []
    )
    frontier_hotspots = falln_frontier_rows(falln_rows, args.top_k) if falln_rows else []

    write_csv(
        out_dir / "structural_fiber_stress_hotspots.csv",
        (
            "step",
            "section_gp",
            "material_role",
            "zone",
            "y",
            "z",
            "area",
            "fall_n_stress_xx_MPa",
            "opensees_stress_xx_MPa",
            "abs_stress_xx_MPa_error",
            "rel_stress_xx_MPa_error",
            "curvature_y",
            "drift_m",
            "zero_curvature_anchor",
        ),
        stress_hotspots,
    )
    write_csv(
        out_dir / "structural_fiber_tangent_hotspots.csv",
        (
            "step",
            "section_gp",
            "material_role",
            "zone",
            "y",
            "z",
            "area",
            "fall_n_tangent_xx_MPa",
            "opensees_tangent_xx_MPa",
            "abs_tangent_xx_MPa_error",
            "rel_tangent_xx_MPa_error",
            "curvature_y",
            "drift_m",
            "zero_curvature_anchor",
        ),
        tangent_hotspots,
    )
    write_csv(
        out_dir / "falln_frontier_fiber_hotspots.csv",
        (
            "step",
            "section_gp",
            "material_role",
            "zone",
            "y",
            "z",
            "area",
            "curvature_y",
            "drift_m",
            "strain_xx",
            "stress_xx_MPa",
            "tangent_xx_MPa",
            "abs_strain_xx",
            "abs_stress_xx_MPa",
            "abs_tangent_xx_MPa",
        ),
        frontier_hotspots,
    )

    summary = {
        "bundle_dir": str(bundle_dir),
        "stress_hotspot_count": len(stress_hotspots),
        "tangent_hotspot_count": len(tangent_hotspots),
        "frontier_hotspot_count": len(frontier_hotspots),
        "top_stress_hotspot": stress_hotspots[0] if stress_hotspots else None,
        "top_tangent_hotspot": tangent_hotspots[0] if tangent_hotspots else None,
        "top_frontier_hotspot": frontier_hotspots[0] if frontier_hotspots else None,
        "artifacts": {
            "stress_hotspots_csv": str(out_dir / "structural_fiber_stress_hotspots.csv"),
            "tangent_hotspots_csv": str(out_dir / "structural_fiber_tangent_hotspots.csv"),
            "falln_frontier_hotspots_csv": str(out_dir / "falln_frontier_fiber_hotspots.csv"),
        },
    }
    (out_dir / "hotspot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
