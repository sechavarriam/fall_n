#!/usr/bin/env python3
"""Audit fall_n publication VTK outputs.

The script intentionally depends only on the Python standard library so it can
run on a headless node before opening the case in ParaView.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


CRACK_FIELDS = {
    "crack_opening",
    "crack_opening_max",
    "crack_visible",
    "crack_normal",
    "crack_opening_vector",
    "crack_state",
    "site_id",
    "parent_element_id",
}

GAUSS_FIELDS = {
    "displacement",
    "gauss_id",
    "element_id",
    "material_id",
    "site_id",
    "parent_element_id",
}
GAUSS_MINIMAL_DISALLOWED_SUFFIXES = {
    "_voigt",
}
REBAR_FIELDS = {
    "TubeRadius",
    "bar_id",
    "bar_area",
    "axial_strain",
    "axial_stress",
    "yield_ratio",
}


def issue(severity: str, path: Path | str, message: str) -> dict[str, str]:
    return {"severity": severity, "path": str(path), "message": message}


def category(path: Path) -> str:
    name = path.name
    if name.endswith("_cracks_visible.vtu"):
        return "cracks_visible"
    if name.endswith("_cracks.vtu"):
        return "cracks_raw"
    if name.endswith("_rebar_tubes.vtu"):
        return "rebar_tubes"
    if name.endswith("_rebar.vtu"):
        return "rebar"
    if name.endswith("_gauss.vtu"):
        return "gauss"
    if name.endswith("_mesh.vtu"):
        return "mesh"
    return "other"


def prefix_for(path: Path) -> Path:
    suffixes = [
        "_cracks_visible.vtu",
        "_cracks.vtu",
        "_rebar_tubes.vtu",
        "_rebar.vtu",
        "_gauss.vtu",
        "_mesh.vtu",
    ]
    text = str(path)
    for suffix in suffixes:
        if text.endswith(suffix):
            return Path(text[: -len(suffix)])
    return path.with_suffix("")


def parse_vtu(path: Path) -> dict[str, Any]:
    tree = ET.parse(path)
    root = tree.getroot()
    piece = root.find(".//Piece")
    point_data = root.find(".//PointData")
    cell_data = root.find(".//CellData")

    def arrays(parent: ET.Element | None) -> dict[str, ET.Element]:
        if parent is None:
            return {}
        found: dict[str, ET.Element] = {}
        for arr in parent.findall("DataArray"):
            name = arr.attrib.get("Name")
            if name:
                found[name] = arr
        return found

    return {
        "points": int(piece.attrib.get("NumberOfPoints", "0")) if piece is not None else 0,
        "cells": int(piece.attrib.get("NumberOfCells", "0")) if piece is not None else 0,
        "point_data": arrays(point_data),
        "cell_data": arrays(cell_data),
        "active_vectors": point_data.attrib.get("Vectors", "") if point_data is not None else "",
        "active_scalars": point_data.attrib.get("Scalars", "") if point_data is not None else "",
    }


def parse_ascii_numbers(array: ET.Element | None) -> list[float]:
    if array is None:
        return []
    fmt = array.attrib.get("format", "ascii").lower()
    if fmt not in {"", "ascii"}:
        return []
    text = array.text or ""
    if not text.strip():
        return []
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]


def audit_vtu(
    path: Path,
    info: dict[str, Any],
    gauss_fields_profile: str,
    crack_opening_threshold: float | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    issues: list[dict[str, str]] = []
    cat = category(path)
    point_names = set(info["point_data"])
    cell_names = set(info["cell_data"])
    metrics: dict[str, Any] = {}

    if "displacement" not in point_names:
        issues.append(issue("error", path, "missing PointData/displacement"))
    elif info.get("active_vectors") != "displacement":
        issues.append(issue("error", path, "PointData/displacement is not the active vector"))

    if cat in {"cracks_raw", "cracks_visible"}:
        missing = sorted(CRACK_FIELDS - cell_names)
        for field in missing:
            issues.append(issue("error", path, f"missing crack CellData/{field}"))
        if cat == "cracks_visible":
            metrics["visible_crack_cells"] = info["cells"]
            if crack_opening_threshold is not None:
                openings = parse_ascii_numbers(info["cell_data"].get("crack_opening"))
                opening_max = parse_ascii_numbers(info["cell_data"].get("crack_opening_max"))
                count = max(len(openings), len(opening_max))
                min_visible_opening = math.inf
                for i in range(count):
                    current = abs(openings[i]) if i < len(openings) else 0.0
                    historical = abs(opening_max[i]) if i < len(opening_max) else 0.0
                    visible_opening = max(current, historical)
                    min_visible_opening = min(min_visible_opening, visible_opening)
                    if visible_opening <= crack_opening_threshold:
                        issues.append(issue(
                            "error",
                            path,
                            "visible crack opening does not exceed "
                            f"threshold {crack_opening_threshold:.6e} m",
                        ))
                        break
                if math.isfinite(min_visible_opening):
                    metrics["min_visible_crack_opening_m"] = min_visible_opening
        else:
            visible = parse_ascii_numbers(info["cell_data"].get("crack_visible"))
            if visible:
                metrics["visible_crack_cells"] = sum(1 for value in visible if value >= 0.5)
        families = parse_ascii_numbers(info["cell_data"].get("crack_family_id"))
        if families:
            metrics["crack_families"] = sorted({int(round(value)) for value in families})

    if cat == "gauss":
        for field in sorted(GAUSS_FIELDS - point_names):
            issues.append(issue("error", path, f"missing Gauss PointData/{field}"))
        if not any(name.startswith("qp_stress") or name.startswith("stress") for name in point_names):
            issues.append(issue("warning", path, "no Gauss stress field found"))
        if not any(name.startswith("qp_strain") or name.startswith("strain") for name in point_names):
            issues.append(issue("warning", path, "no Gauss strain field found"))
        if not any(("damage" in name or "crack" in name) for name in point_names):
            issues.append(issue("warning", path, "no Gauss damage/crack field found"))
        if gauss_fields_profile in {"minimal", "visual"}:
            heavy = sorted(
                name for name in point_names
                if any(name.endswith(suffix) for suffix in GAUSS_MINIMAL_DISALLOWED_SUFFIXES)
            )
            for field in heavy:
                issues.append(issue("error", path, f"{gauss_fields_profile} Gauss profile contains heavy field {field}"))
        metrics["gauss_points"] = info["points"]

    if cat in {"rebar", "rebar_tubes"}:
        for field in sorted(REBAR_FIELDS - cell_names):
            issues.append(issue("error", path, f"missing rebar CellData/{field}"))

    return issues, metrics


def audit_transforms(root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    issues: list[dict[str, str]] = []
    metrics: dict[str, Any] = {"transform_records": 0, "transform_coherent": False}
    path = root / "recorders" / "local_site_transform.json"
    if not path.exists():
        issues.append(issue("warning", path, "missing local_site_transform.json"))
        return issues, metrics

    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records", [])
    metrics["transform_records"] = len(records)
    coherent = True
    for idx, record in enumerate(records):
        R = record.get("R")
        if not isinstance(R, list) or len(R) != 3:
            issues.append(issue("error", path, f"record {idx} has invalid R"))
            coherent = False
            continue
        try:
            cols = [[float(R[r][c]) for r in range(3)] for c in range(3)]
        except (TypeError, ValueError, IndexError):
            issues.append(issue("error", path, f"record {idx} has nonnumeric R"))
            coherent = False
            continue
        for c, col in enumerate(cols):
            norm = math.sqrt(sum(v * v for v in col))
            if abs(norm - 1.0) > 1.0e-6:
                issues.append(issue("error", path, f"record {idx} column {c} norm is {norm:g}"))
                coherent = False
        for a in range(3):
            for b in range(a + 1, 3):
                dot = sum(cols[a][r] * cols[b][r] for r in range(3))
                if abs(dot) > 1.0e-6:
                    issues.append(issue("error", path, f"record {idx} columns {a}/{b} dot is {dot:g}"))
                    coherent = False
        det = (
            cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1])
            - cols[1][0] * (cols[0][1] * cols[2][2] - cols[0][2] * cols[2][1])
            + cols[2][0] * (cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1])
        )
        if abs(det - 1.0) > 1.0e-6:
            issues.append(issue("error", path, f"record {idx} determinant is {det:g}"))
            coherent = False
    metrics["transform_coherent"] = coherent
    return issues, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="fall_n output root")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument("--warn-only", action="store_true", help="always exit 0")
    parser.add_argument(
        "--gauss-fields-profile",
        choices=("visual", "minimal", "full", "debug"),
        default="minimal",
        help="expected Gauss field profile for publication checks",
    )
    parser.add_argument(
        "--max-files-per-category",
        type=int,
        default=0,
        help="sample at most this many VTUs per category; 0 audits all",
    )
    parser.add_argument(
        "--crack-opening-threshold",
        type=float,
        default=None,
        help="if set, require every *_cracks_visible.vtu cell to exceed this opening threshold in metres",
    )
    args = parser.parse_args()

    root = args.root
    report: dict[str, Any] = {
        "schema": "fall_n_publication_vtk_audit_v1",
        "root": str(root),
        "files": {},
        "missing_files": [],
        "issues": [],
        "visible_crack_cells": 0,
        "gauss_points": 0,
        "kobathe_crack_families_active": [],
        "transform_records": 0,
        "transform_coherent": False,
        "gauss_fields_profile": args.gauss_fields_profile,
        "crack_opening_threshold_m": args.crack_opening_threshold,
        "min_visible_crack_opening_m": None,
        "sampled_files": 0,
        "total_vtu_files": 0,
    }

    if not root.exists():
        report["issues"].append(issue("error", root, "output root does not exist"))
    else:
        all_vtus = sorted(root.rglob("*.vtu"))
        report["total_vtu_files"] = len(all_vtus)
        if args.max_files_per_category > 0:
            selected: list[Path] = []
            grouped: dict[str, list[Path]] = {}
            for path in all_vtus:
                grouped.setdefault(category(path), []).append(path)
            for paths in grouped.values():
                if len(paths) <= args.max_files_per_category:
                    selected.extend(paths)
                    continue
                if args.max_files_per_category == 1:
                    selected.append(paths[0])
                    continue
                last = len(paths) - 1
                indices = {
                    round(i * last / (args.max_files_per_category - 1))
                    for i in range(args.max_files_per_category)
                }
                selected.extend(paths[i] for i in sorted(indices))
            vtus = sorted(set(selected))
        else:
            vtus = all_vtus
        report["sampled_files"] = len(vtus)
        by_category: dict[str, list[str]] = {}
        infos: dict[Path, dict[str, Any]] = {}
        for path in vtus:
            cat = category(path)
            by_category.setdefault(cat, []).append(str(path.relative_to(root)))
            try:
                info = parse_vtu(path)
            except Exception as exc:  # noqa: BLE001 - report bad XML, keep auditing.
                report["issues"].append(issue("error", path, f"cannot parse VTU: {exc}"))
                continue
            infos[path] = info
            issues, metrics = audit_vtu(
                path,
                info,
                args.gauss_fields_profile,
                args.crack_opening_threshold,
            )
            report["issues"].extend(issues)
            report["visible_crack_cells"] += int(metrics.get("visible_crack_cells", 0))
            report["gauss_points"] += int(metrics.get("gauss_points", 0))
            min_visible = metrics.get("min_visible_crack_opening_m")
            if min_visible is not None:
                report["min_visible_crack_opening_m"] = min(
                    min_visible,
                    report["min_visible_crack_opening_m"]
                    if report["min_visible_crack_opening_m"] is not None
                    else min_visible,
                )
            for family in metrics.get("crack_families", []):
                if family not in report["kobathe_crack_families_active"]:
                    report["kobathe_crack_families_active"].append(family)

        report["files"] = {key: len(value) for key, value in sorted(by_category.items())}
        report["kobathe_crack_families_active"].sort()

        prefixes = {
            prefix_for(path)
            for path in vtus
            if "_current_" not in path.name
            and category(path) in {"mesh", "gauss", "cracks_raw", "cracks_visible"}
        }
        required_suffixes = ["_mesh.vtu", "_gauss.vtu", "_cracks.vtu", "_cracks_visible.vtu"]
        for prefix in sorted(prefixes, key=str):
            for suffix in required_suffixes:
                candidate = Path(str(prefix) + suffix)
                if not candidate.exists():
                    report["missing_files"].append(str(candidate.relative_to(root)))
                    report["issues"].append(issue("error", candidate, "missing publication VTU sibling"))
            rebar = Path(str(prefix) + "_rebar.vtu")
            tubes = Path(str(prefix) + "_rebar_tubes.vtu")
            if rebar.exists() != tubes.exists():
                missing = tubes if rebar.exists() else rebar
                report["missing_files"].append(str(missing.relative_to(root)))
                report["issues"].append(issue("error", missing, "missing paired rebar/tube VTU"))

        transform_issues, transform_metrics = audit_transforms(root)
        report["issues"].extend(transform_issues)
        report.update(transform_metrics)

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    has_error = any(item["severity"] == "error" for item in report["issues"])
    return 0 if args.warn_only or not has_error else 1


if __name__ == "__main__":
    sys.exit(main())
