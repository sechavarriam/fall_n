#!/usr/bin/env python3
"""Audit fall_n publication VTK outputs.

The script intentionally depends only on the Python standard library so it can
run on a headless node before opening the case in ParaView.
"""

from __future__ import annotations

import argparse
import csv
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


def triples(values: list[float]) -> list[list[float]]:
    return [values[i : i + 3] for i in range(0, len(values), 3)]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def vec_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def vec_dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def vec_norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def vec_mean(points: list[list[float]]) -> list[float]:
    if not points:
        return [math.nan, math.nan, math.nan]
    return [sum(p[i] for p in points) / len(points) for i in range(3)]


def parse_points_from_piece(piece: ET.Element | None) -> list[list[float]]:
    if piece is None:
        return []
    points = piece.find("./Points/DataArray")
    return triples(parse_ascii_numbers(points))


def parse_point_vector(piece: ET.Element | None, name: str) -> list[list[float]]:
    if piece is None:
        return []
    for array in piece.findall("./PointData/DataArray"):
        if array.attrib.get("Name") == name:
            return triples(parse_ascii_numbers(array))
    return []


def parse_lines(piece: ET.Element | None) -> list[list[int]]:
    if piece is None:
        return []
    conn_el = piece.find("./Lines/DataArray[@Name='connectivity']")
    off_el = piece.find("./Lines/DataArray[@Name='offsets']")
    conn = [int(round(x)) for x in parse_ascii_numbers(conn_el)]
    offsets = [int(round(x)) for x in parse_ascii_numbers(off_el)]
    lines: list[list[int]] = []
    start = 0
    for offset in offsets:
        lines.append(conn[start:offset])
        start = offset
    return lines


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


def parse_vtm_axis_path(vtm_path: Path) -> Path | None:
    try:
        root = ET.parse(vtm_path).getroot()
    except Exception:
        return None
    for data_set in root.findall(".//DataSet"):
        if data_set.attrib.get("name") == "axis":
            rel = data_set.attrib.get("file")
            return vtm_path.parent / rel if rel else None
    return None


def current_frame_path_for(local_path: str) -> str:
    if local_path.endswith("_mesh.vtu"):
        return local_path[:-len("_mesh.vtu")] + "_current_mesh.vtu"
    return ""


def current_frame_path_from_notes(notes: str, local_path: str) -> str:
    match = re.search(r"(?:^|\s)current_mesh=([^,\s]+)", notes)
    if match:
        return match.group(1)
    return current_frame_path_for(local_path)


def local_face_centroids(
    reference_mesh: Path,
    sample_mesh: Path,
    transform_record: dict[str, Any],
    warp_sample: bool = False,
) -> tuple[list[float], list[float]]:
    ref_piece = ET.parse(reference_mesh).getroot().find(".//Piece")
    sample_piece = ET.parse(sample_mesh).getroot().find(".//Piece")
    ref_points = parse_points_from_piece(ref_piece)
    sample_points = parse_points_from_piece(sample_piece)
    if warp_sample:
        sample_displacement = parse_point_vector(sample_piece, "displacement")
        if sample_displacement:
            sample_points = [
                vec_add(point, sample_displacement[i])
                for i, point in enumerate(sample_points)
            ]
    if len(ref_points) != len(sample_points) or not ref_points:
        return [math.nan] * 3, [math.nan] * 3

    origin = [float(x) for x in transform_record.get("origin", [0.0, 0.0, 0.0])]
    R = transform_record.get("R", [[1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0]])
    cols = [[float(R[r][c]) for r in range(3)] for c in range(3)]

    def local_z(point: list[float]) -> float:
        shifted = vec_sub(point, origin)
        return vec_dot(shifted, cols[2])

    z_values = [local_z(point) for point in ref_points]
    z_min = min(z_values)
    z_max = max(z_values)
    tol = max(1.0e-9, 1.0e-8 * max(1.0, abs(z_max - z_min)))
    bottom = [sample_points[i] for i, z in enumerate(z_values) if abs(z - z_min) <= tol]
    top = [sample_points[i] for i, z in enumerate(z_values) if abs(z - z_max) <= tol]
    return vec_mean(bottom), vec_mean(top)


def global_axis_endpoints(axis_path: Path, macro_element_id: int) -> tuple[list[float], list[float]]:
    piece = ET.parse(axis_path).getroot().find(".//Piece")
    points = parse_points_from_piece(piece)
    displacement = parse_point_vector(piece, "displacement")
    lines = parse_lines(piece)
    if macro_element_id < 0 or macro_element_id >= len(lines):
        return [math.nan] * 3, [math.nan] * 3
    ids = lines[macro_element_id]
    if not ids:
        return [math.nan] * 3, [math.nan] * 3
    first = ids[0]
    last = ids[-1]
    if not displacement:
        displacement = [[0.0, 0.0, 0.0] for _ in points]
    return vec_add(points[first], displacement[first]), vec_add(points[last], displacement[last])


def audit_time_index_and_endpoint_coincidence(
    root: Path,
    tolerance: float,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    issues: list[dict[str, str]] = []
    metrics: dict[str, Any] = {
        "local_global_endpoint_checks": 0,
        "local_global_endpoint_max_gap_m": None,
        "local_global_current_endpoint_max_gap_m": None,
        "local_global_reference_warp_endpoint_max_gap_m": None,
    }
    index_path = root / "recorders" / "multiscale_time_index.csv"
    if not index_path.exists():
        issues.append(issue("warning", index_path, "missing multiscale_time_index.csv"))
        return issues, metrics

    transform_path = root / "recorders" / "local_site_transform.json"
    if not transform_path.exists():
        issues.append(issue("warning", transform_path, "missing local_site_transform.json"))
        return issues, metrics
    transform_payload = json.loads(transform_path.read_text(encoding="utf-8"))
    transforms = {
        int(record.get("local_site_index", -1)): record
        for record in transform_payload.get("records", [])
    }

    rows: list[dict[str, str]]
    with index_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    global_times = {
        float(row["physical_time"])
        for row in rows
        if row.get("role") == "global_frame"
        and row.get("case_kind") == "fe2_one_way"
        and row.get("global_vtk_path")
    }
    local_times = {
        float(row["physical_time"])
        for row in rows
        if row.get("role", "").startswith("local_")
        and row.get("case_kind") == "fe2_one_way"
        and row.get("local_vtk_path")
    }
    initial_time = min(local_times) if local_times else None
    for time in sorted(global_times):
        if initial_time is not None and abs(time - initial_time) <= 1.0e-9:
            continue
        if not any(abs(time - local_time) <= 1.0e-9 for local_time in local_times):
            issues.append(issue(
                "error",
                index_path,
                f"global frame at t={time:.9g} has no local VTK row at the same time",
            ))

    axis_cache: dict[str, Path | None] = {}
    max_gap = 0.0
    current_max_gap = 0.0
    reference_warp_max_gap = 0.0
    for row in rows:
        role = row.get("role", "")
        if not role.startswith("local_"):
            continue
        local_path_text = row.get("local_vtk_path", "")
        global_path_text = row.get("global_vtk_path", "")
        if not local_path_text or not global_path_text:
            continue
        current_path_text = current_frame_path_from_notes(
            row.get("notes", ""), local_path_text)
        if not current_path_text:
            continue
        site_id = int(row.get("local_site_index") or -1)
        macro_element_id = int(row.get("macro_element_id") or -1)
        transform = transforms.get(site_id)
        if transform is None:
            issues.append(issue("error", transform_path, f"missing transform for local site {site_id}"))
            continue

        reference_mesh = root / local_path_text
        current_mesh = root / current_path_text
        global_vtm = root / global_path_text
        if not reference_mesh.exists() or not current_mesh.exists() or not global_vtm.exists():
            continue
        axis_key = str(global_vtm)
        if axis_key not in axis_cache:
            axis_cache[axis_key] = parse_vtm_axis_path(global_vtm)
        axis_path = axis_cache[axis_key]
        if axis_path is None or not axis_path.exists():
            issues.append(issue("error", global_vtm, "cannot resolve global axis block"))
            continue

        local_bottom, local_top = local_face_centroids(
            reference_mesh, current_mesh, transform, warp_sample=False)
        warped_bottom, warped_top = local_face_centroids(
            reference_mesh, reference_mesh, transform, warp_sample=True)
        global_bottom, global_top = global_axis_endpoints(
            axis_path, macro_element_id)
        bottom_gap = vec_norm(vec_sub(local_bottom, global_bottom))
        top_gap = vec_norm(vec_sub(local_top, global_top))
        warped_bottom_gap = vec_norm(vec_sub(warped_bottom, global_bottom))
        warped_top_gap = vec_norm(vec_sub(warped_top, global_top))
        current_gap = max(bottom_gap, top_gap)
        reference_warp_gap = max(warped_bottom_gap, warped_top_gap)
        gap = max(current_gap, reference_warp_gap)
        if math.isfinite(gap):
            max_gap = max(max_gap, gap)
            current_max_gap = max(current_max_gap, current_gap)
            reference_warp_max_gap = max(reference_warp_max_gap, reference_warp_gap)
            metrics["local_global_endpoint_checks"] += 1
        if gap > tolerance:
            issues.append(issue(
                "error",
                current_mesh,
                "local face centroids do not match global axis endpoints "
                f"for element {macro_element_id} at t={row.get('physical_time')} "
                f"(current_bottom_gap={bottom_gap:.6e} m, "
                f"current_top_gap={top_gap:.6e} m, "
                f"reference_warp_bottom_gap={warped_bottom_gap:.6e} m, "
                f"reference_warp_top_gap={warped_top_gap:.6e} m, "
                f"tol={tolerance:.6e} m)",
            ))

    if metrics["local_global_endpoint_checks"]:
        metrics["local_global_endpoint_max_gap_m"] = max_gap
        metrics["local_global_current_endpoint_max_gap_m"] = current_max_gap
        metrics["local_global_reference_warp_endpoint_max_gap_m"] = reference_warp_max_gap
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
    parser.add_argument(
        "--check-local-global-endpoints",
        action="store_true",
        help="check local current face centroids against the matching global axis endpoints at the same time",
    )
    parser.add_argument(
        "--endpoint-tolerance",
        type=float,
        default=5.0e-5,
        help="endpoint coincidence tolerance in metres for --check-local-global-endpoints",
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
        "local_global_endpoint_checks": 0,
        "local_global_endpoint_max_gap_m": None,
        "local_global_current_endpoint_max_gap_m": None,
        "local_global_reference_warp_endpoint_max_gap_m": None,
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

        if args.check_local_global_endpoints:
            endpoint_issues, endpoint_metrics = (
                audit_time_index_and_endpoint_coincidence(
                    root, args.endpoint_tolerance))
            report["issues"].extend(endpoint_issues)
            report.update(endpoint_metrics)

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)

    has_error = any(item["severity"] == "error" for item in report["issues"])
    return 0 if args.warn_only or not has_error else 1


if __name__ == "__main__":
    sys.exit(main())
