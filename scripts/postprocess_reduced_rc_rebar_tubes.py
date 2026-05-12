#!/usr/bin/env python3
"""Extract embedded Truss3 rebar cells from reduced-RC continuum VTU files.

The continuum cyclic benchmark writes Hex27 host cells and quadratic truss
bars in one mesh VTU.  This helper creates a separate rebar-tube VTU sequence
with the publication fields expected by the thesis/ParaView workflow.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


VTK_QUADRATIC_EDGE = 21
DEFAULT_STEEL_FY_MPA = 420.0
DEFAULT_BAR_RADIUS_M = 0.008
DEFAULT_BAR_AREA_M2 = math.pi * DEFAULT_BAR_RADIUS_M * DEFAULT_BAR_RADIUS_M


def _numbers(text: str, cast=float):
    return [cast(x) for x in (text or "").split()]


def _format(values) -> str:
    return "\n          " + " ".join(
        f"{v:.12g}" if isinstance(v, float) else str(v) for v in values
    ) + "\n        "


def _find_data_array(parent: ET.Element, name: str | None = None) -> ET.Element:
    for arr in parent.findall("DataArray"):
        if name is None or arr.attrib.get("Name") == name:
            return arr
    raise KeyError(f"DataArray not found: {name}")


def _read_rebar_history(path: Path):
    by_step_cell: dict[int, dict[int, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if not path.exists():
        return by_step_cell
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(row["runtime_step"])
                cell = int(row["bar_element_index"])
            except (KeyError, ValueError):
                continue
            by_step_cell[step][cell].append(row)
    return by_step_cell


def _mean(rows, key: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except ValueError:
            value = math.nan
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else 0.0


def _first_int(rows, key: str, default: int = -1) -> int:
    for row in rows:
        try:
            return int(float(row.get(key, "")))
        except ValueError:
            continue
    return default


def _extract_step(path: Path) -> int | None:
    match = re.search(r"step_(\d+)_mesh\.vtu$", path.name)
    return int(match.group(1)) if match else None


def _bond_state(
    gap: float,
    slip_reference_m: float,
    residual_ratio: float,
    adaptive_reference_max_factor: float,
    adaptive_residual_ratio_floor: float,
):
    s0 = max(abs(slip_reference_m), 1.0e-12)
    r0 = min(max(residual_ratio, 0.0), 1.0)
    if adaptive_reference_max_factor <= 1.0 and adaptive_residual_ratio_floor < 0.0:
        return s0, r0, 0.0
    x = abs(gap) / s0
    m = (x * x) / (1.0 + x * x)
    max_factor = max(1.0, adaptive_reference_max_factor)
    floor = r0 if adaptive_residual_ratio_floor < 0.0 else min(max(adaptive_residual_ratio_floor, 0.0), r0)
    return s0 * (1.0 + (max_factor - 1.0) * m), r0 - (r0 - floor) * m, m


def _bond_response(
    gap: float,
    penalty_alpha: float,
    slip_reference_m: float,
    residual_ratio: float,
    adaptive_reference_max_factor: float,
    adaptive_residual_ratio_floor: float,
):
    alpha = max(0.0, penalty_alpha)
    s_ref, ratio, fraction = _bond_state(
        gap,
        slip_reference_m,
        residual_ratio,
        adaptive_reference_max_factor,
        adaptive_residual_ratio_floor,
    )
    arg = max(-50.0, min(50.0, gap / s_ref))
    th = math.tanh(arg)
    sech2 = max(0.0, 1.0 - th * th)
    force = alpha * (ratio * gap + (1.0 - ratio) * s_ref * th)
    tangent = alpha * (ratio + (1.0 - ratio) * sech2)
    return force, tangent, s_ref, ratio, fraction


def write_rebar_tube_vtu(mesh_path: Path, rows_by_cell, output_path: Path, args) -> int:
    tree = ET.parse(mesh_path)
    piece = tree.getroot().find("./UnstructuredGrid/Piece")
    if piece is None:
        raise RuntimeError(f"Unsupported VTU layout: {mesh_path}")

    points_arr = _find_data_array(piece.find("Points"))
    point_values = _numbers(points_arr.text, float)
    points = [
        point_values[i : i + 3] for i in range(0, len(point_values), 3)
    ]

    point_data = piece.find("PointData")
    disp_arr = _find_data_array(point_data, "displacement") if point_data is not None else None
    disp_values = _numbers(disp_arr.text, float) if disp_arr is not None else [0.0] * (3 * len(points))
    displacements = [
        disp_values[i : i + 3] for i in range(0, len(disp_values), 3)
    ]

    cells = piece.find("Cells")
    connectivity = _numbers(_find_data_array(cells, "connectivity").text, int)
    offsets = _numbers(_find_data_array(cells, "offsets").text, int)
    types = _numbers(_find_data_array(cells, "types").text, int)

    out_points: list[float] = []
    out_displacements: list[float] = []
    out_connectivity: list[int] = []
    out_offsets: list[int] = []
    out_types: list[int] = []
    tube_radius: list[float] = []
    bar_id: list[int] = []
    bar_area: list[float] = []
    axial_strain: list[float] = []
    axial_stress: list[float] = []
    yield_ratio: list[float] = []
    slip: list[float] = []
    bond_slip: list[float] = []
    bond_slip_axial: list[float] = []
    bond_slip_transverse: list[float] = []
    bond_slip_vector: list[float] = []
    bond_force: list[float] = []
    bond_force_axial: list[float] = []
    bond_force_transverse: list[float] = []
    bond_tangent_axial: list[float] = []
    bond_tangent_ratio: list[float] = []
    bond_effective_slip_reference: list[float] = []
    bond_residual_stiffness_ratio: list[float] = []
    bond_adaptive_fraction: list[float] = []
    bond_slip_ratio: list[float] = []
    bond_slip_valid: list[float] = []

    start = 0
    out_node = 0
    for cell_index, (offset, cell_type) in enumerate(zip(offsets, types)):
        cell_conn = connectivity[start:offset]
        start = offset
        if cell_type != VTK_QUADRATIC_EDGE:
            continue

        rows = rows_by_cell.get(cell_index, [])
        stress = _mean(rows, "stress_xx_MPa")
        strain = _mean(rows, "axial_strain")
        gap = max(
            [_mean(rows, "projected_gap_norm_m"), _mean(rows, "projected_axial_gap_m")],
            key=abs,
        )
        axial_gap = _mean(rows, "projected_axial_gap_m")
        gap_norm = abs(_mean(rows, "projected_gap_norm_m")) or abs(axial_gap)
        p0 = points[cell_conn[0]]
        p1 = points[cell_conn[-1]]
        axis = [p1[i] - p0[i] for i in range(3)]
        length = math.sqrt(sum(v * v for v in axis))
        unit = [v / length for v in axis] if length > 1.0e-14 else [0.0, 0.0, 1.0]
        transverse_gap = math.sqrt(max(0.0, gap_norm * gap_norm - axial_gap * axial_gap))
        force_axial, tangent_axial, s_eff, r_eff, adapt_frac = _bond_response(
            axial_gap,
            args.penalty_alpha,
            args.bond_slip_reference,
            args.bond_slip_residual_ratio,
            args.bond_slip_adaptive_reference_max_factor,
            args.bond_slip_adaptive_residual_ratio_floor,
        )

        for nid in cell_conn:
            out_points.extend(points[nid])
            out_displacements.extend(displacements[nid])
            out_connectivity.append(out_node)
            out_node += 1

        out_offsets.append(len(out_connectivity))
        out_types.append(VTK_QUADRATIC_EDGE)
        tube_radius.append(DEFAULT_BAR_RADIUS_M)
        bar_id.append(_first_int(rows, "bar_index", len(bar_id)))
        bar_area.append(DEFAULT_BAR_AREA_M2)
        axial_strain.append(strain)
        axial_stress.append(stress)
        yield_ratio.append(abs(stress) / DEFAULT_STEEL_FY_MPA)
        slip.append(abs(gap))
        bond_slip.append(gap_norm)
        bond_slip_axial.append(axial_gap)
        bond_slip_transverse.append(transverse_gap)
        bond_slip_vector.extend([axial_gap * unit[i] for i in range(3)])
        bond_force.append(abs(force_axial))
        bond_force_axial.append(force_axial)
        bond_force_transverse.append(0.0)
        bond_tangent_axial.append(tangent_axial)
        bond_tangent_ratio.append(
            tangent_axial / args.penalty_alpha if args.penalty_alpha > 0.0 else 0.0
        )
        bond_effective_slip_reference.append(s_eff)
        bond_residual_stiffness_ratio.append(r_eff)
        bond_adaptive_fraction.append(adapt_frac)
        bond_slip_ratio.append(gap_norm / max(args.bond_slip_reference, 1.0e-12))
        bond_slip_valid.append(1.0 if rows else 0.0)

    n_points = len(out_points) // 3
    n_cells = len(out_types)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1", byte_order="LittleEndian")
    grid = ET.SubElement(root, "UnstructuredGrid")
    out_piece = ET.SubElement(
        grid,
        "Piece",
        NumberOfPoints=str(n_points),
        NumberOfCells=str(n_cells),
    )
    pd = ET.SubElement(out_piece, "PointData", Vectors="displacement")
    ET.SubElement(
        pd,
        "DataArray",
        type="Float64",
        Name="displacement",
        NumberOfComponents="3",
        format="ascii",
    ).text = _format(out_displacements)

    cd = ET.SubElement(out_piece, "CellData", Scalars="axial_stress")
    for name, typ, values in [
        ("TubeRadius", "Float64", tube_radius),
        ("bar_id", "Int32", bar_id),
        ("bar_area", "Float64", bar_area),
        ("axial_strain", "Float64", axial_strain),
        ("axial_stress", "Float64", axial_stress),
        ("yield_ratio", "Float64", yield_ratio),
        ("slip", "Float64", slip),
        ("bond_slip", "Float64", bond_slip),
        ("bond_slip_axial", "Float64", bond_slip_axial),
        ("bond_slip_transverse", "Float64", bond_slip_transverse),
        ("bond_force", "Float64", bond_force),
        ("bond_force_axial", "Float64", bond_force_axial),
        ("bond_force_transverse", "Float64", bond_force_transverse),
        ("bond_tangent_axial", "Float64", bond_tangent_axial),
        ("bond_tangent_ratio", "Float64", bond_tangent_ratio),
        ("bond_effective_slip_reference", "Float64", bond_effective_slip_reference),
        ("bond_residual_stiffness_ratio", "Float64", bond_residual_stiffness_ratio),
        ("bond_adaptive_fraction", "Float64", bond_adaptive_fraction),
        ("bond_slip_ratio", "Float64", bond_slip_ratio),
        ("bond_slip_valid", "Float64", bond_slip_valid),
    ]:
        ET.SubElement(cd, "DataArray", type=typ, Name=name, format="ascii").text = _format(values)
    ET.SubElement(
        cd,
        "DataArray",
        type="Float64",
        Name="bond_slip_vector",
        NumberOfComponents="3",
        format="ascii",
    ).text = _format(bond_slip_vector)

    pts = ET.SubElement(out_piece, "Points")
    ET.SubElement(
        pts,
        "DataArray",
        type="Float64",
        NumberOfComponents="3",
        format="ascii",
    ).text = _format(out_points)

    out_cells = ET.SubElement(out_piece, "Cells")
    ET.SubElement(out_cells, "DataArray", type="Int64", Name="connectivity", format="ascii").text = _format(out_connectivity)
    ET.SubElement(out_cells, "DataArray", type="Int64", Name="offsets", format="ascii").text = _format(out_offsets)
    ET.SubElement(out_cells, "DataArray", type="UInt8", Name="types", format="ascii").text = _format(out_types)

    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return n_cells


def write_pvd(path: Path, entries: list[tuple[int, Path]]) -> None:
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for step, file_path in entries:
        rel = file_path.name
        lines.append(
            f'    <DataSet timestep="{step}" group="" part="0" file="{rel}"/>'
        )
    lines.extend(["  </Collection>", "</VTKFile>", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--penalty-alpha", type=float, default=0.0)
    parser.add_argument("--bond-slip-reference", type=float, default=5.0e-4)
    parser.add_argument("--bond-slip-residual-ratio", type=float, default=0.2)
    parser.add_argument(
        "--bond-slip-adaptive-reference-max-factor", type=float, default=1.0
    )
    parser.add_argument(
        "--bond-slip-adaptive-residual-ratio-floor", type=float, default=-1.0
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    vtk_dir = run_dir / "vtk"
    history = _read_rebar_history(run_dir / "rebar_history.csv")
    entries: list[tuple[int, Path]] = []
    total_cells = 0
    for mesh in sorted(vtk_dir.glob("continuum_step_*_mesh.vtu")):
        step = _extract_step(mesh)
        if step is None:
            continue
        out = vtk_dir / mesh.name.replace("_mesh.vtu", "_rebar_tubes.vtu")
        total_cells += write_rebar_tube_vtu(mesh, history.get(step, {}), out, args)
        entries.append((step, out))

    if entries:
        write_pvd(vtk_dir / "continuum_rebar_tubes.pvd", entries)
    print(
        f"wrote {len(entries)} rebar tube VTU files "
        f"({total_cells} quadratic truss cells) in {vtk_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
