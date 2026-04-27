#!/usr/bin/env python3
"""
OpenSeesPy 3D-continuum reference for the reduced RC-column campaign.

This bridge is intentionally conservative.  It does not try to emulate the
fall_n embedded-bar formulation directly.  Instead it builds an external,
conforming OpenSees model in which the longitudinal steel bars share nodes with
the brick mesh.  That makes the comparison useful as a physical trend check:
if a conforming OpenSees solid behaves like the fall_n solid and unlike the
fiber beam, the gap is likely kinematic/material rather than an implementation
bug in fall_n's embedded coupling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

from opensees_reduced_rc_column_reference import (
    ProtocolPoint,
    ReducedRCColumnReferenceSpec,
    analyze_displacement_increment_summary,
    bar_area,
    build_protocol_points,
    concrete_initial_modulus,
    configure_static_analysis,
    ensure_dir,
    external_mapping_policy_catalog,
    mander_confined_strength,
    mpa_to_pa,
    parse_amplitudes_mm,
    rc_section_rebar_layout,
    safe_test_iterations,
    safe_test_norm,
    structural_convergence_profile_families,
    structural_increment_convergence_profiles,
    try_import_opensees,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a conforming 3D-continuum OpenSeesPy reference for the "
            "reduced RC-column benchmark."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis", choices=("monotonic", "cyclic"), default="cyclic")
    parser.add_argument(
        "--solid-element",
        choices=("stdBrick", "bbarBrick", "SSPbrick"),
        default="SSPbrick",
        help=(
            "OpenSees solid element. SSPbrick is the default cheap locking-"
            "resistant Hex8 reference; stdBrick and bbarBrick are retained as "
            "sensitivity checks."
        ),
    )
    parser.add_argument(
        "--concrete-model",
        choices=("elastic-isotropic", "ASDConcrete3D", "Damage2p"),
        default="elastic-isotropic",
        help=(
            "OpenSees nDMaterial used by the brick host. The elastic route is "
            "the first kinematic/assembly control; ASDConcrete3D/Damage2p are "
            "nonlinear sensitivity paths whose calibration is external to "
            "fall_n's Ko-Bathe/crack-band laws."
        ),
    )
    parser.add_argument(
        "--steel-model",
        choices=("Steel02", "Elastic"),
        default="Steel02",
        help="Uniaxial steel law used by conforming truss bars.",
    )
    parser.add_argument(
        "--reinforcement-mode",
        choices=("none", "conforming-eight-bar"),
        default="conforming-eight-bar",
    )
    parser.add_argument(
        "--mapping-policy",
        choices=tuple(external_mapping_policy_catalog().keys()),
        default=None,
        help="Steel02 mapping policy reused from the structural OpenSees bridge.",
    )
    parser.add_argument("--steel-r0", type=float, default=None)
    parser.add_argument("--steel-cr1", type=float, default=None)
    parser.add_argument("--steel-cr2", type=float, default=None)
    parser.add_argument("--steel-a1", type=float, default=None)
    parser.add_argument("--steel-a2", type=float, default=None)
    parser.add_argument("--steel-a3", type=float, default=None)
    parser.add_argument("--steel-a4", type=float, default=None)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=8)
    parser.add_argument(
        "--longitudinal-bias-power",
        type=float,
        default=1.5,
        help="Power-law clustering of z stations toward selected end(s).",
    )
    parser.add_argument(
        "--longitudinal-bias-location",
        choices=("fixed-end", "loaded-end", "both-ends"),
        default="fixed-end",
        help=(
            "Where to place the longitudinal clustering: fixed-end (base), "
            "loaded-end (top displacement face), or both-ends."
        ),
    )
    parser.add_argument(
        "--host-concrete-zoning-mode",
        choices=("uniform", "cover-core-split"),
        default="cover-core-split",
    )
    parser.add_argument(
        "--lateral-control-mode",
        choices=("sp-path", "single-node-displacement-control"),
        default="sp-path",
        help=(
            "sp-path imposes the target lateral displacement on the whole top "
            "face through a Path time series. single-node-displacement-control "
            "is kept only as a diagnostic and controls one top node."
        ),
    )
    parser.add_argument(
        "--solver-profile-family",
        choices=structural_convergence_profile_families(),
        default="robust-large-amplitude",
    )
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--monotonic-tip-mm", type=float, default=2.5)
    parser.add_argument("--monotonic-steps", type=int, default=8)
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--reversal-substep-factor", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage manifest and mesh contracts without importing OpenSeesPy.",
    )
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def selected_steel_mapping_policy(args: argparse.Namespace):
    if args.mapping_policy is not None:
        label = args.mapping_policy
    elif args.steel_model == "Elastic":
        label = "elasticized-parity"
    else:
        label = "cyclic-diagnostic" if args.analysis == "cyclic" else "monotonic-reference"
    policy = external_mapping_policy_catalog()[label]
    overrides = {
        key: value
        for key, value in (
            ("steel_r0", args.steel_r0),
            ("steel_cr1", args.steel_cr1),
            ("steel_cr2", args.steel_cr2),
            ("steel_a1", args.steel_a1),
            ("steel_a2", args.steel_a2),
            ("steel_a3", args.steel_a3),
            ("steel_a4", args.steel_a4),
        )
        if value is not None
    }
    return replace(policy, label=f"{policy.label}+steel_override", **overrides) if overrides else policy


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def unique_sorted(values: Iterable[float], *, tol: float = 1.0e-12) -> list[float]:
    out: list[float] = []
    for value in sorted(values):
        if not out or abs(value - out[-1]) > tol:
            out.append(float(value))
    return out


def conforming_transverse_coordinates(
    *,
    half_width: float,
    requested_elements: int,
    mandatory_points: Iterable[float],
) -> list[float]:
    uniform = [
        -half_width + 2.0 * half_width * i / max(requested_elements, 1)
        for i in range(max(requested_elements, 1) + 1)
    ]
    return unique_sorted([*uniform, *mandatory_points, -half_width, half_width, 0.0])


def biased_longitudinal_coordinates(
    height: float,
    element_count: int,
    bias_power: float,
    bias_location: str = "fixed-end",
) -> list[float]:
    n = max(element_count, 1)
    p = max(bias_power, 1.0e-9)
    coords: list[float] = []
    for i in range(n + 1):
        s = i / n
        if bias_location == "loaded-end":
            biased = 1.0 - (1.0 - s) ** p
        elif bias_location == "both-ends":
            biased = 0.5 * (2.0 * s) ** p if s <= 0.5 else 1.0 - 0.5 * (2.0 * (1.0 - s)) ** p
        else:
            biased = s ** p
        coords.append(height * biased)
    coords[0] = 0.0
    coords[-1] = height
    return coords


def section_bar_positions_global(
    spec: ReducedRCColumnReferenceSpec,
) -> list[tuple[float, float]]:
    # Existing fall_n/OpenSees structural helpers use section coordinates
    # (y, z).  The continuum benchmark bends by imposing global-x drift, so
    # section z maps to global x and section y maps to global y.
    return [(section_z, section_y) for section_y, section_z in rc_section_rebar_layout(spec)]


def find_coordinate_index(values: list[float], target: float, *, tol: float = 1.0e-10) -> int:
    for idx, value in enumerate(values):
        if abs(value - target) <= tol:
            return idx
    raise RuntimeError(f"Conforming mesh does not contain required coordinate {target:.16e}.")


def node_tributary_weight(values: list[float], index: int) -> float:
    if len(values) == 1:
        return 1.0
    left = values[index] - values[index - 1] if index > 0 else values[1] - values[0]
    right = values[index + 1] - values[index] if index + 1 < len(values) else values[-1] - values[-2]
    return 0.5 * (left + right)


def top_face_load_weights(x_coords: list[float], y_coords: list[float]) -> dict[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    total = 0.0
    for i in range(len(x_coords)):
        wx = node_tributary_weight(x_coords, i)
        for j in range(len(y_coords)):
            wy = node_tributary_weight(y_coords, j)
            w = wx * wy
            weights[(i, j)] = w
            total += w
    return {key: value / total for key, value in weights.items()}


def concrete_core_bounds(spec: ReducedRCColumnReferenceSpec) -> tuple[float, float, float, float]:
    x_core = 0.5 * spec.section_h_m - spec.cover_m
    y_core = 0.5 * spec.section_b_m - spec.cover_m
    return -x_core, x_core, -y_core, y_core


def element_is_core(
    x_mid: float,
    y_mid: float,
    spec: ReducedRCColumnReferenceSpec,
    zoning_mode: str,
) -> bool:
    if zoning_mode != "cover-core-split":
        return False
    x_min, x_max, y_min, y_max = concrete_core_bounds(spec)
    return x_min <= x_mid <= x_max and y_min <= y_mid <= y_max


def define_concrete_materials(
    ops: object,
    args: argparse.Namespace,
    spec: ReducedRCColumnReferenceSpec,
) -> dict[str, int]:
    e_pa = mpa_to_pa(concrete_initial_modulus(spec.concrete_fpc_mpa))
    nu = spec.concrete_nu
    cover_tag = 1
    core_tag = 2
    ft_pa = 0.10 * mpa_to_pa(spec.concrete_fpc_mpa)
    cover_fc_pa = mpa_to_pa(spec.concrete_fpc_mpa)
    core_fc_pa = mpa_to_pa(
        mander_confined_strength(spec.concrete_fpc_mpa, spec.rho_s, spec.tie_fy_mpa)
    )

    def define_one(tag: int, fc_pa: float) -> None:
        if args.concrete_model == "elastic-isotropic":
            ops.nDMaterial("ElasticIsotropic", tag, e_pa, nu)
            return
        if args.concrete_model == "ASDConcrete3D":
            ops.nDMaterial("ASDConcrete3D", tag, e_pa, nu, "-fc", fc_pa, "-ft", ft_pa)
            return
        if args.concrete_model == "Damage2p":
            ops.nDMaterial(
                "Damage2p",
                tag,
                fc_pa,
                "-fct",
                ft_pa,
                "-E",
                e_pa,
                "-ni",
                nu,
                "-Gt",
                100.0,
                "-Gc",
                10_000.0,
                "-rho_bar",
                0.0,
                "-H",
                0.0,
                "-theta",
                0.0,
                "-tangent",
                1,
            )
            return
        raise ValueError(f"Unsupported concrete model: {args.concrete_model}")

    define_one(cover_tag, cover_fc_pa)
    define_one(core_tag, core_fc_pa)
    return {"cover": cover_tag, "core": core_tag}


def define_steel_material(
    ops: object,
    args: argparse.Namespace,
    spec: ReducedRCColumnReferenceSpec,
) -> int:
    tag = 100
    mapping_policy = selected_steel_mapping_policy(args)
    if args.steel_model == "Elastic":
        ops.uniaxialMaterial("Elastic", tag, mpa_to_pa(spec.steel_E_mpa))
    else:
        ops.uniaxialMaterial(
            "Steel02",
            tag,
            mpa_to_pa(spec.steel_fy_mpa),
            mpa_to_pa(spec.steel_E_mpa),
            spec.steel_b,
            mapping_policy.steel_r0,
            mapping_policy.steel_cr1,
            mapping_policy.steel_cr2,
            mapping_policy.steel_a1,
            mapping_policy.steel_a2,
            mapping_policy.steel_a3,
            mapping_policy.steel_a4,
        )
    return tag


def solid_element_command(
    element_name: str,
    tag: int,
    nodes: tuple[int, ...],
    mat_tag: int,
) -> tuple[object, ...]:
    if element_name in {"stdBrick", "bbarBrick", "SSPbrick"}:
        return (element_name, tag, *nodes, mat_tag)
    raise ValueError(f"Unsupported solid element {element_name!r}")


def build_conforming_continuum_model(
    ops: object,
    args: argparse.Namespace,
    spec: ReducedRCColumnReferenceSpec,
) -> dict[str, object]:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)

    bar_positions = section_bar_positions_global(spec)
    mandatory_x = [x for x, _ in bar_positions]
    mandatory_y = [y for _, y in bar_positions]
    x_coords = conforming_transverse_coordinates(
        half_width=0.5 * spec.section_h_m,
        requested_elements=args.nx,
        mandatory_points=mandatory_x,
    )
    y_coords = conforming_transverse_coordinates(
        half_width=0.5 * spec.section_b_m,
        requested_elements=args.ny,
        mandatory_points=mandatory_y,
    )
    z_coords = biased_longitudinal_coordinates(
        spec.column_height_m,
        args.nz,
        args.longitudinal_bias_power,
        args.longitudinal_bias_location,
    )

    def node_tag(i: int, j: int, k: int) -> int:
        return 1 + i + len(x_coords) * (j + len(y_coords) * k)

    for k, z in enumerate(z_coords):
        for j, y in enumerate(y_coords):
            for i, x in enumerate(x_coords):
                ops.node(node_tag(i, j, k), x, y, z)

    base_nodes = [node_tag(i, j, 0) for j in range(len(y_coords)) for i in range(len(x_coords))]
    top_nodes = [
        node_tag(i, j, len(z_coords) - 1)
        for j in range(len(y_coords))
        for i in range(len(x_coords))
    ]
    for tag in base_nodes:
        ops.fix(tag, 1, 1, 1)

    concrete_tags = define_concrete_materials(ops, args, spec)
    steel_tag = define_steel_material(ops, args, spec)

    solid_elements: list[dict[str, object]] = []
    element_tag = 1
    for k in range(len(z_coords) - 1):
        for j in range(len(y_coords) - 1):
            for i in range(len(x_coords) - 1):
                nodes = (
                    node_tag(i, j, k),
                    node_tag(i + 1, j, k),
                    node_tag(i + 1, j + 1, k),
                    node_tag(i, j + 1, k),
                    node_tag(i, j, k + 1),
                    node_tag(i + 1, j, k + 1),
                    node_tag(i + 1, j + 1, k + 1),
                    node_tag(i, j + 1, k + 1),
                )
                x_mid = 0.5 * (x_coords[i] + x_coords[i + 1])
                y_mid = 0.5 * (y_coords[j] + y_coords[j + 1])
                is_core = element_is_core(x_mid, y_mid, spec, args.host_concrete_zoning_mode)
                mat_tag = concrete_tags["core" if is_core else "cover"]
                ops.element(*solid_element_command(args.solid_element, element_tag, nodes, mat_tag))
                solid_elements.append(
                    {
                        "element_tag": element_tag,
                        "i": i,
                        "j": j,
                        "k": k,
                        "x_mid": x_mid,
                        "y_mid": y_mid,
                        "z_mid": 0.5 * (z_coords[k] + z_coords[k + 1]),
                        "zone": "confined_core" if is_core else "unconfined_cover",
                        "material_tag": mat_tag,
                    }
                )
                element_tag += 1

    truss_elements: list[dict[str, object]] = []
    if args.reinforcement_mode == "conforming-eight-bar":
        area = bar_area(spec.longitudinal_bar_diameter_m)
        for bar_index, (x_bar, y_bar) in enumerate(bar_positions):
            i = find_coordinate_index(x_coords, x_bar)
            j = find_coordinate_index(y_coords, y_bar)
            for k in range(len(z_coords) - 1):
                n_i = node_tag(i, j, k)
                n_j = node_tag(i, j, k + 1)
                ops.element("truss", element_tag, n_i, n_j, area, steel_tag)
                truss_elements.append(
                    {
                        "element_tag": element_tag,
                        "bar_index": bar_index,
                        "segment_index": k,
                        "node_i": n_i,
                        "node_j": n_j,
                        "x": x_bar,
                        "y": y_bar,
                        "z_i": z_coords[k],
                        "z_j": z_coords[k + 1],
                        "area_m2": area,
                        "material_tag": steel_tag,
                    }
                )
                element_tag += 1

    return {
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "node_tag": node_tag,
        "base_nodes": base_nodes,
        "top_nodes": top_nodes,
        "solid_elements": solid_elements,
        "truss_elements": truss_elements,
        "concrete_material_tags": concrete_tags,
        "steel_material_tag": steel_tag,
    }


def run_axial_preload(
    ops: object,
    args: argparse.Namespace,
    model: dict[str, object],
) -> dict[str, object]:
    if args.axial_compression_mn <= 0.0:
        return {"status": "skipped", "accepted_steps": 0}

    x_coords = list(model["x_coords"])
    y_coords = list(model["y_coords"])
    node_tag = model["node_tag"]
    top_k = len(model["z_coords"]) - 1
    weights = top_face_load_weights(x_coords, y_coords)
    total_force_n = -args.axial_compression_mn * 1.0e6

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for (i, j), weight in weights.items():
        ops.load(node_tag(i, j, top_k), 0.0, 0.0, total_force_n * weight)

    configure_static_analysis(ops, profile_family=args.solver_profile_family)
    steps = max(args.axial_preload_steps, 1)
    ops.integrator("LoadControl", 1.0 / steps)
    ops.analysis("Static")
    ok = ops.analyze(steps)
    if ok != 0:
        raise RuntimeError("OpenSees continuum axial preload did not converge.")
    ops.loadConst("-time", 0.0)
    return {
        "status": "completed",
        "accepted_steps": steps,
        "total_force_MN": args.axial_compression_mn,
    }


def configure_lateral_sp_path(
    ops: object,
    model: dict[str, object],
    protocol: list[ProtocolPoint],
) -> None:
    path_args: list[object] = ["-time"]
    path_args.extend(float(point.p) for point in protocol)
    path_args.append("-values")
    path_args.extend(float(point.target_drift_m) for point in protocol)
    path_args.append("-useLast")
    ops.timeSeries("Path", 2, *path_args)
    ops.pattern("Plain", 2, 2)
    for node in model["top_nodes"]:
        ops.sp(int(node), 1, 1.0)


def safe_revert_to_last_commit(ops: object) -> None:
    try:
        ops.revertToLastCommit()
    except Exception:
        pass


def sp_path_single_step(
    ops: object,
    *,
    pseudo_time_increment: float,
    top_probe_node: int,
    target_drift_m: float,
    profile,
) -> dict[str, object]:
    time_before = float(ops.getTime())
    dof_before = float(ops.nodeDisp(top_probe_node, 1))
    configure_static_analysis(ops, profile=profile)
    ops.integrator("LoadControl", pseudo_time_increment)
    ops.analysis("Static")
    converged = ops.analyze(1) == 0
    time_after = float(ops.getTime())
    dof_after = float(ops.nodeDisp(top_probe_node, 1))
    error = dof_after - target_drift_m
    tolerance = 1.0e-9 + 1.0e-6 * max(abs(target_drift_m), 1.0e-12)
    success = bool(converged and abs(error) <= tolerance)
    if not success:
        safe_revert_to_last_commit(ops)
    return {
        "success": success,
        "solver_converged": bool(converged),
        "accepted_substep_count": 1 if success else 0,
        "max_bisection_level": 0,
        "newton_iterations": safe_test_iterations(ops),
        "test_norm": safe_test_norm(ops),
        "domain_time_before": time_before,
        "domain_time_after": time_after,
        "control_dof_before": dof_before,
        "control_dof_after": dof_after,
        "control_increment_requested": target_drift_m - dof_before,
        "control_increment_achieved": dof_after - dof_before,
        "control_increment_error": error,
        "solver_profile_label": profile.label,
        "solver_algorithm_name": profile.algorithm_name,
    }


def sp_path_step(
    ops: object,
    *,
    p_start: float,
    p_end: float,
    drift_start_m: float,
    drift_end_m: float,
    top_probe_node: int,
    profile_family: str,
    max_bisections: int,
) -> dict[str, object]:
    profiles = structural_increment_convergence_profiles(profile_family)

    def recurse(
        local_p_start: float,
        local_p_end: float,
        local_drift_start: float,
        local_drift_end: float,
        depth: int,
    ) -> dict[str, object]:
        increment = local_p_end - local_p_start
        for profile in profiles:
            control = sp_path_single_step(
                ops,
                pseudo_time_increment=increment,
                top_probe_node=top_probe_node,
                target_drift_m=local_drift_end,
                profile=profile,
            )
            if control["success"]:
                control["max_bisection_level"] = depth
                return control

        if depth >= max(max_bisections, 0):
            return {
                "success": False,
                "solver_converged": False,
                "accepted_substep_count": 0,
                "max_bisection_level": depth,
                "newton_iterations": math.nan,
                "test_norm": math.nan,
                "domain_time_before": math.nan,
                "domain_time_after": math.nan,
                "control_dof_before": math.nan,
                "control_dof_after": math.nan,
                "control_increment_requested": local_drift_end - local_drift_start,
                "control_increment_achieved": math.nan,
                "control_increment_error": math.nan,
                "solver_profile_label": "",
                "solver_algorithm_name": "",
            }

        mid_p = 0.5 * (local_p_start + local_p_end)
        mid_drift = 0.5 * (local_drift_start + local_drift_end)
        first = recurse(local_p_start, mid_p, local_drift_start, mid_drift, depth + 1)
        if not first["success"]:
            return first
        second = recurse(mid_p, local_p_end, mid_drift, local_drift_end, depth + 1)
        if not second["success"]:
            return second

        first_iterations = float(first["newton_iterations"])
        second_iterations = float(second["newton_iterations"])
        requested = local_drift_end - local_drift_start
        achieved = float(second["control_dof_after"]) - float(first["control_dof_before"])
        return {
            "success": True,
            "solver_converged": bool(first["solver_converged"] and second["solver_converged"]),
            "accepted_substep_count": int(first["accepted_substep_count"])
            + int(second["accepted_substep_count"]),
            "max_bisection_level": max(
                int(first["max_bisection_level"]),
                int(second["max_bisection_level"]),
            ),
            "newton_iterations": first_iterations + second_iterations
            if math.isfinite(first_iterations) and math.isfinite(second_iterations)
            else math.nan,
            "test_norm": second["test_norm"],
            "domain_time_before": first["domain_time_before"],
            "domain_time_after": second["domain_time_after"],
            "control_dof_before": first["control_dof_before"],
            "control_dof_after": second["control_dof_after"],
            "control_increment_requested": requested,
            "control_increment_achieved": achieved,
            "control_increment_error": achieved - requested,
            "solver_profile_label": second["solver_profile_label"],
            "solver_algorithm_name": second["solver_algorithm_name"],
        }

    return recurse(p_start, p_end, drift_start_m, drift_end_m, 0)


def sum_reaction(ops: object, nodes: Iterable[int], dof: int) -> float:
    return sum(float(ops.nodeReaction(int(node), dof)) for node in nodes)


def average_displacement(ops: object, nodes: Iterable[int], dof: int) -> float:
    values = [float(ops.nodeDisp(int(node), dof)) for node in nodes]
    return sum(values) / len(values) if values else math.nan


def axial_force_from_truss_response(ops: object, tag: int) -> float:
    for query in (("axialForce",), ("basicForce",), ("force",)):
        try:
            values = ops.eleResponse(tag, *query)
        except Exception:
            values = None
        if values is None:
            continue
        if isinstance(values, (float, int)):
            return float(values)
        if values:
            return float(values[0])
    return math.nan


def truss_green_strain_from_nodes(ops: object, node_i: int, node_j: int) -> float:
    xi = [float(value) for value in ops.nodeCoord(node_i)]
    xj = [float(value) for value in ops.nodeCoord(node_j)]
    ui = [float(value) for value in ops.nodeDisp(node_i)]
    uj = [float(value) for value in ops.nodeDisp(node_j)]
    ref = math.sqrt(sum((xj[a] - xi[a]) ** 2 for a in range(3)))
    cur = math.sqrt(sum((xj[a] + uj[a] - xi[a] - ui[a]) ** 2 for a in range(3)))
    return (cur - ref) / ref if ref > 0.0 else math.nan


def sample_bar_rows(
    ops: object,
    model: dict[str, object],
    point: ProtocolPoint,
    drift_m: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for truss in model["truss_elements"]:
        force = axial_force_from_truss_response(ops, int(truss["element_tag"]))
        area = float(truss["area_m2"])
        rows.append(
            {
                "step": point.step,
                "p": point.p,
                "drift_m": drift_m,
                "bar_index": int(truss["bar_index"]),
                "segment_index": int(truss["segment_index"]),
                "element_tag": int(truss["element_tag"]),
                "x": float(truss["x"]),
                "y": float(truss["y"]),
                "z_mid": 0.5 * (float(truss["z_i"]) + float(truss["z_j"])),
                "axial_strain": truss_green_strain_from_nodes(
                    ops,
                    int(truss["node_i"]),
                    int(truss["node_j"]),
                ),
                "axial_force_MN": force / 1.0e6 if math.isfinite(force) else math.nan,
                "axial_stress_MPa": force / area / 1.0e6
                if math.isfinite(force) and area > 0.0
                else math.nan,
            }
        )
    return rows


def sample_state(
    ops: object,
    model: dict[str, object],
    point: ProtocolPoint,
    control: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    ops.reactions()
    top_nodes = [int(node) for node in model["top_nodes"]]
    base_nodes = [int(node) for node in model["base_nodes"]]
    drift = average_displacement(ops, top_nodes, 1)
    top_axial = average_displacement(ops, top_nodes, 3)
    base_shear = sum_reaction(ops, base_nodes, 1) / 1.0e6
    base_axial = sum_reaction(ops, base_nodes, 3) / 1.0e6
    h = {
        "step": point.step,
        "p": point.p,
        "drift_m": drift,
        "base_shear_MN": base_shear,
    }
    c = {
        "step": point.step,
        "p": point.p,
        "target_drift_m": point.target_drift_m,
        "actual_tip_drift_m": drift,
        "top_axial_displacement_m": top_axial,
        "base_shear_MN": base_shear,
        "base_axial_reaction_MN": base_axial,
        "accepted_substep_count": control.get("accepted_substep_count", 0) if control else 0,
        "max_bisection_level": control.get("max_bisection_level", 0) if control else 0,
        "newton_iterations": control.get("newton_iterations", 0.0) if control else 0.0,
        "test_norm": control.get("test_norm", math.nan) if control else math.nan,
        "solver_profile_label": control.get("solver_profile_label", "") if control else "",
        "solver_algorithm_name": control.get("solver_algorithm_name", "") if control else "",
        "solver_converged": int(control.get("solver_converged", True)) if control else 1,
        "control_increment_requested": control.get("control_increment_requested", 0.0)
        if control
        else 0.0,
        "control_increment_achieved": control.get("control_increment_achieved", 0.0)
        if control
        else 0.0,
        "control_increment_error": control.get("control_increment_error", 0.0)
        if control
        else 0.0,
    }
    return h, c, sample_bar_rows(ops, model, point, drift)


def mesh_contract_rows(model: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    coord_rows: list[dict[str, object]] = []
    for axis, values in (
        ("x", model["x_coords"]),
        ("y", model["y_coords"]),
        ("z", model["z_coords"]),
    ):
        for index, value in enumerate(values):
            coord_rows.append({"axis": axis, "index": index, "value_m": value})
    bar_rows = [dict(row) for row in model["truss_elements"]]
    return coord_rows, bar_rows


def run_analysis(args: argparse.Namespace) -> int:
    spec = ReducedRCColumnReferenceSpec()
    protocol = build_protocol_points(args)
    mapping_policy = selected_steel_mapping_policy(args)
    out_dir = args.output_dir.resolve()
    ensure_dir(out_dir)

    manifest: dict[str, object] = {
        "benchmark_kind": "opensees_continuum_conforming_rc_column_reference",
        "tool": "OpenSeesPy",
        "status": "staged" if args.dry_run else "running",
        "analysis": args.analysis,
        "solid_element": args.solid_element,
        "concrete_model": args.concrete_model,
        "steel_model": args.steel_model,
        "reinforcement_mode": args.reinforcement_mode,
        "mesh_request": {
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "longitudinal_bias_power": args.longitudinal_bias_power,
            "longitudinal_bias_location": args.longitudinal_bias_location,
            "host_concrete_zoning_mode": args.host_concrete_zoning_mode,
        },
        "lateral_control_mode": args.lateral_control_mode,
        "solver_profile_family": args.solver_profile_family,
        "reference_spec": asdict(spec),
        "mapping_policy": asdict(mapping_policy),
        "cyclic_amplitudes_mm": [1000.0 * value for value in parse_amplitudes_mm(args.amplitudes_mm)],
        "steps_per_segment": args.steps_per_segment,
        "reversal_substep_factor": args.reversal_substep_factor,
    }

    if args.dry_run:
        write_csv(
            out_dir / "protocol.csv",
            ("step", "p", "target_drift_m"),
            ({"step": p.step, "p": p.p, "target_drift_m": p.target_drift_m} for p in protocol),
        )
        write_json(out_dir / "reference_manifest.json", manifest)
        return 0

    total_start = time.perf_counter()
    ops = try_import_opensees()
    hysteresis_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    bar_rows: list[dict[str, object]] = []
    try:
        analysis_start = time.perf_counter()
        model = build_conforming_continuum_model(ops, args, spec)
        preload = run_axial_preload(ops, args, model)
        if args.lateral_control_mode == "sp-path":
            configure_lateral_sp_path(ops, model, protocol)
        else:
            ops.timeSeries("Linear", 2)
            ops.pattern("Plain", 2, 2)
            ops.load(int(model["top_nodes"][0]), 1.0, 0.0, 0.0)

        h, c, b = sample_state(ops, model, protocol[0], None)
        hysteresis_rows.append(h)
        control_rows.append(c)
        bar_rows.extend(b)

        previous = protocol[0]
        top_probe = int(model["top_nodes"][len(model["top_nodes"]) // 2])
        for point in protocol[1:]:
            if args.lateral_control_mode == "sp-path":
                control = sp_path_step(
                    ops,
                    p_start=previous.p,
                    p_end=point.p,
                    drift_start_m=previous.target_drift_m,
                    drift_end_m=point.target_drift_m,
                    top_probe_node=top_probe,
                    profile_family=args.solver_profile_family,
                    max_bisections=args.max_bisections,
                )
            else:
                control_summary = analyze_displacement_increment_summary(
                    ops,
                    top_probe,
                    1,
                    point.target_drift_m - previous.target_drift_m,
                    args.max_bisections,
                    profile_family=args.solver_profile_family,
                )
                control = {
                    "success": control_summary.success,
                    "solver_converged": control_summary.solver_converged,
                    "accepted_substep_count": control_summary.accepted_substep_count,
                    "max_bisection_level": control_summary.max_bisection_level,
                    "newton_iterations": control_summary.newton_iterations,
                    "test_norm": control_summary.test_norm,
                    "control_increment_requested": control_summary.control_increment_requested,
                    "control_increment_achieved": control_summary.control_increment_achieved,
                    "control_increment_error": control_summary.control_increment_error,
                    "solver_profile_label": control_summary.solver_profile_label,
                    "solver_algorithm_name": control_summary.solver_algorithm_name,
                }
            if not control["success"]:
                manifest["status"] = "failed"
                manifest["failure_step"] = point.step
                manifest["failure_target_drift_m"] = point.target_drift_m
                manifest["failure_reason"] = "lateral_control_failed"
                break
            h, c, b = sample_state(ops, model, point, control)
            hysteresis_rows.append(h)
            control_rows.append(c)
            bar_rows.extend(b)
            previous = point
            if args.print_progress:
                print(
                    f"OpenSees continuum step={point.step:4d} "
                    f"ux={1000.0*h['drift_m']:+.3f} mm "
                    f"V={1000.0*h['base_shear_MN']:+.3f} kN"
                )
        else:
            manifest["status"] = "completed"

        analysis_elapsed = time.perf_counter() - analysis_start
        coord_rows, mesh_bar_rows = mesh_contract_rows(model)
        write_csv(
            out_dir / "mesh_coordinates.csv",
            ("axis", "index", "value_m"),
            coord_rows,
        )
        if mesh_bar_rows:
            write_csv(
                out_dir / "bar_layout.csv",
                (
                    "element_tag",
                    "bar_index",
                    "segment_index",
                    "node_i",
                    "node_j",
                    "x",
                    "y",
                    "z_i",
                    "z_j",
                    "area_m2",
                    "material_tag",
                ),
                mesh_bar_rows,
            )
        manifest["preload_summary"] = preload
        manifest["mesh_actual"] = {
            "nx": len(model["x_coords"]) - 1,
            "ny": len(model["y_coords"]) - 1,
            "nz": len(model["z_coords"]) - 1,
            "node_count": len(model["x_coords"]) * len(model["y_coords"]) * len(model["z_coords"]),
            "solid_element_count": len(model["solid_elements"]),
            "truss_element_count": len(model["truss_elements"]),
        }
        manifest["timing"] = {
            "analysis_wall_seconds": analysis_elapsed,
            "total_wall_seconds": time.perf_counter() - total_start,
        }
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_reason"] = str(exc)
        manifest["failure_exception_type"] = type(exc).__name__
        manifest["failure_traceback"] = traceback.format_exc()

    write_csv(
        out_dir / "protocol.csv",
        ("step", "p", "target_drift_m"),
        ({"step": p.step, "p": p.p, "target_drift_m": p.target_drift_m} for p in protocol),
    )
    if hysteresis_rows:
        write_csv(out_dir / "hysteresis.csv", hysteresis_rows[0].keys(), hysteresis_rows)
    if control_rows:
        write_csv(out_dir / "control_state.csv", control_rows[0].keys(), control_rows)
    if bar_rows:
        write_csv(out_dir / "steel_bar_response.csv", bar_rows[0].keys(), bar_rows)
    manifest["hysteresis_point_count"] = len(hysteresis_rows)
    manifest["control_state_record_count"] = len(control_rows)
    manifest["steel_bar_record_count"] = len(bar_rows)
    write_json(out_dir / "reference_manifest.json", manifest)
    return 0 if manifest["status"] == "completed" else 2


def main() -> int:
    return run_analysis(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
