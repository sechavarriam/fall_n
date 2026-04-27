#!/usr/bin/env python3
"""
High-fidelity OpenSeesPy structural reference for the reduced RC-column study.

This script is intentionally narrower than the general external benchmark:

  - structural slice only
  - multiple beam-column elements along the cantilever height
  - global observables (hysteresis/control/timing) as the primary contract

It exists to support the TimoshenkoBeamN family study in fall_n, where we are
currently interested in physical coherence of the structural response more than
pointwise station equivalence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from opensees_reduced_rc_column_reference import (
    ReducedRCColumnReferenceSpec,
    analyze_displacement_increment_summary,
    build_protocol_points,
    build_section_layout_rows,
    build_station_layout_rows,
    configure_static_analysis,
    define_rc_fiber_section,
    ensure_dir,
    external_mapping_policy_catalog,
    parse_amplitudes_mm,
    safe_test_iterations,
    safe_test_norm,
    selected_mapping_policy,
    structural_convergence_profile_families,
    structural_increment_convergence_profiles,
    try_import_opensees,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-element OpenSeesPy high-fidelity RC-column reference "
            "for the reduced structural validation campaign."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis", choices=("monotonic", "cyclic"), default="cyclic")
    parser.add_argument(
        "--material-mode",
        choices=("nonlinear", "elasticized"),
        default="nonlinear",
    )
    parser.add_argument(
        "--model-dimension",
        choices=("2d", "3d"),
        default="3d",
        help=(
            "Structural hi-fi comparator dimension. The 2D slice preserves the same "
            "RC fiber section and axial preload but removes 3D frame noise for the "
            "single-direction reduced-column benchmark."
        ),
    )
    parser.add_argument(
        "--mapping-policy",
        choices=tuple(external_mapping_policy_catalog().keys()),
        default=None,
    )
    parser.add_argument(
        "--solver-profile-family",
        choices=structural_convergence_profile_families(),
        default="robust-large-amplitude",
        help=(
            "Declared OpenSees convergence-profile family for the hi-fi structural "
            "reference. The default promotes a less artificially fragile solve "
            "surface at large amplitude while step acceptance is still checked "
            "against displacement-control admissibility."
        ),
    )
    parser.add_argument(
        "--concrete-model",
        choices=("Elastic", "Concrete01", "Concrete02"),
        default=None,
    )
    parser.add_argument("--concrete-lambda", type=float, default=None)
    parser.add_argument("--concrete-ft-ratio", type=float, default=None)
    parser.add_argument("--concrete-softening-multiplier", type=float, default=None)
    parser.add_argument(
        "--concrete-unconfined-residual-ratio", type=float, default=None
    )
    parser.add_argument(
        "--concrete-confined-residual-ratio", type=float, default=None
    )
    parser.add_argument("--concrete-ultimate-strain", type=float, default=None)
    parser.add_argument("--steel-r0", type=float, default=None)
    parser.add_argument("--steel-cr1", type=float, default=None)
    parser.add_argument("--steel-cr2", type=float, default=None)
    parser.add_argument("--steel-a1", type=float, default=None)
    parser.add_argument("--steel-a2", type=float, default=None)
    parser.add_argument("--steel-a3", type=float, default=None)
    parser.add_argument("--steel-a4", type=float, default=None)
    parser.add_argument(
        "--beam-element-family",
        choices=("disp", "force"),
        default="force",
    )
    parser.add_argument(
        "--beam-integration",
        choices=("legendre", "lobatto"),
        default="lobatto",
    )
    parser.add_argument("--integration-points", type=int, default=5)
    parser.add_argument("--structural-element-count", type=int, default=12)
    parser.add_argument(
        "--element-local-iterations",
        type=int,
        default=0,
        help=(
            "Optional local compatibility iterations for forceBeamColumn. "
            "When positive, the hi-fi runner appends '-iter maxIter tol' to "
            "the OpenSees element command."
        ),
    )
    parser.add_argument(
        "--element-local-tolerance",
        type=float,
        default=1.0e-12,
        help="Tolerance used with --element-local-iterations for forceBeamColumn.",
    )
    parser.add_argument("--geom-transf", choices=("linear", "pdelta"), default="linear")
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--monotonic-tip-mm", type=float, default=2.5)
    parser.add_argument("--monotonic-steps", type=int, default=8)
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--reversal-substep-factor", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument(
        "--lateral-control-mode",
        choices=("displacement-control", "sp-path"),
        default="displacement-control",
        help=(
            "Lateral Dirichlet-control implementation. `displacement-control` "
            "uses OpenSees' DisplacementControl integrator increment-by-increment. "
            "`sp-path` prescribes the same tip displacement through a Path "
            "TimeSeries and an SP constraint, so OpenSees time remains monotone "
            "while the imposed displacement can reverse sign."
        ),
    )
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def beam_integration_name(family: str) -> str:
    return "Lobatto" if family == "lobatto" else "Legendre"


def build_model(
    ops: object,
    args: argparse.Namespace,
    spec: ReducedRCColumnReferenceSpec,
    mapping_policy,
) -> dict[str, object]:
    ops.wipe()
    is_2d = args.model_dimension == "2d"
    ops.model("basic", "-ndm", 2 if is_2d else 3, "-ndf", 3 if is_2d else 6)

    element_count = max(args.structural_element_count, 1)
    node_tags: list[int] = []
    for idx in range(element_count + 1):
        tag = idx + 1
        z = spec.column_height_m * idx / element_count
        if is_2d:
            ops.node(tag, 0.0, z)
        else:
            ops.node(tag, 0.0, 0.0, z)
        node_tags.append(tag)
    ops.fix(node_tags[0], *(1, 1, 1) if is_2d else (1, 1, 1, 1, 1, 1))

    transf_tag = 1
    sec_tag = 10
    integ_tag = 20
    geom_name = "PDelta" if args.geom_transf == "pdelta" else "Linear"
    if is_2d:
        ops.geomTransf(geom_name, transf_tag)
    else:
        ops.geomTransf(geom_name, transf_tag, 1.0, 0.0, 0.0)

    define_rc_fiber_section(
        ops,
        spec,
        sec_tag=sec_tag,
        material_mode=args.material_mode,
        mapping_policy=mapping_policy,
        include_shear_aggregator=True,
        frame_dimension=args.model_dimension,
    )
    ops.beamIntegration(
        beam_integration_name(args.beam_integration),
        integ_tag,
        sec_tag,
        max(args.integration_points, 1),
    )

    beam_element_name = "dispBeamColumn" if args.beam_element_family == "disp" else "forceBeamColumn"
    element_tags: list[int] = []
    for idx in range(element_count):
        tag = 100 + idx
        element_command: list[object] = [
            beam_element_name,
            tag,
            node_tags[idx],
            node_tags[idx + 1],
            transf_tag,
            integ_tag,
        ]
        if (
            args.beam_element_family == "force"
            and args.element_local_iterations > 0
        ):
            element_command.extend(
                (
                    "-iter",
                    int(args.element_local_iterations),
                    float(args.element_local_tolerance),
                )
            )
        ops.element(*element_command)
        element_tags.append(tag)

    local_xi = tuple(
        {
            "legendre": {
                1: (0.0,),
                2: (-0.5773502691896257, 0.5773502691896257),
                3: (-0.7745966692414834, 0.0, 0.7745966692414834),
                4: (
                    -0.8611363115940526,
                    -0.3399810435848563,
                    0.3399810435848563,
                    0.8611363115940526,
                ),
                5: (
                    -0.9061798459386640,
                    -0.5384693101056831,
                    0.0,
                    0.5384693101056831,
                    0.9061798459386640,
                ),
            },
            "lobatto": {
                2: (-1.0, 1.0),
                3: (-1.0, 0.0, 1.0),
                4: (-1.0, -0.4472135954999579, 0.4472135954999579, 1.0),
                5: (-1.0, -0.6546536707079771, 0.0, 0.6546536707079771, 1.0),
            },
        }[args.beam_integration][max(args.integration_points, 1)]
    )

    station_rows: list[dict[str, object]] = []
    section_gp = 0
    for elem_idx, element_tag in enumerate(element_tags):
        z0 = spec.column_height_m * elem_idx / element_count
        z1 = spec.column_height_m * (elem_idx + 1) / element_count
        for local_sec_idx, xi in enumerate(local_xi, start=1):
            z = z0 + 0.5 * (xi + 1.0) * (z1 - z0)
            xi_global = 2.0 * z / spec.column_height_m - 1.0
            station_rows.append(
                {
                    "section_gp": section_gp,
                    "xi": xi_global,
                    "element_index": elem_idx,
                    "element_tag": element_tag,
                    "local_section_index": local_sec_idx,
                    "z_m": z,
                }
            )
            section_gp += 1

    return {
        "base_node": node_tags[0],
        "top_node": node_tags[-1],
        "element_tags": element_tags,
        "station_rows": station_rows,
        "beam_element_name": beam_element_name,
        "model_dimension": args.model_dimension,
    }


def run_adaptive_axial_preload(
    ops: object,
    *,
    top_node: int,
    axial_compression_mn: float,
    requested_steps: int,
    solver_profile_family: str,
    model_dimension: str,
    max_refinements: int = 6,
) -> dict[str, object]:
    if axial_compression_mn <= 0.0:
        return {"status": "skipped", "accepted_steps": 0, "refinement_level": 0}

    for refinement_level in range(max_refinements + 1):
        steps = max(requested_steps, 1) * (2**refinement_level)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        if model_dimension == "2d":
            ops.load(top_node, 0.0, -axial_compression_mn * 1.0e6, 0.0)
        else:
            ops.load(top_node, 0.0, 0.0, -axial_compression_mn * 1.0e6, 0.0, 0.0, 0.0)
        configure_static_analysis(ops, profile_family=solver_profile_family)
        ops.integrator("LoadControl", 1.0 / steps)
        ops.analysis("Static")
        ok = ops.analyze(steps)
        if ok == 0:
            ops.loadConst("-time", 0.0)
            return {
                "status": "completed",
                "accepted_steps": steps,
                "refinement_level": refinement_level,
            }
        ops.wipeAnalysis()
        ops.remove("loadPattern", 1)
        ops.remove("timeSeries", 1)

    raise RuntimeError(
        "OpenSees high-fidelity preload stage failed after adaptive load-control refinements."
    )


def analyze_sp_path_increment_summary(
    ops: object,
    *,
    top_node: int,
    dof: int,
    pseudo_time_increment: float,
    target_drift_m: float,
    solver_profile_family: str,
) -> SimpleNamespace:
    """Advance one prescribed-displacement path step with load control.

    OpenSees' DisplacementControl integrator solves for an auxiliary load factor.
    Near cyclic reversals that factor can become extremely ill-conditioned for
    softening fiber sections. The SP-path route instead keeps the displacement as
    the primary Dirichlet datum and uses monotonically increasing pseudo-time only
    to evaluate the Path TimeSeries.
    """

    profile = structural_increment_convergence_profiles(solver_profile_family)[0]
    domain_time_before = float(ops.getTime())
    control_dof_before = float(ops.nodeDisp(top_node, dof))
    configure_static_analysis(ops, profile=profile)
    ops.integrator("LoadControl", pseudo_time_increment)
    ops.analysis("Static")
    solver_converged = ops.analyze(1) == 0
    domain_time_after = float(ops.getTime())
    control_dof_after = float(ops.nodeDisp(top_node, dof))
    drift_error = control_dof_after - target_drift_m
    target_reached = abs(drift_error) <= (1.0e-9 + 1.0e-6 * max(abs(target_drift_m), 1.0e-12))
    return SimpleNamespace(
        success=bool(solver_converged and target_reached),
        accepted_substep_count=1 if solver_converged and target_reached else 0,
        max_bisection_level=0,
        newton_iterations=safe_test_iterations(ops),
        test_norm=safe_test_norm(ops),
        domain_time_before=domain_time_before,
        domain_time_after=domain_time_after,
        control_dof_before=control_dof_before,
        control_dof_after=control_dof_after,
        solver_profile_label=profile.label,
        solver_test_name=profile.test_name,
        solver_algorithm_name=profile.algorithm_name,
        solver_converged=solver_converged,
        control_increment_requested=target_drift_m - control_dof_before,
        control_increment_achieved=control_dof_after - control_dof_before,
        control_increment_error=drift_error,
        control_relative_increment_error=(
            drift_error / target_drift_m if abs(target_drift_m) > 1.0e-14 else math.nan
        ),
        control_direction_admissible=True,
        control_magnitude_admissible=target_reached,
    )


def sample_base_section_response(
    ops: object,
    station_rows: list[dict[str, object]],
    step: int,
    p: float,
    drift_m: float,
    model_dimension: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    is_2d = model_dimension == "2d"
    for row in station_rows:
        element_tag = int(row["element_tag"])
        section_index = int(row["local_section_index"])
        deformation = list(ops.eleResponse(element_tag, "section", section_index, "deformation") or [])
        force = list(ops.eleResponse(element_tag, "section", section_index, "force") or [])
        moment_index = 1 if is_2d else 2
        rows.append(
            {
                "step": step,
                "p": p,
                "drift_m": drift_m,
                "section_gp": int(row["section_gp"]),
                "xi": float(row["xi"]),
                "z_m": float(row["z_m"]),
                "axial_strain": float(deformation[0]) if len(deformation) > 0 else math.nan,
                # In the 2D slice OpenSees exposes the single bending curvature as the
                # second generalized section deformation [eps, kappa_z, gamma_y]. We
                # map that scalar into the benchmark's canonical curvature_y column so
                # the hysteresis/moment-curvature contracts stay shared across 2D/3D.
                "curvature_y": (
                    float(deformation[1]) if is_2d and len(deformation) > 1
                    else float(deformation[2]) if len(deformation) > 2
                    else math.nan
                ),
                "axial_force_MN": float(force[0]) / 1.0e6 if len(force) > 0 else math.nan,
                "moment_y_MNm": (
                    float(force[moment_index]) / 1.0e6 if len(force) > moment_index else math.nan
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    ensure_dir(out_dir)

    spec = ReducedRCColumnReferenceSpec()
    mapping_policy = selected_mapping_policy(args)
    protocol = build_protocol_points(args)
    manifest = {
        "benchmark_kind": "opensees_structural_high_fidelity_reference",
        "tool": "OpenSeesPy",
        "model_dimension": args.model_dimension,
        "analysis": args.analysis,
        "material_mode": args.material_mode,
        "beam_element_family": args.beam_element_family,
        "beam_integration": args.beam_integration,
        "integration_points": args.integration_points,
        "structural_element_count": args.structural_element_count,
        "element_local_iterations": args.element_local_iterations,
        "element_local_tolerance": args.element_local_tolerance,
        "geom_transf": args.geom_transf,
        "axial_compression_mn": args.axial_compression_mn,
        "axial_preload_steps": args.axial_preload_steps,
        "cyclic_amplitudes_mm": [1.0e3 * value for value in parse_amplitudes_mm(args.amplitudes_mm)],
        "steps_per_segment": args.steps_per_segment,
        "reversal_substep_factor": args.reversal_substep_factor,
        "max_bisections": args.max_bisections,
        "lateral_control_mode": args.lateral_control_mode,
        "reference_spec": asdict(spec),
        "mapping_policy": asdict(mapping_policy),
        "solver_profile_family": args.solver_profile_family,
    }

    total_start = time.perf_counter()
    ops = try_import_opensees()
    analysis_start = time.perf_counter()
    hysteresis_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    section_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    preload_summary: dict[str, object] | None = None

    try:
        model = build_model(ops, args, spec, mapping_policy)
        base_node = int(model["base_node"])
        top_node = int(model["top_node"])
        station_rows = list(model["station_rows"])
        model_dimension = str(model["model_dimension"])

        preload_summary = run_adaptive_axial_preload(
            ops,
            top_node=top_node,
            axial_compression_mn=args.axial_compression_mn,
            requested_steps=args.axial_preload_steps,
            solver_profile_family=args.solver_profile_family,
            model_dimension=model_dimension,
        )
        if args.axial_compression_mn <= 0.0:
            ops.loadConst("-time", 0.0)

        if args.lateral_control_mode == "sp-path":
            path_args: list[object] = ["-time"]
            path_args.extend(float(point.p) for point in protocol)
            path_args.append("-values")
            path_args.extend(float(point.target_drift_m) for point in protocol)
            path_args.append("-useLast")
            ops.timeSeries("Path", 2, *path_args)
            ops.pattern("Plain", 2, 2)
            ops.sp(top_node, 1, 1.0)
        else:
            ops.timeSeries("Linear", 2)
            ops.pattern("Plain", 2, 2)
            if model_dimension == "2d":
                ops.load(top_node, 1.0, 0.0, 0.0)
            else:
                ops.load(top_node, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        def sample(point: object, stage: str, increment_summary=None) -> None:
            ops.reactions()
            drift = float(ops.nodeDisp(top_node, 1))
            top_axial = float(ops.nodeDisp(top_node, 2 if model_dimension == "2d" else 3))
            base_shear = float(ops.nodeReaction(base_node, 1)) / 1.0e6
            base_axial = float(ops.nodeReaction(base_node, 2 if model_dimension == "2d" else 3)) / 1.0e6
            hysteresis_rows.append(
                {
                    "step": point.step,
                    "p": point.p,
                    "drift_m": drift,
                    "base_shear_MN": base_shear,
                }
            )
            control_rows.append(
                {
                    "step": point.step,
                    "p": point.p,
                    "target_drift_m": point.target_drift_m,
                    "actual_tip_drift_m": drift,
                    "top_axial_displacement_m": top_axial,
                    "base_shear_MN": base_shear,
                    "base_axial_reaction_MN": base_axial,
                    "stage": stage,
                    "accepted_substep_count": (
                        increment_summary.accepted_substep_count if increment_summary else 0
                    ),
                    "max_bisection_level": (
                        increment_summary.max_bisection_level if increment_summary else 0
                    ),
                    "newton_iterations": (
                        increment_summary.newton_iterations if increment_summary else 0.0
                    ),
                    "newton_iterations_per_substep": (
                        increment_summary.newton_iterations / increment_summary.accepted_substep_count
                        if increment_summary and increment_summary.accepted_substep_count > 0
                        else 0.0
                    ),
                    "solver_profile_label": (
                        increment_summary.solver_profile_label if increment_summary else ""
                    ),
                    "solver_algorithm_name": (
                        increment_summary.solver_algorithm_name if increment_summary else ""
                    ),
                    "solver_converged": int(
                        increment_summary.solver_converged if increment_summary else True
                    ),
                }
            )
            section_rows.extend(
                sample_base_section_response(
                    ops,
                    station_rows,
                    point.step,
                    point.p,
                    drift,
                    model_dimension,
                )
            )

        sample(
            protocol[0],
            "preload_equilibrated"
            if args.axial_compression_mn > 0.0 and args.axial_preload_steps > 0
            else "lateral_branch",
            None,
        )
        previous = protocol[0]
        for point in protocol[1:]:
            delta = point.target_drift_m - previous.target_drift_m
            if abs(delta) <= 1.0e-14:
                sample(point, "lateral_branch", None)
                previous = point
                continue
            if args.lateral_control_mode == "sp-path":
                summary = analyze_sp_path_increment_summary(
                    ops,
                    top_node=top_node,
                    dof=1,
                    pseudo_time_increment=point.p - previous.p,
                    target_drift_m=point.target_drift_m,
                    solver_profile_family=args.solver_profile_family,
                )
            else:
                summary = analyze_displacement_increment_summary(
                    ops,
                    top_node,
                    1,
                    delta,
                    args.max_bisections,
                    profile_family=args.solver_profile_family,
                )
            if not summary.success:
                manifest["status"] = "failed"
                manifest["failure_step"] = point.step
                manifest["failure_target_drift_m"] = point.target_drift_m
                manifest["failure_reason"] = (
                    "OpenSees lateral Dirichlet-control stage failed before "
                    "reaching the declared protocol point."
                )
                break
            sample(point, "lateral_branch", summary)
            previous = point
        else:
            manifest["status"] = "completed"

    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure_reason"] = str(exc)
        manifest["failure_exception_type"] = type(exc).__name__
        manifest["failure_traceback"] = traceback.format_exc()

    analysis_elapsed = time.perf_counter() - analysis_start
    output_start = time.perf_counter()
    if hysteresis_rows:
        write_csv(
            out_dir / "hysteresis.csv",
            ("step", "p", "drift_m", "base_shear_MN"),
            hysteresis_rows,
        )
    if control_rows:
        write_csv(
            out_dir / "control_state.csv",
            tuple(control_rows[0].keys()),
            control_rows,
        )
    if station_rows:
        write_csv(
            out_dir / "section_station_layout.csv",
            tuple(station_rows[0].keys()),
            station_rows,
        )
    if section_rows:
        write_csv(
            out_dir / "section_response.csv",
            tuple(section_rows[0].keys()),
            section_rows,
        )
    write_csv(
        out_dir / "section_layout.csv",
        ("fiber_index", "y", "z", "area", "zone", "material_role", "material_tag"),
        build_section_layout_rows(spec),
    )
    output_elapsed = time.perf_counter() - output_start

    if preload_summary is not None:
        manifest["preload_summary"] = preload_summary
    manifest["hysteresis_point_count"] = len(hysteresis_rows)
    manifest["control_state_record_count"] = len(control_rows)
    manifest["section_record_count"] = len(section_rows)
    manifest["station_count"] = len(station_rows)
    manifest["timing"] = {
        "total_wall_seconds": time.perf_counter() - total_start,
        "analysis_wall_seconds": analysis_elapsed,
        "output_write_wall_seconds": output_elapsed,
    }
    write_json(out_dir / "reference_manifest.json", manifest)
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
