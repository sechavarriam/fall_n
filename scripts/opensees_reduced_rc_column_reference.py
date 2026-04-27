#!/usr/bin/env python3
"""
OpenSeesPy external computational reference for the reduced RC-column campaign.

This script is intentionally staged as an external *computational* benchmark,
not as a replacement for the later experimental/literature closure.

Current scope:
  - single cantilever RC column
  - one lateral direction
  - optional constant axial compression
  - monotonic or cyclic tip-displacement control
  - 3D fiber-section beam-column reference
  - CSV contracts aligned with the reduced-column validation surface in fall_n

The constitutive mapping is deliberately honest and explicit:
  - monotonic-reference: baseline Concrete02 + Steel02 bridge;
  - cyclic-diagnostic: reduced-tension Concrete02 + tuned Steel02 bridge from
    the uniaxial audit;
  - elasticized-parity: same fiber layout with elastic materials only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReducedRCColumnReferenceSpec:
    column_height_m: float = 3.2
    section_b_m: float = 0.25
    section_h_m: float = 0.25
    cover_m: float = 0.03
    longitudinal_bar_diameter_m: float = 0.016
    tie_spacing_m: float = 0.08
    concrete_fpc_mpa: float = 30.0
    concrete_nu: float = 0.20
    steel_E_mpa: float = 200_000.0
    steel_fy_mpa: float = 420.0
    steel_b: float = 0.01
    tie_fy_mpa: float = 420.0
    rho_s: float = 0.015
    kappa_y: float = 5.0 / 6.0
    kappa_z: float = 5.0 / 6.0


@dataclass(frozen=True)
class ProtocolPoint:
    step: int
    p: float
    target_drift_m: float


@dataclass(frozen=True)
class SectionRecord:
    step: int
    p: float
    drift_m: float
    section_gp: int
    xi: float
    axial_strain: float
    curvature_y: float
    curvature_z: float
    axial_force_mn: float
    moment_y_mnm: float
    moment_z_mnm: float
    tangent_ea: float
    tangent_eiy: float
    tangent_eiz: float
    tangent_eiy_direct_raw: float = math.nan
    tangent_eiz_direct_raw: float = math.nan
    raw_tangent_k00: float = math.nan
    raw_tangent_k0y: float = math.nan
    raw_tangent_ky0: float = math.nan
    raw_tangent_kyy: float = math.nan


@dataclass(frozen=True)
class ElasticSectionTargets:
    ea_n: float
    eiy_nm2: float
    eiz_nm2: float
    ga_y_n: float
    ga_z_n: float
    gj_nm2: float
    weighted_centroid_y_m: float
    weighted_centroid_z_m: float


@dataclass(frozen=True)
class StepRecord:
    step: int
    p: float
    drift_m: float
    base_shear_mn: float


@dataclass(frozen=True)
class ControlStateRecord:
    step: int
    p: float
    target_drift_m: float
    actual_tip_drift_m: float
    top_axial_displacement_m: float
    base_shear_mn: float
    base_axial_reaction_mn: float
    stage: str
    accepted_substep_count: int = 0
    max_bisection_level: int = 0
    newton_iterations: float = math.nan
    newton_iterations_per_substep: float = math.nan
    test_norm: float = math.nan
    solver_profile_label: str = ""
    solver_test_name: str = ""
    solver_algorithm_name: str = ""
    solver_converged: bool = False
    control_increment_requested: float = math.nan
    control_increment_achieved: float = math.nan
    control_increment_error: float = math.nan
    control_relative_increment_error: float = math.nan
    control_direction_admissible: bool = False
    control_magnitude_admissible: bool = False


@dataclass(frozen=True)
class SectionBaselineRecord:
    step: int
    load_factor: float
    target_axial_force_mn: float
    solved_axial_strain: float
    curvature_y: float
    curvature_z: float
    axial_force_mn: float
    moment_y_mnm: float
    moment_z_mnm: float
    tangent_ea: float
    tangent_eiy: float
    tangent_eiz: float
    tangent_eiy_direct_raw: float
    tangent_eiz_direct_raw: float
    newton_iterations: int
    final_axial_force_residual_mn: float
    raw_tangent_k00: float = math.nan
    raw_tangent_k0y: float = math.nan
    raw_tangent_ky0: float = math.nan
    raw_tangent_kyy: float = math.nan
    support_axial_reaction_mn: float = math.nan
    support_moment_y_reaction_mnm: float = math.nan
    axial_equilibrium_gap_mn: float = math.nan
    moment_equilibrium_gap_mnm: float = math.nan


@dataclass(frozen=True)
class SectionFiberStateRecord:
    step: int
    load_factor: float
    solved_axial_strain: float
    curvature_y: float
    zero_curvature_anchor: bool
    fiber_index: int
    y: float
    z: float
    area: float
    zone: str
    material_role: str
    material_tag: int
    strain_xx: float
    stress_xx_mpa: float
    tangent_xx_mpa: float
    axial_force_contribution_mn: float
    moment_y_contribution_mnm: float
    raw_tangent_k00_contribution: float
    raw_tangent_k0y_contribution: float
    raw_tangent_kyy_contribution: float
    p: float = 0.0
    drift_m: float = 0.0
    section_gp: int = 0
    xi: float = 0.0


@dataclass(frozen=True)
class SectionControlTraceRecord:
    step: int
    load_factor: float
    stage: str
    target_curvature_y: float
    actual_curvature_y: float
    delta_target_curvature_y: float
    delta_actual_curvature_y: float
    pseudo_time_before: float
    pseudo_time_after: float
    pseudo_time_increment: float
    domain_time_before: float
    domain_time_after: float
    domain_time_increment: float
    control_dof_before: float
    control_dof_after: float
    target_increment_direction: int
    actual_increment_direction: int
    protocol_branch_id: int
    reversal_index: int
    branch_step_index: int
    accepted_substep_count: int
    max_bisection_level: int
    newton_iterations: float
    newton_iterations_per_substep: float
    test_norm: float
    target_axial_force_mn: float
    actual_axial_force_mn: float
    axial_force_residual_mn: float
    solver_converged: bool = False
    control_increment_requested: float = math.nan
    control_increment_achieved: float = math.nan
    control_increment_error: float = math.nan
    control_relative_increment_error: float = math.nan
    control_direction_admissible: bool = False
    control_magnitude_admissible: bool = False


@dataclass(frozen=True)
class IncrementControlSummary:
    success: bool
    accepted_substep_count: int
    max_bisection_level: int
    newton_iterations: float
    test_norm: float
    domain_time_before: float
    domain_time_after: float
    control_dof_before: float
    control_dof_after: float
    solver_profile_label: str = ""
    solver_test_name: str = ""
    solver_algorithm_name: str = ""
    solver_converged: bool = False
    control_increment_requested: float = math.nan
    control_increment_achieved: float = math.nan
    control_increment_error: float = math.nan
    control_relative_increment_error: float = math.nan
    control_direction_admissible: bool = False
    control_magnitude_admissible: bool = False
    accepted_replay_segments: tuple["AcceptedIncrementReplaySegment", ...] = ()


@dataclass(frozen=True)
class AcceptedIncrementReplaySegment:
    delta: float
    solver_profile_label: str
    solver_test_name: str = ""
    solver_algorithm_name: str = ""


@dataclass(frozen=True)
class ControlIncrementAcceptance:
    solver_converged: bool
    accepted: bool
    requested_increment: float
    achieved_increment: float
    increment_error: float
    relative_increment_error: float
    direction_admissible: bool
    magnitude_admissible: bool


@dataclass(frozen=True)
class SectionFailureTrialState:
    section_row: SectionBaselineRecord | None
    fiber_rows: tuple[SectionFiberStateRecord, ...]
    control_trace_row: SectionControlTraceRecord


@dataclass(frozen=True)
class PartialSectionAnalysisState:
    rows: tuple[SectionBaselineRecord, ...]
    fiber_rows: tuple[SectionFiberStateRecord, ...]
    control_trace_rows: tuple[SectionControlTraceRecord, ...]
    failure_trial: SectionFailureTrialState | None = None


class OpenSeesReferenceAnalysisFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        partial_section_state: PartialSectionAnalysisState | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_section_state = partial_section_state


@dataclass(frozen=True)
class OpenSeesConvergenceProfile:
    family: str
    label: str
    test_name: str
    tolerance: float
    max_iterations: int
    algorithm_name: str
    test_print_flag: int = 0
    algorithm_args: tuple[object, ...] = ()


@dataclass(frozen=True)
class ExternalConstitutiveMappingPolicy:
    label: str
    concrete_model: str
    steel_r0: float
    steel_cr1: float
    steel_cr2: float
    steel_a1: float
    steel_a2: float
    steel_a3: float
    steel_a4: float
    concrete_lambda: float
    concrete_ft_ratio: float
    concrete_softening_multiplier: float
    concrete_unconfined_residual_ratio: float
    concrete_confined_residual_ratio: float
    concrete_ultimate_strain: float


def default_mapping_policy_label(material_mode: str, analysis: str) -> str:
    if material_mode == "elasticized":
        return "elasticized-parity"
    return "cyclic-diagnostic" if analysis == "cyclic" else "monotonic-reference"


def external_mapping_policy_catalog() -> dict[str, ExternalConstitutiveMappingPolicy]:
    return {
        "elasticized-parity": ExternalConstitutiveMappingPolicy(
            label="elasticized-parity",
            concrete_model="Elastic",
            steel_r0=20.0,
            steel_cr1=18.5,
            steel_cr2=0.15,
            steel_a1=0.0,
            steel_a2=1.0,
            steel_a3=0.0,
            steel_a4=1.0,
            concrete_lambda=0.10,
            concrete_ft_ratio=0.10,
            concrete_softening_multiplier=0.10,
            concrete_unconfined_residual_ratio=0.10,
            concrete_confined_residual_ratio=0.10,
            concrete_ultimate_strain=-0.006,
        ),
        "monotonic-reference": ExternalConstitutiveMappingPolicy(
            label="monotonic-reference",
            concrete_model="Concrete02",
            steel_r0=20.0,
            steel_cr1=18.5,
            steel_cr2=0.15,
            steel_a1=0.0,
            steel_a2=1.0,
            steel_a3=0.0,
            steel_a4=1.0,
            concrete_lambda=0.10,
            concrete_ft_ratio=0.0,
            concrete_softening_multiplier=0.10,
            concrete_unconfined_residual_ratio=0.10,
            concrete_confined_residual_ratio=0.20,
            concrete_ultimate_strain=-0.006,
        ),
        "cyclic-diagnostic": ExternalConstitutiveMappingPolicy(
            label="cyclic-diagnostic",
            concrete_model="Concrete02",
            steel_r0=30.0,
            steel_cr1=8.0,
            steel_cr2=0.30,
            steel_a1=0.0,
            steel_a2=1.0,
            steel_a3=0.0,
            steel_a4=1.0,
            concrete_lambda=0.10,
            concrete_ft_ratio=0.02,
            concrete_softening_multiplier=0.50,
            concrete_unconfined_residual_ratio=0.20,
            concrete_confined_residual_ratio=0.20,
            concrete_ultimate_strain=-0.006,
        ),
    }


def structural_profile_family_specs() -> dict[str, tuple[tuple[str, str, float, int, str, tuple[object, ...]], ...]]:
    robust_specs = (
        ("newton_unbalance", "NormUnbalance", 1.0e-6, 80, "Newton", ()),
        (
            "line_search_unbalance",
            "NormUnbalance",
            1.0e-6,
            100,
            "NewtonLineSearch",
            (0.8,),
        ),
        (
            "line_search_disp",
            "NormDispIncr",
            1.0e-8,
            120,
            "NewtonLineSearch",
            (0.8,),
        ),
        ("krylov_disp", "NormDispIncr", 1.0e-8, 140, "KrylovNewton", ()),
        ("broyden_disp", "NormDispIncr", 1.0e-8, 160, "Broyden", ()),
        ("secant_disp", "NormDispIncr", 1.0e-8, 160, "SecantNewton", ()),
        ("bfgs_disp", "NormDispIncr", 1.0e-8, 160, "BFGS", ()),
        (
            "modified_initial_energy",
            "EnergyIncr",
            1.0e-8,
            160,
            "ModifiedNewton",
            ("-initial",),
        ),
    )
    specs = {
        "strict": (
            ("newton_unbalance", "NormUnbalance", 1.0e-8, 50, "Newton", ()),
            (
                "line_search_unbalance",
                "NormUnbalance",
                1.0e-8,
                60,
                "NewtonLineSearch",
                (0.8,),
            ),
            (
                "line_search_disp",
                "NormDispIncr",
                1.0e-8,
                80,
                "NewtonLineSearch",
                (0.8,),
            ),
            ("krylov_disp", "NormDispIncr", 1.0e-8, 100, "KrylovNewton", ()),
            ("broyden_disp", "NormDispIncr", 1.0e-8, 120, "Broyden", ()),
            ("secant_disp", "NormDispIncr", 1.0e-8, 120, "SecantNewton", ()),
            ("bfgs_disp", "NormDispIncr", 1.0e-8, 120, "BFGS", ()),
            (
                "modified_initial_energy",
                "EnergyIncr",
                1.0e-9,
                120,
                "ModifiedNewton",
                ("-initial",),
            ),
        ),
        "balanced": (
            ("newton_unbalance", "NormUnbalance", 1.0e-7, 60, "Newton", ()),
            (
                "line_search_unbalance",
                "NormUnbalance",
                1.0e-7,
                80,
                "NewtonLineSearch",
                (0.8,),
            ),
            (
                "line_search_disp",
                "NormDispIncr",
                1.0e-8,
                100,
                "NewtonLineSearch",
                (0.8,),
            ),
            ("krylov_disp", "NormDispIncr", 1.0e-8, 120, "KrylovNewton", ()),
            ("broyden_disp", "NormDispIncr", 1.0e-8, 140, "Broyden", ()),
            ("secant_disp", "NormDispIncr", 1.0e-8, 140, "SecantNewton", ()),
            ("bfgs_disp", "NormDispIncr", 1.0e-8, 140, "BFGS", ()),
            (
                "modified_initial_energy",
                "EnergyIncr",
                1.0e-8,
                140,
                "ModifiedNewton",
                ("-initial",),
            ),
        ),
        "robust-large-amplitude": robust_specs,
        "pure-newton-unbalance": (robust_specs[0],),
        "pure-line-search-unbalance": (robust_specs[1],),
        "pure-line-search-disp": (robust_specs[2],),
        "pure-krylov-disp": (robust_specs[3],),
        "pure-broyden-disp": (robust_specs[4],),
        "pure-secant-disp": (robust_specs[5],),
        "pure-bfgs-disp": (robust_specs[6],),
        "pure-modified-initial-energy": (robust_specs[7],),
    }
    return specs


def structural_convergence_profile_families() -> tuple[str, ...]:
    return tuple(structural_profile_family_specs().keys())


def parse_args() -> argparse.Namespace:
    mapping_policy_labels = tuple(external_mapping_policy_catalog().keys())
    parser = argparse.ArgumentParser(
        description="Run or stage an OpenSeesPy external reference for the "
        "reduced RC-column validation campaign."
    )
    parser.add_argument(
        "--model-kind",
        choices=("structural", "section"),
        default="structural",
        help="Reference slice to run: full structural column or isolated section baseline.",
    )
    parser.add_argument(
        "--material-mode",
        choices=("nonlinear", "elasticized"),
        default="nonlinear",
        help="Material family for the reference slice: audited nonlinear mapping or elasticized parity control.",
    )
    parser.add_argument(
        "--mapping-policy",
        choices=mapping_policy_labels,
        default=None,
        help=(
            "Declared external constitutive policy. Defaults to "
            "`monotonic-reference` for nonlinear monotonic runs, "
            "`cyclic-diagnostic` for nonlinear cyclic runs, and "
            "`elasticized-parity` for elasticized control runs."
        ),
    )
    parser.add_argument(
        "--solver-profile-family",
        choices=structural_convergence_profile_families(),
        default="strict",
        help=(
            "Declared OpenSees convergence-profile family. "
            "`strict` preserves the original audit surface, "
            "`balanced` relaxes the algebraic solve mildly, and "
            "`robust-large-amplitude` is intended for large-amplitude "
            "fiber-beam structural references where displacement-control "
            "admissibility is checked explicitly after each accepted step."
        ),
    )
    parser.add_argument(
        "--concrete-model",
        choices=("Elastic", "Concrete01", "Concrete02"),
        default=None,
        help="Optional override for the external concrete constitutive family.",
    )
    parser.add_argument(
        "--concrete-lambda",
        type=float,
        default=None,
        help="Optional override for the Concrete02 unloading-slope ratio.",
    )
    parser.add_argument(
        "--concrete-ft-ratio",
        type=float,
        default=None,
        help="Optional override for the external tensile-strength ratio ft/f'c.",
    )
    parser.add_argument(
        "--concrete-softening-multiplier",
        type=float,
        default=None,
        help="Optional override for the external tension-softening span multiplier.",
    )
    parser.add_argument(
        "--concrete-unconfined-residual-ratio",
        type=float,
        default=None,
        help="Optional override for the unconfined residual-stress ratio.",
    )
    parser.add_argument(
        "--concrete-confined-residual-ratio",
        type=float,
        default=None,
        help="Optional override for the confined residual-stress ratio.",
    )
    parser.add_argument(
        "--concrete-ultimate-strain",
        type=float,
        default=None,
        help="Optional override for the crushing strain of the external bridge.",
    )
    parser.add_argument(
        "--steel-r0",
        type=float,
        default=None,
        help="Optional override for Steel02 R0.",
    )
    parser.add_argument(
        "--steel-cr1",
        type=float,
        default=None,
        help="Optional override for Steel02 cR1.",
    )
    parser.add_argument(
        "--steel-cr2",
        type=float,
        default=None,
        help="Optional override for Steel02 cR2.",
    )
    parser.add_argument(
        "--steel-a1",
        type=float,
        default=None,
        help="Optional override for Steel02 a1.",
    )
    parser.add_argument(
        "--steel-a2",
        type=float,
        default=None,
        help="Optional override for Steel02 a2.",
    )
    parser.add_argument(
        "--steel-a3",
        type=float,
        default=None,
        help="Optional override for Steel02 a3.",
    )
    parser.add_argument(
        "--steel-a4",
        type=float,
        default=None,
        help="Optional override for Steel02 a4.",
    )
    parser.add_argument(
        "--analysis",
        choices=("monotonic", "cyclic"),
        default="monotonic",
        help="Loading protocol family.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where manifest and CSV artifacts will be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifest and CSV contracts without requiring OpenSeesPy.",
    )
    parser.add_argument(
        "--beam-integration",
        choices=("legendre", "lobatto"),
        default="lobatto",
        help="Beam integration family for the OpenSees structural reference element.",
    )
    parser.add_argument(
        "--beam-element-family",
        choices=("disp", "force", "elastic-timoshenko"),
        default="disp",
        help=(
            "OpenSees beam-column formulation for the structural slice. "
            "Use `disp` as the parity anchor against fall_n's current "
            "displacement-based Timoshenko benchmark; keep `force` as a "
            "sensitivity path; `elastic-timoshenko` is an elastic-only "
            "control path with stiffness-equivalent section properties."
        ),
    )
    parser.add_argument(
        "--integration-points",
        type=int,
        default=3,
        help="Number of section stations in the OpenSees reference element.",
    )
    parser.add_argument(
        "--geom-transf",
        choices=("linear", "pdelta"),
        default="linear",
        help=(
            "Geometric transformation for the structural reference column. "
            "Use `linear` for the current small-strain parity slice; `pdelta` "
            "is kept only as a sensitivity path."
        ),
    )
    parser.add_argument(
        "--axial-compression-mn",
        type=float,
        default=0.02,
        help="Constant compressive axial load in MN.",
    )
    parser.add_argument(
        "--axial-preload-steps",
        type=int,
        default=4,
        help="Number of preload steps before lateral control starts.",
    )
    parser.add_argument(
        "--monotonic-tip-mm",
        type=float,
        default=2.5,
        help="Target monotonic tip displacement in mm.",
    )
    parser.add_argument(
        "--monotonic-steps",
        type=int,
        default=8,
        help="Number of monotonic displacement-control steps.",
    )
    parser.add_argument(
        "--amplitudes-mm",
        default="1.25,2.50",
        help="Comma-separated cyclic amplitude levels in mm.",
    )
    parser.add_argument(
        "--steps-per-segment",
        type=int,
        default=2,
        help="Base number of steps per cyclic segment.",
    )
    parser.add_argument(
        "--reversal-substep-factor",
        type=int,
        default=2,
        help="Extra substep factor used to guard cyclic reversal segments.",
    )
    parser.add_argument(
        "--max-bisections",
        type=int,
        default=8,
        help="Maximum binary cutbacks for a failed displacement increment.",
    )
    parser.add_argument(
        "--falln-hysteresis",
        type=Path,
        help="Optional fall_n hysteresis.csv to compare against the OpenSees reference.",
    )
    parser.add_argument(
        "--falln-moment-curvature",
        type=Path,
        help="Optional fall_n moment_curvature_base.csv to compare against the OpenSees reference.",
    )
    parser.add_argument(
        "--falln-section-response",
        type=Path,
        help="Optional fall_n section_response.csv to audit structural section-path parity station by station.",
    )
    parser.add_argument(
        "--falln-control-state",
        type=Path,
        help="Optional fall_n control_state.csv to audit preload and displacement-control parity point by point.",
    )
    parser.add_argument(
        "--falln-section-layout",
        type=Path,
        help="Optional fall_n section_layout.csv to audit fiber-cloud parity against the OpenSees reference.",
    )
    parser.add_argument(
        "--falln-station-layout",
        type=Path,
        help="Optional fall_n section_station_layout.csv to audit beam-integration station parity.",
    )
    parser.add_argument(
        "--falln-section-baseline",
        type=Path,
        help="Optional fall_n section_moment_curvature_baseline.csv to compare against the OpenSees section reference.",
    )
    parser.add_argument(
        "--falln-section-fiber-history",
        type=Path,
        help="Optional fall_n section_fiber_state_history.csv to compare against the OpenSees section-fiber history.",
    )
    parser.add_argument(
        "--falln-section-control-trace",
        type=Path,
        help="Optional fall_n section_control_trace.csv to compare against the OpenSees section control schedule.",
    )
    parser.add_argument(
        "--max-curvature-y",
        type=float,
        default=0.03,
        help="Target positive curvature for the isolated section benchmark.",
    )
    parser.add_argument(
        "--section-steps",
        type=int,
        default=120,
        help="Number of curvature-control steps for the isolated section benchmark.",
    )
    parser.add_argument(
        "--section-amplitudes-curvature-y",
        default="",
        help=(
            "Comma-separated positive curvature amplitudes for cyclic section "
            "benchmarks. Defaults to max-curvature-y when omitted."
        ),
    )
    return parser.parse_args()


def selected_mapping_policy(
    args: argparse.Namespace,
) -> ExternalConstitutiveMappingPolicy:
    label = args.mapping_policy or default_mapping_policy_label(
        args.material_mode, args.analysis
    )
    policy = external_mapping_policy_catalog()[label]
    overrides = {
        field_name: value
        for field_name, value in (
            ("concrete_model", args.concrete_model),
            ("concrete_lambda", args.concrete_lambda),
            ("concrete_ft_ratio", args.concrete_ft_ratio),
            (
                "concrete_softening_multiplier",
                args.concrete_softening_multiplier,
            ),
            (
                "concrete_unconfined_residual_ratio",
                args.concrete_unconfined_residual_ratio,
            ),
            (
                "concrete_confined_residual_ratio",
                args.concrete_confined_residual_ratio,
            ),
            ("concrete_ultimate_strain", args.concrete_ultimate_strain),
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
    if not overrides:
        return policy
    return replace(policy, label=f"{policy.label}+override", **overrides)


def parse_amplitudes_mm(raw: str) -> list[float]:
    return [1.0e-3 * float(token.strip()) for token in raw.split(",") if token.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def annotate_failure_manifest(
    manifest: dict[str, object], failure_message: str
) -> None:
    manifest["failure_message"] = failure_message
    section_match = re.search(
        r"reaching step=(?P<step>\d+), target_curvature=(?P<target>[+-]?\d+\.\d+e[+-]\d+)",
        failure_message,
    )
    if section_match:
        manifest["failure_step"] = int(section_match.group("step"))
        manifest["failure_target_curvature_y"] = float(section_match.group("target"))
        manifest["failure_scope"] = "section_curvature_control"
        return

    structural_match = re.search(
        r"reaching step=(?P<step>\d+), target_drift=(?P<target>[+-]?\d+\.\d+e[+-]\d+)",
        failure_message,
    )
    if structural_match:
        manifest["failure_step"] = int(structural_match.group("step"))
        manifest["failure_target_drift_m"] = float(structural_match.group("target"))
        manifest["failure_scope"] = "structural_displacement_control"


def relative_error(lhs: float, rhs: float) -> float:
    scale = max(abs(rhs), 1.0e-12)
    return abs(lhs - rhs) / scale


def cyclic_segment_count(num_levels: int) -> int:
    return 3 * num_levels


def cyclic_displacement(p: float, amplitudes_m: list[float]) -> float:
    if not amplitudes_m:
        return 0.0
    n_seg = cyclic_segment_count(len(amplitudes_m))
    t = min(max(p, 0.0), 1.0) * n_seg
    seg = min(max(int(t), 0), n_seg - 1)
    frac = t - float(seg)
    level = seg // 3
    phase = seg % 3
    amp = amplitudes_m[level]
    if phase == 0:
        return frac * amp
    if phase == 1:
        return amp * (1.0 - 2.0 * frac)
    return -amp * (1.0 - frac)


def build_protocol_points(args: argparse.Namespace) -> list[ProtocolPoint]:
    points = [ProtocolPoint(step=0, p=0.0, target_drift_m=0.0)]

    if args.analysis == "monotonic":
        steps = max(args.monotonic_steps, 1)
        target = 1.0e-3 * args.monotonic_tip_mm
        for step in range(1, steps + 1):
            p = step / steps
            points.append(
                ProtocolPoint(step=step, p=p, target_drift_m=target * p)
            )
        return points

    amplitudes_m = parse_amplitudes_mm(args.amplitudes_mm)
    segment_steps = max(args.steps_per_segment, 1) * max(args.reversal_substep_factor, 1)
    total_steps = cyclic_segment_count(len(amplitudes_m)) * segment_steps
    for step in range(1, total_steps + 1):
        p = step / total_steps
        points.append(
            ProtocolPoint(step=step, p=p, target_drift_m=cyclic_displacement(p, amplitudes_m))
        )
    return points


def build_section_protocol(args: argparse.Namespace) -> list[ProtocolPoint]:
    if args.analysis == "monotonic":
        steps = max(args.section_steps, 1)
        return [
            ProtocolPoint(
                step=step,
                p=step / steps,
                target_drift_m=(step / steps) * args.max_curvature_y,
            )
            for step in range(steps + 1)
        ]

    levels = [abs(value) for value in parse_csv_floats(args.section_amplitudes_curvature_y)]
    if not levels:
        levels = [abs(args.max_curvature_y)]
    segment_steps = max(args.steps_per_segment, 1) * max(args.reversal_substep_factor, 1)
    protocol = [ProtocolPoint(step=0, p=0.0, target_drift_m=0.0)]

    def add_branch(start: float, end: float, n_steps: int) -> None:
        base_step = len(protocol)
        for i in range(n_steps):
            protocol.append(
                ProtocolPoint(
                    step=base_step + i,
                    p=0.0,
                    target_drift_m=start + (i + 1) * (end - start) / n_steps,
                )
            )

    current = 0.0
    for level in levels:
        add_branch(current, level, segment_steps)
        add_branch(level, -level, 2 * segment_steps)
        add_branch(-level, 0.0, segment_steps)
        current = 0.0

    if len(protocol) > 1:
        denom = len(protocol) - 1
        protocol = [
            replace(point, p=idx / denom) for idx, point in enumerate(protocol)
        ]
    return protocol


def rc_section_patch_layout(
    spec: ReducedRCColumnReferenceSpec,
) -> tuple[dict[str, object], ...]:
    y_edge = 0.5 * spec.section_b_m
    z_edge = 0.5 * spec.section_h_m
    y_core = y_edge - spec.cover_m
    z_core = z_edge - spec.cover_m
    return (
        {
            "y_min": -y_edge,
            "y_max": y_edge,
            "ny": 8,
            "z_min": -z_edge,
            "z_max": -z_core,
            "nz": 2,
            "zone": "cover_bottom",
            "material_role": "unconfined_concrete",
        },
        {
            "y_min": -y_edge,
            "y_max": y_edge,
            "ny": 8,
            "z_min": z_core,
            "z_max": z_edge,
            "nz": 2,
            "zone": "cover_top",
            "material_role": "unconfined_concrete",
        },
        {
            "y_min": -y_edge,
            "y_max": -y_core,
            "ny": 2,
            "z_min": -z_core,
            "z_max": z_core,
            "nz": 4,
            "zone": "cover_left",
            "material_role": "unconfined_concrete",
        },
        {
            "y_min": y_core,
            "y_max": y_edge,
            "ny": 2,
            "z_min": -z_core,
            "z_max": z_core,
            "nz": 4,
            "zone": "cover_right",
            "material_role": "unconfined_concrete",
        },
        {
            "y_min": -y_core,
            "y_max": y_core,
            "ny": 6,
            "z_min": -z_core,
            "z_max": z_core,
            "nz": 6,
            "zone": "confined_core",
            "material_role": "confined_concrete",
        },
    )


def rc_section_rebar_layout(
    spec: ReducedRCColumnReferenceSpec,
) -> tuple[tuple[float, float], ...]:
    y_bar = 0.5 * spec.section_b_m - spec.cover_m
    z_bar = 0.5 * spec.section_h_m - spec.cover_m
    return (
        (-y_bar, -z_bar),
        (y_bar, -z_bar),
        (-y_bar, z_bar),
        (y_bar, z_bar),
        (0.0, -z_bar),
        (0.0, z_bar),
        (-y_bar, 0.0),
        (y_bar, 0.0),
    )


def write_csv(path: Path, header: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)


def build_section_layout_rows(
    spec: ReducedRCColumnReferenceSpec,
    material_tags: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    area_bar = bar_area(spec.longitudinal_bar_diameter_m)
    role_to_tag = material_tags or {
        "unconfined_concrete": 1,
        "confined_concrete": 2,
        "reinforcing_steel": 3,
    }

    layout: list[dict[str, object]] = []

    def append_patch(
        *,
        y_min: float,
        y_max: float,
        ny: int,
        z_min: float,
        z_max: float,
        nz: int,
        zone: str,
        material_role: str,
    ) -> None:
        dy = (y_max - y_min) / ny
        dz = (z_max - z_min) / nz
        area = dy * dz
        for iy in range(ny):
            for iz in range(nz):
                layout.append(
                    {
                        "fiber_index": len(layout),
                        "y": y_min + (iy + 0.5) * dy,
                        "z": z_min + (iz + 0.5) * dz,
                        "area": area,
                        "zone": zone,
                        "material_role": material_role,
                        "material_tag": role_to_tag[material_role],
                    }
                )

    for patch in rc_section_patch_layout(spec):
        append_patch(**patch)

    for y, z in rc_section_rebar_layout(spec):
        layout.append(
            {
                "fiber_index": len(layout),
                "y": y,
                "z": z,
                "area": area_bar,
                "zone": "longitudinal_steel",
                "material_role": "reinforcing_steel",
                "material_tag": role_to_tag["reinforcing_steel"],
            }
        )

    return layout


def build_station_layout_rows(xi_values: Iterable[float]) -> list[dict[str, object]]:
    return [
        {"section_gp": section_gp, "xi": xi}
        for section_gp, xi in enumerate(tuple(xi_values))
    ]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def square_matrix_width(values: list[float]) -> int:
    if not values:
        return 0
    width = int(round(math.sqrt(len(values))))
    return width if width * width == len(values) else 0


def matrix_entry_or_nan(values: list[float], row: int, col: int) -> float:
    width = square_matrix_width(values)
    if width == 0:
        return math.nan
    flat_index = row * width + col
    return float(values[flat_index]) if flat_index < len(values) else math.nan


def diag_or_nan(values: list[float], diag_index: int) -> float:
    return matrix_entry_or_nan(values, diag_index, diag_index)


def condensed_bending_tangent_or_nan(
    values: list[float], axial_index: int, bending_index: int
) -> float:
    axial_tangent = matrix_entry_or_nan(values, axial_index, axial_index)
    bending_tangent = matrix_entry_or_nan(values, bending_index, bending_index)
    coupling_left = matrix_entry_or_nan(values, bending_index, axial_index)
    coupling_right = matrix_entry_or_nan(values, axial_index, bending_index)
    if not math.isfinite(bending_tangent):
        return math.nan
    if not math.isfinite(axial_tangent) or abs(axial_tangent) <= 1.0e-12:
        return bending_tangent
    if not math.isfinite(coupling_left) or not math.isfinite(coupling_right):
        return bending_tangent
    return bending_tangent - coupling_left * coupling_right / axial_tangent


def first_finite_response(values: object) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (float, int)):
        return [float(values)]
    try:
        response = [float(value) for value in values]
    except TypeError:
        return []
    return response


def section_fiber_response(
    ops: object,
    element_tag: int,
    y: float,
    z: float,
    material_tag: int,
    response_kind: str,
) -> list[float]:
    query_variants = (
        ("section", 1, "fiber", y, z, material_tag, response_kind),
        ("section", 1, "fiber", y, z, response_kind),
        ("fiber", y, z, material_tag, response_kind),
        ("fiber", y, z, response_kind),
    )
    for variant in query_variants:
        try:
            values = first_finite_response(ops.eleResponse(element_tag, *variant))
        except Exception:
            values = []
        if values:
            return values
    return []


def section_fiber_response_at_station(
    ops: object,
    element_tag: int,
    section_index: int,
    y: float,
    z: float,
    material_tag: int,
    response_kind: str,
) -> list[float]:
    query_variants = (
        ("section", section_index, "fiber", y, z, material_tag, response_kind),
        ("section", section_index, "fiber", y, z, response_kind),
        ("section", 1, "fiber", y, z, material_tag, response_kind),
        ("section", 1, "fiber", y, z, response_kind),
        ("fiber", y, z, material_tag, response_kind),
        ("fiber", y, z, response_kind),
    )
    for variant in query_variants:
        try:
            values = first_finite_response(ops.eleResponse(element_tag, *variant))
        except Exception:
            values = []
        if values:
            return values
    return []


def spatial_parity_summary(
    falln_section_layout: Path | None,
    falln_station_layout: Path | None,
    opensees_section_layout: list[dict[str, object]],
    opensees_station_layout: list[dict[str, object]],
) -> dict[str, object]:
    summary: dict[str, object] = {
        "status": "not_requested",
        "section_layout": {},
        "station_layout": {},
    }

    if not falln_section_layout and not falln_station_layout:
        return summary

    summary["status"] = "completed"

    def sorted_section_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(
            rows,
            key=lambda row: (
                str(row["material_role"]),
                str(row["zone"]),
                round(float(row["y"]), 14),
                round(float(row["z"]), 14),
                round(float(row["area"]), 14),
            ),
        )

    if falln_section_layout:
        lhs = sorted_section_rows(read_csv_dicts(falln_section_layout))
        rhs = sorted_section_rows(opensees_section_layout)
        count_match = len(lhs) == len(rhs)
        shared = min(len(lhs), len(rhs))
        max_y = 0.0
        max_z = 0.0
        max_area = 0.0
        zone_mismatches = 0
        role_mismatches = 0
        for i in range(shared):
            max_y = max(max_y, abs(float(lhs[i]["y"]) - float(rhs[i]["y"])))
            max_z = max(max_z, abs(float(lhs[i]["z"]) - float(rhs[i]["z"])))
            max_area = max(
                max_area, abs(float(lhs[i]["area"]) - float(rhs[i]["area"]))
            )
            zone_mismatches += int(lhs[i]["zone"] != rhs[i]["zone"])
            role_mismatches += int(
                lhs[i]["material_role"] != rhs[i]["material_role"]
            )
        summary["section_layout"] = {
            "count_match": count_match,
            "falln_count": len(lhs),
            "opensees_count": len(rhs),
            "max_abs_y_error": max_y,
            "max_abs_z_error": max_z,
            "max_abs_area_error": max_area,
            "zone_mismatch_count": zone_mismatches,
            "material_role_mismatch_count": role_mismatches,
        }

    if falln_station_layout:
        lhs = sorted(
            (
                (int(row["section_gp"]), float(row["xi"]))
                for row in read_csv_dicts(falln_station_layout)
            )
        )
        rhs = sorted(
            (int(row["section_gp"]), float(row["xi"]))
            for row in opensees_station_layout
        )
        count_match = len(lhs) == len(rhs)
        shared = min(len(lhs), len(rhs))
        max_xi = 0.0
        gp_mismatches = 0
        for i in range(shared):
            gp_mismatches += int(lhs[i][0] != rhs[i][0])
            max_xi = max(max_xi, abs(lhs[i][1] - rhs[i][1]))
        summary["station_layout"] = {
            "count_match": count_match,
            "falln_count": len(lhs),
            "opensees_count": len(rhs),
            "section_gp_mismatch_count": gp_mismatches,
            "max_abs_xi_error": max_xi,
        }

    return summary


def write_structural_csv_contracts_only(out_dir: Path) -> None:
    contracts = (
        (
            "hysteresis.csv",
            ("step", "p", "drift_m", "base_shear_MN"),
        ),
        (
            "section_response.csv",
            (
                "step",
                "p",
                "drift_m",
                "section_gp",
                "xi",
                "axial_strain",
                "curvature_y",
                "curvature_z",
                "axial_force_MN",
                "moment_y_MNm",
                "moment_z_MNm",
                "tangent_ea",
                "tangent_eiy",
                "tangent_eiz",
                "tangent_eiy_direct_raw",
                "tangent_eiz_direct_raw",
                "raw_k00",
                "raw_k0y",
                "raw_ky0",
                "raw_kyy",
            ),
        ),
        (
            "moment_curvature_base.csv",
            (
                "step",
                "p",
                "drift_m",
                "section_gp",
                "xi",
                "curvature_y",
                "moment_y_MNm",
                "axial_force_MN",
                "tangent_eiy",
            ),
        ),
        (
            "control_state.csv",
            (
                "step",
                "p",
                "target_drift_m",
                "actual_tip_drift_m",
                "top_axial_displacement_m",
                "base_shear_MN",
                "base_axial_reaction_MN",
                "stage",
            ),
        ),
        (
            "section_fiber_state_history.csv",
            (
                "step",
                "p",
                "drift_m",
                "section_gp",
                "xi",
                "axial_strain",
                "curvature_y",
                "zero_curvature_anchor",
                "fiber_index",
                "y",
                "z",
                "area",
                "zone",
                "material_role",
                "material_tag",
                "strain_xx",
                "stress_xx_MPa",
                "tangent_xx_MPa",
                "axial_force_contribution_MN",
                "moment_y_contribution_MNm",
                "raw_k00_contribution",
                "raw_k0y_contribution",
                "raw_kyy_contribution",
            ),
        ),
    )
    for name, header in contracts:
        write_csv(out_dir / name, header, ())
    (out_dir / "preload_state.json").write_text(
        json.dumps({"status": "contract_only"}, indent=2), encoding="utf-8"
    )


def write_section_csv_contracts_only(out_dir: Path) -> None:
    write_csv(
        out_dir / "section_moment_curvature_baseline.csv",
        (
            "step",
            "load_factor",
            "target_axial_force_MN",
            "solved_axial_strain",
            "curvature_y",
            "curvature_z",
            "axial_force_MN",
            "moment_y_MNm",
            "moment_z_MNm",
            "tangent_ea",
            "tangent_eiy",
            "tangent_eiz",
            "tangent_eiy_direct_raw",
            "tangent_eiz_direct_raw",
            "newton_iterations",
            "final_axial_force_residual_MN",
            "support_axial_reaction_MN",
            "support_moment_y_reaction_MNm",
            "axial_equilibrium_gap_MN",
            "moment_equilibrium_gap_MNm",
        ),
        (),
    )
    write_csv(
        out_dir / "section_tangent_diagnostics.csv",
        (
            "step",
            "load_factor",
            "curvature_y",
            "moment_y_MNm",
            "zero_curvature_anchor",
            "tangent_eiy_condensed",
            "tangent_eiy_direct",
            "tangent_eiy_numerical",
            "tangent_eiy_left",
            "tangent_eiy_right",
            "tangent_consistency_rel_error",
            "raw_k00",
            "raw_k0y",
            "raw_ky0",
            "raw_kyy",
        ),
        (),
    )
    write_csv(
        out_dir / "section_fiber_state_history.csv",
        (
            "step",
            "load_factor",
            "solved_axial_strain",
            "curvature_y",
            "zero_curvature_anchor",
            "fiber_index",
            "y",
            "z",
            "area",
            "zone",
            "material_role",
            "material_tag",
            "strain_xx",
            "stress_xx_MPa",
            "tangent_xx_MPa",
            "axial_force_contribution_MN",
            "moment_y_contribution_MNm",
            "raw_k00_contribution",
            "raw_k0y_contribution",
            "raw_kyy_contribution",
        ),
        (),
    )
    write_csv(
        out_dir / "section_control_trace.csv",
        (
            "step",
            "load_factor",
            "stage",
            "target_curvature_y",
            "actual_curvature_y",
            "delta_target_curvature_y",
            "delta_actual_curvature_y",
            "pseudo_time_before",
            "pseudo_time_after",
            "pseudo_time_increment",
            "domain_time_before",
            "domain_time_after",
            "domain_time_increment",
            "control_dof_before",
            "control_dof_after",
            "target_increment_direction",
            "actual_increment_direction",
            "protocol_branch_id",
            "reversal_index",
            "branch_step_index",
            "accepted_substep_count",
            "max_bisection_level",
            "newton_iterations",
            "newton_iterations_per_substep",
            "test_norm",
            "target_axial_force_MN",
            "actual_axial_force_MN",
            "axial_force_residual_MN",
        ),
        (),
    )


def rectangular_torsion_constant(width: float, height: float) -> float:
    b_min = min(width, height)
    h_max = max(width, height)
    return (b_min**3 * h_max / 3.0) * (1.0 - 0.63 * b_min / h_max)


def bar_area(diameter: float) -> float:
    return math.pi * diameter * diameter / 4.0


def isotropic_shear_modulus(elastic_modulus: float, poisson: float) -> float:
    return elastic_modulus / (2.0 * (1.0 + poisson))


def concrete_initial_modulus(fpc_mpa: float) -> float:
    return 1000.0 * fpc_mpa


def mpa_to_pa(value_mpa: float) -> float:
    return 1.0e6 * value_mpa


def gauss_legendre_nodes_1_to_10() -> dict[int, tuple[float, ...]]:
    return {
        1: (0.0,),
        2: (-0.5773502691896258, 0.5773502691896258),
        3: (-0.7745966692414834, 0.0, 0.7745966692414834),
        4: (-0.8611363115940526, -0.33998104358485626, 0.33998104358485626, 0.8611363115940526),
        5: (-0.906179845938664, -0.5384693101056831, 0.0, 0.5384693101056831, 0.906179845938664),
        6: (-0.932469514203152, -0.6612093864662645, -0.2386191860831969, 0.2386191860831969, 0.6612093864662645, 0.932469514203152),
        7: (-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0, 0.4058451513773972, 0.7415311855993945, 0.9491079123427585),
        8: (-0.9602898564975363, -0.7966664774136267, -0.525532409916329, -0.1834346424956498, 0.1834346424956498, 0.525532409916329, 0.7966664774136267, 0.9602898564975363),
        9: (-0.9681602395076261, -0.8360311073266358, -0.6133714327005904, -0.3242534234038089, 0.0, 0.3242534234038089, 0.6133714327005904, 0.8360311073266358, 0.9681602395076261),
        10: (-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.14887433898163122, 0.14887433898163122, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717),
    }


def gauss_lobatto_nodes_1_to_10() -> dict[int, tuple[float, ...]]:
    return {
        1: (0.0,),
        2: (-1.0, 1.0),
        3: (-1.0, 0.0, 1.0),
        4: (-1.0, -0.4472135954999579, 0.4472135954999579, 1.0),
        5: (-1.0, -0.6546536707079771, 0.0, 0.6546536707079771, 1.0),
        6: (-1.0, -0.7650553239294647, -0.2852315164806451, 0.2852315164806451, 0.7650553239294647, 1.0),
        7: (-1.0, -0.8302238962785669, -0.4688487934707142, 0.0, 0.4688487934707142, 0.8302238962785669, 1.0),
        8: (-1.0, -0.8717401485096066, -0.5917001814331423, -0.20929921790247887, 0.20929921790247887, 0.5917001814331423, 0.8717401485096066, 1.0),
        9: (-1.0, -0.8997579954114601, -0.6771862795107377, -0.36311746382617816, 0.0, 0.36311746382617816, 0.6771862795107377, 0.8997579954114601, 1.0),
        10: (-1.0, -0.9195339081664588, -0.7387738651055051, -0.4779249498104445, -0.16527895766638698, 0.16527895766638698, 0.4779249498104445, 0.7387738651055051, 0.9195339081664588, 1.0),
    }


def beam_integration_nodes(family: str, n_points: int) -> tuple[float, ...]:
    tables = {
        "legendre": gauss_legendre_nodes_1_to_10(),
        "lobatto": gauss_lobatto_nodes_1_to_10(),
    }
    table = tables[family]
    if n_points not in table:
        raise ValueError(
            f"{family} integration is currently tabulated only for 1 <= N <= 10; got N={n_points}."
        )
    return table[n_points]


def make_manifest(
    args: argparse.Namespace,
    spec: ReducedRCColumnReferenceSpec,
    protocol: list[ProtocolPoint],
    mapping_policy: ExternalConstitutiveMappingPolicy,
) -> dict[str, object]:
    amplitudes_mm = [1.0e3 * amp for amp in parse_amplitudes_mm(args.amplitudes_mm)]
    return {
        "benchmark_kind": "external_computational_reference",
        "tool": "OpenSeesPy",
        "model_kind": args.model_kind,
        "analysis": args.analysis,
        "material_mode": args.material_mode,
        "beam_element_family": args.beam_element_family,
        "beam_integration": args.beam_integration,
        "integration_points": args.integration_points,
        "geom_transf": args.geom_transf,
        "axial_compression_mn": args.axial_compression_mn,
        "axial_preload_steps": args.axial_preload_steps,
        "monotonic_tip_mm": args.monotonic_tip_mm,
        "monotonic_steps": args.monotonic_steps,
        "cyclic_amplitudes_mm": amplitudes_mm,
        "steps_per_segment": args.steps_per_segment,
        "reversal_substep_factor": args.reversal_substep_factor,
        "max_bisections": args.max_bisections,
        "protocol_step_count": len(protocol) - 1,
        "section_max_curvature_y": args.max_curvature_y,
        "section_steps": args.section_steps,
        "section_amplitudes_curvature_y": parse_csv_floats(
            args.section_amplitudes_curvature_y
        ),
        "reference_spec": asdict(spec),
        "mapping_policy": asdict(mapping_policy),
        "solver_policy": {
            "profile_family": args.solver_profile_family,
            "increment_control": (
                "recursive displacement-control cutback with audited OpenSees convergence-profile cascade"
            ),
            "profiles": [
                {
                    "label": profile.label,
                    "test_name": profile.test_name,
                    "tolerance": profile.tolerance,
                    "max_iterations": profile.max_iterations,
                    "algorithm_name": profile.algorithm_name,
                    "algorithm_args": list(profile.algorithm_args),
                }
                for profile in structural_increment_convergence_profiles(
                    args.solver_profile_family
                )
            ],
        },
        "equivalence_scope": {
            "spatial_model": (
                "single 3D cantilever column, one lateral direction driven explicitly (ElasticTimoshenkoBeam stiffness-equivalent elastic control)"
                if args.model_kind == "structural"
                and args.beam_element_family == "elastic-timoshenko"
                else (
                    f"single 3D cantilever column, one lateral direction driven explicitly ({args.beam_element_family}BeamColumn reference)"
                    if args.model_kind == "structural"
                    else "single 3D zeroLengthSection slice driven by axial force plus rotation-control"
                )
            ),
            "beam_formulation_note": (
                "ElasticTimoshenkoBeam is used only as an elastic control path. Its properties are stiffness-equivalent to the audited RC section (EA, EIy, EIz, GAy, GAz, GJ), so it is useful for shear-flexible elastic parity but not for nonlinear fiber-history closure."
                if args.model_kind == "structural"
                and args.beam_element_family == "elastic-timoshenko"
                else (
                    "dispBeamColumn is the parity anchor for the current fall_n reduced-column benchmark because both sides are displacement-based beam slices, but it remains an OpenSees displacement-based beam-column element with weak equilibrium, not a literal clone of fall_n's mixed-interpolation TimoshenkoBeamN."
                    if args.model_kind == "structural"
                    and args.beam_element_family == "disp"
                    else (
                        "forceBeamColumn is kept only as a sensitivity path; it is not the primary structural parity slice against the current fall_n beam benchmark."
                        if args.model_kind == "structural"
                        else "section-level slice uses zeroLengthSection and bypasses beam-formulation differences."
                    )
                )
            ),
            "beam_theory_note": (
                "fall_n uses a mixed-interpolation Timoshenko beam with explicit shear strains. This OpenSees slice uses ElasticTimoshenkoBeam with stiffness-equivalent section targets condensed from the same audited RC fiber layout, so it is a fair elastic shear-flexible control but not a nonlinear fiber comparator."
                if args.model_kind == "structural"
                and args.beam_element_family == "elastic-timoshenko"
                else (
                    "fall_n uses a mixed-interpolation Timoshenko beam with explicit shear strains. OpenSees does provide an ElasticTimoshenkoBeam element for elastic shear-flexible controls, but the audited nonlinear structural bridge uses dispBeamColumn or forceBeamColumn with a fiber section plus an uncoupled SectionAggregator(Vy,Vz); this is a close comparator for the reduced-column benchmark, but not the explicit ElasticTimoshenkoBeam family."
                    if args.model_kind == "structural"
                    else "section-level slice bypasses beam-theory differences and audits the condensed fiber-section response directly."
                )
            ),
            "section_kinematics_note": (
                "The structural slice wraps the audited Fiber section in a SectionAggregator with elastic Vy/Vz branches derived from the same kappa·G·A rule used by fall_n's Timoshenko fiber section. The aggregator is uncoupled across added DOFs, so this parity slice matches fiber cloud and section stations but not every internal formulation detail."
                if args.model_kind == "structural"
                else "The section-level slice bypasses beam-level shear interpolation and compares the condensed fiber section directly."
            ),
            "tangent_policy_note": (
                "Primary tangent comparisons use the axial-force-condensed section tangent dMy/dkappa_y|N=const; raw Kyy and the axial-flexural coupling terms K00/K0y/Ky0 are exported separately for audit."
            ),
            "station_observable_contract": (
                structural_station_observable_contract(args.beam_element_family)
                if args.model_kind == "structural"
                else {
                    "global_hysteresis": "not_applicable",
                    "base_side_moment_curvature": "normative_primary_benchmark_observable",
                    "support_axial_reaction": "normative_primary_benchmark_observable",
                    "tip_axial_shortening": "diagnostic_control_observable",
                    "station_section_path": "normative_section_benchmark_observable",
                    "station_section_axial_force": "normative_under_N_const_contract",
                    "station_section_tangent": "normative_under_N_const_contract",
                    "interpretation_note": (
                        "The zeroLengthSection benchmark is declared directly on "
                        "the condensed section slice, so section force/tangent "
                        "observables are normative rather than formulation-side "
                        "diagnostics."
                    ),
                }
            ),
            "solver_policy_note": (
                "Large-amplitude displacement-control steps are audited under a declared convergence-profile cascade (Newton / NewtonLineSearch / NormDispIncr+Krylov / ModifiedNewton-energy) plus recursive bisection, so the external bridge is not artificially limited by a single OpenSees solve profile."
            ),
            "axis_mapping": "global X drift -> local z translation -> local My response, matching the fall_n moment_y observable by construction",
            "fiber_history_convention": (
                "native_falln_parity"
                if args.model_kind == "section"
                else "native_opensees_structural"
            ),
            "constitutive_mapping": {
                "concrete": (
                    "Elastic uniaxial parity control over the same fiber layout"
                    if args.material_mode == "elasticized"
                    else (
                        "Concrete02 monotonic-reference bridge for external section/column comparison"
                        if mapping_policy.label == "monotonic-reference"
                        else (
                            "Reduced-tension Concrete02 cyclic-diagnostic bridge used to localize external cyclic concrete mismatch"
                            if mapping_policy.label == "cyclic-diagnostic"
                            else f"{mapping_policy.concrete_model} external comparable mapping"
                        )
                    )
                ),
                "steel": (
                    "Elastic uniaxial parity control over the same fiber layout"
                    if args.material_mode == "elasticized"
                    else (
                        "Steel02 baseline Menegotto-Pinto-family bridge"
                        if mapping_policy.label == "monotonic-reference"
                        else "Steel02 tuned cyclic-diagnostic bridge from the constitutive mapping audit"
                    )
                ),
            },
            "validation_role": (
                "external computational reference before later experimental/literature closure; "
                "not a substitute for physical validation."
            ),
        },
    }


def structural_station_observable_contract(
    beam_element_family: str,
) -> dict[str, str]:
    if beam_element_family == "disp":
        return {
            "global_hysteresis": "normative_primary_benchmark_observable",
            "base_side_moment_curvature": "normative_primary_benchmark_observable",
            "support_axial_reaction": "normative_primary_benchmark_observable",
            "tip_axial_shortening": "normative_secondary_diagnostic",
            "station_section_path": (
                "diagnostic_localizer_only_for_distributed_response_drift"
            ),
            "station_section_axial_force": (
                "diagnostic_only_family_dependent_under_weak_equilibrium"
            ),
            "station_section_tangent": (
                "diagnostic_only_family_dependent_under_weak_equilibrium"
            ),
            "interpretation_note": (
                "dispBeamColumn is the declared nonlinear parity anchor, but its "
                "station-wise section resultants are formulation-dependent weak-"
                "equilibrium diagnostics rather than strong-equilibrium acceptance "
                "gates."
            ),
        }
    if beam_element_family == "force":
        return {
            "global_hysteresis": "normative_primary_benchmark_observable",
            "base_side_moment_curvature": "normative_primary_benchmark_observable",
            "support_axial_reaction": "normative_primary_benchmark_observable",
            "tip_axial_shortening": "normative_secondary_diagnostic",
            "station_section_path": "diagnostic_localizer_plus_equilibrium_sensitivity",
            "station_section_axial_force": (
                "family_control_with_stronger_station_equilibrium"
            ),
            "station_section_tangent": (
                "family_control_with_stronger_station_equilibrium"
            ),
            "interpretation_note": (
                "forceBeamColumn is kept as an equilibrium-richer sensitivity path. "
                "Its station-wise section resultants are more meaningful as local "
                "equilibrium controls, but it is still not the primary nonlinear "
                "parity anchor for the current fall_n beam slice."
            ),
        }
    if beam_element_family == "elastic-timoshenko":
        return {
            "global_hysteresis": "elastic_control_only",
            "base_side_moment_curvature": "elastic_control_only",
            "support_axial_reaction": "elastic_control_only",
            "tip_axial_shortening": "elastic_control_only",
            "station_section_path": "elastic_stiffness_equivalent_control",
            "station_section_axial_force": "elastic_stiffness_equivalent_control",
            "station_section_tangent": "elastic_stiffness_equivalent_control",
            "interpretation_note": (
                "ElasticTimoshenkoBeam is a stiffness-equivalent elastic control. "
                "Its station-wise section response is useful to verify beam-theory "
                "parity, but it is not a nonlinear fiber-history benchmark."
            ),
        }
    return {
        "global_hysteresis": "unspecified",
        "base_side_moment_curvature": "unspecified",
        "support_axial_reaction": "unspecified",
        "tip_axial_shortening": "unspecified",
        "station_section_path": "unspecified",
        "station_section_axial_force": "unspecified",
        "station_section_tangent": "unspecified",
        "interpretation_note": "No structural station-observable contract declared.",
    }


def try_import_opensees() -> object:
    try:
        import openseespy.opensees as ops  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenSeesPy is not installed in the active Python environment. "
            "Use `pip install openseespy` or rerun this script with --dry-run "
            "to stage only the manifest and CSV contracts."
        ) from exc
    except RuntimeError as exc:
        raise RuntimeError(
            "OpenSeesPy is installed but could not be loaded by the active "
            "Python runtime. On Windows this usually means a wheel/runtime "
            "mismatch. The audited local runtime for this script is currently "
            "Python 3.12 with OpenSeesPy 3.8.x. Rerun with that runtime or "
            "use --dry-run to stage only the manifest and CSV contracts."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "OpenSeesPy is present but one of its transitive native "
            "dependencies could not be loaded. On Windows this often points "
            "to a Python-version mismatch or a missing DLL in the OpenSeesPy "
            "wheel. Rerun with the audited Python 3.12 runtime or use "
            "--dry-run to stage the external-reference contract only."
        ) from exc
    return ops


def mander_confined_strength(fpc_mpa: float, rho_s: float, fyh_mpa: float) -> float:
    lateral_pressure = rho_s * fyh_mpa
    ratio = max(lateral_pressure / max(fpc_mpa, 1.0e-12), 0.0)
    factor = 2.254 * math.sqrt(1.0 + 7.94 * ratio) - 2.0 * ratio - 1.254
    return max(fpc_mpa * factor, fpc_mpa)


def define_rc_fiber_section(
    ops: object,
    spec: ReducedRCColumnReferenceSpec,
    *,
    sec_tag: int,
    material_mode: str,
    mapping_policy: ExternalConstitutiveMappingPolicy,
    include_shear_aggregator: bool = False,
    frame_dimension: str = "3d",
) -> int:
    ec_pa = mpa_to_pa(concrete_initial_modulus(spec.concrete_fpc_mpa))
    gc_pa = isotropic_shear_modulus(ec_pa, spec.concrete_nu)
    gj_pa_m4 = gc_pa * rectangular_torsion_constant(spec.section_b_m, spec.section_h_m)
    area_m2 = spec.section_b_m * spec.section_h_m

    unconfined_tag = 1
    confined_tag = 2
    steel_tag = 3
    vy_tag = 4
    vz_tag = 5
    fiber_sec_tag = sec_tag if not include_shear_aggregator else sec_tag + 1000

    if material_mode == "elasticized":
        ops.uniaxialMaterial("Elastic", unconfined_tag, ec_pa)
        ops.uniaxialMaterial("Elastic", confined_tag, ec_pa)
        ops.uniaxialMaterial("Elastic", steel_tag, mpa_to_pa(spec.steel_E_mpa))
    else:
        cover_fpc = -mpa_to_pa(abs(spec.concrete_fpc_mpa))
        cover_epsc0 = -0.002
        cover_fpcu = -0.10 * mpa_to_pa(abs(spec.concrete_fpc_mpa))
        cover_epsu = -0.006

        confined_fpc_abs = mander_confined_strength(
            spec.concrete_fpc_mpa, spec.rho_s, spec.tie_fy_mpa
        )
        confined_fpc = -mpa_to_pa(confined_fpc_abs)
        confined_epsc0 = -0.002 * (
            1.0 + 5.0 * max(confined_fpc_abs / spec.concrete_fpc_mpa - 1.0, 0.0)
        )
        confined_fpcu = -0.20 * mpa_to_pa(confined_fpc_abs)
        confined_epsu = min(-0.010, 5.0 * confined_epsc0)

        ft_pa = mpa_to_pa(mapping_policy.concrete_ft_ratio * spec.concrete_fpc_mpa)
        ec_mpa = concrete_initial_modulus(spec.concrete_fpc_mpa)
        cracking_strain = (
            mapping_policy.concrete_ft_ratio * spec.concrete_fpc_mpa
        ) / max(ec_mpa, 1.0e-12)
        softening_span = max(
            mapping_policy.concrete_softening_multiplier * cracking_strain,
            1.0e-5,
        )
        ets_pa = mpa_to_pa(
            (mapping_policy.concrete_ft_ratio * spec.concrete_fpc_mpa)
            / softening_span
        )

        define_concrete = (
            lambda tag, fpc, epsc0, fpcu, epsu: ops.uniaxialMaterial(
                "Concrete01",
                tag,
                fpc,
                epsc0,
                fpcu,
                epsu,
            )
            if mapping_policy.concrete_model == "Concrete01"
            else ops.uniaxialMaterial(
                "Concrete02",
                tag,
                fpc,
                epsc0,
                fpcu,
                epsu,
                mapping_policy.concrete_lambda,
                ft_pa,
                ets_pa,
            )
        )
        define_concrete(
            unconfined_tag,
            cover_fpc,
            cover_epsc0,
            -mapping_policy.concrete_unconfined_residual_ratio
            * mpa_to_pa(abs(spec.concrete_fpc_mpa)),
            mapping_policy.concrete_ultimate_strain,
        )
        define_concrete(
            confined_tag,
            confined_fpc,
            confined_epsc0,
            -mapping_policy.concrete_confined_residual_ratio
            * mpa_to_pa(confined_fpc_abs),
            min(mapping_policy.concrete_ultimate_strain, 5.0 * confined_epsc0),
        )
        ops.uniaxialMaterial(
            "Steel02",
            steel_tag,
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

    area_bar = bar_area(spec.longitudinal_bar_diameter_m)
    patches = rc_section_patch_layout(spec)
    rebars = rc_section_rebar_layout(spec)

    if frame_dimension not in {"2d", "3d"}:
        raise ValueError(
            f"Unsupported OpenSees frame dimension for RC fiber section: {frame_dimension!r}"
        )

    if frame_dimension == "3d":
        ops.section("Fiber", fiber_sec_tag, "-GJ", gj_pa_m4)
    else:
        ops.section("Fiber", fiber_sec_tag)
    material_tag = {
        "unconfined_concrete": unconfined_tag,
        "confined_concrete": confined_tag,
    }
    for patch in patches:
        ops.patch(
            "rect",
            material_tag[patch["material_role"]],
            patch["ny"],
            patch["nz"],
            patch["y_min"],
            patch["z_min"],
            patch["y_max"],
            patch["z_max"],
        )

    bottom_face = [rebars[i] for i in (0, 4, 1)]
    top_face = [rebars[i] for i in (2, 5, 3)]
    left_mid = rebars[6]
    right_mid = rebars[7]
    ops.layer(
        "straight",
        steel_tag,
        len(bottom_face),
        area_bar,
        bottom_face[0][0],
        bottom_face[0][1],
        bottom_face[-1][0],
        bottom_face[-1][1],
    )
    ops.layer(
        "straight",
        steel_tag,
        len(top_face),
        area_bar,
        top_face[0][0],
        top_face[0][1],
        top_face[-1][0],
        top_face[-1][1],
    )
    ops.layer("straight", steel_tag, 1, area_bar, left_mid[0], left_mid[1], left_mid[0], left_mid[1])
    ops.layer(
        "straight", steel_tag, 1, area_bar, right_mid[0], right_mid[1], right_mid[0], right_mid[1]
    )

    if include_shear_aggregator:
        ops.uniaxialMaterial("Elastic", vy_tag, spec.kappa_y * gc_pa * area_m2)
        if frame_dimension == "3d":
            ops.uniaxialMaterial("Elastic", vz_tag, spec.kappa_z * gc_pa * area_m2)
            ops.section(
                "Aggregator",
                sec_tag,
                vy_tag,
                "Vy",
                vz_tag,
                "Vz",
                "-section",
                fiber_sec_tag,
            )
        else:
            ops.section(
                "Aggregator",
                sec_tag,
                vy_tag,
                "Vy",
                "-section",
                fiber_sec_tag,
            )

    return sec_tag


def build_structural_model(
    ops: object,
    args: argparse.Namespace,
    spec: ReducedRCColumnReferenceSpec,
    mapping_policy: ExternalConstitutiveMappingPolicy,
) -> dict[str, object]:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    base_node = 1
    top_node = 2
    ele_tag = 1
    transf_tag = 1
    sec_tag = 10
    integ_tag = 20

    ops.node(base_node, 0.0, 0.0, 0.0)
    ops.node(top_node, 0.0, 0.0, spec.column_height_m)
    ops.fix(base_node, 1, 1, 1, 1, 1, 1)

    geom_name = "PDelta" if args.geom_transf == "pdelta" else "Linear"
    # vecxz = global X, so global-X drift maps to local-z bending and therefore
    # to the local My resultants/deformations queried below.
    ops.geomTransf(geom_name, transf_tag, 1.0, 0.0, 0.0)
    elastic_targets = None
    if args.beam_element_family == "elastic-timoshenko":
        if args.material_mode != "elasticized":
            raise RuntimeError(
                "ElasticTimoshenkoBeam is only admitted as an elasticized control path in the reduced RC-column benchmark."
            )
        elastic_targets = calibrated_elasticized_section_targets(
            spec, args.falln_section_response
        )
        ops.element(
            "ElasticTimoshenkoBeam",
            ele_tag,
            base_node,
            top_node,
            1.0,
            1.0,
            elastic_targets.ea_n,
            elastic_targets.gj_nm2,
            elastic_targets.eiy_nm2,
            elastic_targets.eiz_nm2,
            elastic_targets.ga_y_n,
            elastic_targets.ga_z_n,
            transf_tag,
        )
        beam_element_name = "ElasticTimoshenkoBeam"
    else:
        define_rc_fiber_section(
            ops,
            spec,
            sec_tag=sec_tag,
            material_mode=args.material_mode,
            mapping_policy=mapping_policy,
            include_shear_aggregator=True,
        )

        ops.beamIntegration(
            args.beam_integration.capitalize(), integ_tag, sec_tag, args.integration_points
        )
        beam_element_name = (
            "dispBeamColumn" if args.beam_element_family == "disp" else "forceBeamColumn"
        )
        ops.element(beam_element_name, ele_tag, base_node, top_node, transf_tag, integ_tag)

    return {
        "base_node": base_node,
        "top_node": top_node,
        "element_tag": ele_tag,
        "beam_element_name": beam_element_name,
        "integration_xi": beam_integration_nodes(args.beam_integration, args.integration_points),
        "beam_element_family": args.beam_element_family,
        "elastic_section_targets": elastic_targets,
    }


def build_section_model(
    ops: object,
    spec: ReducedRCColumnReferenceSpec,
    *,
    material_mode: str,
    mapping_policy: ExternalConstitutiveMappingPolicy,
) -> dict[str, object]:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    base_node = 1
    top_node = 2
    ele_tag = 1
    sec_tag = 10

    ops.node(base_node, 0.0, 0.0, 0.0)
    ops.node(top_node, 0.0, 0.0, 0.0)
    ops.fix(base_node, 1, 1, 1, 1, 1, 1)
    # Release only axial translation (global Z) and rotation about global Y.
    #
    # Parity note:
    #   We want the section slice to share the same physical bending sign
    #   convention as the structural cantilever benchmark, where positive
    #   global-X drift induces positive local-z translation and therefore a
    #   positive My response. With local x aligned to global Z, choosing
    #   local y = -global Y makes the section local z axis coincide with
    #   +global X through the right-hand rule:
    #
    #       z_local = x_local × y_local = Z × (-Y) = +X
    #
    #   That keeps the zeroLengthSection generalized deformation/resultant
    #   order [P, Mz, My, ...] aligned with the structural slice instead of
    #   requiring a post-hoc z/zone flip in the section benchmark itself.
    ops.fix(top_node, 1, 1, 0, 1, 0, 1)

    define_rc_fiber_section(
        ops,
        spec,
        sec_tag=sec_tag,
        material_mode=material_mode,
        mapping_policy=mapping_policy,
        include_shear_aggregator=False,
    )
    ops.element(
        "zeroLengthSection",
        ele_tag,
        base_node,
        top_node,
        sec_tag,
        "-orient",
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
        0.0,
    )

    return {
        "base_node": base_node,
        "top_node": top_node,
        "element_tag": ele_tag,
        "integration_xi": (0.0,),
    }


def structural_increment_convergence_profiles(
    family: str = "strict",
) -> tuple[OpenSeesConvergenceProfile, ...]:
    profile_specs = structural_profile_family_specs()
    if family not in profile_specs:
        raise ValueError(
            f"Unknown OpenSees convergence-profile family '{family}'."
        )
    return tuple(
        OpenSeesConvergenceProfile(
            family=family,
            label=f"{family}:{label}",
            test_name=test_name,
            tolerance=tolerance,
            max_iterations=max_iterations,
            algorithm_name=algorithm_name,
            algorithm_args=algorithm_args,
        )
        for label, test_name, tolerance, max_iterations, algorithm_name, algorithm_args in profile_specs[
            family
        ]
    )


def configure_static_analysis(
    ops: object,
    *,
    profile: OpenSeesConvergenceProfile | None = None,
    algorithm: str = "Newton",
    profile_family: str = "strict",
) -> None:
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    if profile is None:
        base_profile = structural_increment_convergence_profiles(profile_family)[0]
        profile = OpenSeesConvergenceProfile(
            family=profile_family,
            label=f"{profile_family}:legacy_{algorithm.lower()}",
            test_name=base_profile.test_name,
            tolerance=base_profile.tolerance,
            max_iterations=base_profile.max_iterations,
            algorithm_name=algorithm,
            algorithm_args=(0.8,) if algorithm == "NewtonLineSearch" else (),
        )
    ops.test(
        profile.test_name,
        profile.tolerance,
        profile.max_iterations,
        profile.test_print_flag,
    )
    ops.algorithm(profile.algorithm_name, *profile.algorithm_args)


def safe_test_iterations(ops: object) -> float:
    try:
        return float(ops.testIter())
    except Exception:
        return math.nan


def safe_test_norm(ops: object) -> float:
    try:
        values = ops.testNorm()
    except Exception:
        return math.nan
    if values is None:
        return math.nan
    if isinstance(values, (list, tuple)):
        finite = [float(value) for value in values if math.isfinite(float(value))]
        return finite[-1] if finite else math.nan
    value = float(values)
    return value if math.isfinite(value) else math.nan


def signum(value: float, *, tol: float = 1.0e-14) -> int:
    return 1 if value > tol else (-1 if value < -tol else 0)


def assess_displacement_control_acceptance(
    *,
    solver_converged: bool,
    requested_increment: float,
    control_dof_before: float,
    control_dof_after: float,
    direction_tol: float = 1.0e-14,
    increment_abs_tol: float = 1.0e-10,
    increment_rel_tol: float = 1.0e-3,
) -> ControlIncrementAcceptance:
    achieved_increment = control_dof_after - control_dof_before
    increment_error = achieved_increment - requested_increment
    scale = max(abs(requested_increment), abs(achieved_increment), 1.0e-12)
    relative_increment_error = abs(increment_error) / scale

    if not (
        math.isfinite(control_dof_before)
        and math.isfinite(control_dof_after)
        and math.isfinite(requested_increment)
        and math.isfinite(achieved_increment)
    ):
        return ControlIncrementAcceptance(
            solver_converged=solver_converged,
            accepted=False,
            requested_increment=requested_increment,
            achieved_increment=achieved_increment,
            increment_error=increment_error,
            relative_increment_error=math.nan,
            direction_admissible=False,
            magnitude_admissible=False,
        )

    requested_direction = signum(requested_increment, tol=direction_tol)
    achieved_direction = signum(achieved_increment, tol=direction_tol)
    if requested_direction == 0:
        direction_admissible = achieved_direction == 0
    else:
        direction_admissible = achieved_direction == requested_direction

    magnitude_admissible = abs(increment_error) <= (
        increment_abs_tol + increment_rel_tol * max(abs(requested_increment), 1.0e-12)
    )
    accepted = solver_converged and direction_admissible and magnitude_admissible
    return ControlIncrementAcceptance(
        solver_converged=solver_converged,
        accepted=accepted,
        requested_increment=requested_increment,
        achieved_increment=achieved_increment,
        increment_error=increment_error,
        relative_increment_error=relative_increment_error,
        direction_admissible=direction_admissible,
        magnitude_admissible=magnitude_admissible,
    )


def single_profile_displacement_increment_summary(
    ops: object,
    node_tag: int,
    dof: int,
    delta: float,
    profile: OpenSeesConvergenceProfile,
) -> IncrementControlSummary:
    domain_time_before = float(ops.getTime())
    control_dof_before = float(ops.nodeDisp(node_tag, dof))
    configure_static_analysis(ops, profile=profile)
    ops.integrator("DisplacementControl", node_tag, dof, delta)
    ops.analysis("Static")
    solver_converged = ops.analyze(1) == 0
    domain_time_after = float(ops.getTime())
    control_dof_after = float(ops.nodeDisp(node_tag, dof))
    acceptance = assess_displacement_control_acceptance(
        solver_converged=solver_converged,
        requested_increment=delta,
        control_dof_before=control_dof_before,
        control_dof_after=control_dof_after,
    )
    return IncrementControlSummary(
        success=acceptance.accepted,
        accepted_substep_count=1 if acceptance.accepted else 0,
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
        solver_converged=acceptance.solver_converged,
        control_increment_requested=acceptance.requested_increment,
        control_increment_achieved=acceptance.achieved_increment,
        control_increment_error=acceptance.increment_error,
        control_relative_increment_error=acceptance.relative_increment_error,
        control_direction_admissible=acceptance.direction_admissible,
        control_magnitude_admissible=acceptance.magnitude_admissible,
        accepted_replay_segments=(
            (
                AcceptedIncrementReplaySegment(
                    delta=delta,
                    solver_profile_label=profile.label,
                    solver_test_name=profile.test_name,
                    solver_algorithm_name=profile.algorithm_name,
                ),
            )
            if acceptance.accepted
            else ()
        ),
    )


def convergence_profile_by_label(
    label: str,
) -> OpenSeesConvergenceProfile | None:
    profiles = {
        profile.label: profile
        for family in structural_convergence_profile_families()
        for profile in structural_increment_convergence_profiles(family)
    }
    return profiles.get(label)


def analyze_displacement_increment_summary(
    ops: object,
    node_tag: int,
    dof: int,
    delta: float,
    max_bisections: int,
    profile_family: str = "strict",
) -> IncrementControlSummary:
    profiles = structural_increment_convergence_profiles(profile_family)

    def recurse(
        incr: float,
        depth: int,
    ) -> IncrementControlSummary:
        for profile in profiles:
            summary = single_profile_displacement_increment_summary(
                ops, node_tag, dof, incr, profile
            )
            if summary.success:
                return replace(summary, max_bisection_level=depth)

        if depth >= max(max_bisections, 0):
            return IncrementControlSummary(
                success=False,
                accepted_substep_count=0,
                max_bisection_level=depth,
                newton_iterations=math.nan,
                test_norm=math.nan,
                domain_time_before=math.nan,
                domain_time_after=math.nan,
                control_dof_before=math.nan,
                control_dof_after=math.nan,
                control_increment_requested=incr,
            )

        half_increment = 0.5 * incr
        first = recurse(half_increment, depth + 1)
        if not first.success:
            return first
        second = recurse(half_increment, depth + 1)
        if not second.success:
            return second
        acceptance = assess_displacement_control_acceptance(
            solver_converged=first.solver_converged and second.solver_converged,
            requested_increment=incr,
            control_dof_before=first.control_dof_before,
            control_dof_after=second.control_dof_after,
        )
        accepted_substep_count = (
            first.accepted_substep_count + second.accepted_substep_count
            if acceptance.accepted
            else 0
        )
        return IncrementControlSummary(
            success=acceptance.accepted,
            accepted_substep_count=accepted_substep_count,
            max_bisection_level=max(
                first.max_bisection_level, second.max_bisection_level
            ),
            newton_iterations=(
                first.newton_iterations + second.newton_iterations
                if math.isfinite(first.newton_iterations)
                and math.isfinite(second.newton_iterations)
                else math.nan
            ),
            test_norm=second.test_norm,
            domain_time_before=first.domain_time_before,
            domain_time_after=second.domain_time_after,
            control_dof_before=first.control_dof_before,
            control_dof_after=second.control_dof_after,
            solver_profile_label=second.solver_profile_label,
            solver_test_name=second.solver_test_name,
            solver_algorithm_name=second.solver_algorithm_name,
            solver_converged=acceptance.solver_converged,
            control_increment_requested=acceptance.requested_increment,
            control_increment_achieved=acceptance.achieved_increment,
            control_increment_error=acceptance.increment_error,
            control_relative_increment_error=acceptance.relative_increment_error,
            control_direction_admissible=acceptance.direction_admissible,
            control_magnitude_admissible=acceptance.magnitude_admissible,
            accepted_replay_segments=(
                first.accepted_replay_segments + second.accepted_replay_segments
                if acceptance.accepted
                else ()
            ),
        )

    return recurse(delta, 0)


def analyze_displacement_increment(
    ops: object,
    node_tag: int,
    dof: int,
    delta: float,
    max_bisections: int,
) -> bool:
    return analyze_displacement_increment_summary(
        ops, node_tag, dof, delta, max_bisections
    ).success


def sample_section_records(
    ops: object,
    element_tag: int,
    xi_values: tuple[float, ...],
    step_point: ProtocolPoint,
) -> list[SectionRecord]:
    rows: list[SectionRecord] = []

    def scalar_or_nan(values: list[float], index: int) -> float:
        return float(values[index]) if index < len(values) else math.nan

    for sec_idx, xi in enumerate(xi_values, start=1):
        deformation = list(ops.eleResponse(element_tag, "section", sec_idx, "deformation") or [])
        force = list(ops.eleResponse(element_tag, "section", sec_idx, "force") or [])
        stiffness = list(ops.eleResponse(element_tag, "section", sec_idx, "stiffness") or [])

        rows.append(
            SectionRecord(
                step=step_point.step,
                p=step_point.p,
                drift_m=step_point.target_drift_m,
                section_gp=sec_idx - 1,
                xi=xi,
                axial_strain=scalar_or_nan(deformation, 0),
                curvature_y=scalar_or_nan(deformation, 2),
                curvature_z=scalar_or_nan(deformation, 1),
                axial_force_mn=scalar_or_nan(force, 0) / 1.0e6,
                moment_y_mnm=scalar_or_nan(force, 2) / 1.0e6,
                moment_z_mnm=scalar_or_nan(force, 1) / 1.0e6,
                tangent_ea=diag_or_nan(stiffness, 0) / 1.0e6,
                tangent_eiy=condensed_bending_tangent_or_nan(stiffness, 0, 2) / 1.0e6,
                tangent_eiz=condensed_bending_tangent_or_nan(stiffness, 0, 1) / 1.0e6,
                tangent_eiy_direct_raw=matrix_entry_or_nan(stiffness, 2, 2) / 1.0e6,
                tangent_eiz_direct_raw=matrix_entry_or_nan(stiffness, 1, 1) / 1.0e6,
                raw_tangent_k00=matrix_entry_or_nan(stiffness, 0, 0) / 1.0e6,
                raw_tangent_k0y=matrix_entry_or_nan(stiffness, 0, 2) / 1.0e6,
                raw_tangent_ky0=matrix_entry_or_nan(stiffness, 2, 0) / 1.0e6,
                raw_tangent_kyy=matrix_entry_or_nan(stiffness, 2, 2) / 1.0e6,
            )
        )

    return rows


def synthesize_elastic_timoshenko_section_records(
    *,
    step_point: ProtocolPoint,
    xi_values: tuple[float, ...],
    column_height_m: float,
    base_shear_mn: float,
    base_axial_reaction_mn: float,
    elastic_targets: ElasticSectionTargets,
) -> list[SectionRecord]:
    rows: list[SectionRecord] = []
    for sec_idx, xi in enumerate(xi_values, start=1):
        x_from_base = 0.5 * (xi + 1.0) * column_height_m
        moment_y_mnm = base_shear_mn * max(column_height_m - x_from_base, 0.0)
        axial_force_n = -base_axial_reaction_mn * 1.0e6
        moment_y_nm = moment_y_mnm * 1.0e6
        curvature_y = (
            moment_y_nm / elastic_targets.eiy_nm2
            if abs(elastic_targets.eiy_nm2) > 1.0e-18
            else math.nan
        )
        axial_strain = (
            axial_force_n / elastic_targets.ea_n
            if abs(elastic_targets.ea_n) > 1.0e-18
            else math.nan
        )
        rows.append(
            SectionRecord(
                step=step_point.step,
                p=step_point.p,
                drift_m=step_point.target_drift_m,
                section_gp=sec_idx - 1,
                xi=xi,
                axial_strain=axial_strain,
                curvature_y=curvature_y,
                curvature_z=0.0,
                axial_force_mn=-base_axial_reaction_mn,
                moment_y_mnm=moment_y_mnm,
                moment_z_mnm=0.0,
                tangent_ea=elastic_targets.ea_n / 1.0e6,
                tangent_eiy=elastic_targets.eiy_nm2 / 1.0e6,
                tangent_eiz=elastic_targets.eiz_nm2 / 1.0e6,
                tangent_eiy_direct_raw=elastic_targets.eiy_nm2 / 1.0e6,
                tangent_eiz_direct_raw=elastic_targets.eiz_nm2 / 1.0e6,
                raw_tangent_k00=elastic_targets.ea_n / 1.0e6,
                raw_tangent_k0y=0.0,
                raw_tangent_ky0=0.0,
                raw_tangent_kyy=elastic_targets.eiy_nm2 / 1.0e6,
            )
        )
    return rows


def elasticized_section_targets(
    spec: ReducedRCColumnReferenceSpec,
) -> ElasticSectionTargets:
    section_layout_rows = build_section_layout_rows(spec)
    ec_pa = mpa_to_pa(concrete_initial_modulus(spec.concrete_fpc_mpa))
    gc_pa = isotropic_shear_modulus(ec_pa, spec.concrete_nu)
    area_m2 = spec.section_b_m * spec.section_h_m

    def elastic_modulus_pa(material_role: str) -> float:
        return (
            mpa_to_pa(spec.steel_E_mpa)
            if material_role == "steel"
            else ec_pa
        )

    ea_n = 0.0
    e_weighted_y = 0.0
    e_weighted_z = 0.0
    for row in section_layout_rows:
        modulus = elastic_modulus_pa(str(row["material_role"]))
        area = float(row["area"])
        y = float(row["y"])
        z = float(row["z"])
        stiffness_weight = modulus * area
        ea_n += stiffness_weight
        e_weighted_y += stiffness_weight * y
        e_weighted_z += stiffness_weight * z

    centroid_y = e_weighted_y / ea_n if abs(ea_n) > 1.0e-18 else 0.0
    centroid_z = e_weighted_z / ea_n if abs(ea_n) > 1.0e-18 else 0.0

    eiy_nm2 = 0.0
    eiz_nm2 = 0.0
    for row in section_layout_rows:
        modulus = elastic_modulus_pa(str(row["material_role"]))
        area = float(row["area"])
        y = float(row["y"]) - centroid_y
        z = float(row["z"]) - centroid_z
        eiy_nm2 += modulus * area * z * z
        eiz_nm2 += modulus * area * y * y

    return ElasticSectionTargets(
        ea_n=ea_n,
        eiy_nm2=eiy_nm2,
        eiz_nm2=eiz_nm2,
        ga_y_n=spec.kappa_y * gc_pa * area_m2,
        ga_z_n=spec.kappa_z * gc_pa * area_m2,
        gj_nm2=gc_pa * rectangular_torsion_constant(spec.section_b_m, spec.section_h_m),
        weighted_centroid_y_m=centroid_y,
        weighted_centroid_z_m=centroid_z,
    )


def calibrated_elasticized_section_targets(
    spec: ReducedRCColumnReferenceSpec,
    falln_section_response: Path | None,
) -> ElasticSectionTargets:
    fallback = elasticized_section_targets(spec)
    if not falln_section_response or not falln_section_response.exists():
        return fallback

    with falln_section_response.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    preload_rows = [row for row in rows if int(row["step"]) == 0]
    if not preload_rows:
        return fallback

    mean_or = lambda key, default: (
        sum(float(row[key]) for row in preload_rows) / len(preload_rows)
        if preload_rows
        else default
    )
    ea_n = abs(mean_or("raw_k00", fallback.ea_n / 1.0e6)) * 1.0e6
    eiy_nm2 = abs(mean_or("raw_kyy", fallback.eiy_nm2 / 1.0e6)) * 1.0e6
    eiz_nm2 = abs(
        mean_or(
            "tangent_eiz_direct_raw",
            mean_or("tangent_eiz", fallback.eiz_nm2 / 1.0e6),
        )
    ) * 1.0e6
    return replace(
        fallback,
        ea_n=ea_n,
        eiy_nm2=eiy_nm2,
        eiz_nm2=eiz_nm2,
    )


def numerical_tangent(values_x: list[float], values_y: list[float], index: int) -> float:
    if not values_x or len(values_x) != len(values_y):
        return math.nan

    def slope(i0: int, i1: int) -> float:
        dx = values_x[i1] - values_x[i0]
        if abs(dx) <= 1.0e-14:
            return math.nan
        return (values_y[i1] - values_y[i0]) / dx

    if index <= 0:
        return slope(0, min(1, len(values_x) - 1))
    if index >= len(values_x) - 1:
        return slope(max(len(values_x) - 2, 0), len(values_x) - 1)
    left = slope(index - 1, index)
    right = slope(index, index + 1)
    if math.isfinite(left) and math.isfinite(right):
        return 0.5 * (left + right)
    return left if math.isfinite(left) else right


def numerical_tangent_pair(
    values_x: list[float], values_y: list[float], index: int
) -> tuple[float, float, float]:
    if not values_x or len(values_x) != len(values_y):
        return (math.nan, math.nan, math.nan)

    def slope(i0: int, i1: int) -> float:
        dx = values_x[i1] - values_x[i0]
        if abs(dx) <= 1.0e-14:
            return math.nan
        return (values_y[i1] - values_y[i0]) / dx

    left = math.nan if index <= 0 else slope(index - 1, index)
    right = math.nan if index >= len(values_x) - 1 else slope(index, index + 1)
    if math.isfinite(left) and math.isfinite(right):
        numerical = 0.5 * (left + right)
    else:
        numerical = left if math.isfinite(left) else right
    return numerical, left, right


def make_base_side_history(rows: list[SectionRecord]) -> list[dict[str, object]]:
    if not rows:
        return []

    controlling_gp = min(rows, key=lambda row: (row.xi, row.section_gp)).section_gp
    base_rows = [row for row in rows if row.section_gp == controlling_gp]
    kappas = [row.curvature_y for row in base_rows]
    moments = [row.moment_y_mnm for row in base_rows]

    derived: list[dict[str, object]] = []
    for idx, row in enumerate(base_rows):
        tangent = row.tangent_eiy
        if not math.isfinite(tangent):
            tangent = numerical_tangent(kappas, moments, idx)
        derived.append(
            {
                "step": row.step,
                "p": row.p,
                "drift_m": row.drift_m,
                "section_gp": row.section_gp,
                "xi": row.xi,
                "curvature_y": row.curvature_y,
                "moment_y_MNm": row.moment_y_mnm,
                "axial_force_MN": row.axial_force_mn,
                "tangent_eiy": tangent,
            }
        )
    return derived


def run_analysis(
    ops: object,
    args: argparse.Namespace,
    spec: ReducedRCColumnReferenceSpec,
    protocol: list[ProtocolPoint],
    mapping_policy: ExternalConstitutiveMappingPolicy,
) -> tuple[
    list[StepRecord],
    list[SectionRecord],
    list[ControlStateRecord],
    list[SectionFiberStateRecord],
]:
    model = build_structural_model(ops, args, spec, mapping_policy)
    base_node = int(model["base_node"])
    top_node = int(model["top_node"])
    ele_tag = int(model["element_tag"])
    xi_values = tuple(float(x) for x in model["integration_xi"])
    beam_element_family = str(model.get("beam_element_family", args.beam_element_family))
    elastic_targets = model.get("elastic_section_targets")
    section_layout_rows = build_section_layout_rows(spec)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    if args.axial_compression_mn > 0.0:
        ops.load(top_node, 0.0, 0.0, -args.axial_compression_mn * 1.0e6, 0.0, 0.0, 0.0)
        configure_static_analysis(ops, profile_family=args.solver_profile_family)
        ops.integrator("LoadControl", 1.0 / max(args.axial_preload_steps, 1))
        ops.analysis("Static")
        ok = ops.analyze(max(args.axial_preload_steps, 1))
        if ok != 0:
            raise RuntimeError("OpenSees axial preload stage failed before lateral control started.")
        ops.loadConst("-time", 0.0)
    else:
        ops.loadConst("-time", 0.0)

    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(top_node, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    steps: list[StepRecord] = []
    sections: list[SectionRecord] = []
    control_states: list[ControlStateRecord] = []
    fiber_rows: list[SectionFiberStateRecord] = []
    last_increment_summary = IncrementControlSummary(
        success=True,
        accepted_substep_count=0,
        max_bisection_level=0,
        newton_iterations=math.nan,
        test_norm=math.nan,
        domain_time_before=float(ops.getTime()),
        domain_time_after=float(ops.getTime()),
        control_dof_before=float(ops.nodeDisp(top_node, 1)),
        control_dof_after=float(ops.nodeDisp(top_node, 1)),
    )

    def sample_structural_section_fiber_records(
        point: ProtocolPoint,
        section_rows_for_step: list[SectionRecord],
    ) -> list[SectionFiberStateRecord]:
        records: list[SectionFiberStateRecord] = []
        for section_row in section_rows_for_step:
            section_index = int(section_row.section_gp) + 1
            for row in section_layout_rows:
                y = float(row["y"])
                z = float(row["z"])
                query_z = -z
                area = float(row["area"])
                material_tag = int(row["material_tag"])
                stress_strain = section_fiber_response_at_station(
                    ops,
                    ele_tag,
                    section_index,
                    y,
                    query_z,
                    material_tag,
                    "stressStrain",
                )
                if len(stress_strain) >= 2:
                    stress_xx_mpa = float(stress_strain[0]) / 1.0e6
                    strain_xx = float(stress_strain[1])
                else:
                    stress_values = section_fiber_response_at_station(
                        ops,
                        ele_tag,
                        section_index,
                        y,
                        query_z,
                        material_tag,
                        "stress",
                    )
                    strain_values = section_fiber_response_at_station(
                        ops,
                        ele_tag,
                        section_index,
                        y,
                        query_z,
                        material_tag,
                        "strain",
                    )
                    stress_xx_mpa = (
                        float(stress_values[0]) / 1.0e6
                        if len(stress_values) >= 1
                        else math.nan
                    )
                    strain_xx = (
                        float(strain_values[0]) if len(strain_values) >= 1 else math.nan
                    )

                tangent_values = section_fiber_response_at_station(
                    ops,
                    ele_tag,
                    section_index,
                    y,
                    query_z,
                    material_tag,
                    "tangent",
                )
                tangent_xx_mpa = (
                    float(tangent_values[0]) / 1.0e6
                    if len(tangent_values) >= 1
                    else math.nan
                )

                axial_force_contribution = (
                    stress_xx_mpa * area if math.isfinite(stress_xx_mpa) else math.nan
                )
                moment_y_contribution = (
                    -stress_xx_mpa * z * area
                    if math.isfinite(stress_xx_mpa)
                    else math.nan
                )
                raw_tangent_k00_contribution = (
                    tangent_xx_mpa * area
                    if math.isfinite(tangent_xx_mpa)
                    else math.nan
                )
                raw_tangent_k0y_contribution = (
                    -tangent_xx_mpa * z * area
                    if math.isfinite(tangent_xx_mpa)
                    else math.nan
                )
                raw_tangent_kyy_contribution = (
                    tangent_xx_mpa * z * z * area
                    if math.isfinite(tangent_xx_mpa)
                    else math.nan
                )

                records.append(
                    SectionFiberStateRecord(
                        step=point.step,
                        load_factor=point.p,
                        solved_axial_strain=section_row.axial_strain,
                        curvature_y=section_row.curvature_y,
                        zero_curvature_anchor=abs(section_row.curvature_y) <= 1.0e-12,
                        fiber_index=int(row["fiber_index"]),
                        y=y,
                        z=z,
                        area=area,
                        zone=str(row["zone"]),
                        material_role=str(row["material_role"]),
                        material_tag=material_tag,
                        strain_xx=strain_xx,
                        stress_xx_mpa=stress_xx_mpa,
                        tangent_xx_mpa=tangent_xx_mpa,
                        axial_force_contribution_mn=axial_force_contribution,
                        moment_y_contribution_mnm=moment_y_contribution,
                        raw_tangent_k00_contribution=raw_tangent_k00_contribution,
                        raw_tangent_k0y_contribution=raw_tangent_k0y_contribution,
                        raw_tangent_kyy_contribution=raw_tangent_kyy_contribution,
                        p=point.p,
                        drift_m=point.target_drift_m,
                        section_gp=section_row.section_gp,
                        xi=section_row.xi,
                    )
                )
        return records

    def sample(point: ProtocolPoint, stage: str) -> None:
        ops.reactions()
        drift = float(ops.nodeDisp(top_node, 1))
        top_axial_displacement = float(ops.nodeDisp(top_node, 3))
        base_shear = float(ops.nodeReaction(base_node, 1)) / 1.0e6
        base_axial_reaction = float(ops.nodeReaction(base_node, 3)) / 1.0e6
        steps.append(
            StepRecord(
                step=point.step,
                p=point.p,
                drift_m=drift,
                # Keep the global base-shear observable on the same structural
                # sign convention used by fall_n's resisting-force extraction:
                # positive imposed drift in +X yields a negative resisting
                # base shear on the cantilever slice.
                base_shear_mn=base_shear,
            )
        )
        control_states.append(
            ControlStateRecord(
                step=point.step,
                p=point.p,
                target_drift_m=point.target_drift_m,
                actual_tip_drift_m=drift,
                top_axial_displacement_m=top_axial_displacement,
                base_shear_mn=base_shear,
                base_axial_reaction_mn=base_axial_reaction,
                stage=stage,
                accepted_substep_count=last_increment_summary.accepted_substep_count,
                max_bisection_level=last_increment_summary.max_bisection_level,
                newton_iterations=last_increment_summary.newton_iterations,
                newton_iterations_per_substep=(
                    last_increment_summary.newton_iterations
                    / last_increment_summary.accepted_substep_count
                    if last_increment_summary.accepted_substep_count > 0
                    else math.nan
                ),
                test_norm=last_increment_summary.test_norm,
                solver_profile_label=last_increment_summary.solver_profile_label,
                solver_test_name=last_increment_summary.solver_test_name,
                solver_algorithm_name=last_increment_summary.solver_algorithm_name,
                solver_converged=last_increment_summary.solver_converged,
                control_increment_requested=last_increment_summary.control_increment_requested,
                control_increment_achieved=last_increment_summary.control_increment_achieved,
                control_increment_error=last_increment_summary.control_increment_error,
                control_relative_increment_error=(
                    last_increment_summary.control_relative_increment_error
                ),
                control_direction_admissible=(
                    last_increment_summary.control_direction_admissible
                ),
                control_magnitude_admissible=(
                    last_increment_summary.control_magnitude_admissible
                ),
            )
        )
        if beam_element_family == "elastic-timoshenko" and elastic_targets is not None:
            sections.extend(
                synthesize_elastic_timoshenko_section_records(
                    step_point=point,
                    xi_values=xi_values,
                    column_height_m=spec.column_height_m,
                    base_shear_mn=base_shear,
                    base_axial_reaction_mn=base_axial_reaction,
                    elastic_targets=elastic_targets,
                )
            )
        else:
            step_section_rows = sample_section_records(ops, ele_tag, xi_values, point)
            sections.extend(step_section_rows)
            fiber_rows.extend(
                sample_structural_section_fiber_records(point, step_section_rows)
            )

    sample(
        protocol[0],
        "preload_equilibrated"
        if args.axial_compression_mn > 0.0 and args.axial_preload_steps > 0
        else "lateral_branch",
    )
    previous = protocol[0]
    for point in protocol[1:]:
        delta = point.target_drift_m - previous.target_drift_m
        if abs(delta) <= 1.0e-14:
            sample(point, "lateral_branch")
            previous = point
            continue
        last_increment_summary = analyze_displacement_increment_summary(
            ops,
            top_node,
            1,
            delta,
            args.max_bisections,
            profile_family=args.solver_profile_family,
        )
        if not last_increment_summary.success:
            raise RuntimeError(
                "OpenSees displacement-control stage failed before reaching the "
                f"declared protocol point step={point.step}, target_drift={point.target_drift_m:+.6e} m."
            )
        sample(point, "lateral_branch")
        previous = point

    return steps, sections, control_states, fiber_rows


def sample_zero_length_section_record(
    ops: object,
    base_node: int,
    element_tag: int,
    point: ProtocolPoint,
    target_axial_force_mn: float,
    newton_iterations: int,
) -> SectionBaselineRecord:
    deformation = list(ops.eleResponse(element_tag, "section", 1, "deformation") or [])
    force = list(ops.eleResponse(element_tag, "section", 1, "force") or [])
    stiffness = list(ops.eleResponse(element_tag, "section", 1, "stiffness") or [])

    scalar_or_nan = lambda values, index: float(values[index]) if index < len(values) else math.nan

    # The parity slice uses the section generalized response as the canonical
    # source for [P, Mz, My] / [eps, kappa_z, kappa_y], but the chosen local
    # orientation still requires the same sign reconciliation against the
    # fall_n convention used elsewhere in this benchmark.
    axial_force_n = scalar_or_nan(force, 0)
    moment_y_nm = scalar_or_nan(force, 2)
    mapped_curvature_y = -scalar_or_nan(deformation, 2)
    raw_k0y = -matrix_entry_or_nan(stiffness, 0, 2) / 1.0e6
    raw_ky0 = -matrix_entry_or_nan(stiffness, 2, 0) / 1.0e6
    raw_kyy = matrix_entry_or_nan(stiffness, 2, 2) / 1.0e6
    raw_k00 = matrix_entry_or_nan(stiffness, 0, 0) / 1.0e6
    ops.reactions()
    support_axial_reaction_mn = -float(ops.nodeReaction(base_node, 3)) / 1.0e6
    support_moment_y_reaction_mnm = -float(ops.nodeReaction(base_node, 5)) / 1.0e6
    mapped_moment_y_mnm = -moment_y_nm / 1.0e6
    mapped_axial_force_mn = axial_force_n / 1.0e6

    def condensed_from_transformed_entries(
        k00: float,
        k0y_value: float,
        ky0_value: float,
        kyy_value: float,
    ) -> float:
        if not math.isfinite(k00) or abs(k00) <= 1.0e-12:
            return kyy_value
        return kyy_value - k0y_value * ky0_value / k00

    return SectionBaselineRecord(
        step=point.step,
        load_factor=point.p,
        target_axial_force_mn=target_axial_force_mn,
        solved_axial_strain=scalar_or_nan(deformation, 0),
        curvature_y=mapped_curvature_y,
        curvature_z=scalar_or_nan(deformation, 1),
        axial_force_mn=mapped_axial_force_mn,
        moment_y_mnm=mapped_moment_y_mnm,
        moment_z_mnm=0.0,
        tangent_ea=diag_or_nan(stiffness, 0) / 1.0e6,
        tangent_eiy=condensed_from_transformed_entries(
            raw_k00, raw_k0y, raw_ky0, raw_kyy
        ),
        tangent_eiz=condensed_bending_tangent_or_nan(stiffness, 0, 1) / 1.0e6,
        tangent_eiy_direct_raw=raw_kyy,
        tangent_eiz_direct_raw=matrix_entry_or_nan(stiffness, 1, 1) / 1.0e6,
        newton_iterations=newton_iterations,
        final_axial_force_residual_mn=mapped_axial_force_mn - target_axial_force_mn,
        raw_tangent_k00=raw_k00,
        raw_tangent_k0y=raw_k0y,
        raw_tangent_ky0=raw_ky0,
        raw_tangent_kyy=raw_kyy,
        support_axial_reaction_mn=support_axial_reaction_mn,
        support_moment_y_reaction_mnm=support_moment_y_reaction_mnm,
        axial_equilibrium_gap_mn=mapped_axial_force_mn - support_axial_reaction_mn,
        moment_equilibrium_gap_mnm=mapped_moment_y_mnm - support_moment_y_reaction_mnm,
    )


def sample_zero_length_section_fiber_records(
    ops: object,
    element_tag: int,
    point: ProtocolPoint,
    section_row: SectionBaselineRecord,
    section_layout_rows: list[dict[str, object]],
) -> list[SectionFiberStateRecord]:
    records: list[SectionFiberStateRecord] = []

    for row in section_layout_rows:
        y = float(row["y"])
        z = float(row["z"])
        area = float(row["area"])
        material_tag = int(row["material_tag"])
        stress_strain = section_fiber_response(
            ops, element_tag, y, z, material_tag, "stressStrain"
        )
        if len(stress_strain) >= 2:
            stress_xx_mpa = float(stress_strain[0]) / 1.0e6
            strain_xx = float(stress_strain[1])
        else:
            stress_values = section_fiber_response(
                ops, element_tag, y, z, material_tag, "stress"
            )
            strain_values = section_fiber_response(
                ops, element_tag, y, z, material_tag, "strain"
            )
            stress_xx_mpa = (
                float(stress_values[0]) / 1.0e6 if len(stress_values) >= 1 else math.nan
            )
            strain_xx = float(strain_values[0]) if len(strain_values) >= 1 else math.nan

        tangent_values = section_fiber_response(
            ops, element_tag, y, z, material_tag, "tangent"
        )
        tangent_xx_mpa = (
            float(tangent_values[0]) / 1.0e6 if len(tangent_values) >= 1 else math.nan
        )

        axial_force_contribution = stress_xx_mpa * area if math.isfinite(stress_xx_mpa) else math.nan
        moment_y_contribution = (
            -stress_xx_mpa * z * area if math.isfinite(stress_xx_mpa) else math.nan
        )
        raw_tangent_k00_contribution = (
            tangent_xx_mpa * area if math.isfinite(tangent_xx_mpa) else math.nan
        )
        raw_tangent_k0y_contribution = (
            -tangent_xx_mpa * z * area if math.isfinite(tangent_xx_mpa) else math.nan
        )
        raw_tangent_kyy_contribution = (
            tangent_xx_mpa * z * z * area if math.isfinite(tangent_xx_mpa) else math.nan
        )

        records.append(
            SectionFiberStateRecord(
                step=point.step,
                load_factor=point.p,
                solved_axial_strain=section_row.solved_axial_strain,
                curvature_y=section_row.curvature_y,
                zero_curvature_anchor=abs(section_row.curvature_y) <= 1.0e-12,
                fiber_index=int(row["fiber_index"]),
                y=y,
                z=z,
                area=area,
                zone=str(row["zone"]),
                material_role=str(row["material_role"]),
                material_tag=material_tag,
                strain_xx=strain_xx,
                stress_xx_mpa=stress_xx_mpa,
                tangent_xx_mpa=tangent_xx_mpa,
                axial_force_contribution_mn=axial_force_contribution,
                moment_y_contribution_mnm=moment_y_contribution,
                raw_tangent_k00_contribution=raw_tangent_k00_contribution,
                raw_tangent_k0y_contribution=raw_tangent_k0y_contribution,
                raw_tangent_kyy_contribution=raw_tangent_kyy_contribution,
            )
        )

    return records


def run_section_analysis(
    ops: object,
    args: argparse.Namespace,
    spec: ReducedRCColumnReferenceSpec,
    mapping_policy: ExternalConstitutiveMappingPolicy,
) -> tuple[
    list[SectionBaselineRecord],
    list[SectionFiberStateRecord],
    list[SectionControlTraceRecord],
]:
    section_layout_rows = build_section_layout_rows(spec)
    target_axial_force_mn = -abs(args.axial_compression_mn)
    axial_force_admissibility_tolerance_mn = max(
        1.0e-6, 1.0e-4 * max(abs(target_axial_force_mn), 1.0e-3)
    )

    def initialize_section_state() -> dict[str, object]:
        model = build_section_model(
            ops,
            spec,
            material_mode=args.material_mode,
            mapping_policy=mapping_policy,
        )
        top_node = int(model["top_node"])
        if args.axial_compression_mn > 0.0:
            ops.timeSeries("Linear", 1)
            ops.pattern("Plain", 1, 1)
            ops.load(
                top_node,
                0.0,
                0.0,
                target_axial_force_mn * 1.0e6,
                0.0,
                0.0,
                0.0,
            )
            configure_static_analysis(ops, profile_family=args.solver_profile_family)
            ops.integrator("LoadControl", 1.0 / max(args.axial_preload_steps, 1))
            ops.analysis("Static")
            if ops.analyze(max(args.axial_preload_steps, 1)) != 0:
                raise RuntimeError(
                    "OpenSees zeroLengthSection preload stage failed before curvature control started."
                )
            ops.loadConst("-time", 0.0)
        else:
            ops.loadConst("-time", 0.0)

        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        ops.load(top_node, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        return model

    def replay_accepted_section_history(
        accepted_targets: list[float],
        accepted_summaries: list[IncrementControlSummary],
    ) -> dict[str, object] | None:
        model = initialize_section_state()
        top_node = int(model["top_node"])
        previous_target = 0.0
        for target, summary in zip(accepted_targets, accepted_summaries, strict=True):
            delta = target - previous_target
            if abs(delta) <= 1.0e-14:
                previous_target = target
                continue
            if summary.accepted_replay_segments:
                replay_delta = 0.0
                for segment in summary.accepted_replay_segments:
                    profile = convergence_profile_by_label(segment.solver_profile_label)
                    if profile is None:
                        return None
                    replay_summary = single_profile_displacement_increment_summary(
                        ops, top_node, 5, segment.delta, profile
                    )
                    if not replay_summary.success:
                        return None
                    replay_delta += segment.delta
                # Reconstructing an accepted path from replay segments is the
                # real state-preserving contract. The segment sum is only an
                # audit sanity check and should not veto a path that has just
                # been replayed successfully because floating-point cutback
                # trees can accumulate machine-noise at the subsegment level.
                if abs(replay_delta - delta) > 1.0e-9 * max(abs(delta), 1.0):
                    return None
            elif summary.accepted_substep_count == 1 and summary.solver_profile_label:
                profile = convergence_profile_by_label(summary.solver_profile_label)
                if profile is None:
                    return None
                replay_summary = single_profile_displacement_increment_summary(
                    ops, top_node, 5, delta, profile
                )
                if not replay_summary.success:
                    return None
            else:
                return None
            previous_target = target
        return model

    def analyze_section_increment_from_clean_history(
        accepted_targets: list[float],
        accepted_summaries: list[IncrementControlSummary],
        delta: float,
        max_bisections: int,
    ) -> tuple[IncrementControlSummary, dict[str, object] | None]:
        profiles = structural_increment_convergence_profiles(
            args.solver_profile_family
        )

        def recurse(
            history_targets: list[float],
            history_summaries: list[IncrementControlSummary],
            local_delta: float,
            depth: int,
        ) -> tuple[IncrementControlSummary, dict[str, object] | None]:
            current_target = history_targets[-1] if history_targets else 0.0
            trial_target = current_target + local_delta
            for profile in profiles:
                model = replay_accepted_section_history(history_targets, history_summaries)
                if model is None:
                    break
                candidate_base_node = int(model["base_node"])
                top_node = int(model["top_node"])
                candidate_element_tag = int(model["element_tag"])
                summary = single_profile_displacement_increment_summary(
                    ops, top_node, 5, local_delta, profile
                )
                if summary.success:
                    candidate_row = sample_zero_length_section_record(
                        ops,
                        candidate_base_node,
                        candidate_element_tag,
                        ProtocolPoint(step=-1, p=math.nan, target_drift_m=trial_target),
                        target_axial_force_mn,
                        int(summary.newton_iterations)
                        if math.isfinite(summary.newton_iterations)
                        else 0,
                    )
                    if (
                        abs(candidate_row.final_axial_force_residual_mn)
                        <= axial_force_admissibility_tolerance_mn
                    ):
                        return replace(summary, max_bisection_level=depth), model

            if depth >= max(max_bisections, 0):
                return (
                    IncrementControlSummary(
                        success=False,
                        accepted_substep_count=0,
                        max_bisection_level=depth,
                        newton_iterations=math.nan,
                        test_norm=math.nan,
                        domain_time_before=math.nan,
                        domain_time_after=math.nan,
                        control_dof_before=math.nan,
                        control_dof_after=math.nan,
                        control_increment_requested=local_delta,
                    ),
                    None,
                )

            half_delta = 0.5 * local_delta
            first_summary, _ = recurse(
                history_targets, history_summaries, half_delta, depth + 1
            )
            if not first_summary.success:
                return first_summary, None

            current_target = history_targets[-1] if history_targets else 0.0
            mid_target = current_target + half_delta
            second_summary, second_model = recurse(
                [*history_targets, mid_target],
                [*history_summaries, first_summary],
                half_delta,
                depth + 1,
            )
            if not second_summary.success:
                return second_summary, None

            acceptance = assess_displacement_control_acceptance(
                solver_converged=(
                    first_summary.solver_converged and second_summary.solver_converged
                ),
                requested_increment=local_delta,
                control_dof_before=history_targets[-1] if history_targets else 0.0,
                control_dof_after=second_summary.control_dof_after,
            )
            second_base_node = int(second_model["base_node"]) if second_model is not None else base_node
            second_element_tag = (
                int(second_model["element_tag"]) if second_model is not None else element_tag
            )
            second_row = sample_zero_length_section_record(
                ops,
                second_base_node,
                second_element_tag,
                ProtocolPoint(step=-1, p=math.nan, target_drift_m=trial_target),
                target_axial_force_mn,
                int(second_summary.newton_iterations)
                if math.isfinite(second_summary.newton_iterations)
                else 0,
            )
            axial_force_admissible = (
                abs(second_row.final_axial_force_residual_mn)
                <= axial_force_admissibility_tolerance_mn
            )
            accepted_substep_count = (
                first_summary.accepted_substep_count
                + second_summary.accepted_substep_count
                if acceptance.accepted and axial_force_admissible
                else 0
            )
            return (
                IncrementControlSummary(
                    success=(acceptance.accepted and axial_force_admissible),
                    accepted_substep_count=accepted_substep_count,
                    max_bisection_level=max(
                        first_summary.max_bisection_level,
                        second_summary.max_bisection_level,
                    ),
                    newton_iterations=(
                        first_summary.newton_iterations
                        + second_summary.newton_iterations
                        if math.isfinite(first_summary.newton_iterations)
                        and math.isfinite(second_summary.newton_iterations)
                        else math.nan
                    ),
                    test_norm=second_summary.test_norm,
                    domain_time_before=first_summary.domain_time_before,
                    domain_time_after=second_summary.domain_time_after,
                    control_dof_before=history_targets[-1]
                    if history_targets
                    else 0.0,
                    control_dof_after=(history_targets[-1] if history_targets else 0.0)
                    + local_delta,
                    solver_profile_label=second_summary.solver_profile_label,
                    solver_test_name=second_summary.solver_test_name,
                    solver_algorithm_name=second_summary.solver_algorithm_name,
                    solver_converged=acceptance.solver_converged,
                    control_increment_requested=acceptance.requested_increment,
                    control_increment_achieved=acceptance.achieved_increment,
                    control_increment_error=acceptance.increment_error,
                    control_relative_increment_error=acceptance.relative_increment_error,
                    control_direction_admissible=acceptance.direction_admissible,
                    control_magnitude_admissible=acceptance.magnitude_admissible,
                    accepted_replay_segments=(
                        first_summary.accepted_replay_segments
                        + second_summary.accepted_replay_segments
                        if acceptance.accepted and axial_force_admissible
                        else ()
                    ),
                ),
                second_model,
            )

        return recurse(accepted_targets, accepted_summaries, delta, 0)

    def capture_section_failure_trial_state(
        point: ProtocolPoint,
        increment_summary: IncrementControlSummary,
        previous_row: SectionBaselineRecord,
        protocol_branch_id_value: int,
        reversal_index_value: int,
        branch_step_index_value: int,
    ) -> SectionFailureTrialState:
        trial_curvature_y = float(ops.nodeDisp(top_node, 5))
        section_row: SectionBaselineRecord | None = None
        fiber_trial_rows: list[SectionFiberStateRecord] = []
        try:
            section_row = sample_zero_length_section_record(
                ops,
                base_node,
                element_tag,
                point,
                target_axial_force_mn,
                int(increment_summary.newton_iterations)
                if math.isfinite(increment_summary.newton_iterations)
                else 0,
            )
            fiber_trial_rows = sample_zero_length_section_fiber_records(
                ops,
                element_tag,
                point,
                section_row,
                section_layout_rows,
            )
            trial_curvature_y = section_row.curvature_y
        except Exception:
            section_row = None
            fiber_trial_rows = []

        target_increment_direction = signum(point.target_drift_m - previous.target_drift_m)
        actual_increment_direction = signum(trial_curvature_y - previous_row.curvature_y)
        control_trace_row = SectionControlTraceRecord(
            step=point.step,
            load_factor=point.p,
            stage="curvature_branch_failed_trial",
            target_curvature_y=point.target_drift_m,
            actual_curvature_y=trial_curvature_y,
            delta_target_curvature_y=point.target_drift_m - previous.target_drift_m,
            delta_actual_curvature_y=trial_curvature_y - previous_row.curvature_y,
            pseudo_time_before=previous.p,
            pseudo_time_after=point.p,
            pseudo_time_increment=point.p - previous.p,
            domain_time_before=increment_summary.domain_time_before,
            domain_time_after=increment_summary.domain_time_after,
            domain_time_increment=(
                increment_summary.domain_time_after - increment_summary.domain_time_before
            ),
            control_dof_before=increment_summary.control_dof_before,
            control_dof_after=increment_summary.control_dof_after,
            target_increment_direction=target_increment_direction,
            actual_increment_direction=actual_increment_direction,
            protocol_branch_id=protocol_branch_id_value,
            reversal_index=reversal_index_value,
            branch_step_index=branch_step_index_value,
            accepted_substep_count=increment_summary.accepted_substep_count,
            max_bisection_level=increment_summary.max_bisection_level,
            newton_iterations=increment_summary.newton_iterations,
            newton_iterations_per_substep=(
                increment_summary.newton_iterations / increment_summary.accepted_substep_count
                if increment_summary.accepted_substep_count > 0
                else math.nan
            ),
            test_norm=increment_summary.test_norm,
            target_axial_force_mn=target_axial_force_mn,
            actual_axial_force_mn=(
                section_row.axial_force_mn if section_row is not None else math.nan
            ),
            axial_force_residual_mn=(
                section_row.final_axial_force_residual_mn
                if section_row is not None
                else math.nan
            ),
            solver_converged=increment_summary.solver_converged,
            control_increment_requested=increment_summary.control_increment_requested,
            control_increment_achieved=increment_summary.control_increment_achieved,
            control_increment_error=increment_summary.control_increment_error,
            control_relative_increment_error=(
                increment_summary.control_relative_increment_error
            ),
            control_direction_admissible=(
                increment_summary.control_direction_admissible
            ),
            control_magnitude_admissible=(
                increment_summary.control_magnitude_admissible
            ),
        )
        return SectionFailureTrialState(
            section_row=section_row,
            fiber_rows=tuple(fiber_trial_rows),
            control_trace_row=control_trace_row,
        )

    model = initialize_section_state()
    base_node = int(model["base_node"])
    top_node = int(model["top_node"])
    element_tag = int(model["element_tag"])

    protocol = build_section_protocol(args)
    rows = [
        sample_zero_length_section_record(
            ops, base_node, element_tag, protocol[0], target_axial_force_mn, 0
        )
    ]
    fiber_rows = sample_zero_length_section_fiber_records(
        ops, element_tag, protocol[0], rows[0], section_layout_rows
    )
    control_trace_rows = [
        SectionControlTraceRecord(
            step=protocol[0].step,
            load_factor=protocol[0].p,
            stage="preload_equilibrated",
            target_curvature_y=protocol[0].target_drift_m,
            actual_curvature_y=rows[0].curvature_y,
            delta_target_curvature_y=0.0,
            delta_actual_curvature_y=0.0,
            pseudo_time_before=0.0,
            pseudo_time_after=protocol[0].p,
            pseudo_time_increment=0.0,
            domain_time_before=float(ops.getTime()),
            domain_time_after=float(ops.getTime()),
            domain_time_increment=0.0,
            control_dof_before=float(ops.nodeDisp(top_node, 5)),
            control_dof_after=float(ops.nodeDisp(top_node, 5)),
            target_increment_direction=0,
            actual_increment_direction=0,
            protocol_branch_id=0,
            reversal_index=0,
            branch_step_index=0,
            accepted_substep_count=0,
            max_bisection_level=0,
            newton_iterations=0.0,
            newton_iterations_per_substep=0.0,
            test_norm=math.nan,
            target_axial_force_mn=target_axial_force_mn,
            actual_axial_force_mn=rows[0].axial_force_mn,
            axial_force_residual_mn=rows[0].final_axial_force_residual_mn,
            solver_converged=True,
            control_direction_admissible=True,
            control_magnitude_admissible=True,
        )
    ]
    previous = protocol[0]
    previous_target_direction = 0
    protocol_branch_id = 0
    reversal_index = 0
    branch_step_index = 0
    accepted_targets: list[float] = []
    accepted_increment_summaries: list[IncrementControlSummary] = []
    for point in protocol[1:]:
        delta = point.target_drift_m - previous.target_drift_m
        increment_summary = IncrementControlSummary(
            success=True,
            accepted_substep_count=0,
            max_bisection_level=0,
            newton_iterations=0.0,
            test_norm=math.nan,
            domain_time_before=float(ops.getTime()),
            domain_time_after=float(ops.getTime()),
            control_dof_before=float(ops.nodeDisp(top_node, 5)),
            control_dof_after=float(ops.nodeDisp(top_node, 5)),
        )
        if abs(delta) > 1.0e-14:
            increment_summary, model = analyze_section_increment_from_clean_history(
                accepted_targets,
                accepted_increment_summaries,
                delta,
                args.max_bisections,
            )
            top_node = int(model["top_node"]) if model is not None else top_node
            base_node = int(model["base_node"]) if model is not None else base_node
            element_tag = int(model["element_tag"]) if model is not None else element_tag
        if abs(delta) > 1.0e-14 and not increment_summary.success:
            target_increment_direction = signum(delta)
            failure_protocol_branch_id = protocol_branch_id
            failure_reversal_index = reversal_index
            failure_branch_step_index = branch_step_index
            if target_increment_direction != 0:
                if previous_target_direction == 0:
                    failure_protocol_branch_id = 1
                    failure_branch_step_index = 1
                elif target_increment_direction != previous_target_direction:
                    failure_protocol_branch_id += 1
                    failure_reversal_index += 1
                    failure_branch_step_index = 1
                else:
                    failure_branch_step_index += 1
            failure_state = PartialSectionAnalysisState(
                rows=tuple(rows),
                fiber_rows=tuple(fiber_rows),
                control_trace_rows=tuple(control_trace_rows),
                failure_trial=capture_section_failure_trial_state(
                    point,
                    increment_summary,
                    rows[-1],
                    failure_protocol_branch_id,
                    failure_reversal_index,
                    failure_branch_step_index,
                ),
            )
            raise OpenSeesReferenceAnalysisFailure(
                "OpenSees zeroLengthSection curvature-control stage failed before "
                f"reaching step={point.step}, target_curvature={point.target_drift_m:+.6e}.",
                partial_section_state=failure_state,
            )
        rows.append(
            sample_zero_length_section_record(
                ops, base_node, element_tag, point, target_axial_force_mn, 1
            )
        )
        fiber_rows.extend(
            sample_zero_length_section_fiber_records(
                ops, element_tag, point, rows[-1], section_layout_rows
            )
        )
        target_increment_direction = signum(delta)
        actual_increment_direction = signum(rows[-1].curvature_y - rows[-2].curvature_y)
        if target_increment_direction != 0:
            if previous_target_direction == 0:
                protocol_branch_id = 1
                branch_step_index = 1
            elif target_increment_direction != previous_target_direction:
                protocol_branch_id += 1
                reversal_index += 1
                branch_step_index = 1
            else:
                branch_step_index += 1
            previous_target_direction = target_increment_direction
        newton_iterations_per_substep = (
            increment_summary.newton_iterations / increment_summary.accepted_substep_count
            if increment_summary.accepted_substep_count > 0
            else 0.0
        )
        control_trace_rows.append(
            SectionControlTraceRecord(
                step=point.step,
                load_factor=point.p,
                stage="curvature_branch",
                target_curvature_y=point.target_drift_m,
                actual_curvature_y=rows[-1].curvature_y,
                delta_target_curvature_y=delta,
                delta_actual_curvature_y=rows[-1].curvature_y - rows[-2].curvature_y,
                pseudo_time_before=previous.p,
                pseudo_time_after=point.p,
                pseudo_time_increment=point.p - previous.p,
                domain_time_before=increment_summary.domain_time_before,
                domain_time_after=increment_summary.domain_time_after,
                domain_time_increment=(
                    increment_summary.domain_time_after
                    - increment_summary.domain_time_before
                ),
                control_dof_before=increment_summary.control_dof_before,
                control_dof_after=increment_summary.control_dof_after,
                target_increment_direction=target_increment_direction,
                actual_increment_direction=actual_increment_direction,
                protocol_branch_id=protocol_branch_id,
                reversal_index=reversal_index,
                branch_step_index=branch_step_index,
                accepted_substep_count=increment_summary.accepted_substep_count,
                max_bisection_level=increment_summary.max_bisection_level,
                newton_iterations=increment_summary.newton_iterations,
                newton_iterations_per_substep=newton_iterations_per_substep,
                test_norm=increment_summary.test_norm,
                target_axial_force_mn=target_axial_force_mn,
                actual_axial_force_mn=rows[-1].axial_force_mn,
                axial_force_residual_mn=rows[-1].final_axial_force_residual_mn,
                solver_converged=increment_summary.solver_converged,
                control_increment_requested=increment_summary.control_increment_requested,
                control_increment_achieved=increment_summary.control_increment_achieved,
                control_increment_error=increment_summary.control_increment_error,
                control_relative_increment_error=(
                    increment_summary.control_relative_increment_error
                ),
                control_direction_admissible=(
                    increment_summary.control_direction_admissible
                ),
                control_magnitude_admissible=(
                    increment_summary.control_magnitude_admissible
                ),
            )
        )
        accepted_targets.append(point.target_drift_m)
        accepted_increment_summaries.append(increment_summary)
        previous = point

    kappas = [row.curvature_y for row in rows]
    moments = [row.moment_y_mnm for row in rows]
    finalized_rows = [
        replace(
            row,
            tangent_eiy=(
                row.tangent_eiy
                if math.isfinite(row.tangent_eiy)
                else numerical_tangent(kappas, moments, idx)
            ),
        )
        for idx, row in enumerate(rows)
    ]
    return finalized_rows, fiber_rows, control_trace_rows


def write_structural_outputs(
    out_dir: Path,
    manifest: dict[str, object],
    protocol: list[ProtocolPoint],
    hysteresis_rows: list[StepRecord],
    section_rows: list[SectionRecord],
    control_rows: list[ControlStateRecord],
    fiber_rows: list[SectionFiberStateRecord],
    section_layout_rows: list[dict[str, object]],
    station_layout_rows: list[dict[str, object]],
) -> None:
    ensure_dir(out_dir)
    (out_dir / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    write_csv(
        out_dir / "protocol.csv",
        ("step", "p", "target_drift_m"),
        ({"step": p.step, "p": p.p, "target_drift_m": p.target_drift_m} for p in protocol),
    )
    write_csv(
        out_dir / "hysteresis.csv",
        ("step", "p", "drift_m", "base_shear_MN"),
        (
            {
                "step": row.step,
                "p": row.p,
                "drift_m": row.drift_m,
                "base_shear_MN": row.base_shear_mn,
            }
            for row in hysteresis_rows
        ),
    )
    write_csv(
        out_dir / "section_response.csv",
        (
            "step",
            "p",
            "drift_m",
            "section_gp",
            "xi",
            "axial_strain",
            "curvature_y",
            "curvature_z",
            "axial_force_MN",
            "moment_y_MNm",
            "moment_z_MNm",
            "tangent_ea",
            "tangent_eiy",
            "tangent_eiz",
            "tangent_eiy_direct_raw",
            "tangent_eiz_direct_raw",
            "raw_k00",
            "raw_k0y",
            "raw_ky0",
            "raw_kyy",
        ),
        (
            {
                "step": row.step,
                "p": row.p,
                "drift_m": row.drift_m,
                "section_gp": row.section_gp,
                "xi": row.xi,
                "axial_strain": row.axial_strain,
                "curvature_y": row.curvature_y,
                "curvature_z": row.curvature_z,
                "axial_force_MN": row.axial_force_mn,
                "moment_y_MNm": row.moment_y_mnm,
                "moment_z_MNm": row.moment_z_mnm,
                "tangent_ea": row.tangent_ea,
                "tangent_eiy": row.tangent_eiy,
                "tangent_eiz": row.tangent_eiz,
                "tangent_eiy_direct_raw": row.tangent_eiy_direct_raw,
                "tangent_eiz_direct_raw": row.tangent_eiz_direct_raw,
                "raw_k00": row.raw_tangent_k00,
                "raw_k0y": row.raw_tangent_k0y,
                "raw_ky0": row.raw_tangent_ky0,
                "raw_kyy": row.raw_tangent_kyy,
            }
            for row in section_rows
        ),
    )
    write_csv(
        out_dir / "moment_curvature_base.csv",
        (
            "step",
            "p",
            "drift_m",
            "section_gp",
            "xi",
            "curvature_y",
            "moment_y_MNm",
            "axial_force_MN",
            "tangent_eiy",
        ),
        make_base_side_history(section_rows),
    )
    write_csv(
        out_dir / "control_state.csv",
        (
            "step",
            "p",
            "target_drift_m",
            "actual_tip_drift_m",
            "top_axial_displacement_m",
            "base_shear_MN",
            "base_axial_reaction_MN",
            "accepted_substep_count",
            "max_bisection_level",
            "newton_iterations",
            "newton_iterations_per_substep",
            "test_norm",
            "solver_profile_label",
            "solver_test_name",
            "solver_algorithm_name",
            "stage",
        ),
        (
            {
                "step": row.step,
                "p": row.p,
                "target_drift_m": row.target_drift_m,
                "actual_tip_drift_m": row.actual_tip_drift_m,
                "top_axial_displacement_m": row.top_axial_displacement_m,
                "base_shear_MN": row.base_shear_mn,
                "base_axial_reaction_MN": row.base_axial_reaction_mn,
                "accepted_substep_count": row.accepted_substep_count,
                "max_bisection_level": row.max_bisection_level,
                "newton_iterations": row.newton_iterations,
                "newton_iterations_per_substep": row.newton_iterations_per_substep,
                "test_norm": row.test_norm,
                "solver_profile_label": row.solver_profile_label,
                "solver_test_name": row.solver_test_name,
                "solver_algorithm_name": row.solver_algorithm_name,
                "stage": row.stage,
            }
            for row in control_rows
        ),
    )
    write_csv(
        out_dir / "section_fiber_state_history.csv",
        (
            "step",
            "p",
            "drift_m",
            "section_gp",
            "xi",
            "axial_strain",
            "curvature_y",
            "zero_curvature_anchor",
            "fiber_index",
            "y",
            "z",
            "area",
            "zone",
            "material_role",
            "material_tag",
            "strain_xx",
            "stress_xx_MPa",
            "tangent_xx_MPa",
            "axial_force_contribution_MN",
            "moment_y_contribution_MNm",
            "raw_k00_contribution",
            "raw_k0y_contribution",
            "raw_kyy_contribution",
        ),
        (
            {
                "step": row.step,
                "p": row.p,
                "drift_m": row.drift_m,
                "section_gp": row.section_gp,
                "xi": row.xi,
                "axial_strain": row.solved_axial_strain,
                "curvature_y": row.curvature_y,
                "zero_curvature_anchor": int(row.zero_curvature_anchor),
                "fiber_index": row.fiber_index,
                "y": row.y,
                "z": row.z,
                "area": row.area,
                "zone": row.zone,
                "material_role": row.material_role,
                "material_tag": row.material_tag,
                "strain_xx": row.strain_xx,
                "stress_xx_MPa": row.stress_xx_mpa,
                "tangent_xx_MPa": row.tangent_xx_mpa,
                "axial_force_contribution_MN": row.axial_force_contribution_mn,
                "moment_y_contribution_MNm": row.moment_y_contribution_mnm,
                "raw_k00_contribution": row.raw_tangent_k00_contribution,
                "raw_k0y_contribution": row.raw_tangent_k0y_contribution,
                "raw_kyy_contribution": row.raw_tangent_kyy_contribution,
            }
            for row in fiber_rows
        ),
    )
    write_csv(
        out_dir / "section_layout.csv",
        ("fiber_index", "y", "z", "area", "zone", "material_role", "material_tag"),
        section_layout_rows,
    )
    write_csv(
        out_dir / "section_station_layout.csv",
        ("section_gp", "xi"),
        station_layout_rows,
    )
    preload_row = next((row for row in control_rows if row.step == 0), None)
    preload_section_rows = [row for row in section_rows if row.step == 0]
    preload_summary = {"status": "unavailable"}
    if preload_row is not None:
        mean = lambda values: sum(values) / len(values) if values else math.nan
        preload_summary = {
            "status": "available",
            "step": preload_row.step,
            "p": preload_row.p,
            "target_drift_m": preload_row.target_drift_m,
            "actual_tip_drift_m": preload_row.actual_tip_drift_m,
            "top_axial_displacement_m": preload_row.top_axial_displacement_m,
            "base_shear_MN": preload_row.base_shear_mn,
            "base_axial_reaction_MN": preload_row.base_axial_reaction_mn,
            "mean_section_axial_strain": mean(
                [row.axial_strain for row in preload_section_rows]
            ),
            "mean_section_axial_force_MN": mean(
                [row.axial_force_mn for row in preload_section_rows]
            ),
            "mean_section_tangent_eiy": mean(
                [row.tangent_eiy for row in preload_section_rows]
            ),
            "section_station_count": len(preload_section_rows),
        }
    (out_dir / "preload_state.json").write_text(
        json.dumps(preload_summary, indent=2), encoding="utf-8"
    )


def write_section_outputs(
    out_dir: Path,
    manifest: dict[str, object],
    protocol: list[ProtocolPoint],
    rows: list[SectionBaselineRecord],
    fiber_rows: list[SectionFiberStateRecord],
    control_trace_rows: list[SectionControlTraceRecord],
    section_layout_rows: list[dict[str, object]],
) -> None:
    ensure_dir(out_dir)
    (out_dir / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_csv(
        out_dir / "protocol.csv",
        ("step", "p", "target_curvature_y"),
        (
            {
                "step": point.step,
                "p": point.p,
                "target_curvature_y": point.target_drift_m,
            }
            for point in protocol
        ),
    )
    write_csv(
        out_dir / "section_moment_curvature_baseline.csv",
        (
            "step",
            "load_factor",
            "target_axial_force_MN",
            "solved_axial_strain",
            "curvature_y",
            "curvature_z",
            "axial_force_MN",
            "moment_y_MNm",
            "moment_z_MNm",
            "tangent_ea",
            "tangent_eiy",
            "tangent_eiz",
            "tangent_eiy_direct_raw",
            "tangent_eiz_direct_raw",
            "newton_iterations",
            "final_axial_force_residual_MN",
            "support_axial_reaction_MN",
            "support_moment_y_reaction_MNm",
            "axial_equilibrium_gap_MN",
            "moment_equilibrium_gap_MNm",
        ),
        (
            {
                "step": row.step,
                "load_factor": row.load_factor,
                "target_axial_force_MN": row.target_axial_force_mn,
                "solved_axial_strain": row.solved_axial_strain,
                "curvature_y": row.curvature_y,
                "curvature_z": row.curvature_z,
                "axial_force_MN": row.axial_force_mn,
                "moment_y_MNm": row.moment_y_mnm,
                "moment_z_MNm": row.moment_z_mnm,
                "tangent_ea": row.tangent_ea,
                "tangent_eiy": row.tangent_eiy,
                "tangent_eiz": row.tangent_eiz,
                "tangent_eiy_direct_raw": row.tangent_eiy_direct_raw,
                "tangent_eiz_direct_raw": row.tangent_eiz_direct_raw,
                "newton_iterations": row.newton_iterations,
                "final_axial_force_residual_MN": row.final_axial_force_residual_mn,
                "support_axial_reaction_MN": row.support_axial_reaction_mn,
                "support_moment_y_reaction_MNm": row.support_moment_y_reaction_mnm,
                "axial_equilibrium_gap_MN": row.axial_equilibrium_gap_mn,
                "moment_equilibrium_gap_MNm": row.moment_equilibrium_gap_mnm,
            }
            for row in rows
        ),
    )
    kappas = [row.curvature_y for row in rows]
    moments = [row.moment_y_mnm for row in rows]
    write_csv(
        out_dir / "section_tangent_diagnostics.csv",
        (
            "step",
            "load_factor",
            "curvature_y",
            "moment_y_MNm",
            "zero_curvature_anchor",
            "tangent_eiy_condensed",
            "tangent_eiy_direct",
            "tangent_eiy_numerical",
            "tangent_eiy_left",
            "tangent_eiy_right",
            "tangent_consistency_rel_error",
            "raw_k00",
            "raw_k0y",
            "raw_ky0",
            "raw_kyy",
            "support_axial_reaction_MN",
            "support_moment_y_reaction_MNm",
            "axial_equilibrium_gap_MN",
            "moment_equilibrium_gap_MNm",
        ),
        (
            {
                "step": row.step,
                "load_factor": row.load_factor,
                "curvature_y": row.curvature_y,
                "moment_y_MNm": row.moment_y_mnm,
                "zero_curvature_anchor": int(abs(row.curvature_y) <= 1.0e-12),
                "tangent_eiy_condensed": row.tangent_eiy,
                "tangent_eiy_direct": row.tangent_eiy_direct_raw,
                "tangent_eiy_numerical": numerical_tangent_pair(kappas, moments, idx)[0],
                "tangent_eiy_left": numerical_tangent_pair(kappas, moments, idx)[1],
                "tangent_eiy_right": numerical_tangent_pair(kappas, moments, idx)[2],
                "tangent_consistency_rel_error": relative_error(
                    row.tangent_eiy, numerical_tangent_pair(kappas, moments, idx)[0]
                ),
                "raw_k00": row.raw_tangent_k00,
                "raw_k0y": row.raw_tangent_k0y,
                "raw_ky0": row.raw_tangent_ky0,
                "raw_kyy": row.raw_tangent_kyy,
                "support_axial_reaction_MN": row.support_axial_reaction_mn,
                "support_moment_y_reaction_MNm": row.support_moment_y_reaction_mnm,
                "axial_equilibrium_gap_MN": row.axial_equilibrium_gap_mn,
                "moment_equilibrium_gap_MNm": row.moment_equilibrium_gap_mnm,
            }
            for idx, row in enumerate(rows)
        ),
    )
    write_csv(
        out_dir / "section_fiber_state_history.csv",
        (
            "step",
            "load_factor",
            "solved_axial_strain",
            "curvature_y",
            "zero_curvature_anchor",
            "fiber_index",
            "y",
            "z",
            "area",
            "zone",
            "material_role",
            "material_tag",
            "strain_xx",
            "stress_xx_MPa",
            "tangent_xx_MPa",
            "axial_force_contribution_MN",
            "moment_y_contribution_MNm",
            "raw_k00_contribution",
            "raw_k0y_contribution",
            "raw_kyy_contribution",
        ),
        (
            {
                "step": row.step,
                "load_factor": row.load_factor,
                "solved_axial_strain": row.solved_axial_strain,
                "curvature_y": row.curvature_y,
                "zero_curvature_anchor": int(row.zero_curvature_anchor),
                "fiber_index": row.fiber_index,
                "y": row.y,
                "z": row.z,
                "area": row.area,
                "zone": row.zone,
                "material_role": row.material_role,
                "material_tag": row.material_tag,
                "strain_xx": row.strain_xx,
                "stress_xx_MPa": row.stress_xx_mpa,
                "tangent_xx_MPa": row.tangent_xx_mpa,
                "axial_force_contribution_MN": row.axial_force_contribution_mn,
                "moment_y_contribution_MNm": row.moment_y_contribution_mnm,
                "raw_k00_contribution": row.raw_tangent_k00_contribution,
                "raw_k0y_contribution": row.raw_tangent_k0y_contribution,
                "raw_kyy_contribution": row.raw_tangent_kyy_contribution,
            }
            for row in fiber_rows
        ),
    )
    write_csv(
        out_dir / "section_control_trace.csv",
        (
            "step",
            "load_factor",
            "stage",
            "target_curvature_y",
            "actual_curvature_y",
            "delta_target_curvature_y",
            "delta_actual_curvature_y",
            "pseudo_time_before",
            "pseudo_time_after",
            "pseudo_time_increment",
            "domain_time_before",
            "domain_time_after",
            "domain_time_increment",
            "control_dof_before",
            "control_dof_after",
            "target_increment_direction",
            "actual_increment_direction",
            "protocol_branch_id",
            "reversal_index",
            "branch_step_index",
            "accepted_substep_count",
            "max_bisection_level",
            "newton_iterations",
            "newton_iterations_per_substep",
            "test_norm",
            "target_axial_force_MN",
            "actual_axial_force_MN",
            "axial_force_residual_MN",
        ),
        (
            {
                "step": row.step,
                "load_factor": row.load_factor,
                "stage": row.stage,
                "target_curvature_y": row.target_curvature_y,
                "actual_curvature_y": row.actual_curvature_y,
                "delta_target_curvature_y": row.delta_target_curvature_y,
                "delta_actual_curvature_y": row.delta_actual_curvature_y,
                "pseudo_time_before": row.pseudo_time_before,
                "pseudo_time_after": row.pseudo_time_after,
                "pseudo_time_increment": row.pseudo_time_increment,
                "domain_time_before": row.domain_time_before,
                "domain_time_after": row.domain_time_after,
                "domain_time_increment": row.domain_time_increment,
                "control_dof_before": row.control_dof_before,
                "control_dof_after": row.control_dof_after,
                "target_increment_direction": row.target_increment_direction,
                "actual_increment_direction": row.actual_increment_direction,
                "protocol_branch_id": row.protocol_branch_id,
                "reversal_index": row.reversal_index,
                "branch_step_index": row.branch_step_index,
                "accepted_substep_count": row.accepted_substep_count,
                "max_bisection_level": row.max_bisection_level,
                "newton_iterations": row.newton_iterations,
                "newton_iterations_per_substep": row.newton_iterations_per_substep,
                "test_norm": row.test_norm,
                "target_axial_force_MN": row.target_axial_force_mn,
                "actual_axial_force_MN": row.actual_axial_force_mn,
                "axial_force_residual_MN": row.axial_force_residual_mn,
            }
            for row in control_trace_rows
        ),
    )
    write_csv(
        out_dir / "section_layout.csv",
        ("fiber_index", "y", "z", "area", "zone", "material_role", "material_tag"),
        section_layout_rows,
    )
    write_csv(
        out_dir / "section_station_layout.csv",
        ("section_gp", "xi"),
        build_station_layout_rows((0.0,)),
    )


def write_section_failure_trial_outputs(
    out_dir: Path,
    failure_trial: SectionFailureTrialState,
) -> None:
    ensure_dir(out_dir)
    write_csv(
        out_dir / "section_control_trace_failure_trial.csv",
        (
            "step",
            "load_factor",
            "stage",
            "target_curvature_y",
            "actual_curvature_y",
            "delta_target_curvature_y",
            "delta_actual_curvature_y",
            "pseudo_time_before",
            "pseudo_time_after",
            "pseudo_time_increment",
            "domain_time_before",
            "domain_time_after",
            "domain_time_increment",
            "control_dof_before",
            "control_dof_after",
            "target_increment_direction",
            "actual_increment_direction",
            "protocol_branch_id",
            "reversal_index",
            "branch_step_index",
            "accepted_substep_count",
            "max_bisection_level",
            "newton_iterations",
            "newton_iterations_per_substep",
            "test_norm",
            "target_axial_force_MN",
            "actual_axial_force_MN",
            "axial_force_residual_MN",
            "solver_converged",
            "control_increment_requested",
            "control_increment_achieved",
            "control_increment_error",
            "control_relative_increment_error",
            "control_direction_admissible",
            "control_magnitude_admissible",
        ),
        (
            {
                "step": failure_trial.control_trace_row.step,
                "load_factor": failure_trial.control_trace_row.load_factor,
                "stage": failure_trial.control_trace_row.stage,
                "target_curvature_y": failure_trial.control_trace_row.target_curvature_y,
                "actual_curvature_y": failure_trial.control_trace_row.actual_curvature_y,
                "delta_target_curvature_y": failure_trial.control_trace_row.delta_target_curvature_y,
                "delta_actual_curvature_y": failure_trial.control_trace_row.delta_actual_curvature_y,
                "pseudo_time_before": failure_trial.control_trace_row.pseudo_time_before,
                "pseudo_time_after": failure_trial.control_trace_row.pseudo_time_after,
                "pseudo_time_increment": failure_trial.control_trace_row.pseudo_time_increment,
                "domain_time_before": failure_trial.control_trace_row.domain_time_before,
                "domain_time_after": failure_trial.control_trace_row.domain_time_after,
                "domain_time_increment": failure_trial.control_trace_row.domain_time_increment,
                "control_dof_before": failure_trial.control_trace_row.control_dof_before,
                "control_dof_after": failure_trial.control_trace_row.control_dof_after,
                "target_increment_direction": failure_trial.control_trace_row.target_increment_direction,
                "actual_increment_direction": failure_trial.control_trace_row.actual_increment_direction,
                "protocol_branch_id": failure_trial.control_trace_row.protocol_branch_id,
                "reversal_index": failure_trial.control_trace_row.reversal_index,
                "branch_step_index": failure_trial.control_trace_row.branch_step_index,
                "accepted_substep_count": failure_trial.control_trace_row.accepted_substep_count,
                "max_bisection_level": failure_trial.control_trace_row.max_bisection_level,
                "newton_iterations": failure_trial.control_trace_row.newton_iterations,
                "newton_iterations_per_substep": failure_trial.control_trace_row.newton_iterations_per_substep,
                "test_norm": failure_trial.control_trace_row.test_norm,
                "target_axial_force_MN": failure_trial.control_trace_row.target_axial_force_mn,
                "actual_axial_force_MN": failure_trial.control_trace_row.actual_axial_force_mn,
                "axial_force_residual_MN": failure_trial.control_trace_row.axial_force_residual_mn,
                "solver_converged": int(failure_trial.control_trace_row.solver_converged),
                "control_increment_requested": failure_trial.control_trace_row.control_increment_requested,
                "control_increment_achieved": failure_trial.control_trace_row.control_increment_achieved,
                "control_increment_error": failure_trial.control_trace_row.control_increment_error,
                "control_relative_increment_error": failure_trial.control_trace_row.control_relative_increment_error,
                "control_direction_admissible": int(failure_trial.control_trace_row.control_direction_admissible),
                "control_magnitude_admissible": int(failure_trial.control_trace_row.control_magnitude_admissible),
            },
        ),
    )
    if failure_trial.section_row is not None:
        write_csv(
            out_dir / "section_failure_trial.csv",
            (
                "step",
                "load_factor",
                "target_axial_force_MN",
                "solved_axial_strain",
                "curvature_y",
                "curvature_z",
                "axial_force_MN",
                "moment_y_MNm",
                "moment_z_MNm",
                "tangent_ea",
                "tangent_eiy",
                "tangent_eiz",
                "tangent_eiy_direct_raw",
                "tangent_eiz_direct_raw",
                "newton_iterations",
                "final_axial_force_residual_MN",
                "raw_k00",
                "raw_k0y",
                "raw_ky0",
                "raw_kyy",
                "support_axial_reaction_MN",
                "support_moment_y_reaction_MNm",
                "axial_equilibrium_gap_MN",
                "moment_equilibrium_gap_MNm",
            ),
            (
                {
                    "step": failure_trial.section_row.step,
                    "load_factor": failure_trial.section_row.load_factor,
                    "target_axial_force_MN": failure_trial.section_row.target_axial_force_mn,
                    "solved_axial_strain": failure_trial.section_row.solved_axial_strain,
                    "curvature_y": failure_trial.section_row.curvature_y,
                    "curvature_z": failure_trial.section_row.curvature_z,
                    "axial_force_MN": failure_trial.section_row.axial_force_mn,
                    "moment_y_MNm": failure_trial.section_row.moment_y_mnm,
                    "moment_z_MNm": failure_trial.section_row.moment_z_mnm,
                    "tangent_ea": failure_trial.section_row.tangent_ea,
                    "tangent_eiy": failure_trial.section_row.tangent_eiy,
                    "tangent_eiz": failure_trial.section_row.tangent_eiz,
                    "tangent_eiy_direct_raw": failure_trial.section_row.tangent_eiy_direct_raw,
                    "tangent_eiz_direct_raw": failure_trial.section_row.tangent_eiz_direct_raw,
                    "newton_iterations": failure_trial.section_row.newton_iterations,
                    "final_axial_force_residual_MN": failure_trial.section_row.final_axial_force_residual_mn,
                    "raw_k00": failure_trial.section_row.raw_tangent_k00,
                    "raw_k0y": failure_trial.section_row.raw_tangent_k0y,
                    "raw_ky0": failure_trial.section_row.raw_tangent_ky0,
                    "raw_kyy": failure_trial.section_row.raw_tangent_kyy,
                    "support_axial_reaction_MN": failure_trial.section_row.support_axial_reaction_mn,
                    "support_moment_y_reaction_MNm": failure_trial.section_row.support_moment_y_reaction_mnm,
                    "axial_equilibrium_gap_MN": failure_trial.section_row.axial_equilibrium_gap_mn,
                    "moment_equilibrium_gap_MNm": failure_trial.section_row.moment_equilibrium_gap_mnm,
                },
            ),
        )
    if failure_trial.fiber_rows:
        write_csv(
            out_dir / "section_failure_trial_fibers.csv",
            (
                "step",
                "load_factor",
                "solved_axial_strain",
                "curvature_y",
                "zero_curvature_anchor",
                "fiber_index",
                "y",
                "z",
                "area",
                "zone",
                "material_role",
                "material_tag",
                "strain_xx",
                "stress_xx_MPa",
                "tangent_xx_MPa",
                "axial_force_contribution_MN",
                "moment_y_contribution_MNm",
                "raw_k00_contribution",
                "raw_k0y_contribution",
                "raw_kyy_contribution",
            ),
            (
                {
                    "step": row.step,
                    "load_factor": row.load_factor,
                    "solved_axial_strain": row.solved_axial_strain,
                    "curvature_y": row.curvature_y,
                    "zero_curvature_anchor": int(row.zero_curvature_anchor),
                    "fiber_index": row.fiber_index,
                    "y": row.y,
                    "z": row.z,
                    "area": row.area,
                    "zone": row.zone,
                    "material_role": row.material_role,
                    "material_tag": row.material_tag,
                    "strain_xx": row.strain_xx,
                    "stress_xx_MPa": row.stress_xx_mpa,
                    "tangent_xx_MPa": row.tangent_xx_mpa,
                    "axial_force_contribution_MN": row.axial_force_contribution_mn,
                    "moment_y_contribution_MNm": row.moment_y_contribution_mnm,
                    "raw_k00_contribution": row.raw_tangent_k00_contribution,
                    "raw_k0y_contribution": row.raw_tangent_k0y_contribution,
                    "raw_kyy_contribution": row.raw_tangent_kyy_contribution,
                }
                for row in failure_trial.fiber_rows
            ),
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def maybe_write_comparison(
    out_dir: Path,
    falln_hysteresis: Path | None,
    falln_mk: Path | None,
    falln_section_response: Path | None,
    falln_control_state: Path | None,
    falln_section_layout: Path | None,
    falln_station_layout: Path | None,
    falln_section_baseline: Path | None,
    falln_section_fiber_history: Path | None,
    falln_section_control_trace: Path | None,
    model_kind: str,
    beam_element_family: str | None = None,
) -> None:
    if (
        not falln_hysteresis
        and not falln_mk
        and not falln_section_response
        and not falln_control_state
        and not falln_section_layout
        and not falln_station_layout
        and not falln_section_baseline
        and not falln_section_fiber_history
        and not falln_section_control_trace
    ):
        return

    summary: dict[str, object] = {}

    if model_kind == "structural":
        summary["structural_station_observable_contract"] = (
            structural_station_observable_contract(beam_element_family or "unknown")
        )

    def compare_by_step(
        lhs_rows: list[dict[str, str]],
        rhs_rows: list[dict[str, str]],
        lhs_key: str,
        rhs_key: str,
        label: str,
    ) -> dict[str, object]:
        lhs_by_step = {int(row["step"]): float(row[lhs_key]) for row in lhs_rows}
        rhs_by_step = {int(row["step"]): float(row[rhs_key]) for row in rhs_rows}
        common_steps = sorted(set(lhs_by_step) & set(rhs_by_step))
        if not common_steps:
            return {"shared_step_count": 0}
        reference_peak = max(abs(rhs_by_step[step]) for step in common_steps)
        activity_floor = max(5.0e-2 * reference_peak, 1.0e-12)
        active_steps = [
            step for step in common_steps if abs(rhs_by_step[step]) >= activity_floor
        ]
        rel_steps = active_steps or common_steps
        rel_errors = [
            relative_error(lhs_by_step[step], rhs_by_step[step]) for step in rel_steps
        ]
        abs_errors = [
            abs(lhs_by_step[step] - rhs_by_step[step]) for step in common_steps
        ]
        rms = math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors))
        return {
            "shared_step_count": len(common_steps),
            "active_step_count": len(active_steps),
            "activity_floor": activity_floor,
            f"max_abs_{label}": max(abs_errors),
            f"max_rel_{label}": max(rel_errors),
            f"rms_rel_{label}": rms,
        }

    def compare_derived_by_step(
        lhs_rows: list[dict[str, str]],
        rhs_rows: list[dict[str, str]],
        lhs_of,
        rhs_of,
        label: str,
    ) -> dict[str, object]:
        lhs_by_step = {int(row["step"]): float(lhs_of(row)) for row in lhs_rows}
        rhs_by_step = {int(row["step"]): float(rhs_of(row)) for row in rhs_rows}
        common_steps = sorted(set(lhs_by_step) & set(rhs_by_step))
        if not common_steps:
            return {"shared_step_count": 0}
        reference_peak = max(abs(rhs_by_step[step]) for step in common_steps)
        activity_floor = max(5.0e-2 * reference_peak, 1.0e-12)
        active_steps = [
            step for step in common_steps if abs(rhs_by_step[step]) >= activity_floor
        ]
        rel_steps = active_steps or common_steps
        rel_errors = [
            relative_error(lhs_by_step[step], rhs_by_step[step]) for step in rel_steps
        ]
        abs_errors = [
            abs(lhs_by_step[step] - rhs_by_step[step]) for step in common_steps
        ]
        rms = math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors))
        return {
            "shared_step_count": len(common_steps),
            "active_step_count": len(active_steps),
            "activity_floor": activity_floor,
            f"max_abs_{label}": max(abs_errors),
            f"max_rel_{label}": max(rel_errors),
            f"rms_rel_{label}": rms,
        }

    def compare_integer_by_step(
        lhs_rows: list[dict[str, str]],
        rhs_rows: list[dict[str, str]],
        lhs_key: str,
        rhs_key: str,
        label: str,
    ) -> dict[str, object]:
        lhs_by_step = {
            int(row["step"]): int(round(float(row[lhs_key]))) for row in lhs_rows
        }
        rhs_by_step = {
            int(row["step"]): int(round(float(row[rhs_key]))) for row in rhs_rows
        }
        common_steps = sorted(set(lhs_by_step) & set(rhs_by_step))
        if not common_steps:
            return {"shared_step_count": 0}

        abs_errors = [
            abs(lhs_by_step[step] - rhs_by_step[step]) for step in common_steps
        ]
        mismatched_steps = [
            step for step in common_steps if lhs_by_step[step] != rhs_by_step[step]
        ]
        rms_abs = math.sqrt(sum(err * err for err in abs_errors) / len(abs_errors))
        return {
            "shared_step_count": len(common_steps),
            f"max_abs_{label}": max(abs_errors),
            f"rms_abs_{label}": rms_abs,
            f"mismatched_{label}_step_count": len(mismatched_steps),
            f"exact_match_{label}": len(mismatched_steps) == 0,
        }

    def numerical_tangent_rows(
        rows: list[dict[str, str]],
        x_key: str,
        y_key: str,
    ) -> list[dict[str, object]]:
        xs = [float(row[x_key]) for row in rows]
        ys = [float(row[y_key]) for row in rows]
        diagnostics: list[dict[str, object]] = []
        for idx, row in enumerate(rows):
            numerical, left, right = numerical_tangent_pair(xs, ys, idx)
            diagnostics.append(
                {
                    "step": int(row["step"]),
                    "curvature_y": float(row[x_key]),
                    "direct": float(row["tangent_eiy"]),
                    "numerical": numerical,
                    "left": left,
                    "right": right,
                    "zero_curvature_anchor": abs(float(row[x_key])) <= 1.0e-12,
                }
            )
        return diagnostics

    def compare_direct_vs_numerical(
        diagnostics: list[dict[str, object]],
    ) -> dict[str, object]:
        if not diagnostics:
            return {"shared_step_count": 0}

        rel_errors = [
            relative_error(float(row["direct"]), float(row["numerical"]))
            for row in diagnostics
            if math.isfinite(float(row["numerical"]))
        ]
        zero_anchor_rows = [
            row for row in diagnostics if bool(row["zero_curvature_anchor"])
        ]
        branch_rows = [
            row for row in diagnostics if not bool(row["zero_curvature_anchor"])
        ]

        def summarize(rows_subset: list[dict[str, object]]) -> dict[str, object]:
            if not rows_subset:
                return {"shared_step_count": 0}
            subset_errors = [
                relative_error(float(row["direct"]), float(row["numerical"]))
                for row in rows_subset
                if math.isfinite(float(row["numerical"]))
            ]
            if not subset_errors:
                return {"shared_step_count": len(rows_subset)}
            return {
                "shared_step_count": len(rows_subset),
                "max_rel_tangent_consistency_error": max(subset_errors),
                "rms_rel_tangent_consistency_error": math.sqrt(
                    sum(err * err for err in subset_errors) / len(subset_errors)
                ),
            }

        return {
            "shared_step_count": len(diagnostics),
            "max_rel_tangent_consistency_error": max(rel_errors),
            "rms_rel_tangent_consistency_error": math.sqrt(
                sum(err * err for err in rel_errors) / len(rel_errors)
            ),
            "branch_only": summarize(branch_rows),
            "zero_curvature_anchor_only": summarize(zero_anchor_rows),
        }

    def compare_by_step_subset(
        lhs_rows: list[dict[str, str]],
        rhs_rows: list[dict[str, str]],
        lhs_key: str,
        rhs_key: str,
        label: str,
        predicate,
    ) -> dict[str, object]:
        lhs_filtered = [row for row in lhs_rows if predicate(row)]
        rhs_filtered = [row for row in rhs_rows if predicate(row)]
        return compare_by_step(lhs_filtered, rhs_filtered, lhs_key, rhs_key, label)

    def compare_by_step_and_station(
        lhs_rows: list[dict[str, str]],
        rhs_rows: list[dict[str, str]],
        lhs_key: str,
        rhs_key: str,
        label: str,
    ) -> dict[str, object]:
        key_of = lambda row: (int(row["step"]), int(row["section_gp"]))
        lhs_by_key = {key_of(row): float(row[lhs_key]) for row in lhs_rows}
        rhs_by_key = {key_of(row): float(row[rhs_key]) for row in rhs_rows}
        common_keys = sorted(set(lhs_by_key) & set(rhs_by_key))
        if not common_keys:
            return {"shared_point_count": 0}

        reference_peak = max(abs(rhs_by_key[key]) for key in common_keys)
        activity_floor = max(5.0e-2 * reference_peak, 1.0e-12)
        active_keys = [
            key for key in common_keys if abs(rhs_by_key[key]) >= activity_floor
        ]
        rel_keys = active_keys or common_keys

        rel_errors = [
            relative_error(lhs_by_key[key], rhs_by_key[key]) for key in rel_keys
        ]
        abs_errors = [
            abs(lhs_by_key[key] - rhs_by_key[key]) for key in common_keys
        ]
        rms = math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors))

        by_station: dict[str, dict[str, float]] = {}
        for station in sorted({key[1] for key in common_keys}):
            station_keys = [key for key in common_keys if key[1] == station]
            station_active = [
                key for key in station_keys if abs(rhs_by_key[key]) >= activity_floor
            ]
            station_rel_keys = station_active or station_keys
            station_rel_errors = [
                relative_error(lhs_by_key[key], rhs_by_key[key])
                for key in station_rel_keys
            ]
            by_station[str(station)] = {
                "shared_point_count": len(station_keys),
                "active_point_count": len(station_active),
                f"max_rel_{label}": max(station_rel_errors),
                f"rms_rel_{label}": math.sqrt(
                    sum(err * err for err in station_rel_errors)
                    / len(station_rel_errors)
                ),
            }

        return {
            "shared_point_count": len(common_keys),
            "active_point_count": len(active_keys),
            "activity_floor": activity_floor,
            f"max_abs_{label}": max(abs_errors),
            f"max_rel_{label}": max(rel_errors),
            f"rms_rel_{label}": rms,
            "by_station": by_station,
        }

    def compare_derived_by_step_and_station(
        lhs_rows: list[dict[str, str]],
        rhs_rows: list[dict[str, str]],
        lhs_of,
        rhs_of,
        label: str,
    ) -> dict[str, object]:
        key_of = lambda row: (int(row["step"]), int(row["section_gp"]))
        lhs_by_key = {key_of(row): float(lhs_of(row)) for row in lhs_rows}
        rhs_by_key = {key_of(row): float(rhs_of(row)) for row in rhs_rows}
        common_keys = sorted(set(lhs_by_key) & set(rhs_by_key))
        if not common_keys:
            return {"shared_point_count": 0}

        reference_peak = max(abs(rhs_by_key[key]) for key in common_keys)
        activity_floor = max(5.0e-2 * reference_peak, 1.0e-12)
        active_keys = [
            key for key in common_keys if abs(rhs_by_key[key]) >= activity_floor
        ]
        rel_keys = active_keys or common_keys
        rel_errors = [
            relative_error(lhs_by_key[key], rhs_by_key[key]) for key in rel_keys
        ]
        abs_errors = [
            abs(lhs_by_key[key] - rhs_by_key[key]) for key in common_keys
        ]
        rms = math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors))

        by_station: dict[str, dict[str, float]] = {}
        for station in sorted({key[1] for key in common_keys}):
            station_keys = [key for key in common_keys if key[1] == station]
            station_active = [
                key for key in station_keys if abs(rhs_by_key[key]) >= activity_floor
            ]
            station_rel_keys = station_active or station_keys
            station_rel_errors = [
                relative_error(lhs_by_key[key], rhs_by_key[key])
                for key in station_rel_keys
            ]
            by_station[str(station)] = {
                "shared_point_count": len(station_keys),
                "active_point_count": len(station_active),
                f"max_rel_{label}": max(station_rel_errors),
                f"rms_rel_{label}": math.sqrt(
                    sum(err * err for err in station_rel_errors)
                    / len(station_rel_errors)
                ),
            }

        return {
            "shared_point_count": len(common_keys),
            "active_point_count": len(active_keys),
            "activity_floor": activity_floor,
            f"max_abs_{label}": max(abs_errors),
            f"max_rel_{label}": max(rel_errors),
            f"rms_rel_{label}": rms,
            "by_station": by_station,
        }

    def compare_by_step_and_fiber_spatial(
        lhs_rows: list[dict[str, str]],
        rhs_rows: list[dict[str, str]],
        lhs_key: str,
        rhs_key: str,
        label: str,
        predicate=None,
    ) -> dict[str, object]:
        predicate = predicate or (lambda row: True)
        key_of = lambda row: (
            int(row["step"]),
            row["material_role"],
            row["zone"],
            round(float(row["y"]), 10),
            round(float(row["z"]), 10),
            round(float(row["area"]), 12),
        )
        lhs_by_key = {
            key_of(row): float(row[lhs_key]) for row in lhs_rows if predicate(row)
        }
        rhs_by_key = {
            key_of(row): float(row[rhs_key]) for row in rhs_rows if predicate(row)
        }
        common_keys = sorted(set(lhs_by_key) & set(rhs_by_key))
        if not common_keys:
            return {"shared_point_count": 0}

        reference_peak = max(abs(rhs_by_key[key]) for key in common_keys)
        activity_floor = max(5.0e-2 * reference_peak, 1.0e-12)
        active_keys = [
            key for key in common_keys if abs(rhs_by_key[key]) >= activity_floor
        ]
        rel_keys = active_keys or common_keys
        rel_errors = [
            relative_error(lhs_by_key[key], rhs_by_key[key]) for key in rel_keys
        ]
        abs_errors = [
            abs(lhs_by_key[key] - rhs_by_key[key]) for key in common_keys
        ]
        rms = math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors))
        return {
            "shared_point_count": len(common_keys),
            "active_point_count": len(active_keys),
            "activity_floor": activity_floor,
            f"max_abs_{label}": max(abs_errors),
            f"max_rel_{label}": max(rel_errors),
            f"rms_rel_{label}": rms,
        }

    def compare_by_step_and_station_fiber_spatial(
        lhs_rows: list[dict[str, str]],
        rhs_rows: list[dict[str, str]],
        lhs_key: str,
        rhs_key: str,
        label: str,
        predicate=None,
    ) -> dict[str, object]:
        predicate = predicate or (lambda row: True)
        key_of = lambda row: (
            int(row["step"]),
            int(row["section_gp"]),
            row["material_role"],
            row["zone"],
            round(float(row["y"]), 10),
            round(float(row["z"]), 10),
            round(float(row["area"]), 12),
        )
        lhs_by_key = {
            key_of(row): float(row[lhs_key]) for row in lhs_rows if predicate(row)
        }
        rhs_by_key = {
            key_of(row): float(row[rhs_key]) for row in rhs_rows if predicate(row)
        }
        common_keys = sorted(set(lhs_by_key) & set(rhs_by_key))
        if not common_keys:
            return {"shared_point_count": 0}

        reference_peak = max(abs(rhs_by_key[key]) for key in common_keys)
        activity_floor = max(5.0e-2 * reference_peak, 1.0e-12)
        active_keys = [
            key for key in common_keys if abs(rhs_by_key[key]) >= activity_floor
        ]
        rel_keys = active_keys or common_keys
        rel_errors = [
            relative_error(lhs_by_key[key], rhs_by_key[key]) for key in rel_keys
        ]
        abs_errors = [abs(lhs_by_key[key] - rhs_by_key[key]) for key in common_keys]
        rms = math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors))

        by_station: dict[str, object] = {}
        stations = sorted({key[1] for key in common_keys})
        for station in stations:
            station_keys = [key for key in common_keys if key[1] == station]
            station_active = [key for key in active_keys if key[1] == station]
            station_rel_keys = station_active or station_keys
            station_rel_errors = [
                relative_error(lhs_by_key[key], rhs_by_key[key])
                for key in station_rel_keys
            ]
            station_abs_errors = [
                abs(lhs_by_key[key] - rhs_by_key[key]) for key in station_keys
            ]
            by_station[str(station)] = {
                "shared_point_count": len(station_keys),
                "active_point_count": len(station_active),
                f"max_abs_{label}": max(station_abs_errors),
                f"max_rel_{label}": max(station_rel_errors),
                f"rms_rel_{label}": math.sqrt(
                    sum(err * err for err in station_rel_errors)
                    / len(station_rel_errors)
                ),
            }

        return {
            "shared_point_count": len(common_keys),
            "active_point_count": len(active_keys),
            "activity_floor": activity_floor,
            f"max_abs_{label}": max(abs_errors),
            f"max_rel_{label}": max(rel_errors),
            f"rms_rel_{label}": rms,
            "by_station": by_station,
        }

    def grouped_fiber_contributions(
        rows: list[dict[str, str]],
        predicate,
        key_fields: tuple[str, ...],
    ) -> dict[tuple[object, ...], dict[str, float]]:
        sums: dict[tuple[object, ...], dict[str, float]] = {}
        numeric_fields = (
            "axial_force_contribution_MN",
            "moment_y_contribution_MNm",
            "raw_k00_contribution",
            "raw_k0y_contribution",
            "raw_kyy_contribution",
        )
        for row in rows:
            if not predicate(row):
                continue
            key_parts: list[object] = []
            for field in key_fields:
                if field == "step":
                    key_parts.append(int(row[field]))
                else:
                    key_parts.append(row[field])
            key = tuple(key_parts)
            bucket = sums.setdefault(key, {field: 0.0 for field in numeric_fields})
            for field in numeric_fields:
                bucket[field] += float(row[field])
        return sums

    def condensed_tangent_from_entries(
        k00: float,
        k0y: float,
        ky0: float,
        kyy: float,
    ) -> float:
        if not math.isfinite(kyy):
            return math.nan
        if not math.isfinite(k00) or abs(k00) <= 1.0e-12:
            return kyy
        if not math.isfinite(k0y) or not math.isfinite(ky0):
            return kyy
        return kyy - k0y * ky0 / k00

    def write_grouped_fiber_summary_csv(
        path: Path,
        lhs_rows: list[dict[str, str]],
        rhs_rows: list[dict[str, str]],
        predicate,
        key_fields: tuple[str, ...],
    ) -> dict[str, object]:
        lhs_grouped = grouped_fiber_contributions(lhs_rows, predicate, key_fields)
        rhs_grouped = grouped_fiber_contributions(rhs_rows, predicate, key_fields)
        common_keys = sorted(set(lhs_grouped) & set(rhs_grouped))
        if not common_keys:
            return {"shared_group_count": 0}

        rows_to_write: list[dict[str, object]] = []
        max_condensed_rel_error = 0.0
        rms_terms: list[float] = []
        for key in common_keys:
            lhs = lhs_grouped[key]
            rhs = rhs_grouped[key]
            lhs_condensed = condensed_tangent_from_entries(
                lhs["raw_k00_contribution"],
                lhs["raw_k0y_contribution"],
                lhs["raw_k0y_contribution"],
                lhs["raw_kyy_contribution"],
            )
            rhs_condensed = condensed_tangent_from_entries(
                rhs["raw_k00_contribution"],
                rhs["raw_k0y_contribution"],
                rhs["raw_k0y_contribution"],
                rhs["raw_kyy_contribution"],
            )
            condensed_rel_error = relative_error(lhs_condensed, rhs_condensed)
            max_condensed_rel_error = max(max_condensed_rel_error, condensed_rel_error)
            rms_terms.append(condensed_rel_error * condensed_rel_error)

            row_payload = {
                field: value for field, value in zip(key_fields, key, strict=True)
            }
            row_payload.update(
                {
                    "lhs_axial_force_contribution_MN": lhs["axial_force_contribution_MN"],
                    "rhs_axial_force_contribution_MN": rhs["axial_force_contribution_MN"],
                    "lhs_moment_y_contribution_MNm": lhs["moment_y_contribution_MNm"],
                    "rhs_moment_y_contribution_MNm": rhs["moment_y_contribution_MNm"],
                    "lhs_raw_k00_contribution": lhs["raw_k00_contribution"],
                    "rhs_raw_k00_contribution": rhs["raw_k00_contribution"],
                    "lhs_raw_k0y_contribution": lhs["raw_k0y_contribution"],
                    "rhs_raw_k0y_contribution": rhs["raw_k0y_contribution"],
                    "lhs_raw_kyy_contribution": lhs["raw_kyy_contribution"],
                    "rhs_raw_kyy_contribution": rhs["raw_kyy_contribution"],
                    "lhs_condensed_tangent": lhs_condensed,
                    "rhs_condensed_tangent": rhs_condensed,
                    "rel_error_axial_force": relative_error(
                        lhs["axial_force_contribution_MN"],
                        rhs["axial_force_contribution_MN"],
                    ),
                    "rel_error_moment_y": relative_error(
                        lhs["moment_y_contribution_MNm"],
                        rhs["moment_y_contribution_MNm"],
                    ),
                    "rel_error_raw_k00": relative_error(
                        lhs["raw_k00_contribution"],
                        rhs["raw_k00_contribution"],
                    ),
                    "rel_error_raw_k0y": relative_error(
                        lhs["raw_k0y_contribution"],
                        rhs["raw_k0y_contribution"],
                    ),
                    "rel_error_raw_kyy": relative_error(
                        lhs["raw_kyy_contribution"],
                        rhs["raw_kyy_contribution"],
                    ),
                    "rel_error_condensed_tangent": condensed_rel_error,
                }
            )
            rows_to_write.append(row_payload)

        write_csv(
            path,
            (
                *key_fields,
                "lhs_axial_force_contribution_MN",
                "rhs_axial_force_contribution_MN",
                "lhs_moment_y_contribution_MNm",
                "rhs_moment_y_contribution_MNm",
                "lhs_raw_k00_contribution",
                "rhs_raw_k00_contribution",
                "lhs_raw_k0y_contribution",
                "rhs_raw_k0y_contribution",
                "lhs_raw_kyy_contribution",
                "rhs_raw_kyy_contribution",
                "lhs_condensed_tangent",
                "rhs_condensed_tangent",
                "rel_error_axial_force",
                "rel_error_moment_y",
                "rel_error_raw_k00",
                "rel_error_raw_k0y",
                "rel_error_raw_kyy",
                "rel_error_condensed_tangent",
            ),
            rows_to_write,
        )
        return {
            "shared_group_count": len(common_keys),
            "max_rel_condensed_tangent_error": max_condensed_rel_error,
            "rms_rel_condensed_tangent_error": math.sqrt(
                sum(rms_terms) / len(rms_terms)
            ),
            "csv": str(path),
        }

    def summarize_domain_time_trace(
        rows: list[dict[str, str]],
    ) -> dict[str, object]:
        if not rows:
            return {"shared_step_count": 0}
        domain_times = [float(row["domain_time_after"]) for row in rows]
        pseudo_times = [float(row["pseudo_time_after"]) for row in rows]
        domain_increments = [float(row["domain_time_increment"]) for row in rows]
        pseudo_increments = [float(row["pseudo_time_increment"]) for row in rows]
        return {
            "shared_step_count": len(rows),
            "domain_time_monotone": all(
                domain_times[idx] + 1.0e-14 >= domain_times[idx - 1]
                for idx in range(1, len(domain_times))
            ),
            "final_domain_time": domain_times[-1],
            "final_pseudo_time": pseudo_times[-1],
            "max_abs_domain_vs_pseudo_time": max(
                abs(domain_times[idx] - pseudo_times[idx])
                for idx in range(len(rows))
            ),
            "max_abs_increment_vs_pseudo_increment": max(
                abs(domain_increments[idx] - pseudo_increments[idx])
                for idx in range(len(rows))
            ),
            "max_accepted_substep_count": max(
                int(row["accepted_substep_count"]) for row in rows
            ),
            "max_bisection_level": max(
                int(row["max_bisection_level"]) for row in rows
            ),
        }

    if model_kind == "structural" and falln_hysteresis:
        summary["hysteresis"] = compare_by_step(
            read_csv_rows(out_dir / "hysteresis.csv"),
            read_csv_rows(falln_hysteresis),
            "base_shear_MN",
            "base_shear_MN",
            "base_shear_error",
        )

    if model_kind == "structural" and falln_mk:
        summary["moment_curvature_base"] = compare_by_step(
            read_csv_rows(out_dir / "moment_curvature_base.csv"),
            read_csv_rows(falln_mk),
            "moment_y_MNm",
            "moment_y_MNm",
            "moment_error",
        )

    if model_kind == "structural" and falln_section_response:
        lhs_section_rows = read_csv_rows(out_dir / "section_response.csv")
        rhs_section_rows = read_csv_rows(falln_section_response)
        summary["section_response_moment"] = compare_by_step_and_station(
            lhs_section_rows,
            rhs_section_rows,
            "moment_y_MNm",
            "moment_y_MNm",
            "moment_error",
        )
        summary["section_response_curvature"] = compare_by_step_and_station(
            lhs_section_rows,
            rhs_section_rows,
            "curvature_y",
            "curvature_y",
            "curvature_error",
        )
        summary["section_response_axial_force"] = compare_by_step_and_station(
            lhs_section_rows,
            rhs_section_rows,
            "axial_force_MN",
            "axial_force_MN",
            "axial_force_error",
        )
        summary["section_response_tangent"] = compare_by_step_and_station(
            lhs_section_rows,
            rhs_section_rows,
            "tangent_eiy",
            "tangent_eiy",
            "tangent_error",
        )
        summary["section_response_tangent_direct_raw"] = compare_by_step_and_station(
            lhs_section_rows,
            rhs_section_rows,
            "tangent_eiy_direct_raw",
            "tangent_eiy_direct_raw",
            "tangent_direct_raw_error",
        )
        summary["section_response_tangent_condensation_gap"] = compare_derived_by_step_and_station(
            lhs_section_rows,
            rhs_section_rows,
            lambda row: float(row["tangent_eiy_direct_raw"]) - float(row["tangent_eiy"]),
            lambda row: float(row["tangent_eiy_direct_raw"]) - float(row["tangent_eiy"]),
            "tangent_condensation_gap_error",
        )
        summary["section_response_raw_k00"] = compare_by_step_and_station(
            lhs_section_rows,
            rhs_section_rows,
            "raw_k00",
            "raw_k00",
            "raw_k00_error",
        )
        summary["section_response_raw_k0y"] = compare_by_step_and_station(
            lhs_section_rows,
            rhs_section_rows,
            "raw_k0y",
            "raw_k0y",
            "raw_k0y_error",
        )

    if model_kind == "structural" and falln_control_state:
        lhs_control_rows = read_csv_rows(out_dir / "control_state.csv")
        rhs_control_rows = read_csv_rows(falln_control_state)
        summary["control_state_tip_drift"] = compare_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "actual_tip_drift_m",
            "actual_tip_drift_m",
            "tip_drift_error",
        )
        summary["control_state_tip_axial_displacement"] = compare_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "top_axial_displacement_m",
            "top_axial_displacement_m",
            "tip_axial_displacement_error",
        )
        summary["control_state_base_axial_reaction"] = compare_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "base_axial_reaction_MN",
            "base_axial_reaction_MN",
            "base_axial_reaction_error",
        )

    if model_kind == "section" and falln_section_baseline:
        lhs_rows = read_csv_rows(out_dir / "section_moment_curvature_baseline.csv")
        rhs_rows = read_csv_rows(falln_section_baseline)
        summary["section_moment_curvature"] = compare_by_step(
            lhs_rows,
            rhs_rows,
            "moment_y_MNm",
            "moment_y_MNm",
            "moment_error",
        )
        summary["section_tangent"] = compare_by_step(
            lhs_rows,
            rhs_rows,
            "tangent_eiy",
            "tangent_eiy",
            "tangent_error",
        )
        summary["section_tangent_direct_raw"] = compare_by_step(
            lhs_rows,
            rhs_rows,
            "tangent_eiy_direct_raw",
            "tangent_eiy_direct_raw",
            "tangent_direct_raw_error",
        )
        summary["section_tangent_condensation_gap"] = compare_derived_by_step(
            lhs_rows,
            rhs_rows,
            lambda row: float(row["tangent_eiy_direct_raw"]) - float(row["tangent_eiy"]),
            lambda row: float(row["tangent_eiy_direct_raw"]) - float(row["tangent_eiy"]),
            "tangent_condensation_gap_error",
        )
        zero_curvature = lambda row: abs(float(row["curvature_y"])) <= 1.0e-12
        summary["section_tangent_branch_only"] = compare_by_step_subset(
            lhs_rows,
            rhs_rows,
            "tangent_eiy",
            "tangent_eiy",
            "tangent_error",
            lambda row: not zero_curvature(row),
        )
        summary["section_tangent_zero_curvature_anchor_only"] = compare_by_step_subset(
            lhs_rows,
            rhs_rows,
            "tangent_eiy",
            "tangent_eiy",
            "tangent_error",
            zero_curvature,
        )
        summary["section_axial_force"] = compare_by_step(
            lhs_rows,
            rhs_rows,
            "axial_force_MN",
            "axial_force_MN",
            "axial_force_error",
        )
        summary["section_tangent_consistency_fall_n"] = compare_direct_vs_numerical(
            numerical_tangent_rows(rhs_rows, "curvature_y", "moment_y_MNm")
        )
        summary["section_tangent_consistency_opensees"] = compare_direct_vs_numerical(
            numerical_tangent_rows(lhs_rows, "curvature_y", "moment_y_MNm")
        )

    if model_kind == "section" and falln_section_control_trace:
        lhs_control_rows = read_csv_rows(out_dir / "section_control_trace.csv")
        rhs_control_rows = read_csv_rows(falln_section_control_trace)
        summary["section_control_target_curvature"] = compare_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "target_curvature_y",
            "target_curvature_y",
            "target_curvature_error",
        )
        summary["section_control_actual_curvature"] = compare_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "actual_curvature_y",
            "actual_curvature_y",
            "actual_curvature_error",
        )
        summary["section_control_delta_actual_curvature"] = compare_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "delta_actual_curvature_y",
            "delta_actual_curvature_y",
            "delta_actual_curvature_error",
        )
        summary["section_control_pseudotime"] = compare_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "pseudo_time_after",
            "pseudo_time_after",
            "pseudo_time_error",
        )
        summary["section_control_newton_iterations"] = compare_integer_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "newton_iterations",
            "newton_iterations",
            "newton_iteration_error",
        )
        summary["section_control_newton_iterations_per_substep"] = compare_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "newton_iterations_per_substep",
            "newton_iterations_per_substep",
            "newton_iterations_per_substep_error",
        )
        summary["section_control_accepted_substeps"] = compare_integer_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "accepted_substep_count",
            "accepted_substep_count",
            "accepted_substep_error",
        )
        summary["section_control_bisection_level"] = compare_integer_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "max_bisection_level",
            "max_bisection_level",
            "bisection_level_error",
        )
        summary["section_control_target_increment_direction"] = compare_integer_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "target_increment_direction",
            "target_increment_direction",
            "target_increment_direction_error",
        )
        summary["section_control_actual_increment_direction"] = compare_integer_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "actual_increment_direction",
            "actual_increment_direction",
            "actual_increment_direction_error",
        )
        summary["section_control_protocol_branch_id"] = compare_integer_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "protocol_branch_id",
            "protocol_branch_id",
            "protocol_branch_id_error",
        )
        summary["section_control_reversal_index"] = compare_integer_by_step(
            lhs_control_rows,
            rhs_control_rows,
            "reversal_index",
            "reversal_index",
            "reversal_index_error",
        )
        summary["section_control_domain_time_opensees"] = summarize_domain_time_trace(
            lhs_control_rows
        )

    if model_kind == "section" and falln_section_fiber_history:
        lhs_fiber_rows = read_csv_rows(out_dir / "section_fiber_state_history.csv")
        rhs_fiber_rows = read_csv_rows(falln_section_fiber_history)
        zero_curvature = lambda row: int(row["zero_curvature_anchor"]) == 1
        summary["section_fiber_stress"] = compare_by_step_and_fiber_spatial(
            lhs_fiber_rows,
            rhs_fiber_rows,
            "stress_xx_MPa",
            "stress_xx_MPa",
            "fiber_stress_error",
        )
        summary["section_fiber_tangent"] = compare_by_step_and_fiber_spatial(
            lhs_fiber_rows,
            rhs_fiber_rows,
            "tangent_xx_MPa",
            "tangent_xx_MPa",
            "fiber_tangent_error",
        )
        summary["section_fiber_stress_zero_curvature_anchor_only"] = compare_by_step_and_fiber_spatial(
            lhs_fiber_rows,
            rhs_fiber_rows,
            "stress_xx_MPa",
            "stress_xx_MPa",
            "fiber_stress_error",
            zero_curvature,
        )
        summary["section_fiber_tangent_zero_curvature_anchor_only"] = compare_by_step_and_fiber_spatial(
            lhs_fiber_rows,
            rhs_fiber_rows,
            "tangent_xx_MPa",
            "tangent_xx_MPa",
            "fiber_tangent_error",
            zero_curvature,
        )
        summary["section_fiber_anchor_total_summary"] = write_grouped_fiber_summary_csv(
            out_dir / "section_fiber_anchor_total_summary.csv",
            lhs_fiber_rows,
            rhs_fiber_rows,
            zero_curvature,
            ("step",),
        )
        summary["section_fiber_anchor_material_role_summary"] = write_grouped_fiber_summary_csv(
            out_dir / "section_fiber_anchor_material_role_summary.csv",
            lhs_fiber_rows,
            rhs_fiber_rows,
            zero_curvature,
            ("step", "material_role"),
        )
        summary["section_fiber_anchor_zone_summary"] = write_grouped_fiber_summary_csv(
            out_dir / "section_fiber_anchor_zone_summary.csv",
            lhs_fiber_rows,
            rhs_fiber_rows,
            zero_curvature,
            ("step", "zone"),
        )

    if model_kind == "structural" and falln_section_fiber_history:
        lhs_fiber_rows = read_csv_rows(out_dir / "section_fiber_state_history.csv")
        rhs_fiber_rows = read_csv_rows(falln_section_fiber_history)
        zero_curvature = lambda row: int(row["zero_curvature_anchor"]) == 1
        summary["structural_section_fiber_stress"] = (
            compare_by_step_and_station_fiber_spatial(
                lhs_fiber_rows,
                rhs_fiber_rows,
                "stress_xx_MPa",
                "stress_xx_MPa",
                "fiber_stress_error",
            )
        )
        summary["structural_section_fiber_tangent"] = (
            compare_by_step_and_station_fiber_spatial(
                lhs_fiber_rows,
                rhs_fiber_rows,
                "tangent_xx_MPa",
                "tangent_xx_MPa",
                "fiber_tangent_error",
            )
        )
        summary["structural_section_fiber_stress_zero_curvature_anchor_only"] = (
            compare_by_step_and_station_fiber_spatial(
                lhs_fiber_rows,
                rhs_fiber_rows,
                "stress_xx_MPa",
                "stress_xx_MPa",
                "fiber_stress_error",
                zero_curvature,
            )
        )
        summary["structural_section_fiber_tangent_zero_curvature_anchor_only"] = (
            compare_by_step_and_station_fiber_spatial(
                lhs_fiber_rows,
                rhs_fiber_rows,
                "tangent_xx_MPa",
                "tangent_xx_MPa",
                "fiber_tangent_error",
                zero_curvature,
            )
        )
        summary["structural_section_fiber_anchor_station_zone_summary"] = (
            write_grouped_fiber_summary_csv(
                out_dir / "structural_section_fiber_anchor_station_zone_summary.csv",
                lhs_fiber_rows,
                rhs_fiber_rows,
                zero_curvature,
                ("step", "section_gp", "zone"),
            )
        )

    summary["spatial_parity"] = spatial_parity_summary(
        falln_section_layout,
        falln_station_layout,
        read_csv_rows(out_dir / "section_layout.csv"),
        read_csv_rows(out_dir / "section_station_layout.csv"),
    )

    (out_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    spec = ReducedRCColumnReferenceSpec()
    mapping_policy = selected_mapping_policy(args)
    protocol = (
        build_protocol_points(args)
        if args.model_kind == "structural"
        else build_section_protocol(args)
    )
    manifest = make_manifest(args, spec, protocol, mapping_policy)

    ensure_dir(out_dir)
    if args.dry_run:
        manifest["dependency_status"] = "dry_run_without_openseespy"
        manifest["timing"] = {
            "total_wall_seconds": 0.0,
            "analysis_wall_seconds": 0.0,
            "output_write_wall_seconds": 0.0,
        }
        section_layout_rows = build_section_layout_rows(spec)
        (out_dir / "reference_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if args.model_kind == "structural":
            write_csv(
                out_dir / "protocol.csv",
                ("step", "p", "target_drift_m"),
                ({"step": p.step, "p": p.p, "target_drift_m": p.target_drift_m} for p in protocol),
            )
            write_structural_csv_contracts_only(out_dir)
            write_csv(
                out_dir / "section_layout.csv",
                ("fiber_index", "y", "z", "area", "zone", "material_role", "material_tag"),
                section_layout_rows,
            )
            write_csv(
                out_dir / "section_station_layout.csv",
                ("section_gp", "xi"),
                build_station_layout_rows(
                    beam_integration_nodes(
                        args.beam_integration, args.integration_points
                    )
                ),
            )
        else:
            write_csv(
                out_dir / "protocol.csv",
                ("step", "p", "target_curvature_y"),
                ({"step": p.step, "p": p.p, "target_curvature_y": p.target_drift_m} for p in protocol),
            )
            write_section_csv_contracts_only(out_dir)
            write_csv(
                out_dir / "section_layout.csv",
                ("fiber_index", "y", "z", "area", "zone", "material_role", "material_tag"),
                section_layout_rows,
            )
            write_csv(
                out_dir / "section_station_layout.csv",
                ("section_gp", "xi"),
                build_station_layout_rows((0.0,)),
            )
        return 0

    total_start = time.perf_counter()
    analysis_start: float | None = None
    try:
        ops = try_import_opensees()
        analysis_start = time.perf_counter()
        structural_result = (
            run_analysis(ops, args, spec, protocol, mapping_policy)
            if args.model_kind == "structural"
            else None
        )
        section_result = (
            None
            if args.model_kind == "structural"
            else run_section_analysis(ops, args, spec, mapping_policy)
        )
        analysis_elapsed = time.perf_counter() - analysis_start
    except Exception as exc:
        manifest["status"] = "failed"
        annotate_failure_manifest(manifest, str(exc))
        output_elapsed = 0.0
        analysis_elapsed = (
            time.perf_counter() - analysis_start if analysis_start is not None else 0.0
        )
        section_layout_rows = build_section_layout_rows(spec)
        if (
            args.model_kind == "section"
            and isinstance(exc, OpenSeesReferenceAnalysisFailure)
            and exc.partial_section_state is not None
        ):
            partial_state = exc.partial_section_state
            manifest["section_baseline_record_count"] = len(partial_state.rows)
            manifest["section_fiber_record_count"] = len(partial_state.fiber_rows)
            manifest["section_control_trace_record_count"] = len(
                partial_state.control_trace_rows
            )
            manifest["section_layout_fiber_count"] = len(section_layout_rows)
            manifest["section_station_count"] = 1
            if partial_state.failure_trial is not None:
                manifest["failure_trial_recorded"] = True
                manifest["failure_trial_actual_curvature_y"] = (
                    partial_state.failure_trial.control_trace_row.actual_curvature_y
                )
                manifest["failure_trial_control_increment_error"] = (
                    partial_state.failure_trial.control_trace_row.control_increment_error
                )
                manifest["failure_trial_control_direction_admissible"] = (
                    partial_state.failure_trial.control_trace_row.control_direction_admissible
                )
                manifest["failure_trial_control_magnitude_admissible"] = (
                    partial_state.failure_trial.control_trace_row.control_magnitude_admissible
                )
            output_start = time.perf_counter()
            write_section_outputs(
                out_dir,
                manifest,
                protocol,
                list(partial_state.rows),
                list(partial_state.fiber_rows),
                list(partial_state.control_trace_rows),
                section_layout_rows,
            )
            if partial_state.failure_trial is not None:
                write_section_failure_trial_outputs(out_dir, partial_state.failure_trial)
            maybe_write_comparison(
                out_dir,
                args.falln_hysteresis,
                args.falln_moment_curvature,
                args.falln_section_response,
                args.falln_control_state,
                args.falln_section_layout,
                args.falln_station_layout,
                args.falln_section_baseline,
                args.falln_section_fiber_history,
                args.falln_section_control_trace,
                args.model_kind,
                args.beam_element_family,
            )
            output_elapsed = time.perf_counter() - output_start
        manifest["timing"] = {
            "total_wall_seconds": time.perf_counter() - total_start,
            "analysis_wall_seconds": analysis_elapsed,
            "output_write_wall_seconds": output_elapsed,
        }
        (out_dir / "reference_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        raise

    manifest["status"] = "completed"
    section_layout_rows = build_section_layout_rows(spec)
    if args.model_kind == "structural":
        assert structural_result is not None
        hysteresis, sections, control_states, structural_fiber_rows = structural_result
        manifest["hysteresis_point_count"] = len(hysteresis)
        manifest["section_record_count"] = len(sections)
        manifest["control_state_record_count"] = len(control_states)
        manifest["section_fiber_record_count"] = len(structural_fiber_rows)
        station_layout_rows = build_station_layout_rows(
            tuple(row.xi for row in sections[: args.integration_points])
        )
    else:
        assert section_result is not None
        section_rows, section_fiber_rows, section_control_trace_rows = section_result
        manifest["section_baseline_record_count"] = len(section_rows)
        manifest["section_fiber_record_count"] = len(section_fiber_rows)
        manifest["section_control_trace_record_count"] = len(section_control_trace_rows)
        station_layout_rows = build_station_layout_rows((0.0,))

    manifest["section_layout_fiber_count"] = len(section_layout_rows)
    manifest["section_station_count"] = len(station_layout_rows)

    output_start = time.perf_counter()
    if args.model_kind == "structural":
        write_structural_outputs(
            out_dir,
            manifest,
            protocol,
            hysteresis,
            sections,
            control_states,
            structural_fiber_rows,
            section_layout_rows,
            station_layout_rows,
        )
    else:
        write_section_outputs(
            out_dir,
            manifest,
            protocol,
            section_rows,
            section_fiber_rows,
            section_control_trace_rows,
            section_layout_rows,
        )
    maybe_write_comparison(
        out_dir,
        args.falln_hysteresis,
        args.falln_moment_curvature,
        args.falln_section_response,
        args.falln_control_state,
        args.falln_section_layout,
        args.falln_station_layout,
        args.falln_section_baseline,
        args.falln_section_fiber_history,
        args.falln_section_control_trace,
        args.model_kind,
        args.beam_element_family,
    )
    output_elapsed = time.perf_counter() - output_start
    manifest["timing"] = {
        "total_wall_seconds": time.perf_counter() - total_start,
        "analysis_wall_seconds": analysis_elapsed,
        "output_write_wall_seconds": output_elapsed,
    }
    (out_dir / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
