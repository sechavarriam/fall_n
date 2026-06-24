#!/usr/bin/env python3
"""
OpenSeesPy 3D RC-column Hi-Fi cyclic reference, written as a teaching script.

This file reproduces the successful 200 mm OpenSees reference used in the
fall_n reduced-RC-column validation campaign, but keeps the implementation
plain and readable for undergraduate Civil Engineering classes.

The goal is not to hide the details behind framework code. The script shows:

1. Geometry and section discretization.
2. Concrete and steel constitutive models.
3. Multi-element 3D beam-column discretization.
4. Axial preload.
5. Cyclic displacement-control loading up to 200 mm.
6. Newton/Secant-Newton convergence settings and cutback logic.
7. CSV and figure outputs for base-shear hysteresis and base response.

Run from the repository root with:

    py -3.12 scripts/opensees_reduced_rc_column_hifi_teaching.py

Dependencies:

    py -3.12 -m pip install openseespy matplotlib

Important units:

    OpenSees receives SI units: N, m, Pa.
    Output CSV files report MN, MPa, m, and dimensionless strains.

Replace the constants in the block below to adapt the example to another
column, reinforcement layout, material calibration, or loading protocol.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


# =============================================================================
# 1. User-editable parameters
# =============================================================================

# Output directory. Change this path for a clean new run.
OUT_DIR = Path(
    "data/output/cyclic_validation/opensees_hifi_teaching_3d_200mm"
)

# Geometry. This is the validated reduced-column reference.
L_COLUMN = 3.20          # column height [m]
SECTION_B = 0.25        # section width in fiber y direction [m]
SECTION_H = 0.25        # section depth in fiber z direction [m]
COVER = 0.03            # cover to bar centers used by the reference [m]
BAR_DIAMETER = 0.016    # longitudinal bar diameter [m]

# Material parameters. Replace these values with calibrated data if available.
FPC_MPA = 30.0          # concrete compressive strength magnitude [MPa]
NU_CONCRETE = 0.20
STEEL_E_MPA = 200_000.0
STEEL_FY_MPA = 420.0
STEEL_B = 0.01
TIE_FY_MPA = 420.0
RHO_S = 0.015           # transverse confinement ratio used by the audit model

# Cyclic OpenSees mapping used by the successful hi-fi reference.
# These numbers are intentionally explicit because they are part of the audit.
CONCRETE_LAMBDA = 0.10
CONCRETE_FT_RATIO = 0.02
CONCRETE_SOFTENING_MULTIPLIER = 0.50
CONCRETE_UNCONFINED_RESIDUAL_RATIO = 0.20
CONCRETE_CONFINED_RESIDUAL_RATIO = 0.20
CONCRETE_ULTIMATE_STRAIN = -0.006

STEEL_R0 = 20.0
STEEL_CR1 = 0.925
STEEL_CR2 = 0.15
STEEL_A1 = 0.0
STEEL_A2 = 1.0
STEEL_A3 = 0.0
STEEL_A4 = 1.0

# Structural discretization. This is the successful publication comparator.
N_ELEMENTS = 20
BEAM_INTEGRATION = "Legendre"
N_INTEGRATION_POINTS = 5
GEOMETRIC_TRANSFORMATION = "PDelta"

# 3D axes used by this teaching model:
#   global Z = column axis and gravity/preload direction
#   global X = imposed cyclic lateral displacement direction
#   global Y = out-of-plane direction, free at the top
LATERAL_DOF = 1
OUT_OF_PLANE_DOF = 2
AXIAL_DOF = 3

# Constant axial compression. Positive value means compression magnitude.
AXIAL_COMPRESSION_MN = 0.02
AXIAL_PRELOAD_STEPS = 4

# Cyclic displacement protocol. Each amplitude follows 0 -> +A -> -A -> 0.
AMPLITUDES_MM = [50.0, 100.0, 150.0, 200.0]
STEPS_PER_SEGMENT = 32
REVERSAL_SUBSTEP_FACTOR = 2

# Nonlinear solution settings. This is the "pure-secant-disp" profile used by
# the successful artifact. If a step fails, the target increment is bisected.
NEWTON_TEST = "NormDispIncr"
NEWTON_TOL = 1.0e-8
NEWTON_MAX_ITER = 160
NEWTON_ALGORITHM = "SecantNewton"
MAX_BISECTIONS = 10


# =============================================================================
# 2. Small utilities
# =============================================================================

def mpa_to_pa(value_mpa: float) -> float:
    return value_mpa * 1.0e6


def bar_area(diameter_m: float) -> float:
    return math.pi * diameter_m**2 / 4.0


def concrete_initial_modulus_mpa(fpc_mpa: float) -> float:
    # This deliberately matches the validated fall_n/OpenSees audit bridge.
    # If using a design-code expression such as 4700*sqrt(f'c), replace here.
    return 1000.0 * fpc_mpa


def shear_modulus_pa(elastic_modulus_pa: float, poisson: float) -> float:
    return elastic_modulus_pa / (2.0 * (1.0 + poisson))


def rectangular_torsion_constant_m4(width_m: float, height_m: float) -> float:
    # Saint-Venant torsion approximation for a rectangular section.
    b_min = min(width_m, height_m)
    h_max = max(width_m, height_m)
    return (b_min**3 * h_max / 3.0) * (1.0 - 0.63 * b_min / h_max)


def mander_confined_strength_mpa(fpc_mpa: float, rho_s: float, fyh_mpa: float) -> float:
    # Compact form used by the audit model to obtain the confined-core peak.
    lateral_pressure = rho_s * fyh_mpa
    ratio = max(lateral_pressure / max(fpc_mpa, 1.0e-12), 0.0)
    factor = 2.254 * math.sqrt(1.0 + 7.94 * ratio) - 2.0 * ratio - 1.254
    return max(fpc_mpa * factor, fpc_mpa)


def cyclic_displacement_m(pseudo_time: float, amplitudes_mm: list[float]) -> float:
    """Triangular cyclic protocol 0 -> +A -> -A -> 0 per amplitude."""
    amplitudes_m = [amp * 1.0e-3 for amp in amplitudes_mm]
    n_segments = 3 * len(amplitudes_m)
    t = max(0.0, min(1.0, pseudo_time)) * n_segments
    segment = min(max(int(t), 0), n_segments - 1)
    frac = t - float(segment)
    level = segment // 3
    phase = segment % 3
    amp = amplitudes_m[level]

    if phase == 0:
        return frac * amp
    if phase == 1:
        return amp * (1.0 - 2.0 * frac)
    return -amp * (1.0 - frac)


def build_protocol() -> list[dict[str, float]]:
    segment_steps = STEPS_PER_SEGMENT * REVERSAL_SUBSTEP_FACTOR
    total_steps = 3 * len(AMPLITUDES_MM) * segment_steps
    protocol = [{"step": 0, "p": 0.0, "target_drift_m": 0.0}]
    for step in range(1, total_steps + 1):
        p = step / total_steps
        protocol.append(
            {
                "step": step,
                "p": p,
                "target_drift_m": cyclic_displacement_m(p, AMPLITUDES_MM),
            }
        )
    return protocol


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# 3. OpenSees model definition
# =============================================================================

def import_opensees():
    try:
        import openseespy.opensees as ops
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenSeesPy is not installed. Try `py -3.12 -m pip install openseespy`."
        ) from exc
    return ops


def define_materials(ops) -> dict[str, int]:
    """Create unconfined concrete, confined concrete, steel, and shear materials."""
    unconfined = 1
    confined = 2
    steel = 3
    shear_y = 4
    shear_z = 5

    fpc_pa = mpa_to_pa(FPC_MPA)
    ec_pa = mpa_to_pa(concrete_initial_modulus_mpa(FPC_MPA))
    gc_pa = shear_modulus_pa(ec_pa, NU_CONCRETE)

    # Concrete02 arguments:
    # tag, fpc, epsc0, fpcu, epsU, lambda, ft, Ets
    # Compression is negative in OpenSees.
    ft_pa = mpa_to_pa(CONCRETE_FT_RATIO * FPC_MPA)
    cracking_strain = (CONCRETE_FT_RATIO * FPC_MPA) / max(
        concrete_initial_modulus_mpa(FPC_MPA), 1.0e-12
    )
    softening_span = max(CONCRETE_SOFTENING_MULTIPLIER * cracking_strain, 1.0e-5)
    ets_pa = ft_pa / softening_span

    ops.uniaxialMaterial(
        "Concrete02",
        unconfined,
        -fpc_pa,
        -0.002,
        -CONCRETE_UNCONFINED_RESIDUAL_RATIO * fpc_pa,
        CONCRETE_ULTIMATE_STRAIN,
        CONCRETE_LAMBDA,
        ft_pa,
        ets_pa,
    )

    confined_fpc_mpa = mander_confined_strength_mpa(FPC_MPA, RHO_S, TIE_FY_MPA)
    confined_epsc0 = -0.002 * (1.0 + 5.0 * max(confined_fpc_mpa / FPC_MPA - 1.0, 0.0))
    confined_epsu = min(CONCRETE_ULTIMATE_STRAIN, 5.0 * confined_epsc0)
    ops.uniaxialMaterial(
        "Concrete02",
        confined,
        -mpa_to_pa(confined_fpc_mpa),
        confined_epsc0,
        -CONCRETE_CONFINED_RESIDUAL_RATIO * mpa_to_pa(confined_fpc_mpa),
        confined_epsu,
        CONCRETE_LAMBDA,
        ft_pa,
        ets_pa,
    )

    # Steel02 is Menegotto-Pinto steel in OpenSees.
    ops.uniaxialMaterial(
        "Steel02",
        steel,
        mpa_to_pa(STEEL_FY_MPA),
        mpa_to_pa(STEEL_E_MPA),
        STEEL_B,
        STEEL_R0,
        STEEL_CR1,
        STEEL_CR2,
        STEEL_A1,
        STEEL_A2,
        STEEL_A3,
        STEEL_A4,
    )

    # Elastic shear springs aggregated to the 3D fiber section.
    shear_area = (5.0 / 6.0) * gc_pa * SECTION_B * SECTION_H
    ops.uniaxialMaterial("Elastic", shear_y, shear_area)
    ops.uniaxialMaterial("Elastic", shear_z, shear_area)

    return {
        "unconfined": unconfined,
        "confined": confined,
        "steel": steel,
        "shear_y": shear_y,
        "shear_z": shear_z,
        "gc_pa": gc_pa,
    }


def add_rectangular_patch(
    ops,
    material_tag: int,
    ny: int,
    nz: int,
    y_min: float,
    z_min: float,
    y_max: float,
    z_max: float,
) -> None:
    ops.patch("rect", material_tag, ny, nz, y_min, z_min, y_max, z_max)


def define_fiber_section(ops, sec_tag: int, material_tags: dict[str, int]) -> list[dict[str, object]]:
    """Define the 3D RC fiber section and return a table of fiber locations."""
    fiber_sec_tag = sec_tag + 1000
    gj = material_tags["gc_pa"] * rectangular_torsion_constant_m4(SECTION_B, SECTION_H)
    ops.section("Fiber", fiber_sec_tag, "-GJ", gj)

    y_edge = SECTION_B / 2.0
    z_edge = SECTION_H / 2.0
    y_core = y_edge - COVER
    z_core = z_edge - COVER

    # Concrete patch discretization. The patch counts are part of the validated
    # comparator. Increase them for a finer section study.
    patches = [
        # y_min, y_max, ny, z_min, z_max, nz, material, name
        (-y_edge, y_edge, 8, -z_edge, -z_core, 2, material_tags["unconfined"], "cover_bottom"),
        (-y_edge, y_edge, 8, z_core, z_edge, 2, material_tags["unconfined"], "cover_top"),
        (-y_edge, -y_core, 2, -z_core, z_core, 4, material_tags["unconfined"], "cover_left"),
        (y_core, y_edge, 2, -z_core, z_core, 4, material_tags["unconfined"], "cover_right"),
        (-y_core, y_core, 6, -z_core, z_core, 6, material_tags["confined"], "confined_core"),
    ]

    section_layout: list[dict[str, object]] = []
    for y_min, y_max, ny, z_min, z_max, nz, mat, zone in patches:
        add_rectangular_patch(ops, mat, ny, nz, y_min, z_min, y_max, z_max)
        dy = (y_max - y_min) / ny
        dz = (z_max - z_min) / nz
        for iy in range(ny):
            for iz in range(nz):
                section_layout.append(
                    {
                        "fiber_index": len(section_layout),
                        "y": y_min + (iy + 0.5) * dy,
                        "z": z_min + (iz + 0.5) * dz,
                        "area": dy * dz,
                        "material_role": (
                            "confined_concrete"
                            if mat == material_tags["confined"]
                            else "unconfined_concrete"
                        ),
                        "zone": zone,
                        "material_tag": mat,
                    }
                )

    # Eight longitudinal bars. Replace this list for a different cage layout.
    y_bar = y_edge - COVER
    z_bar = z_edge - COVER
    bars = [
        (-y_bar, -z_bar),
        (y_bar, -z_bar),
        (-y_bar, z_bar),
        (y_bar, z_bar),
        (0.0, -z_bar),
        (0.0, z_bar),
        (-y_bar, 0.0),
        (y_bar, 0.0),
    ]
    area_bar = bar_area(BAR_DIAMETER)

    # Layers keep the script close to how a conventional OpenSees section is
    # usually taught. Single-bar layers are used for the mid-side bars.
    bottom_face = [bars[i] for i in (0, 4, 1)]
    top_face = [bars[i] for i in (2, 5, 3)]
    left_mid = bars[6]
    right_mid = bars[7]
    ops.layer(
        "straight",
        material_tags["steel"],
        len(bottom_face),
        area_bar,
        bottom_face[0][0],
        bottom_face[0][1],
        bottom_face[-1][0],
        bottom_face[-1][1],
    )
    ops.layer(
        "straight",
        material_tags["steel"],
        len(top_face),
        area_bar,
        top_face[0][0],
        top_face[0][1],
        top_face[-1][0],
        top_face[-1][1],
    )
    ops.layer("straight", material_tags["steel"], 1, area_bar, left_mid[0], left_mid[1], left_mid[0], left_mid[1])
    ops.layer("straight", material_tags["steel"], 1, area_bar, right_mid[0], right_mid[1], right_mid[0], right_mid[1])

    for y, z in bars:
        section_layout.append(
            {
                "fiber_index": len(section_layout),
                "y": y,
                "z": z,
                "area": area_bar,
                "material_role": "reinforcing_steel",
                "zone": "longitudinal_steel",
                "material_tag": material_tags["steel"],
            }
        )

    # Add both elastic shear directions to the 3D fiber section. The fiber
    # section already supplies axial force, biaxial bending and torsion.
    ops.section(
        "Aggregator",
        sec_tag,
        material_tags["shear_y"],
        "Vy",
        material_tags["shear_z"],
        "Vz",
        "-section",
        fiber_sec_tag,
    )
    return section_layout


def build_model(ops) -> tuple[int, int, list[int], list[dict[str, object]]]:
    """Create the 3D cantilever model and return base/top nodes and elements."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    # Nodes along the column height.
    # Translational DOFs are 1=X, 2=Y, 3=Z. Rotations are 4, 5, 6.
    node_tags = []
    for i in range(N_ELEMENTS + 1):
        tag = i + 1
        z = L_COLUMN * i / N_ELEMENTS
        ops.node(tag, 0.0, 0.0, z)
        node_tags.append(tag)

    base_node = node_tags[0]
    top_node = node_tags[-1]
    ops.fix(base_node, 1, 1, 1, 1, 1, 1)

    material_tags = define_materials(ops)
    section_layout = define_fiber_section(ops, sec_tag=10, material_tags=material_tags)

    transf_tag = 1
    integ_tag = 20
    # The element local x-axis follows the column from base to top. The vector
    # (1,0,0) places global X in the local x-z plane, so global-X drift is the
    # in-plane bending direction used by the fall_n comparison.
    ops.geomTransf(GEOMETRIC_TRANSFORMATION, transf_tag, 1.0, 0.0, 0.0)
    ops.beamIntegration(BEAM_INTEGRATION, integ_tag, 10, N_INTEGRATION_POINTS)

    element_tags = []
    for i in range(N_ELEMENTS):
        tag = 100 + i
        ops.element(
            "dispBeamColumn",
            tag,
            node_tags[i],
            node_tags[i + 1],
            transf_tag,
            integ_tag,
        )
        element_tags.append(tag)

    return base_node, top_node, element_tags, section_layout


# =============================================================================
# 4. Nonlinear solver configuration and step cutback
# =============================================================================

def configure_static_solver(ops) -> None:
    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test(NEWTON_TEST, NEWTON_TOL, NEWTON_MAX_ITER, 0)
    ops.algorithm(NEWTON_ALGORITHM)


def analyze_displacement_increment(ops, top_node: int, delta: float, depth: int = 0) -> dict[str, object]:
    """Try one displacement increment. If it fails, bisect recursively."""
    configure_static_solver(ops)
    before = float(ops.nodeDisp(top_node, LATERAL_DOF))
    ops.integrator("DisplacementControl", top_node, LATERAL_DOF, delta)
    ops.analysis("Static")
    ok = ops.analyze(1)
    after = float(ops.nodeDisp(top_node, LATERAL_DOF))

    reached = abs((after - before) - delta) <= 1.0e-10 + 1.0e-3 * max(abs(delta), 1.0e-12)
    if ok == 0 and reached:
        return {
            "success": True,
            "substeps": 1,
            "max_bisection": depth,
            "iterations": float(ops.testIter()),
            "test_norm": float(ops.testNorm()[-1]) if ops.testNorm() else math.nan,
        }

    if depth >= MAX_BISECTIONS:
        return {
            "success": False,
            "substeps": 0,
            "max_bisection": depth,
            "iterations": math.nan,
            "test_norm": math.nan,
        }

    half = 0.5 * delta
    first = analyze_displacement_increment(ops, top_node, half, depth + 1)
    if not first["success"]:
        return first
    second = analyze_displacement_increment(ops, top_node, half, depth + 1)
    if not second["success"]:
        return second
    return {
        "success": True,
        "substeps": int(first["substeps"]) + int(second["substeps"]),
        "max_bisection": max(int(first["max_bisection"]), int(second["max_bisection"])),
        "iterations": float(first["iterations"]) + float(second["iterations"]),
        "test_norm": second["test_norm"],
    }


# =============================================================================
# 5. Sampling output quantities
# =============================================================================

def sample_state(
    ops,
    *,
    step: int,
    p: float,
    target_drift_m: float,
    base_node: int,
    top_node: int,
    element_tags: list[int],
    section_layout: list[dict[str, object]],
    increment_summary: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], list[dict[str, object]]]:
    """Read global, base-section, and base-fiber response."""
    ops.reactions()
    top_drift = float(ops.nodeDisp(top_node, LATERAL_DOF))
    top_out_of_plane = float(ops.nodeDisp(top_node, OUT_OF_PLANE_DOF))
    top_axial = float(ops.nodeDisp(top_node, AXIAL_DOF))
    base_shear_mn = float(ops.nodeReaction(base_node, LATERAL_DOF)) / 1.0e6
    base_axial_mn = float(ops.nodeReaction(base_node, AXIAL_DOF)) / 1.0e6

    # Base station = first integration section of first element.
    base_element = element_tags[0]
    base_section_index = 1
    deformation = list(ops.eleResponse(base_element, "section", base_section_index, "deformation") or [])
    force = list(ops.eleResponse(base_element, "section", base_section_index, "force") or [])
    axial_strain = float(deformation[0]) if len(deformation) > 0 else math.nan
    # In this 3D orientation, global-X drift is reported in the local bending
    # component used by the validated reference as curvature_y / moment_y.
    curvature_y = float(deformation[2]) if len(deformation) > 2 else math.nan
    axial_force_mn = float(force[0]) / 1.0e6 if len(force) > 0 else math.nan
    moment_y_mnm = float(force[2]) / 1.0e6 if len(force) > 2 else math.nan

    hysteresis = {
        "step": step,
        "p": p,
        "drift_m": top_drift,
        "base_shear_MN": base_shear_mn,
    }
    control = {
        "step": step,
        "p": p,
        "target_drift_m": target_drift_m,
        "actual_tip_drift_m": top_drift,
        "top_out_of_plane_displacement_m": top_out_of_plane,
        "top_axial_displacement_m": top_axial,
        "base_shear_MN": base_shear_mn,
        "base_axial_reaction_MN": base_axial_mn,
        "accepted_substep_count": increment_summary["substeps"] if increment_summary else 0,
        "max_bisection_level": increment_summary["max_bisection"] if increment_summary else 0,
        "newton_iterations": increment_summary["iterations"] if increment_summary else 0.0,
        "test_norm": increment_summary["test_norm"] if increment_summary else 0.0,
    }
    section = {
        "step": step,
        "p": p,
        "drift_m": top_drift,
        "section_location": "base_first_integration_point",
        "axial_strain": axial_strain,
        "curvature_y": curvature_y,
        "axial_force_MN": axial_force_mn,
        "moment_y_MNm": moment_y_mnm,
    }

    fiber_rows = []
    for fiber in section_layout:
        y = float(fiber["y"])
        z = float(fiber["z"])
        mat = int(fiber["material_tag"])
        values = (
            ops.eleResponse(
                base_element,
                "section",
                base_section_index,
                "fiber",
                y,
                z,
                mat,
                "stressStrain",
            )
            or []
        )
        stress_mpa = float(values[0]) / 1.0e6 if len(values) >= 1 else math.nan
        strain = float(values[1]) if len(values) >= 2 else math.nan
        fiber_rows.append(
            {
                "step": step,
                "p": p,
                "drift_m": top_drift,
                "fiber_index": fiber["fiber_index"],
                "y": y,
                "z": z,
                "area": fiber["area"],
                "zone": fiber["zone"],
                "material_role": fiber["material_role"],
                "material_tag": mat,
                "strain_xx": strain,
                "stress_xx_MPa": stress_mpa,
            }
        )

    return hysteresis, control, section, fiber_rows


# =============================================================================
# 6. Plotting
# =============================================================================

def plot_hysteresis(rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; CSV files were written but no plot was created.")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(
        [1000.0 * float(row["drift_m"]) for row in rows],
        [1000.0 * float(row["base_shear_MN"]) for row in rows],
        color="#d97706",
        lw=1.4,
        label="OpenSees Hi-Fi teaching script",
    )
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.set_xlabel("Tip displacement [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("RC column cyclic reference to 200 mm")
    ax.grid(True, alpha=0.30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hysteresis.png", dpi=220)
    fig.savefig(OUT_DIR / "hysteresis.pdf")
    plt.close(fig)


# =============================================================================
# 7. Main analysis
# =============================================================================

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ops = import_opensees()
    # OpenSees prints a warning every time an attempted increment fails before
    # bisection. Those failed trials are expected in a cutback strategy, so the
    # warnings are redirected to a log file and the console only reports accepted
    # milestone steps.
    ops.logFile(str(OUT_DIR / "opensees_warnings.log"), "-noEcho")

    base_node, top_node, element_tags, section_layout = build_model(ops)
    protocol = build_protocol()

    # Axial preload. This represents the constant gravity/axial compression
    # used before applying lateral cyclic displacement.
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(top_node, 0.0, 0.0, -AXIAL_COMPRESSION_MN * 1.0e6, 0.0, 0.0, 0.0)
    configure_static_solver(ops)
    ops.integrator("LoadControl", 1.0 / AXIAL_PRELOAD_STEPS)
    ops.analysis("Static")
    preload_ok = ops.analyze(AXIAL_PRELOAD_STEPS)
    if preload_ok != 0:
        raise RuntimeError("Axial preload did not converge.")
    ops.loadConst("-time", 0.0)

    # Lateral reference load. DisplacementControl uses it to define the loaded
    # degree of freedom, while the target displacement is enforced by the integrator.
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(top_node, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    hysteresis_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    section_rows: list[dict[str, object]] = []
    fiber_rows: list[dict[str, object]] = []

    h, c, s, f = sample_state(
        ops,
        step=0,
        p=0.0,
        target_drift_m=0.0,
        base_node=base_node,
        top_node=top_node,
        element_tags=element_tags,
        section_layout=section_layout,
        increment_summary=None,
    )
    hysteresis_rows.append(h)
    control_rows.append(c)
    section_rows.append(s)
    fiber_rows.extend(f)

    previous_target = 0.0
    status = "completed"
    failure: dict[str, object] = {}
    for point in protocol[1:]:
        target = float(point["target_drift_m"])
        delta = target - previous_target
        summary = analyze_displacement_increment(ops, top_node, delta)
        if not summary["success"]:
            status = "failed"
            failure = {
                "step": point["step"],
                "p": point["p"],
                "target_drift_m": target,
                "delta_m": delta,
                "reason": "displacement-control increment failed after cutbacks",
            }
            print(f"FAILED at step {point['step']}: target drift = {target:.6e} m")
            break

        h, c, s, f = sample_state(
            ops,
            step=int(point["step"]),
            p=float(point["p"]),
            target_drift_m=target,
            base_node=base_node,
            top_node=top_node,
            element_tags=element_tags,
            section_layout=section_layout,
            increment_summary=summary,
        )
        hysteresis_rows.append(h)
        control_rows.append(c)
        section_rows.append(s)
        fiber_rows.extend(f)
        previous_target = target

        if int(point["step"]) % 64 == 0:
            print(
                f"step {point['step']:4d}  "
                f"drift={1000.0 * float(h['drift_m']):+8.2f} mm  "
                f"V={1000.0 * float(h['base_shear_MN']):+8.2f} kN  "
                f"substeps={summary['substeps']}"
            )

    write_csv(
        OUT_DIR / "hysteresis.csv",
        hysteresis_rows,
        ["step", "p", "drift_m", "base_shear_MN"],
    )
    write_csv(
        OUT_DIR / "control_state.csv",
        control_rows,
        [
            "step",
            "p",
            "target_drift_m",
            "actual_tip_drift_m",
            "top_out_of_plane_displacement_m",
            "top_axial_displacement_m",
            "base_shear_MN",
            "base_axial_reaction_MN",
            "accepted_substep_count",
            "max_bisection_level",
            "newton_iterations",
            "test_norm",
        ],
    )
    write_csv(
        OUT_DIR / "base_moment_curvature.csv",
        section_rows,
        [
            "step",
            "p",
            "drift_m",
            "section_location",
            "axial_strain",
            "curvature_y",
            "axial_force_MN",
            "moment_y_MNm",
        ],
    )
    write_csv(
        OUT_DIR / "base_fiber_history.csv",
        fiber_rows,
        [
            "step",
            "p",
            "drift_m",
            "fiber_index",
            "y",
            "z",
            "area",
            "zone",
            "material_role",
            "material_tag",
            "strain_xx",
            "stress_xx_MPa",
        ],
    )
    write_csv(
        OUT_DIR / "section_layout.csv",
        section_layout,
        ["fiber_index", "y", "z", "area", "zone", "material_role", "material_tag"],
    )

    peak_drift_mm = max(abs(float(row["drift_m"])) for row in hysteresis_rows) * 1000.0
    peak_shear_kn = max(abs(float(row["base_shear_MN"])) for row in hysteresis_rows) * 1000.0
    manifest = {
        "script": Path(__file__).name,
        "status": status,
        "failure": failure,
        "model_dimension": "3d",
        "axis_convention": {
            "column_axis": "global Z",
            "lateral_control": "global X displacement at the top node",
            "out_of_plane_axis": "global Y",
            "section_component_reported": "curvature_y and moment_y from the 3D section response",
        },
        "geometry": {
            "L_COLUMN_m": L_COLUMN,
            "SECTION_B_m": SECTION_B,
            "SECTION_H_m": SECTION_H,
            "COVER_m": COVER,
            "BAR_DIAMETER_m": BAR_DIAMETER,
        },
        "materials": {
            "concrete": {
                "model": "Concrete02",
                "FPC_MPa": FPC_MPA,
                "lambda": CONCRETE_LAMBDA,
                "ft_ratio": CONCRETE_FT_RATIO,
                "softening_multiplier": CONCRETE_SOFTENING_MULTIPLIER,
                "ultimate_strain": CONCRETE_ULTIMATE_STRAIN,
            },
            "steel": {
                "model": "Steel02 Menegotto-Pinto",
                "E_MPa": STEEL_E_MPA,
                "fy_MPa": STEEL_FY_MPA,
                "b": STEEL_B,
                "R0": STEEL_R0,
                "cR1": STEEL_CR1,
                "cR2": STEEL_CR2,
                "a1": STEEL_A1,
                "a2": STEEL_A2,
                "a3": STEEL_A3,
                "a4": STEEL_A4,
            },
        },
        "discretization": {
            "N_ELEMENTS": N_ELEMENTS,
            "element": "dispBeamColumn",
            "beam_integration": BEAM_INTEGRATION,
            "N_INTEGRATION_POINTS": N_INTEGRATION_POINTS,
            "geometric_transformation": GEOMETRIC_TRANSFORMATION,
            "node_ndm": 3,
            "node_ndf": 6,
            "section_fiber_count": len(section_layout),
        },
        "loading": {
            "AXIAL_COMPRESSION_MN": AXIAL_COMPRESSION_MN,
            "AXIAL_PRELOAD_STEPS": AXIAL_PRELOAD_STEPS,
            "AMPLITUDES_MM": AMPLITUDES_MM,
            "STEPS_PER_SEGMENT": STEPS_PER_SEGMENT,
            "REVERSAL_SUBSTEP_FACTOR": REVERSAL_SUBSTEP_FACTOR,
        },
        "solver": {
            "test": NEWTON_TEST,
            "tolerance": NEWTON_TOL,
            "max_iterations": NEWTON_MAX_ITER,
            "algorithm": NEWTON_ALGORITHM,
            "max_bisections": MAX_BISECTIONS,
        },
        "summary": {
            "records": len(hysteresis_rows),
            "peak_abs_drift_mm": peak_drift_mm,
            "peak_abs_base_shear_kN": peak_shear_kn,
        },
    }
    (OUT_DIR / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    plot_hysteresis(hysteresis_rows)

    print(f"\nStatus: {status}")
    print(f"Output: {OUT_DIR.resolve()}")
    print(f"Peak drift: {peak_drift_mm:.3f} mm")
    print(f"Peak |base shear|: {peak_shear_kn:.3f} kN")
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
