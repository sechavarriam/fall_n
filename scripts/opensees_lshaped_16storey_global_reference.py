#!/usr/bin/env python3
"""OpenSeesPy global reference for the 16-storey L-shaped RC frame.

The script is intentionally conservative: it creates the external single-scale
global comparator required by the FE2 seismic campaign, writes the same recorder
families expected from fall_n, and supports ``--dry-run`` so CI/documentation can
verify the campaign manifest without requiring OpenSeesPy.

Units follow the fall_n driver convention: m, MN, MPa (= MN/m^2), s.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Iterable


X_GRID = [0.0, 5.0, 10.0, 15.0, 20.0]
Y_GRID = [0.0, 4.0, 8.0, 12.0]
NUM_STORIES = 16
STORY_HEIGHT = 3.2
CUTOUT_X_START = 1
CUTOUT_X_END = 4
CUTOUT_Y_START = 1
CUTOUT_Y_END = 3

COL_B = [0.50, 0.40, 0.30]
COL_H = [0.50, 0.40, 0.30]
COL_FPC = [35.0, 28.0, 21.0]
COL_BAR = 0.020
COL_RHO_S = 0.015
BM_B = 0.30
BM_H = 0.60
BM_FPC = 25.0
BM_BAR = 0.016
NU_RC = 0.20
STEEL_FY = 420.0
STEEL_E = 200000.0
STEEL_B = 0.01
TIE_FY = 420.0
CONCRETE_TENSION_RATIO = 0.10


def concrete_ec(fpc: float) -> float:
    # fall_n's RCSectionBuilder currently uses concrete_initial_modulus(fpc)
    # = 1000*f'c for fiber-section axial stiffness.  Keeping this default here
    # makes the external elastic parity slice compare the same material scale.
    return 1000.0 * abs(fpc)


def rectangular_elastic_props(b: float, h: float, fpc: float) -> tuple[float, float, float, float, float, float]:
    E = concrete_ec(fpc)
    G = E / (2.0 * (1.0 + NU_RC))
    A = b * h
    Iy = h * b**3 / 12.0
    Iz = b * h**3 / 12.0
    J = Iy + Iz
    return A, E, G, J, Iy, Iz


def rectangular_torsion_constant(width: float, height: float) -> float:
    b_min = min(width, height)
    h_max = max(width, height)
    return (b_min**3 * h_max / 3.0) * (1.0 - 0.63 * b_min / h_max)


RC_DENSITY = 2.4e-3


def transformed_rc_column_props(b: float, h: float, fpc: float) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Return E=G=1 equivalent properties matching fall_n elasticized fibers.

    OpenSees ElasticTimoshenkoBeam uses E*A, E*Iy, E*Iz, G*J, G*Avy and
    G*Avz.  By setting E=G=1 we can pass the already-transformed stiffnesses
    assembled from the same concrete patches and longitudinal bars used by
    fall_n's elasticized RC section.
    """
    ec = concrete_ec(fpc)
    gc = ec / (2.0 * (1.0 + NU_RC))
    cover = 0.04
    bar_diameter = COL_BAR
    steel_area = math.pi * (bar_diameter / 2.0) ** 2
    y_bar = 0.5 * b - cover
    z_bar = 0.5 * h - cover
    bars = [
        (-y_bar, -z_bar), (y_bar, -z_bar),
        (-y_bar, z_bar), (y_bar, z_bar),
        (0.0, -z_bar), (0.0, z_bar),
        (-y_bar, 0.0), (y_bar, 0.0),
    ]
    concrete_area = b * h
    ea = ec * concrete_area + STEEL_E * steel_area * len(bars)
    eiy = ec * (b * h**3 / 12.0)
    eiz = ec * (h * b**3 / 12.0)
    for y, z in bars:
        eiy += STEEL_E * steel_area * z * z
        eiz += STEEL_E * steel_area * y * y
    area_equiv = concrete_area + steel_area * len(bars)
    gj = gc * rectangular_torsion_constant(b, h)
    avy = (5.0 / 6.0) * gc * area_equiv
    avz = (5.0 / 6.0) * gc * area_equiv
    mass_per_length = RC_DENSITY * area_equiv
    return ea, 1.0, 1.0, gj, eiy, eiz, avy, avz, mass_per_length


def transformed_rc_beam_props(b: float, h: float, fpc: float) -> tuple[float, float, float, float, float, float, float, float, float]:
    ec = concrete_ec(fpc)
    gc = ec / (2.0 * (1.0 + NU_RC))
    cover = 0.04
    bar_diameter = BM_BAR
    steel_area = math.pi * (bar_diameter / 2.0) ** 2
    y_bar = 0.5 * b - cover
    z_bar = 0.5 * h - cover
    bars = [
        (-y_bar, -z_bar), (0.0, -z_bar), (y_bar, -z_bar),
        (-y_bar, z_bar), (0.0, z_bar), (y_bar, z_bar),
    ]
    concrete_area = b * h
    ea = ec * concrete_area + STEEL_E * steel_area * len(bars)
    eiy = ec * (b * h**3 / 12.0)
    eiz = ec * (h * b**3 / 12.0)
    for y, z in bars:
        eiy += STEEL_E * steel_area * z * z
        eiz += STEEL_E * steel_area * y * y
    area_equiv = concrete_area + steel_area * len(bars)
    gj = gc * rectangular_torsion_constant(b, h)
    avy = (5.0 / 6.0) * gc * area_equiv
    avz = (5.0 / 6.0) * gc * area_equiv
    mass_per_length = RC_DENSITY * area_equiv
    return ea, 1.0, 1.0, gj, eiy, eiz, avy, avz, mass_per_length


def active_bay(ix: int, iy: int) -> bool:
    return not (
        CUTOUT_X_START <= ix < CUTOUT_X_END
        and CUTOUT_Y_START <= iy < CUTOUT_Y_END
    )


def active_grid_point(ix: int, iy: int) -> bool:
    """Match fall_n's bay-derived node activity for the L-shaped plan.

    A grid node is kept if at least one adjacent bay is active.  This is more
    precise than cutting nodes by rectangle and prevents OpenSees from using a
    different structural graph than BuildingDomainBuilder.
    """
    for bx in (ix - 1, ix):
        for by in (iy - 1, iy):
            if (
                0 <= bx < len(X_GRID) - 1
                and 0 <= by < len(Y_GRID) - 1
                and active_bay(bx, by)
            ):
                return True
    return False


def story_range(story: int) -> int:
    if story < 5:
        return 0
    if story < 11:
        return 1
    return 2


def kent_park_unconfined_proxy(
    fpc: float,
    *,
    tension_ratio: float = CONCRETE_TENSION_RATIO,
    ets_ratio: float = 1.0,
) -> dict[str, float]:
    """Return the compression/tension envelope used by fall_n's Kent-Park law.

    OpenSees does not provide the exact fall_n implementation, including the
    reclosure bridge and residual tensile tangent.  These parameters therefore
    define an auditable Concrete02 proxy with the same initial modulus,
    compressive peak, residual branch, and tensile cracking stress.
    """
    eps0 = -0.002
    eps_50u = max((3.0 + 0.29 * fpc) / (145.0 * fpc - 1000.0), 1.0e-6)
    z_slope = 0.5 / max(eps_50u + eps0, 1.0e-6)
    eps_u = eps0 - 0.8 / z_slope
    return {
        "fpc_peak": fpc,
        "eps0": eps0,
        "fpc_residual": 0.20 * fpc,
        "eps_u": eps_u,
        "ft": tension_ratio * fpc,
        "ets": ets_ratio * concrete_ec(fpc),
        "lambda": 0.10,
        "kconf": 1.0,
    }


def kent_park_confined_proxy(
    fpc: float,
    *,
    b: float,
    h: float,
    tension_ratio: float = CONCRETE_TENSION_RATIO,
    ets_ratio: float = 1.0,
    cover: float = 0.04,
    rho_s: float = COL_RHO_S,
    tie_fy: float = TIE_FY,
    tie_spacing: float = 0.10,
) -> dict[str, float]:
    y_core = 0.5 * b - cover
    z_core = 0.5 * h - cover
    h_prime = 2.0 * min(y_core, z_core)
    kconf = 1.0 + rho_s * tie_fy / fpc
    eps0 = -0.002 * kconf
    eps_50u = max((3.0 + 0.29 * fpc) / (145.0 * fpc - 1000.0), 1.0e-6)
    eps_50h = 0.75 * rho_s * math.sqrt(max(h_prime / tie_spacing, 1.0e-12))
    z_slope = 0.5 / max(eps_50u + eps_50h + eps0, 1.0e-6)
    eps_u = eps0 - 0.8 / z_slope
    return {
        "fpc_peak": kconf * fpc,
        "eps0": eps0,
        "fpc_residual": 0.20 * kconf * fpc,
        "eps_u": eps_u,
        "ft": tension_ratio * fpc,
        "ets": ets_ratio * concrete_ec(fpc),
        "lambda": 0.10,
        "kconf": kconf,
    }


def parse_knet(path: Path) -> tuple[float, list[float]]:
    """Return dt and acceleration values in m/s^2 from a K-NET-like file."""
    dt = None
    scale = 1.0
    values: list[float] = []
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("sampling freq"):
            nums = [float(tok) for tok in line.replace("Hz", "").split() if _is_float(tok)]
            if nums:
                dt = 1.0 / nums[-1]
        elif low.startswith("scale factor"):
            match = re.search(
                r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?:\(gal\))?\s*/\s*"
                r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                line,
            )
            if match and float(match.group(2)) != 0.0:
                scale = float(match.group(1)) / float(match.group(2))
            else:
                parts = line.split()
                for token in reversed(parts):
                    if _is_float(token):
                        scale = float(token)
                        break
        elif line[0].isdigit() or line[0] in "+-.":
            for token in line.split():
                if _is_float(token):
                    # K-NET acceleration records are commonly in gal.
                    values.append(float(token) * scale * 0.01)
    if dt is None:
        dt = 0.01
    return dt, values


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_empty_recorders(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, header in {
        "roof_displacement.csv": "time,ux,uy,uz\n",
        "story_drift.csv": "time,storey,drift_x,drift_y\n",
        "base_shear.csv": "time,Vx,Vy,Vz\n",
        "section_critical.csv": "time,element,section,N,My,Mz\n",
    }.items():
        (out_dir / name).write_text(header, encoding="utf-8")


def _windowed(values: list[float], dt: float, start_time: float, duration: float) -> list[float]:
    i0 = max(0, int(round(start_time / dt)))
    n = max(1, int(round(duration / dt)))
    return values[i0 : i0 + n]


def build_and_run(args: argparse.Namespace) -> dict:
    import openseespy.opensees as ops  # type: ignore

    t0_wall = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    node_id: dict[tuple[int, int, int], int] = {}
    tag = 1
    nodal_mass_total = 0.0
    for k in range(NUM_STORIES + 1):
        z = k * STORY_HEIGHT
        for ix, x in enumerate(X_GRID):
            for iy, y in enumerate(Y_GRID):
                if not active_grid_point(ix, iy):
                    continue
                node_id[(ix, iy, k)] = tag
                ops.node(tag, x, y, z)
                if k == 0:
                    ops.fix(tag, 1, 1, 1, 1, 1, 1)
                elif args.mass_model == "nodal":
                    mass = args.floor_mass / max(1, len(X_GRID) * len(Y_GRID))
                    ops.mass(tag, mass, mass, mass, 0.0, 0.0, 0.0)
                    nodal_mass_total += mass
                tag += 1

    # Materials. Negative strengths follow OpenSees concrete convention.
    mat_tag = 1
    steel_tag = mat_tag
    if args.elasticized_fiber_sections:
        ops.uniaxialMaterial("Elastic", steel_tag, STEEL_E)
    else:
        ops.uniaxialMaterial("Steel02", steel_tag, STEEL_FY, STEEL_E, STEEL_B)
    mat_tag += 1
    concrete_tags: list[dict[str, int | float | str]] = []
    concrete_material_audit: list[dict[str, float | int | str]] = []

    def add_concrete_material(
        *,
        role: str,
        fpc: float,
        b: float,
        h: float,
        confined: bool,
    ) -> int:
        nonlocal mat_tag
        tag_c = mat_tag
        mat_tag += 1
        if args.elasticized_fiber_sections:
            ops.uniaxialMaterial("Elastic", tag_c, concrete_ec(fpc))
            concrete_material_audit.append({
                "tag": tag_c,
                "role": role,
                "model": "Elastic",
                "fpc_mpa": fpc,
                "Ec_mpa": concrete_ec(fpc),
            })
        else:
            if args.concrete_model == "concrete01":
                ec0 = -0.002
                fpcu = -0.20 * fpc
                ecu = -0.006
                ops.uniaxialMaterial("Concrete01", tag_c, -fpc, ec0, fpcu, ecu)
                concrete_material_audit.append({
                    "tag": tag_c,
                    "role": role,
                    "model": "Concrete01",
                    "fpc_mpa": fpc,
                    "eps0": ec0,
                    "fpcu_mpa": fpcu,
                    "epsu": ecu,
                })
            else:
                if confined:
                    params = kent_park_confined_proxy(
                        fpc,
                        b=b,
                        h=h,
                        tension_ratio=args.concrete_tension_ratio,
                        ets_ratio=args.concrete_ets_ratio,
                    )
                else:
                    params = kent_park_unconfined_proxy(
                        fpc,
                        tension_ratio=args.concrete_tension_ratio,
                        ets_ratio=args.concrete_ets_ratio,
                    )
                ops.uniaxialMaterial(
                    "Concrete02",
                    tag_c,
                    -params["fpc_peak"],
                    params["eps0"],
                    -params["fpc_residual"],
                    params["eps_u"],
                    params["lambda"],
                    params["ft"],
                    params["ets"],
                )
                concrete_material_audit.append({
                    "tag": tag_c,
                    "role": role,
                    "model": "Concrete02_falln_kent_park_proxy",
                    "fpc_mpa": fpc,
                    "confined": confined,
                    **params,
                })
        return tag_c

    for i, fpc in enumerate(COL_FPC):
        cover_tag = add_concrete_material(
            role=f"column_range_{i}_cover",
            fpc=fpc,
            b=COL_B[i],
            h=COL_H[i],
            confined=False,
        )
        core_tag = (
            add_concrete_material(
                role=f"column_range_{i}_core",
                fpc=fpc,
                b=COL_B[i],
                h=COL_H[i],
                confined=args.concrete_model == "falln-kent-park-proxy",
            )
            if args.concrete_model == "falln-kent-park-proxy"
            else cover_tag
        )
        concrete_tags.append({
            "cover": cover_tag,
            "core": core_tag,
            "fpc": fpc,
            "kind": "column",
        })
    beam_cover_tag = add_concrete_material(
        role="beam_cover_and_core",
        fpc=BM_FPC,
        b=BM_B,
        h=BM_H,
        confined=False,
    )
    concrete_tags.append({
        "cover": beam_cover_tag,
        "core": beam_cover_tag,
        "fpc": BM_FPC,
        "kind": "beam",
    })

    def make_fiber_section(sec_tag: int, b: float, h: float, conc_tags: dict[str, int | float | str], *, is_beam: bool) -> None:
        gj = 0.10 * 4700.0 * math.sqrt(abs(COL_FPC[0])) * b * h * (b * b + h * h) / 12.0
        ops.section("Fiber", sec_tag, "-GJ", max(gj, 1.0e-6))
        cover = 0.04
        cover_tag = int(conc_tags["cover"])
        core_tag = int(conc_tags["core"])
        # Match fall_n's RCSectionLayout convention: local y is the section
        # width b and local z is the section depth/height h.
        y_edge = b / 2
        z_edge = h / 2
        y_core = y_edge - cover
        z_core = z_edge - cover
        if is_beam:
            ops.patch("rect", cover_tag, 6, 2, -y_edge, -z_edge, y_edge, -z_core)
            ops.patch("rect", cover_tag, 6, 2, -y_edge, z_core, y_edge, z_edge)
            ops.patch("rect", cover_tag, 2, 6, -y_edge, -z_core, -y_core, z_core)
            ops.patch("rect", cover_tag, 2, 6, y_core, -z_core, y_edge, z_core)
            ops.patch("rect", core_tag, 4, 6, -y_core, -z_core, y_core, z_core)
            area = math.pi * (BM_BAR / 2) ** 2
            bars = [
                (-y_core, -z_core), (-y_core, 0.0), (-y_core, z_core),
                ( y_core, -z_core), ( y_core, 0.0), ( y_core, z_core),
            ]
        else:
            # fall_n canonical RCColumnSpec layout mapped to OpenSees section
            # axes: 4 cover patches, one core patch and 8 longitudinal bars.
            ops.patch("rect", cover_tag, 4, 2, -y_edge, -z_edge, y_edge, -z_core)
            ops.patch("rect", cover_tag, 4, 2, -y_edge, z_core, y_edge, z_edge)
            ops.patch("rect", cover_tag, 2, 8, -y_edge, -z_core, -y_core, z_core)
            ops.patch("rect", cover_tag, 2, 8, y_core, -z_core, y_edge, z_core)
            ops.patch("rect", core_tag, 6, 6, -y_core, -z_core, y_core, z_core)
            area = math.pi * (COL_BAR / 2) ** 2
            bars = [
                (-y_core, -z_core), (-y_core, z_core),
                ( y_core, -z_core), ( y_core, z_core),
                (-y_core, 0.0), ( y_core, 0.0),
                (0.0, -z_core), (0.0, z_core),
            ]
        for y, z in bars:
            ops.fiber(y, z, area, steel_tag)

    def make_shear_aggregated_section(sec_tag: int, base_sec_tag: int, b: float, h: float, fpc: float, *, is_beam: bool) -> None:
        nonlocal mat_tag
        if is_beam:
            *_unused, avy, avz, _mass = transformed_rc_beam_props(b, h, fpc)
        else:
            *_unused, avy, avz, _mass = transformed_rc_column_props(b, h, fpc)
        vy_mat = mat_tag
        mat_tag += 1
        vz_mat = mat_tag
        mat_tag += 1
        ops.uniaxialMaterial("Elastic", vy_mat, max(avy, 1.0e-9))
        ops.uniaxialMaterial("Elastic", vz_mat, max(avz, 1.0e-9))
        ops.section("Aggregator", sec_tag, vy_mat, "Vy", vz_mat, "Vz", "-section", base_sec_tag)

    section_tags = {}
    for r in range(3):
        sec = 100 + r
        make_fiber_section(sec, COL_B[r], COL_H[r], concrete_tags[r], is_beam=False)
        section_tags[("col", r, "fiber")] = sec
        shear_sec = 110 + r
        make_shear_aggregated_section(shear_sec, sec, COL_B[r], COL_H[r], COL_FPC[r], is_beam=False)
        section_tags[("col", r, "shear")] = shear_sec
    beam_sec = 200
    make_fiber_section(beam_sec, BM_B, BM_H, concrete_tags[-1], is_beam=True)
    beam_shear_sec = 210
    make_shear_aggregated_section(beam_shear_sec, beam_sec, BM_B, BM_H, BM_FPC, is_beam=True)

    transf_kind = "Linear" if args.geom_transf == "linear" else "PDelta"
    ops.geomTransf(transf_kind, 1, 1.0, 0.0, 0.0)
    ops.geomTransf(transf_kind, 2, 0.0, 0.0, 1.0)
    ops.geomTransf(transf_kind, 3, 0.0, 0.0, 1.0)
    force_section_kind = "shear" if args.beam_element_family == "force-shear" else "fiber"
    ops.beamIntegration("Lobatto", 1, section_tags[("col", 0, force_section_kind)], 5)
    ops.beamIntegration("Lobatto", 2, section_tags[("col", 1, force_section_kind)], 5)
    ops.beamIntegration("Lobatto", 3, section_tags[("col", 2, force_section_kind)], 5)
    ops.beamIntegration("Lobatto", 4, beam_shear_sec if force_section_kind == "shear" else beam_sec, 5)

    ele = 1
    critical_elements: list[int] = []
    element_mass_total = 0.0
    def add_frame_element(
        ele_tag: int,
        ni: int,
        nj: int,
        transf: int,
        integ: int,
        b: float,
        h: float,
        fpc: float,
    ) -> None:
        if args.beam_element_family in {"force", "force-shear"}:
            cmd = ["forceBeamColumn", ele_tag, ni, nj, transf, integ]
            if args.element_iterations > 0:
                cmd.extend(["-iter", args.element_iterations, args.element_tolerance])
            if args.mass_model == "element":
                props = transformed_rc_beam_props(b, h, fpc) if integ == 4 else transformed_rc_column_props(b, h, fpc)
                cmd.extend(["-mass", props[-1]])
            ops.element(*cmd)
        elif args.beam_element_family == "disp":
            ops.element("dispBeamColumn", ele_tag, ni, nj, transf, integ)
        elif args.beam_element_family == "elastic":
            A, E, G, J, Iy, Iz = rectangular_elastic_props(b, h, fpc)
            ops.element(
                "elasticBeamColumn",
                ele_tag,
                ni,
                nj,
                A,
                E,
                G,
                J,
                Iy,
                Iz,
                transf,
            )
        elif args.beam_element_family == "elastic-timoshenko":
            if integ == 4:
                A, E, G, J, Iy, Iz, Avy, Avz, mass_dens = transformed_rc_beam_props(b, h, fpc)
            else:
                A, E, G, J, Iy, Iz, Avy, Avz, mass_dens = transformed_rc_column_props(b, h, fpc)
            cmd = [
                "ElasticTimoshenkoBeam",
                ele_tag, ni, nj, E, G, A, J, Iy, Iz, Avy, Avz, transf,
            ]
            if args.mass_model == "element":
                cmd.extend(["-mass", mass_dens])
                if args.element_mass_form == "consistent":
                    cmd.append("-cMass")
            ops.element(*cmd)
        else:
            raise ValueError(f"Unsupported beam element family: {args.beam_element_family}")

    next_aux_node = max(node_id.values()) + 1

    def add_member(
        first_ele_tag: int,
        ni: int,
        nj: int,
        xi: float,
        yi: float,
        zi: float,
        xj: float,
        yj: float,
        zj: float,
        transf: int,
        integ: int,
        b: float,
        h: float,
        fpc: float,
    ) -> int:
        nonlocal next_aux_node
        nonlocal element_mass_total
        nsub = max(1, args.member_subdivisions)
        length = math.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        if args.mass_model == "element":
            props = transformed_rc_beam_props(b, h, fpc) if integ == 4 else transformed_rc_column_props(b, h, fpc)
            element_mass_total += props[-1] * length
        if nsub == 1:
            add_frame_element(first_ele_tag, ni, nj, transf, integ, b, h, fpc)
            return first_ele_tag + 1

        nodes = [ni]
        for s in range(1, nsub):
            a = s / nsub
            tag_aux = next_aux_node
            next_aux_node += 1
            ops.node(
                tag_aux,
                xi + a * (xj - xi),
                yi + a * (yj - yi),
                zi + a * (zj - zi),
            )
            nodes.append(tag_aux)
        nodes.append(nj)

        ele_tag = first_ele_tag
        for a, bnode in zip(nodes, nodes[1:]):
            add_frame_element(ele_tag, a, bnode, transf, integ, b, h, fpc)
            ele_tag += 1
        return ele_tag

    for k in range(NUM_STORIES):
        r = story_range(k)
        for ix, _x in enumerate(X_GRID):
            for iy, _y in enumerate(Y_GRID):
                if not active_grid_point(ix, iy):
                    continue
                n1 = node_id[(ix, iy, k)]
                n2 = node_id[(ix, iy, k + 1)]
                if k < 3:
                    critical_elements.append(ele)
                x = X_GRID[ix]
                y = Y_GRID[iy]
                z0 = k * STORY_HEIGHT
                z1 = (k + 1) * STORY_HEIGHT
                ele = add_member(ele, n1, n2, x, y, z0, x, y, z1,
                                 1, r + 1, COL_B[r], COL_H[r], COL_FPC[r])

    for k in range(1, NUM_STORIES + 1):
        z = k * STORY_HEIGHT
        for iy, _y in enumerate(Y_GRID):
            for ix in range(len(X_GRID) - 1):
                if (ix, iy, k) in node_id and (ix + 1, iy, k) in node_id:
                    ele = add_member(
                        ele,
                        node_id[(ix, iy, k)],
                        node_id[(ix + 1, iy, k)],
                        X_GRID[ix], Y_GRID[iy], z,
                        X_GRID[ix + 1], Y_GRID[iy], z,
                        2, 4, BM_B, BM_H, BM_FPC)
        for ix, _x in enumerate(X_GRID):
            for iy in range(len(Y_GRID) - 1):
                if (ix, iy, k) in node_id and (ix, iy + 1, k) in node_id:
                    ele = add_member(
                        ele,
                        node_id[(ix, iy, k)],
                        node_id[(ix, iy + 1, k)],
                        X_GRID[ix], Y_GRID[iy], z,
                        X_GRID[ix], Y_GRID[iy + 1], z,
                        3, 4, BM_B, BM_H, BM_FPC)

    roof_node = max(node_id.values())

    # Gravity.
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for key, nid in node_id.items():
        if key[2] > 0:
            ops.load(nid, 0.0, 0.0, -args.gravity_load, 0.0, 0.0, 0.0)
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 20)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    gravity_ok = ops.analyze(10) == 0
    ops.loadConst("-time", 0.0)

    eigen_periods: list[float] = []
    if args.eigen_modes > 0:
        try:
            lambdas = ops.eigen(args.eigen_modes)
            eigen_periods = [
                (2.0 * math.pi / math.sqrt(lam)) if lam > 0.0 else float("nan")
                for lam in lambdas
            ]
        except Exception:
            eigen_periods = []

    if args.dry_gravity_only:
        return {
            "status": "gravity_only",
            "gravity_ok": gravity_ok,
            "beam_element_family": args.beam_element_family,
            "mass_model": args.mass_model,
            "element_mass_form": args.element_mass_form if args.mass_model == "element" else "",
            "elasticized_fiber_sections": bool(args.elasticized_fiber_sections),
            "concrete_model": args.concrete_model,
            "section_axis_convention": "fall_n: y=width_b, z=height_h",
            "concrete_material_audit": concrete_material_audit,
            "member_subdivisions": args.member_subdivisions,
            "geom_transf": args.geom_transf,
            "include_vertical": args.include_vertical,
            "node_count": len(node_id),
            "node_count_with_auxiliary": next_aux_node - 1,
            "element_count": ele - 1,
            "nodal_mass_total_per_direction": nodal_mass_total,
            "element_mass_total_per_direction": element_mass_total,
            "eigen_periods_s": eigen_periods,
            "total_wall_seconds": time.perf_counter() - t0_wall,
        }

    # Dynamic recorders are created after gravity and loadConst("-time", 0.0)
    # so the CSV response is a clean seismic window, comparable to fall_n.
    ops.recorder("Node", "-file", str(out / "roof_displacement.out"), "-time", "-node", roof_node, "-dof", 1, 2, 3, "disp")
    ops.recorder("Node", "-file", str(out / "base_reaction.out"), "-time", "-node", *[node_id[key] for key in node_id if key[2] == 0], "-dof", 1, 2, 3, "reaction")
    for suffix, response in {
        "force": "force",
        "global_force": "globalForce",
        "local_force": "localForce",
        "basic_force": "basicForce",
    }.items():
        ops.recorder(
            "Element",
            "-file",
            str(out / f"selected_element_1_{suffix}.out"),
            "-time",
            "-ele",
            1,
            response,
        )
    for e in critical_elements[:10]:
        ops.recorder("Element", "-file", str(out / f"section_element_{e}.out"), "-time", "-ele", e, "section", 1, "force")

    # Dynamic excitation.
    dt_ns, acc_ns_full = parse_knet(Path(args.eq_ns))
    dt_ew, acc_ew_full = parse_knet(Path(args.eq_ew))
    dt_ud, acc_ud_full = parse_knet(Path(args.eq_ud))
    dt = args.dt
    acc_ns = _windowed(acc_ns_full, dt_ns, args.start_time, args.duration)
    acc_ew = _windowed(acc_ew_full, dt_ew, args.start_time, args.duration)
    acc_ud = _windowed(acc_ud_full, dt_ud, args.start_time, args.duration)
    n = int(round(args.duration / dt))
    acc_ns = [args.scale * a for a in acc_ns]
    acc_ew = [args.scale * a for a in acc_ew]
    acc_ud = [args.scale * a for a in acc_ud]

    ops.timeSeries("Path", 11, "-dt", dt_ns, "-values", *acc_ns)
    ops.timeSeries("Path", 12, "-dt", dt_ew, "-values", *acc_ew)
    ops.pattern("UniformExcitation", 11, 1, "-accel", 11)
    ops.pattern("UniformExcitation", 12, 2, "-accel", 12)
    if args.include_vertical:
        ops.timeSeries("Path", 13, "-dt", dt_ud, "-values", *acc_ud)
        ops.pattern("UniformExcitation", 13, 3, "-accel", 13)

    omega_1 = 2.0 * math.pi / args.rayleigh_t1
    omega_3 = 2.0 * math.pi / args.rayleigh_t3
    beta_k = 2.0 * args.rayleigh_xi / (omega_1 + omega_3)
    alpha_m = beta_k * omega_1 * omega_3
    ops.rayleigh(alpha_m, beta_k, 0.0, 0.0)

    ops.wipeAnalysis()
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", args.test_tolerance, args.test_iterations)
    ops.algorithm("NewtonLineSearch")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    def try_one_step(step_dt: float) -> bool:
        profiles: tuple[tuple[str, tuple], ...]
        if args.beam_element_family in {"elastic", "elastic-timoshenko"}:
            profiles = (("Linear", ()),)
        else:
            profiles = (
            ("NewtonLineSearch", ("-type", "Bisection")),
            ("KrylovNewton", ()),
            ("Newton", ()),
            ("ModifiedNewton", ()),
            )
        tests = (
            ("NormDispIncr", args.test_tolerance),
            ("NormUnbalance", args.unbalance_tolerance),
            ("EnergyIncr", args.energy_tolerance),
        )
        for test_name, tol in tests:
            ops.test(test_name, tol, args.test_iterations)
            for algorithm, extra in profiles:
                if extra:
                    ops.algorithm(algorithm, *extra)
                else:
                    ops.algorithm(algorithm)
                if ops.analyze(1, step_dt) == 0:
                    return True
        return False

    ok_steps = 0
    for _ in range(n):
        if not try_one_step(dt):
            recovered = False
            for sub in (2, 4, 8, 16, 32):
                sub_ok = True
                for _substep in range(sub):
                    if not try_one_step(dt / sub):
                        sub_ok = False
                        break
                if sub_ok:
                    ok_steps += 1
                    recovered = True
                    break
            if not recovered:
                break
        else:
            ok_steps += 1

    return {
        "status": "completed" if ok_steps == n else "incomplete",
        "gravity_ok": gravity_ok,
        "accepted_steps": ok_steps,
        "requested_steps": n,
        "dt": dt,
        "start_time": args.start_time,
        "duration": args.duration,
        "scale": args.scale,
        "beam_element_family": args.beam_element_family,
        "mass_model": args.mass_model,
        "element_mass_form": args.element_mass_form if args.mass_model == "element" else "",
        "elasticized_fiber_sections": bool(args.elasticized_fiber_sections),
        "concrete_model": args.concrete_model,
        "concrete_tension_ratio": args.concrete_tension_ratio,
        "concrete_ets_ratio": args.concrete_ets_ratio,
        "section_axis_convention": "fall_n: y=width_b, z=height_h",
        "steel_model": "Elastic" if args.elasticized_fiber_sections else "Steel02",
        "steel_parameters": {
            "fy_mpa": STEEL_FY,
            "E_mpa": STEEL_E,
            "b": STEEL_B,
            "R0": 20.0,
            "cR1": 18.5,
            "cR2": 0.15,
        },
        "section_layout": {
            "column_bar_diameter_m": COL_BAR,
            "beam_bar_diameter_m": BM_BAR,
            "column_bar_count": 8,
            "beam_bar_count": 6,
            "cover_m": 0.04,
            "column_rho_s": COL_RHO_S,
            "tie_fy_mpa": TIE_FY,
        },
        "concrete_material_audit": concrete_material_audit,
        "member_subdivisions": args.member_subdivisions,
        "geom_transf": args.geom_transf,
        "include_vertical": args.include_vertical,
        "rayleigh_alpha_m": alpha_m,
        "rayleigh_beta_k": beta_k,
        "node_count": len(node_id),
        "node_count_with_auxiliary": next_aux_node - 1,
        "element_count": ele - 1,
        "nodal_mass_total_per_direction": nodal_mass_total,
        "element_mass_total_per_direction": element_mass_total,
        "roof_node": roof_node,
        "critical_elements_recorded": critical_elements[:10],
        "eigen_periods_s": eigen_periods,
        "total_wall_seconds": time.perf_counter() - t0_wall,
    }


def convert_recorders(out_dir: Path) -> None:
    roof_out = out_dir / "roof_displacement.out"
    roof_csv = out_dir / "roof_displacement.csv"
    if roof_out.exists():
        with roof_out.open() as src, roof_csv.open("w", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(["time", "ux", "uy", "uz"])
            for line in src:
                vals = line.split()
                if len(vals) >= 4:
                    writer.writerow(vals[:4])
    for suffix in ("force", "global_force", "local_force", "basic_force"):
        force_out = out_dir / f"selected_element_1_{suffix}.out"
        force_csv = out_dir / f"selected_element_1_{suffix}.csv"
        if not force_out.exists():
            continue
        rows = [line.split() for line in force_out.read_text(encoding="utf-8", errors="ignore").splitlines()]
        rows = [row for row in rows if len(row) >= 2]
        if not rows:
            continue
        width = max(len(row) for row in rows)
        with force_csv.open("w", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(["time"] + [f"f{i}" for i in range(width - 1)])
            for row in rows:
                writer.writerow(row + [""] * (width - len(row)))


def main() -> int:
    p = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    eq_dir = root / "data/input/earthquakes/Japan2011/Tsukidate-MYG004"
    p.add_argument("--output-dir", default=str(root / "data/output/opensees_lshaped_16storey"))
    p.add_argument("--eq-ns", default=str(eq_dir / "MYG0041103111446.NS"))
    p.add_argument("--eq-ew", default=str(eq_dir / "MYG0041103111446.EW"))
    p.add_argument("--eq-ud", default=str(eq_dir / "MYG0041103111446.UD"))
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--start-time", type=float, default=40.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--duration", type=float, default=1.5)
    p.add_argument("--include-vertical", action="store_true")
    p.add_argument("--rayleigh-xi", type=float, default=0.05)
    p.add_argument("--rayleigh-t1", type=float, default=1.60)
    p.add_argument("--rayleigh-t3", type=float, default=0.40)
    p.add_argument(
        "--beam-element-family",
        choices=("force", "force-shear", "disp", "elastic", "elastic-timoshenko"),
        default="disp",
    )
    p.add_argument("--elasticized-fiber-sections", action="store_true")
    p.add_argument(
        "--concrete-model",
        choices=("concrete01", "falln-kent-park-proxy"),
        default="concrete01",
        help=(
            "Concrete01 preserves the legacy OpenSees comparator; "
            "falln-kent-park-proxy uses Concrete02 with separate confined "
            "column core and a tensile cracking proxy matching fall_n's "
            "Kent-Park envelope as closely as standard OpenSees materials allow."
        ),
    )
    p.add_argument(
        "--concrete-tension-ratio",
        type=float,
        default=CONCRETE_TENSION_RATIO,
        help=(
            "Concrete02 tensile strength ratio ft/f'c used by the fall_n "
            "Kent-Park proxy.  Only affects --concrete-model "
            "falln-kent-park-proxy."
        ),
    )
    p.add_argument(
        "--concrete-ets-ratio",
        type=float,
        default=1.0,
        help=(
            "Concrete02 tension-softening slope ratio Ets/Ec.  Values below "
            "one regularize the OpenSees tensile proxy and avoid the nearly "
            "instantaneous post-cracking drop produced by Ets=Ec."
        ),
    )
    p.add_argument("--geom-transf", choices=("linear", "pdelta"), default="linear")
    p.add_argument("--element-iterations", type=int, default=20)
    p.add_argument("--element-tolerance", type=float, default=1.0e-12)
    p.add_argument("--test-tolerance", type=float, default=1.0e-5)
    p.add_argument("--unbalance-tolerance", type=float, default=1.0e-4)
    p.add_argument("--energy-tolerance", type=float, default=1.0e-7)
    p.add_argument("--test-iterations", type=int, default=80)
    p.add_argument("--floor-mass", type=float, default=0.12)
    p.add_argument("--mass-model", choices=("nodal", "element"), default="nodal")
    p.add_argument("--element-mass-form", choices=("lumped", "consistent"), default="consistent")
    p.add_argument("--member-subdivisions", type=int, default=1)
    p.add_argument("--gravity-load", type=float, default=0.005)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-gravity-only", action="store_true")
    p.add_argument("--eigen-modes", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output_dir)
    if args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        write_empty_recorders(out)
        write_manifest(out / "opensees_lshaped_16storey_manifest.json", {
            "schema": "opensees_lshaped_16storey_global_reference_v1",
            "status": "dry_run",
            "model": "forceBeamColumn_fiber_sections",
            "concrete_model": args.concrete_model,
            "section_axis_convention": "fall_n: y=width_b, z=height_h",
            "components": ["NS", "EW"],
            "outputs": [
                "roof_displacement.csv",
                "story_drift.csv",
                "base_shear.csv",
                "section_critical.csv",
            ],
        })
        return 0

    try:
        result = build_and_run(args)
    except ModuleNotFoundError:
        out.mkdir(parents=True, exist_ok=True)
        write_empty_recorders(out)
        result = {
            "schema": "opensees_lshaped_16storey_global_reference_v1",
            "status": "openseespy_unavailable",
            "message": "Install openseespy to run the external reference.",
        }
    convert_recorders(out)
    write_manifest(out / "opensees_lshaped_16storey_manifest.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
