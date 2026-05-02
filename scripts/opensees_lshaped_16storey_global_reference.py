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
BM_B = 0.30
BM_H = 0.60
BM_FPC = 25.0
NU_RC = 0.20
STEEL_FY = 420.0
STEEL_E = 200000.0
STEEL_B = 0.01


def concrete_ec(fpc: float) -> float:
    return 4700.0 * math.sqrt(abs(fpc))


def rectangular_elastic_props(b: float, h: float, fpc: float) -> tuple[float, float, float, float, float, float]:
    E = concrete_ec(fpc)
    G = E / (2.0 * (1.0 + NU_RC))
    A = b * h
    Iy = h * b**3 / 12.0
    Iz = b * h**3 / 12.0
    J = Iy + Iz
    return A, E, G, J, Iy, Iz


def active_grid_point(ix: int, iy: int) -> bool:
    return not (
        CUTOUT_X_START <= ix < CUTOUT_X_END
        and CUTOUT_Y_START <= iy < CUTOUT_Y_END
    )


def story_range(story: int) -> int:
    if story < 5:
        return 0
    if story < 11:
        return 1
    return 2


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
                else:
                    mass = args.floor_mass / max(1, len(X_GRID) * len(Y_GRID))
                    ops.mass(tag, mass, mass, mass, 0.0, 0.0, 0.0)
                tag += 1

    # Materials. Negative strengths follow OpenSees concrete convention.
    mat_tag = 1
    steel_tag = mat_tag
    ops.uniaxialMaterial("Steel02", steel_tag, STEEL_FY, STEEL_E, STEEL_B)
    mat_tag += 1
    concrete_tags: list[int] = []
    for fpc in COL_FPC + [BM_FPC]:
        tag_c = mat_tag
        ec0 = -0.002
        fpcu = -0.20 * fpc
        ecu = -0.006
        ops.uniaxialMaterial("Concrete01", tag_c, -fpc, ec0, fpcu, ecu)
        concrete_tags.append(tag_c)
        mat_tag += 1

    def make_fiber_section(sec_tag: int, b: float, h: float, conc_tag: int) -> None:
        gj = 0.10 * 4700.0 * math.sqrt(abs(COL_FPC[0])) * b * h * (b * b + h * h) / 12.0
        ops.section("Fiber", sec_tag, "-GJ", max(gj, 1.0e-6))
        ops.patch("rect", conc_tag, 10, 10, -h / 2, -b / 2, h / 2, b / 2)
        cover = 0.04
        area = math.pi * (0.016 / 2) ** 2
        y = h / 2 - cover
        z = b / 2 - cover
        ops.layer("straight", steel_tag, 2, area, -y, -z, y, -z)
        ops.layer("straight", steel_tag, 2, area, -y, z, y, z)

    section_tags = {}
    for r in range(3):
        sec = 100 + r
        make_fiber_section(sec, COL_B[r], COL_H[r], concrete_tags[r])
        section_tags[("col", r)] = sec
    beam_sec = 200
    make_fiber_section(beam_sec, BM_B, BM_H, concrete_tags[-1])

    transf_kind = "Linear" if args.geom_transf == "linear" else "PDelta"
    ops.geomTransf(transf_kind, 1, 1.0, 0.0, 0.0)
    ops.geomTransf(transf_kind, 2, 0.0, 0.0, 1.0)
    ops.geomTransf(transf_kind, 3, 0.0, 0.0, 1.0)
    ops.beamIntegration("Lobatto", 1, section_tags[("col", 0)], 5)
    ops.beamIntegration("Lobatto", 2, section_tags[("col", 1)], 5)
    ops.beamIntegration("Lobatto", 3, section_tags[("col", 2)], 5)
    ops.beamIntegration("Lobatto", 4, beam_sec, 5)

    ele = 1
    critical_elements: list[int] = []
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
        if args.beam_element_family == "force":
            if args.element_iterations > 0:
                ops.element(
                    "forceBeamColumn",
                    ele_tag,
                    ni,
                    nj,
                    transf,
                    integ,
                    "-iter",
                    args.element_iterations,
                    args.element_tolerance,
                )
            else:
                ops.element("forceBeamColumn", ele_tag, ni, nj, transf, integ)
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
        else:
            raise ValueError(f"Unsupported beam element family: {args.beam_element_family}")

    for k in range(NUM_STORIES):
        r = story_range(k)
        for ix, _x in enumerate(X_GRID):
            for iy, _y in enumerate(Y_GRID):
                if not active_grid_point(ix, iy):
                    continue
                n1 = node_id[(ix, iy, k)]
                n2 = node_id[(ix, iy, k + 1)]
                add_frame_element(ele, n1, n2, 1, r + 1, COL_B[r], COL_H[r], COL_FPC[r])
                if k < 3:
                    critical_elements.append(ele)
                ele += 1

    for k in range(1, NUM_STORIES + 1):
        for iy, _y in enumerate(Y_GRID):
            active_ix = [ix for ix in range(len(X_GRID)) if active_grid_point(ix, iy)]
            for ix_a, ix_b in zip(active_ix, active_ix[1:]):
                if (ix_a, iy, k) in node_id and (ix_b, iy, k) in node_id:
                    add_frame_element(ele, node_id[(ix_a, iy, k)], node_id[(ix_b, iy, k)], 2, 4, BM_B, BM_H, BM_FPC)
                    ele += 1
        for ix, _x in enumerate(X_GRID):
            active_iy = [iy for iy in range(len(Y_GRID)) if active_grid_point(ix, iy)]
            for iy_a, iy_b in zip(active_iy, active_iy[1:]):
                if (ix, iy_a, k) in node_id and (ix, iy_b, k) in node_id:
                    add_frame_element(ele, node_id[(ix, iy_a, k)], node_id[(ix, iy_b, k)], 3, 4, BM_B, BM_H, BM_FPC)
                    ele += 1

    roof_node = max(node_id.values())
    ops.recorder("Node", "-file", str(out / "roof_displacement.out"), "-time", "-node", roof_node, "-dof", 1, 2, 3, "disp")
    ops.recorder("Node", "-file", str(out / "base_reaction.out"), "-time", "-node", *[node_id[key] for key in node_id if key[2] == 0], "-dof", 1, 2, 3, "reaction")
    for e in critical_elements[:10]:
        ops.recorder("Element", "-file", str(out / f"section_element_{e}.out"), "-time", "-ele", e, "section", 1, "force")

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

    if args.dry_gravity_only:
        return {
            "status": "gravity_only",
            "gravity_ok": gravity_ok,
            "node_count": len(node_id),
            "element_count": ele - 1,
        }

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
        if args.beam_element_family == "elastic":
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
        "geom_transf": args.geom_transf,
        "include_vertical": args.include_vertical,
        "rayleigh_alpha_m": alpha_m,
        "rayleigh_beta_k": beta_k,
        "node_count": len(node_id),
        "element_count": ele - 1,
        "roof_node": roof_node,
        "critical_elements_recorded": critical_elements[:10],
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
    p.add_argument("--beam-element-family", choices=("force", "disp", "elastic"), default="disp")
    p.add_argument("--geom-transf", choices=("linear", "pdelta"), default="linear")
    p.add_argument("--element-iterations", type=int, default=20)
    p.add_argument("--element-tolerance", type=float, default=1.0e-12)
    p.add_argument("--test-tolerance", type=float, default=1.0e-5)
    p.add_argument("--unbalance-tolerance", type=float, default=1.0e-4)
    p.add_argument("--energy-tolerance", type=float, default=1.0e-7)
    p.add_argument("--test-iterations", type=int, default=80)
    p.add_argument("--floor-mass", type=float, default=0.12)
    p.add_argument("--gravity-load", type=float, default=0.005)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dry-gravity-only", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    if args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        write_empty_recorders(out)
        write_manifest(out / "opensees_lshaped_16storey_manifest.json", {
            "schema": "opensees_lshaped_16storey_global_reference_v1",
            "status": "dry_run",
            "model": "forceBeamColumn_fiber_sections",
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
