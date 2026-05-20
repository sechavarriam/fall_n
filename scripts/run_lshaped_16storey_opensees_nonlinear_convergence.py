#!/usr/bin/env python3
"""Run a nodal-mass OpenSeesPy nonlinear convergence study for the frame.

The goal is deliberately structural and single-scale: close the nonlinear
OpenSees comparator against the promoted fall_n global response, without FE2.
All publication candidates keep the mass model that already matched fall_n in
the elastic dynamic audit: nodal masses, physical MYG004 scale 1.0, window
[87.65, 97.65] s, and vertical excitation enabled.

The sweep starts from the most promising nonlinear OpenSees family found in
the previous audit:

  forceBeamColumn + shear aggregator + Concrete02 fall_n Kent-Park proxy
  with regularized tensile softening Ets/Ec = 0.05.

It can also test modest mesh refinement through member subdivisions before a
very fine OpenSees model is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FALLN_PRIMARY_NODAL_TOTAL_MASS_PER_DIRECTION = 0.8865509419199957
OPENSEES_ACTIVE_NODES_PER_FLOOR = 14
OPENSEES_GRID_POINTS_PER_FLOOR = 20
OPENSEES_DYNAMIC_FLOORS = 16
MASSMATCHED_FLOOR_MASS = (
    FALLN_PRIMARY_NODAL_TOTAL_MASS_PER_DIRECTION
    * OPENSEES_GRID_POINTS_PER_FLOOR
    / (OPENSEES_ACTIVE_NODES_PER_FLOOR * OPENSEES_DYNAMIC_FLOORS)
)
DEFAULT_FALLN_ROOF_NODE_ID = 329
DEFAULT_OPENSEES_ROOF_NODE_ID = 238
DEFAULT_ROOF_POINT_COORDS_M = (20.0, 4.0, 51.2)


@dataclass(frozen=True)
class CaseSpec:
    label: str
    output_name: str
    duration: float
    timeout_s: float
    beam_element_family: str = "force-shear"
    beam_member_element_family: str = "same"
    beam_x_member_element_family: str = "same"
    beam_y_member_element_family: str = "same"
    member_subdivisions: int = 1
    geom_transf: str = "linear"
    rayleigh_stiffness: str = "current"
    rayleigh_xi: float = 0.05
    rayleigh_t1: float = 1.60
    rayleigh_t3: float = 0.40
    concrete_ets_ratio: float = 0.05
    concrete_tension_ratio: float = 0.10
    element_iterations: int = 40
    element_tolerance: float = 1.0e-8
    test_tolerance: float = 1.0e-5
    unbalance_tolerance: float = 1.0e-4
    energy_tolerance: float = 1.0e-7
    test_iterations: int = 80
    recovery_substeps: str = "2,4,8,16,32"
    floor_mass: float = MASSMATCHED_FLOOR_MASS
    section_refinement: int = 1
    shear_scale_vy: float = 1.0
    shear_scale_vz: float = 1.0


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run short/medium OpenSees nonlinear global comparator studies "
            "with nodal masses and summarize them against fall_n."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with OpenSeesPy available, e.g. py -3.12 path.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo
        / "data/output/opensees_lshaped_16storey_nonlinear_convergence_massmatched_20260518",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo / "doc/figures/validation_reboot",
    )
    parser.add_argument(
        "--falln-reference",
        type=Path,
        default=repo
        / "data/output/lshaped_16storey_global_only_timegrid_10s_20260511/recorders/roof_displacement_global_reference.csv",
    )
    parser.add_argument(
        "--falln-roof-node-id",
        type=int,
        default=DEFAULT_FALLN_ROOF_NODE_ID,
        help=(
            "fall_n roof node id used as the point observable for the "
            "OpenSees comparison. The default node329 is the same physical "
            "roof point recorded by OpenSees node238: (20, 4, 51.2) m."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=("smoke", "medium", "extended", "shear", "pdelta", "publication"),
        default="smoke",
        help=(
            "smoke: 1 s candidates; medium: 2 s candidates; extended: "
            "4 s material/section candidates; shear: 2 s Vy/Vz Aggregator "
            "scale sweep; pdelta: 4 s second-order geometry candidates; "
            "publication: the current best 10 s candidate."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed cases.")
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only summarize existing outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="lshaped_16_opensees_nonlinear_convergence",
    )
    parser.add_argument(
        "--case-filter",
        default="",
        help=(
            "Comma-separated substrings used to run/summarize only matching "
            "case output names or labels. Empty means all preset cases."
        ),
    )
    return parser.parse_args()


def cases_for_preset(preset: str) -> list[CaseSpec]:
    if preset == "smoke":
        return [
            CaseSpec(
                label="force-shear Ets/Ec=0.05 sub1 1.5s",
                output_name="fs_ets005_sub1_1p5s",
                duration=1.5,
                timeout_s=900.0,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.05 sub2 1.5s",
                output_name="fs_ets005_sub2_1p5s",
                duration=1.5,
                timeout_s=1200.0,
                member_subdivisions=2,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.02 sub1 1.5s",
                output_name="fs_ets002_sub1_1p5s",
                duration=1.5,
                timeout_s=900.0,
                concrete_ets_ratio=0.02,
            ),
            CaseSpec(
                label="dispBeam Ets/Ec=0.05 sub1 1.5s",
                output_name="disp_ets005_sub1_1p5s",
                duration=1.5,
                timeout_s=1200.0,
                beam_element_family="disp",
            ),
        ]
    if preset == "medium":
        return [
            CaseSpec(
                label="force-shear Ets/Ec=0.05 sub1 2s",
                output_name="fs_ets005_sub1_2s",
                duration=2.0,
                timeout_s=1800.0,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.05 sub2 2s",
                output_name="fs_ets005_sub2_2s",
                duration=2.0,
                timeout_s=2400.0,
                member_subdivisions=2,
            ),
            CaseSpec(
                label="force-shear Rayleigh initial Ets/Ec=0.05 sub1 2s",
                output_name="fs_ets005_sub1_rayleigh_initial_2s",
                duration=2.0,
                timeout_s=1800.0,
                rayleigh_stiffness="initial",
            ),
            CaseSpec(
                label="force-shear Rayleigh committed Ets/Ec=0.05 sub1 2s",
                output_name="fs_ets005_sub1_rayleigh_committed_2s",
                duration=2.0,
                timeout_s=1800.0,
                rayleigh_stiffness="committed",
            ),
            CaseSpec(
                label="force-shear Rayleigh initial Ets/Ec=0.05 sub2 2s",
                output_name="fs_ets005_sub2_rayleigh_initial_2s",
                duration=2.0,
                timeout_s=2400.0,
                member_subdivisions=2,
                rayleigh_stiffness="initial",
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.02 sub1 2s",
                output_name="fs_ets002_sub1_2s",
                duration=2.0,
                timeout_s=1800.0,
                concrete_ets_ratio=0.02,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 sub1 2s",
                output_name="fs_ets008_sub1_2s",
                duration=2.0,
                timeout_s=1800.0,
                concrete_ets_ratio=0.08,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.10 sub1 2s",
                output_name="fs_ets010_sub1_2s",
                duration=2.0,
                timeout_s=1800.0,
                concrete_ets_ratio=0.10,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.05 section-ref2 sub1 2s",
                output_name="fs_ets005_sub1_section_ref2_2s",
                duration=2.0,
                timeout_s=2400.0,
                section_refinement=2,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.02 section-ref2 sub1 2s",
                output_name="fs_ets002_sub1_section_ref2_2s",
                duration=2.0,
                timeout_s=2400.0,
                concrete_ets_ratio=0.02,
                section_refinement=2,
            ),
            CaseSpec(
                label="force-shear columns + disp beams Ets/Ec=0.05 sub1 2s",
                output_name="fs_ets005_sub1_beamdisp_2s",
                duration=2.0,
                timeout_s=1800.0,
                beam_member_element_family="disp",
            ),
            CaseSpec(
                label="force-shear columns + elastic Timoshenko beams Ets/Ec=0.05 sub1 2s",
                output_name="fs_ets005_sub1_beamtimo_2s",
                duration=2.0,
                timeout_s=1800.0,
                beam_member_element_family="elastic-timoshenko",
            ),
            CaseSpec(
                label="force-shear with disp Y-beams Ets/Ec=0.05 sub1 2s",
                output_name="fs_ets005_sub1_ybeamdisp_2s",
                duration=2.0,
                timeout_s=1800.0,
                beam_y_member_element_family="disp",
            ),
            CaseSpec(
                label="dispBeam Ets/Ec=0.05 sub1 2s",
                output_name="disp_ets005_sub1_2s",
                duration=2.0,
                timeout_s=2400.0,
                beam_element_family="disp",
            ),
        ]
    if preset == "extended":
        return [
            CaseSpec(
                label="force-shear Ets/Ec=0.05 sub1 4s",
                output_name="fs_ets005_sub1_4s",
                duration=4.0,
                timeout_s=3600.0,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 sub1 4s",
                output_name="fs_ets008_sub1_4s",
                duration=4.0,
                timeout_s=5400.0,
                concrete_ets_ratio=0.08,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.10 sub1 4s",
                output_name="fs_ets010_sub1_4s",
                duration=4.0,
                timeout_s=7200.0,
                concrete_ets_ratio=0.10,
            ),
        ]
    if preset == "shear":
        return [
            CaseSpec(
                label="force-shear Ets/Ec=0.08 shear Vy=Vz=0.50 2s",
                output_name="fs_ets008_shear050_2s",
                duration=2.0,
                timeout_s=2400.0,
                concrete_ets_ratio=0.08,
                shear_scale_vy=0.50,
                shear_scale_vz=0.50,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 shear Vy=Vz=1.00 2s",
                output_name="fs_ets008_shear100_2s",
                duration=2.0,
                timeout_s=2400.0,
                concrete_ets_ratio=0.08,
                shear_scale_vy=1.00,
                shear_scale_vz=1.00,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 shear Vy=Vz=1.50 2s",
                output_name="fs_ets008_shear150_2s",
                duration=2.0,
                timeout_s=2400.0,
                concrete_ets_ratio=0.08,
                shear_scale_vy=1.50,
                shear_scale_vz=1.50,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 shear Vy=Vz=2.00 2s",
                output_name="fs_ets008_shear200_2s",
                duration=2.0,
                timeout_s=2400.0,
                concrete_ets_ratio=0.08,
                shear_scale_vy=2.00,
                shear_scale_vz=2.00,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 shear Vy=Vz=4.00 2s",
                output_name="fs_ets008_shear400_2s",
                duration=2.0,
                timeout_s=2400.0,
                concrete_ets_ratio=0.08,
                shear_scale_vy=4.00,
                shear_scale_vz=4.00,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 shear Vy=Vz=0.50 4s",
                output_name="fs_ets008_shear050_4s",
                duration=4.0,
                timeout_s=5400.0,
                concrete_ets_ratio=0.08,
                shear_scale_vy=0.50,
                shear_scale_vz=0.50,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 shear Vy=Vz=1.00 4s",
                output_name="fs_ets008_shear100_4s",
                duration=4.0,
                timeout_s=5400.0,
                concrete_ets_ratio=0.08,
                shear_scale_vy=1.00,
                shear_scale_vz=1.00,
            ),
        ]
    if preset == "pdelta":
        return [
            CaseSpec(
                label="force-shear Ets/Ec=0.08 linear shear Vy=Vz=0.50 4s",
                output_name="fs_ets008_shear050_linear_4s",
                duration=4.0,
                timeout_s=5400.0,
                concrete_ets_ratio=0.08,
                shear_scale_vy=0.50,
                shear_scale_vz=0.50,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 PDelta shear Vy=Vz=0.50 4s",
                output_name="fs_ets008_shear050_pdelta_4s",
                duration=4.0,
                timeout_s=5400.0,
                geom_transf="pdelta",
                concrete_ets_ratio=0.08,
                shear_scale_vy=0.50,
                shear_scale_vz=0.50,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.08 PDelta shear Vy=Vz=1.00 4s",
                output_name="fs_ets008_shear100_pdelta_4s",
                duration=4.0,
                timeout_s=5400.0,
                geom_transf="pdelta",
                concrete_ets_ratio=0.08,
                shear_scale_vy=1.00,
                shear_scale_vz=1.00,
            ),
            CaseSpec(
                label="force-shear Ets/Ec=0.05 PDelta shear Vy=Vz=1.00 4s",
                output_name="fs_ets005_shear100_pdelta_4s",
                duration=4.0,
                timeout_s=5400.0,
                geom_transf="pdelta",
                concrete_ets_ratio=0.05,
                shear_scale_vy=1.00,
                shear_scale_vz=1.00,
            ),
        ]
    return [
        CaseSpec(
            label="force-shear Ets/Ec=0.05 sub1 10s publication",
            output_name="fs_ets005_sub1_10s_publication",
            duration=10.0,
            timeout_s=0.0,
        ),
        CaseSpec(
            label="force-shear columns + disp beams Ets/Ec=0.05 sub1 10s publication",
            output_name="fs_ets005_sub1_beamdisp_10s_publication",
            duration=10.0,
            timeout_s=0.0,
            beam_member_element_family="disp",
        ),
        CaseSpec(
            label="force-shear Ets/Ec=0.05 sub1 10s deep cutback",
            output_name="fs_ets005_sub1_deepcutback_10s_publication",
            duration=10.0,
            timeout_s=0.0,
            element_iterations=80,
            element_tolerance=1.0e-6,
            recovery_substeps="2,4,8,16,32,64,128",
        ),
        CaseSpec(
            label="force-shear Ets/Ec=0.08 PDelta shear Vy=Vz=0.50 10s material-mapped deep cutback",
            output_name="fs_ets008_shear050_pdelta_materialmapped_deepcutback_10s_publication",
            duration=10.0,
            timeout_s=0.0,
            geom_transf="pdelta",
            concrete_ets_ratio=0.08,
            shear_scale_vy=0.50,
            shear_scale_vz=0.50,
            element_iterations=80,
            element_tolerance=1.0e-6,
            recovery_substeps="2,4,8,16,32,64,128",
        ),
        CaseSpec(
            label="force-shear with disp Y-beams Ets/Ec=0.05 sub1 10s publication",
            output_name="fs_ets005_sub1_ybeamdisp_10s_publication",
            duration=10.0,
            timeout_s=0.0,
            beam_y_member_element_family="disp",
        )
    ]


def command_for_case(args: argparse.Namespace, repo: Path, case: CaseSpec, out: Path) -> list[str]:
    return [
        args.python,
        str(repo / "scripts/opensees_lshaped_16storey_global_reference.py"),
        "--output-dir",
        str(out),
        "--scale",
        "1.0",
        "--start-time",
        "87.65",
        "--dt",
        "0.02",
        "--duration",
        str(case.duration),
        "--include-vertical",
        "--beam-element-family",
        case.beam_element_family,
        "--beam-member-element-family",
        case.beam_member_element_family,
        "--beam-x-member-element-family",
        case.beam_x_member_element_family,
        "--beam-y-member-element-family",
        case.beam_y_member_element_family,
        "--concrete-model",
        "falln-kent-park-proxy",
        "--concrete-tension-ratio",
        str(case.concrete_tension_ratio),
        "--concrete-ets-ratio",
        str(case.concrete_ets_ratio),
        "--element-iterations",
        str(case.element_iterations),
        "--element-tolerance",
        str(case.element_tolerance),
        "--test-tolerance",
        str(case.test_tolerance),
        "--unbalance-tolerance",
        str(case.unbalance_tolerance),
        "--energy-tolerance",
        str(case.energy_tolerance),
        "--test-iterations",
        str(case.test_iterations),
        "--recovery-substeps",
        case.recovery_substeps,
        "--floor-mass",
        str(case.floor_mass),
        "--mass-model",
        "nodal",
        "--member-subdivisions",
        str(case.member_subdivisions),
        "--shear-scale-vy",
        str(case.shear_scale_vy),
        "--shear-scale-vz",
        str(case.shear_scale_vz),
        "--section-refinement",
        str(case.section_refinement),
        "--geom-transf",
        case.geom_transf,
        "--rayleigh-stiffness",
        case.rayleigh_stiffness,
        "--rayleigh-xi",
        str(case.rayleigh_xi),
        "--rayleigh-t1",
        str(case.rayleigh_t1),
        "--rayleigh-t3",
        str(case.rayleigh_t3),
    ]


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {}


def completed(path: Path) -> bool:
    manifest = read_manifest(path / "opensees_lshaped_16storey_manifest.json")
    return manifest.get("status") == "completed"


def run_case(args: argparse.Namespace, repo: Path, case: CaseSpec) -> dict[str, Any]:
    out = args.output_root / case.output_name
    out.mkdir(parents=True, exist_ok=True)
    command = command_for_case(args, repo, case, out)
    result: dict[str, Any] = {
        "label": case.label,
        "output_dir": str(out),
        "case": asdict(case),
        "command": command,
        "skipped_existing": False,
    }
    if args.no_run or (completed(out) and not args.force):
        result["skipped_existing"] = True
        return result
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
            timeout=None if case.timeout_s <= 0.0 else case.timeout_s,
        )
        result["return_code"] = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        result["return_code"] = None
        result["timed_out"] = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
    result["wall_seconds_driver"] = time.perf_counter() - t0
    (out / "driver_stdout.log").write_text(stdout or "", encoding="utf-8")
    (out / "driver_stderr.log").write_text(stderr or "", encoding="utf-8")
    return result


def read_roof(path: Path, *, falln_node_id: int | None = None) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        for raw in csv.DictReader(handle):
            keys = raw.keys()
            if {"ux", "uy", "uz"}.issubset(keys):
                cols = ("ux", "uy", "uz")
            elif falln_node_id is not None:
                cols = (
                    f"node{falln_node_id}_dof0",
                    f"node{falln_node_id}_dof1",
                    f"node{falln_node_id}_dof2",
                )
            else:
                cols = tuple(list(keys)[-3:])
            try:
                rows.append(
                    {
                        "time": float(raw["time"]),
                        "ux": float(raw[cols[0]]),
                        "uy": float(raw[cols[1]]),
                        "uz": float(raw[cols[2]]),
                    }
                )
            except (KeyError, ValueError, IndexError):
                continue
    return rows


def interp(rows: list[dict[str, float]], t: float, comp: str) -> float:
    if not rows:
        return math.nan
    if t <= rows[0]["time"]:
        return rows[0][comp]
    if t >= rows[-1]["time"]:
        return rows[-1][comp]
    lo = 0
    hi = len(rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rows[mid]["time"] <= t:
            lo = mid
        else:
            hi = mid
    a = rows[lo]
    b = rows[hi]
    span = b["time"] - a["time"]
    if span <= 0.0:
        return a[comp]
    alpha = (t - a["time"]) / span
    return (1.0 - alpha) * a[comp] + alpha * b[comp]


def roof_metrics(case_rows: list[dict[str, float]], ref_rows: list[dict[str, float]]) -> dict[str, float]:
    metrics: dict[str, float] = {
        "roof_samples": float(len(case_rows)),
        "last_recorded_time_s": case_rows[-1]["time"] if case_rows else math.nan,
        "comparison_samples": 0.0,
        "comparison_start_s": math.nan,
        "comparison_end_s": math.nan,
    }
    if not case_rows:
        return metrics
    if ref_rows:
        ref_start = ref_rows[0]["time"]
        ref_end = ref_rows[-1]["time"]
        overlap_rows = [row for row in case_rows if ref_start <= row["time"] <= ref_end]
        if overlap_rows:
            metrics["comparison_samples"] = float(len(overlap_rows))
            metrics["comparison_start_s"] = overlap_rows[0]["time"]
            metrics["comparison_end_s"] = overlap_rows[-1]["time"]
    else:
        overlap_rows = []
    for comp in ("ux", "uy", "uz"):
        values = [row[comp] for row in case_rows]
        metrics[f"peak_abs_{comp}_m"] = max(abs(v) for v in values)
        metrics[f"final_{comp}_m"] = values[-1]
        errors = []
        for row in overlap_rows:
            ref = interp(ref_rows, row["time"], comp)
            if math.isfinite(ref):
                errors.append(row[comp] - ref)
        if errors:
            rms = math.sqrt(sum(e * e for e in errors) / len(errors))
            ref_peak = max(
                abs(interp(ref_rows, row["time"], comp)) for row in overlap_rows
            )
            metrics[f"rms_vs_falln_{comp}_m"] = rms
            metrics[f"peak_normalized_rms_vs_falln_{comp}"] = (
                rms / ref_peak if math.isfinite(ref_peak) and ref_peak > 1.0e-14 else math.nan
            )
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    available = [row for row in rows if row.get("roof_samples", 0) > 0]
    if not available:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    labels = [row["label"] for row in available]
    progress = [row.get("last_recorded_time_s", math.nan) for row in available]
    rms_ux = [row.get("rms_vs_falln_ux_m", math.nan) for row in available]
    rms_uy = [row.get("rms_vs_falln_uy_m", math.nan) for row in available]

    plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    x = list(range(len(labels)))
    axes[0].bar(x, progress, color="#2563eb")
    axes[0].set_ylabel("accepted dynamic window [s]")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[1].plot(x, rms_ux, marker="o", label="ux")
    axes[1].plot(x, rms_uy, marker="s", label="uy")
    axes[1].set_ylabel("RMS vs fall_n [m]")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    axes[1].legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = args.figures_dir / f"{args.prefix}_{args.preset}_summary.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def plot_roof_overlays(rows: list[dict[str, Any]], ref_rows: list[dict[str, float]], args: argparse.Namespace) -> list[str]:
    available = [
        (row, read_roof(Path(row["output_dir"]) / "roof_displacement.csv"))
        for row in rows
        if row.get("roof_samples", 0) > 0
    ]
    available = [(row, roof) for row, roof in available if roof]
    if not available or not ref_rows:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    plt.rcParams.update({"font.family": "serif", "axes.grid": True, "grid.alpha": 0.25})

    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.6), sharex=True)
    labels = (r"$u_x$ [m]", r"$u_y$ [m]", r"$u_z$ [m]")
    for ax, comp, ylabel in zip(axes, ("ux", "uy", "uz"), labels):
        ax.plot(
            [r["time"] for r in ref_rows],
            [r[comp] for r in ref_rows],
            color="black",
            linewidth=1.6,
            label=f"fall_n node{args.falln_roof_node_id}",
        )
        for row, roof in available:
            ax.plot(
                [r["time"] for r in roof],
                [r[comp] for r in roof],
                linewidth=1.1,
                label=row["label"],
            )
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("tiempo relativo en la ventana [s]")
    axes[0].legend(loc="best", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = args.figures_dir / f"{args.prefix}_{args.preset}_roof_components.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        outputs.append(str(path))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    ax.plot(
        [r["ux"] for r in ref_rows],
        [r["uy"] for r in ref_rows],
        color="black",
        linewidth=1.6,
        label=f"fall_n node{args.falln_roof_node_id}",
    )
    for row, roof in available:
        ax.plot(
            [r["ux"] for r in roof],
            [r["uy"] for r in roof],
            linewidth=1.1,
            label=row["label"],
        )
    ax.set_xlabel(r"$u_x$ [m]")
    ax.set_ylabel(r"$u_y$ [m]")
    ax.set_title("Orbita de cubierta en planta")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = args.figures_dir / f"{args.prefix}_{args.preset}_plan_orbit.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output_root.mkdir(parents=True, exist_ok=True)
    ref_rows = read_roof(args.falln_reference, falln_node_id=args.falln_roof_node_id)
    if not ref_rows:
        raise SystemExit(f"Missing fall_n reference roof displacement: {args.falln_reference}")

    cases = cases_for_preset(args.preset)
    filters = [item.strip().lower() for item in args.case_filter.split(",") if item.strip()]
    if filters:
        cases = [
            case
            for case in cases
            if any(
                token in case.output_name.lower() or token in case.label.lower()
                for token in filters
            )
        ]
    if not cases:
        raise SystemExit("No cases selected by --case-filter.")
    runs = [run_case(args, repo, case) for case in cases]
    rows: list[dict[str, Any]] = []
    for run in runs:
        out = Path(run["output_dir"])
        manifest = read_manifest(out / "opensees_lshaped_16storey_manifest.json")
        roof_rows = read_roof(out / "roof_displacement.csv")
        row: dict[str, Any] = {
            "label": run["label"],
            "output_dir": run["output_dir"],
            "skipped_existing": run.get("skipped_existing", False),
            "return_code": run.get("return_code", ""),
            "timed_out": run.get("timed_out", False),
            "wall_seconds_driver": run.get("wall_seconds_driver", ""),
            "status": manifest.get("status", "no_manifest"),
            "accepted_steps": manifest.get("accepted_steps", ""),
            "requested_steps": manifest.get("requested_steps", ""),
            "duration_s": manifest.get("duration", run["case"]["duration"]),
            "beam_element_family": manifest.get("beam_element_family", run["case"]["beam_element_family"]),
            "beam_member_element_family": manifest.get("beam_member_element_family", run["case"]["beam_member_element_family"]),
            "beam_x_member_element_family": manifest.get("beam_x_member_element_family", run["case"]["beam_x_member_element_family"]),
            "beam_y_member_element_family": manifest.get("beam_y_member_element_family", run["case"]["beam_y_member_element_family"]),
            "member_subdivisions": manifest.get("member_subdivisions", run["case"]["member_subdivisions"]),
            "geom_transf": manifest.get("geom_transf", run["case"]["geom_transf"]),
            "rayleigh_stiffness": manifest.get("rayleigh_stiffness", run["case"]["rayleigh_stiffness"]),
            "rayleigh_alpha_m": manifest.get("rayleigh_alpha_m", ""),
            "rayleigh_beta_k": manifest.get("rayleigh_beta_k", ""),
            "rayleigh_beta_k_init": manifest.get("rayleigh_beta_k_init", ""),
            "rayleigh_beta_k_comm": manifest.get("rayleigh_beta_k_comm", ""),
            "mass_model": manifest.get("mass_model", "nodal"),
            "floor_mass": manifest.get("floor_mass", run["case"]["floor_mass"]),
            "nodal_mass_total_per_direction": manifest.get("nodal_mass_total_per_direction", ""),
            "concrete_ets_ratio": manifest.get("concrete_ets_ratio", run["case"]["concrete_ets_ratio"]),
            "section_refinement": manifest.get("section_refinement", run["case"]["section_refinement"]),
            "shear_scale_vy": manifest.get("shear_scale_vy", run["case"]["shear_scale_vy"]),
            "shear_scale_vz": manifest.get("shear_scale_vz", run["case"]["shear_scale_vz"]),
            "steel_R0": manifest.get("steel_parameters", {}).get("R0", ""),
            "steel_cR1_falln": manifest.get("steel_parameters", {}).get("cR1", ""),
            "steel_cR1_opensees": manifest.get("steel_parameters", {}).get("opensees_cR1", ""),
            "steel_cR2": manifest.get("steel_parameters", {}).get("cR2", ""),
            "element_iterations": manifest.get("element_iterations", run["case"]["element_iterations"]),
            "element_tolerance": manifest.get("element_tolerance", run["case"]["element_tolerance"]),
            "recovery_substeps": manifest.get("recovery_substeps", run["case"]["recovery_substeps"]),
            "total_wall_seconds": manifest.get("total_wall_seconds", ""),
        }
        row.update(roof_metrics(roof_rows, ref_rows))
        rows.append(row)

    csv_path = args.figures_dir / f"{args.prefix}_{args.preset}_summary.csv"
    json_path = args.figures_dir / f"{args.prefix}_{args.preset}_summary.json"
    write_csv(csv_path, rows)
    figures = plot_summary(rows, args)
    figures.extend(plot_roof_overlays(rows, ref_rows, args))
    payload = {
        "schema": "lshaped_16_opensees_nonlinear_convergence_v1",
        "preset": args.preset,
        "falln_reference": str(args.falln_reference),
        "observable": {
            "kind": "roof_node_displacement",
            "falln_node_id": args.falln_roof_node_id,
            "opensees_node_id": DEFAULT_OPENSEES_ROOF_NODE_ID,
            "coords_m": list(DEFAULT_ROOF_POINT_COORDS_M),
            "components": ["ux", "uy", "uz"],
            "note": (
                "The comparison uses one physical roof point. In fall_n this "
                "is node329_dof0/1/2 by default; in OpenSees it is node238."
            ),
        },
        "falln_reference_start_s": ref_rows[0]["time"] if ref_rows else None,
        "falln_reference_end_s": ref_rows[-1]["time"] if ref_rows else None,
        "output_root": str(args.output_root),
        "rows": rows,
        "figures": figures,
        "csv": str(csv_path),
        "notes": [
            "All cases use primary-grid mass-matched nodal mass, scale=1.0, MYG004 window [87.65, 97.65] s, dt=0.02 s, and vertical excitation.",
            "The current publication candidate is force-shear with Concrete02 fall_n proxy and Ets/Ec=0.05.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
