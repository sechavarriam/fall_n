#!/usr/bin/env python3
"""
Compare the reduced RC-column structural steel-fiber history against the
embedded-truss steel history of the continuum benchmark.

This audit is intentionally focused on physical symmetry:
  * same reduced RC-column reference spec
  * same eight-bar reinforcement layout
  * same cyclic drift protocol
  * structural steel tracked at the base-side section fiber
  * continuum steel tracked at the embedded truss GP closest to the base on
    the equivalent longitudinal bar
  * explicit structural-to-continuum section-axis mapping, because the
    structural active coordinate is not the same CSV coordinate as the prism
    rebar ly/lz pair
  * explicit embedded-bar interpolation policy instead of a hidden default

The goal is not to force numerical identity between beam and continuum
formulations. The goal is to see whether the steel stories are physically
coherent while the continuum validation matures toward larger amplitudes.

This benchmark therefore needs to stay aligned with the promoted local
continuum baseline instead of silently drifting back to older continuum
defaults.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "savefig.bbox": "tight",
        "figure.dpi": 140,
        "savefig.dpi": 300,
    }
)

BLUE = "#0b5fa5"
ORANGE = "#d97706"
GREEN = "#2f855a"


@dataclass(frozen=True)
class StructuralSteelIdentity:
    fiber_index: int
    y: float
    z: float
    area: float
    zone: str
    material_role: str


@dataclass(frozen=True)
class ContinuumSteelIdentity:
    bar_index: int
    bar_element_layer: int
    gp_index: int
    bar_y: float
    bar_z: float
    position_z_m: float


@dataclass(frozen=True)
class StructuralSteelTraceSelection:
    target_position_z_m: float
    lower_section_gp: int
    upper_section_gp: int
    lower_position_z_m: float
    upper_position_z_m: float
    interpolation_alpha: float
    fiber_index: int
    y: float
    z: float


@dataclass(frozen=True)
class StructuralOrientationCandidate:
    fiber_index: int
    y: float
    z: float
    active_coordinate_name: str
    active_coordinate_value: float
    passive_coordinate_name: str
    passive_coordinate_value: float
    elastic_rms_stress_error: float
    elastic_rms_strain_error: float


@dataclass(frozen=True)
class HingeBandDescriptor:
    length_m: float
    selection_rule: str
    note: str


def structural_to_continuum_rebar_coordinates(
    structural_y: float,
    structural_z: float,
    active_coordinate_name: str,
) -> tuple[float, float]:
    """Map structural section coordinates to continuum RebarBar coordinates.

    The structural benchmark stores steel fibers in the section plane as
    (y, z), while the continuum prism stores RebarBar::ly along the local
    x-axis and RebarBar::lz along the local y-axis. For the current lateral
    drift direction the structural active coordinate is z, but the continuum
    strain gradient is carried by ly. Positive continuum ly is therefore the
    opposite side of the structural active coordinate. Keeping this transform
    explicit prevents the comparison from pairing a physically opposite bar
    just because the coordinate labels happen to share letters.
    """

    if active_coordinate_name == "z":
        return -structural_z, structural_y
    return -structural_y, structural_z


def continuum_to_structural_coordinates(
    continuum_bar_y: float,
    continuum_bar_z: float,
    active_coordinate_name: str,
) -> tuple[float, float]:
    """Inverse of structural_to_continuum_rebar_coordinates.

    The continuum CSV fields are named bar_y/bar_z for historical reasons:
    bar_y is RebarBar::ly (local prism x) and bar_z is RebarBar::lz
    (local prism y). The returned tuple is structural (y, z).
    """

    if active_coordinate_name == "z":
        return continuum_bar_z, -continuum_bar_y
    return -continuum_bar_y, continuum_bar_z


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced RC structural-vs-continuum steel-hysteresis audit."
        )
    )
    parser.add_argument(
        "--analysis",
        choices=("monotonic", "cyclic"),
        default="cyclic",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--structural-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument(
        "--continuum-exe",
        type=Path,
        default=repo_root
        / "build"
        / "fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
    )
    parser.add_argument("--beam-nodes", type=int, default=10)
    parser.add_argument(
        "--structural-element-count",
        type=int,
        default=1,
        help=(
            "Number of structural TimoshenkoBeamN elements used by the "
            "structural reference before comparing against the continuum. "
            "Use values greater than one to resolve the plastic-hinge region "
            "with a structural mesh instead of one high-order element."
        ),
    )
    parser.add_argument(
        "--beam-integration",
        choices=("legendre", "lobatto", "radau-left", "radau-right"),
        default="lobatto",
    )
    parser.add_argument("--hex-orders", default="hex20")
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument(
        "--longitudinal-bias-power",
        type=float,
        default=1.0,
        help=(
            "Longitudinal host-mesh bias for the promoted continuum bridge. "
            "The canonical Hex20 4x4x2 interior baseline stays uniform; "
            "strongly biased meshes are tracked separately as higher-cost frontiers."
        ),
    )
    parser.add_argument(
        "--longitudinal-bias-location",
        default="fixed-end",
        choices=("fixed-end", "loaded-end", "both-ends"),
        help=(
            "Longitudinal location where the host mesh is refined when "
            "--longitudinal-bias-power > 1. fixed-end targets the plastic "
            "hinge near the support; loaded-end targets the imposed-motion "
            "face; both-ends splits refinement between both."
        ),
    )
    parser.add_argument(
        "--host-concrete-zoning-mode",
        choices=("uniform-reference", "cover-core-split"),
        default="cover-core-split",
    )
    parser.add_argument(
        "--transverse-mesh-mode",
        choices=("uniform", "cover-aligned"),
        default="cover-aligned",
    )
    parser.add_argument(
        "--continuum-rebar-layout",
        choices=(
            "structural-matched-eight-bar",
            "cover-core-interface-eight-bar",
            "boundary-matched-eight-bar",
            "enriched-twelve-bar",
        ),
        default="structural-matched-eight-bar",
    )
    parser.add_argument(
        "--continuum-concrete-profile",
        default="production-stabilized",
        choices=("benchmark-reference", "production-stabilized"),
    )
    parser.add_argument(
        "--continuum-material-mode",
        default="nonlinear",
        choices=(
            "nonlinear",
            "elasticized",
            "orthotropic-bimodular-proxy",
            "tensile-crack-band-damage-proxy",
            "cyclic-crack-band-concrete",
            "fixed-crack-band-concrete",
            "componentwise-kent-park-concrete",
        ),
        help=(
            "Concrete/host material branch used by the continuum benchmark. "
            "The structural reference remains nonlinear so this switch isolates "
            "the local-continuum host assumption."
        ),
    )
    parser.add_argument(
        "--continuum-kinematics",
        default="corotational",
        choices=(
            "small-strain",
            "total-lagrangian",
            "updated-lagrangian",
            "corotational",
        ),
        help=(
            "Continuum kinematic policy used by the local model. The promoted "
            "benchmark defaults to the corotational branch so rigid-body "
            "motion filtering is explicit in every reproduced bundle."
        ),
    )
    parser.add_argument(
        "--concrete-tension-stiffness-ratio",
        type=float,
        default=0.10,
        help=(
            "Tension/compression stiffness ratio for the low-cost concrete "
            "proxy branches."
        ),
    )
    parser.add_argument("--concrete-fracture-energy-nmm", type=float, default=0.06)
    parser.add_argument("--concrete-reference-length-mm", type=float, default=100.0)
    parser.add_argument(
        "--concrete-crack-band-residual-tension-ratio",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--concrete-crack-band-residual-shear-ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--concrete-crack-band-large-opening-residual-shear-ratio",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--concrete-crack-band-shear-retention-decay-strain",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--concrete-crack-band-shear-transfer-law",
        default="opening-exponential",
        choices=(
            "constant-residual",
            "opening-exponential",
            "compression-gated-opening",
        ),
    )
    parser.add_argument(
        "--concrete-crack-band-closure-shear-gain",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--concrete-crack-band-open-compression-transfer-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--continuum-concrete-tangent-mode",
        default="fracture-secant",
        choices=(
            "fracture-secant",
            "legacy-forward-difference",
            "adaptive-central-difference",
            "adaptive-central-difference-with-secant-fallback",
        ),
    )
    parser.add_argument(
        "--continuum-characteristic-length-mode",
        default="fixed-end-longitudinal-host-edge-mm",
        choices=(
            "fixed-reference-mm",
            "mean-longitudinal-host-edge-mm",
            "fixed-end-longitudinal-host-edge-mm",
            "max-host-edge-mm",
        ),
    )
    parser.add_argument(
        "--continuum-rebar-interpolation",
        default="automatic",
        choices=("automatic", "two-node", "three-node"),
    )
    parser.add_argument(
        "--embedded-boundary-mode",
        default="dirichlet-rebar-endcap",
        choices=("dirichlet-rebar-endcap", "full-penalty-coupling"),
        help=(
            "Embedded-bar boundary policy. The structural-equivalence gate "
            "defaults to dirichlet-rebar-endcap because it preserves the "
            "imposed axial resultant under cyclic top displacement. "
            "full-penalty-coupling remains available as an experimental "
            "interior-coupling path, but it must satisfy the axial-reaction "
            "gate before being promoted."
        ),
    )
    parser.add_argument(
        "--axial-preload-transfer-mode",
        default="composite-section-force-split",
        choices=("host-surface-only", "composite-section-force-split"),
        help=(
            "How the axial preload is introduced in the continuum model. "
            "The structural-equivalence gate uses composite-section-force-"
            "split so the imposed axial compression is shared consistently "
            "between host and embedded bars. host-surface-only remains "
            "available only as an isolation/control route because it can "
            "create large artificial axial-reaction excursions under cyclic "
            "lateral displacement."
        ),
    )
    parser.add_argument(
        "--continuum-top-cap-mode",
        default="lateral-translation-only",
        choices=(
            "lateral-translation-only",
            "uniform-axial-penalty-cap",
            "affine-bending-rotation-penalty-cap",
        ),
        help=(
            "Top-face kinematic cap used by the continuum benchmark. The "
            "default prescribes only lateral translation. The axial penalty "
            "cap ties top-face axial DOFs to a central anchor and approximates "
            "a guided/no-rotation cap for boundary-condition sensitivity. "
            "The affine bending-rotation cap preserves the free mean axial "
            "settlement while tying the top face to a prescribed bending plane."
        ),
    )
    parser.add_argument(
        "--continuum-top-cap-bending-rotation-drift-ratio",
        type=float,
        default=0.0,
        help=(
            "Ratio used by the affine continuum top cap: theta_y = ratio * "
            "drift / column_height. It mirrors the structural drift-ratio "
            "surface without constraining the mean axial settlement."
        ),
    )
    parser.add_argument(
        "--continuum-top-cap-penalty-alpha-scale-over-ec",
        type=float,
        default=1.0e4,
        help="Dimensionless Ec multiplier for the optional top-cap DOF tie.",
    )
    parser.add_argument(
        "--penalty-alpha-scale-over-ec",
        type=float,
        default=1.0e4,
        help=(
            "Dimensionless penalty scale multiplying Ec for the embedded "
            "bar-host constraint."
        ),
    )
    parser.add_argument(
        "--solver-policy",
        default="newton-l2-only",
        choices=(
            "canonical-cascade",
            "newton-backtracking-only",
            "newton-l2-only",
            "newton-l2-lu-symbolic-reuse-only",
            "newton-l2-gmres-ilu1-only",
            "newton-trust-region-only",
            "newton-trust-region-dogleg-only",
            "quasi-newton-only",
            "nonlinear-gmres-only",
            "nonlinear-cg-only",
            "anderson-only",
            "nonlinear-richardson-only",
        ),
    )
    parser.add_argument(
        "--continuation",
        default="reversal-guarded",
        choices=("monolithic", "segmented", "reversal-guarded", "arc-length"),
    )
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--amplitudes-mm", default="5,10,15,20")
    parser.add_argument("--monotonic-tip-mm", type=float, default=20.0)
    parser.add_argument("--monotonic-steps", type=int, default=12)
    parser.add_argument("--column-height-m", type=float, default=3.2)
    parser.add_argument(
        "--steel-hinge-band-length-m",
        type=float,
        default=0.0,
        help=(
            "Length of the fixed-end steel band used for mesh-robust local "
            "comparison. Values <= 0 use one structural-element length, "
            "column_height/structural_element_count. The legacy single-GP "
            "comparison is still exported separately."
        ),
    )
    parser.add_argument(
        "--structural-top-rotation-mode",
        choices=("free", "clamped", "drift-ratio"),
        default="free",
        help=(
            "Tip bending-rotation condition used by the structural beam "
            "reference. The default is free because the current continuum "
            "benchmark prescribes only the lateral top-face translation; "
            "top-face axial displacements remain free, so the solid is not a "
            "guided/clamped-rotation cap unless an explicit cap constraint is "
            "introduced. drift-ratio prescribes theta_y = ratio * drift / L "
            "to match a measured or intended top-section kinematic contract."
        ),
    )
    parser.add_argument(
        "--structural-top-rotation-drift-ratio",
        type=float,
        default=1.0,
        help=(
            "Ratio used when --structural-top-rotation-mode=drift-ratio. "
            "Positive values follow the benchmark sign convention for theta_y."
        ),
    )
    parser.add_argument(
        "--structural-section-fiber-profile",
        choices=("coarse", "canonical", "fine", "ultra", "ultra-fine"),
        default="canonical",
        help=(
            "Concrete fiber refinement profile for the structural section. "
            "Steel area and bar coordinates remain invariant; this isolates "
            "section quadrature/discretization effects from continuum kinematics."
        ),
    )
    parser.add_argument("--structural-timeout-seconds", type=int, default=900)
    parser.add_argument("--continuum-timeout-seconds", type=int, default=1800)
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument(
        "--force-rerun",
        action="store_false",
        dest="reuse_existing",
        help="Ignore existing manifests and regenerate structural/continuum bundles.",
    )
    parser.add_argument("--print-progress", action="store_true")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "doc" / "figures" / "validation_reboot",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=repo_root / "PhD_Thesis" / "Figuras" / "validation_reboot",
    )
    parser.add_argument(
        "--structural-bundle-dir",
        type=Path,
        default=None,
        help="Reuse an existing structural bundle instead of rerunning the structural benchmark.",
    )
    parser.add_argument(
        "--continuum-bundle-dir",
        type=Path,
        default=None,
        help=(
            "Reuse one existing continuum bundle instead of rerunning the "
            "continuum benchmark. This is valid only when --hex-orders selects "
            "a single continuum family; comparison traces are still written "
            "under --output-dir."
        ),
    )
    parser.add_argument(
        "--continuum-bundle-root",
        type=Path,
        default=None,
        help=(
            "Reuse existing continuum bundles from <root>/<hex_order>. This is "
            "useful when comparing multiple already-generated continuum cases "
            "against a new structural reference."
        ),
    )
    parser.add_argument(
        "--skip-figure-export",
        action="store_true",
        help="Do not export the generic overlay figures into doc/ and thesis figure folders.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_floats(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def parse_csv_strings(raw: str) -> list[str]:
    return [token.strip().lower() for token in raw.split(",") if token.strip()]


def amplitude_suffix_from_args(args: argparse.Namespace) -> str:
    def scalar_label(value: float) -> str:
        return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")

    def compact_label(raw: str, aliases: dict[str, str]) -> str:
        normalized = str(raw).replace("_", "-").lower()
        return aliases.get(normalized, normalized.replace("-", "_"))

    if args.analysis == "cyclic":
        peak_mm = max(parse_csv_floats(args.amplitudes_mm))
    else:
        peak_mm = float(args.monotonic_tip_mm)
    label = f"{peak_mm:.3f}".rstrip("0").rstrip(".")
    suffix = label.replace(".", "p") + "mm"
    kinematics = str(
        getattr(args, "continuum_kinematics", "unspecified-kinematics")
    )
    kinematics_label = compact_label(
        kinematics,
        {
            "small-strain": "ss",
            "total-lagrangian": "tl",
            "updated-lagrangian": "ul",
            "corotational": "corot",
        },
    )
    material_mode = getattr(args, "continuum_material_mode", "nonlinear")
    if material_mode == "nonlinear":
        return f"{kinematics_label}_{suffix}"
    material_label = compact_label(
        material_mode,
        {
            "orthotropic-bimodular-proxy": "obp",
            "tensile-crack-band-damage-proxy": "tcbdp",
            "componentwise-kent-park-concrete": "cwkp",
            "cyclic-crack-band-concrete": "ccbc",
            "fixed-crack-band-concrete": "fcbc",
        },
    )
    if material_mode in (
        "orthotropic-bimodular-proxy",
        "tensile-crack-band-damage-proxy",
        "cyclic-crack-band-concrete",
        "fixed-crack-band-concrete",
    ):
        ratio_prefix = (
            "ts"
            if material_mode
            in ("cyclic-crack-band-concrete", "fixed-crack-band-concrete")
            else "et"
        )
        material_label = (
            f"{material_label}_{ratio_prefix}"
            f"{scalar_label(float(args.concrete_tension_stiffness_ratio))}"
        )
    if material_mode in ("cyclic-crack-band-concrete", "fixed-crack-band-concrete"):
        law_label = compact_label(
            str(args.concrete_crack_band_shear_transfer_law),
            {
                "constant-residual": "cres",
                "opening-exponential": "oexp",
                "compression-gated-opening": "cgop",
            },
        )
        material_label = (
            f"{material_label}_sr"
            f"{scalar_label(float(args.concrete_crack_band_residual_shear_ratio))}"
            f"_{law_label}"
            f"_oct"
            f"{scalar_label(float(args.concrete_crack_band_open_compression_transfer_ratio))}"
        )
    mesh_label = f"{int(args.nx)}x{int(args.ny)}x{int(args.nz)}"
    bias = float(getattr(args, "longitudinal_bias_power", 1.0))
    if abs(bias - 1.0) > 1.0e-12:
        bias_label = f"_bias{f'{bias:.3f}'.rstrip('0').rstrip('.').replace('.', 'p')}"
        bias_location_label = (
            "_" + str(getattr(args, "longitudinal_bias_location", "fixed-end"))
            .replace("-", "_")
        )
    else:
        bias_label = ""
        bias_location_label = ""
    rotation_mode = getattr(args, "structural_top_rotation_mode", "free")
    rotation_ratio_label = ""
    if rotation_mode == "drift-ratio":
        rotation_ratio = float(
            getattr(args, "structural_top_rotation_drift_ratio", 1.0)
        )
        rotation_ratio_label = (
            f"_r{f'{rotation_ratio:.3f}'.rstrip('0').rstrip('.').replace('.', 'p')}"
        )
    rotation_mode_label = compact_label(
        rotation_mode,
        {
            "free": "free",
            "zero": "zero",
            "drift-ratio": "drift",
        },
    )
    structural_label = (
        f"struct_{rotation_mode_label}{rotation_ratio_label}"
        f"_n{int(getattr(args, 'beam_nodes', 0))}"
        f"_e{int(getattr(args, 'structural_element_count', 1))}"
        f"_{getattr(args, 'beam_integration', 'unknown')}"
    )
    cap_label = str(
        getattr(args, "continuum_top_cap_mode", "lateral-translation-only")
    )
    cap_mode_label = compact_label(
        cap_label,
        {
            "lateral-translation-only": "lat",
            "uniform-axial-penalty-cap": "uax",
            "affine-bending-rotation-penalty-cap": "affrot",
        },
    )
    if cap_mode_label == "affrot":
        cap_label = (
            f"{cap_mode_label}_r"
            f"{scalar_label(float(args.continuum_top_cap_bending_rotation_drift_ratio))}"
        )
    else:
        cap_label = cap_mode_label
    return (
        f"{kinematics_label}_{material_label}_{mesh_label}{bias_label}"
        f"{bias_location_label}_"
        f"cap_{cap_label}_{structural_label}_{suffix}"
    )


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_existing_bundle(bundle_dir: Path, manifest_name: str) -> dict[str, object]:
    manifest_path = bundle_dir / manifest_name
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest in reused bundle: {manifest_path}")
    return read_json(manifest_path)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def timing_value(manifest: dict[str, Any], *field_names: str) -> float:
    timing = manifest.get("timing")
    if not isinstance(timing, dict):
        return math.nan

    for name in field_names:
        value = timing.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return math.nan


def total_wall_seconds(manifest: dict[str, Any]) -> float:
    return timing_value(manifest, "total_wall_seconds")


def solve_wall_seconds(manifest: dict[str, Any]) -> float:
    return timing_value(
        manifest,
        "solve_wall_seconds",
        "analysis_wall_seconds",
        "total_wall_seconds",
    )


def structural_section_fiber_profile(
    manifest: dict[str, Any],
    fallback: str,
) -> str:
    mesh = manifest.get("section_fiber_mesh")
    if isinstance(mesh, dict):
        profile = mesh.get("profile")
        if isinstance(profile, str) and profile:
            return profile
    return fallback


def clean_optional_number(value: float) -> float | None:
    return None if not math.isfinite(value) else value


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        proc.kill()


def run_command(
    command: list[str], cwd: Path, timeout_seconds: int
) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=creationflags,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        terminate_process_tree(proc)
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            command,
            timeout_seconds,
            output=(exc.output or "") + (stdout or ""),
            stderr=(exc.stderr or "") + (stderr or ""),
        ) from exc
    completed = subprocess.CompletedProcess(
        command,
        proc.returncode,
        stdout,
        stderr,
    )
    return time.perf_counter() - start, completed


def structural_command(
    exe: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    top_rotation_args: list[str]
    if args.structural_top_rotation_mode == "clamped":
        top_rotation_args = ["--clamp-top-bending-rotation"]
    elif args.structural_top_rotation_mode == "drift-ratio":
        top_rotation_args = [
            "--top-bending-rotation-drift-ratio",
            f"{args.structural_top_rotation_drift_ratio}",
        ]
    else:
        top_rotation_args = ["--no-op"]

    return [
        str(exe),
        "--output-dir",
        str(out_dir),
        "--analysis",
        args.analysis,
        "--material-mode",
        "nonlinear",
        "--beam-nodes",
        str(args.beam_nodes),
        "--structural-element-count",
        str(max(args.structural_element_count, 1)),
        "--beam-integration",
        args.beam_integration,
        *top_rotation_args,
        "--solver-policy",
        args.solver_policy,
        "--continuation",
        args.continuation,
        "--continuation-segment-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{args.monotonic_tip_mm}",
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
        "--section-fiber-profile",
        args.structural_section_fiber_profile,
        "--print-progress" if args.print_progress else "--no-op",
    ]


def continuum_command(
    exe: Path,
    out_dir: Path,
    hex_order: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        str(exe),
        "--output-dir",
        str(out_dir),
        "--analysis",
        args.analysis,
        "--material-mode",
        args.continuum_material_mode,
        "--continuum-kinematics",
        args.continuum_kinematics,
        "--concrete-tension-stiffness-ratio",
        f"{args.concrete_tension_stiffness_ratio}",
        "--concrete-fracture-energy-nmm",
        f"{args.concrete_fracture_energy_nmm}",
        "--concrete-reference-length-mm",
        f"{args.concrete_reference_length_mm}",
        "--concrete-crack-band-residual-tension-ratio",
        f"{args.concrete_crack_band_residual_tension_ratio}",
        "--concrete-crack-band-residual-shear-ratio",
        f"{args.concrete_crack_band_residual_shear_ratio}",
        "--concrete-crack-band-large-opening-residual-shear-ratio",
        f"{args.concrete_crack_band_large_opening_residual_shear_ratio}",
        "--concrete-crack-band-shear-retention-decay-strain",
        f"{args.concrete_crack_band_shear_retention_decay_strain}",
        "--concrete-crack-band-shear-transfer-law",
        args.concrete_crack_band_shear_transfer_law,
        "--concrete-crack-band-closure-shear-gain",
        f"{args.concrete_crack_band_closure_shear_gain}",
        "--concrete-crack-band-open-compression-transfer-ratio",
        f"{args.concrete_crack_band_open_compression_transfer_ratio}",
        "--concrete-profile",
        args.continuum_concrete_profile,
        "--concrete-tangent-mode",
        args.continuum_concrete_tangent_mode,
        "--concrete-characteristic-length-mode",
        args.continuum_characteristic_length_mode,
        "--reinforcement-mode",
        "embedded-longitudinal-bars",
        "--rebar-interpolation",
        args.continuum_rebar_interpolation,
        "--rebar-layout",
        args.continuum_rebar_layout,
        "--host-concrete-zoning-mode",
        args.host_concrete_zoning_mode,
        "--transverse-mesh-mode",
        args.transverse_mesh_mode,
        "--hex-order",
        hex_order,
        "--nx",
        str(args.nx),
        "--ny",
        str(args.ny),
        "--nz",
        str(args.nz),
        "--longitudinal-bias-power",
        f"{args.longitudinal_bias_power}",
        "--longitudinal-bias-location",
        args.longitudinal_bias_location,
        "--embedded-boundary-mode",
        args.embedded_boundary_mode,
        "--axial-preload-transfer-mode",
        args.axial_preload_transfer_mode,
        "--top-cap-mode",
        args.continuum_top_cap_mode,
        "--top-cap-penalty-alpha-scale-over-ec",
        f"{args.continuum_top_cap_penalty_alpha_scale_over_ec}",
        "--top-cap-bending-rotation-drift-ratio",
        f"{args.continuum_top_cap_bending_rotation_drift_ratio}",
        "--penalty-alpha-scale-over-ec",
        f"{args.penalty_alpha_scale_over_ec}",
        "--solver-policy",
        args.solver_policy,
        "--continuation",
        args.continuation,
        "--continuation-segment-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--axial-compression-mn",
        f"{args.axial_compression_mn}",
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--monotonic-tip-mm",
        f"{args.monotonic_tip_mm}",
        "--monotonic-steps",
        str(args.monotonic_steps),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def run_case(
    command: list[str],
    output_dir: Path,
    manifest_name: str,
    reuse_existing: bool,
    timeout_seconds: int,
) -> tuple[float, dict[str, object]]:
    ensure_dir(output_dir)
    manifest_path = output_dir / manifest_name
    if reuse_existing and manifest_path.exists():
        return math.nan, read_json(manifest_path)

    try:
        elapsed, proc = run_command(command, output_dir.parent, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        (output_dir / "runner_stdout.log").write_text(
            exc.stdout or "",
            encoding="utf-8",
        )
        (output_dir / "runner_stderr.log").write_text(
            exc.stderr or "",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Command timed out after {timeout_seconds} s:\n{' '.join(command)}"
        ) from exc

    (output_dir / "runner_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "runner_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit {proc.returncode}:\n{' '.join(command)}\n\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if not manifest_path.exists():
        raise RuntimeError(f"Missing manifest after successful run: {manifest_path}")
    return elapsed, read_json(manifest_path)


def select_structural_steel_identity(
    rows: list[dict[str, object]],
) -> StructuralSteelIdentity:
    steel_rows = [
        row
        for row in rows
        if str(row["material_role"]) == "reinforcing_steel"
    ]
    if not steel_rows:
        raise RuntimeError("Structural benchmark did not export reinforcing steel fibers.")
    base_gp = min(int(row["section_gp"]) for row in steel_rows)
    base_rows = [row for row in steel_rows if int(row["section_gp"]) == base_gp]
    seed = min(float(row["step"]) for row in base_rows)
    seed_rows = [row for row in base_rows if float(row["step"]) == seed]
    chosen = max(seed_rows, key=lambda row: (float(row["z"]), float(row["y"])))
    return StructuralSteelIdentity(
        fiber_index=int(chosen["fiber_index"]),
        y=float(chosen["y"]),
        z=float(chosen["z"]),
        area=float(chosen["area"]),
        zone=str(chosen["zone"]),
        material_role=str(chosen["material_role"]),
    )


def infer_structural_active_coordinate(
    section_rows: list[dict[str, object]],
) -> tuple[str, str]:
    if not section_rows:
        return "z", "y"

    ky = [
        abs(float(row["curvature_y"]))
        for row in section_rows
        if "curvature_y" in row
    ]
    kz = [
        abs(float(row["curvature_z"]))
        for row in section_rows
        if "curvature_z" in row
    ]
    ky_ref = sum(ky) / len(ky) if ky else 0.0
    kz_ref = sum(kz) / len(kz) if kz else 0.0
    return ("z", "y") if ky_ref >= kz_ref else ("y", "z")


def select_structural_steel_trace(
    rows: list[dict[str, object]],
    identity: StructuralSteelIdentity,
    target_position_z_m: float,
    column_height_m: float,
) -> tuple[list[dict[str, object]], StructuralSteelTraceSelection]:
    matching_bar_rows = [
        row
        for row in rows
        if int(row["fiber_index"]) == identity.fiber_index
        and abs(float(row["y"]) - identity.y) < 1.0e-12
        and abs(float(row["z"]) - identity.z) < 1.0e-12
    ]
    if not matching_bar_rows:
        raise RuntimeError("Failed to recover the structural steel bar trace.")

    def axial_position(row: dict[str, object]) -> float:
        xi = float(row["xi"])
        return 0.5 * column_height_m * (xi + 1.0)

    def bracket(stations: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, object], float]:
        ordered = sorted(stations, key=axial_position)
        if len(ordered) == 1:
            return ordered[0], ordered[0], 0.0

        first = ordered[0]
        last = ordered[-1]
        if target_position_z_m <= axial_position(first):
            return first, first, 0.0
        if target_position_z_m >= axial_position(last):
            return last, last, 0.0

        for left, right in zip(ordered[:-1], ordered[1:]):
            z0 = axial_position(left)
            z1 = axial_position(right)
            if z0 <= target_position_z_m <= z1:
                if abs(z1 - z0) <= 1.0e-14:
                    return left, right, 0.0
                alpha = (target_position_z_m - z0) / (z1 - z0)
                return left, right, alpha
        return last, last, 0.0

    grouped_by_step: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in matching_bar_rows:
        grouped_by_step.setdefault((float(row["step"]), float(row["p"])), []).append(row)

    interpolated_trace: list[dict[str, object]] = []
    selection: StructuralSteelTraceSelection | None = None
    for _, stations in sorted(grouped_by_step.items(), key=lambda item: item[0]):
        lower, upper, alpha = bracket(stations)

        def lerp(field: str) -> float:
            lhs = float(lower[field])
            rhs = float(upper[field])
            return lhs + alpha * (rhs - lhs)

        interpolated_trace.append(
            {
                "step": float(lower["step"]),
                "p": float(lower["p"]),
                "drift_m": float(lower["drift_m"]),
                "position_z_m": axial_position(lower) + alpha * (axial_position(upper) - axial_position(lower)),
                "strain_xx": lerp("strain_xx"),
                "stress_xx_MPa": lerp("stress_xx_MPa"),
                "tangent_xx_MPa": lerp("tangent_xx_MPa"),
                "fiber_index": identity.fiber_index,
                "y": identity.y,
                "z": identity.z,
            }
        )

        if selection is None:
            selection = StructuralSteelTraceSelection(
                target_position_z_m=target_position_z_m,
                lower_section_gp=int(lower["section_gp"]),
                upper_section_gp=int(upper["section_gp"]),
                lower_position_z_m=axial_position(lower),
                upper_position_z_m=axial_position(upper),
                interpolation_alpha=alpha,
                fiber_index=identity.fiber_index,
                y=identity.y,
                z=identity.z,
            )

    if selection is None:
        raise RuntimeError("Failed to interpolate the structural steel trace.")

    return interpolated_trace, selection


def elastic_window_rows(
    rows: list[dict[str, object]], x_field: str, max_abs_x: float
) -> list[dict[str, object]]:
    tol = 1.0e-12
    return [
        row
        for row in rows
        if abs(float(row[x_field])) <= max_abs_x + tol
    ]


def first_orientation_window_drift(
    rows: list[dict[str, object]], x_field: str, floor_abs_x: float = 5.0e-3
) -> float:
    """Return a drift window that contains at least one real loading increment.

    The orientation audit must not infer the active steel side from the preload
    state alone. Large-amplitude campaigns may have a first nonzero drift larger
    than the small default elastic window; in that case all symmetric bars look
    identical and the comparison can silently pair opposite physical branches.
    """

    nonzero = sorted(
        {
            abs(float(row[x_field]))
            for row in rows
            if abs(float(row[x_field])) > 1.0e-12
        }
    )
    if not nonzero:
        return floor_abs_x
    return max(floor_abs_x, nonzero[0])


def rms_relative_error(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
    x_field: str,
    lhs_y_field: str,
    rhs_y_field: str,
) -> float:
    _, mean_rel = branchwise_interpolation_error(
        lhs_rows,
        rhs_rows,
        x_field,
        lhs_y_field,
        rhs_y_field,
    )
    return mean_rel


def select_continuum_steel_identity(
    rows: list[dict[str, object]],
    target_y: float,
    target_z: float,
) -> ContinuumSteelIdentity:
    if not rows:
        raise RuntimeError("Continuum benchmark did not export rebar_history.csv rows.")
    unique_bars: dict[int, tuple[float, float]] = {}
    for row in rows:
        bar_index = int(row["bar_index"])
        unique_bars.setdefault(
            bar_index, (float(row["bar_y"]), float(row["bar_z"]))
        )
    chosen_bar_index, (bar_y, bar_z) = min(
        unique_bars.items(),
        key=lambda item: abs(item[1][0] - target_y) + abs(item[1][1] - target_z),
    )
    base_candidates = [
        row
        for row in rows
        if int(row["bar_index"]) == chosen_bar_index
        and int(row["bar_element_layer"]) == 0
    ]
    if not base_candidates:
        raise RuntimeError("Continuum benchmark has no base-layer rebar candidates.")
    chosen_gp_row = min(base_candidates, key=lambda row: float(row["position_z_m"]))
    return ContinuumSteelIdentity(
        bar_index=chosen_bar_index,
        bar_element_layer=int(chosen_gp_row["bar_element_layer"]),
        gp_index=int(chosen_gp_row["gp_index"]),
        bar_y=bar_y,
        bar_z=bar_z,
        position_z_m=float(chosen_gp_row["position_z_m"]),
    )


def select_continuum_steel_trace(
    rows: list[dict[str, object]], identity: ContinuumSteelIdentity
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if int(row["bar_index"]) == identity.bar_index
        and int(row["bar_element_layer"]) == identity.bar_element_layer
        and int(row["gp_index"]) == identity.gp_index
    ]
    return sorted(
        selected, key=lambda row: (float(row["runtime_p"]), float(row["runtime_step"]))
    )


def hinge_band_descriptor(args: argparse.Namespace) -> HingeBandDescriptor:
    if args.steel_hinge_band_length_m > 0.0:
        return HingeBandDescriptor(
            length_m=args.steel_hinge_band_length_m,
            selection_rule="user-specified",
            note=(
                "Fixed-end steel histories are averaged over the user-declared "
                "physical band instead of a single Gauss point."
            ),
        )

    structural_element_count = max(int(args.structural_element_count), 1)
    return HingeBandDescriptor(
        length_m=float(args.column_height_m) / float(structural_element_count),
        selection_rule="one-structural-element-length",
        note=(
            "Fixed-end steel histories are averaged over the first structural "
            "element length. This avoids treating a mesh-dependent continuum "
            "Gauss point as the whole plastic-hinge story."
        ),
    )


def axial_voronoi_weights(
    all_positions: list[float],
    included_positions: list[float],
    band_length_m: float,
) -> dict[float, float]:
    """Return normalized one-dimensional support weights for axial stations."""

    unique_all = sorted({round(value, 12) for value in all_positions})
    unique_included = sorted({round(value, 12) for value in included_positions})
    if not unique_included:
        return {}

    raw_weights: dict[float, float] = {}
    for position in unique_included:
        index = unique_all.index(position)
        left = 0.0 if index == 0 else 0.5 * (unique_all[index - 1] + position)
        right = (
            band_length_m
            if index + 1 >= len(unique_all)
            else 0.5 * (position + unique_all[index + 1])
        )
        left = max(0.0, min(left, band_length_m))
        right = max(0.0, min(right, band_length_m))
        raw_weights[position] = max(right - left, 0.0)

    total = sum(raw_weights.values())
    if total <= 1.0e-14:
        equal = 1.0 / float(len(unique_included))
        return {position: equal for position in unique_included}
    return {position: weight / total for position, weight in raw_weights.items()}


def aggregate_position_group(
    rows: list[dict[str, object]],
    position_field: str,
    value_fields: list[str],
    band_length_m: float,
) -> tuple[dict[str, float], dict[str, float]]:
    positions = [float(row[position_field]) for row in rows]
    included_positions = [
        position for position in positions if position <= band_length_m + 1.0e-12
    ]
    if not included_positions and positions:
        included_positions = [min(positions)]

    weights = axial_voronoi_weights(positions, included_positions, band_length_m)
    by_position: dict[float, list[dict[str, object]]] = {}
    for row in rows:
        position = round(float(row[position_field]), 12)
        if position in weights:
            by_position.setdefault(position, []).append(row)

    averaged: dict[str, float] = {}
    for field in value_fields:
        total = 0.0
        for position, station_rows in by_position.items():
            station_value = sum(float(row[field]) for row in station_rows) / float(
                len(station_rows)
            )
            total += weights[position] * station_value
        averaged[field] = total

    metadata = {
        "included_station_count": float(sum(len(v) for v in by_position.values())),
        "unique_position_count": float(len(by_position)),
        "min_position_z_m": min(by_position, default=math.nan),
        "max_position_z_m": max(by_position, default=math.nan),
        "band_length_m": band_length_m,
    }
    return averaged, metadata


def continuum_hinge_band_trace(
    rows: list[dict[str, object]],
    identity: ContinuumSteelIdentity,
    band_length_m: float,
) -> list[dict[str, object]]:
    bar_rows = [
        row for row in rows if int(row["bar_index"]) == identity.bar_index
    ]
    grouped: dict[tuple[float, float, float, float], list[dict[str, object]]] = {}
    for row in bar_rows:
        grouped.setdefault(
            (
                float(row["runtime_step"]),
                float(row["step"]),
                float(row["p"]),
                float(row["runtime_p"]),
            ),
            [],
        ).append(row)

    trace: list[dict[str, object]] = []
    for (runtime_step, step, p_value, runtime_p), group_rows in sorted(
        grouped.items(), key=lambda item: (item[0][3], item[0][0])
    ):
        averaged, metadata = aggregate_position_group(
            group_rows,
            "position_z_m",
            [
                "axial_strain",
                "stress_xx_MPa",
                "projected_axial_strain_gap",
                "projected_axial_gap_m",
                "projected_gap_norm_m",
            ],
            band_length_m,
        )
        trace.append(
            {
                "runtime_step": runtime_step,
                "step": step,
                "p": p_value,
                "runtime_p": runtime_p,
                "drift_m": float(group_rows[0]["drift_m"]),
                "bar_index": identity.bar_index,
                "bar_y": identity.bar_y,
                "bar_z": identity.bar_z,
                "band_length_m": metadata["band_length_m"],
                "included_station_count": metadata["included_station_count"],
                "unique_position_count": metadata["unique_position_count"],
                "min_position_z_m": metadata["min_position_z_m"],
                "max_position_z_m": metadata["max_position_z_m"],
                "axial_strain": averaged["axial_strain"],
                "stress_xx_MPa": averaged["stress_xx_MPa"],
                "projected_axial_strain_gap": averaged["projected_axial_strain_gap"],
                "projected_axial_gap_m": averaged["projected_axial_gap_m"],
                "projected_gap_norm_m": averaged["projected_gap_norm_m"],
            }
        )
    return trace


def structural_hinge_band_trace(
    rows: list[dict[str, object]],
    identity: StructuralSteelIdentity,
    column_height_m: float,
    band_length_m: float,
) -> list[dict[str, object]]:
    matching_bar_rows = [
        row
        for row in rows
        if int(row["fiber_index"]) == identity.fiber_index
        and abs(float(row["y"]) - identity.y) < 1.0e-12
        and abs(float(row["z"]) - identity.z) < 1.0e-12
    ]

    def with_position(row: dict[str, object]) -> dict[str, object]:
        out = dict(row)
        out["position_z_m"] = 0.5 * column_height_m * (float(row["xi"]) + 1.0)
        return out

    grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in matching_bar_rows:
        grouped.setdefault((float(row["step"]), float(row["p"])), []).append(
            with_position(row)
        )

    trace: list[dict[str, object]] = []
    for (step, p_value), group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        averaged, metadata = aggregate_position_group(
            group_rows,
            "position_z_m",
            ["strain_xx", "stress_xx_MPa", "tangent_xx_MPa"],
            band_length_m,
        )
        trace.append(
            {
                "step": step,
                "p": p_value,
                "drift_m": float(group_rows[0]["drift_m"]),
                "fiber_index": identity.fiber_index,
                "y": identity.y,
                "z": identity.z,
                "band_length_m": metadata["band_length_m"],
                "included_station_count": metadata["included_station_count"],
                "unique_position_count": metadata["unique_position_count"],
                "min_position_z_m": metadata["min_position_z_m"],
                "max_position_z_m": metadata["max_position_z_m"],
                "strain_xx": averaged["strain_xx"],
                "stress_xx_MPa": averaged["stress_xx_MPa"],
                "tangent_xx_MPa": averaged["tangent_xx_MPa"],
            }
        )
    return trace


def select_structural_orientation_matched_trace(
    structural_rows: list[dict[str, object]],
    section_rows: list[dict[str, object]],
    continuum_trace: list[dict[str, object]],
    continuum_identity: ContinuumSteelIdentity,
    column_height_m: float,
) -> tuple[
    StructuralSteelIdentity,
    list[dict[str, object]],
    StructuralSteelTraceSelection,
    list[StructuralOrientationCandidate],
]:
    steel_rows = [
        row for row in structural_rows if str(row["material_role"]) == "reinforcing_steel"
    ]
    if not steel_rows:
        raise RuntimeError("Structural benchmark did not export reinforcing steel fibers.")

    active_name, passive_name = infer_structural_active_coordinate(section_rows)
    target_structural_y, target_structural_z = continuum_to_structural_coordinates(
        continuum_identity.bar_y,
        continuum_identity.bar_z,
        active_name,
    )
    target_structural_active = (
        target_structural_z if active_name == "z" else target_structural_y
    )
    target_structural_passive = (
        target_structural_y if passive_name == "y" else target_structural_z
    )
    tol = 1.0e-12

    candidate_keys: dict[tuple[int, float, float, float], StructuralSteelIdentity] = {}
    for row in steel_rows:
        y = float(row["y"])
        z = float(row["z"])
        active = z if active_name == "z" else y
        passive = y if passive_name == "y" else z
        if abs(active - target_structural_active) > tol:
            continue
        if abs(passive - target_structural_passive) > tol:
            continue

        identity = StructuralSteelIdentity(
            fiber_index=int(row["fiber_index"]),
            y=y,
            z=z,
            area=float(row["area"]),
            zone=str(row["zone"]),
            material_role=str(row["material_role"]),
        )
        key = (identity.fiber_index, identity.y, identity.z, identity.area)
        candidate_keys[key] = identity

    if not candidate_keys:
        raise RuntimeError(
            "Failed to find structural steel fibers compatible with the matched "
            "continuum bar under the active bending coordinate."
        )

    max_elastic_drift = first_orientation_window_drift(
        continuum_trace,
        "drift_m",
    )
    continuum_elastic = elastic_window_rows(continuum_trace, "drift_m", max_elastic_drift)
    if not continuum_elastic:
        continuum_elastic = continuum_trace

    candidates: list[StructuralOrientationCandidate] = []
    best_identity: StructuralSteelIdentity | None = None
    best_trace: list[dict[str, object]] | None = None
    best_selection: StructuralSteelTraceSelection | None = None
    best_score = (math.inf, math.inf, math.inf, math.inf)

    for identity in candidate_keys.values():
        trace, selection = select_structural_steel_trace(
            structural_rows,
            identity,
            continuum_identity.position_z_m,
            column_height_m,
        )
        structural_elastic = elastic_window_rows(trace, "drift_m", max_elastic_drift)
        if not structural_elastic:
            structural_elastic = trace

        stress_err = rms_relative_error(
            continuum_elastic,
            structural_elastic,
            "drift_m",
            "stress_xx_MPa",
            "stress_xx_MPa",
        )
        strain_err = rms_relative_error(
            continuum_elastic,
            structural_elastic,
            "drift_m",
            "axial_strain",
            "strain_xx",
        )
        active = identity.z if active_name == "z" else identity.y
        passive = identity.y if passive_name == "y" else identity.z
        candidates.append(
            StructuralOrientationCandidate(
                fiber_index=identity.fiber_index,
                y=identity.y,
                z=identity.z,
                active_coordinate_name=active_name,
                active_coordinate_value=active,
                passive_coordinate_name=passive_name,
                passive_coordinate_value=passive,
                elastic_rms_stress_error=stress_err,
                elastic_rms_strain_error=strain_err,
            )
        )
        score = (
            (stress_err if math.isfinite(stress_err) else 1.0e12),
            (strain_err if math.isfinite(strain_err) else 1.0e12),
            abs(active - target_structural_active),
            abs(passive - target_structural_passive),
        )
        if score < best_score:
            best_score = score
            best_identity = identity
            best_trace = trace
            best_selection = selection

    if best_identity is None or best_trace is None or best_selection is None:
        raise RuntimeError("Failed to select the orientation-matched structural steel trace.")

    candidates.sort(
        key=lambda row: (
            row.elastic_rms_stress_error if math.isfinite(row.elastic_rms_stress_error) else 1.0e12,
            row.elastic_rms_strain_error if math.isfinite(row.elastic_rms_strain_error) else 1.0e12,
            row.fiber_index,
        )
    )
    return best_identity, best_trace, best_selection, candidates


def branchwise_interpolation_error(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
    x_field: str,
    lhs_y_field: str,
    rhs_y_field: str,
) -> tuple[float, float]:
    def monotone_branches(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
        if len(rows) <= 1:
            return [rows] if rows else []
        tol = 1.0e-12
        branches: list[list[dict[str, object]]] = []
        current = [rows[0]]
        current_direction = 0
        previous_x = float(rows[0][x_field])
        for row in rows[1:]:
            x = float(row[x_field])
            delta = x - previous_x
            direction = 0 if abs(delta) <= tol else (1 if delta > 0.0 else -1)
            if current_direction == 0 and direction != 0:
                current_direction = direction
            elif direction != 0 and current_direction != 0 and direction != current_direction:
                branches.append(current)
                current = [current[-1], row]
                current_direction = direction
            else:
                current.append(row)
            previous_x = x
        if current:
            branches.append(current)
        return branches

    def interpolate(rows: list[dict[str, object]], target_x: float) -> float:
        if not rows:
            return math.nan
        xs = [float(row[x_field]) for row in rows]
        ys = [float(row[rhs_y_field]) for row in rows]
        if len(rows) == 1:
            return ys[0]
        lo = min(xs)
        hi = max(xs)
        tol = 1.0e-12
        if target_x < lo - tol or target_x > hi + tol:
            return math.nan
        for left, right in zip(range(len(rows) - 1), range(1, len(rows))):
            x0 = xs[left]
            x1 = xs[right]
            y0 = ys[left]
            y1 = ys[right]
            if abs(target_x - x0) <= tol:
                return y0
            if abs(target_x - x1) <= tol:
                return y1
            if (x0 <= target_x <= x1) or (x1 <= target_x <= x0):
                if abs(x1 - x0) <= tol:
                    return 0.5 * (y0 + y1)
                alpha = (target_x - x0) / (x1 - x0)
                return y0 + alpha * (y1 - y0)
        return math.nan

    lhs_branches = monotone_branches(lhs_rows)
    rhs_branches = monotone_branches(rhs_rows)
    rel_errors: list[float] = []
    for lhs_branch, rhs_branch in zip(lhs_branches, rhs_branches):
        for row in lhs_branch:
            lhs = float(row[lhs_y_field])
            rhs = interpolate(rhs_branch, float(row[x_field]))
            if not math.isfinite(rhs):
                continue
            rel_errors.append(abs(lhs - rhs) / max(abs(rhs), 1.0e-12))
    if not rel_errors:
        return math.nan, math.nan
    return max(rel_errors), sum(rel_errors) / len(rel_errors)


def branchwise_peak_normalized_error(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
    x_field: str,
    lhs_y_field: str,
    rhs_y_field: str,
) -> tuple[float, float]:
    """Compare histories with errors normalized by the reference peak.

    Pointwise relative errors are useful for sign mistakes, but they become
    misleading on cyclic curves whenever the reference crosses zero. This
    metric keeps the same branchwise interpolation while scaling all absolute
    differences by max(|reference|), which is more meaningful for hysteretic
    amplitude and energy comparisons.
    """

    def monotone_branches(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
        if len(rows) <= 1:
            return [rows] if rows else []
        tol = 1.0e-12
        branches: list[list[dict[str, object]]] = []
        current = [rows[0]]
        current_direction = 0
        previous_x = float(rows[0][x_field])
        for row in rows[1:]:
            x = float(row[x_field])
            delta = x - previous_x
            direction = 0 if abs(delta) <= tol else (1 if delta > 0.0 else -1)
            if current_direction == 0 and direction != 0:
                current_direction = direction
            elif direction != 0 and current_direction != 0 and direction != current_direction:
                branches.append(current)
                current = [current[-1], row]
                current_direction = direction
            else:
                current.append(row)
            previous_x = x
        if current:
            branches.append(current)
        return branches

    def interpolate(rows: list[dict[str, object]], target_x: float) -> float:
        if not rows:
            return math.nan
        xs = [float(row[x_field]) for row in rows]
        ys = [float(row[rhs_y_field]) for row in rows]
        if len(rows) == 1:
            return ys[0]
        lo = min(xs)
        hi = max(xs)
        tol = 1.0e-12
        if target_x < lo - tol or target_x > hi + tol:
            return math.nan
        for left, right in zip(range(len(rows) - 1), range(1, len(rows))):
            x0 = xs[left]
            x1 = xs[right]
            y0 = ys[left]
            y1 = ys[right]
            if abs(target_x - x0) <= tol:
                return y0
            if abs(target_x - x1) <= tol:
                return y1
            if (x0 <= target_x <= x1) or (x1 <= target_x <= x0):
                if abs(x1 - x0) <= tol:
                    return 0.5 * (y0 + y1)
                alpha = (target_x - x0) / (x1 - x0)
                return y0 + alpha * (y1 - y0)
        return math.nan

    reference_peak = max(
        (abs(float(row[rhs_y_field])) for row in rhs_rows),
        default=math.nan,
    )
    if not math.isfinite(reference_peak) or reference_peak <= 1.0e-12:
        return math.nan, math.nan

    lhs_branches = monotone_branches(lhs_rows)
    rhs_branches = monotone_branches(rhs_rows)
    errors: list[float] = []
    for lhs_branch, rhs_branch in zip(lhs_branches, rhs_branches):
        for row in lhs_branch:
            lhs = float(row[lhs_y_field])
            rhs = interpolate(rhs_branch, float(row[x_field]))
            if not math.isfinite(rhs):
                continue
            errors.append(abs(lhs - rhs) / reference_peak)
    if not errors:
        return math.nan, math.nan
    return max(errors), math.sqrt(sum(value * value for value in errors) / len(errors))


def cyclic_loop_work(rows: list[dict[str, object]], x_field: str, y_field: str) -> float:
    if len(rows) < 2:
        return math.nan
    total = 0.0
    for prev, curr in zip(rows[:-1], rows[1:]):
        dx = float(curr[x_field]) - float(prev[x_field])
        y_avg = 0.5 * (float(curr[y_field]) + float(prev[y_field]))
        total += y_avg * dx
    return total


def save(fig: plt.Figure, out_dirs: list[Path], stem: str) -> None:
    for out_dir in out_dirs:
        ensure_dir(out_dir)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def make_overlay_figures(
    structural_hist: list[dict[str, object]],
    structural_steel_traces: dict[str, list[dict[str, object]]],
    continuum_histories: dict[str, list[dict[str, object]]],
    continuum_steel_histories: dict[str, list[dict[str, object]]],
    out_dirs: list[Path],
    suffix: str,
) -> None:
    palette = {"hex8": ORANGE, "hex20": ORANGE, "hex27": GREEN}
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    ax.plot(
        [1.0e3 * float(row["drift_m"]) for row in structural_hist],
        [1.0e3 * float(row["base_shear_MN"]) for row in structural_hist],
        color=BLUE,
        linewidth=1.6,
        label="Structural Timoshenko",
    )
    for hex_order, rows in continuum_histories.items():
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [1.0e3 * float(row["base_shear_MN"]) for row in rows],
            color=palette.get(hex_order, "#444444"),
            linewidth=1.3,
            linestyle="--",
            label=f"Continuum {hex_order.upper()}",
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Reduced RC structural vs continuum hysteresis")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dirs, f"reduced_rc_structural_continuum_hysteresis_overlay_{suffix}")

    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for hex_order, structural_rows in structural_steel_traces.items():
        color = palette.get(hex_order, BLUE)
        ax.plot(
            [float(row["strain_xx"]) for row in structural_rows],
            [float(row["stress_xx_MPa"]) for row in structural_rows],
            color=color,
            linewidth=1.6,
            label=f"Structural steel fiber ({hex_order.upper()} orientation-matched)",
        )
        rows = continuum_steel_histories[hex_order]
        ax.plot(
            [float(row["axial_strain"]) for row in rows],
            [float(row["stress_xx_MPa"]) for row in rows],
            color=color,
            linewidth=1.3,
            linestyle="--",
            label=f"Continuum embedded steel {hex_order.upper()}",
        )
    ax.set_xlabel(r"Steel strain $\varepsilon$")
    ax.set_ylabel(r"Steel stress $\sigma$ [MPa]")
    ax.set_title(
        "Steel hysteresis: orientation-matched structural fiber vs embedded truss"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dirs, f"reduced_rc_structural_continuum_steel_hysteresis_overlay_{suffix}")

    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for hex_order, structural_rows in structural_steel_traces.items():
        color = palette.get(hex_order, BLUE)
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in structural_rows],
            [float(row["stress_xx_MPa"]) for row in structural_rows],
            color=color,
            linewidth=1.6,
            label=f"Structural steel fiber ({hex_order.upper()} orientation-matched)",
        )
        rows = continuum_steel_histories[hex_order]
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [float(row["stress_xx_MPa"]) for row in rows],
            color=color,
            linewidth=1.3,
            linestyle="--",
            label=f"Continuum embedded steel {hex_order.upper()}",
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel(r"Steel stress $\sigma$ [MPa]")
    ax.set_title("Steel stress response along the cyclic protocol")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dirs, f"reduced_rc_structural_continuum_steel_stress_vs_drift_{suffix}")
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for hex_order, structural_rows in structural_steel_traces.items():
        color = palette.get(hex_order, BLUE)
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in structural_rows],
            [float(row["strain_xx"]) for row in structural_rows],
            color=color,
            linewidth=1.6,
            label=f"Structural steel fiber ({hex_order.upper()} orientation-matched)",
        )
        rows = continuum_steel_histories[hex_order]
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [float(row["axial_strain"]) for row in rows],
            color=color,
            linewidth=1.3,
            linestyle="--",
            label=f"Continuum embedded steel {hex_order.upper()}",
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel(r"Steel strain $\varepsilon$")
    ax.set_title("Steel strain response along the cyclic protocol")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dirs, f"reduced_rc_structural_continuum_steel_strain_vs_drift_{suffix}")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for hex_order, structural_rows in structural_steel_traces.items():
        color = palette.get(hex_order, BLUE)
        rows = continuum_steel_histories[hex_order]
        # The host-projected strain is the key local audit: if it tracks the
        # embedded bar almost exactly, then the open structural-vs-continuum
        # gap is no longer credibly explained by the host-bar transfer layer.
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in structural_rows],
            [float(row["strain_xx"]) for row in structural_rows],
            color=color,
            linewidth=1.5,
            label=f"Structural fiber ({hex_order.upper()})",
        )
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [float(row["axial_strain"]) for row in rows],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=f"Continuum bar ({hex_order.upper()})",
        )
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [float(row["host_projected_axial_strain"]) for row in rows],
            color=color,
            linewidth=1.1,
            linestyle=":",
            label=f"Continuum projected host ({hex_order.upper()})",
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel(r"Axial strain $\varepsilon$")
    ax.set_title("Embedded steel vs projected host axial strain")
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()
    save(fig, out_dirs, f"reduced_rc_structural_continuum_embedded_transfer_strain_{suffix}")

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for hex_order, rows in continuum_steel_histories.items():
        color = palette.get(hex_order, BLUE)
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [float(row["projected_axial_strain_gap"]) for row in rows],
            color=color,
            linewidth=1.5,
            label=f"Host-bar strain gap ({hex_order.upper()})",
        )
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in rows],
            [float(row["projected_axial_gap_m"]) for row in rows],
            color=color,
            linewidth=1.2,
            linestyle="--",
            label=f"Projected axial gap ({hex_order.upper()})",
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Embedded transfer mismatch")
    ax.set_title("Embedded host-vs-bar transfer diagnostics")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dirs, f"reduced_rc_structural_continuum_embedded_transfer_gap_{suffix}")


def make_hinge_band_figures(
    structural_hinge_traces: dict[str, list[dict[str, object]]],
    continuum_hinge_traces: dict[str, list[dict[str, object]]],
    out_dirs: list[Path],
    suffix: str,
) -> None:
    palette = {"hex8": ORANGE, "hex20": ORANGE, "hex27": GREEN}

    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for hex_order, structural_rows in structural_hinge_traces.items():
        color = palette.get(hex_order, BLUE)
        continuum_rows = continuum_hinge_traces[hex_order]
        ax.plot(
            [float(row["strain_xx"]) for row in structural_rows],
            [float(row["stress_xx_MPa"]) for row in structural_rows],
            color=color,
            linewidth=1.6,
            label=f"Structural hinge-band steel ({hex_order.upper()})",
        )
        ax.plot(
            [float(row["axial_strain"]) for row in continuum_rows],
            [float(row["stress_xx_MPa"]) for row in continuum_rows],
            color=color,
            linewidth=1.3,
            linestyle="--",
            label=f"Continuum hinge-band steel ({hex_order.upper()})",
        )
    ax.set_xlabel(r"Band-averaged steel strain $\bar{\varepsilon}$")
    ax.set_ylabel(r"Band-averaged steel stress $\bar{\sigma}$ [MPa]")
    ax.set_title("Fixed-end hinge-band steel hysteresis")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(
        fig,
        out_dirs,
        f"reduced_rc_structural_continuum_steel_hinge_band_hysteresis_{suffix}",
    )

    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    for hex_order, structural_rows in structural_hinge_traces.items():
        color = palette.get(hex_order, BLUE)
        continuum_rows = continuum_hinge_traces[hex_order]
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in structural_rows],
            [float(row["stress_xx_MPa"]) for row in structural_rows],
            color=color,
            linewidth=1.6,
            label=f"Structural hinge-band steel ({hex_order.upper()})",
        )
        ax.plot(
            [1.0e3 * float(row["drift_m"]) for row in continuum_rows],
            [float(row["stress_xx_MPa"]) for row in continuum_rows],
            color=color,
            linewidth=1.3,
            linestyle="--",
            label=f"Continuum hinge-band steel ({hex_order.upper()})",
        )
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel(r"Band-averaged steel stress $\bar{\sigma}$ [MPa]")
    ax.set_title("Fixed-end hinge-band steel stress along protocol")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(
        fig,
        out_dirs,
        f"reduced_rc_structural_continuum_steel_hinge_band_stress_vs_drift_{suffix}",
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    out_dirs = (
        []
        if args.skip_figure_export
        else [args.figures_dir.resolve(), args.secondary_figures_dir.resolve()]
    )

    existing_summary_path = output_dir / "structural_continuum_steel_hysteresis_summary.json"
    if args.reuse_existing and existing_summary_path.exists():
        existing_protocol = read_json(existing_summary_path).get("protocol", {})
        if isinstance(existing_protocol, dict):
            existing_amplitudes = existing_protocol.get("amplitudes_mm")
            if args.analysis == "cyclic" and isinstance(existing_amplitudes, list) and existing_amplitudes:
                args.amplitudes_mm = ",".join(
                    str(float(value)) for value in existing_amplitudes
                )
            args.monotonic_tip_mm = float(
                existing_protocol.get("monotonic_tip_mm", args.monotonic_tip_mm)
            )
            args.monotonic_steps = int(
                existing_protocol.get("monotonic_steps", args.monotonic_steps)
            )
            args.steps_per_segment = int(
                existing_protocol.get("steps_per_segment", args.steps_per_segment)
            )
            args.continuation = str(
                existing_protocol.get("continuation", args.continuation)
            )
            args.continuation_segment_substep_factor = int(
                existing_protocol.get(
                    "continuation_segment_substep_factor",
                    args.continuation_segment_substep_factor,
                )
            )
            args.axial_compression_mn = float(
                existing_protocol.get("axial_compression_mn", args.axial_compression_mn)
            )
            args.axial_preload_steps = int(
                existing_protocol.get("axial_preload_steps", args.axial_preload_steps)
            )

    amplitudes_mm = parse_csv_floats(args.amplitudes_mm)
    structural_dir = (
        args.structural_bundle_dir.resolve()
        if args.structural_bundle_dir is not None
        else output_dir / "structural"
    )
    if args.structural_bundle_dir is not None:
        structural_elapsed = math.nan
        structural_manifest = require_existing_bundle(
            structural_dir, "runtime_manifest.json"
        )
    else:
        structural_elapsed, structural_manifest = run_case(
            [
                arg
                for arg in structural_command(
                    args.structural_exe.resolve(), structural_dir, args
                )
                if arg != "--no-op"
            ],
            structural_dir,
            "runtime_manifest.json",
            args.reuse_existing,
            timeout_seconds=args.structural_timeout_seconds,
        )
    structural_hist = read_csv_rows(structural_dir / "hysteresis.csv")
    structural_fiber_rows = read_csv_rows(structural_dir / "section_fiber_state_history.csv")
    section_response_rows = read_csv_rows(structural_dir / "section_response.csv")
    nominal_structural_identity = select_structural_steel_identity(structural_fiber_rows)
    structural_active_coordinate, _ = infer_structural_active_coordinate(
        section_response_rows
    )
    target_continuum_bar_y, target_continuum_bar_z = (
        structural_to_continuum_rebar_coordinates(
            nominal_structural_identity.y,
            nominal_structural_identity.z,
            structural_active_coordinate,
        )
    )

    continuum_histories: dict[str, list[dict[str, object]]] = {}
    continuum_steel_histories: dict[str, list[dict[str, object]]] = {}
    structural_steel_traces: dict[str, list[dict[str, object]]] = {}
    structural_trace_selections: dict[str, StructuralSteelTraceSelection] = {}
    structural_orientation_candidates: dict[str, list[StructuralOrientationCandidate]] = {}
    continuum_summaries: dict[str, dict[str, object]] = {}
    continuum_identities: dict[str, ContinuumSteelIdentity] = {}
    selected_structural_identities: dict[str, StructuralSteelIdentity] = {}
    structural_hinge_traces: dict[str, list[dict[str, object]]] = {}
    continuum_hinge_traces: dict[str, list[dict[str, object]]] = {}
    hinge_band = hinge_band_descriptor(args)

    hex_orders = parse_csv_strings(args.hex_orders)
    if args.continuum_bundle_dir is not None and len(hex_orders) != 1:
        raise RuntimeError(
            "--continuum-bundle-dir can only be used with a single --hex-orders value."
        )
    if args.continuum_bundle_dir is not None and args.continuum_bundle_root is not None:
        raise RuntimeError(
            "Use either --continuum-bundle-dir or --continuum-bundle-root, not both."
        )

    for hex_order in hex_orders:
        artifact_dir = output_dir / hex_order
        ensure_dir(artifact_dir)
        if args.continuum_bundle_dir is not None:
            case_dir = args.continuum_bundle_dir.resolve()
            elapsed = math.nan
            manifest = require_existing_bundle(case_dir, "runtime_manifest.json")
        elif args.continuum_bundle_root is not None:
            case_dir = (args.continuum_bundle_root.resolve() / hex_order)
            elapsed = math.nan
            manifest = require_existing_bundle(case_dir, "runtime_manifest.json")
        else:
            case_dir = artifact_dir
            elapsed, manifest = run_case(
                continuum_command(args.continuum_exe.resolve(), case_dir, hex_order, args),
                case_dir,
                "runtime_manifest.json",
                args.reuse_existing,
                timeout_seconds=args.continuum_timeout_seconds,
            )
        hist_rows = read_csv_rows(case_dir / "hysteresis.csv")
        rebar_rows = read_csv_rows(case_dir / "rebar_history.csv")
        if not bool(manifest.get("completed_successfully", False)):
            continuum_summaries[hex_order] = {
                "process_wall_seconds": clean_optional_number(elapsed),
                "reported_total_wall_seconds": total_wall_seconds(manifest),
                "reported_solve_wall_seconds": solve_wall_seconds(manifest),
                "completed_successfully": False,
                "termination_reason": manifest.get("termination_reason", "failed"),
                "solve_summary": manifest.get("solve_summary", {}),
                "bundle_dir": str(case_dir),
            }
            continue
        if not rebar_rows:
            raise RuntimeError(
                "Continuum benchmark completed successfully but did not export "
                "rebar_history.csv rows."
            )
        continuum_identity = select_continuum_steel_identity(
            rebar_rows, target_continuum_bar_y, target_continuum_bar_z
        )
        (
            structural_identity,
            structural_steel_trace,
            structural_trace_selection,
            orientation_candidates,
        ) = select_structural_orientation_matched_trace(
            structural_fiber_rows,
            section_response_rows,
            steel_trace := select_continuum_steel_trace(rebar_rows, continuum_identity),
            continuum_identity,
            args.column_height_m
        )
        continuum_histories[hex_order] = hist_rows
        structural_steel_traces[hex_order] = structural_steel_trace
        structural_trace_selections[hex_order] = structural_trace_selection
        structural_orientation_candidates[hex_order] = orientation_candidates
        continuum_steel_histories[hex_order] = steel_trace
        continuum_identities[hex_order] = continuum_identity
        selected_structural_identities[hex_order] = structural_identity
        continuum_hinge_trace = continuum_hinge_band_trace(
            rebar_rows,
            continuum_identity,
            hinge_band.length_m,
        )
        structural_hinge_trace = structural_hinge_band_trace(
            structural_fiber_rows,
            structural_identity,
            args.column_height_m,
            hinge_band.length_m,
        )
        continuum_hinge_traces[hex_order] = continuum_hinge_trace
        structural_hinge_traces[hex_order] = structural_hinge_trace
        continuum_summaries[hex_order] = {
            "bundle_dir": str(case_dir),
            "process_wall_seconds": clean_optional_number(elapsed),
            "reported_total_wall_seconds": total_wall_seconds(manifest),
            "reported_solve_wall_seconds": solve_wall_seconds(manifest),
            "completed_successfully": manifest["completed_successfully"],
            "matched_structural_station": asdict(structural_trace_selection),
            "selected_structural_fiber": asdict(structural_identity),
            "selected_bar": asdict(continuum_identity),
        }

        write_csv(artifact_dir / "selected_continuum_steel_trace.csv", steel_trace)
        write_csv(
            artifact_dir / "matched_structural_steel_trace.csv",
            structural_steel_trace,
        )
        write_csv(
            artifact_dir / "structural_orientation_candidates.csv",
            [asdict(candidate) for candidate in orientation_candidates],
        )
        write_csv(
            artifact_dir / "selected_continuum_steel_hinge_band_trace.csv",
            continuum_hinge_trace,
        )
        write_csv(
            artifact_dir / "matched_structural_steel_hinge_band_trace.csv",
            structural_hinge_trace,
        )

    suffix = amplitude_suffix_from_args(args)
    if out_dirs and continuum_histories:
        make_overlay_figures(
            structural_hist,
            structural_steel_traces,
            continuum_histories,
            continuum_steel_histories,
            out_dirs,
            suffix,
        )
        make_hinge_band_figures(
            structural_hinge_traces,
            continuum_hinge_traces,
            out_dirs,
            suffix,
        )

    global_comparison: dict[str, dict[str, float]] = {}
    steel_comparison: dict[str, dict[str, float]] = {}
    steel_hinge_band_comparison: dict[str, dict[str, float]] = {}
    embedded_transfer_comparison: dict[str, dict[str, float]] = {}
    for hex_order, hist_rows in continuum_histories.items():
        global_max, global_rms = branchwise_interpolation_error(
            hist_rows,
            structural_hist,
            "drift_m",
            "base_shear_MN",
            "base_shear_MN",
        )
        global_peak_max, global_peak_rms = branchwise_peak_normalized_error(
            hist_rows,
            structural_hist,
            "drift_m",
            "base_shear_MN",
            "base_shear_MN",
        )
        steel_stress_max, steel_stress_rms = branchwise_interpolation_error(
            continuum_steel_histories[hex_order],
            structural_steel_traces[hex_order],
            "drift_m",
            "stress_xx_MPa",
            "stress_xx_MPa",
        )
        steel_stress_peak_max, steel_stress_peak_rms = (
            branchwise_peak_normalized_error(
                continuum_steel_histories[hex_order],
                structural_steel_traces[hex_order],
                "drift_m",
                "stress_xx_MPa",
                "stress_xx_MPa",
            )
        )
        steel_strain_max, steel_strain_rms = branchwise_interpolation_error(
            continuum_steel_histories[hex_order],
            structural_steel_traces[hex_order],
            "drift_m",
            "axial_strain",
            "strain_xx",
        )
        steel_strain_peak_max, steel_strain_peak_rms = (
            branchwise_peak_normalized_error(
                continuum_steel_histories[hex_order],
                structural_steel_traces[hex_order],
                "drift_m",
                "axial_strain",
                "strain_xx",
            )
        )
        structural_loop_work = cyclic_loop_work(
            structural_steel_traces[hex_order], "strain_xx", "stress_xx_MPa"
        )
        continuum_loop_work = cyclic_loop_work(
            continuum_steel_histories[hex_order], "axial_strain", "stress_xx_MPa"
        )
        hinge_stress_max, hinge_stress_rms = branchwise_interpolation_error(
            continuum_hinge_traces[hex_order],
            structural_hinge_traces[hex_order],
            "drift_m",
            "stress_xx_MPa",
            "stress_xx_MPa",
        )
        hinge_stress_peak_max, hinge_stress_peak_rms = (
            branchwise_peak_normalized_error(
                continuum_hinge_traces[hex_order],
                structural_hinge_traces[hex_order],
                "drift_m",
                "stress_xx_MPa",
                "stress_xx_MPa",
            )
        )
        hinge_strain_max, hinge_strain_rms = branchwise_interpolation_error(
            continuum_hinge_traces[hex_order],
            structural_hinge_traces[hex_order],
            "drift_m",
            "axial_strain",
            "strain_xx",
        )
        hinge_strain_peak_max, hinge_strain_peak_rms = (
            branchwise_peak_normalized_error(
                continuum_hinge_traces[hex_order],
                structural_hinge_traces[hex_order],
                "drift_m",
                "axial_strain",
                "strain_xx",
            )
        )
        structural_hinge_loop_work = cyclic_loop_work(
            structural_hinge_traces[hex_order], "strain_xx", "stress_xx_MPa"
        )
        continuum_hinge_loop_work = cyclic_loop_work(
            continuum_hinge_traces[hex_order], "axial_strain", "stress_xx_MPa"
        )
        global_comparison[hex_order] = {
            "max_rel_base_shear_error": global_max,
            "rms_rel_base_shear_error": global_rms,
            "peak_normalized_max_base_shear_error": global_peak_max,
            "peak_normalized_rms_base_shear_error": global_peak_rms,
        }
        steel_comparison[hex_order] = {
            "max_rel_stress_vs_drift_error": steel_stress_max,
            "rms_rel_stress_vs_drift_error": steel_stress_rms,
            "peak_normalized_max_stress_vs_drift_error": steel_stress_peak_max,
            "peak_normalized_rms_stress_vs_drift_error": steel_stress_peak_rms,
            "max_rel_strain_vs_drift_error": steel_strain_max,
            "rms_rel_strain_vs_drift_error": steel_strain_rms,
            "peak_normalized_max_strain_vs_drift_error": steel_strain_peak_max,
            "peak_normalized_rms_strain_vs_drift_error": steel_strain_peak_rms,
            "structural_loop_work_mpa": structural_loop_work,
            "continuum_loop_work_mpa": continuum_loop_work,
            "continuum_to_structural_loop_work_ratio": (
                continuum_loop_work / structural_loop_work
                if abs(structural_loop_work) > 1.0e-12
                else math.nan
            ),
        }
        steel_hinge_band_comparison[hex_order] = {
            "max_rel_stress_vs_drift_error": hinge_stress_max,
            "rms_rel_stress_vs_drift_error": hinge_stress_rms,
            "peak_normalized_max_stress_vs_drift_error": hinge_stress_peak_max,
            "peak_normalized_rms_stress_vs_drift_error": hinge_stress_peak_rms,
            "max_rel_strain_vs_drift_error": hinge_strain_max,
            "rms_rel_strain_vs_drift_error": hinge_strain_rms,
            "peak_normalized_max_strain_vs_drift_error": hinge_strain_peak_max,
            "peak_normalized_rms_strain_vs_drift_error": hinge_strain_peak_rms,
            "structural_loop_work_mpa": structural_hinge_loop_work,
            "continuum_loop_work_mpa": continuum_hinge_loop_work,
            "continuum_to_structural_loop_work_ratio": (
                continuum_hinge_loop_work / structural_hinge_loop_work
                if abs(structural_hinge_loop_work) > 1.0e-12
                else math.nan
            ),
            "band_length_m": hinge_band.length_m,
            "selection_rule": hinge_band.selection_rule,
            "structural_unique_position_count": (
                structural_hinge_traces[hex_order][0]["unique_position_count"]
                if structural_hinge_traces[hex_order]
                else math.nan
            ),
            "continuum_unique_position_count": (
                continuum_hinge_traces[hex_order][0]["unique_position_count"]
                if continuum_hinge_traces[hex_order]
                else math.nan
            ),
        }
        rows = continuum_steel_histories[hex_order]
        strain_gaps = [abs(float(row["projected_axial_strain_gap"])) for row in rows]
        kinematic_gaps = [
            abs(
                float(row["axial_strain"])
                - float(row["rebar_projected_axial_strain"])
            )
            for row in rows
        ]
        axial_gaps = [abs(float(row["projected_axial_gap_m"])) for row in rows]
        norm_gaps = [abs(float(row["projected_gap_norm_m"])) for row in rows]
        embedded_transfer_comparison[hex_order] = {
            "max_abs_host_bar_axial_strain_gap": max(strain_gaps, default=math.nan),
            "rms_abs_host_bar_axial_strain_gap": (
                math.sqrt(sum(value * value for value in strain_gaps) / len(strain_gaps))
                if strain_gaps
                else math.nan
            ),
            "max_abs_rebar_kinematic_consistency_gap": max(
                kinematic_gaps, default=math.nan
            ),
            "max_abs_projected_axial_gap_m": max(axial_gaps, default=math.nan),
            "max_abs_projected_gap_norm_m": max(norm_gaps, default=math.nan),
        }

    representative_hex_order = next(iter(continuum_histories), "")
    summary = {
        "protocol": {
            "analysis": args.analysis,
            "amplitudes_mm": amplitudes_mm,
            "monotonic_tip_mm": args.monotonic_tip_mm,
            "monotonic_steps": args.monotonic_steps,
            "steps_per_segment": args.steps_per_segment,
            "continuation": args.continuation,
            "continuation_segment_substep_factor": args.continuation_segment_substep_factor,
            "axial_compression_mn": args.axial_compression_mn,
            "axial_preload_steps": args.axial_preload_steps,
        },
        "structural_reference": {
            "bundle_dir": str(structural_dir),
            "beam_nodes": args.beam_nodes,
            "beam_integration": args.beam_integration,
            "top_rotation_mode": args.structural_top_rotation_mode,
            "top_rotation_drift_ratio": args.structural_top_rotation_drift_ratio,
            "section_fiber_profile": structural_section_fiber_profile(
                structural_manifest,
                args.structural_section_fiber_profile,
            ),
            "solver_policy": args.solver_policy,
            "process_wall_seconds": clean_optional_number(structural_elapsed),
            "reported_total_wall_seconds": total_wall_seconds(structural_manifest),
            "reported_solve_wall_seconds": solve_wall_seconds(structural_manifest),
            "nominal_selected_steel_bar": asdict(nominal_structural_identity),
            "structural_active_coordinate": structural_active_coordinate,
            "target_continuum_bar_coordinates": {
                "bar_y_local_x": target_continuum_bar_y,
                "bar_z_local_y": target_continuum_bar_z,
                "mapping": (
                    "structural active coordinate maps with opposite sign to "
                    "continuum RebarBar::ly; structural passive coordinate "
                    "maps directly to RebarBar::lz"
                ),
            },
        },
        "continuum_cases": continuum_summaries,
        "continuum_reference_spec": {
            "nx": args.nx,
            "ny": args.ny,
            "nz": args.nz,
            "longitudinal_bias_power": args.longitudinal_bias_power,
            "longitudinal_bias_location": args.longitudinal_bias_location,
            "material_mode": args.continuum_material_mode,
            "kinematics": args.continuum_kinematics,
            "concrete_tension_stiffness_ratio": (
                args.concrete_tension_stiffness_ratio
            ),
            "concrete_fracture_energy_nmm": args.concrete_fracture_energy_nmm,
            "concrete_reference_length_mm": args.concrete_reference_length_mm,
            "concrete_crack_band_residual_tension_ratio": (
                args.concrete_crack_band_residual_tension_ratio
            ),
            "concrete_crack_band_residual_shear_ratio": (
                args.concrete_crack_band_residual_shear_ratio
            ),
            "concrete_crack_band_large_opening_residual_shear_ratio": (
                args.concrete_crack_band_large_opening_residual_shear_ratio
            ),
            "concrete_crack_band_shear_retention_decay_strain": (
                args.concrete_crack_band_shear_retention_decay_strain
            ),
            "concrete_crack_band_shear_transfer_law": (
                args.concrete_crack_band_shear_transfer_law
            ),
            "concrete_crack_band_closure_shear_gain": (
                args.concrete_crack_band_closure_shear_gain
            ),
            "concrete_crack_band_open_compression_transfer_ratio": (
                args.concrete_crack_band_open_compression_transfer_ratio
            ),
            "concrete_profile": args.continuum_concrete_profile,
            "concrete_tangent_mode": args.continuum_concrete_tangent_mode,
            "characteristic_length_mode": args.continuum_characteristic_length_mode,
            "host_concrete_zoning_mode": args.host_concrete_zoning_mode,
            "transverse_mesh_mode": args.transverse_mesh_mode,
            "rebar_layout": args.continuum_rebar_layout,
            "rebar_interpolation": args.continuum_rebar_interpolation,
            "embedded_boundary_mode": args.embedded_boundary_mode,
            "axial_preload_transfer_mode": args.axial_preload_transfer_mode,
            "top_cap_mode": args.continuum_top_cap_mode,
            "top_cap_penalty_alpha_scale_over_ec": (
                args.continuum_top_cap_penalty_alpha_scale_over_ec
            ),
            "top_cap_bending_rotation_drift_ratio": (
                args.continuum_top_cap_bending_rotation_drift_ratio
            ),
            "penalty_alpha_scale_over_ec": args.penalty_alpha_scale_over_ec,
            "structural_to_continuum_axis_map": {
                "continuum_bar_y_field": "RebarBar::ly, prism local x",
                "continuum_bar_z_field": "RebarBar::lz, prism local y",
                "active_bending_sign": "structural active = -continuum bar_y",
                "passive_bending_sign": "structural passive = continuum bar_z",
            },
        },
        "global_comparison": global_comparison,
        "steel_local_comparison": steel_comparison,
        "steel_hinge_band": asdict(hinge_band),
        "steel_hinge_band_comparison": steel_hinge_band_comparison,
        "embedded_transfer_comparison": embedded_transfer_comparison,
        "artifacts": {
            "structural_trace_csv": (
                str(
                    output_dir
                    / representative_hex_order
                    / "matched_structural_steel_trace.csv"
                )
                if representative_hex_order
                else ""
            ),
            "amplitude_suffix": suffix,
            "figure_hysteresis_overlay": str(
                args.figures_dir
                / f"reduced_rc_structural_continuum_hysteresis_overlay_{suffix}.png"
            )
            if representative_hex_order
            else "",
            "figure_steel_hysteresis_overlay": str(
                args.figures_dir
                / f"reduced_rc_structural_continuum_steel_hysteresis_overlay_{suffix}.png"
            )
            if representative_hex_order
            else "",
            "figure_steel_stress_vs_drift": str(
                args.figures_dir
                / f"reduced_rc_structural_continuum_steel_stress_vs_drift_{suffix}.png"
            )
            if representative_hex_order
            else "",
            "figure_steel_strain_vs_drift": str(
                args.figures_dir
                / f"reduced_rc_structural_continuum_steel_strain_vs_drift_{suffix}.png"
            )
            if representative_hex_order
            else "",
            "figure_embedded_transfer_strain": str(
                args.figures_dir
                / f"reduced_rc_structural_continuum_embedded_transfer_strain_{suffix}.png"
            )
            if representative_hex_order
            else "",
            "figure_embedded_transfer_gap": str(
                args.figures_dir
                / f"reduced_rc_structural_continuum_embedded_transfer_gap_{suffix}.png"
            )
            if representative_hex_order
            else "",
            "figure_steel_hinge_band_hysteresis": str(
                args.figures_dir
                / f"reduced_rc_structural_continuum_steel_hinge_band_hysteresis_{suffix}.png"
            )
            if representative_hex_order
            else "",
            "figure_steel_hinge_band_stress_vs_drift": str(
                args.figures_dir
                / f"reduced_rc_structural_continuum_steel_hinge_band_stress_vs_drift_{suffix}.png"
            )
            if representative_hex_order
            else "",
        },
    }
    for hex_order in continuum_histories:
        summary["artifacts"][f"{hex_order}_orientation_candidates_csv"] = str(
            output_dir / hex_order / "structural_orientation_candidates.csv"
        )
        summary["artifacts"][f"{hex_order}_trace_csv"] = str(
            output_dir / hex_order / "selected_continuum_steel_trace.csv"
        )
        summary["artifacts"][f"{hex_order}_matched_structural_trace_csv"] = str(
            output_dir / hex_order / "matched_structural_steel_trace.csv"
        )
        summary["artifacts"][f"{hex_order}_hinge_band_trace_csv"] = str(
            output_dir / hex_order / "selected_continuum_steel_hinge_band_trace.csv"
        )
        summary["artifacts"][f"{hex_order}_matched_structural_hinge_band_trace_csv"] = str(
            output_dir / hex_order / "matched_structural_steel_hinge_band_trace.csv"
        )
    summary["selected_structural_fibers"] = {
        hex_order: asdict(identity)
        for hex_order, identity in selected_structural_identities.items()
    }
    summary["structural_orientation_candidates"] = {
        hex_order: [asdict(candidate) for candidate in candidates]
        for hex_order, candidates in structural_orientation_candidates.items()
    }

    write_json(output_dir / "structural_continuum_steel_hysteresis_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
