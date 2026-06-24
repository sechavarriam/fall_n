#!/usr/bin/env python3
"""Run and postprocess the reduced-RC local model promotion matrix.

The campaign compares Ko-Bathe continuum and shifted-Heaviside XFEM local
models against the audited cyclic column references. Each case writes a local
manifest, a global campaign manifest, ranking metrics, and comparison figures.

Notes for publication use:

* Ko-Bathe VTK comes from the native continuum benchmark when ``--write-vtk``
  is enabled.
* XFEM currently writes CSV state histories only. This script therefore writes
  an explicit observable-axis/crack-plane VTK proxy and marks it as proxy in
  the case manifest. It is useful for quick inspection, but it is not a
  substitute for native XFEM volumetric VTK.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

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
        "savefig.dpi": 250,
    }
)


FAMILY_LABEL = {
    "kobathe": "Ko-Bathe",
    "xfem": "XFEM",
}

HEX_LABEL = {
    "hex8": "Hex8",
    "hex20": "Hex20",
    "hex27": "Hex27",
}

BIAS_LABEL = {
    "uniform": "uniform",
    "fixed-end": "fixed-end bias",
    "loaded-end": "loaded-end bias",
    "both-ends": "both-ends bias",
}


@dataclass(frozen=True)
class CaseSpec:
    family: str
    hex_order: str
    nx: int
    ny: int
    nz: int
    bias_mode: str

    @property
    def case_id(self) -> str:
        bias = self.bias_mode.replace("-", "")
        return (
            f"{self.family}_{self.hex_order}_nx{self.nx}_ny{self.ny}_"
            f"nz{self.nz}_{bias}"
        )


@dataclass
class CaseResult:
    case_id: str
    family: str
    hex_order: str
    nx: int
    ny: int
    nz: int
    bias_mode: str
    status: str
    returncode: int | None
    elapsed_seconds: float
    completed_200mm: bool
    peak_abs_drift_mm: float
    peak_abs_base_shear_mn: float
    work_mn_m: float
    opensees_rms_rel: float
    opensees_max_rel: float
    opensees_work_rel: float
    lobatto_n4_rms_rel: float
    lobatto_n6_rms_rel: float
    lobatto_n8_rms_rel: float
    steel_metric_status: str
    vtk_status: str
    output_dir: str
    log_path: str
    manifest_path: str
    failure_reason: str = ""


def parse_csv_list(value: str, cast=str) -> list[Any]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(cast(item))
    return out


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    return (
        root
        / "data/output/cyclic_validation"
        / f"local_model_promotion_matrix_200mm_{stamp}"
    )


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="exhaustive_200mm")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir(root))
    parser.add_argument(
        "--families",
        default="kobathe,xfem",
        help="Comma list selected from kobathe,xfem.",
    )
    parser.add_argument("--hex-orders", default="hex27")
    parser.add_argument("--nx-list", default="1,2,3")
    parser.add_argument("--ny-list", default="1,2")
    parser.add_argument("--nz-list", default="1,2,3,4")
    parser.add_argument("--bias-modes", default="uniform,fixed-end,loaded-end")
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
    parser.add_argument("--steps-per-segment", type=int, default=4)
    parser.add_argument("--max-bisections", type=int, default=12)
    parser.add_argument("--vtk-stride", type=int, default=1)
    parser.add_argument("--write-vtk", action="store_true")
    parser.add_argument(
        "--abort-base-shear-threshold-kn",
        type=float,
        default=50.0,
        help="Abort a case after the first accepted state exceeding this |V_base|. Use <=0 to disable.",
    )
    parser.add_argument("--kobathe-rebar-interpolation", default="automatic")
    parser.add_argument("--kobathe-bond-slip-reference", default="5e-4")
    parser.add_argument("--kobathe-bond-slip-residual-ratio", default="0.2")
    parser.add_argument(
        "--kobathe-bond-slip-adaptive-reference-max-factor",
        default="1",
    )
    parser.add_argument(
        "--kobathe-bond-slip-adaptive-residual-ratio-floor",
        default="-1",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Regenerate summary tables and figures from existing case manifests without launching solvers.",
    )
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-filter", default="")
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--print-progress", action="store_true")
    parser.add_argument(
        "--kobathe-exe",
        type=Path,
        default=root / "build/fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
    )
    parser.add_argument(
        "--xfem-exe",
        type=Path,
        default=root / "build/fall_n_reduced_rc_xfem_reference_benchmark.exe",
    )
    parser.add_argument(
        "--opensees-hifi-hysteresis",
        type=Path,
        default=root
        / "data/output/cyclic_validation/opensees_hifi_timoshenko_matrix_200mm_publication_20260518/hysteresis.csv",
    )
    parser.add_argument(
        "--structural-matrix-dir",
        type=Path,
        default=root
        / "data/output/cyclic_validation/timoshenko_matrix_reproduced_historical_closure_20260520/fall_n_matrix",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=root / "doc/figures/validation_reboot/local_model_promotion_matrix_200mm",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=root
        / "PhD_Thesis/Figuras/validation_reboot/local_model_promotion_matrix_200mm",
    )
    parser.add_argument("--column-height-m", type=float, default=3.0)
    parser.add_argument(
        "--xfem-proxy-vtk",
        action="store_true",
        help="Write explicit proxy VTK for XFEM CSV histories.",
    )
    parser.set_defaults(write_vtk=False, xfem_proxy_vtk=True)
    return parser.parse_args()


def json_sanitized(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_sanitized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_sanitized(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_sanitized(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def case_result_from_dict(payload: dict[str, Any]) -> CaseResult:
    data = dict(payload)
    for key in (
        "elapsed_seconds",
        "peak_abs_drift_mm",
        "peak_abs_base_shear_mn",
        "work_mn_m",
        "opensees_rms_rel",
        "opensees_max_rel",
        "opensees_work_rel",
        "lobatto_n4_rms_rel",
        "lobatto_n6_rms_rel",
        "lobatto_n8_rms_rel",
    ):
        if data.get(key) is None:
            data[key] = math.nan
    return CaseResult(**data)


def fmt_float(value: float, fmt: str = ".3g") -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return format(float(value), fmt)
    return "n/a"


def case_specs(args: argparse.Namespace) -> list[CaseSpec]:
    families = parse_csv_list(args.families)
    hex_orders = parse_csv_list(args.hex_orders)
    nx_values = parse_csv_list(args.nx_list, int)
    ny_values = parse_csv_list(args.ny_list, int)
    nz_values = parse_csv_list(args.nz_list, int)
    bias_modes = parse_csv_list(args.bias_modes)
    specs: list[CaseSpec] = []
    for family in families:
        if family not in FAMILY_LABEL:
            raise ValueError(f"Unsupported family {family}")
        for hex_order in hex_orders:
            if hex_order not in HEX_LABEL:
                raise ValueError(f"Unsupported hex order {hex_order}")
            for nx in nx_values:
                for ny in ny_values:
                    for nz in nz_values:
                        for bias in bias_modes:
                            if bias not in BIAS_LABEL:
                                raise ValueError(f"Unsupported bias mode {bias}")
                            spec = CaseSpec(family, hex_order, nx, ny, nz, bias)
                            if args.case_filter and args.case_filter not in spec.case_id:
                                continue
                            specs.append(spec)
    if args.case_limit > 0:
        specs = specs[: args.case_limit]
    return specs


def bias_power(spec: CaseSpec) -> str:
    return "1.0" if spec.bias_mode == "uniform" else "2.5"


def bias_location(spec: CaseSpec) -> str:
    if spec.bias_mode == "uniform":
        return "fixed-end"
    return spec.bias_mode


def build_command(args: argparse.Namespace, spec: CaseSpec, case_dir: Path) -> list[str]:
    if spec.family == "kobathe":
        cmd = [
            str(args.kobathe_exe),
            "--output-dir",
            str(case_dir),
            "--analysis",
            "cyclic",
            "--continuum-kinematics",
            "corotational",
            "--material-mode",
            "nonlinear",
            "--concrete-profile",
            "production-stabilized",
            "--hex-order",
            spec.hex_order,
            "--nx",
            str(spec.nx),
            "--ny",
            str(spec.ny),
            "--nz",
            str(spec.nz),
            "--longitudinal-bias-power",
            bias_power(spec),
            "--longitudinal-bias-location",
            bias_location(spec),
            "--reinforcement-mode",
            "embedded-longitudinal-bars",
            "--rebar-interpolation",
            args.kobathe_rebar_interpolation,
            "--rebar-layout",
            "structural-matched-eight-bar",
            "--axial-compression-mn",
            "0.02",
            "--axial-preload-steps",
            "4",
            "--amplitudes-mm",
            args.amplitudes_mm,
            "--steps-per-segment",
            str(args.steps_per_segment),
            "--max-bisections",
            str(args.max_bisections),
            "--continuation",
            "reversal-guarded",
            "--continuation-segment-substep-factor",
            "2",
            "--bond-slip",
            "--bond-slip-reference",
            args.kobathe_bond_slip_reference,
            "--bond-slip-residual-ratio",
            args.kobathe_bond_slip_residual_ratio,
            "--bond-slip-adaptive-reference-max-factor",
            args.kobathe_bond_slip_adaptive_reference_max_factor,
            "--bond-slip-adaptive-residual-ratio-floor",
            args.kobathe_bond_slip_adaptive_residual_ratio_floor,
            "--print-progress",
        ]
        if args.abort_base_shear_threshold_kn > 0.0:
            cmd += [
                "--abort-base-shear-threshold-kn",
                f"{args.abort_base_shear_threshold_kn:g}",
            ]
        if args.write_vtk:
            cmd += ["--write-vtk", "--vtk-stride", str(args.vtk_stride)]
        return cmd

    crack_plane = "20:3:0,0,0.6:0.0990147542977,0.0990147542977,0.990147542977"
    return [
        str(args.xfem_exe),
        "--output-dir",
        str(case_dir),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--axial-compression-mn",
        "0.02",
        "--global-xfem-hex-order",
        spec.hex_order,
        "--global-xfem-nx",
        str(spec.nx),
        "--global-xfem-ny",
        str(spec.ny),
        "--global-xfem-nz",
        str(spec.nz),
        "--global-xfem-bias-power",
        bias_power(spec),
        "--global-xfem-bias-location",
        bias_location(spec),
        "--global-xfem-kinematic-formulation",
        "corotational",
        "--global-xfem-concrete-material",
        "cyclic-crack-band",
        "--global-xfem-shear-cap-mpa",
        "0.02",
        "--global-xfem-continuation",
        "mixed-arc-length",
        "--global-xfem-mixed-arc-target",
        "0.5",
        "--global-xfem-solver-profile",
        "l2",
        "--global-xfem-adaptive-increments",
        "--global-xfem-max-bisections",
        str(max(args.max_bisections, 8)),
        "--global-xfem-solver-max-iterations",
        "120",
        "--global-xfem-crack-crossing-rebar-area-scale",
        "1",
        "--global-xfem-crack-crossing-rebar-mode",
        "dowel-x",
        "--global-xfem-crack-crossing-bridge-law",
        "bounded-slip",
        "--global-xfem-crack-crossing-axis-frame",
        "corotational-host",
        "--global-xfem-crack-crossing-host-axis-tangent",
        "finite-difference",
        "--global-xfem-crack-crossing-yield-force-mn",
        "0.0019",
        "--global-xfem-crack-crossing-force-cap-mn",
        "0.0019",
        "--xfem-crack-plane",
        crack_plane,
    ]


def run_command(cmd: list[str], log_path: Path, cwd: Path, timeout: float) -> tuple[str, int | None, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout if timeout > 0 else None,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout} seconds\n")
            return "timeout", None, f"timeout after {timeout} seconds"
    if proc.returncode == 0:
        return "ran", proc.returncode, ""
    return "failed_process", proc.returncode, f"process return code {proc.returncode}"


def scalar(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return math.nan


def normalized_hysteresis_rows(case_dir: Path, family: str) -> list[dict[str, float]]:
    paths = (
        [case_dir / "global_xfem_newton_hysteresis.csv", case_dir / "hysteresis.csv"]
        if family == "xfem"
        else [case_dir / "hysteresis.csv"]
    )
    for path in paths:
        rows = read_csv(path)
        if rows:
            out: list[dict[str, float]] = []
            for row in rows:
                drift_m = scalar(row, "drift_m")
                if not math.isfinite(drift_m):
                    drift_mm = scalar(row, "drift_mm")
                    drift_m = drift_mm / 1000.0 if math.isfinite(drift_mm) else math.nan
                out.append(
                    {
                        "step": scalar(row, "step"),
                        "p": scalar(row, "p"),
                        "drift_m": drift_m,
                        "drift_mm": drift_m * 1000.0 if math.isfinite(drift_m) else math.nan,
                        "base_shear_MN": scalar(row, "base_shear_MN", "base_shear_mn"),
                    }
                )
            return out
    return []


def hysteretic_work(rows: list[dict[str, float]]) -> float:
    clean = [
        row
        for row in rows
        if math.isfinite(row.get("drift_m", math.nan))
        and math.isfinite(row.get("base_shear_MN", math.nan))
    ]
    work = 0.0
    for a, b in zip(clean, clean[1:]):
        work += 0.5 * (a["base_shear_MN"] + b["base_shear_MN"]) * (
            b["drift_m"] - a["drift_m"]
        )
    return work


def nearest_by_drift(rows: list[dict[str, float]], drift_m: float) -> float:
    clean = [
        row
        for row in rows
        if math.isfinite(row.get("drift_m", math.nan))
        and math.isfinite(row.get("base_shear_MN", math.nan))
    ]
    if not clean:
        return math.nan
    return min(clean, key=lambda row: abs(row["drift_m"] - drift_m))[
        "base_shear_MN"
    ]


def compare_hysteresis(
    reference: list[dict[str, float]],
    candidate: list[dict[str, float]],
) -> dict[str, float]:
    errors: list[float] = []
    ref_values: list[float] = []
    for row in candidate:
        drift = row.get("drift_m", math.nan)
        cand = row.get("base_shear_MN", math.nan)
        if not math.isfinite(drift) or not math.isfinite(cand):
            continue
        ref = nearest_by_drift(reference, drift)
        if not math.isfinite(ref):
            continue
        errors.append(cand - ref)
        ref_values.append(ref)
    if not errors:
        return {"rms_rel": math.nan, "max_rel": math.nan, "work_rel": math.nan}
    ref_scale = max(max(abs(v) for v in ref_values), 1.0e-12)
    rms = math.sqrt(sum(e * e for e in errors) / len(errors))
    max_err = max(abs(e) for e in errors)
    ref_work = hysteretic_work(reference)
    cand_work = hysteretic_work(candidate)
    work_scale = max(abs(ref_work), 1.0e-12)
    return {
        "rms_rel": rms / ref_scale,
        "max_rel": max_err / ref_scale,
        "work_rel": abs(cand_work - ref_work) / work_scale,
    }


def read_reference_rows(args: argparse.Namespace) -> dict[str, list[dict[str, float]]]:
    refs: dict[str, list[dict[str, float]]] = {
        "opensees_hifi": normalized_csv_reference(args.opensees_hifi_hysteresis)
    }
    for n in (4, 6, 8):
        refs[f"lobatto_n{n}"] = normalized_csv_reference(
            args.structural_matrix_dir / f"n{n:02d}_lobatto/hysteresis.csv"
        )
    return refs


def normalized_csv_reference(path: Path) -> list[dict[str, float]]:
    rows = read_csv(path)
    out: list[dict[str, float]] = []
    for row in rows:
        drift_m = scalar(row, "drift_m")
        if not math.isfinite(drift_m):
            drift_mm = scalar(row, "drift_mm")
            drift_m = drift_mm / 1000.0 if math.isfinite(drift_mm) else math.nan
        out.append(
            {
                "drift_m": drift_m,
                "drift_mm": drift_m * 1000.0 if math.isfinite(drift_m) else math.nan,
                "base_shear_MN": scalar(row, "base_shear_MN", "base_shear_mn"),
            }
        )
    return out


def vtk_inventory(case_dir: Path, family: str) -> dict[str, Any]:
    pvd = sorted(case_dir.rglob("*.pvd"))
    vtu = sorted(case_dir.rglob("*.vtu"))
    all_vtk = pvd + vtu
    proxy = bool(all_vtk) and all(
        "vtk_proxy" in str(path).replace("\\", "/") for path in all_vtk
    )
    native = bool(all_vtk) and not proxy
    return {
        "status": "native" if native else ("proxy" if proxy or family == "xfem" else "missing"),
        "pvd_count": len(pvd),
        "vtu_count": len(vtu),
        "pvd_files": [str(path) for path in pvd[:20]],
        "vtu_files_sample": [str(path) for path in vtu[:20]],
    }


def write_native_pvd_collections(case_dir: Path) -> list[str]:
    outputs: list[str] = []
    step_re = re.compile(r"^(?P<prefix>.+)_step_(?P<step>\d+)_(?P<kind>.+)\.vtu$")
    grouped: dict[tuple[Path, str, str], list[tuple[int, Path]]] = {}
    for vtu in case_dir.rglob("*.vtu"):
        if "vtk_proxy" in str(vtu).replace("\\", "/"):
            continue
        match = step_re.match(vtu.name)
        if not match:
            continue
        key = (vtu.parent, match.group("prefix"), match.group("kind"))
        grouped.setdefault(key, []).append((int(match.group("step")), vtu))
    for (folder, prefix, kind), items in grouped.items():
        items.sort(key=lambda item: item[0])
        pvd = folder / f"{prefix}_{kind}.pvd"
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
            "  <Collection>",
        ]
        for step, vtu in items:
            rel = vtu.name
            lines.append(
                f'    <DataSet timestep="{step}" group="" part="0" file="{rel}"/>'
            )
        lines += ["  </Collection>", "</VTKFile>"]
        pvd.write_text("\n".join(lines) + "\n", encoding="utf-8")
        outputs.append(str(pvd))
    return outputs


def write_xfem_proxy_vtk(case_dir: Path, rows: list[dict[str, float]], height_m: float) -> dict[str, Any]:
    if not rows:
        return {"status": "skipped_no_rows"}
    vtk_dir = case_dir / "vtk_proxy"
    vtk_dir.mkdir(parents=True, exist_ok=True)
    pvd_path = vtk_dir / "xfem_observable_axis.pvd"
    pvd_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
        '  <Collection>',
    ]
    for idx, row in enumerate(rows):
        drift = row.get("drift_m", 0.0)
        shear = row.get("base_shear_MN", 0.0)
        step = int(row.get("step", idx) if math.isfinite(row.get("step", math.nan)) else idx)
        name = f"xfem_axis_step_{idx:06d}.vtu"
        pvd_lines.append(
            f'    <DataSet timestep="{row.get("p", idx)}" group="" part="0" file="{name}"/>'
        )
        vtu = vtk_dir / name
        write_axis_vtu(vtu, drift, shear, height_m, step)
    pvd_lines += ["  </Collection>", "</VTKFile>"]
    pvd_path.write_text("\n".join(pvd_lines) + "\n", encoding="utf-8")
    return {
        "status": "proxy",
        "pvd": str(pvd_path),
        "vtu_count": len(rows),
        "proxy_kind": "two-node observable axis with displacement and base_shear",
    }


def write_axis_vtu(path: Path, drift_m: float, base_shear_mn: float, height_m: float, step: int) -> None:
    text = f'''<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="2" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          0 0 0
          0 0 {height_m:.12g}
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int32" Name="connectivity" format="ascii">0 1</DataArray>
        <DataArray type="Int32" Name="offsets" format="ascii">2</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">3</DataArray>
      </Cells>
      <PointData Vectors="displacement">
        <DataArray type="Float64" Name="displacement" NumberOfComponents="3" format="ascii">
          0 0 0
          {drift_m:.12g} 0 0
        </DataArray>
      </PointData>
      <CellData Scalars="base_shear_MN">
        <DataArray type="Float64" Name="base_shear_MN" format="ascii">{base_shear_mn:.12g}</DataArray>
        <DataArray type="Int32" Name="step" format="ascii">{step}</DataArray>
      </CellData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
'''
    path.write_text(text, encoding="utf-8")


def case_completion_from_manifest(case_dir: Path, family: str) -> tuple[bool, str, str]:
    if family == "kobathe":
        manifest = read_json(case_dir / "runtime_manifest.json")
        if manifest:
            return (
                bool(manifest.get("completed_successfully", False)),
                str(manifest.get("status", "unknown")),
                str(manifest.get("termination_reason", "")),
            )
    manifest = read_json(case_dir / "global_xfem_newton_manifest.json")
    if manifest:
        return (
            bool(manifest.get("completed_successfully", False)),
            str(manifest.get("status", "unknown")),
            str(manifest.get("failure_reason", "")),
        )
    return False, "unknown", "manifest missing"


def steel_metric_status(case_dir: Path, family: str) -> str:
    if family == "xfem" and (case_dir / "steel_history.csv").exists():
        return "available"
    if family == "kobathe" and (case_dir / "rebar_history.csv").exists():
        return "available"
    return "unavailable"


def summarize_case(
    args: argparse.Namespace,
    spec: CaseSpec,
    case_dir: Path,
    log_path: Path,
    run_status: str,
    returncode: int | None,
    elapsed: float,
    failure_reason: str,
    refs: dict[str, list[dict[str, float]]],
) -> CaseResult:
    rows = normalized_hysteresis_rows(case_dir, spec.family)
    completed, manifest_status, manifest_failure = case_completion_from_manifest(case_dir, spec.family)
    if run_status == "skipped_existing":
        status = manifest_status if manifest_status != "unknown" else "skipped_existing"
    elif run_status == "ran":
        status = "completed" if completed else manifest_status
    else:
        status = run_status
    if not failure_reason:
        failure_reason = manifest_failure
    if failure_reason == "base_shear_threshold_exceeded":
        status = "aborted"
    peak_drift = max((abs(row["drift_mm"]) for row in rows if math.isfinite(row["drift_mm"])), default=math.nan)
    peak_shear = max((abs(row["base_shear_MN"]) for row in rows if math.isfinite(row["base_shear_MN"])), default=math.nan)
    work = hysteretic_work(rows)
    cmp_open = compare_hysteresis(refs.get("opensees_hifi", []), rows)
    cmp_n4 = compare_hysteresis(refs.get("lobatto_n4", []), rows)
    cmp_n6 = compare_hysteresis(refs.get("lobatto_n6", []), rows)
    cmp_n8 = compare_hysteresis(refs.get("lobatto_n8", []), rows)
    if spec.family == "xfem" and args.xfem_proxy_vtk:
        write_xfem_proxy_vtk(case_dir, rows, args.column_height_m)
    write_native_pvd_collections(case_dir)
    vtk = vtk_inventory(case_dir, spec.family)
    case_manifest = case_dir / "case_manifest.json"
    result = CaseResult(
        case_id=spec.case_id,
        family=spec.family,
        hex_order=spec.hex_order,
        nx=spec.nx,
        ny=spec.ny,
        nz=spec.nz,
        bias_mode=spec.bias_mode,
        status=status,
        returncode=returncode,
        elapsed_seconds=elapsed,
        completed_200mm=completed and peak_drift >= 199.0,
        peak_abs_drift_mm=peak_drift,
        peak_abs_base_shear_mn=peak_shear,
        work_mn_m=work,
        opensees_rms_rel=cmp_open["rms_rel"],
        opensees_max_rel=cmp_open["max_rel"],
        opensees_work_rel=cmp_open["work_rel"],
        lobatto_n4_rms_rel=cmp_n4["rms_rel"],
        lobatto_n6_rms_rel=cmp_n6["rms_rel"],
        lobatto_n8_rms_rel=cmp_n8["rms_rel"],
        steel_metric_status=steel_metric_status(case_dir, spec.family),
        vtk_status=vtk["status"],
        output_dir=str(case_dir),
        log_path=str(log_path),
        manifest_path=str(case_manifest),
        failure_reason=failure_reason,
    )
    write_json(
        case_manifest,
        {
            "schema": "reduced_rc_local_model_promotion_case_v1",
            "case": asdict(spec),
            "result": asdict(result),
            "vtk_inventory": vtk,
            "notes": (
                "XFEM VTK is an observable proxy unless vtk_status is native. "
                "Ko-Bathe native VTK is produced by the continuum benchmark."
            ),
        },
    )
    return result


def score(result: CaseResult) -> float:
    terms = [
        result.opensees_rms_rel,
        result.opensees_work_rel,
        0.2 * result.lobatto_n4_rms_rel,
        0.03 * math.log10(max(result.elapsed_seconds, 1.0)),
    ]
    if not result.completed_200mm:
        return math.inf
    finite = [v for v in terms if math.isfinite(v)]
    return sum(finite) if finite else math.inf


def promote(results: list[CaseResult]) -> dict[str, Any]:
    promotions: dict[str, Any] = {}
    for family in sorted(set(r.family for r in results)):
        candidates = [r for r in results if r.family == family and r.completed_200mm]
        candidates.sort(key=score)
        if candidates:
            best = candidates[0]
            promotions[family] = {
                "status": "promoted_candidate",
                "case_id": best.case_id,
                "score": score(best),
                "rationale": (
                    "Lowest combined OpenSees RMS/work error, Lobatto N=4 "
                    "distance, and runtime penalty among completed cases."
                ),
                "case": asdict(best),
            }
        else:
            promotions[family] = {
                "status": "no_completed_200mm_case",
                "rationale": "No case reached the full 200 mm protocol in this run.",
            }
    return promotions


def save_fig(fig: plt.Figure, path: Path, secondary: Path) -> list[str]:
    outputs = []
    path.parent.mkdir(parents=True, exist_ok=True)
    secondary.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, dpi=250, bbox_inches="tight")
        outputs.append(str(out))
        copy = secondary / out.name
        fig.savefig(copy, dpi=250, bbox_inches="tight")
        outputs.append(str(copy))
    plt.close(fig)
    return outputs


def plot_base_shear_overlays(
    args: argparse.Namespace,
    refs: dict[str, list[dict[str, float]]],
    results: list[CaseResult],
) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for key, label, color, lw in [
        ("opensees_hifi", "OpenSees Hi-Fi", "black", 2.0),
        ("lobatto_n4", "fall_n Lobatto N=4", "#d97706", 1.4),
        ("lobatto_n6", "fall_n Lobatto N=6", "#0b5fa5", 1.2),
        ("lobatto_n8", "fall_n Lobatto N=8", "#2f855a", 1.2),
    ]:
        rows = refs.get(key, [])
        ax.plot(
            [row["drift_m"] * 1000.0 for row in rows],
            [row["base_shear_MN"] * 1000.0 for row in rows],
            label=label,
            color=color,
            lw=lw,
        )
    completed = sorted([r for r in results if r.completed_200mm], key=score)[:8]
    partial = sorted(
        [
            r for r in results
            if not r.completed_200mm
            and math.isfinite(r.peak_abs_drift_mm)
            and r.peak_abs_drift_mm > 0.0
        ],
        key=lambda item: (-item.peak_abs_drift_mm, item.peak_abs_base_shear_mn),
    )[:12]
    candidates = completed + partial
    for result in candidates:
        rows = normalized_hysteresis_rows(Path(result.output_dir), result.family)
        if not rows:
            continue
        label = (
            f"{FAMILY_LABEL[result.family]} {HEX_LABEL[result.hex_order]} "
            f"{result.nx}x{result.ny}x{result.nz} {result.bias_mode}"
        )
        if not result.completed_200mm:
            label += f" ({result.status})"
        ax.plot(
            [row["drift_m"] * 1000.0 for row in rows],
            [row["base_shear_MN"] * 1000.0 for row in rows],
            label=label,
            lw=0.9,
            alpha=0.85,
        )
    ax.set_xlabel("Tip displacement [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Reduced RC local promotion candidates and partial cutoffs")
    if args.abort_base_shear_threshold_kn > 0.0:
        limit = args.abort_base_shear_threshold_kn
        ax.axhline(
            limit,
            color="0.35",
            lw=0.8,
            ls="--",
            label=f"+{limit:g} kN cutoff",
        )
        ax.axhline(
            -limit,
            color="0.35",
            lw=0.8,
            ls=":",
            label=f"-{limit:g} kN cutoff",
        )
        ax.set_ylim(-limit, limit)
    ax.legend(fontsize=7, ncol=2)
    return save_fig(
        fig,
        args.figures_dir / "local_model_promotion_base_shear_overlays.pdf",
        args.secondary_figures_dir,
    )


def plot_ranking(args: argparse.Namespace, results: list[CaseResult]) -> list[str]:
    completed = [r for r in results if r.completed_200mm]
    completed.sort(key=score)
    top = completed[:24]
    if not top:
        return []
    fig, ax = plt.subplots(figsize=(7.5, max(3.2, 0.26 * len(top))))
    labels = [r.case_id.replace("_", " ") for r in top]
    values = [score(r) for r in top]
    colors = ["#0b5fa5" if r.family == "kobathe" else "#d97706" for r in top]
    ax.barh(range(len(top)), values, color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Promotion score, lower is better")
    ax.set_title("Completed 200 mm cases ranked by error and cost")
    return save_fig(
        fig,
        args.figures_dir / "local_model_promotion_ranking.pdf",
        args.secondary_figures_dir,
    )


def write_summary_tables(args: argparse.Namespace, results: list[CaseResult]) -> None:
    fields = list(asdict(results[0]).keys()) if results else list(CaseResult.__annotations__.keys())
    write_csv(args.output_dir / "case_summary.csv", [asdict(r) for r in results], fields)
    promotions = promote(results)
    write_json(args.output_dir / "promotion_summary.json", promotions)


def run_case(
    args: argparse.Namespace,
    spec: CaseSpec,
    refs: dict[str, list[dict[str, float]]],
) -> CaseResult:
    root = repo_root()
    case_dir = args.output_dir / spec.case_id
    log_path = case_dir / "logs/run.log"
    command = build_command(args, spec, case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        case_dir / "case_input.json",
        {
            "schema": "reduced_rc_local_model_promotion_case_input_v1",
            "case": asdict(spec),
            "command": command,
        },
    )

    existing = read_json(case_dir / "case_manifest.json")
    if args.resume and existing.get("result", {}).get("status") in {
        "completed",
        "failed",
        "aborted",
        "timeout",
    }:
        return case_result_from_dict(existing["result"])

    if args.dry_run:
        return summarize_case(
            args,
            spec,
            case_dir,
            log_path,
            "dry_run",
            None,
            0.0,
            "",
            refs,
        )

    tic = time.monotonic()
    run_status, returncode, failure = run_command(
        command,
        log_path,
        root,
        args.timeout_seconds,
    )
    elapsed = time.monotonic() - tic
    return summarize_case(
        args,
        spec,
        case_dir,
        log_path,
        run_status,
        returncode,
        elapsed,
        failure,
        refs,
    )


def write_campaign_manifest(
    args: argparse.Namespace,
    specs: list[CaseSpec],
    results: list[CaseResult],
    figure_paths: list[str],
) -> None:
    payload = {
        "schema": "reduced_rc_local_model_promotion_campaign_v1",
        "campaign": args.campaign,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(args.output_dir),
        "case_count_requested": len(specs),
        "case_count_completed": sum(1 for r in results if r.completed_200mm),
        "case_count_run": len(results),
        "references": {
            "opensees_hifi_hysteresis": str(args.opensees_hifi_hysteresis),
            "structural_matrix_dir": str(args.structural_matrix_dir),
            "structural_promoted": ["lobatto_n4", "lobatto_n6", "lobatto_n8"],
        },
        "policy": {
            "families": parse_csv_list(args.families),
            "hex_orders": parse_csv_list(args.hex_orders),
            "nx_list": parse_csv_list(args.nx_list, int),
            "ny_list": parse_csv_list(args.ny_list, int),
            "nz_list": parse_csv_list(args.nz_list, int),
            "bias_modes": parse_csv_list(args.bias_modes),
            "amplitudes_mm": args.amplitudes_mm,
            "steps_per_segment": args.steps_per_segment,
            "write_vtk": args.write_vtk,
            "vtk_stride": args.vtk_stride,
            "abort_base_shear_threshold_kn": args.abort_base_shear_threshold_kn,
            "kobathe_rebar_interpolation": args.kobathe_rebar_interpolation,
            "kobathe_bond_slip_reference": args.kobathe_bond_slip_reference,
            "kobathe_bond_slip_residual_ratio": args.kobathe_bond_slip_residual_ratio,
            "kobathe_bond_slip_adaptive_reference_max_factor": args.kobathe_bond_slip_adaptive_reference_max_factor,
            "kobathe_bond_slip_adaptive_residual_ratio_floor": args.kobathe_bond_slip_adaptive_residual_ratio_floor,
        },
        "promotions": promote(results),
        "figures": figure_paths,
        "results": [asdict(r) for r in results],
    }
    write_json(args.output_dir / "campaign_manifest.json", payload)


def refresh_campaign_outputs(
    args: argparse.Namespace,
    refs: dict[str, list[dict[str, float]]],
    specs: list[CaseSpec],
    results: list[CaseResult],
) -> list[str]:
    if results:
        write_summary_tables(args, results)
    figures: list[str] = []
    figures += plot_base_shear_overlays(args, refs, results)
    figures += plot_ranking(args, results)
    write_campaign_manifest(args, specs, results, figures)
    return figures


def read_existing_case_results(output_dir: Path) -> list[CaseResult]:
    results: list[CaseResult] = []
    for manifest_path in sorted(output_dir.rglob("case_manifest.json")):
        manifest = read_json(manifest_path)
        payload = manifest.get("result", {})
        if not payload:
            continue
        try:
            results.append(case_result_from_dict(payload))
        except TypeError:
            continue
    return results


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.figures_dir = args.figures_dir.resolve()
    args.secondary_figures_dir = args.secondary_figures_dir.resolve()
    specs = case_specs(args)
    refs = read_reference_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.postprocess_only:
        results = read_existing_case_results(args.output_dir)
        refresh_campaign_outputs(args, refs, specs, results)
        if args.print_progress:
            print(
                f"postprocessed {len(results)} existing case manifests from {args.output_dir}",
                flush=True,
            )
        return 0
    write_json(
        args.output_dir / "campaign_input.json",
        {
            "schema": "reduced_rc_local_model_promotion_campaign_input_v1",
            "argv": sys.argv,
            "specs": [asdict(spec) for spec in specs],
            "abort_base_shear_threshold_kn": args.abort_base_shear_threshold_kn,
        },
    )

    results: list[CaseResult] = []
    for index, spec in enumerate(specs, start=1):
        if args.print_progress:
            print(f"[{index}/{len(specs)}] {spec.case_id}", flush=True)
        result = run_case(args, spec, refs)
        results.append(result)
        if args.print_progress:
            print(
                f"  -> {result.status}, drift={fmt_float(result.peak_abs_drift_mm)} mm, "
                f"score={fmt_float(score(result), '.4g')}",
                flush=True,
            )
        refresh_campaign_outputs(args, refs, specs, results)
        if (
            not args.continue_on_failure
            and result.status in {"failed_process", "timeout"}
            and not args.dry_run
        ):
            break

    refresh_campaign_outputs(args, refs, specs, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
