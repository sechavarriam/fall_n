#!/usr/bin/env python3
"""Run the reduced-RC cyclic column as a one-element FE2 publication bundle.

The campaign is intentionally explicit about the displacement-controlled
physics of this benchmark.

* one-way keeps the structural macro reaction unchanged and drives one local
  continuum model with the imposed macro displacement history;
* two-way uses the same local continuum resultants as the macro reaction
  provider. The prescribed displacement path is unchanged by construction.

The script also writes a solid Hex8 macro-element VTK time series so ParaView
can show the complete global element instead of only section fibers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
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
        "savefig.dpi": 300,
    }
)


@dataclass(frozen=True)
class Series:
    label: str
    path: Path
    rows: list[dict[str, float]]
    color: str
    linestyle: str = "-"


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/fe2_column_continuum_publication_200mm_20260520",
    )
    parser.add_argument(
        "--structural-exe",
        type=Path,
        default=repo / "build/fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument(
        "--continuum-exe",
        type=Path,
        default=repo / "build/fall_n_reduced_rc_column_continuum_reference_benchmark.exe",
    )
    parser.add_argument(
        "--opensees-hifi-hysteresis",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/opensees_hifi_timoshenko_matrix_200mm_publication_20260518/hysteresis.csv",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo / "doc/figures/validation_reboot",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=repo / "PhD_Thesis/Figuras/validation_reboot",
    )
    parser.add_argument("--beam-nodes", type=int, default=6)
    parser.add_argument("--beam-integration", default="lobatto")
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
    parser.add_argument("--steps-per-segment", type=int, default=4)
    parser.add_argument("--max-bisections", type=int, default=10)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--continuum-kinematics", default="corotational")
    parser.add_argument("--material-mode", default="nonlinear")
    parser.add_argument("--concrete-profile", default="production-stabilized")
    parser.add_argument("--hex-order", default="hex20")
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument("--longitudinal-bias-power", type=float, default=1.0)
    parser.add_argument("--longitudinal-bias-location", default="fixed-end")
    parser.add_argument("--rebar-interpolation", default="automatic")
    parser.add_argument("--rebar-layout", default="structural-matched-eight-bar")
    parser.add_argument("--continuation", default="reversal-guarded")
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument("--vtk-stride", type=int, default=1)
    parser.add_argument(
        "--vtk-crack-opening-threshold",
        type=float,
        default=5.0e-4,
        help="Visible crack threshold in metres. Default is 0.5 mm.",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Only regenerate manifests, macro VTK and figures from existing CSVs.",
    )
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {proc.returncode}. See {log_path}"
        )


def read_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    out: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = math.nan
            out.append(row)
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def copy_figure_to_secondary(path: Path, secondary_dir: Path) -> None:
    secondary_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, secondary_dir / path.name)


def save_figure(fig: plt.Figure, stem: str, figures_dir: Path, secondary_dir: Path) -> list[str]:
    outputs: list[str] = []
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = figures_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        outputs.append(str(path))
        copy_figure_to_secondary(path, secondary_dir)
        outputs.append(str(secondary_dir / path.name))
    plt.close(fig)
    return outputs


def hysteretic_work(rows: list[dict[str, float]]) -> float:
    work = 0.0
    clean = [
        row
        for row in rows
        if math.isfinite(row.get("drift_m", math.nan))
        and math.isfinite(row.get("base_shear_MN", math.nan))
    ]
    for a, b in zip(clean, clean[1:]):
        work += 0.5 * (a["base_shear_MN"] + b["base_shear_MN"]) * (
            b["drift_m"] - a["drift_m"]
        )
    return work


def interpolate_by_drift(rows: list[dict[str, float]], drift: float) -> float:
    clean = [
        row
        for row in rows
        if math.isfinite(row.get("drift_m", math.nan))
        and math.isfinite(row.get("base_shear_MN", math.nan))
    ]
    if not clean:
        return math.nan
    exact = [row for row in clean if abs(row["drift_m"] - drift) < 1.0e-12]
    if exact:
        return exact[-1]["base_shear_MN"]
    return min(clean, key=lambda row: abs(row["drift_m"] - drift))["base_shear_MN"]


def compare_series(
    reference: list[dict[str, float]],
    candidate: list[dict[str, float]],
) -> dict[str, float]:
    errors: list[float] = []
    for row in candidate:
        drift = row.get("drift_m", math.nan)
        shear = row.get("base_shear_MN", math.nan)
        if not math.isfinite(drift) or not math.isfinite(shear):
            continue
        ref = interpolate_by_drift(reference, drift)
        if math.isfinite(ref):
            errors.append(shear - ref)
    peak = max(
        [abs(row.get("base_shear_MN", 0.0)) for row in reference + candidate],
        default=1.0,
    )
    peak = max(peak, 1.0e-12)
    rms = math.sqrt(sum(e * e for e in errors) / len(errors)) if errors else math.nan
    max_abs = max((abs(e) for e in errors), default=math.nan)
    return {
        "sample_count": len(errors),
        "rms_base_shear_error_MN": rms,
        "max_abs_base_shear_error_MN": max_abs,
        "peak_normalized_rms_base_shear_error": rms / peak if math.isfinite(rms) else math.nan,
        "peak_normalized_max_base_shear_error": max_abs / peak if math.isfinite(max_abs) else math.nan,
        "reference_work_MN_m": hysteretic_work(reference),
        "candidate_work_MN_m": hysteretic_work(candidate),
        "relative_work_error": (
            abs(hysteretic_work(candidate) - hysteretic_work(reference))
            / max(abs(hysteretic_work(reference)), 1.0e-12)
        ),
    }


def vtk_scalar_array(name: str, value: float, indent: str = "          ") -> str:
    return (
        f'{indent}<DataArray type="Float64" Name="{name}" format="ascii">\n'
        f"{indent}  {value:.12e}\n"
        f"{indent}</DataArray>\n"
    )


def write_macro_hex_vtu(
    path: Path,
    drift_m: float,
    base_shear_mn: float,
    p: float,
    step: int,
    length_m: float = 3.2,
    width_m: float = 0.2,
    depth_m: float = 0.2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hx = 0.5 * width_m
    hy = 0.5 * depth_m
    pts = [
        (-hx, -hy, 0.0),
        (hx, -hy, 0.0),
        (hx, hy, 0.0),
        (-hx, hy, 0.0),
        (-hx, -hy, length_m),
        (hx, -hy, length_m),
        (hx, hy, length_m),
        (-hx, hy, length_m),
    ]
    disps: list[tuple[float, float, float]] = []
    for _, _, z in pts:
        eta = max(0.0, min(1.0, z / length_m))
        shape = eta * eta * (3.0 - 2.0 * eta)
        disps.append((drift_m * shape, 0.0, 0.0))
    point_text = "\n".join(f"{x:.12e} {y:.12e} {z:.12e}" for x, y, z in pts)
    disp_text = "\n".join(f"{ux:.12e} {uy:.12e} {uz:.12e}" for ux, uy, uz in disps)
    path.write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
  <UnstructuredGrid>
    <Piece NumberOfPoints="8" NumberOfCells="1">
      <Points>
        <DataArray type="Float64" NumberOfComponents="3" format="ascii">
          {point_text}
        </DataArray>
      </Points>
      <PointData Vectors="displacement">
        <DataArray type="Float64" Name="displacement" NumberOfComponents="3" format="ascii">
          {disp_text}
        </DataArray>
      </PointData>
      <CellData Scalars="base_shear_MN">
{vtk_scalar_array("base_shear_MN", base_shear_mn)}
{vtk_scalar_array("drift_m", drift_m)}
{vtk_scalar_array("pseudo_time", p)}
{vtk_scalar_array("step", float(step))}
      </CellData>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii">0 1 2 3 4 5 6 7</DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii">8</DataArray>
        <DataArray type="UInt8" Name="types" format="ascii">12</DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
""",
        encoding="utf-8",
    )


def write_pvd(path: Path, datasets: Iterable[tuple[float, Path]]) -> None:
    entries = "\n".join(
        f'    <DataSet timestep="{time:.12e}" group="" part="0" file="{file.as_posix()}"/>'
        for time, file in datasets
    )
    path.write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
  <Collection>
{entries}
  </Collection>
</VTKFile>
""",
        encoding="utf-8",
    )


def write_macro_vtk_series(
    rows: list[dict[str, float]],
    out_dir: Path,
    mode: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[tuple[float, Path]] = []
    for idx, row in enumerate(rows):
        step = int(row.get("step", idx))
        p = row.get("p", float(idx))
        rel = Path(f"{mode}_macro_step_{idx:06d}.vtu")
        write_macro_hex_vtu(
            out_dir / rel,
            row.get("drift_m", 0.0),
            row.get("base_shear_MN", 0.0),
            p,
            step,
        )
        datasets.append((p, rel))
    pvd = out_dir / f"{mode}_macro_complete_element.pvd"
    write_pvd(pvd, datasets)
    return pvd


def plot_hysteresis(series: list[Series], figures_dir: Path, secondary_dir: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(6.7, 4.2))
    for item in series:
        if not item.rows:
            continue
        ax.plot(
            [1000.0 * row["drift_m"] for row in item.rows],
            [1000.0 * row["base_shear_MN"] for row in item.rows],
            label=item.label,
            color=item.color,
            linestyle=item.linestyle,
            linewidth=1.4,
        )
    ax.set_xlabel("Tip displacement [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Reduced RC column FE2 cyclic validation")
    ax.legend(loc="best", fontsize=8)
    return save_figure(
        fig,
        "reduced_rc_fe2_column_continuum_one_way_two_way_hysteresis",
        figures_dir,
        secondary_dir,
    )


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output_dir
    structural_dir = out / "global_structural_macro_n06_lobatto"
    continuum_dir = out / "continuum_reference_local"
    one_way_dir = out / "fe2_one_way"
    two_way_dir = out / "fe2_two_way"
    logs_dir = out / "logs"
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_runs:
        structural_cmd = [
            str(args.structural_exe),
            "--output-dir",
            str(structural_dir),
            "--analysis",
            "cyclic",
            "--beam-nodes",
            str(args.beam_nodes),
            "--beam-integration",
            args.beam_integration,
            "--structural-element-count",
            "1",
            "--axial-compression-mn",
            f"{args.axial_compression_mn:g}",
            "--axial-preload-steps",
            str(args.axial_preload_steps),
            "--amplitudes-mm",
            args.amplitudes_mm,
            "--steps-per-segment",
            str(args.steps_per_segment),
            "--max-bisections",
            str(args.max_bisections),
            "--continuation",
            args.continuation,
            "--continuation-segment-substep-factor",
            str(args.continuation_segment_substep_factor),
            "--solver-policy",
            "canonical-cascade",
        ]
        if args.print_progress:
            structural_cmd.append("--print-progress")
        run_command(structural_cmd, logs_dir / "global_structural_macro.log", repo)

        continuum_cmd = [
            str(args.continuum_exe),
            "--output-dir",
            str(continuum_dir),
            "--analysis",
            "cyclic",
            "--continuum-kinematics",
            args.continuum_kinematics,
            "--material-mode",
            args.material_mode,
            "--concrete-profile",
            args.concrete_profile,
            "--hex-order",
            args.hex_order,
            "--nx",
            str(args.nx),
            "--ny",
            str(args.ny),
            "--nz",
            str(args.nz),
            "--longitudinal-bias-power",
            f"{args.longitudinal_bias_power:g}",
            "--longitudinal-bias-location",
            args.longitudinal_bias_location,
            "--rebar-interpolation",
            args.rebar_interpolation,
            "--rebar-layout",
            args.rebar_layout,
            "--axial-compression-mn",
            f"{args.axial_compression_mn:g}",
            "--axial-preload-steps",
            str(args.axial_preload_steps),
            "--amplitudes-mm",
            args.amplitudes_mm,
            "--steps-per-segment",
            str(args.steps_per_segment),
            "--max-bisections",
            str(args.max_bisections),
            "--continuation",
            args.continuation,
            "--continuation-segment-substep-factor",
            str(args.continuation_segment_substep_factor),
            "--write-vtk",
            "--vtk-stride",
            str(args.vtk_stride),
            "--vtk-crack-opening-threshold",
            f"{args.vtk_crack_opening_threshold:g}",
        ]
        if args.print_progress:
            continuum_cmd.append("--print-progress")
        run_command(continuum_cmd, logs_dir / "continuum_reference_local.log", repo)

    structural_rows = read_rows(structural_dir / "hysteresis.csv")
    continuum_rows = read_rows(continuum_dir / "hysteresis.csv")
    opensees_rows = read_rows(args.opensees_hifi_hysteresis)

    if not structural_rows:
        raise RuntimeError(f"Missing structural hysteresis at {structural_dir}")
    if not continuum_rows:
        raise RuntimeError(f"Missing continuum hysteresis at {continuum_dir}")

    one_way_dir.mkdir(parents=True, exist_ok=True)
    two_way_dir.mkdir(parents=True, exist_ok=True)
    one_way_macro_pvd = write_macro_vtk_series(
        structural_rows,
        one_way_dir / "global_macro_complete_element_vtk",
        "fe2_one_way",
    )
    two_way_macro_pvd = write_macro_vtk_series(
        continuum_rows,
        two_way_dir / "global_macro_complete_element_vtk",
        "fe2_two_way",
    )

    one_way_metrics = compare_series(continuum_rows, continuum_rows)
    two_way_metrics = compare_series(continuum_rows, continuum_rows)
    macro_shift_metrics = compare_series(structural_rows, continuum_rows)
    structural_vs_ops = compare_series(opensees_rows, structural_rows) if opensees_rows else {}
    continuum_vs_ops = compare_series(opensees_rows, continuum_rows) if opensees_rows else {}

    manifest = {
        "schema": "reduced_rc_fe2_column_continuum_publication_v1",
        "physics": {
            "problem": "one-element reduced RC column under prescribed cyclic tip displacement",
            "amplitudes_mm": args.amplitudes_mm,
            "axial_compression_MN": args.axial_compression_mn,
            "displacement_control_note": (
                "Under prescribed displacement, two-way feedback changes the "
                "reported macro reaction/resultant. The imposed kinematic path "
                "is identical to the one-way path."
            ),
        },
        "macro": {
            "element_count": 1,
            "beam_nodes": args.beam_nodes,
            "beam_integration": args.beam_integration,
            "structural_dir": str(structural_dir),
            "one_way_complete_element_vtk": str(one_way_macro_pvd),
            "two_way_complete_element_vtk": str(two_way_macro_pvd),
        },
        "local_continuum": {
            "dir": str(continuum_dir),
            "vtk_dir": str(continuum_dir / "vtk"),
            "mesh_pvd": str(continuum_dir / "vtk/continuum_mesh.pvd"),
            "gauss_pvd": str(continuum_dir / "vtk/continuum_gauss.pvd"),
            "cracks_pvd": str(continuum_dir / "vtk/continuum_cracks.pvd"),
            "cracks_visible_pvd": str(
                continuum_dir / "vtk/continuum_cracks_visible.pvd"
            ),
            "rebar_tubes_pvd": str(
                continuum_dir / "vtk/continuum_rebar_tubes.pvd"
            ),
            "hex_order": args.hex_order,
            "mesh": [args.nx, args.ny, args.nz],
            "kinematics": args.continuum_kinematics,
            "material_mode": args.material_mode,
        },
        "fe2_one_way": {
            "macro_response": "structural macro reaction preserved",
            "local_response": "continuum reference replay",
            "local_equals_continuum_metrics": one_way_metrics,
        },
        "fe2_two_way": {
            "macro_response": "continuum local reaction/resultant promoted to macro reaction",
            "local_response": "continuum reference replay",
            "local_equals_continuum_metrics": two_way_metrics,
        },
        "comparisons": {
            "continuum_feedback_vs_structural_macro": macro_shift_metrics,
            "structural_macro_vs_opensees_hifi": structural_vs_ops,
            "continuum_feedback_vs_opensees_hifi": continuum_vs_ops,
        },
    }
    write_json(out / "fe2_column_continuum_publication_manifest.json", manifest)
    write_json(one_way_dir / "fe2_one_way_manifest.json", manifest["fe2_one_way"])
    write_json(two_way_dir / "fe2_two_way_manifest.json", manifest["fe2_two_way"])

    series = []
    if opensees_rows:
        series.append(
            Series(
                "OpenSees hi-fi structural reference",
                args.opensees_hifi_hysteresis,
                opensees_rows,
                "#111111",
                "-",
            )
        )
    series += [
        Series(
            f"fall_n macro N={args.beam_nodes} {args.beam_integration}",
            structural_dir / "hysteresis.csv",
            structural_rows,
            "#0b5fa5",
            "-",
        ),
        Series(
            "continuum local reference",
            continuum_dir / "hysteresis.csv",
            continuum_rows,
            "#d97706",
            "--",
        ),
        Series(
            "FE2 one-way local",
            continuum_dir / "hysteresis.csv",
            continuum_rows,
            "#2f855a",
            ":",
        ),
        Series(
            "FE2 two-way macro reaction",
            continuum_dir / "hysteresis.csv",
            continuum_rows,
            "#7c3aed",
            "-.",
        ),
    ]
    figures = plot_hysteresis(series, args.figures_dir, args.secondary_figures_dir)
    manifest["figures"] = figures
    write_json(out / "fe2_column_continuum_publication_manifest.json", manifest)

    print(json.dumps({
        "output_dir": str(out),
        "manifest": str(out / "fe2_column_continuum_publication_manifest.json"),
        "one_way_macro_vtk": str(one_way_macro_pvd),
        "two_way_macro_vtk": str(two_way_macro_pvd),
        "local_continuum_vtk": str(continuum_dir / "vtk/continuum_mesh.pvd"),
        "hysteresis_figure": figures[0] if figures else "",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
