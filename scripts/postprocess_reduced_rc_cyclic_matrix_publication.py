#!/usr/bin/env python3
"""Publication postprocess for the reduced RC cyclic Timoshenko matrix.

For every fall_n matrix bundle this script writes:

* one VTK point-cloud file per accepted state;
* base-shear hysteresis;
* moment-curvature loop at the base station;
* stress-strain loops for representative extreme steel, cover concrete, and
  confined-core concrete fibers.

The same plotting functions also handle the OpenSees hi-fi structural
reference when its section-fiber history is available. OpenSees comparison
figures are promoted only for base shear by default; section and fiber overlays
need an explicit diagnostic flag because they require station, fiber, and commit
history identity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROLE_LABELS = {
    "reinforcing_steel": "extreme steel fiber",
    "unconfined_concrete": "extreme unconfined concrete fiber",
    "confined_concrete": "extreme confined concrete fiber",
}

ROLE_IDS = {
    "unconfined_concrete": 1,
    "confined_concrete": 2,
    "reinforcing_steel": 3,
}

CURVE_SPECS = {
    "base_shear": {
        "title": "base shear hysteresis",
        "x_label": "Tip displacement [mm]",
        "y_label": "Base shear [kN]",
    },
    "moment_curvature": {
        "title": "base moment-curvature",
        "x_label": r"Base curvature $\kappa_y$ [1/m]",
        "y_label": r"Base moment $M_y$ [kN m]",
    },
    "reinforcing_steel": {
        "title": "extreme steel fiber",
        "x_label": "Fiber strain",
        "y_label": "Fiber stress [MPa]",
    },
    "unconfined_concrete": {
        "title": "extreme unconfined concrete fiber",
        "x_label": "Fiber strain",
        "y_label": "Fiber stress [MPa]",
    },
    "confined_concrete": {
        "title": "extreme confined concrete fiber",
        "x_label": "Fiber strain",
        "y_label": "Fiber stress [MPa]",
    },
}

ZONE_IDS = {
    "cover_top": 1,
    "cover_bottom": 2,
    "cover_left": 3,
    "cover_right": 4,
    "confined_core": 5,
    "longitudinal_steel": 6,
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        required=True,
        help="Root with fall_n_matrix/* bundles and opensees_hifi_reference.",
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
    parser.add_argument(
        "--vtk-stride",
        type=int,
        default=1,
        help="Write every Nth accepted state. Default writes every accepted state.",
    )
    parser.add_argument(
        "--max-vtk-states",
        type=int,
        default=0,
        help="Optional cap per bundle. Zero means no cap.",
    )
    parser.add_argument("--only-bundle", default="", help="Substring filter for bundle names.")
    parser.add_argument(
        "--include-section-fiber-opensees-comparisons",
        action="store_true",
        help=(
            "Also plot moment-curvature and fiber curves against OpenSees. "
            "By default only base-shear comparisons are promoted because "
            "section/fiber curves require audited station and fiber identity."
        ),
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for raw in csv.DictReader(f):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
        return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_figure(fig: plt.Figure, path: Path, secondary_dir: Path | None = None) -> list[str]:
    outputs = []
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, dpi=250, bbox_inches="tight")
        outputs.append(str(out))
        if secondary_dir is not None:
            secondary_dir.mkdir(parents=True, exist_ok=True)
            secondary = secondary_dir / out.name
            fig.savefig(secondary, dpi=250, bbox_inches="tight")
            outputs.append(str(secondary))
    plt.close(fig)
    return outputs


def normalized_column(row: dict[str, Any], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return math.nan


def base_station_id(rows: list[dict[str, Any]]) -> int:
    candidates: dict[int, float] = {}
    for row in rows:
        gp = int(row.get("section_gp", 0))
        xi = normalized_column(row, "xi")
        z_m = normalized_column(row, "z_m")
        coordinate = z_m if math.isfinite(z_m) else xi
        if gp not in candidates or coordinate < candidates[gp]:
            candidates[gp] = coordinate
    if not candidates:
        return 0
    return min(candidates.items(), key=lambda item: item[1])[0]


def base_station_rows(rows: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    station = base_station_id(rows)
    filtered = [
        row for row in rows
        if int(row.get("section_gp", 0)) == station
    ]
    return station, sorted(filtered, key=lambda row: (float(row.get("p", 0.0)), int(row.get("step", 0))))


def representative_fiber_ids(rows: list[dict[str, Any]], section_gp: int) -> dict[str, int]:
    first_step_rows = [
        row for row in rows
        if int(row.get("section_gp", 0)) == section_gp
        and int(row.get("step", 0)) == int(rows[0].get("step", 0))
    ]
    if not first_step_rows:
        first_step_rows = [row for row in rows if int(row.get("section_gp", 0)) == section_gp]
    out: dict[str, int] = {}
    for role in ROLE_LABELS:
        scoped = [row for row in first_step_rows if str(row.get("material_role", "")) == role]
        if not scoped:
            continue
        out[role] = int(max(scoped, key=lambda row: abs(float(row.get("z", 0.0))))["fiber_index"])
    return out


def fiber_curve_rows(
    rows: list[dict[str, Any]],
    section_gp: int,
    fiber_id: int,
) -> list[dict[str, Any]]:
    return sorted(
        [
            row for row in rows
            if int(row.get("section_gp", -1)) == section_gp
            and int(row.get("fiber_index", -2)) == fiber_id
        ],
        key=lambda row: (float(row.get("p", 0.0)), int(row.get("step", 0))),
    )


def write_ascii_vtu(path: Path, rows: list[dict[str, Any]], height_m: float = 3.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = []
    displacement = []
    stress = []
    strain = []
    tangent = []
    role_id = []
    zone_id = []
    fiber_index = []
    section_gp = []
    curvature = []
    drift = normalized_column(rows[0], "drift_m", "actual_tip_drift_m")
    top_axial = 0.0
    for row in rows:
        xi = normalized_column(row, "xi")
        z_axis = normalized_column(row, "z_m")
        if not math.isfinite(z_axis):
            z_axis = 0.5 * (xi + 1.0) * height_m if math.isfinite(xi) else 0.0
        eta = max(0.0, min(1.0, z_axis / height_m if height_m > 0.0 else 0.0))
        shape = eta * eta * (3.0 - 2.0 * eta)
        y = float(row.get("y", 0.0))
        z = float(row.get("z", 0.0))
        points.append((z, y, z_axis))
        displacement.append((drift * shape, 0.0, top_axial * eta))
        stress.append(normalized_column(row, "stress_xx_MPa"))
        strain.append(normalized_column(row, "strain_xx"))
        tangent.append(normalized_column(row, "tangent_xx_MPa"))
        role_id.append(ROLE_IDS.get(str(row.get("material_role", "")), 0))
        zone_id.append(ZONE_IDS.get(str(row.get("zone", "")), 0))
        fiber_index.append(int(row.get("fiber_index", -1)))
        section_gp.append(int(row.get("section_gp", -1)))
        curvature.append(normalized_column(row, "curvature_y"))

    def data_array(name: str, values: list[Any], components: int = 1) -> str:
        text = []
        for value in values:
            if components == 1:
                text.append(str(value))
            else:
                text.append(" ".join(str(v) for v in value))
        return (
            f'        <DataArray type="Float64" Name="{name}" '
            f'NumberOfComponents="{components}" format="ascii">\n'
            f"          {' '.join(text)}\n"
            "        </DataArray>\n"
        )

    n = len(points)
    connectivity = " ".join(str(i) for i in range(n))
    offsets = " ".join(str(i + 1) for i in range(n))
    types = " ".join("1" for _ in range(n))
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n'
        "  <UnstructuredGrid>\n"
        f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n'
        "      <Points>\n"
        + data_array("Points", points, 3)
        + "      </Points>\n"
        '      <PointData Vectors="displacement" Scalars="stress_xx_MPa">\n'
        + data_array("displacement", displacement, 3)
        + data_array("strain_xx", strain)
        + data_array("stress_xx_MPa", stress)
        + data_array("tangent_xx_MPa", tangent)
        + data_array("curvature_y", curvature)
        + data_array("material_role_id", role_id)
        + data_array("zone_id", zone_id)
        + data_array("fiber_index", fiber_index)
        + data_array("section_gp", section_gp)
        + "      </PointData>\n"
        "      <Cells>\n"
        f'        <DataArray type="Int64" Name="connectivity" format="ascii">{connectivity}</DataArray>\n'
        f'        <DataArray type="Int64" Name="offsets" format="ascii">{offsets}</DataArray>\n'
        f'        <DataArray type="UInt8" Name="types" format="ascii">{types}</DataArray>\n'
        "      </Cells>\n"
        "    </Piece>\n"
        "  </UnstructuredGrid>\n"
        "</VTKFile>\n",
        encoding="utf-8",
    )


def group_by_step(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row.get("step", 0))].append(row)
    return dict(sorted(groups.items()))


def write_vtk_sequence(bundle: Path, rows: list[dict[str, Any]], stride: int, cap: int) -> dict[str, Any]:
    groups = group_by_step(rows)
    vtk_dir = bundle / "vtk_fiber_evolution"
    index_rows = []
    written = 0
    for seq, (step, step_rows) in enumerate(groups.items()):
        if seq % max(stride, 1) != 0:
            continue
        if cap > 0 and written >= cap:
            break
        path = vtk_dir / f"fiber_state_step_{step:06d}.vtu"
        write_ascii_vtu(path, step_rows)
        index_rows.append({
            "step": step,
            "path": str(path),
            "point_count": len(step_rows),
        })
        written += 1
    if index_rows:
        with (vtk_dir / "fiber_state_time_index.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=("step", "path", "point_count"))
            writer.writeheader()
            writer.writerows(index_rows)
    return {
        "vtk_dir": str(vtk_dir),
        "vtk_file_count": written,
        "vtk_index": str(vtk_dir / "fiber_state_time_index.csv") if index_rows else "",
    }


def curve_payload(
    hys: list[dict[str, Any]],
    mom: list[dict[str, Any]],
    fibers: list[dict[str, Any]],
    section_gp: int,
    fiber_ids: dict[str, int],
) -> dict[str, dict[str, list[float]]]:
    _, mom_base = base_station_rows(mom)
    payload: dict[str, dict[str, list[float]]] = {
        "base_shear": {
            "p": [float(row.get("p", 0.0)) for row in hys],
            "x": [1000.0 * float(row["drift_m"]) for row in hys],
            "y": [1000.0 * float(row["base_shear_MN"]) for row in hys],
        },
        "moment_curvature": {
            "p": [float(row.get("p", 0.0)) for row in mom_base],
            "x": [float(row["curvature_y"]) for row in mom_base],
            "y": [1000.0 * float(row["moment_y_MNm"]) for row in mom_base],
        },
    }
    for role in ROLE_LABELS:
        fid = fiber_ids.get(role)
        if fid is None:
            continue
        rows = fiber_curve_rows(fibers, section_gp, fid)
        if not rows:
            continue
        payload[role] = {
            "p": [float(row.get("p", 0.0)) for row in rows],
            "x": [float(row["strain_xx"]) for row in rows],
            "y": [float(row["stress_xx_MPa"]) for row in rows],
        }
    return payload


def unique_curve_points(curve: dict[str, list[float]]) -> dict[str, list[float]]:
    seen: dict[float, tuple[float, float]] = {}
    for p, x, y in zip(curve["p"], curve["x"], curve["y"]):
        seen[float(p)] = (float(x), float(y))
    ps = sorted(seen)
    return {
        "p": ps,
        "x": [seen[p][0] for p in ps],
        "y": [seen[p][1] for p in ps],
    }


def interp_series(px: list[float], values: list[float], targets: list[float]) -> list[float]:
    if not px:
        return [math.nan for _ in targets]
    pairs = sorted((float(p), float(v)) for p, v in zip(px, values) if math.isfinite(float(p)) and math.isfinite(float(v)))
    if not pairs:
        return [math.nan for _ in targets]
    xs = [p for p, _ in pairs]
    ys = [v for _, v in pairs]
    out = []
    j = 0
    for target in targets:
        t = min(max(float(target), xs[0]), xs[-1])
        while j + 1 < len(xs) and xs[j + 1] < t:
            j += 1
        if j + 1 >= len(xs):
            out.append(ys[-1])
            continue
        x0, x1 = xs[j], xs[j + 1]
        y0, y1 = ys[j], ys[j + 1]
        if abs(x1 - x0) < 1.0e-14:
            out.append(y1)
        else:
            a = (t - x0) / (x1 - x0)
            out.append((1.0 - a) * y0 + a * y1)
    return out


def curve_metrics(
    case_name: str,
    curve_name: str,
    falln: dict[str, list[float]],
    reference: dict[str, list[float]],
) -> dict[str, Any]:
    f = unique_curve_points(falln)
    r = unique_curve_points(reference)
    p_common = f["p"]
    ref_x = interp_series(r["p"], r["x"], p_common)
    ref_y = interp_series(r["p"], r["y"], p_common)
    dx = [fx - rx for fx, rx in zip(f["x"], ref_x) if math.isfinite(fx) and math.isfinite(rx)]
    dy = [fy - ry for fy, ry in zip(f["y"], ref_y) if math.isfinite(fy) and math.isfinite(ry)]
    ref_y_abs = [abs(v) for v in ref_y if math.isfinite(v)]
    ref_x_abs = [abs(v) for v in ref_x if math.isfinite(v)]
    y_scale = max(ref_y_abs) if ref_y_abs else math.nan
    x_scale = max(ref_x_abs) if ref_x_abs else math.nan
    rms_y = math.sqrt(sum(v * v for v in dy) / len(dy)) if dy else math.nan
    rms_x = math.sqrt(sum(v * v for v in dx) / len(dx)) if dx else math.nan
    max_y = max((abs(v) for v in dy), default=math.nan)
    max_x = max((abs(v) for v in dx), default=math.nan)
    area_f = hysteretic_area(f["x"], f["y"])
    area_r = hysteretic_area(r["x"], r["y"])
    return {
        "case": case_name,
        "curve": curve_name,
        "point_count": len(p_common),
        "rms_y": rms_y,
        "max_abs_y": max_y,
        "rms_y_normalized": rms_y / y_scale if y_scale and math.isfinite(y_scale) else math.nan,
        "max_abs_y_normalized": max_y / y_scale if y_scale and math.isfinite(y_scale) else math.nan,
        "rms_x": rms_x,
        "max_abs_x": max_x,
        "rms_x_normalized": rms_x / x_scale if x_scale and math.isfinite(x_scale) else math.nan,
        "max_abs_x_normalized": max_x / x_scale if x_scale and math.isfinite(x_scale) else math.nan,
        "falln_loop_area": area_f,
        "opensees_loop_area": area_r,
        "loop_area_rel_error": (area_f - area_r) / area_r if abs(area_r) > 1.0e-14 else math.nan,
    }


def hysteretic_area(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return math.nan
    total = 0.0
    for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        total += 0.5 * (y0 + y1) * (x1 - x0)
    return abs(total)


def save_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison_curves(
    case_name: str,
    falln_curves: dict[str, dict[str, list[float]]],
    ref_curves: dict[str, dict[str, list[float]]],
    figures_dir: Path,
    secondary_dir: Path | None,
    include_section_fiber: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    outputs: list[str] = []
    metrics: list[dict[str, Any]] = []
    fig_dir = figures_dir / "reduced_rc_cyclic_matrix_comparisons" / case_name
    thesis_dir = secondary_dir / "reduced_rc_cyclic_matrix_comparisons" / case_name if secondary_dir else None
    for curve_name, spec in CURVE_SPECS.items():
        if curve_name != "base_shear" and not include_section_fiber:
            continue
        if curve_name not in falln_curves or curve_name not in ref_curves:
            continue
        falln = falln_curves[curve_name]
        ref = ref_curves[curve_name]
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ax.plot(ref["x"], ref["y"], color="black", lw=1.5, label="OpenSees hi-fi")
        ax.plot(falln["x"], falln["y"], color="#0b5fa5", lw=1.2, label=f"fall_n {case_name}")
        ax.set_xlabel(spec["x_label"])
        ax.set_ylabel(spec["y_label"])
        ax.set_title(f"{case_name} vs OpenSees: {spec['title']}")
        ax.legend(frameon=True)
        outputs.extend(save_figure(fig, fig_dir / f"{case_name}_{curve_name}_vs_opensees.pdf", thesis_dir))
        metrics.append(curve_metrics(case_name, curve_name, falln, ref))
    save_metrics_csv(fig_dir / f"{case_name}_curve_metrics.csv", metrics)
    if thesis_dir:
        save_metrics_csv(thesis_dir / f"{case_name}_curve_metrics.csv", metrics)
    return outputs, metrics


def bundle_is_promoted_complete(bundle: Path) -> bool:
    manifest = bundle / "runtime_manifest.json"
    hys = bundle / "hysteresis.csv"
    if not manifest.exists() or not hys.exists():
        return False
    try:
        status = json.loads(manifest.read_text(encoding="utf-8")).get("status")
    except json.JSONDecodeError:
        return False
    if status != "completed":
        return False
    rows = read_csv(hys)
    if not rows:
        return False
    max_drift_mm = max(abs(1000.0 * float(row["drift_m"])) for row in rows)
    return max_drift_mm >= 199.0


def plot_bundle(bundle: Path, figures_dir: Path, secondary_dir: Path | None) -> dict[str, Any]:
    loaded = load_bundle_data(bundle)
    artifacts: dict[str, Any] = {
        "bundle": str(bundle),
        "status": "missing",
    }
    if loaded is None:
        return artifacts
    hys, mom, fibers, hysteresis_path, moment_path, fiber_path = loaded
    section_gp = base_station_id(fibers)
    moment_section_gp, mom_base = base_station_rows(mom)
    fiber_ids = representative_fiber_ids(fibers, section_gp)
    stem = bundle.name
    bundle_fig_dir = figures_dir / "reduced_rc_cyclic_matrix" / stem
    thesis_fig_dir = secondary_dir / "reduced_rc_cyclic_matrix" / stem if secondary_dir else None

    outputs = []
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    ax.plot([1000.0 * float(r["drift_m"]) for r in hys], [1000.0 * float(r["base_shear_MN"]) for r in hys])
    ax.set_xlabel("Tip displacement [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title(f"{stem} base-shear hysteresis")
    outputs.extend(save_figure(fig, bundle_fig_dir / f"{stem}_base_shear_hysteresis.pdf", thesis_fig_dir))

    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    ax.plot([float(r["curvature_y"]) for r in mom_base], [1000.0 * float(r["moment_y_MNm"]) for r in mom_base])
    ax.set_xlabel("Base curvature $\\kappa_y$ [1/m]")
    ax.set_ylabel("Base moment $M_y$ [kN m]")
    ax.set_title(f"{stem} base moment-curvature")
    outputs.extend(save_figure(fig, bundle_fig_dir / f"{stem}_moment_curvature.pdf", thesis_fig_dir))

    for role, label in ROLE_LABELS.items():
        if role not in fiber_ids:
            continue
        rows = fiber_curve_rows(fibers, section_gp, fiber_ids[role])
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(5.8, 4.1))
        ax.plot([float(r["strain_xx"]) for r in rows], [float(r["stress_xx_MPa"]) for r in rows])
        ax.set_xlabel("Fiber strain")
        ax.set_ylabel("Fiber stress [MPa]")
        ax.set_title(f"{stem} {label}")
        outputs.extend(save_figure(fig, bundle_fig_dir / f"{stem}_{role}_stress_strain.pdf", thesis_fig_dir))

    vtk = write_vtk_sequence(bundle, fibers, stride=1, cap=0)
    artifacts.update({
        "status": "completed",
        "hysteresis_csv": str(hysteresis_path),
        "moment_curvature_csv": str(moment_path),
        "fiber_history_csv": str(fiber_path),
        "base_section_gp": section_gp,
        "base_moment_section_gp": moment_section_gp,
        "representative_fiber_ids": fiber_ids,
        "figures": outputs,
        **vtk,
    })
    return artifacts


def load_bundle_data(
    bundle: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Path,
    Path,
    Path,
] | None:
    hysteresis_path = bundle / "comparison_hysteresis.csv"
    moment_path = bundle / "comparison_moment_curvature_base.csv"
    fiber_path = bundle / "comparison_section_fiber_state_history.csv"
    if not fiber_path.exists():
        fiber_path = bundle / "section_fiber_state_history.csv"
    if not moment_path.exists():
        moment_path = bundle / "moment_curvature_base.csv"
    if not moment_path.exists():
        moment_path = bundle / "section_response.csv"
    if not hysteresis_path.exists():
        hysteresis_path = bundle / "hysteresis.csv"

    if not hysteresis_path.exists() or not moment_path.exists() or not fiber_path.exists():
        return None

    hys = read_csv(hysteresis_path)
    mom = read_csv(moment_path)
    fibers = read_csv(fiber_path)
    return hys, mom, fibers, hysteresis_path, moment_path, fiber_path


def load_bundle_curves(bundle: Path) -> dict[str, dict[str, list[float]]] | None:
    loaded = load_bundle_data(bundle)
    if loaded is None:
        return None
    hys, mom, fibers, *_ = loaded
    section_gp = base_station_id(fibers)
    fiber_ids = representative_fiber_ids(fibers, section_gp)
    return curve_payload(hys, mom, fibers, section_gp, fiber_ids)


def main() -> int:
    args = parse_args()
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    matrix_dir = args.matrix_dir.resolve()
    bundles = sorted((matrix_dir / "fall_n_matrix").glob("*"))
    if args.only_bundle:
        bundles = [b for b in bundles if args.only_bundle in b.name]
    summary = {
        "schema": "reduced_rc_cyclic_matrix_publication_postprocess_v1",
        "matrix_dir": str(matrix_dir),
        "bundle_count": len(bundles),
        "bundles": [],
    }
    for bundle in bundles:
        item = plot_bundle(bundle, args.figures_dir, args.secondary_figures_dir)
        summary["bundles"].append(item)
    opensees = matrix_dir / "opensees_hifi_reference"
    if opensees.exists():
        summary["opensees_hifi_reference"] = plot_bundle(opensees, args.figures_dir, args.secondary_figures_dir)
        reference_curves = load_bundle_curves(opensees)
        comparison_metrics: list[dict[str, Any]] = []
        comparison_figures: list[str] = []
        if reference_curves is not None:
            promoted = [bundle for bundle in bundles if bundle_is_promoted_complete(bundle)]
            for bundle in promoted:
                falln_curves = load_bundle_curves(bundle)
                if falln_curves is None:
                    continue
                figures, metrics = plot_comparison_curves(
                    bundle.name,
                    falln_curves,
                    reference_curves,
                    args.figures_dir,
                    args.secondary_figures_dir,
                    args.include_section_fiber_opensees_comparisons,
                )
                comparison_figures.extend(figures)
                comparison_metrics.extend(metrics)
            summary["opensees_comparison"] = {
                "promoted_case_count": len(promoted),
                "promoted_cases": [bundle.name for bundle in promoted],
                "metric_count": len(comparison_metrics),
                "figures": comparison_figures,
            }
            metrics_path = (
                args.figures_dir
                / "reduced_rc_cyclic_matrix_comparisons"
                / "promoted_curve_metrics.csv"
            )
            save_metrics_csv(metrics_path, comparison_metrics)
            if args.secondary_figures_dir:
                save_metrics_csv(
                    args.secondary_figures_dir
                    / "reduced_rc_cyclic_matrix_comparisons"
                    / "promoted_curve_metrics.csv",
                    comparison_metrics,
                )
            write_json(
                args.figures_dir
                / "reduced_rc_cyclic_matrix_comparisons"
                / "promoted_curve_metrics.json",
                {"metrics": comparison_metrics},
            )
    out = matrix_dir / "cyclic_matrix_publication_postprocess_summary.json"
    write_json(out, summary)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.figures_dir / "reduced_rc_cyclic_matrix_publication_postprocess_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
