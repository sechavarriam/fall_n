#!/usr/bin/env python3
"""Extreme steel-fiber hysteresis selection plots for the reduced RC matrix.

This postprocess mirrors the moment-curvature selection audit, but uses the
stress-strain loop of the extreme longitudinal steel fiber at the base station.
Rows whose base-section axial resultant violates the imposed axial preload
contract are excluded from the steel selection metrics and written to an
outlier CSV.  Raw campaign CSV files are left untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


QUAD_LABEL = {
    "legendre": "Gauss-Legendre",
    "lobatto": "Gauss-Lobatto",
    "radau-left": "Gauss-Radau left",
    "radau-right": "Gauss-Radau right",
}

QUAD_COLOR = {
    "legendre": "#1f77b4",
    "lobatto": "#d97706",
    "radau-left": "#2e8b57",
    "radau-right": "#7c3aed",
}

DEFAULT_EXPECTED_AXIAL_MN = 0.02
AXIAL_CONTRACT_TOLERANCE_MN = 0.08


@dataclass(frozen=True)
class SteelMetric:
    beam_nodes: int
    beam_integration: str
    status: str
    bundle_dir: str
    wall_seconds: float
    hysteresis_rms_rel: float
    work_rel_error: float
    steel_stress_rms_rel: float
    steel_strain_rms_rel: float
    steel_loop_work_rel_error: float
    steel_outlier_count: int
    steel_section_gp: int
    steel_fiber_index: int
    steel_fiber_y: float
    steel_fiber_z: float
    steel_score: float
    production_score: float


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
        "timoshenko_matrix_reproduced_historical_closure_20260520",
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
    parser.add_argument("--min-n", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=10)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows: list[dict[str, Any]] = []
        for raw in csv.DictReader(f):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = value
            rows.append(row)
        return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def finite(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def expected_axial_mn(bundle_dir: Path) -> float:
    for name in ("runtime_manifest.json", "reference_manifest.json", "manifest.json"):
        payload = read_json(bundle_dir / name)
        for key in ("axial_compression_mn", "target_axial_force_mn", "axial_force_mn"):
            value = finite(payload.get(key))
            if math.isfinite(value) and abs(value) > 0.0:
                return abs(value)
    return DEFAULT_EXPECTED_AXIAL_MN


def moment_curvature_path(path: Path) -> Path:
    if (path / "comparison_moment_curvature_base.csv").exists():
        return path / "comparison_moment_curvature_base.csv"
    if (path / "moment_curvature_base.csv").exists():
        return path / "moment_curvature_base.csv"
    return path / "section_response.csv"


def fiber_history_path(path: Path) -> Path:
    if (path / "comparison_section_fiber_state_history.csv").exists():
        return path / "comparison_section_fiber_state_history.csv"
    return path / "section_fiber_state_history.csv"


def base_station_id(rows: list[dict[str, Any]]) -> int:
    candidates: dict[int, float] = {}
    for row in rows:
        gp = int(finite(row.get("section_gp"), 0))
        xi = finite(row.get("xi"))
        z_m = finite(row.get("z_m"))
        coordinate = z_m if math.isfinite(z_m) else xi
        if gp not in candidates or coordinate < candidates[gp]:
            candidates[gp] = coordinate
    if not candidates:
        return 0
    return min(candidates.items(), key=lambda item: item[1])[0]


def base_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gp = base_station_id(rows)
    return [row for row in rows if int(finite(row.get("section_gp"), 0)) == gp]


def axial_rejection_by_step(bundle_dir: Path, case_name: str) -> tuple[set[int], list[dict[str, Any]]]:
    expected = expected_axial_mn(bundle_dir)
    rejected_steps: set[int] = set()
    rejected_rows: list[dict[str, Any]] = []
    for row in base_rows(read_csv(moment_curvature_path(bundle_dir))):
        axial = finite(row.get("axial_force_MN"))
        step = int(finite(row.get("step"), -1))
        if math.isfinite(axial) and abs(abs(axial) - expected) > AXIAL_CONTRACT_TOLERANCE_MN:
            rejected_steps.add(step)
            rejected_rows.append(
                {
                    "case": case_name,
                    "step": step,
                    "p": finite(row.get("p")),
                    "section_gp": row.get("section_gp", ""),
                    "xi": row.get("xi", ""),
                    "curvature_y": finite(row.get("curvature_y")),
                    "moment_y_MNm": finite(row.get("moment_y_MNm")),
                    "axial_force_MN": axial,
                    "expected_abs_axial_MN": expected,
                    "axial_tolerance_MN": AXIAL_CONTRACT_TOLERANCE_MN,
                    "reason": "axial_force_out_of_contract",
                }
            )
    return rejected_steps, rejected_rows


def choose_extreme_steel_fiber(rows: list[dict[str, Any]], section_gp: int) -> dict[str, Any]:
    first_step = min(
        (int(finite(row.get("step"), 0)) for row in rows if int(finite(row.get("section_gp"), -1)) == section_gp),
        default=0,
    )
    candidates = [
        row
        for row in rows
        if int(finite(row.get("section_gp"), -1)) == section_gp
        and int(finite(row.get("step"), -1)) == first_step
        and str(row.get("material_role", "")) == "reinforcing_steel"
    ]
    if not candidates:
        return {}
    return max(
        candidates,
        key=lambda row: (
            abs(finite(row.get("z"))),
            abs(finite(row.get("y"))),
            -finite(row.get("fiber_index"), 1.0e12),
        ),
    )


def steel_curve(
    bundle_dir: Path,
    case_name: str,
) -> tuple[list[dict[str, float]], dict[str, Any], list[dict[str, Any]]]:
    fibers = read_csv(fiber_history_path(bundle_dir))
    section_gp = base_station_id(fibers)
    selected = choose_extreme_steel_fiber(fibers, section_gp)
    if not selected:
        return [], {}, []
    fiber_index = int(finite(selected.get("fiber_index"), -1))
    rejected_steps, rejected_rows = axial_rejection_by_step(bundle_dir, case_name)
    curve: list[dict[str, float]] = []
    for row in fibers:
        if int(finite(row.get("section_gp"), -1)) != section_gp:
            continue
        if int(finite(row.get("fiber_index"), -2)) != fiber_index:
            continue
        step = int(finite(row.get("step"), -1))
        if step in rejected_steps:
            continue
        p = finite(row.get("p"))
        strain = finite(row.get("strain_xx"))
        stress = finite(row.get("stress_xx_MPa"))
        if not (math.isfinite(p) and math.isfinite(strain) and math.isfinite(stress)):
            rejected_rows.append(
                {
                    "case": case_name,
                    "step": step,
                    "p": p,
                    "section_gp": section_gp,
                    "fiber_index": fiber_index,
                    "strain_xx": strain,
                    "stress_xx_MPa": stress,
                    "reason": "nonfinite_steel_fiber_row",
                }
            )
            continue
        curve.append({"p": p, "strain": strain, "stress": stress})
    info = {
        "section_gp": section_gp,
        "fiber_index": fiber_index,
        "y": finite(selected.get("y")),
        "z": finite(selected.get("z")),
        "zone": selected.get("zone", ""),
    }
    return sorted(curve, key=lambda row: (row["p"], row["strain"])), info, rejected_rows


def unique_by_p(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    seen: dict[float, dict[str, float]] = {}
    for row in rows:
        seen[row["p"]] = row
    return [seen[p] for p in sorted(seen)]


def interpolate(ref: list[dict[str, float]], query_p: float, field: str) -> float:
    if not ref:
        return math.nan
    if query_p <= ref[0]["p"]:
        return ref[0][field]
    if query_p >= ref[-1]["p"]:
        return ref[-1][field]
    lo = 0
    hi = len(ref) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ref[mid]["p"] <= query_p:
            lo = mid
        else:
            hi = mid
    p0 = ref[lo]["p"]
    p1 = ref[hi]["p"]
    if abs(p1 - p0) <= 1.0e-15:
        return ref[lo][field]
    a = (query_p - p0) / (p1 - p0)
    return (1.0 - a) * ref[lo][field] + a * ref[hi][field]


def rms_rel_at_p(lhs: list[dict[str, float]], ref: list[dict[str, float]], field: str) -> float:
    scale = max((abs(row[field]) for row in ref if math.isfinite(row[field])), default=math.nan)
    if not math.isfinite(scale) or scale <= 1.0e-15:
        return math.nan
    errors = []
    for row in lhs:
        rv = interpolate(ref, row["p"], field)
        if math.isfinite(rv) and math.isfinite(row[field]):
            errors.append((row[field] - rv) / scale)
    if not errors:
        return math.nan
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def loop_area(rows: list[dict[str, float]]) -> float:
    if len(rows) < 2:
        return math.nan
    total = 0.0
    for a, b in zip(rows[:-1], rows[1:]):
        total += 0.5 * (a["stress"] + b["stress"]) * (b["strain"] - a["strain"])
    return abs(total)


def rel_area_error(lhs: list[dict[str, float]], ref: list[dict[str, float]]) -> float:
    la = loop_area(lhs)
    ra = loop_area(ref)
    if not math.isfinite(la) or not math.isfinite(ra) or abs(ra) <= 1.0e-15:
        return math.nan
    return abs(la - ra) / abs(ra)


def save_figure(fig: plt.Figure, stem: str, figures_dir: Path, secondary_dir: Path | None) -> list[str]:
    outputs: list[str] = []
    for base in [figures_dir, *([secondary_dir] if secondary_dir else [])]:
        if base is None:
            continue
        base.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            out = base / f"{stem}.{ext}"
            try:
                fig.savefig(out, dpi=250, bbox_inches="tight")
            except PermissionError:
                out = base / f"{stem}_updated.{ext}"
                fig.savefig(out, dpi=250, bbox_inches="tight")
            outputs.append(str(out))
    plt.close(fig)
    return outputs


def load_case_rows(matrix_dir: Path) -> dict[tuple[int, str], dict[str, Any]]:
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in read_csv(matrix_dir / "timoshenko_matrix_cases.csv"):
        n = int(finite(row.get("beam_nodes")))
        q = str(row.get("beam_integration"))
        out[(n, q)] = row
    return out


def compute_metrics(
    args: argparse.Namespace,
) -> tuple[list[SteelMetric], list[dict[str, float]], dict[tuple[int, str], list[dict[str, float]]], list[dict[str, Any]], dict[str, Any]]:
    case_rows = load_case_rows(args.matrix_dir)
    ref_dir = args.matrix_dir / "opensees_hifi_reference"
    ref_curve_raw, ref_info, ref_outliers = steel_curve(ref_dir, "opensees_hifi_reference")
    ref_curve = unique_by_p(ref_curve_raw)
    metrics: list[SteelMetric] = []
    series_cache: dict[tuple[int, str], list[dict[str, float]]] = {}
    outliers: list[dict[str, Any]] = ref_outliers
    for (n, q), case in sorted(case_rows.items()):
        if n < args.min_n or n > args.max_n:
            continue
        status = str(case.get("status", ""))
        bundle = Path(str(case.get("bundle_dir", "")))
        if status != "completed" or not bundle.exists():
            metrics.append(
                SteelMetric(
                    n,
                    q,
                    status,
                    str(bundle),
                    finite(case.get("process_wall_seconds")),
                    finite(case.get("hifi_hysteresis_rms_rel_error")),
                    finite(case.get("hifi_total_work_rel_error")),
                    math.nan,
                    math.nan,
                    math.nan,
                    0,
                    -1,
                    -1,
                    math.nan,
                    math.nan,
                    math.nan,
                    math.nan,
                )
            )
            continue
        case_name = f"n{n:02d}_{q}"
        curve_raw, info, case_outliers = steel_curve(bundle, case_name)
        outliers.extend(case_outliers)
        curve = unique_by_p(curve_raw)
        series_cache[(n, q)] = curve
        stress_rms = rms_rel_at_p(curve, ref_curve, "stress")
        strain_rms = rms_rel_at_p(curve, ref_curve, "strain")
        area = rel_area_error(curve, ref_curve)
        hys = finite(case.get("hifi_hysteresis_rms_rel_error"))
        work = finite(case.get("hifi_total_work_rel_error"))
        wall = finite(case.get("process_wall_seconds"))
        steel_score = (
            0.45 * stress_rms + 0.35 * strain_rms + 0.20 * area
            if all(math.isfinite(v) for v in (stress_rms, strain_rms, area))
            else math.nan
        )
        production_score = (
            0.20 * hys
            + 0.15 * work
            + 0.30 * stress_rms
            + 0.20 * strain_rms
            + 0.15 * area
            + 0.03 * math.log10(max(wall, 1.0))
            if all(math.isfinite(v) for v in (hys, work, stress_rms, strain_rms, area, wall))
            else math.nan
        )
        metrics.append(
            SteelMetric(
                n,
                q,
                status,
                str(bundle),
                wall,
                hys,
                work,
                stress_rms,
                strain_rms,
                area,
                len(case_outliers),
                int(info.get("section_gp", -1)),
                int(info.get("fiber_index", -1)),
                finite(info.get("y")),
                finite(info.get("z")),
                steel_score,
                production_score,
            )
        )
    return metrics, ref_curve, series_cache, outliers, ref_info


def plot_metrics(
    metrics: list[SteelMetric],
    figures_dir: Path,
    secondary_dir: Path | None,
) -> list[str]:
    completed = [m for m in metrics if m.status == "completed"]
    quadratures = [q for q in QUAD_LABEL if any(m.beam_integration == q for m in completed)]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=True)
    for q in quadratures:
        scoped = sorted([m for m in completed if m.beam_integration == q], key=lambda m: m.beam_nodes)
        x = [m.beam_nodes for m in scoped]
        color = QUAD_COLOR[q]
        axes[0].plot(x, [m.steel_stress_rms_rel for m in scoped], marker="o", color=color, label=QUAD_LABEL[q])
        axes[1].plot(x, [m.steel_strain_rms_rel for m in scoped], marker="o", color=color, label=QUAD_LABEL[q])
        axes[2].plot(x, [m.production_score for m in scoped], marker="o", color=color, label=QUAD_LABEL[q])
    axes[0].set_ylabel("RMS relative error")
    axes[0].set_title("Steel stress")
    axes[1].set_ylabel("RMS relative error")
    axes[1].set_title("Steel strain")
    axes[2].set_ylabel("Weighted score")
    axes[2].set_title("Production score")
    for ax in axes:
        ax.set_xlabel("Beam nodes N")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    return save_figure(
        fig,
        "reduced_rc_timoshenko_matrix_extreme_steel_selection",
        figures_dir,
        secondary_dir,
    )


def plot_steel_overlays(
    metrics: list[SteelMetric],
    ref_curve: list[dict[str, float]],
    series_cache: dict[tuple[int, str], list[dict[str, float]]],
    figures_dir: Path,
    secondary_dir: Path | None,
) -> list[str]:
    completed = [m for m in metrics if m.status == "completed"]
    quadratures = [q for q in QUAD_LABEL if any(m.beam_integration == q for m in completed)]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, q in zip(axes.flat, quadratures, strict=False):
        ax.plot(
            [row["strain"] for row in ref_curve],
            [row["stress"] for row in ref_curve],
            color="black",
            lw=1.7,
            label="OpenSees hi-fi",
        )
        scoped = sorted([m for m in completed if m.beam_integration == q], key=lambda m: m.beam_nodes)
        for m in scoped:
            rows = series_cache.get((m.beam_nodes, m.beam_integration), [])
            color = cmap((m.beam_nodes - 3) / max(10 - 3, 1))
            lw = 1.5 if m.beam_nodes in (4, 10) else 0.9
            alpha = 0.95 if m.beam_nodes in (4, 10) else 0.45
            label = f"N={m.beam_nodes}" if m.beam_nodes in (3, 4, 6, 8, 10) else None
            ax.plot(
                [row["strain"] for row in rows],
                [row["stress"] for row in rows],
                color=color,
                lw=lw,
                alpha=alpha,
                label=label,
            )
        ax.set_title(QUAD_LABEL[q])
        ax.set_xlabel("Extreme steel strain")
        ax.set_ylabel("Extreme steel stress [MPa]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    return save_figure(
        fig,
        "reduced_rc_timoshenko_matrix_extreme_steel_overlays",
        figures_dir,
        secondary_dir,
    )


def best_payload(metrics: list[SteelMetric], ref_info: dict[str, Any]) -> dict[str, Any]:
    completed = [m for m in metrics if m.status == "completed"]
    by_steel = min(completed, key=lambda m: m.steel_score if math.isfinite(m.steel_score) else math.inf)
    by_production = min(completed, key=lambda m: m.production_score if math.isfinite(m.production_score) else math.inf)
    lobatto = [m for m in completed if m.beam_integration == "lobatto"]
    by_lobatto_steel = min(
        lobatto,
        key=lambda m: m.steel_score if math.isfinite(m.steel_score) else math.inf,
    ) if lobatto else None
    by_lobatto_production = min(
        lobatto,
        key=lambda m: m.production_score if math.isfinite(m.production_score) else math.inf,
    ) if lobatto else None
    by_quad: dict[str, Any] = {}
    for q in QUAD_LABEL:
        scoped = [m for m in completed if m.beam_integration == q]
        if scoped:
            by_quad[q] = asdict(min(scoped, key=lambda m: m.production_score if math.isfinite(m.production_score) else math.inf))
    return {
        "steel_fiber_selection_rule": (
            "Base station; material_role=reinforcing_steel; maximum |z|, then maximum |y|, "
            "then lowest fiber_index. This selects the same corner bar family for fall_n and OpenSees."
        ),
        "opensees_reference_fiber": ref_info,
        "selection_rule": (
            "Steel metrics are computed after the same axial-equilibrium filter used by the moment-curvature audit. "
            "steel_score = 0.45*stress_rms + 0.35*strain_rms + 0.20*steel_loop_error; "
            "production_score = 0.20*base_shear_rms + 0.15*hysteretic_work_error + "
            "0.30*steel_stress_rms + 0.20*steel_strain_rms + 0.15*steel_loop_error + "
            "0.03*log10(wall_seconds)."
        ),
        "best_steel": asdict(by_steel),
        "best_production": asdict(by_production),
        "best_lobatto_steel": asdict(by_lobatto_steel) if by_lobatto_steel else None,
        "best_lobatto_production": asdict(by_lobatto_production) if by_lobatto_production else None,
        "best_production_by_quadrature": by_quad,
    }


def main() -> int:
    args = parse_args()
    metrics, ref_curve, series_cache, outliers, ref_info = compute_metrics(args)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    if args.secondary_figures_dir:
        args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(m) for m in metrics]
    csv_path = args.figures_dir / "reduced_rc_timoshenko_matrix_extreme_steel_selection.csv"
    write_csv(csv_path, rows)
    outlier_csv_path = args.figures_dir / "reduced_rc_timoshenko_matrix_extreme_steel_outliers.csv"
    write_csv(outlier_csv_path, outliers)
    if args.secondary_figures_dir:
        write_csv(
            args.secondary_figures_dir / "reduced_rc_timoshenko_matrix_extreme_steel_selection.csv",
            rows,
        )
        write_csv(
            args.secondary_figures_dir / "reduced_rc_timoshenko_matrix_extreme_steel_outliers.csv",
            outliers,
        )
    figures: list[str] = []
    figures.extend(plot_metrics(metrics, args.figures_dir, args.secondary_figures_dir))
    figures.extend(plot_steel_overlays(metrics, ref_curve, series_cache, args.figures_dir, args.secondary_figures_dir))
    payload = {
        "matrix_dir": str(args.matrix_dir),
        "metrics_csv": str(csv_path),
        "outlier_csv": str(outlier_csv_path),
        "outlier_filter": {
            "contract": "Extreme steel selection excludes steps whose base-section axial force violates the imposed axial compression.",
            "expected_abs_axial_MN_default": DEFAULT_EXPECTED_AXIAL_MN,
            "axial_tolerance_MN": AXIAL_CONTRACT_TOLERANCE_MN,
            "outlier_count": len(outliers),
        },
        "figures": figures,
        **best_payload(metrics, ref_info),
    }
    json_path = args.figures_dir / "reduced_rc_timoshenko_matrix_extreme_steel_selection.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.secondary_figures_dir:
        (
            args.secondary_figures_dir / "reduced_rc_timoshenko_matrix_extreme_steel_selection.json"
        ).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
