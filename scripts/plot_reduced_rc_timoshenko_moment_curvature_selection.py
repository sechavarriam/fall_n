#!/usr/bin/env python3
"""Moment-curvature selection plots for the reduced RC Timoshenko matrix.

This postprocess reads an already completed matrix campaign.  It compares each
fall_n bundle against the OpenSees hi-fi base station at matching pseudo-time
and writes publication figures plus a CSV/JSON ranking for N=3..10.
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
class SelectionMetric:
    beam_nodes: int
    beam_integration: str
    status: str
    bundle_dir: str
    wall_seconds: float
    hysteresis_rms_rel: float
    work_rel_error: float
    mc_moment_rms_rel: float
    mc_curvature_rms_rel: float
    mc_loop_work_rel_error: float
    mc_outlier_count: int
    section_score: float
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


def base_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    gp = min({str(row.get("section_gp", "0")) for row in rows}, key=lambda s: float(s))
    return [row for row in rows if str(row.get("section_gp", gp)) == gp]


def moment_curvature_path(path: Path) -> Path:
    if (path / "comparison_moment_curvature_base.csv").exists():
        return path / "comparison_moment_curvature_base.csv"
    if (path / "moment_curvature_base.csv").exists():
        return path / "moment_curvature_base.csv"
    return path / "section_response.csv"


def expected_axial_mn(bundle_dir: Path) -> float:
    for name in ("runtime_manifest.json", "reference_manifest.json", "manifest.json"):
        payload = read_json(bundle_dir / name)
        for key in ("axial_compression_mn", "target_axial_force_mn", "axial_force_mn"):
            value = finite(payload.get(key))
            if math.isfinite(value) and abs(value) > 0.0:
                return abs(value)
    return DEFAULT_EXPECTED_AXIAL_MN


def filter_section_rows(
    rows: list[dict[str, Any]],
    *,
    case_name: str,
    expected_abs_axial_mn: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        p = finite(row.get("p"))
        kappa = finite(row.get("curvature_y"))
        moment = finite(row.get("moment_y_MNm"))
        axial = finite(row.get("axial_force_MN"))
        reasons: list[str] = []
        if not math.isfinite(p):
            reasons.append("missing_p")
        if not math.isfinite(kappa):
            reasons.append("missing_curvature")
        if not math.isfinite(moment):
            reasons.append("missing_moment")
        if math.isfinite(axial) and expected_abs_axial_mn > 0.0:
            if abs(abs(axial) - expected_abs_axial_mn) > AXIAL_CONTRACT_TOLERANCE_MN:
                reasons.append("axial_force_out_of_contract")
        if reasons:
            rejected.append(
                {
                    "case": case_name,
                    "step": row.get("step", ""),
                    "p": p,
                    "section_gp": row.get("section_gp", ""),
                    "xi": row.get("xi", ""),
                    "curvature_y": kappa,
                    "moment_y_MNm": moment,
                    "axial_force_MN": axial,
                    "expected_abs_axial_MN": expected_abs_axial_mn,
                    "axial_tolerance_MN": AXIAL_CONTRACT_TOLERANCE_MN,
                    "reason": ";".join(reasons),
                }
            )
        else:
            kept.append(row)
    return kept, rejected


def audited_base_rows(bundle_dir: Path, case_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = base_rows(read_csv(moment_curvature_path(bundle_dir)))
    return filter_section_rows(
        rows,
        case_name=case_name,
        expected_abs_axial_mn=expected_axial_mn(bundle_dir),
    )


def normalized_p_series(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            "p": finite(row.get("p")),
            "kappa": finite(row.get("curvature_y")),
            "moment": 1000.0 * finite(row.get("moment_y_MNm")),
        }
        for row in rows
        if math.isfinite(finite(row.get("p")))
    ]


def interpolate_series(
    ref: list[dict[str, float]], query_p: float, field: str
) -> float:
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


def rms_rel_at_p(
    lhs: list[dict[str, float]], ref: list[dict[str, float]], field: str
) -> float:
    if not lhs or not ref:
        return math.nan
    scale = max(abs(row[field]) for row in ref if math.isfinite(row[field]))
    if scale <= 1.0e-15:
        return math.nan
    errors = []
    for row in lhs:
        rv = interpolate_series(ref, row["p"], field)
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
        dk = b["kappa"] - a["kappa"]
        avg_m = 0.5 * (a["moment"] + b["moment"])
        total += avg_m * dk
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
) -> tuple[
    list[SelectionMetric],
    list[dict[str, float]],
    dict[tuple[int, str], list[dict[str, float]]],
    list[dict[str, Any]],
]:
    case_rows = load_case_rows(args.matrix_dir)
    opensees_dir = args.matrix_dir / "opensees_hifi_reference"
    ref_rows, ref_outliers = audited_base_rows(opensees_dir, "opensees_hifi_reference")
    ref_series = normalized_p_series(ref_rows)
    metrics: list[SelectionMetric] = []
    series_cache: dict[tuple[int, str], list[dict[str, float]]] = {}
    outliers: list[dict[str, Any]] = ref_outliers
    for (n, q), case in sorted(case_rows.items()):
        if n < args.min_n or n > args.max_n:
            continue
        status = str(case.get("status", ""))
        bundle = Path(str(case.get("bundle_dir", "")))
        if status != "completed" or not bundle.exists():
            metrics.append(
                SelectionMetric(
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
                    math.nan,
                    math.nan,
                )
            )
            continue
        case_name = f"n{n:02d}_{q}"
        lhs_rows, lhs_outliers = audited_base_rows(bundle, case_name)
        outliers.extend(lhs_outliers)
        lhs = normalized_p_series(lhs_rows)
        series_cache[(n, q)] = lhs
        mc_m = rms_rel_at_p(lhs, ref_series, "moment")
        mc_k = rms_rel_at_p(lhs, ref_series, "kappa")
        mc_area = rel_area_error(lhs, ref_series)
        hys = finite(case.get("hifi_hysteresis_rms_rel_error"))
        work = finite(case.get("hifi_total_work_rel_error"))
        wall = finite(case.get("process_wall_seconds"))
        section_score = (
            0.35 * mc_m + 0.45 * mc_k + 0.20 * mc_area
            if all(math.isfinite(v) for v in (mc_m, mc_k, mc_area))
            else math.nan
        )
        production_score = (
            0.20 * hys
            + 0.15 * work
            + 0.25 * mc_m
            + 0.25 * mc_k
            + 0.15 * mc_area
            + 0.03 * math.log10(max(wall, 1.0))
            if all(math.isfinite(v) for v in (hys, work, mc_m, mc_k, mc_area, wall))
            else math.nan
        )
        metrics.append(
            SelectionMetric(
                n,
                q,
                status,
                str(bundle),
                wall,
                hys,
                work,
                mc_m,
                mc_k,
                mc_area,
                len(lhs_outliers),
                section_score,
                production_score,
            )
        )
    return metrics, ref_series, series_cache, outliers


def plot_metrics(
    metrics: list[SelectionMetric],
    figures_dir: Path,
    secondary_dir: Path | None,
) -> list[str]:
    outputs: list[str] = []
    completed = [m for m in metrics if m.status == "completed"]
    quadratures = [q for q in QUAD_LABEL if any(m.beam_integration == q for m in completed)]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharex=True)
    for q in quadratures:
        scoped = sorted([m for m in completed if m.beam_integration == q], key=lambda m: m.beam_nodes)
        x = [m.beam_nodes for m in scoped]
        color = QUAD_COLOR[q]
        axes[0].plot(x, [m.mc_moment_rms_rel for m in scoped], marker="o", color=color, label=QUAD_LABEL[q])
        axes[1].plot(x, [m.mc_curvature_rms_rel for m in scoped], marker="o", color=color, label=QUAD_LABEL[q])
        axes[2].plot(x, [m.production_score for m in scoped], marker="o", color=color, label=QUAD_LABEL[q])
    axes[0].set_ylabel("RMS relative error")
    axes[0].set_title("Base moment")
    axes[1].set_ylabel("RMS relative error")
    axes[1].set_title("Base curvature")
    axes[2].set_ylabel("Weighted score")
    axes[2].set_title("Production score")
    for ax in axes:
        ax.set_xlabel("Beam nodes N")
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    outputs += save_figure(
        fig,
        "reduced_rc_timoshenko_matrix_moment_curvature_selection",
        figures_dir,
        secondary_dir,
    )

    return outputs


def plot_moment_curvature_overlays(
    metrics: list[SelectionMetric],
    ref_series: list[dict[str, float]],
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
            [row["kappa"] for row in ref_series],
            [row["moment"] for row in ref_series],
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
                [row["kappa"] for row in rows],
                [row["moment"] for row in rows],
                color=color,
                lw=lw,
                alpha=alpha,
                label=label,
            )
        ax.set_title(QUAD_LABEL[q])
        ax.set_xlabel(r"Base curvature $\kappa$ [1/m]")
        ax.set_ylabel(r"Base moment $M$ [kN m]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    return save_figure(
        fig,
        "reduced_rc_timoshenko_matrix_moment_curvature_overlays",
        figures_dir,
        secondary_dir,
    )


def best_payload(metrics: list[SelectionMetric]) -> dict[str, Any]:
    completed = [m for m in metrics if m.status == "completed"]
    by_section = min(completed, key=lambda m: m.section_score if math.isfinite(m.section_score) else math.inf)
    by_production = min(
        completed,
        key=lambda m: m.production_score if math.isfinite(m.production_score) else math.inf,
    )
    lobatto = [m for m in completed if m.beam_integration == "lobatto"]
    by_lobatto_section = min(
        lobatto,
        key=lambda m: m.section_score if math.isfinite(m.section_score) else math.inf,
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
        "selection_rule": (
            "Moment-curvature metrics are computed after an audited axial-equilibrium filter on base-section rows; "
            "rejected points remain in reduced_rc_timoshenko_matrix_moment_curvature_outliers.csv. "
            "section_score = 0.35*moment_rms + 0.45*curvature_rms + 0.20*moment_curvature_loop_error; "
            "production_score = 0.20*base_shear_rms + 0.15*hysteretic_work_error + "
            "0.25*moment_rms + 0.25*curvature_rms + 0.15*moment_curvature_loop_error + "
            "0.03*log10(wall_seconds)."
        ),
        "best_sectional": asdict(by_section),
        "best_production": asdict(by_production),
        "best_lobatto_sectional": asdict(by_lobatto_section) if by_lobatto_section else None,
        "best_lobatto_production": asdict(by_lobatto_production) if by_lobatto_production else None,
        "best_production_by_quadrature": by_quad,
    }


def main() -> int:
    args = parse_args()
    metrics, ref_series, series_cache, outliers = compute_metrics(args)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    if args.secondary_figures_dir:
        args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(m) for m in metrics]
    csv_path = args.figures_dir / "reduced_rc_timoshenko_matrix_moment_curvature_selection.csv"
    write_csv(csv_path, rows)
    if args.secondary_figures_dir:
        write_csv(
            args.secondary_figures_dir
            / "reduced_rc_timoshenko_matrix_moment_curvature_selection.csv",
            rows,
        )
    outlier_csv_path = args.figures_dir / "reduced_rc_timoshenko_matrix_moment_curvature_outliers.csv"
    write_csv(outlier_csv_path, outliers)
    if args.secondary_figures_dir:
        write_csv(
            args.secondary_figures_dir
            / "reduced_rc_timoshenko_matrix_moment_curvature_outliers.csv",
            outliers,
        )
    figures = []
    figures.extend(plot_metrics(metrics, args.figures_dir, args.secondary_figures_dir))
    figures.extend(
        plot_moment_curvature_overlays(
            metrics,
            ref_series,
            series_cache,
            args.figures_dir,
            args.secondary_figures_dir,
        )
    )
    payload = {
        "matrix_dir": str(args.matrix_dir),
        "metrics_csv": str(csv_path),
        "outlier_csv": str(outlier_csv_path),
        "outlier_filter": {
            "contract": "Moment-curvature selection uses base-section rows whose axial force remains compatible with the imposed column axial compression.",
            "expected_abs_axial_MN_default": DEFAULT_EXPECTED_AXIAL_MN,
            "axial_tolerance_MN": AXIAL_CONTRACT_TOLERANCE_MN,
            "outlier_count": len(outliers),
        },
        "figures": figures,
        **best_payload(metrics),
    }
    json_path = args.figures_dir / "reduced_rc_timoshenko_matrix_moment_curvature_selection.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.secondary_figures_dir:
        (
            args.secondary_figures_dir
            / "reduced_rc_timoshenko_matrix_moment_curvature_selection.json"
        ).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
