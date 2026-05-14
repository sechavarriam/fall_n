#!/usr/bin/env python3
"""Plot guarded oblique-plane XFEM mesh branches against the structural column."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from bisect import bisect_left
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare guarded oblique XFEM mesh hysteresis branches."
    )
    parser.add_argument(
        "--structural-dir",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
        "reboot_external_benchmark_cyclic_200mm_publication_20260511/fall_n",
    )
    parser.add_argument(
        "--case",
        action="append",
        nargs=2,
        metavar=("LABEL", "DIR"),
        help="Case label and output directory. Can be repeated.",
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
        "--basename",
        default="xfem_oblique_multimesh_hysteresis_200mm",
    )
    return parser.parse_args()


def default_cases(repo: Path) -> list[tuple[str, Path]]:
    root = repo / "data/output/cyclic_validation"
    return [
        (
            "Hex8 1x1x4",
            root / "xfem_oblique_mild_guarded_mesh_1x1x4_200mm_20260513",
        ),
        (
            "Hex8 2x2x4",
            root / "xfem_oblique_mild_guarded_mesh_2x2x4_200mm_20260513",
        ),
        (
            "Hex8 3x3x8",
            root / "xfem_oblique_mild_guarded_mesh_3x3x8_200mm_20260513",
        ),
    ]


def coerce(value: str) -> Any:
    if value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def interpolate_by_p(rows: list[dict[str, float]], p: float, key: str) -> float:
    ps = [row["p"] for row in rows]
    index = bisect_left(ps, p)
    if index <= 0:
        return rows[0][key]
    if index >= len(rows):
        return rows[-1][key]
    lo = rows[index - 1]
    hi = rows[index]
    span = hi["p"] - lo["p"]
    if abs(span) < 1.0e-15:
        return hi[key]
    alpha = (p - lo["p"]) / span
    return (1.0 - alpha) * lo[key] + alpha * hi[key]


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / max(len(values), 1))


def peak_abs(values: list[float]) -> float:
    return max((abs(value) for value in values), default=math.nan)


def loop_work(rows: list[dict[str, float]], shear_key: str, drift_key: str) -> float:
    work = 0.0
    for previous, current in zip(rows, rows[1:]):
        work += (
            0.5
            * (previous[shear_key] + current[shear_key])
            * (current[drift_key] - previous[drift_key])
        )
    return work


def aligned_structural_rows(
    structural: list[dict[str, float]],
    xfem_rows: list[dict[str, float]],
) -> tuple[float, list[dict[str, float]]]:
    sampled = [
        {
            "p": row["p"],
            "drift_mm": 1000.0 * interpolate_by_p(structural, row["p"], "drift_m"),
            "base_shear_MN": interpolate_by_p(
                structural, row["p"], "base_shear_MN"
            ),
        }
        for row in xfem_rows
    ]
    xfem_shear = [row["base_shear_MN"] for row in xfem_rows]
    structural_shear = [row["base_shear_MN"] for row in sampled]
    raw = rms([candidate - reference for candidate, reference in zip(xfem_shear, structural_shear)])
    flipped = rms([candidate + reference for candidate, reference in zip(xfem_shear, structural_shear)])
    sign = -1.0 if flipped < raw else 1.0
    return sign, [
        {
            "p": row["p"],
            "drift_mm": row["drift_mm"],
            "base_shear_MN": sign * row["base_shear_MN"],
        }
        for row in sampled
    ]


def case_metrics(
    label: str,
    case_dir: Path,
    structural: list[dict[str, float]],
) -> dict[str, Any]:
    rows = read_csv(case_dir / "global_xfem_newton_hysteresis.csv")
    manifest = read_json(case_dir / "global_xfem_newton_manifest.json")
    sign, aligned = aligned_structural_rows(structural, rows)
    xfem_shear = [row["base_shear_MN"] for row in rows]
    structural_shear = [row["base_shear_MN"] for row in aligned]
    errors = [
        candidate - reference
        for candidate, reference in zip(xfem_shear, structural_shear)
    ]
    peak_xfem = peak_abs(xfem_shear)
    peak_structural = peak_abs(structural_shear)
    norm = max(peak_xfem, peak_structural, 1.0e-12)
    guard = (
        manifest.get("physics", {})
        .get("crack_plane_guard", {})
        .get("records", [{}])[0]
    )
    return {
        "label": label,
        "case_dir": str(case_dir),
        "rows": rows,
        "aligned_structural_rows": aligned,
        "manifest": manifest,
        "sign_factor": sign,
        "completed": bool(manifest.get("completed_successfully", False)),
        "status": manifest.get("status", "unknown"),
        "mesh": manifest.get("mesh", {}),
        "guard": guard,
        "peak_abs_xfem_base_shear_MN": peak_xfem,
        "peak_abs_structural_base_shear_MN": peak_structural,
        "xfem_to_structural_peak_base_shear_ratio": peak_xfem / peak_structural,
        "peak_normalized_rms_base_shear_error": rms(errors) / norm,
        "peak_normalized_max_base_shear_error": peak_abs(errors) / norm,
        "xfem_loop_work_MN_mm": loop_work(rows, "base_shear_MN", "drift_mm"),
        "structural_loop_work_MN_mm": loop_work(
            aligned, "base_shear_MN", "drift_mm"
        ),
    }


def save_figure(fig: Any, figures_dir: Path, secondary_dir: Path, stem: str) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for extension in ("png", "pdf"):
        path = figures_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        target = secondary_dir / path.name
        shutil.copy2(path, target)
        artifacts[extension] = str(path)
        artifacts[f"secondary_{extension}"] = str(target)
    return artifacts


def write_metrics_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    columns = [
        "label",
        "status",
        "completed",
        "nx",
        "ny",
        "nz",
        "cut_element_count",
        "min_abs_node_signed_distance_m",
        "min_relative_intersection_area",
        "peak_abs_xfem_base_shear_MN",
        "peak_abs_structural_base_shear_MN",
        "xfem_to_structural_peak_base_shear_ratio",
        "peak_normalized_rms_base_shear_error",
        "peak_normalized_max_base_shear_error",
        "xfem_loop_work_MN_mm",
        "structural_loop_work_MN_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in cases:
            mesh = case["mesh"]
            guard = case["guard"]
            writer.writerow(
                {
                    "label": case["label"],
                    "status": case["status"],
                    "completed": int(case["completed"]),
                    "nx": mesh.get("nx"),
                    "ny": mesh.get("ny"),
                    "nz": mesh.get("nz"),
                    "cut_element_count": guard.get("cut_element_count"),
                    "min_abs_node_signed_distance_m": guard.get(
                        "min_abs_node_signed_distance_m"
                    ),
                    "min_relative_intersection_area": guard.get(
                        "min_relative_intersection_area"
                    ),
                    "peak_abs_xfem_base_shear_MN": case[
                        "peak_abs_xfem_base_shear_MN"
                    ],
                    "peak_abs_structural_base_shear_MN": case[
                        "peak_abs_structural_base_shear_MN"
                    ],
                    "xfem_to_structural_peak_base_shear_ratio": case[
                        "xfem_to_structural_peak_base_shear_ratio"
                    ],
                    "peak_normalized_rms_base_shear_error": case[
                        "peak_normalized_rms_base_shear_error"
                    ],
                    "peak_normalized_max_base_shear_error": case[
                        "peak_normalized_max_base_shear_error"
                    ],
                    "xfem_loop_work_MN_mm": case["xfem_loop_work_MN_mm"],
                    "structural_loop_work_MN_mm": case[
                        "structural_loop_work_MN_mm"
                    ],
                }
            )


def extract_plane_definition(cases: list[dict[str, Any]]) -> dict[str, Any]:
    for case in cases:
        planes = (
            case.get("manifest", {})
            .get("physics", {})
            .get("crack_planes", [])
        )
        if planes:
            plane = planes[0]
            return {
                "point_m": plane.get("point", []),
                "normal_unit": plane.get("normal", []),
                "source": plane.get("source", "prescribed"),
                "plane_id": plane.get("plane_id"),
                "sequence_id": plane.get("sequence_id"),
            }
    return {
        "point_m": [0.0, 0.0, 0.10],
        "normal_raw": [0.10, 0.10, 1.0],
        "normal_unit": [0.0990147543, 0.0990147543, 0.990147543],
        "source": "prescribed",
    }


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    cases = (
        [(label, Path(path)) for label, path in args.case]
        if args.case
        else default_cases(repo)
    )
    structural = read_csv(args.structural_dir / "comparison_hysteresis.csv")
    metrics = [case_metrics(label, case_dir, structural) for label, case_dir in cases]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "savefig.bbox": "tight",
        }
    )

    colors = ["#0f766e", "#1d4ed8", "#7c3aed", "#b45309", "#be123c"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    ax = axes[0]
    reference_rows = metrics[-1]["aligned_structural_rows"]
    ax.plot(
        [row["drift_mm"] for row in reference_rows],
        [1000.0 * row["base_shear_MN"] for row in reference_rows],
        color="#111827",
        lw=1.8,
        label="Structural reference",
    )
    for color, case in zip(colors, metrics):
        rows = case["rows"]
        ax.plot(
            [row["drift_mm"] for row in rows],
            [1000.0 * row["base_shear_MN"] for row in rows],
            color=color,
            lw=1.2,
            label=case["label"],
        )
    ax.set_title("Cyclic hysteresis, guarded oblique plane")
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.legend(fontsize=7.6)

    ax = axes[1]
    labels = [case["label"].replace("Hex8 ", "") for case in metrics]
    ratios = [case["xfem_to_structural_peak_base_shear_ratio"] for case in metrics]
    rms_errors = [case["peak_normalized_rms_base_shear_error"] for case in metrics]
    x = list(range(len(metrics)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], ratios, width, color="#2563eb", label="Peak ratio")
    ax.bar([i + width / 2 for i in x], rms_errors, width, color="#dc2626", label="RMS error")
    ax.axhline(1.0, color="#111827", lw=0.9, ls=":")
    ax.set_title("Mesh comparison metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, max(1.2, 1.12 * max(ratios + rms_errors)))
    ax.set_ylabel("Dimensionless")
    ax.legend(fontsize=8)

    fig.suptitle(
        "Guarded arbitrary-plane XFEM against the 200 mm structural protocol",
        y=1.03,
        fontsize=12,
    )
    artifacts = save_figure(
        fig,
        args.figures_dir,
        args.secondary_figures_dir,
        args.basename,
    )
    plt.close(fig)

    summary_cases: list[dict[str, Any]] = []
    for case in metrics:
        keep = {
            key: value
            for key, value in case.items()
            if key not in {"rows", "aligned_structural_rows", "manifest"}
        }
        keep["manifest_excerpt"] = {
            "observables": case["manifest"].get("observables", {}),
            "timing": case["manifest"].get("timing", {}),
            "solve_control": case["manifest"].get("solve_control", {}),
        }
        summary_cases.append(keep)

    summary = {
        "scope": "guarded_oblique_xfem_multimesh_200mm",
        "status": "completed" if all(case["completed"] for case in metrics) else "partial",
        "structural_bundle": str(args.structural_dir),
        "case_count": len(metrics),
        "plane_definition": extract_plane_definition(metrics),
        "interpretation": (
            "The guarded prescribed oblique-plane path completed the 200 mm "
            "cyclic protocol on the reported Hex8 mesh sequence. The structural "
            "reference is sign-aligned because the two artifacts use opposite "
            "base-reaction conventions."
        ),
        "cases": summary_cases,
        "artifacts": artifacts,
    }
    summary_path = args.figures_dir / f"{args.basename}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copy2(summary_path, args.secondary_figures_dir / summary_path.name)
    csv_path = args.figures_dir / f"{args.basename}_metrics.csv"
    write_metrics_csv(csv_path, metrics)
    shutil.copy2(csv_path, args.secondary_figures_dir / csv_path.name)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
