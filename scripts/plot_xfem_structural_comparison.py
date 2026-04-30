#!/usr/bin/env python3
"""Compare the global XFEM RC-column smoke branch against a structural run."""

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
        description="Overlay global XFEM and structural reduced-column hysteresis."
    )
    parser.add_argument(
        "--structural-dir",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
        "reboot_structural_reference_n10_lobatto_200mm_for_xfem_secant_compare",
    )
    parser.add_argument(
        "--xfem-dir",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
        "xfem_global_cyclic_crack_band_secant_200mm_structural_compare",
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
        default="xfem_global_secant_vs_structural_n10_lobatto_200mm",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    def coerce(value: str) -> Any:
        if value == "":
            return math.nan
        try:
            return float(value)
        except ValueError:
            return value

    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def interpolate_by_p(
    rows: list[dict[str, float]],
    p: float,
    key: str,
) -> float:
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
    t = (p - lo["p"]) / span
    return (1.0 - t) * lo[key] + t * hi[key]


def rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def peak_abs(values: list[float]) -> float:
    return max(abs(value) for value in values) if values else math.nan


def loop_work(rows: list[dict[str, float]], shear_key: str, drift_key: str) -> float:
    work = 0.0
    for previous, current in zip(rows, rows[1:]):
        work += (
            0.5
            * (previous[shear_key] + current[shear_key])
            * (current[drift_key] - previous[drift_key])
        )
    return work


def make_promotion_gate(metrics: dict[str, float], xfem_manifest: dict[str, Any]) -> dict[str, Any]:
    criteria = {
        "max_peak_normalized_rms_base_shear_error": 0.10,
        "max_peak_normalized_max_base_shear_error": 0.30,
        "min_peak_base_shear_ratio": 0.90,
        "max_peak_base_shear_ratio": 1.15,
        "min_peak_steel_stress_MPa": 420.0,
    }
    ratio = metrics["xfem_to_structural_peak_base_shear_ratio"]
    observables = xfem_manifest.get("observables", {})
    peak_steel = float(observables.get("peak_abs_steel_stress_mpa", math.nan))
    passed = (
        metrics["peak_normalized_rms_base_shear_error"]
        <= criteria["max_peak_normalized_rms_base_shear_error"]
        and metrics["peak_normalized_max_base_shear_error"]
        <= criteria["max_peak_normalized_max_base_shear_error"]
        and criteria["min_peak_base_shear_ratio"]
        <= ratio
        <= criteria["max_peak_base_shear_ratio"]
        and peak_steel >= criteria["min_peak_steel_stress_MPa"]
    )
    return {
        "status": "passed" if passed else "not_passed",
        "criteria": criteria,
        "peak_abs_steel_stress_MPa": peak_steel,
        "interpretation": (
            "This gate is a reduced local-model equivalence gate against the "
            "N=10 Lobatto structural reference, not a final production "
            "robustness certificate."
        ),
    }


def save_figure(fig: Any, figures_dir: Path, secondary_dir: Path, stem: str) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for extension in ("png", "pdf"):
        path = figures_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        shutil.copy2(path, secondary_dir / path.name)
        artifacts[extension] = str(path)
        artifacts[f"secondary_{extension}"] = str(secondary_dir / path.name)
    return artifacts


def main() -> int:
    args = parse_args()
    structural = read_csv(args.structural_dir / "comparison_hysteresis.csv")
    xfem = read_csv(args.xfem_dir / "global_xfem_newton_hysteresis.csv")
    structural_manifest = read_json(args.structural_dir / "runtime_manifest.json")
    xfem_manifest = read_json(args.xfem_dir / "global_xfem_newton_manifest.json")

    if not bool(xfem_manifest.get("completed_successfully", False)):
        summary = {
            "scope": "global_xfem_secant_vs_structural_reference_200mm",
            "status": "incomplete_xfem_run",
            "structural_bundle": str(args.structural_dir),
            "xfem_bundle": str(args.xfem_dir),
            "failure_reason": xfem_manifest.get("failure_reason", ""),
            "xfem_manifest": {
                "mesh": xfem_manifest.get("mesh"),
                "reinforcement": xfem_manifest.get("reinforcement"),
                "physics": xfem_manifest.get("physics"),
                "solve_control": xfem_manifest.get("solve_control"),
                "observables": xfem_manifest.get("observables"),
                "timing": xfem_manifest.get("timing"),
            },
            "promotion_gate": {
                "status": "not_evaluated",
                "interpretation": (
                    "The XFEM solve did not complete the declared protocol, "
                    "so hysteresis-equivalence metrics would be misleading."
                ),
            },
        }
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.figures_dir / f"{args.basename}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_path, args.secondary_figures_dir / summary_path.name)
        print(json.dumps(summary, indent=2))
        return 2

    structural_for_compare: list[dict[str, float]] = []
    for row in xfem:
        p = row["p"]
        structural_for_compare.append(
            {
                "p": p,
                "drift_mm": 1000.0
                * interpolate_by_p(structural, p, "drift_m"),
                "base_shear_MN": interpolate_by_p(
                    structural,
                    p,
                    "base_shear_MN",
                ),
            }
        )

    xfem_shear = [row["base_shear_MN"] for row in xfem]
    structural_shear = [row["base_shear_MN"] for row in structural_for_compare]
    raw_errors = [candidate - reference for candidate, reference in zip(xfem_shear, structural_shear)]
    flipped_errors = [
        candidate + reference for candidate, reference in zip(xfem_shear, structural_shear)
    ]
    raw_rms = rms(raw_errors)
    flipped_rms = rms(flipped_errors)
    sign_factor = -1.0 if flipped_rms < raw_rms else 1.0
    aligned_structural_shear = [sign_factor * value for value in structural_shear]
    aligned_errors = [
        candidate - reference
        for candidate, reference in zip(xfem_shear, aligned_structural_shear)
    ]
    aligned_structural_rows = [
        {
            "p": row["p"],
            "drift_mm": row["drift_mm"],
            "base_shear_MN": sign_factor * row["base_shear_MN"],
        }
        for row in structural_for_compare
    ]
    xfem_rows_for_work = [
        {
            "drift_mm": row["drift_mm"],
            "base_shear_MN": row["base_shear_MN"],
        }
        for row in xfem
    ]

    peak_xfem = peak_abs(xfem_shear)
    peak_structural = peak_abs(aligned_structural_shear)
    normalization = max(peak_xfem, peak_structural, 1.0e-12)

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
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    axes[0].plot(
        [1000.0 * row["drift_m"] for row in structural],
        [1000.0 * sign_factor * row["base_shear_MN"] for row in structural],
        color="#111827",
        lw=1.5,
        label="Structural N=10 Lobatto, sign-aligned",
    )
    axes[0].plot(
        [row["drift_mm"] for row in xfem],
        [1000.0 * row["base_shear_MN"] for row in xfem],
        color="#0f766e",
        lw=1.4,
        label="Global XFEM crack-band secant",
    )
    axes[0].plot(
        [1000.0 * row["drift_m"] for row in structural],
        [1000.0 * row["base_shear_MN"] for row in structural],
        color="#9ca3af",
        lw=0.8,
        ls=":",
        label="Structural raw sign",
    )
    axes[0].set_title("Hysteresis overlay")
    axes[0].set_xlabel("Tip drift [mm]")
    axes[0].set_ylabel("Base shear [kN]")
    axes[0].legend(fontsize=7)

    axes[1].plot(
        [row["p"] for row in xfem],
        [1000.0 * error for error in aligned_errors],
        color="#b91c1c",
        lw=1.2,
        label="XFEM - structural",
    )
    axes[1].axhline(0.0, color="#111827", lw=0.8)
    axes[1].set_title("Aligned residual over protocol")
    axes[1].set_xlabel("Protocol pseudo-time p")
    axes[1].set_ylabel("Base-shear residual [kN]")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Global XFEM secant branch vs structural N=10 Lobatto reference\n"
        f"peak-normalized RMS={rms(aligned_errors) / normalization:.3f}, "
        f"peak ratio XFEM/struct={peak_xfem / peak_structural:.3f}",
        y=1.05,
    )
    artifacts = save_figure(
        fig,
        args.figures_dir,
        args.secondary_figures_dir,
        f"{args.basename}_hysteresis",
    )
    plt.close(fig)

    metrics = {
        "point_count": len(xfem),
        "peak_abs_structural_base_shear_MN": peak_structural,
        "peak_abs_xfem_base_shear_MN": peak_xfem,
        "xfem_to_structural_peak_base_shear_ratio": peak_xfem
        / max(peak_structural, 1.0e-12),
        "max_abs_base_shear_error_MN": peak_abs(aligned_errors),
        "rms_base_shear_error_MN": rms(aligned_errors),
        "peak_normalized_max_base_shear_error": peak_abs(aligned_errors)
        / normalization,
        "peak_normalized_rms_base_shear_error": rms(aligned_errors)
        / normalization,
        "structural_loop_work_MN_mm": loop_work(
            aligned_structural_rows,
            "base_shear_MN",
            "drift_mm",
        ),
        "xfem_loop_work_MN_mm": loop_work(
            xfem_rows_for_work,
            "base_shear_MN",
            "drift_mm",
        ),
    }

    summary = {
        "scope": "global_xfem_secant_vs_structural_reference_200mm",
        "status": "completed",
        "structural_bundle": str(args.structural_dir),
        "xfem_bundle": str(args.xfem_dir),
        "sign_convention": {
            "applied_to_structural_base_shear": sign_factor,
            "raw_rms_error_MN": raw_rms,
            "sign_aligned_rms_error_MN": rms(aligned_errors),
            "interpretation": (
                "A factor of -1 indicates opposite base-shear reaction "
                "conventions between the structural and XFEM artifacts."
            ),
        },
        "metrics": metrics,
        "promotion_gate": make_promotion_gate(metrics, xfem_manifest),
        "structural_manifest": {
            "beam_nodes": structural_manifest.get("beam_nodes"),
            "beam_integration": structural_manifest.get("beam_integration"),
            "section_fiber_mesh": structural_manifest.get("section_fiber_mesh"),
            "continuation_kind": structural_manifest.get("continuation_kind"),
            "timing": structural_manifest.get("timing"),
        },
        "xfem_manifest": {
            "mesh": xfem_manifest.get("mesh"),
            "reinforcement": xfem_manifest.get("reinforcement"),
            "physics": xfem_manifest.get("physics"),
            "solve_control": xfem_manifest.get("solve_control"),
            "observables": xfem_manifest.get("observables"),
            "timing": xfem_manifest.get("timing"),
        },
        "artifacts": artifacts,
    }

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.figures_dir / f"{args.basename}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, args.secondary_figures_dir / summary_path.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
