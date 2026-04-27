#!/usr/bin/env python3
"""Create a two-panel 200 mm continuum external hysteresis figure.

The paired runner already emits per-bundle plots.  This script creates the
validation-facing figure that compares the two relevant external-solver
controls side by side:

* an elastic-with-steel bbarBrick control that reaches 200 mm on both solvers;
* a nonlinear ASDConcrete3D/SSPbrick control where fall_n reaches 200 mm but
  the external OpenSees comparator stops at its convergence frontier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 200 mm paired continuum hysteresis controls."
    )
    parser.add_argument(
        "--elastic-bundle",
        type=Path,
        default=Path(
            "data/output/cyclic_validation/"
            "continuum_external_elastic_with_steel_200mm_bbarBrick_refined_6x6x12"
        ),
    )
    parser.add_argument(
        "--nonlinear-bundle",
        type=Path,
        default=Path(
            "data/output/cyclic_validation/"
            "continuum_external_asd_cyclic_200mm_SSPbrick_coarse_v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/figures/validation_reboot"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        out: list[dict[str, float]] = []
        for row in csv.DictReader(fh):
            numeric: dict[str, float] = {}
            for key, value in row.items():
                try:
                    numeric[key] = float(value)
                except (TypeError, ValueError):
                    numeric[key] = math.nan
            out.append(numeric)
        return out


def read_summary(bundle: Path) -> dict[str, object]:
    return json.loads(
        (bundle / "continuum_external_benchmark_summary.json").read_text(
            encoding="utf-8"
        )
    )


def hysteresis(bundle: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    falln = read_rows(bundle / "fall_n" / "hysteresis.csv")
    ops = read_rows(bundle / "opensees" / "hysteresis.csv")

    def points(rows: list[dict[str, float]]) -> list[tuple[float, float]]:
        return [
            (1000.0 * row["drift_m"], 1000.0 * row["base_shear_MN"])
            for row in rows
            if math.isfinite(row.get("drift_m", math.nan))
            and math.isfinite(row.get("base_shear_MN", math.nan))
        ]

    return points(falln), points(ops)


def finite_metric(summary: dict[str, object], metric: str, field: str) -> float:
    try:
        value = summary["comparison"][metric][field]  # type: ignore[index]
        out = float(value)
    except (KeyError, TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def max_abs_x(points: list[tuple[float, float]]) -> float:
    return max((abs(x) for x, _ in points), default=0.0)


def plot_case(ax, bundle: Path, title: str) -> dict[str, object]:
    summary = read_summary(bundle)
    falln, ops = hysteresis(bundle)
    ax.plot(
        [x for x, _ in falln],
        [y for _, y in falln],
        color="#1f77b4",
        linewidth=1.45,
        label="fall_n",
    )
    ax.plot(
        [x for x, _ in ops],
        [y for _, y in ops],
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.35,
        label="OpenSeesPy",
    )
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    ax.grid(True, alpha=0.24)
    ax.set_title(title)
    ax.set_xlabel("Tip drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.legend(fontsize=8)

    base_rms = finite_metric(
        summary,
        "hysteresis_base_shear_at_matched_drift",
        "rms_abs_error",
    )
    steel_rms = finite_metric(
        summary,
        "steel_max_abs_stress_at_matched_drift",
        "rms_abs_error",
    )
    status = str(summary.get("status", "unknown"))
    annotation = (
        f"status: {status}\n"
        f"max drift fall_n/OpenSees: {max_abs_x(falln):.0f}/{max_abs_x(ops):.0f} mm\n"
        f"RMS dV: {1000.0 * base_rms:.1f} kN"
    )
    if math.isfinite(steel_rms):
        annotation += f"\nRMS dsigma_s: {steel_rms:.1f} MPa"
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.86},
    )
    return {
        "bundle": str(bundle),
        "status": status,
        "falln_max_abs_drift_mm": max_abs_x(falln),
        "opensees_max_abs_drift_mm": max_abs_x(ops),
        "base_shear_rms_error_MN": base_rms,
        "steel_stress_rms_error_MPa": steel_rms,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=False)
    rows = [
        plot_case(
            axes[0],
            args.elastic_bundle,
            "Elastic host + steel control to 200 mm",
        ),
        plot_case(
            axes[1],
            args.nonlinear_bundle,
            "Nonlinear ASDConcrete3D frontier",
        ),
    ]
    fig.suptitle("Continuum external hysteresis comparison at 200 mm", y=1.03)
    fig.tight_layout()

    outputs: dict[str, str] = {}
    for ext in ("png", "pdf", "svg"):
        path = args.output_dir / f"continuum_external_hysteresis_200mm_panel.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=250 if ext == "png" else None)
        outputs[ext] = str(path)
    plt.close(fig)

    summary = {
        "status": "completed",
        "cases": rows,
        "figures": outputs,
    }
    (args.output_dir / "continuum_external_hysteresis_200mm_panel_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
