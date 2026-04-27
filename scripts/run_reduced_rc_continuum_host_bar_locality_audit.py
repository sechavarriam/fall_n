#!/usr/bin/env python3
"""
Audit the local host-concrete neighborhood around the tracked embedded steel bar.

The structural-vs-continuum steel bridge already showed that host↔bar transfer
closes almost to machine precision on the promoted Hex20 slice. The next
scientific question is therefore local and physical:

  * what axial strain/stress does the nearest host Gauss point carry while the
    embedded steel follows its hysteresis?
  * how cracked is that host neighborhood?

This script consumes the promoted structural-vs-continuum steel bundle and
turns the selected continuum steel trace into a host-vs-bar locality summary
with figures. It does not rerun the expensive benchmark by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def amplitude_suffix_from_summary(summary: dict[str, Any]) -> str:
    protocol = summary.get("protocol", {})
    amplitudes = protocol.get("amplitudes_mm")
    if isinstance(amplitudes, list) and amplitudes:
        peak_mm = max(float(value) for value in amplitudes)
    else:
        peak_mm = float(protocol.get("monotonic_tip_mm", 0.0))
    label = f"{peak_mm:.3f}".rstrip("0").rstrip(".")
    return label.replace(".", "p") + "mm"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Summarize the local host-concrete neighborhood around the promoted embedded steel trace."
    )
    parser.add_argument(
        "--steel-audit-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_continuum_promoted_cyclic_30mm_audit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_continuum_host_bar_locality_promoted_30mm_audit",
    )
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
    parser.add_argument("--skip-figure-export", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def clean_number(value: float) -> float | None:
    return None if not math.isfinite(value) else value


def relative_error(lhs: float, rhs: float, floor: float = 1.0e-12) -> float:
    scale = max(abs(rhs), floor)
    return abs(lhs - rhs) / scale


def save_figure(fig: plt.Figure, path: Path, secondary: Path | None) -> None:
    ensure_dir(path.parent)
    fig.savefig(path)
    if secondary is not None:
        ensure_dir(secondary.parent)
        fig.savefig(secondary)
    plt.close(fig)


def max_or_zero(values: list[float]) -> float:
    return max(values) if values else 0.0


def rms(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else 0.0


def summarize_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strain_errors = [
        relative_error(
            float(row["axial_strain"]),
            float(row["nearest_host_axial_strain"]),
            floor=1.0e-9,
        )
        for row in rows
    ]
    stress_errors = [
        relative_error(
            float(row["stress_xx_MPa"]),
            float(row["nearest_host_axial_stress_MPa"]),
            floor=1.0e-6,
        )
        for row in rows
    ]
    drifts = [float(row["drift_m"]) for row in rows]
    crack_openings = [float(row["nearest_host_max_crack_opening"]) for row in rows]
    crack_counts = [int(row["nearest_host_num_cracks"]) for row in rows]
    distances = [float(row["nearest_host_gp_distance_m"]) for row in rows]

    first_cracked_runtime_step = next(
        (
            int(row["runtime_step"])
            for row in rows
            if int(row["nearest_host_num_cracks"]) > 0
            or float(row["nearest_host_max_crack_opening"]) > 0.0
        ),
        -1,
    )
    first_cracked_drift = next(
        (
            float(row["drift_m"])
            for row in rows
            if int(row["nearest_host_num_cracks"]) > 0
            or float(row["nearest_host_max_crack_opening"]) > 0.0
        ),
        0.0,
    )

    return {
        "max_rel_bar_vs_host_axial_strain_error": max_or_zero(strain_errors),
        "rms_rel_bar_vs_host_axial_strain_error": rms(strain_errors),
        "max_rel_bar_vs_host_axial_stress_error": max_or_zero(stress_errors),
        "rms_rel_bar_vs_host_axial_stress_error": rms(stress_errors),
        "max_abs_host_crack_opening": max_or_zero(crack_openings),
        "peak_host_crack_count": max(crack_counts) if crack_counts else 0,
        "max_nearest_host_gp_distance_m": max_or_zero(distances),
        "first_host_cracked_runtime_step": first_cracked_runtime_step,
        "first_host_cracked_drift_m": first_cracked_drift,
        "max_abs_drift_m": max((abs(v) for v in drifts), default=0.0),
    }


def plot_trace(
    hex_order: str,
    rows: list[dict[str, Any]],
    out_dirs: list[Path],
    suffix: str,
) -> None:
    drifts_mm = [1.0e3 * float(row["drift_m"]) for row in rows]
    bar_strain = [float(row["axial_strain"]) for row in rows]
    host_strain = [float(row["nearest_host_axial_strain"]) for row in rows]
    bar_stress = [float(row["stress_xx_MPa"]) for row in rows]
    host_stress = [float(row["nearest_host_axial_stress_MPa"]) for row in rows]
    crack_opening_mm = [
        1.0e3 * float(row["nearest_host_max_crack_opening"]) for row in rows
    ]
    crack_count = [float(row["nearest_host_num_cracks"]) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.0), sharex=True)

    axes[0].plot(drifts_mm, bar_strain, color=BLUE, label="Embedded bar")
    axes[0].plot(
        drifts_mm,
        host_strain,
        color=ORANGE,
        linestyle="--",
        label="Nearest host GP",
    )
    axes[0].set_ylabel("Axial strain")
    axes[0].set_title(
        f"Continuum host-vs-bar locality audit ({hex_order.upper()}, {suffix})"
    )
    axes[0].legend(loc="best")

    axes[1].plot(drifts_mm, bar_stress, color=BLUE, label="Embedded bar")
    axes[1].plot(
        drifts_mm,
        host_stress,
        color=ORANGE,
        linestyle="--",
        label="Nearest host GP",
    )
    axes[1].set_ylabel("Axial stress [MPa]")
    axes[1].legend(loc="best")

    axes[2].plot(
        drifts_mm,
        crack_opening_mm,
        color=GREEN,
        label="Nearest host crack opening",
    )
    ax2 = axes[2].twinx()
    ax2.plot(
        drifts_mm,
        crack_count,
        color=ORANGE,
        linestyle="--",
        label="Nearest host crack count",
    )
    axes[2].set_ylabel("Crack opening [mm]")
    ax2.set_ylabel("Crack count")
    axes[2].set_xlabel("Tip drift [mm]")

    lines_1, labels_1 = axes[2].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    primary = (
        out_dirs[0]
        / f"reduced_rc_continuum_host_bar_locality_{hex_order}_{suffix}.png"
    )
    secondary = out_dirs[1] / primary.name if len(out_dirs) > 1 else None
    save_figure(fig, primary, secondary)


def main() -> int:
    args = parse_args()
    ensure_dir(args.output_dir)

    summary_path = args.steel_audit_dir / "structural_continuum_steel_hysteresis_summary.json"
    summary = read_json(summary_path)
    suffix = amplitude_suffix_from_summary(summary)
    continuum_cases = summary.get("continuum_cases", {})
    if not isinstance(continuum_cases, dict) or not continuum_cases:
        raise RuntimeError("Steel hysteresis summary does not contain continuum cases.")

    figure_dirs = [] if args.skip_figure_export else [args.figures_dir, args.secondary_figures_dir]
    payload: dict[str, Any] = {
        "source_summary": str(summary_path),
        "hex_orders": {},
    }

    for hex_order, case_payload in continuum_cases.items():
        case_dir = args.steel_audit_dir / hex_order
        rows = read_csv_rows(case_dir / "selected_continuum_steel_trace.csv")
        metrics = summarize_trace(rows)
        payload["hex_orders"][hex_order] = {
            **metrics,
            "selected_trace_csv": str(case_dir / "selected_continuum_steel_trace.csv"),
        }
        if figure_dirs:
            plot_trace(hex_order, rows, figure_dirs, suffix)

    write_json(args.output_dir / "continuum_host_bar_locality_summary.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
