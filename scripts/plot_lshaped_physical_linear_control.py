#!/usr/bin/env python3
"""Plot the physical scale=1 linear-control L-shaped building response."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE = ROOT / "data" / "output" / "lshaped_16storey_physical_scale1_linear_control_20260509"


def read_roof(path: Path) -> tuple[list[float], dict[str, list[float]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        time: list[float] = []
        cols: dict[str, list[float]] = {name: [] for name in reader.fieldnames or [] if name != "time"}
        for row in reader:
            time.append(float(row["time"]))
            for name in cols:
                cols[name].append(float(row[name]))
    return time, cols


def mean_component(cols: dict[str, list[float]], suffix: str) -> list[float]:
    selected = [values for name, values in cols.items() if name.endswith(suffix)]
    if not selected:
        raise ValueError(f"No roof columns ending with {suffix}")
    return [sum(values[i] for values in selected) / len(selected) for i in range(len(selected[0]))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc" / "figures" / "validation_reboot")
    parser.add_argument("--prefix", default="lshaped_16_physical_scale1_linear_control")
    args = parser.parse_args()

    roof_csv = args.case_dir / "recorders" / "roof_displacement_newmark_linear_reference.csv"
    summary_json = args.case_dir / "recorders" / "newmark_linear_reference_summary.json"
    time, cols = read_roof(roof_csv)
    ux = mean_component(cols, "_dof0")
    uy = mean_component(cols, "_dof1")
    uz = mean_component(cols, "_dof2")
    peak = max(max(abs(v) for v in ux), max(abs(v) for v in uy), max(abs(v) for v in uz))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.2))
    ax = axes[0]
    ax.plot(time, ux, label=r"$\bar{u}_x$", lw=0.9)
    ax.plot(time, uy, label=r"$\bar{u}_y$", lw=0.9)
    ax.plot(time, uz, label=r"$\bar{u}_z$", lw=0.9)
    ax.set_ylabel("Roof displacement (m)")
    ax.grid(True, alpha=0.35, lw=0.3)
    ax.legend(ncol=3, loc="upper right")
    ax.set_title("L-shaped 16-storey building, MYG004 physical scale=1 linear-control response")

    orbit = axes[1]
    orbit.plot(ux, uy, color="#333333", lw=0.75)
    orbit.plot([ux[0]], [uy[0]], "o", color="green", ms=4, label="start")
    orbit.plot([ux[-1]], [uy[-1]], "s", color="red", ms=4, label="end")
    orbit.set_xlabel(r"$\bar{u}_x$ (m)")
    orbit.set_ylabel(r"$\bar{u}_y$ (m)")
    orbit.grid(True, alpha=0.35, lw=0.3)
    orbit.legend(loc="best")
    orbit.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    pdf = args.output_dir / f"{args.prefix}.pdf"
    png = args.output_dir / f"{args.prefix}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    source_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    summary = {
        "schema": "fall_n_lshaped_physical_scale1_linear_control_plot_v1",
        "case_dir": str(args.case_dir.relative_to(ROOT)),
        "roof_csv": str(roof_csv.relative_to(ROOT)),
        "source_summary": str(summary_json.relative_to(ROOT)),
        "eq_scale": source_summary.get("eq_scale"),
        "duration_s": source_summary.get("duration_s"),
        "steps": source_summary.get("steps"),
        "peak_abs_roof_centroid_component_m": peak,
        "source_peak_abs_roof_component_m": source_summary.get("peak_abs_roof_component_m"),
        "figures": [str(pdf.relative_to(ROOT)), str(png.relative_to(ROOT))],
    }
    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
