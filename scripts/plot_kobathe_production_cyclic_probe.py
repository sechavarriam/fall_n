#!/usr/bin/env python3
"""Replot the Ko-Bathe production cyclic probe hysteresis from recorded CSV.

Post-processing only: reads ``hysteresis.csv`` from an existing Ko-Bathe
cyclic probe run directory and regenerates the publication figure with the
same stem used by chapter 9a. No solver is executed.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=repo
        / "data/output/fe2_validation/"
        "kobathe_production_hex8_1x1x2_rebar_cyclic_50mm_fixedend_bias2",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo / "doc/figures/validation_reboot",
    )
    parser.add_argument(
        "--basename",
        default="kobathe_production_hex8_1x1x2_rebar_cyclic_50mm_fixedend_bias2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, float]] = []
    with (args.run_dir / "hysteresis.csv").open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            try:
                rows.append(
                    {
                        "drift_m": float(raw["drift_m"]),
                        "base_shear_MN": float(raw["base_shear_MN"]),
                    }
                )
            except (TypeError, ValueError):
                continue
    if not rows:
        raise SystemExit("hysteresis.csv is empty; nothing to plot.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(
        [1000.0 * row["drift_m"] for row in rows],
        [row["base_shear_MN"] for row in rows],
        color="#0f766e",
        linewidth=1.3,
    )
    ax.set_xlabel("Desplazamiento en el extremo [mm]")
    ax.set_ylabel("Cortante basal [MN]")
    ax.set_title("Sonda continua Ko-Bathe: 1x1x2 cíclica a 50 mm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = args.figures_dir / f"{args.basename}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)

    peak_drift_mm = 1000.0 * max(abs(row["drift_m"]) for row in rows)
    peak_shear_mn = max(abs(row["base_shear_MN"]) for row in rows)
    if not (math.isfinite(peak_drift_mm) and math.isfinite(peak_shear_mn)):
        raise SystemExit("non-finite extrema in hysteresis data")
    print(f"peak |drift| = {peak_drift_mm:.1f} mm; peak |V| = {peak_shear_mn:.3f} MN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
