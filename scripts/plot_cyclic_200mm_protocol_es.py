#!/usr/bin/env python3
"""Regenera la figura del protocolo cíclico de ±200 mm con rótulos en español.

Lee ``comparison_protocol.csv`` del bundle de publicación existente (mismos
datos que la figura original en inglés) y NO ejecuta ningún solver.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-csv",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/"
          "reboot_external_benchmark_cyclic_200mm_publication_20260511/"
          "fall_n/comparison_protocol.csv",
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
        "--stem",
        default="cyclic_200mm_protocol_20260509",
        help="Mismo stem que la figura original para que sync_thesis_figures_9a la recoja.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.protocol_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    p = [float(row["p"]) for row in rows]
    drift_mm = [1000.0 * float(row["target_drift_m"]) for row in rows]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
        }
    )
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.plot(p, drift_mm, color="#101828", lw=2.6, zorder=2)
    ax.plot(
        p,
        drift_mm,
        linestyle="none",
        marker="o",
        markersize=6,
        color="#2563eb",
        zorder=3,
    )
    ax.axhline(0.0, color="#6b7280", lw=1.0, zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Coordenada normalizada del protocolo p [-]")
    ax.set_ylabel("Deriva lateral objetivo [mm]")
    ax.set_title("Protocolo cíclico de auditoría de la columna hasta ±200 mm")
    fig.tight_layout()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = args.figures_dir / f"{args.stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        shutil.copy2(out, args.secondary_figures_dir / out.name)
        print(f"[ok] {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
