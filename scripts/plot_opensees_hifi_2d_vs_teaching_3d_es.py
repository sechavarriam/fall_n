#!/usr/bin/env python3
"""Regenera el overlay OpenSees 2D hi-fi vs 3D docente con rótulos en español.

Lee los ``hysteresis.csv`` supervivientes de los bundles OpenSees existentes
(mismos datos que la figura original en inglés: RMS = 0.398 % del pico 2D)
y NO ejecuta ningún solver.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hifi-2d-csv",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/opensees_hifi_teaching_200mm/hysteresis.csv",
    )
    parser.add_argument(
        "--teaching-3d-csv",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/opensees_hifi_teaching_3d_200mm/hysteresis.csv",
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
    parser.add_argument("--stem", default="opensees_hifi_2d_vs_teaching_3d_hysteresis")
    return parser.parse_args()


def load(path: Path) -> tuple[list[float], list[float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    drift_mm = [1000.0 * float(row["drift_m"]) for row in rows]
    shear_kn = [1000.0 * float(row["base_shear_MN"]) for row in rows]
    return drift_mm, shear_kn


def main() -> int:
    args = parse_args()
    drift_2d, shear_2d = load(args.hifi_2d_csv)
    drift_3d, shear_3d = load(args.teaching_3d_csv)

    if len(shear_2d) != len(shear_3d):
        raise SystemExit("Los historiales no comparten la misma grilla de protocolo.")
    diffs = [a - b for a, b in zip(shear_2d, shear_3d)]
    peak_2d = max(abs(v) for v in shear_2d)
    rms_pct = 100.0 * math.sqrt(sum(d * d for d in diffs) / len(diffs)) / peak_2d

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.plot(drift_2d, shear_2d, color="black", lw=1.7, label="OpenSees 2D alta fidelidad")
    ax.plot(
        drift_3d,
        shear_3d,
        color="#1f77b4",
        lw=1.4,
        linestyle="--",
        label="OpenSees 3D docente",
    )
    ax.set_xlabel("Desplazamiento en el extremo [mm]")
    ax.set_ylabel("Cortante basal [kN]")
    ax.set_title("Auditoría de la referencia cíclica OpenSees de alta fidelidad")
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.03,
        0.05,
        f"RMS = {rms_pct:.3f}% del pico 2D",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#9ca3af", "boxstyle": "square,pad=0.35"},
    )
    fig.tight_layout()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = args.figures_dir / f"{args.stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        shutil.copy2(out, args.secondary_figures_dir / out.name)
        print(f"[ok] {out} (RMS={rms_pct:.4f}%)")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
