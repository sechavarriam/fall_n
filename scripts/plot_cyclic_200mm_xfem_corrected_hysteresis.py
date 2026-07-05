#!/usr/bin/env python3
"""Figura de un solo panel para la auditoria XFEM corregida de 200 mm.

Reemplaza el panel triple del benchmark de politicas de solver (cuyos paneles
de coste y robustez no aportan informacion con un unico caso) por la
comparacion de histeresis entre la rama XFEM promovida y la referencia
estructural N=10 de Lobatto. Lee los CSV existentes de la corrida congelada,
sin re-ejecutar el solver, y escribe la salida con el MISMO stem que consume
el manifiesto de sincronizacion de figuras.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def read_csv_cols(path: Path, drift_key: str, shear_key: str,
                  drift_scale: float) -> tuple[list[float], list[float]]:
    drifts: list[float] = []
    shears: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            drifts.append(float(row[drift_key]) * drift_scale)
            shears.append(float(row[shear_key]) * 1000.0)  # MN -> kN
    return drifts, shears


def align_sign(ref_d: list[float], ref_v: list[float],
               d: list[float], v: list[float]) -> float:
    """Signo que maximiza la correlacion drift-cortante con la referencia."""
    corr_ref = sum(a * b for a, b in zip(ref_d, ref_v))
    corr = sum(a * b for a, b in zip(d, v))
    if corr_ref == 0.0 or corr == 0.0:
        return 1.0
    return 1.0 if (corr_ref > 0) == (corr > 0) else -1.0


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xfem-csv",
        type=Path,
        default=root
        / "data/output/cyclic_validation_200mm_rerun_20260509/xfem_corrected/newton_l2/global_xfem_newton_hysteresis.csv",
    )
    parser.add_argument(
        "--structural-csv",
        type=Path,
        default=root
        / "data/output/fe2_validation/structural_n10_lobatto_200mm_preflight/comparison_hysteresis.csv",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=root / "doc/figures/validation_reboot"
    )
    parser.add_argument("--stem", default="cyclic_200mm_xfem_corrected_20260509")
    args = parser.parse_args()

    xfem_d, xfem_v = read_csv_cols(args.xfem_csv, "drift_mm", "base_shear_MN", 1.0)
    str_d, str_v = read_csv_cols(
        args.structural_csv, "drift_m", "base_shear_MN", 1000.0
    )

    # Convencion fisica: deriva positiva -> cortante positivo. La rama XFEM
    # congelada ya la cumple; la referencia estructural registra la reaccion
    # con signo opuesto (sign_factor_applied_to_structural = -1 en el summary
    # congelado). La alineacion por correlacion cubre ambos casos.
    sign_xfem = 1.0 if sum(a * b for a, b in zip(xfem_d, xfem_v)) > 0 else -1.0
    xfem_v = [sign_xfem * v for v in xfem_v]
    sign_str = align_sign(xfem_d, xfem_v, str_d, str_v)
    str_v = [sign_str * v for v in str_v]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(str_d, str_v, color="black", linewidth=2.0,
            label="Referencia estructural $N{=}10$ Lobatto")
    ax.plot(xfem_d, xfem_v, color="tab:blue", linewidth=1.2,
            linestyle="-.", label="XFEM promovido (deslizamiento acotado)")
    ax.axhline(0.0, color="gray", linewidth=0.6)
    ax.axvline(0.0, color="gray", linewidth=0.6)
    ax.set_xlabel("Desplazamiento en el extremo [mm]")
    ax.set_ylabel("Cortante basal [kN]")
    ax.set_title(
        "Auditoría XFEM corregida hasta $\\pm 200\\,$mm frente a la "
        "referencia estructural"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"{args.stem}.{ext}")
    plt.close(fig)
    print(f"figura regenerada: {args.out_dir / (args.stem + '.pdf')}")
    print(f"signos aplicados: xfem={sign_xfem:+.0f}, estructural={sign_str:+.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
