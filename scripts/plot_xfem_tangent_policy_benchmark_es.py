#!/usr/bin/env python3
"""Regenera el benchmark de política de tangente cohesiva XFEM en español.

Lee el JSON de resumen superviviente
``doc/figures/validation_reboot/xfem_promoted_bounded_dowel_tangent_policy_benchmark_200mm_summary.json``
(mismos escalares que la figura original en inglés) y NO ejecuta ningún solver.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=repo
        / "doc/figures/validation_reboot/"
          "xfem_promoted_bounded_dowel_tangent_policy_benchmark_200mm_summary.json",
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
        default="xfem_promoted_bounded_dowel_tangent_policy_benchmark_200mm",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    cases = summary["cases"]
    labels = [case["cohesive_tangent"] for case in cases]
    wall_seconds = [case["wall_seconds"] for case in cases]
    newton_iterations = [case["total_newton_iterations"] for case in cases]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
        }
    )
    colors = ["#2563eb", "#0f766e", "#b91c1c"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))

    axes[0].bar(labels, wall_seconds, width=0.66, color=colors)
    axes[0].set_title("Tiempo de ejecución")
    axes[0].set_ylabel("Tiempo de pared [s]")

    axes[1].bar(labels, newton_iterations, width=0.66, color=colors)
    axes[1].set_title("Trabajo no lineal")
    axes[1].set_ylabel("Iteraciones de Newton totales")

    fig.suptitle(
        "Benchmark de política de tangente cohesiva: XFEM promovido con dowel acotado",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

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
