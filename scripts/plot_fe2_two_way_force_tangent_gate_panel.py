#!/usr/bin/env python3
"""Panel de cierre del gate fuerza-tangente FE2 two-way de la columna.

Combina el resumen fe2_two_way_managed_xfem_force_tangent_gate_summary.json
con las historias staggered_residuals.csv de cada corrida para producir un
panel de tres vistas: residual mixto por iteracion escalonada, error de
momento de retorno y coste local (tiempo de pared e iteraciones no lineales).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def read_residual_history(run_dir: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with (run_dir / "staggered_residuals.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["iteration"]), float(row["relative_staggered_residual"])))
    rows.sort()
    return rows


def main() -> int:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=root
        / "doc/figures/validation_reboot/fe2_two_way_managed_xfem_force_tangent_gate_summary.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "doc/figures/validation_reboot",
    )
    parser.add_argument("--prefix", default="fe2_two_way_force_tangent_gate_panel")
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    cases = summary["cases"]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
    colors = ["tab:orange", "tab:blue"]
    markers = ["o", "s"]

    for case, color, marker in zip(cases, colors, markers):
        run_dir = root / Path(str(case["path"]).replace("\\", "/"))
        history = read_residual_history(run_dir)
        iters = [it for it, _ in history]
        resid = [r for _, r in history]
        axes[0].semilogy(iters, resid, marker=marker, color=color, label=case["label"])
        tol = float(case["label"].split("=")[1])
        axes[0].axhline(tol, color=color, linestyle="--", linewidth=0.9, alpha=0.6)
    axes[0].set_xlabel("iteracion escalonada")
    axes[0].set_ylabel("residual mixto relativo")
    axes[0].set_title("Residual fuerza-tangente")
    axes[0].set_xticks(range(1, 1 + max(int(c["iterations"]) for c in cases)))
    axes[0].legend(fontsize=8)

    labels = [c["label"] for c in cases]
    err = [100.0 * float(c["relative_moment_feedback_error"]) for c in cases]
    axes[1].bar(labels, err, color=colors, width=0.55)
    for x, e in enumerate(err):
        axes[1].text(x, e, f"{e:.2f}%", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("error de momento de retorno [%]")
    axes[1].set_title("Cierre macro-local de momento")

    wall = [float(c["local_elapsed_seconds"]) for c in cases]
    local_iters = [int(c["local_total_nonlinear_iterations"]) for c in cases]
    axes[2].bar(labels, wall, color=colors, width=0.55)
    for x, (w, it) in enumerate(zip(wall, local_iters)):
        axes[2].text(x, w, f"{w:.1f} s\n{it} iters", ha="center", va="bottom", fontsize=8)
    axes[2].set_ylabel("tiempo local de pared [s]")
    axes[2].set_ylim(0, max(wall) * 1.25)
    axes[2].set_title("Coste local acumulado")

    fig.suptitle(
        "FE2 two-way gestionado: gate fuerza-tangente sobre la columna "
        "(3x3x6, N=10 Lobatto, 200 mm)"
    )
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.out_dir / f"{args.prefix}.{ext}")
    plt.close(fig)

    for case in cases:
        print(
            f"{case['label']}: iters={case['iterations']}, "
            f"residual={case['final_residual']:.3e}, "
            f"err_momento={100.0 * case['relative_moment_feedback_error']:.2f}%, "
            f"wall={case['local_elapsed_seconds']:.1f}s"
        )
    print(f"figura: {args.out_dir / (args.prefix + '.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
