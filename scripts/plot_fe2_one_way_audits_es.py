#!/usr/bin/env python3
"""Regenera las tres figuras de auditoría FE2 one-way con rótulos en español.

Lee los JSON de resumen supervivientes en ``data/output/fe2_validation``
(los mismos escalares que las figuras originales en inglés) y NO ejecuta
ningún solver:

* ``fe2_one_way_managed_xfem_downscaling_audit_summary.json``
* ``fe2_one_way_managed_xfem_incremental_closure_summary.json``
* ``fe2_one_way_xfem_downscaling_mode_audit_summary.json``
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
        "--summaries-dir",
        type=Path,
        default=repo / "data/output/fe2_validation",
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
    return parser.parse_args()


def save(fig: plt.Figure, figures_dir: Path, secondary_dir: Path, stem: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = figures_dir / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        shutil.copy2(out, secondary_dir / out.name)
        print(f"[ok] {out}")
    plt.close(fig)


def plot_downscaling_audit(data: dict, figures_dir: Path, secondary_dir: Path) -> None:
    rows = data["rows"]
    meshes = [row["mesh"] for row in rows]
    m_hom = [row["M_hom"] for row in rows]
    converged = [bool(row["converged"]) for row in rows]
    macro = rows[0]["M_macro"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = list(range(len(rows)))
    ax.bar(x, m_hom, width=0.72, color="#2b6a99", label="XFEM local gestionado")
    ax.axhline(macro, color="black", lw=1.6, linestyle="--", label="Estructural N=10 Lobatto")
    for xi, value, ok in zip(x, m_hom, converged):
        if ok:
            ax.annotate(f"{value:.3f}", (xi, value), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=8)
        else:
            ax.annotate("div.", (xi, 0.0), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(meshes)
    ax.set_xlabel("Malla local XFEM")
    ax.set_ylabel(r"Pico $|M_y|$ [MN m]")
    ax.set_title("FE2 unidireccional: auditoría de reducción de escala del XFEM local gestionado")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save(fig, figures_dir, secondary_dir, "fe2_one_way_managed_xfem_downscaling_audit")


def plot_incremental_closure(data: dict, figures_dir: Path, secondary_dir: Path) -> None:
    macro = data["macro_moment_mn_m"]
    cases = data["cases"]
    label_map = {
        "single-step 3x3x6": "paso único 3x3x6",
        "incremental 3x3x6": "incremental 3x3x6",
        "incremental 5x5x10": "incremental 5x5x10",
    }
    labels = [label_map.get(case["label"], case["label"]) for case in cases]
    values = [case["homogenized_moment_mn_m"] for case in cases]
    errors = [case["relative_error"] for case in cases]

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    x = list(range(len(cases)))
    ax.bar(x, values, width=0.66, color="#5f96f0", label="XFEM local")
    ax.axhline(macro, color="black", lw=1.7, linestyle="--", label="macro estructural")
    for xi, value, err in zip(x, values, errors):
        ax.annotate(f"err={err:.1f}x", (xi, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"Pico $|M_y|$ [MN m]")
    ax.set_title("Replay FE2 unidireccional con XFEM gestionado, N=10 Lobatto, 200 mm")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, figures_dir, secondary_dir, "fe2_one_way_managed_xfem_incremental_closure")


def plot_downscaling_mode_audit(data: dict, figures_dir: Path, secondary_dir: Path) -> None:
    cases = data["cases"]
    label_map = {
        "kinematic incremental 3x3x6": "cinemática\nincremental\n3x3x6",
        "kinematic incremental 5x5x10": "cinemática\nincremental\n5x5x10",
        "dual resultant 3x3x6": "resultante\ndual\n3x3x6",
        "dual resultant 5x5x10": "resultante\ndual\n5x5x10",
    }
    labels = [label_map.get(case["label"], case["label"]) for case in cases]
    values = [case["absolute_homogenized_moment_mn_m"] for case in cases]
    errors = [case["relative_error"] for case in cases]
    macro = cases[0]["macro_moment_mn_m"]
    colors = ["#9db9d7", "#7397bd", "#f0b26b", "#e08f36"]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = list(range(len(cases)))
    ax.bar(x, values, width=0.7, color=colors[: len(cases)])
    ax.axhline(macro, color="black", lw=1.6, linestyle="--", label=f"macro {macro:.4f} MN m")
    for xi, value, err in zip(x, values, errors):
        text = f"e={err:.1f}" if err >= 0.1 else f"e={err:.5f}"
        ax.annotate(text, (xi, value), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$|M_{\mathrm{hom}}|$ [MN m]")
    ax.set_title("FE2 unidireccional con XFEM gestionado: reducción cinemática vs resultante dual")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    save(fig, figures_dir, secondary_dir, "fe2_one_way_xfem_downscaling_mode_audit")


def main() -> int:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )
    summaries = args.summaries_dir
    plot_downscaling_audit(
        json.loads((summaries / "fe2_one_way_managed_xfem_downscaling_audit_summary.json").read_text(encoding="utf-8")),
        args.figures_dir,
        args.secondary_figures_dir,
    )
    plot_incremental_closure(
        json.loads((summaries / "fe2_one_way_managed_xfem_incremental_closure_summary.json").read_text(encoding="utf-8")),
        args.figures_dir,
        args.secondary_figures_dir,
    )
    plot_downscaling_mode_audit(
        json.loads((summaries / "fe2_one_way_xfem_downscaling_mode_audit_summary.json").read_text(encoding="utf-8")),
        args.figures_dir,
        args.secondary_figures_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
