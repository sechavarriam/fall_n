#!/usr/bin/env python3
"""Publication summary for the FE2 Hex27 Ko-Bathe local-site run.

The plot intentionally keeps the comparison narrow: one promoted FE2 one-way
window, the corresponding single-scale fall_n nonlinear window, and optional
OpenSees if its recorder is available.  It also plots the local crack-plane
count so the VTK visibility policy can be audited against the structural
response without opening ParaView first.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            {k: float(v) for k, v in row.items() if k and v not in ("", None)}
            for row in csv.DictReader(f)
        ]


def read_roof_falln(path: Path, label: str, node: str) -> dict | None:
    if not path.exists():
        return None
    rows = read_rows(path)
    cols = rows[0].keys() if rows else []
    triplet = (f"{node}_dof0", f"{node}_dof1", f"{node}_dof2")
    if not rows or triplet[0] not in cols:
        # Fall back to the last recorded roof-node triplet.
        names = [c for c in cols if c != "time"]
        if len(names) < 3:
            return None
        triplet = tuple(names[-3:])  # type: ignore[assignment]
    return {
        "label": label,
        "time": [r["time"] for r in rows],
        "ux": [r[triplet[0]] for r in rows],
        "uy": [r[triplet[1]] for r in rows],
        "uz": [r[triplet[2]] for r in rows],
        "columns": list(triplet),
    }


def read_roof_opensees(path: Path, label: str) -> dict | None:
    if not path.exists():
        return None
    rows = read_rows(path)
    if not rows:
        return None
    return {
        "label": label,
        "time": [r["time"] for r in rows],
        "ux": [r["ux"] for r in rows],
        "uy": [r["uy"] for r in rows],
        "uz": [r["uz"] for r in rows],
        "columns": ["ux", "uy", "uz"],
    }


def clip_case(case: dict, t0: float, t1: float) -> dict:
    keep = [i for i, t in enumerate(case["time"]) if t0 <= t <= t1]
    return {
        **case,
        "time": [case["time"][i] for i in keep],
        "ux": [case["ux"][i] for i in keep],
        "uy": [case["uy"][i] for i in keep],
        "uz": [case["uz"][i] for i in keep],
    }


def read_force(path: Path, label: str, t0: float, t1: float) -> dict | None:
    if not path.exists():
        return None
    rows = [r for r in read_rows(path) if t0 <= r["time"] <= t1]
    if not rows:
        return None
    comps = [f"f{i}" for i in range(6) if f"f{i}" in rows[0]]
    return {
        "label": label,
        "time": [r["time"] for r in rows],
        "components": {c: [r[c] for r in rows] for c in comps},
    }


def summarize_roof(case: dict) -> dict:
    out = {"label": case["label"], "samples": len(case["time"])}
    for comp in ("ux", "uy", "uz"):
        vals = case[comp]
        out[f"peak_abs_{comp}_m"] = max((abs(v) for v in vals), default=0.0)
        out[f"final_{comp}_m"] = vals[-1] if vals else 0.0
    out["columns"] = case.get("columns", [])
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fe2-root",
        default=str(root / "data/output/fe2_hex27_oneway_scale1_restart_5p70_publication_persistent_cracks"),
    )
    parser.add_argument(
        "--falln-roof",
        default=str(root / "data/output/stage_c_16storey/falln_n4_full_nonlinear_primary_nodal_10s_roof_displacement.csv"),
    )
    parser.add_argument(
        "--falln-force",
        default=str(root / "data/output/stage_c_16storey/falln_n4_full_nonlinear_primary_nodal_10s_selected_element_0_global_force.csv"),
    )
    parser.add_argument(
        "--opensees-roof",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_disp_falln_proxy_axisfixed/roof_displacement.csv"),
    )
    parser.add_argument("--roof-node", default="node335")
    parser.add_argument("--out-dir", default=str(root / "doc/figures/validation_reboot"))
    args = parser.parse_args()

    fe2_root = Path(args.fe2_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fe2_roof = read_roof_falln(
        fe2_root / "recorders/roof_displacement.csv",
        "fall_n FE2 one-way Hex27 Ko-Bathe",
        args.roof_node,
    )
    if fe2_roof is None:
        raise SystemExit("FE2 roof recorder not found.")
    t0 = min(fe2_roof["time"])
    t1 = max(fe2_roof["time"])

    cases = [
        fe2_roof,
        read_roof_falln(Path(args.falln_roof), "fall_n full nonlinear", args.roof_node),
        read_roof_opensees(Path(args.opensees_roof), "OpenSees dispBeam proxy"),
    ]
    cases = [clip_case(c, t0, t1) for c in cases if c is not None]
    cases = [c for c in cases if c["time"]]

    force_cases = [
        read_force(
            fe2_root / "recorders/selected_element_0_global_force.csv",
            "FE2 macro element 159",
            t0,
            t1,
        ),
        read_force(Path(args.falln_force), "fall_n element 0", t0, t1),
    ]
    force_cases = [c for c in force_cases if c is not None]

    cracks = read_rows(fe2_root / "recorders/crack_evolution.csv")

    plt.rcParams.update({
        "font.family": "serif",
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8))
    ax = axes[0, 0]
    for case in cases:
        ax.plot(case["time"], case["ux"], linewidth=1.3, label=case["label"])
    ax.set_ylabel(r"$u_x$ cubierta [m]")
    ax.legend(loc="best", fontsize=7)

    ax = axes[0, 1]
    for case in cases:
        ax.plot(case["ux"], case["uy"], linewidth=1.3, label=case["label"])
    ax.set_xlabel(r"$u_x$ [m]")
    ax.set_ylabel(r"$u_y$ [m]")
    ax.set_title("Orbita de cubierta")

    ax = axes[1, 0]
    for case in cases:
        ax.plot(case["time"], case["uy"], linewidth=1.3, label=case["label"])
    ax.set_xlabel("tiempo relativo [s]")
    ax.set_ylabel(r"$u_y$ cubierta [m]")

    ax = axes[1, 1]
    ax.plot(
        [r["time"] for r in cracks],
        [r["total_cracks"] for r in cracks],
        color="tab:red",
        linewidth=1.5,
        label="planos visibles",
    )
    ax2 = ax.twinx()
    ax2.plot(
        [r["time"] for r in cracks],
        [1000.0 * r["max_opening"] for r in cracks],
        color="tab:blue",
        linewidth=1.1,
        label="apertura max.",
    )
    ax.set_xlabel("tiempo relativo [s]")
    ax.set_ylabel("planos de fisura visibles")
    ax2.set_ylabel("apertura max. [mm]")
    ax.set_title("Evolucion local Ko-Bathe Hex27")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=7)

    fig.suptitle("FE2 one-way Hex27 Ko-Bathe: respuesta global y observabilidad local")
    fig.text(
        0.5,
        0.01,
        "Nota: esta corrida usa reinicio desde alarma lineal; la comparacion global es diagnostica, no cierre dinamico promovido.",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "lshaped_16_fe2_hex27_oneway_publication_summary.pdf")
    fig.savefig(out_dir / "lshaped_16_fe2_hex27_oneway_publication_summary.png")
    plt.close(fig)

    if force_cases:
        fig, axes = plt.subplots(3, 2, figsize=(10.0, 7.2), sharex=True)
        for ax, comp in zip(axes.ravel(), [f"f{i}" for i in range(6)]):
            for case in force_cases:
                vals = case["components"].get(comp)
                if vals:
                    ax.plot(case["time"], vals, linewidth=1.1, label=case["label"])
            ax.set_ylabel(comp)
        axes[-1, 0].set_xlabel("tiempo relativo [s]")
        axes[-1, 1].set_xlabel("tiempo relativo [s]")
        axes[0, 0].legend(loc="best", fontsize=7)
        fig.suptitle("Elemento seleccionado: componentes de fuerza")
        fig.tight_layout()
        fig.savefig(out_dir / "lshaped_16_fe2_hex27_oneway_force_components.pdf")
        fig.savefig(out_dir / "lshaped_16_fe2_hex27_oneway_force_components.png")
        plt.close(fig)

    summary = {
        "schema": "fe2_hex27_oneway_publication_summary_v1",
        "fe2_root": str(fe2_root),
        "time_window": [t0, t1],
        "roof_cases": [summarize_roof(c) for c in cases],
        "final_visible_cracks": int(cracks[-1]["total_cracks"]) if cracks else 0,
        "peak_visible_cracks": int(max((r["total_cracks"] for r in cracks), default=0)),
        "peak_current_opening_mm": 1000.0 * max((r["max_opening"] for r in cracks), default=0.0),
        "comparison_warning": (
            "The plotted FE2 case uses restart-from-linear-alarm; use it as a "
            "VTK/local-observability diagnostic until a full-nonlinear "
            "checkpoint at first activation is available."
        ),
        "figures": [
            str(out_dir / "lshaped_16_fe2_hex27_oneway_publication_summary.pdf"),
            str(out_dir / "lshaped_16_fe2_hex27_oneway_force_components.pdf"),
        ],
    }
    (out_dir / "lshaped_16_fe2_hex27_oneway_publication_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
