#!/usr/bin/env python3
"""Plot selected base-column force response for the L-shaped benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path, label: str) -> dict | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = [
            {k.strip(): float(v) for k, v in row.items() if k is not None and v not in (None, "")}
            for row in csv.DictReader(f)
        ]
    if not rows:
        return None
    comps = [c for c in rows[0] if c.startswith("f")]
    return {
        "label": label,
        "time": [r["time"] for r in rows],
        "components": {c: [r[c] for r in rows] for c in comps},
    }


def summarize(case: dict, keep: list[str]) -> dict:
    out = {"label": case["label"], "samples": len(case["time"])}
    for comp in keep:
        vals = case["components"].get(comp, [])
        out[f"peak_abs_{comp}"] = max((abs(v) for v in vals), default=0.0)
        out[f"final_{comp}"] = vals[-1] if vals else 0.0
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opensees-nodal",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_elastic_timoshenko_sub1_nodalmass_current/selected_element_1_force.csv"),
    )
    parser.add_argument(
        "--opensees-element",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_elastic_timoshenko_sub1_elementmass_lumped/selected_element_1_force.csv"),
    )
    parser.add_argument(
        "--opensees-force-shear",
        default=str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_force_shear_sub1_nodalmass/selected_element_1_force.csv"),
    )
    parser.add_argument(
        "--falln",
        default=str(root / "data/output/lshaped_multiscale_16/recorders/selected_element_0_global_force.csv"),
    )
    parser.add_argument("--label-opensees-nodal", default="OpenSees ElasticTimoshenko nodal mass")
    parser.add_argument("--label-opensees-element", default="OpenSees ElasticTimoshenko element mass")
    parser.add_argument("--label-opensees-force-shear", default="OpenSees force+Vy/Vz element 1")
    parser.add_argument("--label-falln", default="fall_n element 0")
    parser.add_argument(
        "--out-dir",
        default=str(root / "doc/figures/validation_reboot"),
    )
    args = parser.parse_args()

    cases = [
        read_csv(Path(args.opensees_nodal), args.label_opensees_nodal),
        read_csv(Path(args.opensees_element), args.label_opensees_element),
        read_csv(Path(args.opensees_force_shear), args.label_opensees_force_shear),
        read_csv(Path(args.falln), args.label_falln),
    ]
    cases = [c for c in cases if c is not None]
    if not cases:
        raise SystemExit("No selected-element force histories found.")

    keep = ["f0", "f1", "f2", "f3", "f4", "f5"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif",
        "figure.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, axes = plt.subplots(3, 2, figsize=(9.0, 7.0), sharex=True)
    for ax, comp in zip(axes.ravel(), keep):
        for case in cases:
            vals = case["components"].get(comp)
            if vals is not None:
                ax.plot(case["time"], vals, linewidth=1.2, label=case["label"])
        ax.set_ylabel(comp)
    axes[-1, 0].set_xlabel("tiempo relativo [s]")
    axes[-1, 1].set_xlabel("tiempo relativo [s]")
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("Elemento seleccionado: componentes de fuerza reportadas")
    fig.tight_layout()
    fig.savefig(out_dir / "lshaped_16_selected_element_force_components.pdf")
    fig.savefig(out_dir / "lshaped_16_selected_element_force_components.png")
    plt.close(fig)

    summary = {
        "schema": "lshaped_16_selected_element_force_comparison_v1",
        "cases": [summarize(c, keep) for c in cases],
        "figure": str(out_dir / "lshaped_16_selected_element_force_components.pdf"),
        "note": "Components are recorder-reported element forces; use for amplitude/phase audit before local-force convention closure.",
    }
    (out_dir / "lshaped_16_selected_element_force_comparison_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
