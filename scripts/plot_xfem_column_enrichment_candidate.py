#!/usr/bin/env python3
"""Plot the first XFEM local-model contract for the RC-column benchmark.

The current promoted continuum benchmark still uses smeared crack-band damage.
This small figure keeps the XFEM route explicit: a base crack should enrich
only the cut cells and their nodes, preserving a cheap standard-continuum region
away from the plastic hinge.  The geometry mirrors the 3D mask test in
``tests/test_xfem_enrichment.cpp`` but is intentionally wrapper-side so it can
be regenerated with the rest of the validation figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a lightweight XFEM enrichment candidate figure."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/figures/validation_reboot"),
    )
    parser.add_argument("--crack-z", type=float, default=0.25)
    return parser.parse_args()


def element_is_cut(z_values: tuple[float, ...], crack_z: float) -> bool:
    signed = [z - crack_z for z in z_values]
    return min(signed) <= 0.0 <= max(signed) and min(signed) != max(signed)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # A two-cell column strip shown in x-z projection. The y direction is
    # suppressed in the figure but kept conceptually by the node multiplicity.
    nodes = {
        0: (-1.0, 0.0),
        1: (1.0, 0.0),
        2: (-1.0, 1.0),
        3: (1.0, 1.0),
        4: (-1.0, 2.0),
        5: (1.0, 2.0),
    }
    elements = {
        "base-cell": (0, 1, 3, 2),
        "upper-cell": (2, 3, 5, 4),
    }
    cut_elements = {
        label: element_is_cut(tuple(nodes[node_id][1] for node_id in conn), args.crack_z)
        for label, conn in elements.items()
    }
    enriched_nodes = {
        node_id: any(cut_elements[label] and node_id in conn for label, conn in elements.items())
        for node_id in nodes
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    for label, conn in elements.items():
        polygon = Polygon(
            [nodes[node_id] for node_id in conn],
            closed=True,
            facecolor="#fde68a" if cut_elements[label] else "#e5e7eb",
            edgecolor="#111827",
            linewidth=1.2,
            alpha=0.90,
        )
        ax.add_patch(polygon)
        cx = sum(nodes[node_id][0] for node_id in conn) / len(conn)
        cz = sum(nodes[node_id][1] for node_id in conn) / len(conn)
        ax.text(
            cx,
            cz,
            "cut + enriched" if cut_elements[label] else "standard",
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.axhline(args.crack_z, color="#dc2626", linestyle="--", linewidth=2.0)
    ax.text(
        1.08,
        args.crack_z,
        "crack level set",
        color="#991b1b",
        va="center",
        fontsize=9,
    )

    for node_id, (x, z) in nodes.items():
        enriched = enriched_nodes[node_id]
        ax.scatter(
            [x],
            [z],
            s=72,
            color="#2563eb" if enriched else "#6b7280",
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        ax.text(x, z + 0.06, f"N{node_id}", ha="center", fontsize=8)

    ax.scatter([], [], s=72, color="#2563eb", label="enriched node")
    ax.scatter([], [], s=72, color="#6b7280", label="standard node")
    ax.set_title("XFEM base-crack enrichment candidate")
    ax.set_xlabel("section coordinate x")
    ax.set_ylabel("column coordinate z")
    ax.set_xlim(-1.35, 1.55)
    ax.set_ylim(-0.15, 2.25)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.20)
    ax.legend(loc="upper left", fontsize=8)

    png = args.output_dir / "xfem_column_base_crack_candidate.png"
    pdf = args.output_dir / "xfem_column_base_crack_candidate.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=250)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status": "completed",
        "crack_z": args.crack_z,
        "element_count": len(elements),
        "cut_element_count": sum(1 for value in cut_elements.values() if value),
        "enriched_node_count": sum(1 for value in enriched_nodes.values() if value),
        "standard_node_count": sum(1 for value in enriched_nodes.values() if not value),
        "figures": {"png": str(png), "pdf": str(pdf)},
        "notes": (
            "This is a wrapper-side visualization of the XFEM mask contract: "
            "only crack-cut cells should receive enriched displacement support."
        ),
    }
    (args.output_dir / "xfem_column_base_crack_candidate_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
