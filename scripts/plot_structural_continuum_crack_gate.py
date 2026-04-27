#!/usr/bin/env python3
"""Plot crack-localization diagnostics for the structural-continuum gate.

The global hysteresis alone is not enough to judge the reduced RC continuum
candidate. A smeared host can preserve bar kinematics and still be physically
wrong if cracking spreads through the whole mesh instead of forming a
fixed-end hinge. This script turns the exported crack_state.csv into a compact
gate figure for the validation notes.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_BUNDLE = Path(
    "data/output/cyclic_validation/"
    "reboot_structural_continuum_cyclic_crack_band_4x4x4_200mm_"
    "clamped_dirichlet_composite_audit/hex8"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot crack gate diagnostics for the continuum RC column."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/figures/validation_reboot"),
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(fh)]


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    crack_state = args.bundle / "crack_state.csv"
    rows = read_rows(crack_state)
    if not rows:
        raise RuntimeError(f"No crack-state rows found in {crack_state}")

    drift_mm = [1000.0 * row["runtime_p"] * 0.2 for row in rows]
    cracked = [
        row["cracked_gauss_point_count"] / row["gauss_point_count"] for row in rows
    ]
    open_cracked = [
        row["open_cracked_gauss_point_count"] / row["gauss_point_count"]
        for row in rows
    ]
    opening_mm = [1000.0 * row["max_crack_opening"] for row in rows]

    first_full_row = next(
        (row for row in rows if row["cracked_gauss_point_count"] == row["gauss_point_count"]),
        None,
    )
    first_open_full_row = next(
        (
            row
            for row in rows
            if row["open_cracked_gauss_point_count"] == row["gauss_point_count"]
        ),
        None,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].plot(drift_mm, cracked, label="cracked GPs", color="#b91c1c")
    axes[0].plot(drift_mm, open_cracked, label="open cracked GPs", color="#f97316")
    axes[0].set_xlabel("equivalent imposed drift [mm]")
    axes[0].set_ylabel("fraction of host Gauss points")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(drift_mm, opening_mm, color="#2563eb")
    axes[1].set_xlabel("equivalent imposed drift [mm]")
    axes[1].set_ylabel("max crack opening [mm]")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Continuum crack-localization gate at 200 mm")
    fig.tight_layout()

    outputs: dict[str, str] = {}
    for ext in ("png", "pdf"):
        out = args.output_dir / f"structural_continuum_crack_gate_200mm.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=250 if ext == "png" else None)
        outputs[ext] = str(out)
    plt.close(fig)

    summary = {
        "status": "completed",
        "bundle": str(args.bundle),
        "crack_state_csv": str(crack_state),
        "figures": outputs,
        "record_count": len(rows),
        "max_cracked_fraction": max(cracked),
        "max_open_cracked_fraction": max(open_cracked),
        "max_crack_opening_mm": max(opening_mm),
        "first_full_cracking_runtime_p": (
            first_full_row["runtime_p"] if first_full_row is not None else None
        ),
        "first_full_cracking_equivalent_drift_mm": (
            1000.0 * first_full_row["runtime_p"] * 0.2
            if first_full_row is not None
            else None
        ),
        "first_full_open_cracking_runtime_p": (
            first_open_full_row["runtime_p"] if first_open_full_row is not None else None
        ),
        "first_full_open_cracking_equivalent_drift_mm": (
            1000.0 * first_open_full_row["runtime_p"] * 0.2
            if first_open_full_row is not None
            else None
        ),
        "diagnosis": (
            "The promoted 200 mm continuum gate preserves axial balance and "
            "steel transfer, but the host crack field is fully smeared early "
            "in the protocol. The next model-form target is objective hinge "
            "localization and crack shear-transfer, not embedded-bar kinematics."
        ),
    }
    summary_path = args.output_dir / "structural_continuum_crack_gate_200mm_summary.json"
    write_summary(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
