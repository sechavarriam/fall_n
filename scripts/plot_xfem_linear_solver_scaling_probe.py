#!/usr/bin/env python3
"""Compare LU and ASM-linearized XFEM scaling probes.

The script intentionally keeps the comparison narrow: same XFEM physical model,
same mesh and cyclic protocol, only the PETSc linear solve profile changes.
This makes the resulting figure an algorithmic scaling diagnostic rather than a
new physical validation claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Plot 3x3x8 XFEM LU vs FGMRES+ASM scaling probes."
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
        "--basename",
        default="xfem_ccb_bounded_dowelx_3x3x8_linear_solver_probe",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[
            "25mm_lu:data/output/cyclic_validation/xfem_ccb_bounded_dowelx_3x3x8_l2_25mm_probe",
            "25mm_asm:data/output/cyclic_validation/xfem_ccb_bounded_dowelx_3x3x8_l2_fgmres_asm_25mm_probe",
            "50mm_lu:data/output/cyclic_validation/xfem_ccb_bounded_dowelx_3x3x8_l2_50mm_probe",
            "50mm_asm:data/output/cyclic_validation/xfem_ccb_bounded_dowelx_3x3x8_l2_fgmres_asm_50mm_probe",
        ],
        help="Label:path pair. Can be repeated.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, float | str]]:
    def coerce(value: str) -> float | str:
        if value == "":
            return 0.0
        try:
            return float(value)
        except ValueError:
            return value

    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_case(spec: str, repo: Path) -> dict[str, Any]:
    label, raw_path = spec.split(":", maxsplit=1)
    root = (repo / raw_path).resolve()
    manifest = json.loads(
        (root / "global_xfem_newton_manifest.json").read_text(encoding="utf-8")
    )
    rows = read_csv(root / "global_xfem_newton_hysteresis.csv")
    solve = manifest.get("solve_control", {})
    obs = manifest.get("observables", {})
    timing = manifest.get("timing", {})
    return {
        "label": label,
        "root": str(root),
        "rows": rows,
        "completed_successfully": manifest.get("completed_successfully", False),
        "solver_profile": solve.get("solver_profile", ""),
        "wall_seconds": float(timing.get("total_wall_seconds", 0.0)),
        "iterations": int(solve.get("total_nonlinear_iterations", 0)),
        "failed_attempts": int(solve.get("total_failed_attempts", 0)),
        "max_bisection_level": int(solve.get("max_bisection_level", 0)),
        "peak_base_shear_mn": float(obs.get("peak_abs_base_shear_mn", 0.0)),
        "peak_steel_mpa": float(obs.get("peak_abs_steel_stress_mpa", 0.0)),
    }


def save_figure(fig: Any, figures_dir: Path, secondary_dir: Path, stem: str) -> dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for extension in ("png", "pdf"):
        path = figures_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight")
        shutil.copy2(path, secondary_dir / path.name)
        artifacts[extension] = str(path)
        artifacts[f"secondary_{extension}"] = str(secondary_dir / path.name)
    return artifacts


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    cases = [read_case(spec, repo) for spec in args.case]
    by_amplitude: dict[str, list[dict[str, Any]]] = {"25mm": [], "50mm": []}
    for case in cases:
        amplitude = case["label"].split("_", maxsplit=1)[0]
        by_amplitude.setdefault(amplitude, []).append(case)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "savefig.bbox": "tight",
        }
    )

    colors = {"lu": "#111827", "asm": "#0f766e"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4))
    for col, amplitude in enumerate(("25mm", "50mm")):
        for case in by_amplitude[amplitude]:
            solver_key = case["label"].split("_", maxsplit=1)[1]
            axes[0, col].plot(
                [row["drift_mm"] for row in case["rows"]],
                [1000.0 * row["base_shear_MN"] for row in case["rows"]],
                color=colors.get(solver_key, "#334155"),
                lw=1.4,
                marker="o",
                ms=2.6,
                label=f"{solver_key.upper()} ({case['wall_seconds']:.1f}s)",
            )
        axes[0, col].set_title(f"3x3x8 cyclic probe to {amplitude}")
        axes[0, col].set_xlabel("Tip drift [mm]")
        axes[0, col].set_ylabel("Base shear [kN]")
        axes[0, col].legend(fontsize=8)

        labels = [case["label"].split("_", maxsplit=1)[1].upper() for case in by_amplitude[amplitude]]
        times = [case["wall_seconds"] for case in by_amplitude[amplitude]]
        iterations = [case["iterations"] for case in by_amplitude[amplitude]]
        x = range(len(labels))
        axes[1, col].bar(
            [value - 0.18 for value in x],
            times,
            width=0.36,
            color="#2563eb",
            label="wall [s]",
        )
        axes[1, col].bar(
            [value + 0.18 for value in x],
            iterations,
            width=0.36,
            color="#f97316",
            label="Newton iters",
        )
        axes[1, col].set_xticks(list(x), labels)
        axes[1, col].set_title(f"Cost read at {amplitude}")
        axes[1, col].legend(fontsize=8)

    fig.suptitle(
        "XFEM 3x3x8 PETSc linear-solver scaling probe\n"
        "Same physical model; only KSP/PC policy changes",
        y=1.02,
    )
    artifacts = save_figure(
        fig, args.figures_dir, args.secondary_figures_dir, args.basename
    )
    plt.close(fig)

    summary = {
        "scope": "xfem_3x3x8_linear_solver_scaling_probe",
        "status": "completed",
        "interpretation": (
            "FGMRES+ASM+subLU is stable on the 3x3x8 probes but is not faster "
            "than direct LU at 624 global DOFs. It remains a scalable candidate "
            "for larger meshes because it removes the monolithic global "
            "factorization; it is not promoted as the small-mesh default."
        ),
        "cases": [
            {key: value for key, value in case.items() if key != "rows"}
            for case in cases
        ],
        "artifacts": artifacts,
    }
    summary_path = args.figures_dir / f"{args.basename}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_path, args.secondary_figures_dir / summary_path.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
