#!/usr/bin/env python3
"""Reproduce the OpenSeesPy hi-fi cyclic RC-column reference to 200 mm.

This is the publication recipe for the Timoshenko matrix benchmark used in
Chapter 9. It runs only the OpenSeesPy structural comparator, with the exact
configuration recovered from the validated matrix-experiment summary:

  * 2D RC cantilever, L = 3.2 m, 0.25 m x 0.25 m section
  * 20 dispBeamColumn elements
  * Gauss-Legendre integration, 5 integration points per element
  * PDelta transformation
  * Concrete02 and Steel02 mapping used by the cyclic diagnostic reference
  * axial preload of 0.02 MN
  * cyclic tip displacement amplitudes 50, 100, 150, and 200 mm

The nonlinear OpenSees model itself is delegated to
``opensees_reduced_rc_column_hifi_reference.py`` so the reference and the
published figure are generated from the same implementation used by the audit.
No fall_n executable is called here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run and plot the OpenSeesPy 200 mm hi-fi cyclic reference."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/opensees_hifi_timoshenko_matrix_200mm_publication_20260518",
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
        default="opensees_reduced_rc_200mm_hifi_hysteresis",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to run the OpenSeesPy reference script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun OpenSeesPy even if a completed manifest already exists.",
    )
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def read_hysteresis(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, float] = {}
            for key, value in raw.items():
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    val = math.nan
                row[key] = val
            rows.append(row)
    return rows


def extrema(rows: list[dict[str, float]]) -> dict[str, float]:
    drifts = [abs(row.get("drift_m", math.nan)) for row in rows]
    shears = [abs(row.get("base_shear_MN", math.nan)) for row in rows]
    return {
        "records": float(len(rows)),
        "max_abs_drift_mm": 1000.0 * max(drifts) if drifts else math.nan,
        "peak_abs_base_shear_kN": 1000.0 * max(shears) if shears else math.nan,
    }


def loop_work_mn_m(rows: list[dict[str, float]]) -> float:
    work = 0.0
    for lhs, rhs in zip(rows, rows[1:]):
        work += 0.5 * (
            lhs.get("base_shear_MN", 0.0) + rhs.get("base_shear_MN", 0.0)
        ) * (rhs.get("drift_m", 0.0) - lhs.get("drift_m", 0.0))
    return work


def completed_manifest(path: Path) -> bool:
    manifest = path / "reference_manifest.json"
    hysteresis = path / "hysteresis.csv"
    if not manifest.exists() or not hysteresis.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed"


def build_command(args: argparse.Namespace, repo: Path) -> list[str]:
    command = [
        args.python,
        str(repo / "scripts/opensees_reduced_rc_column_hifi_reference.py"),
        "--output-dir",
        str(args.output_dir),
        "--analysis",
        "cyclic",
        "--material-mode",
        "nonlinear",
        "--model-dimension",
        "2d",
        "--beam-element-family",
        "disp",
        "--beam-integration",
        "legendre",
        "--integration-points",
        "5",
        "--structural-element-count",
        "20",
        "--geom-transf",
        "pdelta",
        "--axial-compression-mn",
        "0.02",
        "--axial-preload-steps",
        "4",
        "--amplitudes-mm",
        "50,100,150,200",
        "--steps-per-segment",
        "32",
        "--reversal-substep-factor",
        "2",
        "--max-bisections",
        "10",
        "--mapping-policy",
        "cyclic-diagnostic",
        "--concrete-model",
        "Concrete02",
        "--concrete-lambda",
        "0.1",
        "--concrete-ft-ratio",
        "0.02",
        "--concrete-softening-multiplier",
        "0.5",
        "--concrete-unconfined-residual-ratio",
        "0.2",
        "--concrete-confined-residual-ratio",
        "0.2",
        "--concrete-ultimate-strain",
        "-0.006",
        "--steel-r0",
        "20.0",
        "--steel-cr1",
        "0.925",
        "--steel-cr2",
        "0.15",
        "--steel-a1",
        "0.0",
        "--steel-a2",
        "1.0",
        "--steel-a3",
        "0.0",
        "--steel-a4",
        "1.0",
        "--solver-profile-family",
        "pure-secant-disp",
    ]
    if args.print_progress:
        command.append("--print-progress")
    return command


def plot_reference(
    rows: list[dict[str, float]], args: argparse.Namespace
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(6.9, 4.6))
    ax.plot(
        [1000.0 * row["drift_m"] for row in rows],
        [1000.0 * row["base_shear_MN"] for row in rows],
        color="#d97706",
        linewidth=1.45,
        label="OpenSeesPy hi-fi structural",
    )
    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_xlabel("Tip displacement [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("OpenSeesPy hi-fi cyclic RC-column reference to 200 mm")
    ax.legend(loc="best")
    fig.tight_layout()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for ext in ("pdf", "png", "svg"):
        path = args.figures_dir / f"{args.basename}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        outputs[ext] = str(path)
        secondary = args.secondary_figures_dir / path.name
        shutil.copy2(path, secondary)
        outputs[f"thesis_{ext}"] = str(secondary)
    plt.close(fig)
    return outputs


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(args, repo)
    proc_payload: dict[str, Any] = {
        "command": command,
        "skipped_existing": False,
    }
    if args.force or not completed_manifest(args.output_dir):
        proc = subprocess.run(
            command,
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        (args.output_dir / "publication_stdout.log").write_text(
            proc.stdout, encoding="utf-8"
        )
        (args.output_dir / "publication_stderr.log").write_text(
            proc.stderr, encoding="utf-8"
        )
        proc_payload.update(
            {
                "return_code": proc.returncode,
                "stdout_log": str(args.output_dir / "publication_stdout.log"),
                "stderr_log": str(args.output_dir / "publication_stderr.log"),
            }
        )
        if proc.returncode != 0:
            print(json.dumps(proc_payload, indent=2))
            return proc.returncode
    else:
        proc_payload["skipped_existing"] = True

    hysteresis = args.output_dir / "hysteresis.csv"
    manifest = args.output_dir / "reference_manifest.json"
    rows = read_hysteresis(hysteresis)
    figures = plot_reference(rows, args)
    manifest_payload: dict[str, Any] = {}
    if manifest.exists():
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

    summary = {
        "status": "completed" if manifest_payload.get("status") == "completed" else "unknown",
        "scope": "opensees_reduced_rc_200mm_hifi_publication",
        "reference_dir": str(args.output_dir),
        "hysteresis_csv": str(hysteresis),
        "manifest": str(manifest),
        "configuration": {
            "model_dimension": "2d",
            "beam_element_family": "dispBeamColumn",
            "structural_element_count": 20,
            "beam_integration": "Gauss-Legendre",
            "integration_points": 5,
            "geom_transf": "PDelta",
            "axial_compression_mn": 0.02,
            "cyclic_amplitudes_mm": [50.0, 100.0, 150.0, 200.0],
            "steps_per_segment": 32,
            "reversal_substep_factor": 2,
            "max_bisections": 10,
            "concrete_model": "Concrete02",
            "steel_model": "Steel02",
            "solver_profile_family": "pure-secant-disp",
        },
        "metrics": {
            **extrema(rows),
            "loop_work_MN_m": loop_work_mn_m(rows),
        },
        "figures": figures,
        "runner": proc_payload,
    }
    summary_path = args.figures_dir / f"{args.basename}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copy2(summary_path, args.secondary_figures_dir / summary_path.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
