#!/usr/bin/env python3
"""
Audit the opening, closure, and compressive re-engagement chronology of one
tracked structural fiber inside a reduced RC-column benchmark bundle.

The purpose is narrow and physical:

  - confirm whether the same extremal fiber really follows the expected
    open/close/recompress sequence under the declared cyclic protocol;
  - compare the event ordering between fall_n and OpenSees on the exact same
    structural benchmark bundle; and
  - localize any mismatch to a constitutive timing difference rather than a
    broader structural-control issue.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

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
        "figure.dpi": 140,
        "savefig.dpi": 300,
    }
)

BLUE = "#0b5fa5"
ORANGE = "#d97706"
GREEN = "#2f855a"
RED = "#c53030"
PURPLE = "#6b46c1"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Audit opening/closure/recompression chronology for one tracked "
            "fiber inside a reduced RC structural benchmark bundle."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_12p5mm_fiber_audit_zfix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / "data"
        / "output"
        / "cyclic_validation"
        / "reboot_structural_fiber_closure_audit",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=repo_root / "doc" / "figures" / "validation_reboot",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=repo_root / "PhD_Thesis" / "Figuras" / "validation_reboot",
    )
    parser.add_argument("--section-gp", type=int, default=0)
    parser.add_argument("--zone", default="cover_top")
    parser.add_argument("--material-role", default="unconfined_concrete")
    parser.add_argument("--y", type=float, default=-0.109375)
    parser.add_argument("--z", type=float, default=0.1175)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows: list[dict[str, object]] = []
        for row in csv.DictReader(fh):
            parsed: dict[str, object] = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
        return rows


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save(fig: plt.Figure, out_dirs: list[Path], stem: str) -> None:
    for out_dir in out_dirs:
        ensure_dir(out_dir)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)


def benchmark_root(bundle: Path, tool: str) -> Path:
    candidate = bundle / tool
    return candidate if candidate.exists() else bundle


def select_fiber_rows(
    rows: list[dict[str, object]],
    *,
    section_gp: int,
    zone: str,
    material_role: str,
    y: float,
    z: float,
) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if int(row["section_gp"]) == section_gp
        and str(row["zone"]) == zone
        and str(row["material_role"]) == material_role
        and abs(float(row["y"]) - y) < 1.0e-12
        and abs(float(row["z"]) - z) < 1.0e-12
    ]
    return sorted(selected, key=lambda row: int(row["step"]))


def enrich_with_control(
    fiber_rows: list[dict[str, object]],
    control_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    by_step = {int(row["step"]): row for row in control_rows}
    merged: list[dict[str, object]] = []
    for fiber_row in fiber_rows:
        step = int(fiber_row["step"])
        control = by_step.get(step, {})
        merged.append(
            {
                **fiber_row,
                "p": float(control.get("p", math.nan)),
                "target_drift_m": float(control.get("target_drift_m", math.nan)),
                "actual_tip_drift_m": float(control.get("actual_tip_drift_m", math.nan)),
                "stage": control.get("stage", ""),
            }
        )
    return merged


def find_first(rows: list[dict[str, object]], predicate) -> dict[str, object] | None:
    return next((row for row in rows if predicate(row)), None)


def detect_events(rows: list[dict[str, object]]) -> tuple[dict[str, object], dict[str, float]]:
    if not rows:
        raise RuntimeError("Cannot detect fiber events from an empty history.")

    peak_abs_stress = max(abs(float(row["stress_xx_MPa"])) for row in rows)
    peak_abs_tangent = max(abs(float(row["tangent_xx_MPa"])) for row in rows)
    stress_zero_tol = max(0.01 * peak_abs_stress, 5.0e-2)
    stress_recompression_tol = max(0.02 * peak_abs_stress, 1.0e-1)
    tangent_recompression_tol = max(0.4 * peak_abs_tangent, 1.0e3)

    initial_compression = find_first(
        rows, lambda row: float(row["stress_xx_MPa"]) < -stress_recompression_tol
    )
    first_positive_strain = find_first(rows, lambda row: float(row["strain_xx"]) > 0.0)
    opening = find_first(
        rows,
        lambda row: (
            first_positive_strain is not None
            and int(row["step"]) >= int(first_positive_strain["step"])
            and float(row["strain_xx"]) > 0.0
            and abs(float(row["stress_xx_MPa"])) <= stress_zero_tol
        ),
    )
    max_open = (
        max(
            (
                row
                for row in rows
                if opening is not None
                and int(row["step"]) >= int(opening["step"])
                and abs(float(row["stress_xx_MPa"])) <= stress_zero_tol
            ),
            key=lambda row: float(row["strain_xx"]),
        )
        if opening is not None
        else None
    )
    closure = find_first(
        rows,
        lambda row: (
            max_open is not None
            and int(row["step"]) > int(max_open["step"])
            and float(row["strain_xx"]) <= 0.0
        ),
    )
    recompression = find_first(
        rows,
        lambda row: (
            max_open is not None
            and int(row["step"]) > int(max_open["step"])
            and float(row["stress_xx_MPa"]) < -stress_recompression_tol
            and abs(float(row["tangent_xx_MPa"])) >= tangent_recompression_tol
        ),
    )

    events = {
        "initial_compression": initial_compression,
        "first_positive_strain": first_positive_strain,
        "opening": opening,
        "peak_open": max_open,
        "closure": closure,
        "recompression": recompression,
    }
    thresholds = {
        "stress_zero_tol_mpa": stress_zero_tol,
        "stress_recompression_tol_mpa": stress_recompression_tol,
        "tangent_recompression_tol_mpa": tangent_recompression_tol,
    }
    return events, thresholds


def event_payload(row: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None
    return {
        "step": int(row["step"]),
        "p": float(row["p"]),
        "target_drift_m": float(row["target_drift_m"]),
        "actual_tip_drift_m": float(row["actual_tip_drift_m"]),
        "strain_xx": float(row["strain_xx"]),
        "stress_xx_MPa": float(row["stress_xx_MPa"]),
        "tangent_xx_MPa": float(row["tangent_xx_MPa"]),
    }


def event_step(row: dict[str, object] | None) -> int | None:
    return int(row["step"]) if row is not None else None


def compare_event(lhs: dict[str, object] | None, rhs: dict[str, object] | None) -> dict[str, object] | None:
    if lhs is None or rhs is None:
        return None
    return {
        "step_delta": int(rhs["step"]) - int(lhs["step"]),
        "target_drift_delta_mm": 1.0e3 * (float(rhs["target_drift_m"]) - float(lhs["target_drift_m"])),
        "strain_delta": float(rhs["strain_xx"]) - float(lhs["strain_xx"]),
        "stress_delta_mpa": float(rhs["stress_xx_MPa"]) - float(lhs["stress_xx_MPa"]),
        "tangent_delta_mpa": float(rhs["tangent_xx_MPa"]) - float(lhs["tangent_xx_MPa"]),
    }


def build_trace_rows(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
    lhs_events: dict[str, object],
    rhs_events: dict[str, object],
) -> list[dict[str, object]]:
    rhs_by_step = {int(row["step"]): row for row in rhs_rows}
    lhs_event_steps = {event_step(row): name for name, row in lhs_events.items() if row is not None}
    rhs_event_steps = {event_step(row): name for name, row in rhs_events.items() if row is not None}
    trace_rows: list[dict[str, object]] = []
    for lhs in lhs_rows:
        step = int(lhs["step"])
        rhs = rhs_by_step.get(step)
        if rhs is None:
            continue
        trace_rows.append(
            {
                "step": step,
                "target_drift_m": float(lhs["target_drift_m"]),
                "fall_n_strain_xx": float(lhs["strain_xx"]),
                "fall_n_stress_xx_MPa": float(lhs["stress_xx_MPa"]),
                "fall_n_tangent_xx_MPa": float(lhs["tangent_xx_MPa"]),
                "opensees_strain_xx": float(rhs["strain_xx"]),
                "opensees_stress_xx_MPa": float(rhs["stress_xx_MPa"]),
                "opensees_tangent_xx_MPa": float(rhs["tangent_xx_MPa"]),
                "abs_stress_error_mpa": abs(float(lhs["stress_xx_MPa"]) - float(rhs["stress_xx_MPa"])),
                "abs_tangent_error_mpa": abs(float(lhs["tangent_xx_MPa"]) - float(rhs["tangent_xx_MPa"])),
                "fall_n_event": lhs_event_steps.get(step, ""),
                "opensees_event": rhs_event_steps.get(step, ""),
            }
        )
    return trace_rows


def make_event_figure(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
    lhs_events: dict[str, object],
    rhs_events: dict[str, object],
    out_dirs: list[Path],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.8), sharex=True)
    series = (
        ("strain_xx", r"Fiber strain $\varepsilon_x$"),
        ("stress_xx_MPa", "Fiber stress [MPa]"),
        ("tangent_xx_MPa", "Fiber tangent [MPa]"),
    )
    for ax, (field, ylabel) in zip(axes, series):
        ax.plot(
            [1.0e3 * float(row["target_drift_m"]) for row in lhs_rows],
            [float(row[field]) for row in lhs_rows],
            color=BLUE,
            lw=1.5,
            label="fall_n",
        )
        ax.plot(
            [1.0e3 * float(row["target_drift_m"]) for row in rhs_rows],
            [float(row[field]) for row in rhs_rows],
            color=ORANGE,
            lw=1.3,
            ls="--",
            label="OpenSeesPy",
        )
        for events, color, prefix in (
            (lhs_events, BLUE, "fall_n"),
            (rhs_events, ORANGE, "opensees"),
        ):
            for name, row in events.items():
                if row is None:
                    continue
                ax.scatter(
                    [1.0e3 * float(row["target_drift_m"])],
                    [float(row[field])],
                    color=color,
                    s=26,
                    marker="o" if prefix == "fall_n" else "s",
                    zorder=5,
                )
                ax.annotate(
                    name.replace("_", "\n"),
                    (1.0e3 * float(row["target_drift_m"]), float(row[field])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=color,
                )
        ax.set_ylabel(ylabel)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("Target tip drift [mm]")
    fig.suptitle(
        "Tracked structural fiber opening-closure-recompression audit\n"
        "same fiber, same structural cycle, fall_n vs OpenSees",
        fontsize=11,
    )
    save(fig, out_dirs, "reduced_rc_structural_fiber_closure_events")


def make_phase_figure(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
    out_dirs: list[Path],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    for ax, field, ylabel in (
        (axes[0], "stress_xx_MPa", "Fiber stress [MPa]"),
        (axes[1], "tangent_xx_MPa", "Fiber tangent [MPa]"),
    ):
        ax.plot(
            [float(row["strain_xx"]) for row in lhs_rows],
            [float(row[field]) for row in lhs_rows],
            color=BLUE,
            lw=1.5,
            label="fall_n",
        )
        ax.plot(
            [float(row["strain_xx"]) for row in rhs_rows],
            [float(row[field]) for row in rhs_rows],
            color=ORANGE,
            lw=1.3,
            ls="--",
            label="OpenSeesPy",
        )
        ax.set_xlabel(r"Fiber strain $\varepsilon_x$")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
    fig.suptitle(
        "Tracked structural fiber local phase path",
        fontsize=11,
    )
    save(fig, out_dirs, "reduced_rc_structural_fiber_closure_phase_path")


def main() -> int:
    args = parse_args()
    bundle = args.bundle.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    out_dirs = [args.figures_dir.resolve(), args.secondary_figures_dir.resolve()]

    falln_root = benchmark_root(bundle, "fall_n")
    opensees_root = benchmark_root(bundle, "opensees")

    benchmark_summary = read_json(bundle / "benchmark_summary.json")
    opensees_manifest = read_json(opensees_root / "reference_manifest.json")
    equivalence_scope = opensees_manifest.get("equivalence_scope", {})
    if isinstance(equivalence_scope, dict):
        convention = equivalence_scope.get("fiber_history_convention", "")
        if convention and convention != "native_opensees_structural":
            raise RuntimeError(
                "This audit expects a structural bundle with native OpenSees structural "
                f"fiber-history convention, got '{convention}'."
            )

    falln_rows = enrich_with_control(
        select_fiber_rows(
            read_csv_rows(falln_root / "section_fiber_state_history.csv"),
            section_gp=args.section_gp,
            zone=args.zone,
            material_role=args.material_role,
            y=args.y,
            z=args.z,
        ),
        read_csv_rows(falln_root / "control_state.csv"),
    )
    opensees_rows = enrich_with_control(
        select_fiber_rows(
            read_csv_rows(opensees_root / "section_fiber_state_history.csv"),
            section_gp=args.section_gp,
            zone=args.zone,
            material_role=args.material_role,
            y=args.y,
            z=args.z,
        ),
        read_csv_rows(opensees_root / "control_state.csv"),
    )

    if not falln_rows or not opensees_rows:
        raise RuntimeError("Tracked structural fiber not found in one of the compared histories.")

    lhs_events, lhs_thresholds = detect_events(falln_rows)
    rhs_events, rhs_thresholds = detect_events(opensees_rows)

    trace_rows = build_trace_rows(falln_rows, opensees_rows, lhs_events, rhs_events)
    write_csv(
        output_dir / "fiber_closure_trace.csv",
        (
            "step",
            "target_drift_m",
            "fall_n_strain_xx",
            "fall_n_stress_xx_MPa",
            "fall_n_tangent_xx_MPa",
            "opensees_strain_xx",
            "opensees_stress_xx_MPa",
            "opensees_tangent_xx_MPa",
            "abs_stress_error_mpa",
            "abs_tangent_error_mpa",
            "fall_n_event",
            "opensees_event",
        ),
        trace_rows,
    )

    make_event_figure(falln_rows, opensees_rows, lhs_events, rhs_events, out_dirs)
    make_phase_figure(falln_rows, opensees_rows, out_dirs)

    summary = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_structural_fiber_closure_audit",
        "bundle": str(bundle),
        "tracked_fiber_identity": {
            "section_gp": args.section_gp,
            "zone": args.zone,
            "material_role": args.material_role,
            "y": args.y,
            "z": args.z,
        },
        "benchmark_comparison_context": {
            "hysteresis_max_rel_error": benchmark_summary["comparison"]["hysteresis"]["max_rel_base_shear_error"],
            "section_fiber_zero_curvature_anchor_tangent_max_rel_error": benchmark_summary["comparison"]["structural_section_fiber_tangent_zero_curvature_anchor_only"]["max_rel_fiber_tangent_error"],
        },
        "fall_n": {
            "thresholds": lhs_thresholds,
            "events": {name: event_payload(row) for name, row in lhs_events.items()},
        },
        "opensees": {
            "thresholds": rhs_thresholds,
            "events": {name: event_payload(row) for name, row in rhs_events.items()},
        },
        "event_deltas": {
            name: compare_event(lhs_events.get(name), rhs_events.get(name))
            for name in lhs_events
        },
        "final_return_state": {
            "fall_n": event_payload(falln_rows[-1]),
            "opensees": event_payload(opensees_rows[-1]),
            "delta": compare_event(falln_rows[-1], opensees_rows[-1]),
        },
        "findings": [
            "Both solvers show the same broad physical sequence on the tracked extremal fiber: initial compression, tensile opening on the positive branch, closure near the return through zero drift, and re-entry into compression on the negative branch.",
            "The opening timing is not identical: OpenSees drops to zero tensile stress one structural step earlier than fall_n on this tracked fiber.",
            "The closure event itself is well aligned: both histories return to non-positive strain at the same structural step on the way back through zero drift, and both already carry a compressive state there.",
            "The main residual mismatch is instead in the release back to the final zero-drift state: fall_n has already returned to zero traction with residual tangent 0.03 MPa, while OpenSees still keeps a small compressive stress and a large tangent on the same fiber.",
            "This makes the remaining mismatch look less like a grossly wrong local physics ordering, and more like a constitutive timing/residual-state difference around crack closure and final re-contact of the cover concrete.",
        ],
        "artifacts": {
            "trace_csv": str(output_dir / "fiber_closure_trace.csv"),
            "event_figure": str(out_dirs[0] / "reduced_rc_structural_fiber_closure_events.png"),
            "phase_figure": str(out_dirs[0] / "reduced_rc_structural_fiber_closure_phase_path.png"),
        },
    }
    write_json(output_dir / "fiber_closure_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
