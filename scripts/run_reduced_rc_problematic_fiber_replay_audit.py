#!/usr/bin/env python3
"""
Replay the exact strain history of the most problematic audited section fiber
through the uniaxial fall_n and OpenSees material bridges.

This isolates whether the remaining anchor mismatch is already constitutive
under an identical strain history, or whether it only emerges after the section
coupling/preload state builds a different local history.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FiberSelection:
    step: int
    zone: str
    material_role: str
    y: float
    z: float
    area: float
    falln_fiber_index: int
    opensees_fiber_index: int
    selection_metric: str
    activity_floor: float
    tangent_rel_error: float
    stress_rel_error: float
    falln_anchor_strain: float
    opensees_anchor_strain: float
    falln_anchor_stress_mpa: float
    opensees_anchor_stress_mpa: float
    falln_anchor_tangent_mpa: float
    opensees_anchor_tangent_mpa: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the exact strain history of the most problematic audited "
            "section fiber through the fall_n and OpenSees uniaxial-material "
            "bridges."
        )
    )
    parser.add_argument(
        "--section-bundle",
        type=Path,
        default=Path(
            "data/output/cyclic_validation/"
            "reboot_external_section_benchmark_fiber_anchor_time_audit"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=Path("build/fall_n_reduced_rc_material_reference_benchmark.exe"),
    )
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument("--anchor-step", type=int, default=8)
    parser.add_argument("--zone", default="cover_top")
    parser.add_argument("--material-role", default="unconfined_concrete")
    parser.add_argument(
        "--selection-metric",
        choices=("tangent", "stress"),
        default="tangent",
    )
    parser.add_argument("--print-progress", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_protocol_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("step", "strain"))
        writer.writeheader()
        writer.writerows(
            {
                "step": int(row["step"]),
                "strain": float(row["strain_xx"]),
            }
            for row in rows
        )


def maybe_read_anchor_control_context(
    bundle_dir: Path,
    anchor_step: int,
) -> dict[str, object] | None:
    control_path = bundle_dir / "section_control_trace.csv"
    if not control_path.exists():
        return None

    rows = read_csv_rows(control_path)
    by_step = {int(row["step"]): row for row in rows}
    if anchor_step not in by_step:
        return None

    row = by_step[anchor_step]
    numeric_keys = (
        "target_curvature_y",
        "actual_curvature_y",
        "delta_target_curvature_y",
        "delta_actual_curvature_y",
        "pseudo_time_before",
        "pseudo_time_after",
        "pseudo_time_increment",
        "domain_time_before",
        "domain_time_after",
        "domain_time_increment",
        "control_dof_before",
        "control_dof_after",
        "newton_iterations",
        "newton_iterations_per_substep",
        "target_axial_force_MN",
        "actual_axial_force_MN",
        "axial_force_residual_MN",
    )
    integer_keys = (
        "target_increment_direction",
        "actual_increment_direction",
        "protocol_branch_id",
        "reversal_index",
        "branch_step_index",
        "accepted_substep_count",
        "max_bisection_level",
    )
    payload: dict[str, object] = {"stage": row.get("stage", "")}
    for key in numeric_keys:
        if key in row:
            payload[key] = float(row[key])
    for key in integer_keys:
        if key in row:
            payload[key] = int(round(float(row[key])))
    return payload


def row_key(row: dict[str, str], include_step: bool) -> tuple[object, ...]:
    key = (
        row["zone"],
        row["material_role"],
        round(float(row["y"]), 12),
        round(float(row["z"]), 12),
        round(float(row["area"]), 12),
    )
    return ((int(row["step"]),) + key) if include_step else key


def opensees_fiber_row_in_falln_convention(
    row: dict[str, str],
) -> dict[str, str]:
    mapped = dict(row)
    mapped["z"] = str(-float(row["z"]))
    mapped_zone = row["zone"]
    if mapped_zone == "cover_top":
        mapped_zone = "cover_bottom"
    elif mapped_zone == "cover_bottom":
        mapped_zone = "cover_top"
    mapped["zone"] = mapped_zone
    return mapped


def opensees_fiber_history_needs_legacy_flip(
    manifest: dict[str, object],
) -> bool:
    equivalence_scope = manifest.get("equivalence_scope", {})
    if not isinstance(equivalence_scope, dict):
        return True
    return (
        equivalence_scope.get("fiber_history_convention")
        != "native_falln_parity"
    )


def relative_error(lhs: float, rhs: float) -> float:
    return abs(lhs - rhs) / max(abs(rhs), 1.0e-12)


def select_problematic_fiber(
    lhs_rows: list[dict[str, str]],
    rhs_rows: list[dict[str, str]],
    *,
    anchor_step: int,
    zone: str,
    material_role: str,
    selection_metric: str,
) -> FiberSelection:
    lhs_by_key = {row_key(row, include_step=True): row for row in lhs_rows}
    rhs_by_key = {row_key(row, include_step=True): row for row in rhs_rows}
    common_keys = sorted(set(lhs_by_key) & set(rhs_by_key))

    candidates: list[FiberSelection] = []
    for key in common_keys:
        step, row_zone, row_role, *_ = key
        if step != anchor_step or row_zone != zone or row_role != material_role:
            continue

        lhs = lhs_by_key[key]
        rhs = rhs_by_key[key]
        tangent_rel_error = relative_error(
            float(lhs["tangent_xx_MPa"]), float(rhs["tangent_xx_MPa"])
        )
        stress_rel_error = relative_error(
            float(lhs["stress_xx_MPa"]), float(rhs["stress_xx_MPa"])
        )
        candidates.append(
            FiberSelection(
                step=anchor_step,
                zone=zone,
                material_role=material_role,
                y=float(lhs["y"]),
                z=float(lhs["z"]),
                area=float(lhs["area"]),
                falln_fiber_index=int(lhs["fiber_index"]),
                opensees_fiber_index=int(rhs["fiber_index"]),
                selection_metric=selection_metric,
                activity_floor=0.0,
                tangent_rel_error=tangent_rel_error,
                stress_rel_error=stress_rel_error,
                falln_anchor_strain=float(lhs["strain_xx"]),
                opensees_anchor_strain=float(rhs["strain_xx"]),
                falln_anchor_stress_mpa=float(lhs["stress_xx_MPa"]),
                opensees_anchor_stress_mpa=float(rhs["stress_xx_MPa"]),
                falln_anchor_tangent_mpa=float(lhs["tangent_xx_MPa"]),
                opensees_anchor_tangent_mpa=float(rhs["tangent_xx_MPa"]),
            )
        )

    if not candidates:
        raise RuntimeError("No shared section fibers matched the requested anchor filter.")

    if selection_metric == "tangent":
        reference_peak = max(
            abs(item.opensees_anchor_tangent_mpa) for item in candidates
        )
        activity_floor = max(5.0e-2 * reference_peak, 1.0e-12)
        active_candidates = [
            item
            for item in candidates
            if abs(item.opensees_anchor_tangent_mpa) >= activity_floor
        ]
        selector = lambda item: (
            item.tangent_rel_error,
            abs(item.opensees_anchor_tangent_mpa),
            abs(item.z),
            abs(item.y),
        )
    else:
        reference_peak = max(abs(item.opensees_anchor_stress_mpa) for item in candidates)
        activity_floor = max(5.0e-2 * reference_peak, 1.0e-12)
        active_candidates = [
            item
            for item in candidates
            if abs(item.opensees_anchor_stress_mpa) >= activity_floor
        ]
        selector = lambda item: (
            item.stress_rel_error,
            abs(item.opensees_anchor_stress_mpa),
            abs(item.z),
            abs(item.y),
        )

    selected = max(active_candidates or candidates, key=selector)
    return FiberSelection(
        step=selected.step,
        zone=selected.zone,
        material_role=selected.material_role,
        y=selected.y,
        z=selected.z,
        area=selected.area,
        falln_fiber_index=selected.falln_fiber_index,
        opensees_fiber_index=selected.opensees_fiber_index,
        selection_metric=selected.selection_metric,
        activity_floor=activity_floor,
        tangent_rel_error=selected.tangent_rel_error,
        stress_rel_error=selected.stress_rel_error,
        falln_anchor_strain=selected.falln_anchor_strain,
        opensees_anchor_strain=selected.opensees_anchor_strain,
        falln_anchor_stress_mpa=selected.falln_anchor_stress_mpa,
        opensees_anchor_stress_mpa=selected.opensees_anchor_stress_mpa,
        falln_anchor_tangent_mpa=selected.falln_anchor_tangent_mpa,
        opensees_anchor_tangent_mpa=selected.opensees_anchor_tangent_mpa,
    )


def rows_for_selected_fiber(
    rows: list[dict[str, str]],
    selection: FiberSelection,
) -> list[dict[str, str]]:
    selected_key = (
        selection.zone,
        selection.material_role,
        round(selection.y, 12),
        round(selection.z, 12),
        round(selection.area, 12),
    )
    fiber_rows = [
        row
        for row in rows
        if row_key(row, include_step=False) == selected_key
    ]
    return sorted(fiber_rows, key=lambda row: int(row["step"]))


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def role_to_material(material_role: str) -> str:
    return "steel" if "steel" in material_role else "concrete"


def mapping_policy_args(mapping_policy: dict[str, object], material: str) -> list[str]:
    args: list[str] = []
    if material == "steel":
        args.extend(
            (
                "--steel-r0",
                str(mapping_policy["steel_r0"]),
                "--steel-cr1",
                str(mapping_policy["steel_cr1"]),
                "--steel-cr2",
                str(mapping_policy["steel_cr2"]),
                "--steel-a1",
                str(mapping_policy["steel_a1"]),
                "--steel-a2",
                str(mapping_policy["steel_a2"]),
                "--steel-a3",
                str(mapping_policy["steel_a3"]),
                "--steel-a4",
                str(mapping_policy["steel_a4"]),
            )
        )
        return args

    args.extend(
        (
            "--concrete-model",
            str(mapping_policy["concrete_model"]),
            "--concrete-lambda",
            str(mapping_policy["concrete_lambda"]),
            "--concrete-ft-ratio",
            str(mapping_policy["concrete_ft_ratio"]),
            "--concrete-softening-multiplier",
            str(mapping_policy["concrete_softening_multiplier"]),
            "--concrete-residual-ratio",
            str(mapping_policy["concrete_unconfined_residual_ratio"]),
            "--concrete-ultimate-strain",
            str(mapping_policy["concrete_ultimate_strain"]),
        )
    )
    return args


def run_replay_case(
    *,
    repo_root: Path,
    output_dir: Path,
    falln_exe: Path,
    python_launcher: str,
    material: str,
    protocol_csv: Path,
    mapping_policy: dict[str, object],
    anchor_step: int,
    print_progress: bool,
) -> dict[str, object]:
    falln_dir = output_dir / "fall_n"
    opensees_dir = output_dir / "opensees"
    ensure_dir(falln_dir)
    ensure_dir(opensees_dir)

    falln_command = [
        str(falln_exe),
        "--output-dir",
        str(falln_dir),
        "--material",
        material,
        "--protocol",
        "cyclic",
        "--protocol-csv",
        str(protocol_csv),
    ]
    if print_progress:
        falln_command.append("--print-progress")
    falln_elapsed, falln_proc = run_command(falln_command, repo_root)
    (falln_dir / "stdout.log").write_text(falln_proc.stdout, encoding="utf-8")
    (falln_dir / "stderr.log").write_text(falln_proc.stderr, encoding="utf-8")
    if falln_proc.returncode != 0:
        raise RuntimeError(f"fall_n replay failed for {output_dir.name}: {falln_proc.stderr}")

    opensees_command = [
        *shlex.split(python_launcher),
        str(repo_root / "scripts/opensees_reduced_rc_material_reference.py"),
        "--output-dir",
        str(opensees_dir),
        "--material",
        material,
        "--protocol",
        "cyclic",
        "--protocol-csv",
        str(protocol_csv),
        "--falln-response",
        str(falln_dir / "uniaxial_response.csv"),
        *mapping_policy_args(mapping_policy, material),
    ]
    opensees_elapsed, opensees_proc = run_command(opensees_command, repo_root)
    (opensees_dir / "stdout.log").write_text(opensees_proc.stdout, encoding="utf-8")
    (opensees_dir / "stderr.log").write_text(opensees_proc.stderr, encoding="utf-8")
    if opensees_proc.returncode != 0:
        raise RuntimeError(
            f"OpenSees replay failed for {output_dir.name}: {opensees_proc.stderr}"
        )

    falln_manifest = json.loads(
        (falln_dir / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    opensees_manifest = json.loads(
        (opensees_dir / "reference_manifest.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (opensees_dir / "comparison_summary.json").read_text(encoding="utf-8")
    )
    falln_rows = read_csv_rows(falln_dir / "uniaxial_response.csv")
    opensees_rows = read_csv_rows(opensees_dir / "uniaxial_response.csv")
    falln_by_step = {int(row["step"]): row for row in falln_rows}
    opensees_by_step = {int(row["step"]): row for row in opensees_rows}
    anchor_step = (
        anchor_step
        if anchor_step in falln_by_step and anchor_step in opensees_by_step
        else sorted(set(falln_by_step) & set(opensees_by_step))[0]
    )
    falln_anchor = falln_by_step[anchor_step]
    opensees_anchor = opensees_by_step[anchor_step]

    return {
        "fall_n": {
            "process_wall_seconds": falln_elapsed,
            "manifest": falln_manifest,
            "dir": str(falln_dir),
        },
        "opensees": {
            "process_wall_seconds": opensees_elapsed,
            "manifest": opensees_manifest,
            "dir": str(opensees_dir),
        },
        "comparison": comparison,
        "anchor_step_diagnostics": {
            "step": anchor_step,
            "fall_n": {
                "strain": float(falln_anchor["strain"]),
                "stress_mpa": float(falln_anchor["stress_MPa"]),
                "tangent_mpa": float(falln_anchor["tangent_MPa"]),
            },
            "opensees": {
                "strain": float(opensees_anchor["strain"]),
                "stress_mpa": float(opensees_anchor["stress_MPa"]),
                "tangent_mpa": float(opensees_anchor["tangent_MPa"]),
            },
            "abs_stress_error_mpa": abs(
                float(falln_anchor["stress_MPa"]) - float(opensees_anchor["stress_MPa"])
            ),
            "abs_tangent_error_mpa": abs(
                float(falln_anchor["tangent_MPa"])
                - float(opensees_anchor["tangent_MPa"])
            ),
            "rel_stress_error": relative_error(
                float(falln_anchor["stress_MPa"]),
                float(opensees_anchor["stress_MPa"]),
            ),
            "rel_tangent_error": relative_error(
                float(falln_anchor["tangent_MPa"]),
                float(opensees_anchor["tangent_MPa"]),
            ),
        },
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    section_bundle = args.section_bundle.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    section_manifest = json.loads(
        (
            section_bundle / "opensees" / "reference_manifest.json"
        ).read_text(encoding="utf-8")
    )
    falln_rows = read_csv_rows(
        section_bundle / "fall_n" / "section_fiber_state_history.csv"
    )
    raw_opensees_rows = read_csv_rows(
        section_bundle / "opensees" / "section_fiber_state_history.csv"
    )
    if opensees_fiber_history_needs_legacy_flip(section_manifest):
        opensees_rows = [
            opensees_fiber_row_in_falln_convention(row)
            for row in raw_opensees_rows
        ]
    else:
        opensees_rows = raw_opensees_rows
    selection = select_problematic_fiber(
        falln_rows,
        opensees_rows,
        anchor_step=args.anchor_step,
        zone=args.zone,
        material_role=args.material_role,
        selection_metric=args.selection_metric,
    )

    selected_falln_rows = rows_for_selected_fiber(falln_rows, selection)
    selected_opensees_rows = rows_for_selected_fiber(opensees_rows, selection)
    selection_payload = {
        "step": selection.step,
        "zone": selection.zone,
        "material_role": selection.material_role,
        "y": selection.y,
        "z": selection.z,
        "area": selection.area,
        "falln_fiber_index": selection.falln_fiber_index,
        "opensees_fiber_index": selection.opensees_fiber_index,
        "selection_metric": selection.selection_metric,
        "activity_floor": selection.activity_floor,
        "tangent_rel_error": selection.tangent_rel_error,
        "stress_rel_error": selection.stress_rel_error,
        "falln_anchor_state": {
            "strain": selection.falln_anchor_strain,
            "stress_mpa": selection.falln_anchor_stress_mpa,
            "tangent_mpa": selection.falln_anchor_tangent_mpa,
        },
        "opensees_anchor_state": {
            "strain": selection.opensees_anchor_strain,
            "stress_mpa": selection.opensees_anchor_stress_mpa,
            "tangent_mpa": selection.opensees_anchor_tangent_mpa,
        },
    }
    write_json(output_dir / "problematic_fiber_selection.json", selection_payload)

    falln_protocol_csv = output_dir / "problematic_fiber_protocol_from_falln.csv"
    opensees_protocol_csv = (
        output_dir / "problematic_fiber_protocol_from_opensees.csv"
    )
    write_protocol_csv(falln_protocol_csv, selected_falln_rows)
    write_protocol_csv(opensees_protocol_csv, selected_opensees_rows)

    mapping_policy = dict(section_manifest["mapping_policy"])
    material = role_to_material(selection.material_role)

    replay_cases = {
        "falln_history_replay": run_replay_case(
            repo_root=repo_root,
            output_dir=output_dir / "falln_history_replay",
            falln_exe=args.falln_exe.resolve(),
            python_launcher=args.python_launcher,
            material=material,
            protocol_csv=falln_protocol_csv,
            mapping_policy=mapping_policy,
            anchor_step=args.anchor_step,
            print_progress=args.print_progress,
        ),
        "opensees_history_replay": run_replay_case(
            repo_root=repo_root,
            output_dir=output_dir / "opensees_history_replay",
            falln_exe=args.falln_exe.resolve(),
            python_launcher=args.python_launcher,
            material=material,
            protocol_csv=opensees_protocol_csv,
            mapping_policy=mapping_policy,
            anchor_step=args.anchor_step,
            print_progress=args.print_progress,
        ),
    }

    top_summary = {
        "status": "completed",
        "benchmark_scope": "reduced_rc_problematic_fiber_uniaxial_replay_audit",
        "section_bundle": str(section_bundle),
        "selection": selection_payload,
        "mapping_policy": mapping_policy,
        "material": material,
        "protocol_alignment": {
            "shared_step_count": min(
                len(selected_falln_rows), len(selected_opensees_rows)
            ),
            "max_abs_strain_difference": max(
                (
                    abs(
                        float(lhs["strain_xx"]) - float(rhs["strain_xx"])
                    )
                    for lhs, rhs in zip(selected_falln_rows, selected_opensees_rows)
                ),
                default=0.0,
            ),
        },
        "anchor_control_context": {
            "fall_n": maybe_read_anchor_control_context(
                section_bundle / "fall_n", selection.step
            ),
            "opensees": maybe_read_anchor_control_context(
                section_bundle / "opensees", selection.step
            ),
        },
        "replays": replay_cases,
        "artifacts": {
            "problematic_fiber_selection_json": str(
                output_dir / "problematic_fiber_selection.json"
            ),
            "falln_protocol_csv": str(falln_protocol_csv),
            "opensees_protocol_csv": str(opensees_protocol_csv),
        },
    }
    write_json(output_dir / "benchmark_summary.json", top_summary)

    print(
        "Problematic fiber replay completed:",
        f"fiber=({selection.y:+.6f}, {selection.z:+.6f})",
        f"material={material}",
        f"output={output_dir}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
