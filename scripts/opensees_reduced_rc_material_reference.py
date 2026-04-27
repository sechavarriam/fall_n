#!/usr/bin/env python3
"""
OpenSeesPy external uniaxial-material reference for the reduced RC-column reboot.

This script isolates the constitutive bridge behind the section/column benchmark.
It uses OpenSees material-testing commands instead of a structural domain so the
comparison focuses on:
  - Menegotto-Pinto steel <-> Steel02
  - Kent-Park concrete   <-> Concrete02
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReducedRCColumnReferenceSpec:
    concrete_fpc_mpa: float = 30.0
    steel_E_mpa: float = 200_000.0
    steel_fy_mpa: float = 420.0
    steel_b: float = 0.01


@dataclass(frozen=True)
class StrainPoint:
    step: int
    strain: float


@dataclass(frozen=True)
class UniaxialRecord:
    step: int
    strain: float
    stress_mpa: float
    tangent_mpa: float
    energy_density_mpa: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or stage an OpenSeesPy uniaxial-material reference for the reduced RC-column reboot."
    )
    parser.add_argument("--material", choices=("steel", "concrete"), required=True)
    parser.add_argument("--protocol", choices=("monotonic", "cyclic"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--monotonic-target-strain", type=float, default=0.0)
    parser.add_argument("--levels", default="")
    parser.add_argument(
        "--protocol-csv",
        type=Path,
        help="Optional CSV with explicit step,strain history to replay exactly.",
    )
    parser.add_argument("--steps-per-branch", type=int, default=40)
    parser.add_argument("--concrete-model", choices=("Concrete01", "Concrete02"), default="Concrete02")
    parser.add_argument("--steel-r0", type=float, default=20.0)
    parser.add_argument("--steel-cr1", type=float, default=18.5)
    parser.add_argument("--steel-cr2", type=float, default=0.15)
    parser.add_argument("--steel-a1", type=float, default=0.0)
    parser.add_argument("--steel-a2", type=float, default=1.0)
    parser.add_argument("--steel-a3", type=float, default=0.0)
    parser.add_argument("--steel-a4", type=float, default=1.0)
    parser.add_argument("--concrete-lambda", type=float, default=0.10)
    parser.add_argument("--concrete-ft-ratio", type=float, default=0.10)
    parser.add_argument("--concrete-softening-multiplier", type=float, default=0.10)
    parser.add_argument("--concrete-residual-ratio", type=float, default=0.10)
    parser.add_argument("--concrete-ultimate-strain", type=float, default=-0.006)
    parser.add_argument(
        "--falln-response",
        type=Path,
        help="Optional fall_n uniaxial_response.csv to compare against.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_levels_csv(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def read_protocol_csv(path: Path) -> list[StrainPoint]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        protocol = [
            StrainPoint(step=int(row["step"]), strain=float(row["strain"]))
            for row in reader
            if row.get("step") and row.get("strain")
        ]
    if any(
        protocol[index].step >= protocol[index + 1].step
        for index in range(len(protocol) - 1)
    ):
        raise ValueError("Custom protocol CSV requires strictly increasing step ids.")
    return protocol


def default_monotonic_target(args: argparse.Namespace) -> float:
    if abs(args.monotonic_target_strain) > 0.0:
        return args.monotonic_target_strain
    return 0.03 if args.material == "steel" else -0.006


def default_levels(args: argparse.Namespace, spec: ReducedRCColumnReferenceSpec) -> list[float]:
    if levels := parse_levels_csv(args.levels):
        return levels
    if args.material == "steel":
        ey = spec.steel_fy_mpa / spec.steel_E_mpa
        return [0.5 * ey, 1.0 * ey]
    return [0.001, 0.002, 0.003, 0.004]


def make_monotonic_protocol(target_strain: float, steps: int) -> list[StrainPoint]:
    return [
        StrainPoint(step=step, strain=(step / steps) * target_strain)
        for step in range(1, steps + 1)
    ]


def make_symmetric_cyclic_protocol(
    amplitudes: list[float], steps_per_excursion: int
) -> list[StrainPoint]:
    protocol: list[StrainPoint] = []
    step = 0
    current = 0.0
    for amplitude in amplitudes:
        for i in range(1, steps_per_excursion + 1):
            t = i / steps_per_excursion
            step += 1
            protocol.append(StrainPoint(step, current + t * (amplitude - current)))
        current = amplitude

        for i in range(1, 2 * steps_per_excursion + 1):
            t = i / (2 * steps_per_excursion)
            step += 1
            protocol.append(StrainPoint(step, amplitude - 2.0 * amplitude * t))
        current = -amplitude

        for i in range(1, steps_per_excursion + 1):
            t = i / steps_per_excursion
            step += 1
            protocol.append(StrainPoint(step, -amplitude * (1.0 - t)))
        current = 0.0
    return protocol


def make_concrete_cyclic_protocol(
    compression_amplitudes: list[float], tension_limit: float, steps_per_branch: int
) -> list[StrainPoint]:
    protocol: list[StrainPoint] = []
    step = 0
    current = 0.0
    for amplitude in compression_amplitudes:
        for i in range(1, steps_per_branch + 1):
            t = i / steps_per_branch
            step += 1
            protocol.append(StrainPoint(step, current + t * (-amplitude - current)))
        current = -amplitude

        for i in range(1, 2 * steps_per_branch + 1):
            t = i / (2 * steps_per_branch)
            step += 1
            protocol.append(StrainPoint(step, -amplitude + t * (tension_limit + amplitude)))
        current = tension_limit

        rebound_steps = max(steps_per_branch // 2, 1)
        for i in range(1, rebound_steps + 1):
            t = i / rebound_steps
            step += 1
            protocol.append(StrainPoint(step, tension_limit * (1.0 - t)))
        current = 0.0
    return protocol


def write_csv(path: Path, header: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)


def write_csv_contract_only(out_dir: Path) -> None:
    write_csv(
        out_dir / "uniaxial_response.csv",
        ("step", "strain", "stress_MPa", "tangent_MPa", "energy_density_MPa"),
        (),
    )


def mpa_to_pa(value: float) -> float:
    return value * 1.0e6


def pa_to_mpa(value: float) -> float:
    return value / 1.0e6


def load_opensees():
    try:
        import openseespy.opensees as ops
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Unable to import openseespy.opensees. Use --dry-run to stage only "
            "the benchmark contract."
        ) from exc
    return ops


def define_uniaxial_material(ops: object, args: argparse.Namespace, spec: ReducedRCColumnReferenceSpec) -> int:
    tag = 1
    if args.material == "steel":
        steel_args = [
            "Steel02",
            tag,
            mpa_to_pa(spec.steel_fy_mpa),
            mpa_to_pa(spec.steel_E_mpa),
            spec.steel_b,
            args.steel_r0,
            args.steel_cr1,
            args.steel_cr2,
        ]
        if any(abs(value) > 0.0 for value in (args.steel_a1, args.steel_a3)):
            steel_args.extend((args.steel_a1, args.steel_a2, args.steel_a3, args.steel_a4))
        ops.uniaxialMaterial(*steel_args)
        return tag

    ft_mpa = args.concrete_ft_ratio * spec.concrete_fpc_mpa
    ec_mpa = 1000.0 * spec.concrete_fpc_mpa
    cracking_strain = ft_mpa / max(ec_mpa, 1.0e-12)
    softening_span = max(args.concrete_softening_multiplier * cracking_strain, 1.0e-5)
    ets_mpa = ft_mpa / softening_span
    fpc_pa = mpa_to_pa(spec.concrete_fpc_mpa)
    fpcu_pa = -args.concrete_residual_ratio * fpc_pa
    if args.concrete_model == "Concrete01":
        ops.uniaxialMaterial(
            "Concrete01",
            tag,
            -fpc_pa,
            -0.002,
            fpcu_pa,
            args.concrete_ultimate_strain,
        )
        return tag

    ops.uniaxialMaterial(
        "Concrete02",
        tag,
        -fpc_pa,
        -0.002,
        fpcu_pa,
        args.concrete_ultimate_strain,
        args.concrete_lambda,
        mpa_to_pa(ft_mpa),
        mpa_to_pa(ets_mpa),
    )
    return tag


def protocol_points(args: argparse.Namespace, spec: ReducedRCColumnReferenceSpec) -> list[StrainPoint]:
    if args.protocol_csv:
        return read_protocol_csv(args.protocol_csv)
    if args.protocol == "monotonic":
        target = default_monotonic_target(args)
        target = abs(target) if args.material == "steel" else -abs(target)
        return make_monotonic_protocol(target, max(args.steps_per_branch, 1))

    levels = default_levels(args, spec)
    return (
        make_symmetric_cyclic_protocol(levels, max(args.steps_per_branch, 1))
        if args.material == "steel"
        else make_concrete_cyclic_protocol(levels, 2.0e-4, max(args.steps_per_branch, 1))
    )


def run_material_reference(
    ops: object, args: argparse.Namespace, spec: ReducedRCColumnReferenceSpec
) -> list[UniaxialRecord]:
    ops.wipe()
    mat_tag = define_uniaxial_material(ops, args, spec)
    ops.testUniaxialMaterial(mat_tag)

    records = [UniaxialRecord(0, 0.0, 0.0, pa_to_mpa(ops.getTangent()), 0.0)]
    prev_strain = 0.0
    prev_stress_mpa = 0.0
    cumulative_energy = 0.0

    for point in protocol_points(args, spec):
        ops.setStrain(point.strain)
        stress_mpa = pa_to_mpa(ops.getStress())
        tangent_mpa = pa_to_mpa(ops.getTangent())
        cumulative_energy += 0.5 * (stress_mpa + prev_stress_mpa) * (point.strain - prev_strain)
        records.append(
            UniaxialRecord(
                step=point.step,
                strain=point.strain,
                stress_mpa=stress_mpa,
                tangent_mpa=tangent_mpa,
                energy_density_mpa=cumulative_energy,
            )
        )
        prev_strain = point.strain
        prev_stress_mpa = stress_mpa

    return records


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def compare_by_step(
    lhs_rows: list[dict[str, str]],
    rhs_rows: list[dict[str, str]],
    lhs_key: str,
    rhs_key: str,
) -> dict[str, float]:
    lhs_by_step = {int(row["step"]): float(row[lhs_key]) for row in lhs_rows}
    rhs_by_step = {int(row["step"]): float(row[rhs_key]) for row in rhs_rows}
    common_steps = sorted(set(lhs_by_step) & set(rhs_by_step))
    finite_steps = [
        step
        for step in common_steps
        if math.isfinite(lhs_by_step[step]) and math.isfinite(rhs_by_step[step])
    ]
    reference_peak = max((abs(rhs_by_step[step]) for step in finite_steps), default=0.0)
    activity_floor = max(5.0e-2 * reference_peak, 1.0e-12)
    active_steps = [step for step in finite_steps if abs(rhs_by_step[step]) >= activity_floor]
    rel_errors = [
        abs(lhs_by_step[step] - rhs_by_step[step]) / max(abs(rhs_by_step[step]), 1.0e-12)
        for step in active_steps
    ]
    abs_errors = [abs(lhs_by_step[step] - rhs_by_step[step]) for step in finite_steps]
    rms = math.sqrt(sum(err * err for err in rel_errors) / len(rel_errors)) if rel_errors else 0.0
    return {
        "shared_step_count": len(common_steps),
        "finite_step_count": len(finite_steps),
        "nonfinite_step_count": len(common_steps) - len(finite_steps),
        "active_step_count": len(active_steps),
        "activity_floor": activity_floor,
        "max_abs_error": max(abs_errors, default=0.0),
        "max_rel_error": max(rel_errors, default=0.0),
        "rms_rel_error": rms,
    }


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    ensure_dir(out_dir)
    spec = ReducedRCColumnReferenceSpec()

    if args.dry_run:
        write_csv_contract_only(out_dir)
        manifest = {
            "benchmark_kind": "external_computational_reference",
            "tool": "OpenSeesPy",
            "reference_scope": "reduced_rc_column_uniaxial_material_reference",
            "material": args.material,
            "protocol": args.protocol,
            "protocol_source": "csv" if args.protocol_csv else "generated",
            "protocol_csv": str(args.protocol_csv) if args.protocol_csv else "",
            "status": "dry_run_only",
            "reference_spec": asdict(spec),
            "mapping_parameters": {
                "concrete_model": args.concrete_model,
                "steel_r0": args.steel_r0,
                "steel_cr1": args.steel_cr1,
                "steel_cr2": args.steel_cr2,
                "steel_a1": args.steel_a1,
                "steel_a2": args.steel_a2,
                "steel_a3": args.steel_a3,
                "steel_a4": args.steel_a4,
                "concrete_lambda": args.concrete_lambda,
                "concrete_ft_ratio": args.concrete_ft_ratio,
                "concrete_softening_multiplier": args.concrete_softening_multiplier,
                "concrete_residual_ratio": args.concrete_residual_ratio,
                "concrete_ultimate_strain": args.concrete_ultimate_strain,
            },
        }
        (out_dir / "reference_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return 0

    ops = load_opensees()
    total_start = time.perf_counter()
    analysis_start = time.perf_counter()
    records = run_material_reference(ops, args, spec)
    analysis_wall = time.perf_counter() - analysis_start

    output_start = time.perf_counter()
    write_csv(
        out_dir / "uniaxial_response.csv",
        ("step", "strain", "stress_MPa", "tangent_MPa", "energy_density_MPa"),
        (
            {
                "step": record.step,
                "strain": record.strain,
                "stress_MPa": record.stress_mpa,
                "tangent_MPa": record.tangent_mpa,
                "energy_density_MPa": record.energy_density_mpa,
            }
            for record in records
        ),
    )

    comparison_summary: dict[str, object] = {}
    if args.falln_response:
        lhs_rows = read_csv_rows(out_dir / "uniaxial_response.csv")
        rhs_rows = read_csv_rows(args.falln_response)
        comparison_summary = {
            "stress": compare_by_step(lhs_rows, rhs_rows, "stress_MPa", "stress_MPa"),
            "tangent": compare_by_step(lhs_rows, rhs_rows, "tangent_MPa", "tangent_MPa"),
            "energy_density": compare_by_step(
                lhs_rows, rhs_rows, "energy_density_MPa", "energy_density_MPa"
            ),
        }
        (out_dir / "comparison_summary.json").write_text(
            json.dumps(comparison_summary, indent=2), encoding="utf-8"
        )

    output_wall = time.perf_counter() - output_start
    total_wall = time.perf_counter() - total_start

    manifest = {
        "benchmark_kind": "external_computational_reference",
        "tool": "OpenSeesPy",
        "reference_scope": "reduced_rc_column_uniaxial_material_reference",
        "material": args.material,
        "protocol": args.protocol,
        "protocol_source": "csv" if args.protocol_csv else "generated",
        "protocol_csv": str(args.protocol_csv) if args.protocol_csv else "",
        "monotonic_target_strain": args.monotonic_target_strain,
        "levels": default_levels(args, spec) if args.protocol == "cyclic" else [],
        "steps_per_branch": args.steps_per_branch,
        "reference_spec": asdict(spec),
        "mapping_parameters": {
            "concrete_model": args.concrete_model,
            "steel_r0": args.steel_r0,
            "steel_cr1": args.steel_cr1,
            "steel_cr2": args.steel_cr2,
            "steel_a1": args.steel_a1,
            "steel_a2": args.steel_a2,
            "steel_a3": args.steel_a3,
            "steel_a4": args.steel_a4,
            "concrete_lambda": args.concrete_lambda,
            "concrete_ft_ratio": args.concrete_ft_ratio,
            "concrete_softening_multiplier": args.concrete_softening_multiplier,
            "concrete_residual_ratio": args.concrete_residual_ratio,
            "concrete_ultimate_strain": args.concrete_ultimate_strain,
        },
        "equivalence_scope": {
            "constitutive_mapping": (
                "Steel02 external comparable mapping with Menegotto-Pinto-like transition parameters"
                if args.material == "steel"
                else f"{args.concrete_model} external comparable mapping for the Kent-Park compression-dominated benchmark"
            ),
            "validation_role": (
                "External computational material bridge used to localize the nonlinear section/column gap "
                "before later experimental or literature closure."
            ),
        },
        "status": "completed",
        "record_count": len(records),
        "timing": {
            "total_wall_seconds": total_wall,
            "analysis_wall_seconds": analysis_wall,
            "output_write_wall_seconds": output_wall,
        },
    }
    (out_dir / "reference_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(
        "OpenSees material reference completed:",
        f"material={args.material}",
        f"protocol={args.protocol}",
        f"total={total_wall:.6f}s",
        f"records={len(records)}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
