#!/usr/bin/env python3
"""Build a reproducible dry-run scaling matrix for reduced-RC XFEM locals.

The script deliberately uses the benchmark executable instead of duplicating
the C++ formulas.  That keeps wrappers, documentation, and long-run planning
anchored to the same input surface that will later launch the actual model.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MeshCase:
    name: str
    nx: int
    ny: int
    nz: int
    bias_power: float = 2.0
    bias_location: str = "fixed-end"


@dataclass(frozen=True)
class MaterialCase:
    name: str
    token: str


DEFAULT_MESHES = (
    MeshCase("baseline_1x1x4", 1, 1, 4),
    MeshCase("probe_2x2x4", 2, 2, 4),
    MeshCase("intermediate_3x3x8", 3, 3, 8),
    MeshCase("rich_5x5x15", 5, 5, 15),
    MeshCase("target_7x7x25", 7, 7, 25),
)

DEFAULT_MATERIALS = (
    MaterialCase("elastic_proxy", "elastic"),
    MaterialCase("cyclic_crack_band_xfem", "cyclic-crack-band"),
    MaterialCase("ko_bathe_cost", "ko-bathe"),
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Run global-XFEM dry-run scale audits over a mesh/material matrix "
            "and publish CSV/JSON/PNG artifacts."
        )
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=repo / "build/fall_n_reduced_rc_xfem_reference_benchmark.exe",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/xfem_scale_audit_matrix",
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
        "--matrix-stem",
        default="xfem_scale_audit_matrix",
        help="Basename used for CSV, JSON, and optional figure artifacts.",
    )
    parser.add_argument(
        "--mesh",
        action="append",
        default=[],
        help=(
            "Mesh case as name:nx:ny:nz[:bias_power[:bias_location]]. "
            "May be passed multiple times. Defaults to the promoted-to-target "
            "matrix."
        ),
    )
    parser.add_argument(
        "--material",
        action="append",
        default=[],
        help=(
            "Material case as name:token. May be passed multiple times. "
            "Defaults to elastic, cyclic-crack-band, and Ko-Bathe cost."
        ),
    )
    parser.add_argument(
        "--crack-crossing-rebar-area-scale",
        type=float,
        default=1.0,
        help="Bridge activation used by the dry-run count estimate.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Write CSV/JSON only. Useful on machines without matplotlib.",
    )
    return parser.parse_args()


def parse_mesh_case(raw: str) -> MeshCase:
    parts = raw.split(":")
    if len(parts) not in (4, 5, 6):
        raise ValueError(
            "--mesh must have form name:nx:ny:nz[:bias_power[:bias_location]]"
        )
    return MeshCase(
        name=parts[0],
        nx=int(parts[1]),
        ny=int(parts[2]),
        nz=int(parts[3]),
        bias_power=float(parts[4]) if len(parts) >= 5 else 2.0,
        bias_location=parts[5] if len(parts) >= 6 else "fixed-end",
    )


def parse_material_case(raw: str) -> MaterialCase:
    parts = raw.split(":", 1)
    if len(parts) != 2:
        raise ValueError("--material must have form name:token")
    return MaterialCase(name=parts[0], token=parts[1])


def selected_meshes(args: argparse.Namespace) -> tuple[MeshCase, ...]:
    return tuple(parse_mesh_case(raw) for raw in args.mesh) or DEFAULT_MESHES


def selected_materials(args: argparse.Namespace) -> tuple[MaterialCase, ...]:
    return (
        tuple(parse_material_case(raw) for raw in args.material)
        or DEFAULT_MATERIALS
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_audit_case(
    exe: Path,
    output_root: Path,
    mesh: MeshCase,
    material: MaterialCase,
    crack_crossing_rebar_area_scale: float,
    force: bool,
) -> dict[str, Any]:
    case_dir = output_root / f"{mesh.name}__{material.name}"
    artifact = case_dir / "global_xfem_scale_audit.json"
    if force and case_dir.exists():
        shutil.rmtree(case_dir)
    if not artifact.exists():
        case_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(exe),
            "--global-xfem-scale-audit-only",
            "--global-xfem-nx",
            str(mesh.nx),
            "--global-xfem-ny",
            str(mesh.ny),
            "--global-xfem-nz",
            str(mesh.nz),
            "--global-xfem-bias-power",
            str(mesh.bias_power),
            "--global-xfem-bias-location",
            mesh.bias_location,
            "--global-xfem-concrete-material",
            material.token,
            "--global-xfem-crack-crossing-rebar-area-scale",
            str(crack_crossing_rebar_area_scale),
            "--output-dir",
            str(case_dir),
        ]
        subprocess.run(command, check=True)
    data = read_json(artifact)
    data["_case_dir"] = str(case_dir)
    data["_mesh_name"] = mesh.name
    data["_material_name"] = material.name
    data["_material_token"] = material.token
    return data


def flatten_row(data: dict[str, Any]) -> dict[str, Any]:
    mesh = data["mesh"]
    physics = data["physics"]
    counts = data["counts"]
    memory = data["memory_mib"]
    recommendations = data["recommendations"]
    return {
        "mesh_name": data["_mesh_name"],
        "material_name": data["_material_name"],
        "material_token": data["_material_token"],
        "nx": mesh["nx"],
        "ny": mesh["ny"],
        "nz": mesh["nz"],
        "host_element_count": counts["host_element_count"],
        "host_node_count": counts["host_node_count"],
        "host_material_point_count": counts["host_material_point_count"],
        "enriched_node_count": counts["enriched_node_count"],
        "estimated_total_state_dofs": counts["estimated_total_state_dofs"],
        "estimated_sparse_nonzeros": counts["estimated_sparse_nonzeros"],
        "sparse_matrix_mib": memory["sparse_matrix"],
        "direct_factorization_risk_mib": memory["direct_factorization_risk"],
        "material_state_mib": memory["material_state"],
        "estimated_hot_state_mib": memory["estimated_hot_state"],
        "solver_advice": recommendations["solver_advice"],
        "seed_state_cache_recommended": recommendations[
            "seed_state_cache_recommended"
        ],
        "newton_warm_start_recommended": recommendations[
            "newton_warm_start_recommended"
        ],
        "site_level_openmp_recommended": recommendations[
            "site_level_openmp_recommended"
        ],
        "global_petsc_assembly_openmp_recommended": recommendations[
            "global_petsc_assembly_openmp_recommended"
        ],
        "symmetric_matrix_storage_recommended": recommendations[
            "symmetric_matrix_storage_recommended"
        ],
        "symmetric_matrix_storage_requires_tangent_audit": recommendations[
            "symmetric_matrix_storage_requires_tangent_audit"
        ],
        "block_matrix_storage_candidate": recommendations[
            "block_matrix_storage_candidate"
        ],
        "field_split_or_asm_preconditioner_recommended": recommendations[
            "field_split_or_asm_preconditioner_recommended"
        ],
        "plain_gmres_ilu_rejected_for_enriched_branch": recommendations[
            "plain_gmres_ilu_rejected_for_enriched_branch"
        ],
        "constitutive_cost_kind": physics["constitutive_cost_kind"],
        "case_dir": data["_case_dir"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    cyclic = [
        row
        for row in rows
        if row["material_name"] == "cyclic_crack_band_xfem"
    ]
    labels = [row["mesh_name"].replace("_", "\n") for row in cyclic]
    x = range(len(cyclic))

    fig, ax1 = plt.subplots(figsize=(9.0, 5.2))
    ax1.plot(
        x,
        [row["estimated_total_state_dofs"] for row in cyclic],
        marker="o",
        label="state DOFs",
    )
    ax1.plot(
        x,
        [row["host_material_point_count"] for row in cyclic],
        marker="s",
        label="material points",
    )
    ax1.set_yscale("log")
    ax1.set_ylabel("count")
    ax1.set_xticks(list(x), labels, rotation=0)
    ax1.grid(True, which="both", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        [row["direct_factorization_risk_mib"] for row in cyclic],
        color="tab:red",
        marker="^",
        label="direct factor risk [MiB]",
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("memory [MiB]")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax1.set_title("Reduced RC XFEM local scale audit")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def copy_if_requested(source: Path, destination_dir: Path | None) -> None:
    if destination_dir is None:
        return
    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination_dir / source.name)


def main() -> int:
    args = parse_args()
    meshes = selected_meshes(args)
    materials = selected_materials(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    raw_results = [
        run_audit_case(
            args.exe,
            args.output_root,
            mesh,
            material,
            args.crack_crossing_rebar_area_scale,
            args.force,
        )
        for mesh in meshes
        for material in materials
    ]
    rows = [flatten_row(data) for data in raw_results]

    csv_path = args.figures_dir / f"{args.matrix_stem}.csv"
    summary_path = args.figures_dir / f"{args.matrix_stem}_summary.json"
    plot_path = args.figures_dir / f"{args.matrix_stem}.png"

    write_csv(csv_path, rows)
    summary = {
        "driver": "run_xfem_scale_audit_matrix.py",
        "case_count": len(rows),
        "mesh_cases": [mesh.__dict__ for mesh in meshes],
        "material_cases": [material.__dict__ for material in materials],
        "rows": rows,
        "diagnosis": (
            "Dry-run scale matrix. Direct LU is treated as an isolated "
            "reference strategy once the audit reports direct_lu_reference_only; "
            "symmetric PETSc storage is kept off until a tangent symmetry/SPD "
            "audit passes; many-site FE2 should combine seed caches, Newton "
            "warm-start, adaptive enriched-site activation, field-split/ASM "
            "preconditioning and OpenMP over independent local sites before "
            "attempting larger cyclic sweeps."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not args.skip_plot:
        write_plot(plot_path, rows)

    for source in (csv_path, summary_path, plot_path):
        if source.exists():
            copy_if_requested(source, args.secondary_figures_dir)

    print(
        "XFEM scale audit matrix completed | "
        f"cases={len(rows)} | csv={csv_path} | summary={summary_path}"
    )
    if plot_path.exists():
        print(f"figure={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
