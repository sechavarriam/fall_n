#!/usr/bin/env python3
"""
Run the structural TimoshenkoBeamN matrix study for the reduced RC-column case.

Scope:
  - fall_n: N=2..10, quadrature families, common cyclic protocol
  - OpenSees: one high-fidelity multi-element structural reference
  - outputs: timing, convergence, global physical-coherence metrics, figures
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opensees_reduced_rc_column_reference import (
    external_mapping_policy_catalog,
    structural_convergence_profile_families,
)
from python_launcher_utils import python_launcher_command


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


QUAD_STYLE = {
    "legendre": {"label": "Gauss-Legendre", "color": "#0b5fa5"},
    "lobatto": {"label": "Gauss-Lobatto", "color": "#d97706"},
    "radau-left": {"label": "Gauss-Radau left", "color": "#2f855a"},
    "radau-right": {"label": "Gauss-Radau right", "color": "#7c3aed"},
}
REP_NODE_STYLES = {2: ":", 4: "-", 10: "--"}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run the reduced RC-column TimoshenkoBeamN matrix experiment and "
            "compare it against a high-fidelity OpenSees structural reference."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--falln-exe",
        type=Path,
        default=repo_root / "build" / "fall_n_reduced_rc_column_reference_benchmark.exe",
    )
    parser.add_argument("--python-launcher", default="py -3.12")
    parser.add_argument("--runner-launcher", default="py -3.11")
    parser.add_argument("--beam-nodes", default="2,3,4,5,6,7,8,9,10")
    parser.add_argument(
        "--quadratures",
        default="legendre,lobatto,radau-left,radau-right",
    )
    parser.add_argument(
        "--solver-policy",
        choices=(
            "canonical-cascade",
            "newton-backtracking-only",
            "newton-l2-only",
            "newton-trust-region-only",
            "newton-trust-region-dogleg-only",
            "quasi-newton-only",
            "nonlinear-gmres-only",
            "nonlinear-cg-only",
            "anderson-only",
            "nonlinear-richardson-only",
        ),
        default="canonical-cascade",
    )
    parser.add_argument(
        "--continuation",
        choices=("monolithic", "segmented", "reversal-guarded", "arc-length"),
        default="reversal-guarded",
    )
    parser.add_argument("--continuation-segment-substep-factor", type=int, default=2)
    parser.add_argument("--steps-per-segment", type=int, default=2)
    parser.add_argument("--max-bisections", type=int, default=8)
    parser.add_argument("--axial-compression-mn", type=float, default=0.02)
    parser.add_argument("--axial-preload-steps", type=int, default=4)
    parser.add_argument("--amplitudes-mm", default="50,100,150,200")
    parser.add_argument(
        "--opensees-beam-element-family",
        choices=("disp", "force"),
        default="disp",
    )
    parser.add_argument(
        "--opensees-model-dimension",
        choices=("2d", "3d"),
        default="3d",
    )
    parser.add_argument(
        "--opensees-beam-integration",
        choices=("legendre", "lobatto"),
        default="lobatto",
    )
    parser.add_argument("--opensees-integration-points", type=int, default=5)
    parser.add_argument("--opensees-structural-element-count", type=int, default=12)
    parser.add_argument("--opensees-geom-transf", choices=("linear", "pdelta"), default="linear")
    parser.add_argument(
        "--opensees-mapping-policy",
        choices=tuple(external_mapping_policy_catalog().keys()),
        default=None,
    )
    parser.add_argument(
        "--opensees-solver-profile-family",
        choices=structural_convergence_profile_families(),
        default=None,
    )
    parser.add_argument("--opensees-concrete-model", choices=("Elastic", "Concrete01", "Concrete02"), default=None)
    parser.add_argument("--opensees-concrete-lambda", type=float, default=None)
    parser.add_argument("--opensees-concrete-ft-ratio", type=float, default=None)
    parser.add_argument("--opensees-concrete-softening-multiplier", type=float, default=None)
    parser.add_argument("--opensees-concrete-unconfined-residual-ratio", type=float, default=None)
    parser.add_argument("--opensees-concrete-confined-residual-ratio", type=float, default=None)
    parser.add_argument("--opensees-concrete-ultimate-strain", type=float, default=None)
    parser.add_argument("--opensees-steel-r0", type=float, default=None)
    parser.add_argument("--opensees-steel-cr1", type=float, default=None)
    parser.add_argument("--opensees-steel-cr2", type=float, default=None)
    parser.add_argument("--opensees-steel-a1", type=float, default=None)
    parser.add_argument("--opensees-steel-a2", type=float, default=None)
    parser.add_argument("--opensees-steel-a3", type=float, default=None)
    parser.add_argument("--opensees-steel-a4", type=float, default=None)
    parser.add_argument("--opensees-element-local-iterations", type=int, default=0)
    parser.add_argument("--opensees-element-local-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--opensees-steps-per-segment", type=int, default=None)
    parser.add_argument("--opensees-reversal-substep-factor", type=int, default=None)
    parser.add_argument("--opensees-max-bisections", type=int, default=None)
    parser.add_argument(
        "--opensees-amplitudes-mm",
        default="",
        help=(
            "Optional cyclic amplitudes for the OpenSees hi-fi reference. "
            "When omitted, the full fall_n amplitudes are reused. This is useful "
            "when the external hi-fi comparator has a smaller validated window "
            "than the full internal fall_n sweep."
        ),
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
    parser.add_argument("--print-progress", action="store_true")
    parser.add_argument(
        "--no-reuse-existing",
        action="store_true",
        help="Force rerun of the hi-fi reference and every fall_n bundle even if artifacts already exist.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class MatrixRow:
    beam_nodes: int
    beam_integration: str
    status: str
    bundle_dir: str
    return_code: int
    process_wall_seconds: float
    reported_total_wall_seconds: float
    reported_analysis_wall_seconds: float
    max_abs_tip_drift_mm: float
    max_abs_base_shear_kn: float
    max_abs_base_moment_knm: float
    max_newton_iterations: float
    avg_newton_iterations: float
    max_bisection_level: int
    avg_bisection_level: float
    total_hysteretic_work_kn_mm: float
    hifi_hysteresis_max_rel_error: float
    hifi_hysteresis_rms_rel_error: float
    hifi_positive_envelope_max_rel_error: float
    hifi_negative_envelope_max_rel_error: float
    hifi_total_work_rel_error: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv(raw: str, caster=float):
    return [caster(token.strip()) for token in raw.split(",") if token.strip()]


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
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


def run_command(command: list[str], cwd: Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    return time.perf_counter() - start, proc


def hysteretic_work_kn_mm(rows: list[dict[str, object]]) -> float:
    if len(rows) < 2:
        return math.nan
    total = 0.0
    for prev, curr in zip(rows[:-1], rows[1:]):
        du_mm = 1.0e3 * (float(curr["drift_m"]) - float(prev["drift_m"]))
        v_avg_kn = 0.5 * 1.0e3 * (float(curr["base_shear_MN"]) + float(prev["base_shear_MN"]))
        total += v_avg_kn * du_mm
    return abs(total)


def envelope_by_amplitude(rows: list[dict[str, object]]) -> dict[float, dict[str, float]]:
    out: dict[float, dict[str, float]] = {}
    for row in rows:
        amp = round(abs(1.0e3 * float(row["drift_m"])), 6)
        if amp <= 0.0:
            continue
        shear_kn = 1.0e3 * float(row["base_shear_MN"])
        payload = out.setdefault(amp, {"positive": -math.inf, "negative": math.inf})
        payload["positive"] = max(payload["positive"], shear_kn)
        payload["negative"] = min(payload["negative"], shear_kn)
    return out


def stepwise_error(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
) -> tuple[float, float]:
    def monotone_branches(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
        if len(rows) <= 1:
            return [rows] if rows else []
        tol = 1.0e-12
        branches: list[list[dict[str, object]]] = []
        current: list[dict[str, object]] = [rows[0]]
        current_direction = 0
        previous_drift = float(rows[0]["drift_m"])
        for row in rows[1:]:
            drift = float(row["drift_m"])
            delta = drift - previous_drift
            direction = 0 if abs(delta) <= tol else (1 if delta > 0.0 else -1)
            if current_direction == 0 and direction != 0:
                current_direction = direction
            elif direction != 0 and current_direction != 0 and direction != current_direction:
                branches.append(current)
                current = [current[-1], row]
                current_direction = direction
            else:
                current.append(row)
            previous_drift = drift
        if current:
            branches.append(current)
        return branches

    def interpolate_branch_shear(
        branch_rows: list[dict[str, object]],
        target_drift_m: float,
    ) -> float:
        if not branch_rows:
            return math.nan
        drifts = [float(r["drift_m"]) for r in branch_rows]
        shears = [1.0e3 * float(r["base_shear_MN"]) for r in branch_rows]
        if len(branch_rows) == 1:
            return shears[0]
        drift_min = min(drifts)
        drift_max = max(drifts)
        tol = 1.0e-12
        if target_drift_m < drift_min - tol or target_drift_m > drift_max + tol:
            return math.nan
        for left, right in zip(range(len(branch_rows) - 1), range(1, len(branch_rows))):
            x0 = drifts[left]
            x1 = drifts[right]
            y0 = shears[left]
            y1 = shears[right]
            if abs(target_drift_m - x0) <= tol:
                return y0
            if abs(target_drift_m - x1) <= tol:
                return y1
            if (x0 <= target_drift_m <= x1) or (x1 <= target_drift_m <= x0):
                if abs(x1 - x0) <= tol:
                    return 0.5 * (y0 + y1)
                alpha = (target_drift_m - x0) / (x1 - x0)
                return y0 + alpha * (y1 - y0)
        return math.nan

    lhs_branches = monotone_branches(lhs_rows)
    rhs_branches = monotone_branches(rhs_rows)
    errors: list[float] = []
    for lhs_branch, rhs_branch in zip(lhs_branches, rhs_branches):
        for row in lhs_branch:
            lhs = 1.0e3 * float(row["base_shear_MN"])
            rhs = interpolate_branch_shear(rhs_branch, float(row["drift_m"]))
            if not math.isfinite(rhs):
                continue
            errors.append(abs(lhs - rhs) / max(abs(rhs), 1.0e-9))
    return (
        max(errors) if errors else math.nan,
        sum(errors) / len(errors) if errors else math.nan,
    )


def envelope_error(
    lhs_rows: list[dict[str, object]],
    rhs_rows: list[dict[str, object]],
) -> tuple[float, float]:
    lhs_env = envelope_by_amplitude(lhs_rows)
    rhs_env = envelope_by_amplitude(rhs_rows)
    pos: list[float] = []
    neg: list[float] = []
    for amp, ref in rhs_env.items():
        if amp not in lhs_env:
            continue
        pos.append(
            abs(lhs_env[amp]["positive"] - ref["positive"])
            / max(abs(ref["positive"]), 1.0e-9)
        )
        neg.append(
            abs(lhs_env[amp]["negative"] - ref["negative"])
            / max(abs(ref["negative"]), 1.0e-9)
        )
    return (
        max(pos) if pos else math.nan,
        max(neg) if neg else math.nan,
    )


def build_falln_command(args: argparse.Namespace, out_dir: Path, beam_nodes: int, quadrature: str) -> list[str]:
    return [
        str(args.falln_exe),
        "--analysis",
        "cyclic",
        "--output-dir",
        str(out_dir),
        "--material-mode",
        "nonlinear",
        "--solver-policy",
        args.solver_policy,
        "--beam-nodes",
        str(beam_nodes),
        "--beam-integration",
        quadrature,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--continuation",
        args.continuation,
        "--continuation-segment-substep-factor",
        str(args.continuation_segment_substep_factor),
        "--amplitudes-mm",
        args.amplitudes_mm,
        "--steps-per-segment",
        str(args.steps_per_segment),
        "--max-bisections",
        str(args.max_bisections),
        *(["--print-progress"] if args.print_progress else []),
    ]


def build_hifi_command(args: argparse.Namespace, repo_root: Path, out_dir: Path) -> list[str]:
    opensees_amplitudes = args.opensees_amplitudes_mm or args.amplitudes_mm
    command = [
        *python_launcher_command(args.python_launcher),
        str(repo_root / "scripts" / "opensees_reduced_rc_column_hifi_reference.py"),
        "--output-dir",
        str(out_dir),
        "--model-dimension",
        args.opensees_model_dimension,
        "--analysis",
        "cyclic",
        "--material-mode",
        "nonlinear",
        "--beam-element-family",
        args.opensees_beam_element_family,
        "--beam-integration",
        args.opensees_beam_integration,
        "--integration-points",
        str(args.opensees_integration_points),
        "--structural-element-count",
        str(args.opensees_structural_element_count),
        "--geom-transf",
        args.opensees_geom_transf,
        "--axial-compression-mn",
        str(args.axial_compression_mn),
        "--axial-preload-steps",
        str(args.axial_preload_steps),
        "--amplitudes-mm",
        opensees_amplitudes,
        "--steps-per-segment",
        str(args.opensees_steps_per_segment or args.steps_per_segment),
        "--reversal-substep-factor",
        str(args.opensees_reversal_substep_factor or args.continuation_segment_substep_factor),
        "--max-bisections",
        str(args.opensees_max_bisections or args.max_bisections),
    ]
    optional_pairs = (
        ("--mapping-policy", args.opensees_mapping_policy),
        ("--solver-profile-family", args.opensees_solver_profile_family),
        ("--concrete-model", args.opensees_concrete_model),
        ("--concrete-lambda", args.opensees_concrete_lambda),
        ("--concrete-ft-ratio", args.opensees_concrete_ft_ratio),
        ("--concrete-softening-multiplier", args.opensees_concrete_softening_multiplier),
        (
            "--concrete-unconfined-residual-ratio",
            args.opensees_concrete_unconfined_residual_ratio,
        ),
        (
            "--concrete-confined-residual-ratio",
            args.opensees_concrete_confined_residual_ratio,
        ),
        ("--concrete-ultimate-strain", args.opensees_concrete_ultimate_strain),
        ("--steel-r0", args.opensees_steel_r0),
        ("--steel-cr1", args.opensees_steel_cr1),
        ("--steel-cr2", args.opensees_steel_cr2),
        ("--steel-a1", args.opensees_steel_a1),
        ("--steel-a2", args.opensees_steel_a2),
        ("--steel-a3", args.opensees_steel_a3),
        ("--steel-a4", args.opensees_steel_a4),
    )
    for flag, value in optional_pairs:
        if value is not None:
            command.extend((flag, str(value)))
    if args.opensees_element_local_iterations > 0:
        command.extend(
            (
                "--element-local-iterations",
                str(args.opensees_element_local_iterations),
                "--element-local-tolerance",
                str(args.opensees_element_local_tolerance),
            )
        )
    if args.print_progress:
        command.append("--print-progress")
    return command


def row_from_bundle(
    beam_nodes: int,
    quadrature: str,
    out_dir: Path,
    elapsed: float,
    return_code: int,
    ref_hysteresis: list[dict[str, object]],
    ref_work: float,
) -> MatrixRow:
    manifest = read_json(out_dir / "runtime_manifest.json")
    hysteresis = read_csv_rows(out_dir / "hysteresis.csv")
    moment_curvature = read_csv_rows(out_dir / "moment_curvature_base.csv")
    control = read_csv_rows(out_dir / "control_state.csv")
    max_err, rms_err = stepwise_error(hysteresis, ref_hysteresis)
    pos_env_err, neg_env_err = envelope_error(hysteresis, ref_hysteresis)
    total_work = hysteretic_work_kn_mm(hysteresis)
    return MatrixRow(
        beam_nodes=beam_nodes,
        beam_integration=quadrature,
        status=str(manifest.get("status", "failed")),
        bundle_dir=str(out_dir),
        return_code=return_code,
        process_wall_seconds=elapsed,
        reported_total_wall_seconds=float((manifest.get("timing") or {}).get("total_wall_seconds", math.nan)),
        reported_analysis_wall_seconds=float((manifest.get("timing") or {}).get("analysis_wall_seconds", math.nan)),
        max_abs_tip_drift_mm=max(abs(1.0e3 * float(r["drift_m"])) for r in hysteresis),
        max_abs_base_shear_kn=max(abs(1.0e3 * float(r["base_shear_MN"])) for r in hysteresis),
        max_abs_base_moment_knm=max(abs(1.0e3 * float(r["moment_y_MNm"])) for r in moment_curvature),
        max_newton_iterations=max(float(r["newton_iterations"]) for r in control),
        avg_newton_iterations=sum(float(r["newton_iterations"]) for r in control) / len(control),
        max_bisection_level=max(int(float(r["max_bisection_level"])) for r in control),
        avg_bisection_level=sum(float(r["max_bisection_level"]) for r in control) / len(control),
        total_hysteretic_work_kn_mm=total_work,
        hifi_hysteresis_max_rel_error=max_err,
        hifi_hysteresis_rms_rel_error=rms_err,
        hifi_positive_envelope_max_rel_error=pos_env_err,
        hifi_negative_envelope_max_rel_error=neg_env_err,
        hifi_total_work_rel_error=abs(total_work - ref_work) / max(abs(ref_work), 1.0e-9),
    )


def write_rows_csv(path: Path, rows: list[MatrixRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def save(fig: plt.Figure, stem: str, figures_dir: Path, secondary_dir: Path | None) -> list[str]:
    outputs: list[str] = []
    for base in [figures_dir, *( [secondary_dir] if secondary_dir else [] )]:
        ensure_dir(base)
        for ext in ("png", "pdf"):
            path = base / f"{stem}.{ext}"
            fig.savefig(path)
            outputs.append(str(path))
    plt.close(fig)
    return outputs


def plot_matrix(
    rows: list[MatrixRow],
    ref_hysteresis: list[dict[str, object]],
    figures_dir: Path,
    secondary_dir: Path | None,
) -> list[str]:
    outputs: list[str] = []
    quadratures = [q for q in QUAD_STYLE if any(r.beam_integration == q for r in rows)]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    for quadrature in quadratures:
        scoped = sorted((r for r in rows if r.beam_integration == quadrature and r.status == "completed"), key=lambda x: x.beam_nodes)
        if not scoped:
            continue
        style = QUAD_STYLE[quadrature]
        xs = [r.beam_nodes for r in scoped]
        axes[0].plot(xs, [r.process_wall_seconds for r in scoped], marker="o", color=style["color"], label=style["label"])
        axes[1].plot(xs, [r.max_newton_iterations for r in scoped], marker="o", color=style["color"], label=style["label"])
    axes[0].set_xlabel("Beam nodes N")
    axes[0].set_ylabel("Process wall time [s]")
    axes[0].set_title("TimoshenkoBeamN matrix timing")
    axes[0].legend()
    axes[1].set_xlabel("Beam nodes N")
    axes[1].set_ylabel("Max Newton iterations")
    axes[1].set_title("TimoshenkoBeamN matrix convergence")
    axes[1].legend()
    outputs += save(fig, "reduced_rc_timoshenko_matrix_timing_convergence", figures_dir, secondary_dir)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1))
    for quadrature in quadratures:
        scoped = sorted((r for r in rows if r.beam_integration == quadrature and r.status == "completed"), key=lambda x: x.beam_nodes)
        if not scoped:
            continue
        style = QUAD_STYLE[quadrature]
        xs = [r.beam_nodes for r in scoped]
        axes[0].plot(xs, [r.hifi_hysteresis_max_rel_error for r in scoped], marker="o", color=style["color"], label=style["label"])
        axes[1].plot(xs, [r.hifi_total_work_rel_error for r in scoped], marker="o", color=style["color"], label=style["label"])
    axes[0].set_xlabel("Beam nodes N")
    axes[0].set_ylabel("Max relative error")
    axes[0].set_title("Physical coherence vs OpenSees hi-fi\nbase-shear hysteresis")
    axes[0].legend()
    axes[1].set_xlabel("Beam nodes N")
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("Physical coherence vs OpenSees hi-fi\ntotal hysteretic work")
    axes[1].legend()
    outputs += save(fig, "reduced_rc_timoshenko_matrix_physical_coherence", figures_dir, secondary_dir)

    representative_nodes = [2, 4, 10]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True)
    axes_map = dict(zip(quadratures, axes.flat, strict=False))
    for quadrature in quadratures:
        ax = axes_map[quadrature]
        style = QUAD_STYLE[quadrature]
        ax.plot(
            [1.0e3 * float(r["drift_m"]) for r in ref_hysteresis],
            [1.0e3 * float(r["base_shear_MN"]) for r in ref_hysteresis],
            color="black",
            lw=1.6,
            label="OpenSees hi-fi",
        )
        for node in representative_nodes:
            row = next((r for r in rows if r.beam_integration == quadrature and r.beam_nodes == node and r.status == "completed"), None)
            if row is None:
                continue
            hysteresis = read_csv_rows(Path(row.bundle_dir) / "hysteresis.csv")
            ax.plot(
                [1.0e3 * float(r["drift_m"]) for r in hysteresis],
                [1.0e3 * float(r["base_shear_MN"]) for r in hysteresis],
                color=style["color"],
                linestyle=REP_NODE_STYLES[node],
                lw=1.2,
                label=f"fall_n N={node}",
            )
        ax.set_title(style["label"])
        ax.set_xlabel("Tip drift [mm]")
        ax.set_ylabel("Base shear [kN]")
        ax.legend(fontsize=7)
    outputs += save(fig, "reduced_rc_timoshenko_matrix_hysteresis_overlays", figures_dir, secondary_dir)

    return outputs


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    root_dir = args.output_dir.resolve()
    ensure_dir(root_dir)

    hifi_dir = root_dir / "opensees_hifi_reference"
    hifi_command = build_hifi_command(args, repo_root, hifi_dir)
    reuse_existing = not args.no_reuse_existing
    hifi_manifest_path = hifi_dir / "reference_manifest.json"

    if (
        reuse_existing
        and hifi_manifest_path.exists()
        and read_json(hifi_manifest_path).get("status") == "completed"
    ):
        ref_manifest = read_json(hifi_manifest_path)
        hifi_elapsed = float((ref_manifest.get("timing") or {}).get("total_wall_seconds", math.nan))
    else:
        hifi_elapsed, hifi_proc = run_command(hifi_command, repo_root)
        (hifi_dir / "audit_stdout.log").write_text(hifi_proc.stdout, encoding="utf-8")
        (hifi_dir / "audit_stderr.log").write_text(hifi_proc.stderr, encoding="utf-8")
        if hifi_proc.returncode != 0:
            hifi_manifest = read_json(hifi_manifest_path) if hifi_manifest_path.exists() else None
            write_json(
                root_dir / "timoshenko_matrix_experiment_summary.json",
                {
                    "status": "failed",
                    "failed_stage": "opensees_hifi_reference",
                    "command": hifi_command,
                    "return_code": hifi_proc.returncode,
                    "opensees_hifi_manifest": hifi_manifest,
                },
            )
            return hifi_proc.returncode
        ref_manifest = read_json(hifi_manifest_path)

    ref_hysteresis = read_csv_rows(hifi_dir / "hysteresis.csv")
    ref_work = hysteretic_work_kn_mm(ref_hysteresis)

    rows: list[MatrixRow] = []
    for beam_nodes in parse_csv(args.beam_nodes, int):
        for quadrature in parse_csv(args.quadratures, str):
            bundle_dir = root_dir / "fall_n_matrix" / f"n{beam_nodes:02d}_{quadrature.replace('-', '_')}"
            ensure_dir(bundle_dir)
            runtime_manifest = bundle_dir / "runtime_manifest.json"
            if reuse_existing and runtime_manifest.exists():
                manifest = read_json(runtime_manifest)
                elapsed = float((manifest.get("timing") or {}).get("total_wall_seconds", math.nan))
                rows.append(
                    row_from_bundle(
                        beam_nodes,
                        quadrature,
                        bundle_dir,
                        elapsed,
                        0,
                        ref_hysteresis,
                        ref_work,
                    )
                )
            else:
                command = build_falln_command(args, bundle_dir, beam_nodes, quadrature)
                elapsed, proc = run_command(command, repo_root)
                (bundle_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
                (bundle_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
                if proc.returncode == 0 and runtime_manifest.exists():
                    rows.append(
                        row_from_bundle(
                            beam_nodes,
                            quadrature,
                            bundle_dir,
                            elapsed,
                            proc.returncode,
                            ref_hysteresis,
                            ref_work,
                        )
                    )
                else:
                    rows.append(
                        MatrixRow(
                            beam_nodes=beam_nodes,
                            beam_integration=quadrature,
                            status="failed",
                            bundle_dir=str(bundle_dir),
                            return_code=proc.returncode,
                            process_wall_seconds=elapsed,
                            reported_total_wall_seconds=math.nan,
                            reported_analysis_wall_seconds=math.nan,
                            max_abs_tip_drift_mm=math.nan,
                            max_abs_base_shear_kn=math.nan,
                            max_abs_base_moment_knm=math.nan,
                            max_newton_iterations=math.nan,
                            avg_newton_iterations=math.nan,
                            max_bisection_level=0,
                            avg_bisection_level=math.nan,
                            total_hysteretic_work_kn_mm=math.nan,
                            hifi_hysteresis_max_rel_error=math.nan,
                            hifi_hysteresis_rms_rel_error=math.nan,
                            hifi_positive_envelope_max_rel_error=math.nan,
                            hifi_negative_envelope_max_rel_error=math.nan,
                            hifi_total_work_rel_error=math.nan,
                        )
                    )
            if args.print_progress:
                row = rows[-1]
                print(
                    f"[N={beam_nodes:02d}, q={quadrature}] status={row.status} "
                    f"t={row.process_wall_seconds:.3f}s err={row.hifi_hysteresis_max_rel_error:.3e}"
                )

    write_rows_csv(root_dir / "timoshenko_matrix_cases.csv", rows)
    figures = plot_matrix(rows, ref_hysteresis, args.figures_dir, args.secondary_figures_dir)

    completed = [row for row in rows if row.status == "completed"]
    summary = {
        "status": "completed",
        "analysis": "cyclic",
        "benchmark_scope": "reduced_rc_timoshenko_matrix_experiment",
        "fall_n_solver_policy": args.solver_policy,
        "fall_n_continuation": args.continuation,
        "cyclic_amplitudes_mm": parse_csv(args.amplitudes_mm, float),
        "opensees_hifi_cyclic_amplitudes_mm": parse_csv(
            args.opensees_amplitudes_mm or args.amplitudes_mm,
            float,
        ),
        "case_count": len(rows),
        "completed_case_count": len(completed),
        "failed_case_count": len(rows) - len(completed),
        "opensees_hifi_reference": {
            "dir": str(hifi_dir),
            "model_dimension": args.opensees_model_dimension,
            "beam_element_family": args.opensees_beam_element_family,
            "beam_integration": args.opensees_beam_integration,
            "integration_points": args.opensees_integration_points,
            "structural_element_count": args.opensees_structural_element_count,
            "process_wall_seconds": hifi_elapsed,
            "manifest": ref_manifest,
        },
        "comparison_scope_note": (
            "OpenSees hi-fi errors are computed over the common declared comparator "
            "window only. If the external hi-fi amplitudes are shorter than the full "
            "fall_n sweep, hysteresis/error metrics remain intentionally local to "
            "that validated external window while fall_n timing/convergence still "
            "cover the full internal amplitude sweep."
        ),
        "fastest_completed_case": (
            asdict(min(completed, key=lambda row: row.process_wall_seconds))
            if completed
            else None
        ),
        "best_hifi_coherence_case": (
            asdict(min(completed, key=lambda row: row.hifi_hysteresis_max_rel_error))
            if completed
            else None
        ),
        "artifacts": {
            "cases_csv": str(root_dir / "timoshenko_matrix_cases.csv"),
            "figures": figures,
        },
    }
    write_json(root_dir / "timoshenko_matrix_experiment_summary.json", summary)
    shutil.copy2(root_dir / "timoshenko_matrix_experiment_summary.json", args.figures_dir / "timoshenko_matrix_experiment_summary.json")
    if args.secondary_figures_dir:
        ensure_dir(args.secondary_figures_dir)
        shutil.copy2(root_dir / "timoshenko_matrix_experiment_summary.json", args.secondary_figures_dir / "timoshenko_matrix_experiment_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
