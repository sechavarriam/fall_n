#!/usr/bin/env python3
"""Plot cyclic RC-column hysteresis against structural and OpenSees references.

The figure is intentionally data-first: every local/continuum candidate is
drawn against the promoted fall_n structural reference and, when available, the
external high-fidelity OpenSeesPy structural comparator.  The same script can
be rerun after a long Ko-Bathe campaign finishes by passing its output
directory or its ``hysteresis.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from bisect import bisect_left
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Create the publication hysteresis overlay for the reduced RC-column "
            "cyclic validation."
        )
    )
    parser.add_argument(
        "--structural",
        type=Path,
        default=repo
        / "data/output/fe2_validation/structural_n10_lobatto_200mm_preflight",
        help="Directory or CSV for the promoted fall_n structural reference.",
    )
    parser.add_argument(
        "--opensees",
        type=Path,
        default=repo
        / "data/output/cyclic_validation/reboot_structural_multielement_hifi_cyclic_50mm_force2d_20260511/opensees_hifi",
        help=(
            "Directory or CSV for the high-fidelity structural OpenSeesPy "
            "comparator. Missing is recorded."
        ),
    )
    parser.add_argument(
        "--kobathe",
        action="append",
        default=[],
        help=(
            "Ko-Bathe candidate as LABEL=PATH or PATH. PATH may be a directory "
            "containing hysteresis.csv. Can be passed more than once."
        ),
    )
    parser.add_argument(
        "--xfem",
        type=Path,
        default=repo / "data/output/cyclic_validation_200mm_rerun_20260509/xfem_corrected/newton_l2",
        help="Optional promoted XFEM directory/CSV, included when present.",
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
        default="reduced_rc_cyclic_hysteresis_vs_structural_opensees",
    )
    return parser.parse_args()


def coerce_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def read_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            rows.append({key: coerce_float(value) for key, value in raw.items()})
    return rows


def resolve_hysteresis(path: Path, preferred: tuple[str, ...] = ()) -> Path | None:
    if path.is_file():
        return path
    if not path.exists():
        return None
    candidates = (*preferred, "comparison_hysteresis.csv", "hysteresis.csv", "global_xfem_newton_hysteresis.csv")
    for name in candidates:
        candidate = path / name
        if candidate.exists():
            return candidate
    nested = sorted(path.glob("**/hysteresis.csv"), key=lambda p: len(p.parts))
    return nested[0] if nested else None


def normalize_rows(path: Path) -> list[dict[str, float]]:
    rows = read_csv(path)
    out: list[dict[str, float]] = []
    for row in rows:
        drift = row.get("drift_m", math.nan)
        if not math.isfinite(drift):
            drift = row.get("tip_drift_m", row.get("actual_tip_drift_m", math.nan))
        if not math.isfinite(drift):
            drift_mm = row.get("drift_mm", row.get("tip_drift_mm", math.nan))
            drift = 1.0e-3 * drift_mm if math.isfinite(drift_mm) else math.nan
        shear = row.get("base_shear_MN", math.nan)
        if not math.isfinite(shear):
            shear = row.get("base_shear", row.get("reaction_MN", math.nan))
        p = row.get("p", math.nan)
        step = row.get("step", float(len(out)))
        if math.isfinite(drift) and math.isfinite(shear):
            out.append({"p": p, "step": step, "drift_m": drift, "base_shear_MN": shear})
    return out


def label_path(text: str) -> tuple[str, Path]:
    if "=" in text:
        label, raw = text.split("=", 1)
        return label.strip(), Path(raw.strip())
    path = Path(text)
    return path.name.replace("_", " "), path


def interpolate_by_p(rows: list[dict[str, float]], p: float) -> float:
    with_p = [row for row in rows if math.isfinite(row.get("p", math.nan))]
    if len(with_p) < 2 or not math.isfinite(p):
        return math.nan
    ps = [row["p"] for row in with_p]
    idx = bisect_left(ps, p)
    if idx <= 0:
        return with_p[0]["base_shear_MN"]
    if idx >= len(with_p):
        return with_p[-1]["base_shear_MN"]
    lo = with_p[idx - 1]
    hi = with_p[idx]
    span = hi["p"] - lo["p"]
    if abs(span) <= 1.0e-15:
        return hi["base_shear_MN"]
    t = (p - lo["p"]) / span
    return (1.0 - t) * lo["base_shear_MN"] + t * hi["base_shear_MN"]


def loop_work(rows: list[dict[str, float]]) -> float:
    work = 0.0
    for lhs, rhs in zip(rows, rows[1:]):
        work += 0.5 * (lhs["base_shear_MN"] + rhs["base_shear_MN"]) * (
            rhs["drift_m"] - lhs["drift_m"]
        )
    return work


def curve_metrics(candidate: list[dict[str, float]], reference: list[dict[str, float]]) -> dict[str, float]:
    peak_ref = max((abs(row["base_shear_MN"]) for row in reference), default=math.nan)
    peak_candidate = max((abs(row["base_shear_MN"]) for row in candidate), default=math.nan)
    max_drift_candidate = max((abs(row["drift_m"]) for row in candidate), default=math.nan)
    max_drift_reference = max((abs(row["drift_m"]) for row in reference), default=math.nan)
    basic = {
        "records": float(len(candidate)),
        "max_abs_drift_mm": 1000.0 * max_drift_candidate if math.isfinite(max_drift_candidate) else math.nan,
        "peak_abs_base_shear_kN": 1000.0 * peak_candidate if math.isfinite(peak_candidate) else math.nan,
        "loop_work_MN_m": loop_work(candidate),
    }
    if (
        not math.isfinite(max_drift_candidate)
        or not math.isfinite(max_drift_reference)
        or max_drift_reference <= 1.0e-14
        or abs(max_drift_candidate - max_drift_reference) / max_drift_reference > 0.02
    ):
        return {
            **basic,
            "metric_scope": (
                "display_only_amplitude_window_mismatch; RMS and loop-work "
                "ratios to the 200 mm structural reference are intentionally "
                "not computed"
            ),
        }
    errors: list[float] = []
    for row in candidate:
        ref = interpolate_by_p(reference, row.get("p", math.nan))
        if math.isfinite(ref):
            errors.append(row["base_shear_MN"] - ref)
    rms = math.sqrt(sum(value * value for value in errors) / len(errors)) if errors else math.nan
    max_abs = max((abs(value) for value in errors), default=math.nan)
    return {
        **basic,
        "metric_scope": "common_protocol_pseudotime",
        "loop_work_MN_m": loop_work(candidate),
        "reference_loop_work_MN_m": loop_work(reference),
        "loop_work_ratio_to_structural": (
            loop_work(candidate) / loop_work(reference)
            if abs(loop_work(reference)) > 1.0e-14
            else math.nan
        ),
        "rms_base_shear_error_kN": 1000.0 * rms if math.isfinite(rms) else math.nan,
        "peak_normalized_rms_base_shear_error": (
            rms / peak_ref if math.isfinite(rms) and peak_ref > 1.0e-14 else math.nan
        ),
        "max_base_shear_error_kN": 1000.0 * max_abs if math.isfinite(max_abs) else math.nan,
    }


def signed_stiffness_proxy(rows: list[dict[str, float]]) -> float:
    return sum(
        row["drift_m"] * row["base_shear_MN"]
        for row in rows
        if math.isfinite(row.get("drift_m", math.nan))
        and math.isfinite(row.get("base_shear_MN", math.nan))
    )


def sign_factor_to_reference(candidate: list[dict[str, float]], reference: list[dict[str, float]]) -> float:
    candidate_proxy = signed_stiffness_proxy(candidate)
    reference_proxy = signed_stiffness_proxy(reference)
    if abs(candidate_proxy) <= 1.0e-14 or abs(reference_proxy) <= 1.0e-14:
        return 1.0
    return -1.0 if candidate_proxy * reference_proxy < 0.0 else 1.0


def paired_hifi_summary(path: Path) -> dict[str, Any] | None:
    roots: list[Path] = []
    if path.is_file():
        roots.extend([path.parent, path.parent.parent])
    else:
        roots.extend([path, path.parent])
    for root in roots:
        candidate = root / "structural_multielement_hifi_audit_summary.json"
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            return {
                "status": payload.get("status"),
                "scope": payload.get("benchmark_scope"),
                "comparison": payload.get("comparison"),
                "timing": payload.get("timing"),
            }
    return None


def scaled_shear(rows: list[dict[str, float]], factor: float) -> list[dict[str, float]]:
    return [
        {
            **row,
            "base_shear_MN": factor * row["base_shear_MN"],
        }
        for row in rows
    ]


def add_curve(
    curves: list[dict[str, Any]],
    *,
    label: str,
    role: str,
    path: Path,
    preferred: tuple[str, ...] = (),
) -> None:
    csv_path = resolve_hysteresis(path, preferred)
    if csv_path is None:
        status = "missing"
        manifest = path / "reference_manifest.json" if path.is_dir() else None
        if manifest and manifest.exists():
            status = "manifest_without_hysteresis"
        root_summary = path.parent / "benchmark_summary.json" if path.is_dir() else None
        payload: dict[str, Any] = {
            "label": label,
            "role": role,
            "path": str(path),
            "status": status,
        }
        if root_summary and root_summary.exists():
            try:
                summary = json.loads(root_summary.read_text(encoding="utf-8"))
                payload["benchmark_status"] = summary.get("status")
                payload["failed_stage"] = summary.get("failed_stage")
            except json.JSONDecodeError:
                pass
        curves.append(payload)
        return
    rows = normalize_rows(csv_path)
    curves.append(
        {
            "label": label,
            "role": role,
            "path": str(csv_path),
            "status": "available" if rows else "empty",
            "rows": rows,
        }
    )


def save_figure(fig: Any, out_dirs: list[Path], basename: str) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        path = out_dirs[0] / f"{basename}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        outputs[ext] = str(path)
        for secondary in out_dirs[1:]:
            shutil.copy2(path, secondary / path.name)
            outputs[f"{secondary.name}_{ext}"] = str(secondary / path.name)
    return outputs


def main() -> int:
    args = parse_args()
    curves: list[dict[str, Any]] = []
    add_curve(
        curves,
        label="fall_n Timoshenko N=10 Lobatto",
        role="structural_reference",
        path=args.structural,
        preferred=("comparison_hysteresis.csv",),
    )
    add_curve(
        curves,
        label="OpenSees hi-fi structural",
        role="external_reference",
        path=args.opensees,
        preferred=("hysteresis.csv",),
    )
    if args.xfem:
        add_curve(
            curves,
            label="XFEM promoted",
            role="xfem_reference",
            path=args.xfem,
            preferred=("global_xfem_newton_hysteresis.csv", "hysteresis.csv"),
        )
    for item in args.kobathe:
        label, path = label_path(item)
        add_curve(curves, label=label, role="kobathe_candidate", path=path)

    structural = next((c for c in curves if c["role"] == "structural_reference"), None)
    if not structural or structural.get("status") != "available":
        raise SystemExit("Structural reference hysteresis is required.")
    reference_rows = structural["rows"]

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
    styles = {
        "structural_reference": ("#111827", "-", 1.9),
        "external_reference": ("#d97706", "--", 1.4),
        "xfem_reference": ("#2563eb", "-.", 1.2),
        "kobathe_candidate": ("#7c3aed", "-", 1.25),
    }

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    summary_curves: list[dict[str, Any]] = []
    for curve in curves:
        row = {key: curve.get(key) for key in ("label", "role", "path", "status")}
        if curve.get("status") != "available":
            summary_curves.append(row)
            continue
        factor = (
            1.0
            if curve["role"] == "structural_reference"
            else sign_factor_to_reference(curve["rows"], reference_rows)
        )
        rows = scaled_shear(curve["rows"], factor)
        color, linestyle, linewidth = styles.get(curve["role"], ("#4b5563", "-", 1.2))
        ax.plot(
            [1000.0 * item["drift_m"] for item in rows],
            [1000.0 * item["base_shear_MN"] for item in rows],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=curve["label"],
        )
        row["base_shear_sign_factor_to_structural"] = factor
        row.update(curve_metrics(rows, reference_rows))
        if curve["role"] == "external_reference":
            hifi_summary = paired_hifi_summary(Path(str(curve["path"])))
            if hifi_summary is not None:
                row["paired_fall_n_multielement_hifi_audit"] = hifi_summary
        summary_curves.append(row)

    ax.axhline(0.0, color="#9ca3af", linewidth=0.8)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.8)
    ax.set_xlabel("Tip displacement [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title("Cyclic RC-column hysteresis: structural, OpenSees hi-fi, and local models")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    outputs = save_figure(
        fig,
        [args.figures_dir, args.secondary_figures_dir],
        args.basename,
    )
    plt.close(fig)

    summary = {
        "status": "completed",
        "scope": "reduced_rc_column_cyclic_hysteresis_publication_overlay",
        "figures": outputs,
        "curves": summary_curves,
        "notes": (
            "The OpenSees curve is the multi-element high-fidelity structural "
            "comparator used to audit the fall_n structural reference, not the "
            "simplified bridge diagnostic. Missing OpenSees or Ko-Bathe curves "
            "are recorded explicitly so the Chapter 9 claim-support audit can "
            "distinguish absent data from a failed physics result. Rerun this "
            "script after each long cyclic candidate finishes."
        ),
    }
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.secondary_figures_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.figures_dir / f"{args.basename}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.copy2(summary_path, args.secondary_figures_dir / summary_path.name)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
