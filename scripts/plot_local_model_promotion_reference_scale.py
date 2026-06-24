#!/usr/bin/env python3
"""Plot reference-scale local model promotion hysteresis.

The plot intentionally excludes overly stiff local cases so the promoted
evidence can be read on the same scale as OpenSees Hi-Fi and the structural
fall_n references. Excluded cases are still written to a CSV audit table.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-roots",
        nargs="*",
        type=Path,
        default=[
            root
            / "data/output/cyclic_validation/local_model_promotion_matrix_phase1_xfem_20260521",
            root
            / "data/output/cyclic_validation/local_model_promotion_matrix_phase1_kobathe_20260521",
            root
            / "data/output/cyclic_validation/local_model_promotion_matrix_phase2_kobathe_nz3_20260521",
        ],
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=root / "doc/figures/validation_reboot/local_model_promotion_matrix_200mm",
    )
    parser.add_argument(
        "--secondary-figures-dir",
        type=Path,
        default=root
        / "PhD_Thesis/Figuras/validation_reboot/local_model_promotion_matrix_200mm",
    )
    parser.add_argument(
        "--opensees-hifi",
        type=Path,
        default=root
        / "data/output/cyclic_validation/opensees_hifi_timoshenko_matrix_200mm_publication_20260518/hysteresis.csv",
    )
    parser.add_argument(
        "--structural-matrix-dir",
        type=Path,
        default=root
        / "data/output/cyclic_validation/timoshenko_matrix_reproduced_historical_closure_20260520/fall_n_matrix",
    )
    parser.add_argument(
        "--max-peak-kN",
        type=float,
        default=80.0,
        help="Exclude local candidates whose absolute base shear exceeds this peak.",
    )
    parser.add_argument("--ylim-kN", type=float, default=50.0)
    return parser.parse_args()


def read_csv_rows(path: Path, sign: float = 1.0) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            def value(*names: str) -> float:
                for name in names:
                    raw_value = raw.get(name)
                    if raw_value is None:
                        continue
                    try:
                        return float(raw_value)
                    except ValueError:
                        continue
                return math.nan

            drift_m = value("drift_m")
            if not math.isfinite(drift_m):
                drift_m = value("drift_mm") / 1000.0
            shear_mn = value("base_shear_MN", "base_shear_mn") * sign
            if math.isfinite(drift_m) and math.isfinite(shear_mn):
                rows.append(
                    {
                        "drift_mm": 1000.0 * drift_m,
                        "shear_kN": 1000.0 * shear_mn,
                    }
                )
    return rows


def score(result: dict[str, Any]) -> float:
    if not result.get("completed_200mm"):
        return math.inf
    terms = [
        result.get("opensees_rms_rel"),
        result.get("opensees_work_rel"),
        0.2 * result.get("lobatto_n4_rms_rel", math.nan),
    ]
    elapsed = result.get("elapsed_seconds", 1.0)
    if isinstance(elapsed, (int, float)) and elapsed > 0:
        terms.append(0.03 * math.log10(max(elapsed, 1.0)))
    finite = [term for term in terms if isinstance(term, (int, float)) and math.isfinite(term)]
    return sum(finite) if finite else math.inf


def candidate_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for root in args.campaign_roots:
        for manifest_path in root.rglob("case_manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            result = manifest.get("result", {})
            if not result.get("completed_200mm"):
                continue
            out_dir = Path(result["output_dir"])
            sign = -1.0 if result.get("family") == "xfem" else 1.0
            history = out_dir / "global_xfem_newton_hysteresis.csv"
            if not history.exists():
                history = out_dir / "hysteresis.csv"
            rows = read_csv_rows(history, sign=sign)
            if not rows:
                continue
            peak = max(abs(row["shear_kN"]) for row in rows)
            item = {
                "case_id": result.get("case_id"),
                "family": result.get("family"),
                "hex_order": result.get("hex_order"),
                "nx": result.get("nx"),
                "ny": result.get("ny"),
                "nz": result.get("nz"),
                "bias_mode": result.get("bias_mode"),
                "peak_abs_kN": peak,
                "score": score(result),
                "vtk_status": result.get("vtk_status"),
                "sign_applied": sign,
                "history": str(history),
                "rows": rows,
            }
            if peak <= args.max_peak_kN:
                selected.append(item)
            else:
                item["exclusion_reason"] = (
                    f"peak_abs_kN={peak:.3g} exceeds reference-scale limit "
                    f"{args.max_peak_kN:.3g} kN"
                )
                excluded.append(item)
    selected.sort(key=lambda item: item["score"])
    excluded.sort(key=lambda item: item["peak_abs_kN"], reverse=True)
    return selected, excluded


def write_audit_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "family",
        "hex_order",
        "nx",
        "ny",
        "nz",
        "bias_mode",
        "peak_abs_kN",
        "score",
        "vtk_status",
        "sign_applied",
        "history",
        "exclusion_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_figure(fig: plt.Figure, path: Path, secondary_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = path.with_suffix(f".{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        outputs.append(out)
        secondary = secondary_dir / out.name
        fig.savefig(secondary, dpi=300, bbox_inches="tight")
        outputs.append(secondary)
    plt.close(fig)
    return outputs


def plot(args: argparse.Namespace, selected: list[dict[str, Any]]) -> list[Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(7.1, 4.5))
    references = [
        ("OpenSees Hi-Fi", args.opensees_hifi, "black", 2.4, "-"),
        (
            "fall_n Lobatto N=4",
            args.structural_matrix_dir / "n04_lobatto/hysteresis.csv",
            "#d97706",
            1.4,
            "-",
        ),
        (
            "fall_n Lobatto N=6",
            args.structural_matrix_dir / "n06_lobatto/hysteresis.csv",
            "#0b5fa5",
            1.1,
            "--",
        ),
        (
            "fall_n Lobatto N=8",
            args.structural_matrix_dir / "n08_lobatto/hysteresis.csv",
            "#2f855a",
            1.1,
            ":",
        ),
    ]
    for label, path, color, width, style in references:
        rows = read_csv_rows(path)
        ax.plot(
            [row["drift_mm"] for row in rows],
            [row["shear_kN"] for row in rows],
            label=label,
            color=color,
            lw=width,
            ls=style,
        )
    colors = {"xfem": "#7c3aed", "kobathe": "#dc2626"}
    for item in selected[:8]:
        label = (
            f"{item['family']} {item['hex_order']} "
            f"{item['nx']}x{item['ny']}x{item['nz']} {item['bias_mode']}"
        )
        ax.plot(
            [row["drift_mm"] for row in item["rows"]],
            [row["shear_kN"] for row in item["rows"]],
            label=label,
            color=colors.get(item["family"], "gray"),
            lw=1.15,
            alpha=0.80,
        )
    ax.set_xlabel("Tip displacement [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_ylim(-args.ylim_kN, args.ylim_kN)
    ax.axhline(args.ylim_kN, color="0.35", lw=0.8, ls="--")
    ax.axhline(-args.ylim_kN, color="0.35", lw=0.8, ls=":")
    ax.set_title("Reference-scale cyclic hysteresis for comparable local models")
    ax.legend(fontsize=6.8, ncol=2)
    return save_figure(
        fig,
        args.figures_dir / "local_model_promotion_reference_scale_hysteresis.pdf",
        args.secondary_figures_dir,
    )


def main() -> int:
    args = parse_args()
    selected, excluded = candidate_rows(args)
    write_audit_csv(
        args.figures_dir / "local_model_promotion_reference_scale_selected.csv",
        selected,
    )
    write_audit_csv(
        args.figures_dir / "local_model_promotion_reference_scale_excluded_rigid.csv",
        excluded,
    )
    outputs = plot(args, selected)
    for output in outputs:
        print(output)
    print(args.figures_dir / "local_model_promotion_reference_scale_selected.csv")
    print(args.figures_dir / "local_model_promotion_reference_scale_excluded_rigid.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
