#!/usr/bin/env python3
"""Audit selected-element force recorder conventions for the L-shaped benchmark.

The goal is deliberately modest: identify which OpenSees recorder response
(`force`, `globalForce`, `localForce`, `basicForce`) is even dimensionally and
component-wise comparable with fall_n's selected element internal-force vector.
The script avoids treating any recorder as ground truth by reporting amplitudes,
component counts and best component-wise correlations over the shared time
window.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def read_force_history(path: Path, label: str) -> dict | None:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return None
    header = [h.strip() for h in lines[0].split(",")]
    starts_with_step = len(header) > 1 and header[1] == "step"
    first_component = 2 if starts_with_step else 1
    rows: list[tuple[float, list[float]]] = []
    for line in lines[1:]:
        raw = [x.strip() for x in line.split(",")]
        if len(raw) <= first_component:
            continue
        try:
            time = float(raw[0])
            comps = [float(x) for x in raw[first_component:] if x != ""]
        except ValueError:
            continue
        rows.append((time, comps))
    if not rows:
        return None
    width = max(len(comps) for _time, comps in rows)
    time = [row_time for row_time, _comps in rows]
    components = {
        f"f{i}": [comps[i] if i < len(comps) else math.nan for _time, comps in rows]
        for i in range(width)
    }
    return {
        "label": label,
        "path": str(path),
        "time": time,
        "components": components,
        "component_count": width,
    }


def discover_opensees_case(path: Path, label: str) -> list[dict]:
    cases = []
    for suffix in ("force", "global_force", "local_force", "basic_force"):
        case = read_force_history(path / f"selected_element_1_{suffix}.csv", f"{label}:{suffix}")
        if case is not None:
            cases.append(case)
    return cases


def finite_pairs(values_a: list[float], values_b: list[float]) -> tuple[list[float], list[float]]:
    a, b = [], []
    for va, vb in zip(values_a, values_b):
        if math.isfinite(va) and math.isfinite(vb):
            a.append(va)
            b.append(vb)
    return a, b


def interp(values_time: list[float], values: list[float], target_time: list[float]) -> list[float]:
    if not values_time:
        return []
    out = []
    j = 0
    n = len(values_time)
    for t in target_time:
        while j + 1 < n and values_time[j + 1] < t:
            j += 1
        if t < values_time[0] or t > values_time[-1] or j + 1 >= n:
            out.append(math.nan)
            continue
        t0, t1 = values_time[j], values_time[j + 1]
        v0, v1 = values[j], values[j + 1]
        if not (math.isfinite(v0) and math.isfinite(v1)) or t1 == t0:
            out.append(math.nan)
            continue
        a = (t - t0) / (t1 - t0)
        out.append((1.0 - a) * v0 + a * v1)
    return out


def corr(a: list[float], b: list[float]) -> float:
    a, b = finite_pairs(a, b)
    if len(a) < 3:
        return math.nan
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    va = sum(x * x for x in da)
    vb = sum(x * x for x in db)
    if va <= 0.0 or vb <= 0.0:
        return math.nan
    return sum(x * y for x, y in zip(da, db)) / math.sqrt(va * vb)


def peak(values: list[float]) -> float:
    finite = [abs(v) for v in values if math.isfinite(v)]
    return max(finite, default=0.0)


def summarize_against_falln(case: dict, falln: dict) -> dict:
    common_time = [
        t for t in falln["time"]
        if case["time"][0] <= t <= case["time"][-1]
    ]
    best = []
    for oc, oval in case["components"].items():
        o_interp = interp(case["time"], oval, common_time)
        op = peak(o_interp)
        best_corr = {"falln_component": None, "corr": math.nan, "peak_ratio_open_to_falln": math.nan}
        for fc, fval in falln["components"].items():
            f_interp = interp(falln["time"], fval, common_time)
            c = corr(o_interp, f_interp)
            if math.isnan(c):
                continue
            if best_corr["falln_component"] is None or abs(c) > abs(best_corr["corr"]):
                fp = peak(f_interp)
                best_corr = {
                    "falln_component": fc,
                    "corr": c,
                    "peak_ratio_open_to_falln": op / fp if fp > 0.0 else math.nan,
                }
        best.append({
            "opensees_component": oc,
            "peak_abs": op,
            **best_corr,
        })
    best.sort(key=lambda row: 0.0 if math.isnan(row["corr"]) else abs(row["corr"]), reverse=True)
    return {
        "label": case["label"],
        "path": case["path"],
        "samples": len(case["time"]),
        "component_count": case["component_count"],
        "shared_samples": len(common_time),
        "top_matches": best[:12],
    }


def plot_top_matches(summary: dict, out_dir: Path) -> None:
    rows = []
    for case in summary["cases"]:
        for match in case["top_matches"][:6]:
            if match["falln_component"] is None or math.isnan(match["corr"]):
                continue
            rows.append((case["label"], match["opensees_component"], match["falln_component"], match["corr"]))
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{label}\n{oc}->{fc}" for label, oc, fc, _c in rows]
    values = [c for _label, _oc, _fc, c in rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.38 * len(rows)), 4.8))
    colors = ["#3572A5" if v >= 0 else "#B24C3A" for v in values]
    ax.bar(range(len(rows)), values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("correlacion de Pearson")
    ax.set_title("Auditoria de convencion: mejor emparejamiento de componentes de fuerza")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "lshaped_16_element_force_convention_audit.pdf")
    fig.savefig(out_dir / "lshaped_16_element_force_convention_audit.png")
    plt.close(fig)


def parse_case_arg(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.name, path
    label, path = text.split("=", 1)
    return label, Path(path)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--falln",
        default=str(root / "data/output/lshaped_multiscale_16/recorders/selected_element_0_global_force.csv"),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="OpenSees case directory, or label=directory. Can be repeated.",
    )
    parser.add_argument("--out-dir", default=str(root / "doc/figures/validation_reboot"))
    args = parser.parse_args()

    falln = read_force_history(Path(args.falln), "fall_n selected element")
    if falln is None:
        raise SystemExit(f"Could not read fall_n selected-element force history: {args.falln}")

    case_args = args.case or [
        str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_elastic_timoshenko_sub3_elementmass_clean"),
        str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_10s_force_shear_sub1_nodalmass"),
        str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_1s_elastic_timoshenko_sub3_response_audit"),
        str(root / "data/output/opensees_lshaped_16storey/scale1p0_window87p65_1s_force_shear_response_audit"),
    ]
    open_cases = []
    for item in case_args:
        label, path = parse_case_arg(item)
        open_cases.extend(discover_opensees_case(path, label))
    if not open_cases:
        raise SystemExit("No OpenSees selected-element force histories found.")

    summary = {
        "schema": "lshaped_16_element_force_convention_audit_v1",
        "falln": {
            "path": falln["path"],
            "samples": len(falln["time"]),
            "component_count": falln["component_count"],
            "note": "fall_n vector is the selected element internal-force vector in element DOF order.",
        },
        "cases": [summarize_against_falln(case, falln) for case in open_cases],
        "notes": [
            "High correlation is a convention clue, not proof of physical equivalence.",
            "OpenSees forceBeamColumn recorders can expose global/local/basic response vectors with different ordering and size.",
            "Peak ratios should be interpreted only after component convention and sign are closed.",
        ],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lshaped_16_element_force_convention_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_top_matches(summary, out_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
