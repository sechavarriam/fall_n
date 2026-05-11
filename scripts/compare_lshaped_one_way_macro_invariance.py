#!/usr/bin/env python3
"""Compare a FE2 one-way run against a global-only macro reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_table(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            rows.append([float(value) for value in row])
    return header, rows


def column_map(header: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(header)}


def interpolate(
    times: list[float],
    values: list[float],
    query: float,
) -> float | None:
    if not times or query < times[0] or query > times[-1]:
        return None
    pos = bisect_right(times, query)
    if pos == 0:
        return values[0]
    if pos >= len(times):
        return values[-1]
    t0, t1 = times[pos - 1], times[pos]
    y0, y1 = values[pos - 1], values[pos]
    if abs(t1 - t0) <= 1.0e-15:
        return y0
    alpha = (query - t0) / (t1 - t0)
    return y0 + alpha * (y1 - y0)


def compare_roof(fe2_dir: Path, global_dir: Path) -> dict[str, object]:
    fe2_header, fe2_rows = read_table(fe2_dir / "recorders" / "roof_displacement.csv")
    glob_header, glob_rows = read_table(
        global_dir / "recorders" / "roof_displacement.csv"
    )
    fe2_cols = column_map(fe2_header)
    glob_cols = column_map(glob_header)
    common = [
        name
        for name in fe2_header
        if name != "time" and name in glob_cols
    ]
    if not common:
        raise RuntimeError("No common roof displacement channels found.")

    fe2_time = [row[fe2_cols["time"]] for row in fe2_rows]
    glob_time = [row[glob_cols["time"]] for row in glob_rows]
    t0 = max(min(fe2_time), min(glob_time))
    t1 = min(max(fe2_time), max(glob_time))
    sample_times = [t for t in fe2_time if t0 <= t <= t1]

    channel_metrics: dict[str, dict[str, float]] = {}
    component_metrics: dict[str, dict[str, float]] = {
        "dof0": {"max_abs_m": 0.0, "rms_abs_m": 0.0, "count": 0.0},
        "dof1": {"max_abs_m": 0.0, "rms_abs_m": 0.0, "count": 0.0},
        "dof2": {"max_abs_m": 0.0, "rms_abs_m": 0.0, "count": 0.0},
    }
    global_max = 0.0
    global_rms_acc = 0.0
    global_count = 0

    for name in common:
        fe2_values = [row[fe2_cols[name]] for row in fe2_rows]
        glob_values = [row[glob_cols[name]] for row in glob_rows]
        max_abs = 0.0
        rms_acc = 0.0
        ref_max = max(abs(v) for v in glob_values) if glob_values else 0.0
        count = 0
        for t in sample_times:
            fe2_value = interpolate(fe2_time, fe2_values, t)
            glob_value = interpolate(glob_time, glob_values, t)
            if fe2_value is None or glob_value is None:
                continue
            diff = fe2_value - glob_value
            abs_diff = abs(diff)
            max_abs = max(max_abs, abs_diff)
            rms_acc += diff * diff
            count += 1
        rms = math.sqrt(rms_acc / count) if count else math.nan
        rel = max_abs / max(ref_max, 1.0e-15)
        channel_metrics[name] = {
            "max_abs_m": max_abs,
            "rms_abs_m": rms,
            "relative_to_reference_peak": rel,
            "samples": count,
        }
        global_max = max(global_max, max_abs)
        global_rms_acc += rms_acc
        global_count += count
        for suffix, comp in component_metrics.items():
            if name.endswith("_" + suffix):
                comp["max_abs_m"] = max(comp["max_abs_m"], max_abs)
                comp["rms_abs_m"] += rms_acc
                comp["count"] += count

    for comp in component_metrics.values():
        count = comp.pop("count")
        comp["rms_abs_m"] = math.sqrt(comp["rms_abs_m"] / count) if count else math.nan

    return {
        "overlap_time_s": [t0, t1],
        "sample_count": len(sample_times),
        "channel_count": len(common),
        "max_abs_m": global_max,
        "rms_abs_m": math.sqrt(global_rms_acc / global_count)
        if global_count
        else math.nan,
        "components": component_metrics,
        "channels": channel_metrics,
    }


def compare_global_history(fe2_dir: Path, global_dir: Path) -> dict[str, object] | None:
    fe2_path = fe2_dir / "recorders" / "global_history.csv"
    glob_path = global_dir / "recorders" / "global_history.csv"
    if not fe2_path.exists() or not glob_path.exists():
        return None
    fe2_header, fe2_rows = read_table(fe2_path)
    glob_header, glob_rows = read_table(glob_path)
    fe2_cols = column_map(fe2_header)
    glob_cols = column_map(glob_header)
    metrics: dict[str, object] = {}
    fe2_time = [row[fe2_cols["time"]] for row in fe2_rows]
    glob_time = [row[glob_cols["time"]] for row in glob_rows]
    t0 = max(min(fe2_time), min(glob_time))
    t1 = min(max(fe2_time), max(glob_time))
    sample_times = [t for t in fe2_time if t0 <= t <= t1]
    for name in ("u_inf", "peak_damage"):
        if name not in fe2_cols or name not in glob_cols:
            continue
        fe2_values = [row[fe2_cols[name]] for row in fe2_rows]
        glob_values = [row[glob_cols[name]] for row in glob_rows]
        max_abs = 0.0
        for t in sample_times:
            fe2_value = interpolate(fe2_time, fe2_values, t)
            glob_value = interpolate(glob_time, glob_values, t)
            if fe2_value is None or glob_value is None:
                continue
            max_abs = max(max_abs, abs(fe2_value - glob_value))
        metrics[name] = {"max_abs": max_abs}
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fe2-run-dir", type=Path, required=True)
    parser.add_argument("--global-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roof-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    fe2_dir = args.fe2_run_dir.resolve()
    global_dir = args.global_run_dir.resolve()
    roof = compare_roof(fe2_dir, global_dir)
    global_history = compare_global_history(fe2_dir, global_dir)
    passed = roof["max_abs_m"] <= args.roof_tolerance
    summary = {
        "schema": "fall_n_lshaped_one_way_macro_invariance_v1",
        "fe2_run_dir": str(fe2_dir.relative_to(ROOT)),
        "global_run_dir": str(global_dir.relative_to(ROOT)),
        "roof_tolerance_m": args.roof_tolerance,
        "passed": passed,
        "roof_displacement": roof,
        "global_history": global_history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
