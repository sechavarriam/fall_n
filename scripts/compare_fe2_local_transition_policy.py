#!/usr/bin/env python3
"""Compare fixed and adaptive managed-XFEM local transition policies.

The script is intentionally light-weight: it reads the FE2 recorder CSV files
that are already emitted by the 16-storey campaign and produces a partial-safe
summary.  It can be run while simulations are still active; only common
accepted times are compared.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def rows_by_time(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {
        row["time"]: row
        for row in rows
        if row.get("phase", "accepted_step") in {"accepted_step", "initialization"}
        and row.get("time")
    }


def transition_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if not row.get("transition_steps"):
            continue
        key = (
            f"{row.get('transition_reason','unknown')}:"
            f"{row.get('transition_steps','?')}/"
            f"{row.get('transition_max_bisections','?')}"
        )
        counts[key] += 1
    return dict(counts.most_common())


def basic_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": float(len(values)),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed-coupling", required=True, type=Path)
    ap.add_argument("--adaptive-coupling", required=True, type=Path)
    ap.add_argument("--adaptive-boundary", required=True, type=Path)
    ap.add_argument("--output-prefix", required=True, type=Path)
    args = ap.parse_args()

    fixed_rows = read_csv(args.fixed_coupling)
    adaptive_rows = read_csv(args.adaptive_coupling)
    adaptive_boundary = read_csv(args.adaptive_boundary)

    fixed_by_time = rows_by_time(fixed_rows)
    adaptive_by_time = rows_by_time(adaptive_rows)
    common_times = sorted(
        set(fixed_by_time).intersection(adaptive_by_time),
        key=lambda x: float(x),
    )

    common: list[dict[str, Any]] = []
    for time in common_times:
        fr = fixed_by_time[time]
        ar = adaptive_by_time[time]
        common.append(
            {
                "time": float(time),
                "fixed_local_seconds": f(fr, "local_total_solve_seconds"),
                "adaptive_local_seconds": f(ar, "local_total_solve_seconds"),
                "fixed_iterations": f(fr, "iterations"),
                "adaptive_iterations": f(ar, "iterations"),
                "fixed_force_residual": f(fr, "max_force_residual_rel"),
                "adaptive_force_residual": f(ar, "max_force_residual_rel"),
                "fixed_tangent_column_residual": f(
                    fr, "max_tangent_column_residual_rel"
                ),
                "adaptive_tangent_column_residual": f(
                    ar, "max_tangent_column_residual_rel"
                ),
            }
        )

    local_ratio = [
        row["adaptive_local_seconds"] / row["fixed_local_seconds"]
        for row in common
        if row["fixed_local_seconds"] > 0.0 and row["adaptive_local_seconds"] > 0.0
    ]
    force_gap = [
        abs(row["adaptive_force_residual"] - row["fixed_force_residual"])
        for row in common
    ]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if common:
        t = [row["time"] for row in common]
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=True)
        axs[0].plot(t, [row["fixed_local_seconds"] for row in common], label="fixed")
        axs[0].plot(
            t,
            [row["adaptive_local_seconds"] for row in common],
            label="adaptive",
        )
        axs[0].set_ylabel("local solve seconds")
        axs[0].legend()
        axs[0].grid(True, alpha=0.25)

        axs[1].plot(t, [row["fixed_force_residual"] for row in common], label="fixed")
        axs[1].plot(
            t,
            [row["adaptive_force_residual"] for row in common],
            label="adaptive",
        )
        axs[1].set_ylabel("max force residual")
        axs[1].set_yscale("symlog", linthresh=1.0e-6)
        axs[1].grid(True, alpha=0.25)

        axs[2].plot(
            t,
            [row["fixed_tangent_column_residual"] for row in common],
            label="fixed",
        )
        axs[2].plot(
            t,
            [row["adaptive_tangent_column_residual"] for row in common],
            label="adaptive",
        )
        axs[2].set_ylabel("max tangent-column residual")
        axs[2].set_xlabel("time [s]")
        axs[2].grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(args.output_prefix.with_suffix(".pdf"))
        fig.savefig(args.output_prefix.with_suffix(".png"), dpi=180)
        plt.close(fig)

    summary = {
        "schema": "fe2_local_transition_policy_comparison_v1",
        "fixed_coupling": str(args.fixed_coupling),
        "adaptive_coupling": str(args.adaptive_coupling),
        "adaptive_boundary": str(args.adaptive_boundary),
        "common_time_count": len(common),
        "fixed_last_time": max((float(r["time"]) for r in fixed_rows if r.get("time")), default=0.0),
        "adaptive_last_time": max((float(r["time"]) for r in adaptive_rows if r.get("time")), default=0.0),
        "adaptive_transition_distribution": transition_distribution(adaptive_boundary),
        "adaptive_to_fixed_local_seconds_ratio": basic_stats(local_ratio),
        "absolute_force_residual_gap": basic_stats(force_gap),
        "common_rows": common[-12:],
    }
    with args.output_prefix.with_suffix(".json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
