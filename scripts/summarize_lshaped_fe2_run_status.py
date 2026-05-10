#!/usr/bin/env python3
"""Summarize a physical L-shaped FE2 run while it is still running or after exit."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv_tail(path: Path, max_rows: int = 100000) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) > max_rows:
        return rows[-max_rows:]
    return rows


def as_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def parse_log(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "exists": False,
            "failure_detected": False,
            "completed": False,
            "failure_time_s": None,
            "failure_reason": "",
            "failed_submodels": None,
        }
    text = path.read_text(encoding="utf-8", errors="replace")
    failure = re.search(
        r"Multiscale step failed at t=([0-9.eE+-]+) s\s+reason=([^,\n]+).*failed_submodels=([0-9]+)",
        text,
    )
    has_final_summary = bool(re.search(r"16-STORY L-SHAPED MULTISCALE SEISMIC ANALYSIS.*SUMMARY", text, re.IGNORECASE))
    return {
        "exists": True,
        "failure_detected": bool(failure),
        "completed": has_final_summary and not bool(failure),
        "failure_time_s": float(failure.group(1)) if failure else None,
        "failure_reason": failure.group(2).strip() if failure else "",
        "failed_submodels": int(failure.group(3)) if failure else None,
    }


def load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(run_dir: Path) -> dict[str, object]:
    recorders = run_dir / "recorders"
    history = read_csv_tail(recorders / "global_history.csv")
    cracks = read_csv_tail(recorders / "crack_evolution.csv")
    fe2 = load_json(recorders / "seismic_fe2_one_way_summary.json")
    audit = load_json(recorders / "publication_vtk_audit.json")
    log = parse_log(run_dir / "run_stdout.log")

    summary: dict[str, object] = {
        "schema": "fall_n_lshaped_fe2_run_status_v1",
        "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
        "has_global_history": bool(history),
        "has_crack_evolution": bool(cracks),
        "log": log,
        "fe2": {
            "eq_scale": fe2.get("eq_scale"),
            "selected_site_count": fe2.get("selected_site_count"),
            "completed_site_count": fe2.get("completed_site_count"),
        },
    }

    if history:
        times = [as_float(row, "time") for row in history]
        u_inf = [as_float(row, "u_inf") for row in history]
        damage = [as_float(row, "peak_damage") for row in history]
        last = history[-1]
        dt = times[-1] - times[-2] if len(times) > 1 else 0.0
        summary["global"] = {
            "last_time_s": times[-1],
            "last_step": as_int(last, "step"),
            "last_phase": as_int(last, "phase"),
            "last_u_inf_m": u_inf[-1],
            "max_u_inf_m": max(u_inf),
            "last_peak_damage": damage[-1],
            "max_peak_damage": max(damage),
            "last_dt_s": dt,
            "samples": len(history),
        }

    if cracks:
        openings = [as_float(row, "max_opening") for row in cracks]
        total_cracks = [as_int(row, "total_cracks") for row in cracks]
        total_cracked_gps = [as_int(row, "total_cracked_gps") for row in cracks]
        last = cracks[-1]
        summary["cracks"] = {
            "last_time_s": as_float(last, "time"),
            "last_total_cracks": total_cracks[-1],
            "max_total_cracks": max(total_cracks),
            "last_cracked_gps": total_cracked_gps[-1],
            "max_cracked_gps": max(total_cracked_gps),
            "last_max_opening_m": openings[-1],
            "max_opening_m": max(openings),
            "samples": len(cracks),
        }

    if audit:
        summary["vtk_audit"] = {
            "issue_count": len(audit.get("issues", [])),
            "endpoint_max_gap_m": audit.get("local_global_endpoint_max_gap_m"),
            "promotable": len(audit.get("issues", [])) == 0,
        }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.run_dir.resolve())
    text = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
