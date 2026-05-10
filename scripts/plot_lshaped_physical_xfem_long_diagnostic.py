#!/usr/bin/env python3
"""Plot the physical scale=1 XFEM long diagnostic frontier."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "data" / "output" / "lshaped_16storey_physical_scale1_xfem_linear_steel_alarm_10s_diag_20260509"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_failure(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, object] = {
        "failure_detected": False,
        "failure_time_s": None,
        "failure_reason": "",
        "failed_submodels": None,
        "evolution_steps": None,
        "summary_final_time_s": None,
        "summary_peak_damage": None,
        "summary_active_cracks": None,
    }
    match = re.search(r"Multiscale step failed at t=([0-9.eE+-]+) s\s+reason=([^,\n]+).*failed_submodels=([0-9]+)", text)
    if match:
        out["failure_detected"] = True
        out["failure_time_s"] = float(match.group(1))
        out["failure_reason"] = match.group(2)
        out["failed_submodels"] = int(match.group(3))
    patterns = {
        "evolution_steps": r"Evolution:\s+([0-9]+)\s+steps",
        "summary_final_time_s": r"t_final\s+=\s+([0-9.eE+-]+)\s+s",
        "summary_peak_damage": r"Peak damage:\s+([0-9.eE+-]+)",
        "summary_active_cracks": r"Active cracks:\s+([0-9]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            out[key] = int(m.group(1)) if key in {"evolution_steps", "summary_active_cracks"} else float(m.group(1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "doc" / "figures" / "validation_reboot")
    parser.add_argument("--prefix", default="lshaped_16_physical_scale1_xfem_long_diagnostic_frontier")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    history = read_csv(run_dir / "recorders" / "global_history.csv")
    cracks = read_csv(run_dir / "recorders" / "crack_evolution.csv")
    audit = json.loads((run_dir / "recorders" / "publication_vtk_audit.json").read_text(encoding="utf-8"))
    fe2_summary = json.loads((run_dir / "recorders" / "seismic_fe2_one_way_summary.json").read_text(encoding="utf-8"))
    failure = parse_failure(run_dir / "run_stdout.log")

    t = [float(r["time"]) for r in history]
    u_inf = [float(r["u_inf"]) for r in history]
    damage = [float(r["peak_damage"]) for r in history]
    steps = [int(float(r["step"])) for r in history]
    dt = [0.0] + [max(t[i] - t[i - 1], 1.0e-12) for i in range(1, len(t))]

    tc = [float(r["time"]) for r in cracks]
    total_cracks = [float(r["total_cracks"]) for r in cracks]
    max_opening_mm = [float(r["max_opening"]) * 1e3 for r in cracks]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    ax = axes[0, 0]
    ax.plot(t, u_inf, lw=0.8, color="#1f77b4")
    ax.set_title("Global response norm, diagnostic only")
    ax.set_ylabel(r"$\|u\|_\infty$ (m)")
    ax.grid(True, alpha=0.35, lw=0.3)

    ax = axes[0, 1]
    ax.plot(t, damage, lw=0.8, color="#8c2d04")
    ax.set_title("Peak macro damage index")
    ax.set_ylabel("damage index")
    ax.grid(True, alpha=0.35, lw=0.3)

    ax = axes[1, 0]
    ax.plot(tc, total_cracks, lw=0.8, color="#4c78a8", label="cracks")
    ax2 = ax.twinx()
    ax2.plot(tc, max_opening_mm, lw=0.8, color="#d62728", label="opening")
    ax.set_title("Local XFEM crack evolution")
    ax.set_xlabel("Physical time since record window start (s)")
    ax.set_ylabel("crack count")
    ax2.set_ylabel("max opening (mm)")
    ax.grid(True, alpha=0.35, lw=0.3)

    ax = axes[1, 1]
    ax.plot(t[1:], dt[1:], lw=0.8, color="#2ca02c")
    ax.set_yscale("log")
    ax.set_title("Adaptive step size before local failure")
    ax.set_xlabel("Physical time since record window start (s)")
    ax.set_ylabel("accepted dt (s)")
    ax.grid(True, alpha=0.35, lw=0.3)

    if failure["failure_time_s"] is not None:
        for ax in axes.ravel():
            ax.axvline(float(failure["failure_time_s"]), color="k", ls="--", lw=0.8)

    fig.suptitle("Physical scale=1 FE2/XFEM long diagnostic frontier", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.output_dir / f"{args.prefix}.pdf"
    png = args.output_dir / f"{args.prefix}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=180)
    plt.close(fig)

    final_accepted_time = t[-1]
    summary = {
        "schema": "fall_n_lshaped_physical_scale1_xfem_long_diagnostic_frontier_v1",
        "run_dir": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
        "eq_scale": fe2_summary["eq_scale"],
        "selected_site_count": fe2_summary["selected_site_count"],
        "completed_site_count": fe2_summary["completed_site_count"],
        "final_accepted_time_s": final_accepted_time,
        "final_accepted_step": steps[-1],
        "max_u_inf_m": max(u_inf),
        "final_u_inf_m": u_inf[-1],
        "max_peak_damage": max(damage),
        "final_peak_damage": damage[-1],
        "max_crack_opening_m": max(v / 1e3 for v in max_opening_mm) if max_opening_mm else 0.0,
        "final_total_cracks": int(total_cracks[-1]) if total_cracks else 0,
        "failure": failure,
        "vtk_issue_count": len(audit["issues"]),
        "vtk_endpoint_max_gap_m": audit["local_global_endpoint_max_gap_m"],
        "vtk_audit_promotable": len(audit["issues"]) == 0,
        "vtk_failure_note": "Final VTK snapshot is not publication-promotable when issue_count > 0.",
        "figures": [str(pdf.relative_to(ROOT)), str(png.relative_to(ROOT))],
    }
    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
