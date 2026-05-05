#!/usr/bin/env python3
"""Plot the managed-XFEM one-way FE2 column replay summary.

The driver intentionally consumes the CSV/JSON artifacts emitted by
``fall_n_fe2_column_cyclic_one_way`` instead of re-running the analysis.  It is
therefore safe to use after long FE2 runs and keeps the thesis figures tied to
auditable recorder files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        rows: list[dict[str, float | str]] = []
        for row in csv.DictReader(f):
            converted: dict[str, float | str] = {}
            for key, value in row.items():
                try:
                    converted[key] = float(value)
                except (TypeError, ValueError):
                    converted[key] = value
            rows.append(converted)
    return rows


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=str(
            root
            / "data/output/fe2_validation/"
              "fe2_one_way_managed_xfem_n10_lobatto_200mm_3x3x6_current"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(root / "doc/figures/validation_reboot"),
    )
    parser.add_argument(
        "--prefix",
        default="fe2_one_way_managed_xfem_n10_lobatto_200mm_3x3x6",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (run_dir / "fe2_column_one_way_cyclic.json").read_text(
            encoding="utf-8", errors="ignore"
        )
    )
    activation = read_csv(run_dir / "site_activation.csv")
    response = read_csv(run_dir / "site_response.csv")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "figure.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    z = [float(r["z_over_l"]) for r in activation]
    drift = [float(r["trigger_drift_mm"]) for r in activation]
    steel = [float(r["peak_abs_steel_stress_mpa"]) for r in activation]
    macro_m = [float(r["peak_macro_moment_y_mn_m"]) for r in response]
    hom_m = [abs(float(r["moment_y_mn_m"])) for r in response]
    err = [100.0 * float(r["relative_moment_envelope_error"]) for r in response]
    iters = [float(r["snes_iters"]) for r in response]

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))

    axes[0].plot(z, drift, marker="o", label="drift de activacion")
    ax2 = axes[0].twinx()
    ax2.plot(z, steel, marker="s", color="tab:red", label="acero pico")
    axes[0].set_xlabel(r"$z/L$")
    axes[0].set_ylabel("drift [mm]")
    ax2.set_ylabel(r"$|\sigma_s|_{\max}$ [MPa]")
    axes[0].set_title("Activacion por sitio")

    width = 0.018
    axes[1].bar([zi - width / 2 for zi in z], macro_m, width=width, label="macro")
    axes[1].bar([zi + width / 2 for zi in z], hom_m, width=width, label="local XFEM")
    axes[1].set_xlabel(r"$z/L$")
    axes[1].set_ylabel(r"$|M_y|$ [MN m]")
    axes[1].set_title("Momento conjugado")
    axes[1].legend(fontsize=8)

    axes[2].bar(z, err, width=width, label="error envolvente [%]")
    ax4 = axes[2].twinx()
    ax4.plot(z, iters, marker="d", color="tab:green", label="SNES iters")
    axes[2].set_xlabel(r"$z/L$")
    axes[2].set_ylabel("error [%]")
    ax4.set_ylabel("iteraciones SNES")
    axes[2].set_title("Compuerta FE2")

    fig.suptitle(
        "FE2 one-way con XFEM gestionado: "
        f"{manifest['selected_site_count']} sitios, "
        f"pass={manifest['overall_pass']}"
    )
    fig.tight_layout()

    figures: list[str] = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{args.prefix}_summary_panel.{ext}"
        fig.savefig(path)
        figures.append(str(path))
    plt.close(fig)

    summary = {
        "schema": "fe2_one_way_managed_xfem_summary_panel_v1",
        "run_dir": str(run_dir),
        "overall_pass": manifest.get("overall_pass"),
        "selected_site_count": manifest.get("selected_site_count"),
        "max_relative_moment_error": max(err) / 100.0 if err else None,
        "max_snes_iters": max(iters) if iters else None,
        "figures": figures,
    }
    summary_path = out_dir / f"{args.prefix}_summary_panel.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
