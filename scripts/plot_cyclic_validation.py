#!/usr/bin/env python3
"""
plot_cyclic_validation.py — Comparative hysteresis plots for the cyclic
validation suite (v2 protocol: geometric amplitudes {2.5, 5, 10, 20} mm).

Cases:  0 (elastic), 1a-1i (beam N=2..10), 2a-2c (continuum),
        3 (table), 4 (FE² one-way), 5 (FE² two-way).

Usage:
    python scripts/plot_cyclic_validation.py [--dir data/output/cyclic_validation]

Produces:
    {dir}/comparison_beam.pdf       — Cases 0/1a/1b/1c/1d overlay
    {dir}/comparison_continuum.pdf  — Case 2a overlay
    {dir}/comparison_fe2.pdf        — Cases 3/4/5 overlay
    {dir}/comparison_all.pdf        — All cases overlay
    {dir}/protocol.pdf              — Cyclic displacement protocol

Also copies PDFs to doc/figures/cyclic_validation/ for LaTeX inclusion.
"""

import argparse
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Cyclic protocol v2 (matches CyclicProtocol.hh) ───────────────────
def cyclic_displacement(p):
    """Absolute geometric amplitudes: {2.5, 5, 10, 20} mm."""
    amps = [0.0025, 0.005, 0.010, 0.020]  # metres
    N_SEG = 12
    t = p * N_SEG
    seg = int(np.clip(t, 0, N_SEG - 1))
    f = t - seg
    level = seg // 3
    phase = seg % 3
    A = amps[level]
    if phase == 0:
        return f * A
    elif phase == 1:
        return A * (1.0 - 2.0 * f)
    else:
        return -A * (1.0 - f)


cyclic_displacement_v = np.vectorize(cyclic_displacement)


def load_case(base_dir, case_id):
    """Load hysteresis CSV for a given case; return (drift, shear) arrays."""
    path = os.path.join(base_dir, f"case{case_id}", "hysteresis.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    return df["drift_m"].values, df["base_shear_MN"].values


def plot_protocol(base_dir):
    """Plot the cyclic displacement protocol."""
    p = np.linspace(0, 1, 1000)
    d = cyclic_displacement_v(p) * 1e3  # mm

    # Also show the 120 discrete step markers
    p_steps = np.linspace(0, 1, 121)
    d_steps = cyclic_displacement_v(p_steps) * 1e3

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(p, d, "k-", linewidth=1.2)
    ax.plot(p_steps, d_steps, "o", color="#1f77b4", markersize=2, alpha=0.5)
    ax.set_xlabel("Control parameter $p$")
    ax.set_ylabel("Drift $d(p)$ [mm]")
    ax.set_title(
        "Cyclic protocol v2: $\\pm\\{2.5, 5, 10, 20\\}$ mm"
        " (geometric progression)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(base_dir, "protocol.pdf"), dpi=150)
    plt.close(fig)
    print(f"  Written: {base_dir}/protocol.pdf")


def plot_group(base_dir, case_ids, labels, colors, title, filename):
    """Overlay hysteresis loops for a group of cases."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    for cid, lbl, col in zip(case_ids, labels, colors):
        data = load_case(base_dir, cid)
        if data is None:
            continue
        drift_mm = data[0] * 1e3
        shear_kN = data[1] * 1e3  # MN → kN
        ax.plot(drift_mm, shear_kN, color=col, linewidth=1.0,
                label=lbl, alpha=0.85)

    ax.set_xlabel("Lateral drift [mm]")
    ax.set_ylabel("Base shear [kN]")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(base_dir, filename)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Written: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cyclic validation hysteresis curves (v2 protocol)")
    parser.add_argument(
        "--dir", default="data/output/cyclic_validation",
        help="Base output directory")
    parser.add_argument(
        "--figures-dir", default="doc/figures/cyclic_validation",
        help="LaTeX figures directory (PDFs are copied here)")
    args = parser.parse_args()

    base = args.dir
    if not os.path.isdir(base):
        print(f"Directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    fig_dir = args.figures_dir
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Plotting from: {base}")

    # Protocol
    plot_protocol(base)

    # Case 1: Beam comparison (including elastic reference)
    plot_group(
        base,
        case_ids=["0", "1a", "1b", "1c", "1d"],
        labels=[
            "Case 0: Elastic (N=3)",
            "Case 1a: N=2 (1 GP)",
            "Case 1b: N=3 (2 GPs)",
            "Case 1c: N=4 (3 GPs)",
            "Case 1d: N=5 (4 GPs)",
        ],
        colors=["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        title="Beam convergence study (single cantilever column)",
        filename="comparison_beam.pdf")

    # Case 2: Continuum comparison
    plot_group(
        base,
        case_ids=["0", "2a"],
        labels=["Case 0: Elastic reference", "Case 2a: Hex8 (no rebar)"],
        colors=["#7f7f7f", "#d62728"],
        title="Continuum column: Hex8 vs elastic reference",
        filename="comparison_continuum.pdf")

    # Case 3/4/5: Structural vs FE² comparison
    plot_group(
        base,
        case_ids=["3", "4", "5"],
        labels=[
            "Case 3: Table (fiber beams)",
            "Case 4: Table + FE² (one-way)",
            "Case 5: Table + FE² (two-way)",
        ],
        colors=["#000000", "#e377c2", "#17becf"],
        title="Structural model vs FE² multiscale coupling",
        filename="comparison_fe2.pdf")

    # All cases overlay (split by scale)
    plot_group(
        base,
        case_ids=["0", "1a", "1b", "1c", "1d", "2a"],
        labels=[
            "0: Elastic", "1a: Beam N=2", "1b: Beam N=3",
            "1c: Beam N=4", "1d: Beam N=5", "2a: Hex8",
        ],
        colors=[
            "#7f7f7f", "#1f77b4", "#ff7f0e",
            "#2ca02c", "#d62728", "#9467bd",
        ],
        title="Single-column cases comparison",
        filename="comparison_all.pdf")

    # Copy PDFs to figures directory for LaTeX
    pdf_names = [
        "protocol.pdf", "comparison_beam.pdf",
        "comparison_continuum.pdf", "comparison_fe2.pdf",
        "comparison_all.pdf",
    ]
    for name in pdf_names:
        src = os.path.join(base, name)
        if os.path.isfile(src):
            dst = os.path.join(fig_dir, name)
            shutil.copy2(src, dst)
            print(f"  Copied → {dst}")

    print("Done.")


if __name__ == "__main__":
    main()
