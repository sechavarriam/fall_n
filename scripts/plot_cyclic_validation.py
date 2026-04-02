#!/usr/bin/env python3
"""
plot_cyclic_validation.py — Comparative hysteresis plots for the cyclic
validation suite (Cases 1a-1c, 2a-2c, 3, 4, 5).

Usage:
    python scripts/plot_cyclic_validation.py [--dir data/output/cyclic_validation]

Produces:
    {dir}/comparison_beam.pdf       — Cases 1a/1b/1c overlay
    {dir}/comparison_continuum.pdf  — Cases 2a/2b/2c overlay
    {dir}/comparison_fe2.pdf        — Cases 3/4/5 overlay (structural vs FE²)
    {dir}/comparison_all.pdf        — All cases overlay
    {dir}/protocol.pdf              — Cyclic displacement protocol
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Cyclic protocol (must match C++ cyclic_displacement()) ────────────
def cyclic_displacement(p, delta_y=0.01):
    amps = [1.0, 2.0, 4.0, 8.0]
    N_SEG = 12
    t = p * N_SEG
    seg = int(np.clip(t, 0, N_SEG - 1))
    f = t - seg
    level = seg // 3
    phase = seg % 3
    A = amps[level] * delta_y
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

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(p, d, "k-", linewidth=1.2)
    ax.set_xlabel("Control parameter $p$")
    ax.set_ylabel("Drift $d(p)$ [mm]")
    ax.set_title("Cyclic displacement protocol: $\\pm 1/2/4/8\\,\\delta_y$")
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
        description="Plot cyclic validation hysteresis curves")
    parser.add_argument(
        "--dir", default="data/output/cyclic_validation",
        help="Base output directory")
    args = parser.parse_args()

    base = args.dir
    if not os.path.isdir(base):
        print(f"Directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    print(f"Plotting from: {base}")

    # Protocol
    plot_protocol(base)

    # Case 1: Beam comparison
    plot_group(
        base,
        case_ids=["1a", "1b", "1c"],
        labels=["N=2 (linear)", "N=3 (quadratic)", "N=4 (cubic)"],
        colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
        title="Case 1: Single beam — TimoshenkoBeamN<N>",
        filename="comparison_beam.pdf")

    # Case 2: Continuum comparison
    plot_group(
        base,
        case_ids=["2a", "2b", "2c"],
        labels=["Hex8 (linear)", "Hex20 (serendipity)", "Hex27 (quadratic)"],
        colors=["#d62728", "#9467bd", "#8c564b"],
        title="Case 2: Single continuum column — Hex only",
        filename="comparison_continuum.pdf")

    # Case 3/4/5: Structural vs FE² comparison
    plot_group(
        base,
        case_ids=["3", "4", "5"],
        labels=[
            "3: Table (fiber beams)",
            "4: Table + FE² (one-way)",
            "5: Table + FE² (two-way)",
        ],
        colors=["#000000", "#e377c2", "#17becf"],
        title="Structural model vs FE² multiscale coupling",
        filename="comparison_fe2.pdf")

    # All cases overlay
    plot_group(
        base,
        case_ids=["1a", "1b", "1c", "2a", "2b", "2c", "3", "4", "5"],
        labels=[
            "1a: Beam N=2", "1b: Beam N=3", "1c: Beam N=4",
            "2a: Hex8",     "2b: Hex20",    "2c: Hex27",
            "3: Full table",
            "4: FE² one-way",
            "5: FE² two-way",
        ],
        colors=[
            "#1f77b4", "#ff7f0e", "#2ca02c",
            "#d62728", "#9467bd", "#8c564b",
            "#000000", "#e377c2", "#17becf",
        ],
        title="Cyclic validation — All cases comparison",
        filename="comparison_all.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
