#!/usr/bin/env python3
"""Generate moment-rotation figures for the RC beam validation chapter."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
data_dir = Path(r"C:\MyLibs\fall_n\data\output\rc_beam_validation")
out_dir  = Path(r"C:\MyLibs\fall_n\doc\figures\rc_beam_validation")
out_dir.mkdir(parents=True, exist_ok=True)

# ── load CSV data ──────────────────────────────────────────────────────
plain = np.loadtxt(data_dir / "moment_rotation_plain.csv",
                   delimiter=",", skiprows=1)
reinf = np.loadtxt(data_dir / "moment_rotation_reinforced.csv",
                   delimiter=",", skiprows=1)

# columns: step, p, theta_rad, My_MNm, max_disp_m, Fz_MN
theta_p  = plain[:, 2] * 1e3   # convert to mrad
My_p     = plain[:, 3] * 1e3   # convert to kN·m
theta_r  = reinf[:, 2] * 1e3
My_r     = reinf[:, 3] * 1e3

# ── style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

c_plain = "#1f77b4"
c_reinf = "#d62728"

# ── Analytical reference: elastic beam theory ──────────────────────────
# Doubly-clamped beam, rotation θ imposed at one end:
#   M_far = 4EI/L · θ   (at the rotating end)
# Geometry: b=0.30, h=0.40, L=2.0
# I_y = h·b³/12 = 0.40 × 0.30³ / 12 = 9.0e-4 m⁴
# Using initial tangent from simulation data to back-calculate E₀
b, h, L = 0.30, 0.40, 2.00
I_y = h * b**3 / 12.0  # bending about Y — strong axis in different sense

# Initial tangent stiffness from data (first non-zero step)
k0_plain = (My_p[1] / theta_p[1]) if theta_p[1] > 0 else 0  # kN·m / mrad
k0_reinf = (My_r[1] / theta_r[1]) if theta_r[1] > 0 else 0

theta_lin = np.linspace(0, theta_p[-1], 100)
My_elastic_plain = k0_plain * theta_lin
My_elastic_reinf = k0_reinf * theta_lin


# =====================================================================
#  Figure 1: Moment–rotation curves (main validation figure)
# =====================================================================
fig1, ax1 = plt.subplots(figsize=(5.5, 3.8))

ax1.plot(theta_p, My_p, '-o', color=c_plain, markersize=2.5,
         label="Plain concrete")
ax1.plot(theta_r, My_r, '-s', color=c_reinf, markersize=2.5,
         label="Reinforced concrete")
ax1.plot(theta_lin, My_elastic_plain, '--', color=c_plain, alpha=0.5,
         linewidth=0.8, label=f"Elastic (plain, $k_0$={k0_plain:.1f} kN$\\cdot$m/mrad)")
ax1.plot(theta_lin, My_elastic_reinf, '--', color=c_reinf, alpha=0.5,
         linewidth=0.8, label=f"Elastic (reinf., $k_0$={k0_reinf:.1f} kN$\\cdot$m/mrad)")

ax1.set_xlabel(r"Imposed rotation $\theta$ [mrad]")
ax1.set_ylabel(r"Reaction moment $M_y$ [kN$\cdot$m]")
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax1.legend(loc="upper left", framealpha=0.9)
ax1.set_title("Moment--rotation response: Ko-Bathe concrete 3D validation")

fig1.tight_layout()
for ext in ("pdf", "png"):
    fig1.savefig(out_dir / f"moment_rotation.{ext}", dpi=300)
print(f"  [1] moment_rotation saved  (peak plain={My_p[-1]:.1f}, reinf={My_r[-1]:.1f} kN·m)")


# =====================================================================
#  Figure 2: Tangent stiffness degradation
# =====================================================================
# Compute tangent stiffness as ΔM/Δθ
def tangent_stiffness(theta, My):
    """Central differences for interior, forward/backward at ends."""
    dt = np.diff(theta)
    dm = np.diff(My)
    k_tang = dm / np.where(dt > 0, dt, 1e-30)
    # Shift to midpoint θ values
    theta_mid = 0.5 * (theta[:-1] + theta[1:])
    return theta_mid, k_tang

theta_mid_p, k_p = tangent_stiffness(theta_p, My_p)
theta_mid_r, k_r = tangent_stiffness(theta_r, My_r)

fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))

ax2.plot(theta_mid_p, k_p, '-o', color=c_plain, markersize=2.5,
         label="Plain concrete")
ax2.plot(theta_mid_r, k_r, '-s', color=c_reinf, markersize=2.5,
         label="Reinforced concrete")

ax2.axhline(k0_plain, color=c_plain, ls=':', alpha=0.5, lw=0.8,
            label=f"Initial $k_0$ (plain) = {k0_plain:.1f}")
ax2.axhline(k0_reinf, color=c_reinf, ls=':', alpha=0.5, lw=0.8,
            label=f"Initial $k_0$ (reinf.) = {k0_reinf:.1f}")

ax2.set_xlabel(r"Rotation $\theta$ [mrad]")
ax2.set_ylabel(r"Tangent stiffness $dM/d\theta$ [kN$\cdot$m/mrad]")
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)
ax2.legend(loc="upper right", framealpha=0.9, fontsize=7)
ax2.set_title("Tangent stiffness degradation (cracking effect)")

fig2.tight_layout()
for ext in ("pdf", "png"):
    fig2.savefig(out_dir / f"stiffness_degradation.{ext}", dpi=300)
print(f"  [2] stiffness_degradation saved  (k0_p={k0_plain:.1f}, k_final_p={k_p[-1]:.1f})")


# =====================================================================
#  Figure 3: Normalised moment ratio (reinforced / plain)
# =====================================================================
# Since both share the same θ grid, we can directly compute the ratio
ratio = My_r[1:] / np.where(np.abs(My_p[1:]) > 1e-15, My_p[1:], 1e-15)

fig3, ax3 = plt.subplots(figsize=(5.5, 3.0))

ax3.plot(theta_p[1:], ratio, '-o', color="#2ca02c", markersize=2.5)
ax3.axhline(1.0, color='gray', ls='--', lw=0.7)
ax3.set_xlabel(r"Rotation $\theta$ [mrad]")
ax3.set_ylabel(r"$M_y^{\mathrm{reinf.}} / M_y^{\mathrm{plain}}$")
ax3.set_xlim(left=0)
ax3.set_ylim(0.95, max(ratio) * 1.05)
ax3.set_title("Moment ratio: reinforced vs.\ plain concrete")

fig3.tight_layout()
for ext in ("pdf", "png"):
    fig3.savefig(out_dir / f"moment_ratio.{ext}", dpi=300)
print(f"  [3] moment_ratio saved  (initial ratio={ratio[0]:.4f}, final={ratio[-1]:.4f})")


# ── Summary ────────────────────────────────────────────────────────────
print(f"\n  Summary:")
print(f"    Plain concrete:      M_max = {My_p[-1]:.2f} kN·m at θ = {theta_p[-1]:.3f} mrad")
print(f"    Reinforced concrete: M_max = {My_r[-1]:.2f} kN·m at θ = {theta_r[-1]:.3f} mrad")
print(f"    Stiffness ratio (reinforced/plain):")
print(f"      Initial k0: {k0_reinf/k0_plain:.4f}")
print(f"      Final tangent: {k_r[-1]/k_p[-1]:.4f}")
print(f"    Moment gain from rebar: {(My_r[-1]/My_p[-1] - 1)*100:.1f}%")
print(f"\n  All figures saved to: {out_dir}")
