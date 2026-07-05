"""
Cyclic uniaxial evaluation of the three concrete proxies used in the
RC continuum pilots of fall_n.

Models (path-independent in this script — all three proxies are evaluated as
pure functions of the current axial strain, matching the C++ implementation):

  1. Isotropic elastic (linear).
  2. Orthotropic bimodular elastic (different E_c, E_t, no yield knee).
  3. Orthotropic bilinear bimodular (with tensile and compressive yield knees,
     post-yield hardening / softening multipliers h_c, h_t).

Output:
  - proxy_envelope_cases.csv         (per-case scalar summary)
  - proxy_envelope_sweep.csv         (sigma(epsilon) on a dense monotonic sweep)
  - proxy_cyclic_history.csv         (proxy responses for a representative
                                      ±0.6% cyclic uniaxial protocol)
  - proxy_bilinear_anisotropic_overlay.pdf / .png

The three proxies share a common compressive initial modulus E_c = 30 GPa and
Poisson-like ratio nu = 0.20, so that the small-strain shear branch is
identical across models. The non-trivial parameters are the tensile modulus
ratio, the yield surrogates (fc, ft), and the post-yield ratios (h_c, h_t).

The numbers reported in proxy_envelope_cases.csv are consumed by the table
tab:validacion-proxy-bilineal in PhD_Thesis/capitulos/9a_modelos_especificos_y_validacion.tex.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Material parameters (representative C25/30-like proxy data, MPa)
# ---------------------------------------------------------------------------
E_C = 30000.0
NU = 0.20
TENSION_RATIO = 1.0
FC = 25.0
FT = 2.5
H_C = 0.05
H_T = 0.01


@dataclass(frozen=True)
class ProxyConfig:
    label: str
    short: str
    E_c: float
    E_t: float
    f_c: float
    f_t: float
    h_c: float
    h_t: float

    def stress(self, eps: np.ndarray) -> np.ndarray:
        s = np.zeros_like(eps)
        pos = eps >= 0.0
        neg = ~pos

        if self.f_t > 0.0:
            eps_yt = self.f_t / self.E_t
        else:
            eps_yt = np.inf
        if self.f_c > 0.0:
            eps_yc = self.f_c / self.E_c
        else:
            eps_yc = np.inf

        eps_pos = eps[pos]
        pre_t = eps_pos <= eps_yt
        post_t = ~pre_t
        s_pos = np.empty_like(eps_pos)
        s_pos[pre_t] = self.E_t * eps_pos[pre_t]
        s_pos[post_t] = self.f_t + self.h_t * self.E_t * (
            eps_pos[post_t] - eps_yt
        )
        s[pos] = s_pos

        eps_neg = eps[neg]
        pre_c = -eps_neg <= eps_yc
        post_c = ~pre_c
        s_neg = np.empty_like(eps_neg)
        s_neg[pre_c] = self.E_c * eps_neg[pre_c]
        s_neg[post_c] = -self.f_c + self.h_c * self.E_c * (
            eps_neg[post_c] + eps_yc
        )
        s[neg] = s_neg

        return s

    def tangent(self, eps: np.ndarray) -> np.ndarray:
        t = np.full_like(eps, self.E_c)
        if self.f_t > 0.0:
            eps_yt = self.f_t / self.E_t
        else:
            eps_yt = np.inf
        if self.f_c > 0.0:
            eps_yc = self.f_c / self.E_c
        else:
            eps_yc = np.inf

        pos = eps >= 0.0
        t[pos] = self.E_t
        t[(eps > eps_yt)] = self.h_t * self.E_t
        t[(eps < -eps_yc)] = self.h_c * self.E_c
        return t


# Three configurations to compare.
ISOTROPIC = ProxyConfig(
    label="Isotr\u00f3pico el\u00e1stico",
    short="isotropic",
    E_c=E_C,
    E_t=E_C,
    f_c=0.0,
    f_t=0.0,
    h_c=1.0,
    h_t=1.0,
)
BIMODULAR = ProxyConfig(
    label="Bimodular el\u00e1stico",
    short="bimodular",
    E_c=E_C,
    E_t=0.1 * E_C,   # E_t = 0.1 E_c captures concrete tensile weakness
    f_c=0.0,
    f_t=0.0,
    h_c=1.0,
    h_t=1.0,
)
BILINEAR = ProxyConfig(
    label="Bilineal bimodular",
    short="bilinear",
    E_c=E_C,
    E_t=TENSION_RATIO * E_C,
    f_c=FC,
    f_t=FT,
    h_c=H_C,
    h_t=H_T,
)

CONFIGS = [ISOTROPIC, BIMODULAR, BILINEAR]


# ---------------------------------------------------------------------------
# Cyclic protocol
# ---------------------------------------------------------------------------
def cyclic_strain_history(
    peaks_microstrain: list[float],
    samples_per_segment: int = 250,
) -> np.ndarray:
    history = [0.0]
    current = 0.0
    for peak in peaks_microstrain:
        target = peak * 1.0e-6
        segment = np.linspace(current, target, samples_per_segment, endpoint=False)
        history.extend(segment[1:])
        history.append(target)
        current = target
    return np.asarray(history)


PEAKS_MICROSTRAIN = [
    +200.0,
    -200.0,
    +500.0,
    -500.0,
    +1500.0,
    -1500.0,
    +3000.0,
    -3000.0,
    +6000.0,
    -6000.0,
    0.0,
]


# ---------------------------------------------------------------------------
# Scalar summaries
# ---------------------------------------------------------------------------
def envelope_summary(cfg: ProxyConfig) -> dict[str, float]:
    eps_sweep = np.linspace(-6.0e-3, +6.0e-3, 24001)
    sigma_sweep = cfg.stress(eps_sweep)

    peak_compression = float(np.min(sigma_sweep))
    peak_tension = float(np.max(sigma_sweep))

    tangent_at_origin_pos = cfg.tangent(np.array([+1.0e-8]))[0]
    tangent_at_origin_neg = cfg.tangent(np.array([-1.0e-8]))[0]
    tangent_at_2pct_t = cfg.tangent(np.array([+2.0e-3]))[0]
    tangent_at_2pct_c = cfg.tangent(np.array([-2.0e-3]))[0]

    # Energy density absorbed under monotonic compressive sweep up to -3e-3.
    eps_mono = np.linspace(0.0, -3.0e-3, 6001)
    sigma_mono = cfg.stress(eps_mono)
    energy_compression_kJpm3 = float(
        abs(np.trapezoid(sigma_mono, eps_mono)) * 1.0e3
    )  # |MPa . strain| * 1e3 = kJ/m^3

    eps_mono_t = np.linspace(0.0, +3.0e-3, 6001)
    sigma_mono_t = cfg.stress(eps_mono_t)
    energy_tension_kJpm3 = float(
        abs(np.trapezoid(sigma_mono_t, eps_mono_t)) * 1.0e3
    )

    return {
        "case": cfg.short,
        "label": cfg.label,
        "E_c_MPa": cfg.E_c,
        "E_t_MPa": cfg.E_t,
        "f_c_MPa": cfg.f_c,
        "f_t_MPa": cfg.f_t,
        "h_c": cfg.h_c,
        "h_t": cfg.h_t,
        "peak_compressive_stress_MPa": peak_compression,
        "peak_tensile_stress_MPa": peak_tension,
        "tangent_origin_tension_MPa": tangent_at_origin_pos,
        "tangent_origin_compression_MPa": tangent_at_origin_neg,
        "tangent_at_+2pct_strain_MPa": tangent_at_2pct_t,
        "tangent_at_-2pct_strain_MPa": tangent_at_2pct_c,
        "energy_compression_to_-0.3pct_kJ_per_m3": energy_compression_kJpm3,
        "energy_tension_to_+0.3pct_kJ_per_m3": energy_tension_kJpm3,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def main():
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "doc" / "figures" / "validation_reboot" / "proxy_bilinear"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Per-case summary
    summary_rows = [envelope_summary(cfg) for cfg in CONFIGS]
    summary_path = out_dir / "proxy_envelope_cases.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # 2) Monotonic dense sweep
    eps_sweep = np.linspace(-6.0e-3, +6.0e-3, 4001)
    sweep_path = out_dir / "proxy_envelope_sweep.csv"
    with sweep_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["strain"]
            + [f"sigma_{cfg.short}_MPa" for cfg in CONFIGS]
            + [f"tangent_{cfg.short}_MPa" for cfg in CONFIGS]
        )
        sigmas = [cfg.stress(eps_sweep) for cfg in CONFIGS]
        tangents = [cfg.tangent(eps_sweep) for cfg in CONFIGS]
        for k, e in enumerate(eps_sweep):
            row = [f"{e:.8e}"]
            row += [f"{s[k]:.8e}" for s in sigmas]
            row += [f"{t[k]:.8e}" for t in tangents]
            writer.writerow(row)

    # 3) Cyclic history
    eps_hist = cyclic_strain_history(PEAKS_MICROSTRAIN)
    cyclic_path = out_dir / "proxy_cyclic_history.csv"
    with cyclic_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["step", "strain"]
            + [f"sigma_{cfg.short}_MPa" for cfg in CONFIGS]
        )
        sigmas_hist = [cfg.stress(eps_hist) for cfg in CONFIGS]
        for k, e in enumerate(eps_hist):
            row = [k, f"{e:.8e}"]
            row += [f"{s[k]:.8e}" for s in sigmas_hist]
            writer.writerow(row)

    # 4) Overlay figure (envelope only — proxies are path-independent)
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    colors = ["#999999", "#1f77b4", "#d62728"]
    styles = ["--", "-.", "-"]
    widths = [1.4, 1.4, 1.9]
    eps_plot = np.linspace(-4.0e-3, +4.0e-3, 2001)
    for cfg, color, style, lw in zip(CONFIGS, colors, styles, widths):
        ax.plot(
            eps_plot * 1.0e3,
            cfg.stress(eps_plot),
            color=color,
            linestyle=style,
            linewidth=lw,
            label=cfg.label,
        )
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("deformación axial $\\varepsilon$ [‰]")
    ax.set_ylabel(r"esfuerzo axial $\sigma$ [MPa]")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-35.0, 15.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"proxy_bilinear_anisotropic_overlay.{ext}", dpi=200)
    plt.close(fig)

    # 5) Cyclic figure
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    sigmas_hist = [cfg.stress(eps_hist) for cfg in CONFIGS]
    for cfg, sig, color, style, lw in zip(
        CONFIGS, sigmas_hist, colors, styles, widths
    ):
        ax.plot(eps_hist * 1.0e3, sig, color=color, linestyle=style,
                linewidth=lw, label=cfg.label)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("deformación axial $\\varepsilon$ [‰]")
    ax.set_ylabel(r"esfuerzo axial $\sigma$ [MPa]")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.set_xlim(-6.5, 6.5)
    ax.set_ylim(-35.0, 20.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"proxy_bilinear_anisotropic_cyclic.{ext}", dpi=200)
    plt.close(fig)

    print("Wrote outputs under", out_dir)
    for row in summary_rows:
        print(
            f"  {row['case']:>9s}: f_c={row['f_c_MPa']:6.2f} MPa, "
            f"f_t={row['f_t_MPa']:5.2f} MPa, "
            f"E_t/E_c={row['E_t_MPa']/row['E_c_MPa']:.2f}, "
            f"peak_sigma_c={row['peak_compressive_stress_MPa']:7.2f} MPa, "
            f"peak_sigma_t={row['peak_tensile_stress_MPa']:6.2f} MPa, "
            f"W_c={row['energy_compression_to_-0.3pct_kJ_per_m3']:7.2f} kJ/m3"
        )


if __name__ == "__main__":
    main()
