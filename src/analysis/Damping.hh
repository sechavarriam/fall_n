#ifndef FN_DAMPING_HH
#define FN_DAMPING_HH

// =============================================================================
//  Damping.hh — Viscous damping models for structural dynamics
// =============================================================================
//
//  Provides damping matrix assembly strategies for use with DynamicAnalysis.
//
//  ─── Rayleigh damping ───────────────────────────────────────────────────
//
//  The most common model in structural dynamics:
//
//    C = α·M + β·K
//
//  where α (mass-proportional) and β (stiffness-proportional) are chosen
//  to match damping ratios at two target frequencies:
//
//    α = 2·ξ·ω₁·ω₂ / (ω₁ + ω₂)
//    β = 2·ξ / (ω₁ + ω₂)
//
//  For the case of two different damping ratios ξ₁ and ξ₂:
//
//    [α]   1         [ ω₁·ω₂   -1 ] [ξ₁]
//    [β] = ───────── [-ω₂       ω₁] [ξ₂]
//          ω₁² - ω₂²
//
//  ─── Customization points ───────────────────────────────────────────────
//
//  The DampingModel is a simple callable that takes (M, K_initial, C_out)
//  and fills C_out.  Users can provide:
//
//    - Rayleigh damping (factory function)
//    - Modal damping (user-provided)
//    - Direct assembly (user-provided)
//    - No damping (default)
//
// =============================================================================

#include <cmath>
#include <functional>
#include <petsc.h>

// Type alias: a DampingAssembler takes (M, K, C_out) and fills C_out.
using DampingAssembler = std::function<void(Mat M, Mat K, Mat C)>;

namespace damping {

// ─── Rayleigh:  C = α·M + β·K ───────────────────────────────────────────
//
//  Creates a DampingAssembler that computes C = α_M·M + β_K·K.
//
//  The matrices M and K must already be assembled.
//  C must have the same sparsity pattern as K (guaranteed if created
//  via DMCreateMatrix on the same DM).

inline DampingAssembler rayleigh(double alpha_M, double beta_K) {
    return [=](Mat M, Mat K, Mat C) {
        // C = β·K
        MatCopy(K, C, DIFFERENT_NONZERO_PATTERN);
        MatScale(C, beta_K);
        // C += α·M
        MatAXPY(C, alpha_M, M, SAME_NONZERO_PATTERN);
    };
}


// ─── Rayleigh from two frequencies and damping ratios ────────────────────
//
//  Computes α and β from two circular frequencies ω₁, ω₂ and
//  damping ratios ξ₁, ξ₂:
//
//    |ξ₁| = 1  | 1/ω₁  ω₁ | |α/2|
//    |ξ₂|   2  | 1/ω₂  ω₂ | |β/2|

inline DampingAssembler rayleigh_from_frequencies(
    double omega1, double omega2,
    double xi1,    double xi2)
{
    // Solve the 2×2 system:
    //   [1/(2ω₁)  ω₁/2] [α]   [ξ₁]
    //   [1/(2ω₂)  ω₂/2] [β] = [ξ₂]
    double alpha = 2.0 * omega1 * omega2 * (xi1 * omega2 - xi2 * omega1) /
                   (omega2 * omega2 - omega1 * omega1);
    double beta  = 2.0 * (xi2 * omega2 - xi1 * omega1) /
                   (omega2 * omega2 - omega1 * omega1);

    return rayleigh(alpha, beta);
}


// ─── Rayleigh from single damping ratio at two frequencies ───────────────
//
//  Common case: ξ₁ = ξ₂ = ξ  →  simplified formulas:
//
//    α = 2·ξ·ω₁·ω₂ / (ω₁ + ω₂)
//    β = 2·ξ / (ω₁ + ω₂)

inline DampingAssembler rayleigh_from_single_ratio(
    double omega1, double omega2, double xi)
{
    double alpha = 2.0 * xi * omega1 * omega2 / (omega1 + omega2);
    double beta  = 2.0 * xi / (omega1 + omega2);
    return rayleigh(alpha, beta);
}


// ─── Mass-proportional only:  C = α·M ───────────────────────────────────

inline DampingAssembler mass_proportional(double alpha_M) {
    return [=](Mat M, Mat /*K*/, Mat C) {
        MatCopy(M, C, DIFFERENT_NONZERO_PATTERN);
        MatScale(C, alpha_M);
    };
}


// ─── Stiffness-proportional only:  C = β·K ──────────────────────────────

inline DampingAssembler stiffness_proportional(double beta_K) {
    return [=](Mat /*M*/, Mat K, Mat C) {
        MatCopy(K, C, DIFFERENT_NONZERO_PATTERN);
        MatScale(C, beta_K);
    };
}


// ─── No damping (default) ────────────────────────────────────────────────

inline DampingAssembler none() {
    return {};  // empty function → null check in DynamicAnalysis
}

} // namespace damping


#endif // FN_DAMPING_HH
