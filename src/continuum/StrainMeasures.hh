#ifndef FALL_N_STRAIN_MEASURES_HH
#define FALL_N_STRAIN_MEASURES_HH

// =============================================================================
//  StrainMeasures.hh — Finite-strain deformation and strain measures
// =============================================================================
//
//  Given a deformation gradient  F = ∂x/∂X  (Tensor2<dim>), this module
//  provides all standard kinematic quantities for nonlinear solid mechanics:
//
//  ─── Deformation measures ───
//    J  = det(F)                     Jacobian determinant (volume ratio)
//    C  = FᵀF                        Right Cauchy-Green  (Lagrangian, SPD)
//    b  = FFᵀ                        Left  Cauchy-Green  (Eulerian, SPD)
//    U  = C^{1/2}                    Right stretch tensor
//    V  = b^{1/2}                    Left  stretch tensor
//    R                               Rotation tensor  (from polar decomposition)
//
//  ─── Strain measures (Lagrangian — reference config) ───
//    E_GL  = ½(C − I)               Green-Lagrange strain
//    E_H   = ½ ln(C)                Hencky (logarithmic) strain
//    E_m   from Seth-Hill family    E^(m) = (1/2m)(U^{2m} − I)
//
//  ─── Strain measures (Eulerian — current config) ───
//    e_A   = ½(I − b⁻¹)            Almansi-Euler strain
//    e_h   = ½ ln(b) = R · E_H · Rᵀ  Eulerian Hencky strain
//    e_m   from Seth-Hill family    e^(m) = (1/2m)(V^{2m} − I)
//
//  ─── Seth-Hill family ───
//    Unified one-parameter family parametrised by m ∈ ℝ:
//      m = +1    →  Green-Lagrange   E_GL
//      m = −1    →  Almansi strain   (via Eulerian form)
//      m → 0     →  Hencky strain    E_H  (logarithmic, as limit)
//      m = +½    →  Biot strain      (engineering stretch − 1)
//
//  All functions are in namespace continuum::strain.
//
// =============================================================================

#include <cstddef>
#include <cmath>
#include <limits>

#include "Tensor2.hh"
#include "SymmetricTensor2.hh"
#include "TensorOperations.hh"

namespace continuum { namespace strain {

// ─────────────────────────────────────────────────────────────────────────────
//  Deformation measures
// ─────────────────────────────────────────────────────────────────────────────

/// Jacobian J = det(F).  Must be > 0 for physical deformations.
template <std::size_t dim>
double jacobian(const Tensor2<dim>& F) {
    return F.determinant();
}

/// Right Cauchy-Green:  C = FᵀF  (symmetric, positive definite).
template <std::size_t dim>
SymmetricTensor2<dim> right_cauchy_green(const Tensor2<dim>& F) {
    return ops::right_cauchy_green(F);
}

/// Left Cauchy-Green:  b = FFᵀ  (symmetric, positive definite).
template <std::size_t dim>
SymmetricTensor2<dim> left_cauchy_green(const Tensor2<dim>& F) {
    return ops::left_cauchy_green(F);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Lagrangian strain measures (reference configuration)
// ─────────────────────────────────────────────────────────────────────────────

/// Green-Lagrange strain:  E = ½(C − I) = ½(FᵀF − I)
///
/// Work-conjugate to the 2nd Piola-Kirchhoff stress S.
/// This is the most common Lagrangian strain measure in Total Lagrangian
/// formulations.  It is quadratic in the displacement gradient:
///   E = ½(∇u + ∇uᵀ + ∇uᵀ·∇u)
/// The linearised part (small strains) is the symmetric gradient of u.
template <std::size_t dim>
SymmetricTensor2<dim> green_lagrange(const Tensor2<dim>& F) {
    auto C = right_cauchy_green(F);
    auto I = SymmetricTensor2<dim>::identity();
    return (C - I) * 0.5;
}

/// Hencky (logarithmic) strain:  E_H = ½ ln(C)
///
/// Work-conjugate to the Mandel stress M = C · S (for symmetric stress
/// measures). The logarithmic strain is preferred for:
///   • Large strains where additive decomposition is needed
///   • Metals (the "true strain" concept)
///   • Hencky-based hyperelasticity (quadratic in E_H → St. Venant-like
///     but with correct large-strain behaviour)
///
/// Requires C positive definite (always true for physical deformation).
template <std::size_t dim>
SymmetricTensor2<dim> hencky(const Tensor2<dim>& F) {
    auto C = right_cauchy_green(F);
    return C.log() * 0.5;
}

/// Biot strain:  E_Biot = U − I
///
/// Work-conjugate to the Biot stress (= R^T · P).
/// Often used in engineering as the "stretch" strain.
/// Requires the right stretch tensor U = C^{1/2}.
template <std::size_t dim>
    requires (dim >= 2)
SymmetricTensor2<dim> biot(const Tensor2<dim>& F) {
    auto C = right_cauchy_green(F);
    auto U = C.sqrt();
    auto I = SymmetricTensor2<dim>::identity();
    return U - I;
}

/// Seth-Hill (generalised) strain:  E^(m) = (1/2m)(U^{2m} − I)
///
/// Unified parametric family that recovers standard measures:
///   m =  1    → Green-Lagrange        ½(FᵀF − I)
///   m =  ½    → Biot strain           U − I
///   m → 0     → Hencky strain         ½ ln(C)   (logarithmic limit)
///   m = −1    → Almansi-like          ½(I − C⁻¹)
///   m = −½    → inverse-Biot          I − U⁻¹
///
/// For m → 0 the formula limits to ½ ln(C). We handle this separately
/// for numerical stability when |m| < ε.
template <std::size_t dim>
SymmetricTensor2<dim> seth_hill(const Tensor2<dim>& F, double m) {
    constexpr double eps = 1e-10;
    
    if (std::abs(m) < eps) {
        // Limit: m → 0  ⇒  E^(0) = ½ ln(C)
        return hencky(F);
    }

    auto C = right_cauchy_green(F);
    auto I = SymmetricTensor2<dim>::identity();
    
    // U^{2m} = C^m  (since U² = C, so U^{2m} = (U²)^m = C^m)
    auto C_m = C.pow(m);
    
    return (C_m - I) * (1.0 / (2.0 * m));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Eulerian strain measures (current configuration)
// ─────────────────────────────────────────────────────────────────────────────

/// Almansi-Euler strain:  e = ½(I − b⁻¹) = ½(I − F⁻ᵀF⁻¹)
///
/// Eulerian counterpart of Green-Lagrange. Work-conjugate to the
/// Cauchy stress σ in the Eulerian description.
template <std::size_t dim>
SymmetricTensor2<dim> almansi(const Tensor2<dim>& F) {
    auto b = left_cauchy_green(F);
    auto I = SymmetricTensor2<dim>::identity();
    return (I - b.inverse()) * 0.5;
}

/// Eulerian Hencky strain:  e_h = ½ ln(b)
///
/// Related to Lagrangian Hencky by:  e_h = R · E_H · Rᵀ
/// (push-forward by rotation only).
template <std::size_t dim>
SymmetricTensor2<dim> eulerian_hencky(const Tensor2<dim>& F) {
    auto b = left_cauchy_green(F);
    return b.log() * 0.5;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Small-strain limit
// ─────────────────────────────────────────────────────────────────────────────

/// Infinitesimal (engineering) strain:  ε = ½(∇u + ∇uᵀ) = sym(∇u)
///
/// This is the linearised strain, valid only for small displacements.
/// Given F = I + ∇u, this extracts the symmetric part of ∇u.
/// All finite-strain measures reduce to this for F ≈ I.
template <std::size_t dim>
SymmetricTensor2<dim> infinitesimal(const Tensor2<dim>& F) {
    auto I = Tensor2<dim>::identity();
    auto grad_u = F - I;  // ∇u = F - I
    return SymmetricTensor2<dim>{grad_u.symmetric_part()};
}

/// Rotation (spin) tensor:  ω = ½(∇u − ∇uᵀ) = skw(∇u)
///
/// The linearised rotation for small displacements.
template <std::size_t dim>
Tensor2<dim> infinitesimal_rotation(const Tensor2<dim>& F) {
    auto I = Tensor2<dim>::identity();
    auto grad_u = F - I;
    return grad_u.skew_part();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Velocity gradient measures (rate form)
// ─────────────────────────────────────────────────────────────────────────────

/// Velocity gradient:  L = Ḟ · F⁻¹
/// Given the rate of deformation gradient Ḟ and the current F.
template <std::size_t dim>
Tensor2<dim> velocity_gradient(const Tensor2<dim>& F_dot,
                                const Tensor2<dim>& F) {
    return Tensor2<dim>{(F_dot.matrix() * F.matrix().inverse()).eval()};
}

/// Rate of deformation:  d = sym(L) = ½(L + Lᵀ)
template <std::size_t dim>
SymmetricTensor2<dim> rate_of_deformation(const Tensor2<dim>& L) {
    return SymmetricTensor2<dim>{L.symmetric_part()};
}

/// Spin tensor:  w = skw(L) = ½(L − Lᵀ)
template <std::size_t dim>
Tensor2<dim> spin_tensor(const Tensor2<dim>& L) {
    return L.skew_part();
}

}} // namespace continuum::strain

#endif // FALL_N_STRAIN_MEASURES_HH
