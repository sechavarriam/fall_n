#ifndef FALL_N_STRESS_MEASURES_HH
#define FALL_N_STRESS_MEASURES_HH

// =============================================================================
//  StressMeasures.hh — Stress tensors and their transformations
// =============================================================================
//
//  Stress measures in continuum mechanics, and the Piola transformations
//  that relate them.  All transformations depend on F (deformation gradient):
//
//  ─── Reference-configuration stresses ───
//    S   2nd Piola-Kirchhoff  (symmetric, work-conjugate to E_GL)
//    P   1st Piola-Kirchhoff  (non-symmetric, two-point tensor, P = F·S)
//    Tₘ  Mandel stress        (Tₘ = C·S, work-conjugate to Hencky E_H)
//
//  ─── Current-configuration stresses ───
//    σ   Cauchy (true)        (symmetric, σ = (1/J) F·S·Fᵀ)
//    τ   Kirchhoff            (τ = J·σ = F·S·Fᵀ, symmetric)
//
//  ─── Piola transformations ───
//    push_forward:  S → σ           σ = (1/J) F·S·Fᵀ
//    pull_back:     σ → S           S = J F⁻¹·σ·F⁻ᵀ
//    P ↔ S:         P = F·S,        S = F⁻¹·P
//    τ ↔ σ:         τ = J σ,        σ = τ/J
//    Mandel:        Tₘ = C·S = Fᵀ·τ·F⁻ᵀ
//
//  All functions are in namespace continuum::stress.
//
// =============================================================================

#include <cstddef>

#include "Tensor2.hh"
#include "SymmetricTensor2.hh"
#include "TensorOperations.hh"

namespace continuum { namespace stress {

// ─────────────────────────────────────────────────────────────────────────────
//  Push-forward / pull-back (delegating to ops:: with clear intent)
// ─────────────────────────────────────────────────────────────────────────────

/// Cauchy stress from 2nd PK:  σ = (1/J) F · S · Fᵀ
template <std::size_t dim>
SymmetricTensor2<dim> cauchy_from_2pk(const SymmetricTensor2<dim>& S,
                                       const Tensor2<dim>& F) {
    return ops::push_forward(S, F);
}

/// 2nd PK from Cauchy stress:  S = J F⁻¹ · σ · F⁻ᵀ
template <std::size_t dim>
SymmetricTensor2<dim> second_pk_from_cauchy(const SymmetricTensor2<dim>& sigma,
                                             const Tensor2<dim>& F) {
    return ops::pull_back(sigma, F);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Kirchhoff stress
// ─────────────────────────────────────────────────────────────────────────────

/// Kirchhoff stress from Cauchy:  τ = J σ
template <std::size_t dim>
SymmetricTensor2<dim> kirchhoff_from_cauchy(const SymmetricTensor2<dim>& sigma,
                                             const Tensor2<dim>& F) {
    return sigma * F.determinant();
}

/// Cauchy from Kirchhoff:  σ = τ / J
template <std::size_t dim>
SymmetricTensor2<dim> cauchy_from_kirchhoff(const SymmetricTensor2<dim>& tau,
                                             const Tensor2<dim>& F) {
    return tau / F.determinant();
}

/// Kirchhoff from 2nd PK:  τ = F · S · Fᵀ  (= J σ)
template <std::size_t dim>
SymmetricTensor2<dim> kirchhoff_from_2pk(const SymmetricTensor2<dim>& S,
                                          const Tensor2<dim>& F) {
    auto result = F.matrix() * S.matrix() * F.matrix().transpose();
    return SymmetricTensor2<dim>{Tensor2<dim>{result.eval()}};
}

/// 2nd PK from Kirchhoff:  S = F⁻¹ · τ · F⁻ᵀ
template <std::size_t dim>
SymmetricTensor2<dim> second_pk_from_kirchhoff(const SymmetricTensor2<dim>& tau,
                                                const Tensor2<dim>& F) {
    auto F_inv = F.matrix().inverse();
    auto result = F_inv * tau.matrix() * F_inv.transpose();
    return SymmetricTensor2<dim>{Tensor2<dim>{result.eval()}};
}

// ─────────────────────────────────────────────────────────────────────────────
//  1st Piola-Kirchhoff (non-symmetric, two-point tensor)
// ─────────────────────────────────────────────────────────────────────────────

/// 1st PK from 2nd PK:  P = F · S
template <std::size_t dim>
Tensor2<dim> first_pk_from_2pk(const SymmetricTensor2<dim>& S,
                                const Tensor2<dim>& F) {
    return ops::first_piola_from_second(S, F);
}

/// 2nd PK from 1st PK:  S = F⁻¹ · P
template <std::size_t dim>
SymmetricTensor2<dim> second_pk_from_first(const Tensor2<dim>& P,
                                            const Tensor2<dim>& F) {
    return ops::second_piola_from_first(P, F);
}

/// 1st PK from Cauchy:  P = J σ · F⁻ᵀ
template <std::size_t dim>
Tensor2<dim> first_pk_from_cauchy(const SymmetricTensor2<dim>& sigma,
                                   const Tensor2<dim>& F) {
    double J = F.determinant();
    auto result = J * sigma.matrix() * F.matrix().inverse().transpose();
    return Tensor2<dim>{result.eval()};
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mandel stress
// ─────────────────────────────────────────────────────────────────────────────

/// Mandel stress:  Tₘ = C · S
///
/// Work-conjugate to the logarithmic (Hencky) strain rate Ḋ = dE_H/dt.
/// The Mandel stress is symmetric for isotropic materials and naturally
/// connects logarithmic stress-strain measures.
///
/// Returns a SymmetricTensor2 (true only for isotropic materials).
/// For anisotropic materials the Mandel stress is generally non-symmetric,
/// and one would need to return Tensor2 instead. The caller must be aware.
template <std::size_t dim>
SymmetricTensor2<dim> mandel(const SymmetricTensor2<dim>& S,
                              const SymmetricTensor2<dim>& C) {
    // Tₘ = C·S  (matrix product of two symmetric tensors — generally non-sym)
    // For isotropic: they commute → Tₘ is symmetric
    auto result = C.matrix() * S.matrix();
    return SymmetricTensor2<dim>{Tensor2<dim>{result.eval()}};
}

/// Von Mises equivalent stress:  σ_eq = √(3/2) ‖dev(σ)‖
///
/// This is independent of the stress type — works for σ, S, or τ.
template <std::size_t dim>
double von_mises(const SymmetricTensor2<dim>& sigma) {
    auto sdev = sigma.deviatoric();
    return std::sqrt(1.5) * sdev.norm();
}

/// Mean (hydrostatic) stress:  p = (1/3) tr(σ)
template <std::size_t dim>
double hydrostatic(const SymmetricTensor2<dim>& sigma) {
    return sigma.trace() / static_cast<double>(dim);
}

/// Stress triaxiality:  η = p / σ_eq
template <std::size_t dim>
double triaxiality(const SymmetricTensor2<dim>& sigma) {
    double seq = von_mises(sigma);
    if (seq < 1e-30) return 0.0;
    return hydrostatic(sigma) / seq;
}

}} // namespace continuum::stress

#endif // FALL_N_STRESS_MEASURES_HH
