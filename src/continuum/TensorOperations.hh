#ifndef FALL_N_TENSOR_OPERATIONS_HH
#define FALL_N_TENSOR_OPERATIONS_HH

// =============================================================================
//  TensorOperations.hh — Continuum mechanics tensor operations
// =============================================================================
//
//  Functions that operate on Tensor2 and SymmetricTensor2 to perform
//  the standard tensor algebra needed in computational mechanics:
//
//    • Polar decomposition:  F = R · U = V · R
//    • Kinematic products:   C = FᵀF,  b = FFᵀ
//    • Push-forward / pull-back of symmetric tensors
//    • Piola transformations  (S ↔ σ ↔ P ↔ τ)
//    • Tangent transformations (ℂ ↔ 𝕔)
//
//  All functions live in namespace continuum::ops.
//
// =============================================================================

#include <cstddef>
#include <cmath>
#include <utility>

#include <Eigen/Dense>

#include "Tensor2.hh"
#include "SymmetricTensor2.hh"
#include "Tensor4.hh"

namespace continuum { namespace ops {

// ─────────────────────────────────────────────────────────────────────────────
//  Kinematic products
// ─────────────────────────────────────────────────────────────────────────────

/// Right Cauchy-Green tensor:  C = Fᵀ · F   (Lagrangian, symmetric, SPD)
template <std::size_t dim>
SymmetricTensor2<dim> right_cauchy_green(const Tensor2<dim>& F) {
    auto FtF = (F.matrix().transpose() * F.matrix()).eval();
    return SymmetricTensor2<dim>{Tensor2<dim>{FtF}};
}

/// Left Cauchy-Green tensor:  b = F · Fᵀ   (Eulerian, symmetric, SPD)
template <std::size_t dim>
SymmetricTensor2<dim> left_cauchy_green(const Tensor2<dim>& F) {
    auto FFt = (F.matrix() * F.matrix().transpose()).eval();
    return SymmetricTensor2<dim>{Tensor2<dim>{FFt}};
}

/// Jacobian determinant:  J = det(F)
template <std::size_t dim>
double jacobian(const Tensor2<dim>& F) {
    return F.determinant();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Polar decomposition:  F = R · U = V · R
// ─────────────────────────────────────────────────────────────────────────────
//
//  R  — proper orthogonal rotation tensor (det(R) = +1)
//  U  — right stretch tensor (symmetric, positive definite) in reference config
//  V  — left stretch tensor (symmetric, positive definite) in current config
//
//  Algorithm:
//    C = FᵀF  →  spectral decomposition  C = Σ λᵢ² Nᵢ⊗Nᵢ
//    U = C^{1/2} = Σ λᵢ Nᵢ⊗Nᵢ
//    R = F · U⁻¹
//    V = R · U · Rᵀ  =  F · Rᵀ

struct PolarDecompositionResult {
    // These are Tensor2 rather than SymmetricTensor2 because
    // R is not symmetric, and keeping uniform type simplifies the API.
    // U and V can be extracted as SymmetricTensor2 via accessors.
    Eigen::Matrix3d R;   // rotation
    Eigen::Matrix3d U;   // right stretch
    Eigen::Matrix3d V;   // left stretch
};

/// Polar decomposition of a 3×3 tensor F.
/// F = R · U  with  R ∈ SO(3),  U = sqrt(FᵀF).
/// Requires det(F) > 0.
template <std::size_t dim>
    requires (dim == 3)
auto polar_decomposition(const Tensor2<dim>& F) {
    // C = FᵀF
    auto C_mat = (F.matrix().transpose() * F.matrix()).eval();
    
    // Spectral decomposition of C (symmetric positive definite)
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C_mat);
    auto evals = solver.eigenvalues();   // λᵢ² in ascending order
    auto evecs = solver.eigenvectors();  // columns = Nᵢ

    // U = C^{1/2} = Σ sqrt(λᵢ²) Nᵢ⊗Nᵢ
    Eigen::Matrix3d U_mat = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d U_inv = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i) {
        double lam = std::sqrt(evals[i]);
        Eigen::Vector3d ni = evecs.col(i);
        U_mat += lam * ni * ni.transpose();
        U_inv += (1.0 / lam) * ni * ni.transpose();
    }

    // R = F · U⁻¹
    Eigen::Matrix3d R_mat = F.matrix() * U_inv;

    // V = F · Rᵀ = R · U · Rᵀ
    Eigen::Matrix3d V_mat = F.matrix() * R_mat.transpose();

    struct Result {
        Tensor2<dim>          R;
        SymmetricTensor2<dim> U;
        SymmetricTensor2<dim> V;
    };

    return Result{
        Tensor2<dim>{R_mat},
        SymmetricTensor2<dim>{Tensor2<dim>{U_mat}},
        SymmetricTensor2<dim>{Tensor2<dim>{V_mat}}
    };
}

/// Polar decomposition for 2D (F ∈ ℝ²ˣ²).
template <std::size_t dim>
    requires (dim == 2)
auto polar_decomposition(const Tensor2<dim>& F) {
    Eigen::Matrix2d FtF = F.matrix().transpose() * F.matrix();
    
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> solver(FtF);
    auto evals = solver.eigenvalues();
    auto evecs = solver.eigenvectors();

    Eigen::Matrix2d U_mat = Eigen::Matrix2d::Zero();
    Eigen::Matrix2d U_inv = Eigen::Matrix2d::Zero();
    for (int i = 0; i < 2; ++i) {
        double lam = std::sqrt(evals[i]);
        Eigen::Vector2d ni = evecs.col(i);
        U_mat += lam * ni * ni.transpose();
        U_inv += (1.0 / lam) * ni * ni.transpose();
    }

    Eigen::Matrix2d R_mat = F.matrix() * U_inv;
    Eigen::Matrix2d V_mat = F.matrix() * R_mat.transpose();

    struct Result {
        Tensor2<dim>          R;
        SymmetricTensor2<dim> U;
        SymmetricTensor2<dim> V;
    };

    return Result{
        Tensor2<dim>{R_mat},
        SymmetricTensor2<dim>{Tensor2<dim>{U_mat}},
        SymmetricTensor2<dim>{Tensor2<dim>{V_mat}}
    };
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stress transformations (Piola transformations)
// ─────────────────────────────────────────────────────────────────────────────
//
//  Given F with J = det(F):
//
//    Cauchy → 2nd Piola-Kirchhoff:  S = J F⁻¹ · σ · F⁻ᵀ   (pull-back)
//    2nd PK → Cauchy:               σ = (1/J) F · S · Fᵀ   (push-forward)
//
//    Cauchy → Kirchhoff:            τ = J σ
//    Kirchhoff → Cauchy:            σ = τ / J
//
//    2nd PK → 1st PK:              P = F · S    (two-point tensor)
//    1st PK → 2nd PK:              S = F⁻¹ · P
//
//    Cauchy → 1st PK:              P = J σ · F⁻ᵀ

/// Push-forward: σ = (1/J) F · S · Fᵀ
template <std::size_t dim>
SymmetricTensor2<dim> push_forward(const SymmetricTensor2<dim>& S,
                                    const Tensor2<dim>& F) {
    const double J = F.determinant();
    auto result = (1.0 / J) * F.matrix() * S.matrix() * F.matrix().transpose();
    return SymmetricTensor2<dim>{Tensor2<dim>{result.eval()}};
}

/// Pull-back: S = J F⁻¹ · σ · F⁻ᵀ
template <std::size_t dim>
SymmetricTensor2<dim> pull_back(const SymmetricTensor2<dim>& sigma,
                                 const Tensor2<dim>& F) {
    const double J = F.determinant();
    auto F_inv = F.matrix().inverse();
    auto result = J * F_inv * sigma.matrix() * F_inv.transpose();
    return SymmetricTensor2<dim>{Tensor2<dim>{result.eval()}};
}

/// Kirchhoff stress: τ = J σ
template <std::size_t dim>
constexpr SymmetricTensor2<dim> kirchhoff_from_cauchy(const SymmetricTensor2<dim>& sigma,
                                                       double J) noexcept {
    return sigma * J;
}

/// Cauchy from Kirchhoff: σ = τ / J
template <std::size_t dim>
constexpr SymmetricTensor2<dim> cauchy_from_kirchhoff(const SymmetricTensor2<dim>& tau,
                                                       double J) noexcept {
    return tau / J;
}

/// 1st Piola-Kirchhoff from 2nd PK:  P = F · S  (not symmetric — returns Tensor2)
template <std::size_t dim>
Tensor2<dim> first_piola_from_second(const SymmetricTensor2<dim>& S,
                                      const Tensor2<dim>& F) {
    return Tensor2<dim>{(F.matrix() * S.matrix()).eval()};
}

/// 2nd PK from 1st PK:  S = F⁻¹ · P
template <std::size_t dim>
SymmetricTensor2<dim> second_piola_from_first(const Tensor2<dim>& P,
                                               const Tensor2<dim>& F) {
    auto result = F.matrix().inverse() * P.matrix();
    return SymmetricTensor2<dim>{Tensor2<dim>{result.eval()}};
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tangent tensor transformations
// ─────────────────────────────────────────────────────────────────────────────
//
//  Push-forward of the material tangent ℂ (4th order):
//    𝕔_ijkl = (1/J) F_iI F_jJ F_kK F_lL ℂ_IJKL
//
//  This transforms the reference-config tangent ℂ = ∂S/∂E to the spatial
//  tangent 𝕔 = ∂σ/∂ε (or more precisely, the Jaumann rate tangent).
//
//  In Voigt notation, this is a congruence transformation:
//    𝕔_Voigt = (1/J) M · ℂ_Voigt · Mᵀ
//  where M is the 6×6 push-forward basis transformation matrix.
//
//  For the full implementation, this requires building M from F.
//  This is deferred to Phase 4 (element-level assembly), since the
//  exact Voigt-form transformation depends on the specific Voigt
//  convention and engineering-strain factors. For now, we provide a
//  placeholder that works through the full 4th-order index contraction.

/// Push-forward of a 4th-order tangent: 𝕔 = (1/J) F⊗F : ℂ : Fᵀ⊗Fᵀ
/// Full index contraction (not optimized — correct reference implementation).
///
/// c_ijkl = (1/J) Σ_{IJKL} F_iI F_jJ ℂ_IJKL F_kK F_lL
///
/// Works for any dim ∈ {1, 2, 3}.
template <std::size_t dim>
Tensor4<dim> push_forward_tangent(const Tensor4<dim>& CC,
                                   const Tensor2<dim>& F) {
    const double J = F.determinant();
    const auto& Fm = F.matrix();

    constexpr auto N = Tensor4<dim>::N;
    using ST2 = SymmetricTensor2<dim>;

    Eigen::Matrix<double, N, N> c_voigt = Eigen::Matrix<double, N, N>::Zero();

    for (std::size_t A = 0; A < N; ++A) {
        auto [i, j] = ST2::tensor_indices(A);
        for (std::size_t B = 0; B < N; ++B) {
            auto [k, l] = ST2::tensor_indices(B);

            double sum = 0.0;
            for (std::size_t I = 0; I < dim; ++I)
            for (std::size_t J = 0; J < dim; ++J) {
                std::size_t IJ = ST2::voigt_index(I, J);
                for (std::size_t K = 0; K < dim; ++K)
                for (std::size_t L = 0; L < dim; ++L) {
                    std::size_t KL = ST2::voigt_index(K, L);
                    sum += Fm(i,I) * Fm(j,J) * CC(IJ, KL) * Fm(k,K) * Fm(l,L);
                }
            }
            c_voigt(A, B) = sum / J;
        }
    }

    return Tensor4<dim>{c_voigt};
}

// ─────────────────────────────────────────────────────────────────────────────
//  Utility: outer product of SymmetricTensor2 → Tensor4
// ─────────────────────────────────────────────────────────────────────────────
//
// A ⊗ B  (dyadic product of two Voigt vectors → Voigt matrix)
// (ℂ_IJKL = A_IJ B_KL)  →  in Voigt: C(I,J) = a(I) * b(J)
//
// Used in constitutive models:
//   ℂ = λ (I⊗I) + 2μ 𝕀ˢʸᵐ

template <std::size_t dim>
constexpr Tensor4<dim> outer_product(const SymmetricTensor2<dim>& A,
                                      const SymmetricTensor2<dim>& B) noexcept {
    return Tensor4<dim>{(A.voigt() * B.voigt().transpose()).eval()};
}

}} // namespace continuum::ops

#endif // FALL_N_TENSOR_OPERATIONS_HH
