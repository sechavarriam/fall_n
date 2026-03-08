#ifndef FALL_N_TENSOR4_HH
#define FALL_N_TENSOR4_HH

// =============================================================================
//  Tensor4<dim> — Fourth-order tensor in ℝ^dim (Voigt matrix storage)
// =============================================================================
//
//  A fourth-order tensor with minor symmetries (ℂ_ijkl = ℂ_jikl = ℂ_ijlk)
//  stored as a Voigt matrix of size  N × N,  where N = voigt_size<dim>().
//
//  This compact representation is the standard in computational mechanics:
//
//    dim=1  →  1×1  matrix
//    dim=2  →  3×3  matrix
//    dim=3  →  6×6  matrix
//
//  Physical meaning:
//    • Material tangent  ℂ = ∂S/∂E    (Lagrangian, material description)
//    • Spatial tangent   𝕔 = ∂σ/∂ε    (Eulerian, spatial description)
//    • Elasticity tensor for hyperelastic:  ℂ = ∂²W/∂E²
//
//  Operations:
//    • Contract with SymmetricTensor2:   σ = ℂ : ε    (matrix-vector product)
//    • Build from standard symmetry groups (isotropic, etc.)
//    • Algebraic operations (+, −, scalar *, composition)
//
//  Major symmetry (ℂ_ijkl = ℂ_klij ↔ ℂ_Voigt is symmetric) is NOT enforced
//  at the type level — it is a property of specific models (hyperelastic).
//
//  constexpr-friendly, zero-cost over Eigen.
//
// =============================================================================

#include <cstddef>
#include <cmath>

#include <Eigen/Dense>

#include "SymmetricTensor2.hh"

namespace continuum {

// =============================================================================
//  Tensor4<dim>
// =============================================================================

template <std::size_t dim>
    requires ValidDim<dim>
class Tensor4 {
public:
    // ── Types and constants ──────────────────────────────────────────────────

    static constexpr std::size_t dimension = dim;
    static constexpr std::size_t N         = voigt_size<dim>();

    using MatrixT       = Eigen::Matrix<double, N, N>;
    using VoigtVectorT  = Eigen::Matrix<double, N, 1>;

    // ── Construction ─────────────────────────────────────────────────────────

    /// Default: zero tensor.
    constexpr Tensor4() noexcept : data_{MatrixT::Zero()} {}

    /// From a Voigt matrix.
    template <typename Derived>
    constexpr explicit Tensor4(const Eigen::MatrixBase<Derived>& m) noexcept
        : data_{m} {}

    /// Static factory: zero tensor.
    static constexpr Tensor4 zero() noexcept { return Tensor4{}; }

    // ── Isotropic elasticity tensor builders ──────────────────────────────────
    //
    // The isotropic linear elasticity tensor is:
    //   ℂ_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)
    //
    // In Voigt notation (3D):
    //   ℂ = | λ+2μ   λ     λ     0  0  0 |
    //       |  λ    λ+2μ   λ     0  0  0 |
    //       |  λ     λ    λ+2μ   0  0  0 |
    //       |  0     0     0     μ  0  0 |
    //       |  0     0     0     0  μ  0 |
    //       |  0     0     0     0  0  μ |
    //
    // Note: This is the STRESS-form tangent (no factor 2 on shear).
    //       When used as C in  σ_Voigt = C · ε_engineering,
    //       the shear block should use μ (not 2μ) because the factor of 2
    //       is already in the engineering shear strain γ = 2ε₁₂.

    /// Build from Lamé parameters (λ, μ).
    static constexpr Tensor4 isotropic_lame(double lambda, double mu) noexcept {
        Tensor4 C;
        // Diagonal (normal components)
        for (std::size_t i = 0; i < dim; ++i)
            C.data_(i, i) = lambda + 2.0 * mu;

        // Off-diagonal (normal-normal coupling)
        for (std::size_t i = 0; i < dim; ++i)
            for (std::size_t j = i + 1; j < dim; ++j) {
                C.data_(i, j) = lambda;
                C.data_(j, i) = lambda;
            }

        // Shear block
        for (std::size_t k = dim; k < N; ++k)
            C.data_(k, k) = mu;

        return C;
    }

    /// Build from Young's modulus E and Poisson's ratio ν.
    static constexpr Tensor4 isotropic_E_nu(double E, double nu) noexcept {
        double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        double mu     = E / (2.0 * (1.0 + nu));
        return isotropic_lame(lambda, mu);
    }

    // ── Fourth-order identity tensors ────────────────────────────────────────
    //
    // Several "identity" tensors arise naturally in continuum mechanics.
    // All are expressed in Voigt matrix form.
    //
    // 𝕀      = δ_ik δ_jl               "major" identity  (maps T to itself)
    // 𝕀ˢʸᵐ   = ½(δ_ik δ_jl + δ_il δ_jk)  symmetric identity (projects to sym)
    // 𝕀ᵛᵒˡ   = ⅓ δ_ij δ_kl              volumetric projector
    // 𝕀ᵈᵉᵛ   = 𝕀ˢʸᵐ − 𝕀ᵛᵒˡ             deviatoric projector

    /// Symmetric fourth-order identity:  𝕀ˢʸᵐ : T = sym(T)
    /// In Voigt:  I_sym = diag(1,1,1, ½,½,½) for 3D
    /// But since we use the stress-form convention (no factor 2 in shear),
    /// this maps ε_tensor to σ_tensor form, so diag(1,...,1, 0.5,...,0.5).
    static constexpr Tensor4 symmetric_identity() noexcept {
        Tensor4 I;
        for (std::size_t i = 0; i < dim; ++i)
            I.data_(i, i) = 1.0;
        for (std::size_t k = dim; k < N; ++k)
            I.data_(k, k) = 0.5;
        return I;
    }

    /// Volumetric projector:  𝕀ᵛᵒˡ : T = ⅓ tr(T) I
    /// In Voigt:  all entries in the normal block = 1/3, rest zero.
    static constexpr Tensor4 volumetric_projector() noexcept {
        Tensor4 P;
        const double val = 1.0 / static_cast<double>(dim);
        for (std::size_t i = 0; i < dim; ++i)
            for (std::size_t j = 0; j < dim; ++j)
                P.data_(i, j) = val;
        return P;
    }

    /// Deviatoric projector:  𝕀ᵈᵉᵛ : T = dev(T)
    /// In Voigt:  𝕀ᵈᵉᵛ = 𝕀ˢʸᵐ − 𝕀ᵛᵒˡ
    static constexpr Tensor4 deviatoric_projector() noexcept {
        auto I_sym = symmetric_identity();
        auto I_vol = volumetric_projector();
        Tensor4 P;
        P.data_ = I_sym.data_ - I_vol.data_;
        return P;
    }

    // ── Element access ───────────────────────────────────────────────────────

    constexpr double  operator()(std::size_t I, std::size_t J) const noexcept { return data_(I, J); }
    constexpr double& operator()(std::size_t I, std::size_t J)       noexcept { return data_(I, J); }

    /// Raw Voigt matrix.
    constexpr const MatrixT& voigt_matrix() const noexcept { return data_; }
    constexpr MatrixT&       voigt_matrix()       noexcept { return data_; }

    // ── Contraction with SymmetricTensor2 ────────────────────────────────────
    //
    // σ = ℂ : ε   in Voigt:   σ_Voigt = C_Voigt · ε_Voigt
    //
    // IMPORTANT: This assumes BOTH σ and ε are in tenor Voigt form
    // (no factor of 2 on shear). If the input is in engineering Voigt
    // (with factor 2 on shear), the caller must convert first.

    constexpr SymmetricTensor2<dim> contract(const SymmetricTensor2<dim>& e) const noexcept {
        return SymmetricTensor2<dim>{(data_ * e.voigt()).eval()};
    }

    // ── Algebraic operations ─────────────────────────────────────────────────

    constexpr Tensor4 operator+(const Tensor4& other) const noexcept {
        return Tensor4{(data_ + other.data_).eval()};
    }

    constexpr Tensor4 operator-(const Tensor4& other) const noexcept {
        return Tensor4{(data_ - other.data_).eval()};
    }

    constexpr Tensor4 operator*(double scalar) const noexcept {
        return Tensor4{(data_ * scalar).eval()};
    }

    constexpr Tensor4 operator/(double scalar) const noexcept {
        return Tensor4{(data_ / scalar).eval()};
    }

    constexpr Tensor4& operator+=(const Tensor4& other) noexcept {
        data_ += other.data_;
        return *this;
    }

    constexpr Tensor4& operator-=(const Tensor4& other) noexcept {
        data_ -= other.data_;
        return *this;
    }

    constexpr Tensor4& operator*=(double scalar) noexcept {
        data_ *= scalar;
        return *this;
    }

    friend constexpr Tensor4 operator*(double s, const Tensor4& t) noexcept {
        return t * s;
    }

    // ── Properties ───────────────────────────────────────────────────────────

    /// Check major symmetry: ℂ_IJ == ℂ_JI  (hyperelasticity).
    bool has_major_symmetry(double tol = 1e-12) const noexcept {
        for (std::size_t I = 0; I < N; ++I)
            for (std::size_t J = I + 1; J < N; ++J)
                if (std::abs(data_(I, J) - data_(J, I)) > tol) return false;
        return true;
    }

    /// Trace of the Voigt matrix.
    constexpr double trace() const noexcept { return data_.trace(); }

    // ── Comparison ───────────────────────────────────────────────────────────

    bool approx_equal(const Tensor4& other, double tol = 1e-12) const noexcept {
        return (data_ - other.data_).norm() < tol;
    }

private:
    MatrixT data_;
};

// ── Free functions ───────────────────────────────────────────────────────────

/// Contract: σ = ℂ : ε
template <std::size_t dim>
constexpr SymmetricTensor2<dim> contract(const Tensor4<dim>& C,
                                          const SymmetricTensor2<dim>& e) noexcept {
    return C.contract(e);
}

} // namespace continuum

#endif // FALL_N_TENSOR4_HH
