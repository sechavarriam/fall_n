#ifndef FALL_N_TENSOR2_HH
#define FALL_N_TENSOR2_HH

// =============================================================================
//  Tensor2<dim> — General second-order tensor in ℝ^dim
// =============================================================================
//
//  A general (non-symmetric) second-order tensor stored as a dense
//  dim × dim  Eigen::Matrix.  This is the natural representation for:
//
//    • Deformation gradient           F = ∂x/∂X
//    • Rotation tensors               R  (from polar decomposition F = R·U)
//    • Velocity gradient              L = ∂v/∂x = Ḟ·F⁻¹
//    • First Piola-Kirchhoff tensor   P = J σ F⁻ᵀ  (two-point tensor)
//
//  Design choices:
//    – constexpr-friendly: all operations are constexpr where Eigen allows.
//    – Wraps Eigen::Matrix at zero cost (no virtual, no heap).
//    – Not Voigt-compressed; for symmetric tensors use SymmetricTensor2<dim>.
//    – dim ∈ {1, 2, 3} enforced by concept.
//
//  Voigt-related conversions live in SymmetricTensor2, not here.
//
//  Thread-safety: value type, no shared mutable state.
//
// =============================================================================

#include <cstddef>
#include <cmath>
#include <type_traits>

#include <Eigen/Dense>

namespace continuum {

// ─── Dimension constraint ────────────────────────────────────────────────────

template <std::size_t D>
concept ValidDim = (D >= 1 && D <= 3);

// =============================================================================
//  Tensor2<dim>
// =============================================================================

template <std::size_t dim>
    requires ValidDim<dim>
class Tensor2 {
public:
    // ── Types ────────────────────────────────────────────────────────────────

    using MatrixT = Eigen::Matrix<double, dim, dim>;
    using VectorT = Eigen::Matrix<double, dim, 1>;

    static constexpr std::size_t dimension = dim;
    static constexpr std::size_t num_full_components = dim * dim;

    // ── Construction ─────────────────────────────────────────────────────────

    /// Default: zero tensor.
    constexpr Tensor2() noexcept : data_{MatrixT::Zero()} {}

    /// From an Eigen matrix expression.
    template <typename Derived>
    constexpr explicit Tensor2(const Eigen::MatrixBase<Derived>& m) noexcept
        : data_{m} {}

    /// Static factory: identity tensor I.
    static constexpr Tensor2 identity() noexcept {
        Tensor2 t;
        t.data_ = MatrixT::Identity();
        return t;
    }

    /// Static factory: zero tensor.
    static constexpr Tensor2 zero() noexcept { return Tensor2{}; }

    /// Dyadic product a ⊗ b   →   T_ij = a_i b_j
    static constexpr Tensor2 dyadic(const VectorT& a, const VectorT& b) noexcept {
        return Tensor2{a * b.transpose()};
    }

    // ── Element access ───────────────────────────────────────────────────────

    constexpr double  operator()(std::size_t i, std::size_t j) const noexcept { return data_(i, j); }
    constexpr double& operator()(std::size_t i, std::size_t j)       noexcept { return data_(i, j); }

    /// Raw Eigen matrix (const).
    constexpr const MatrixT& matrix() const noexcept { return data_; }

    /// Raw Eigen matrix (mutable).
    constexpr MatrixT& matrix() noexcept { return data_; }

    // ── Fundamental tensor operations ────────────────────────────────────────

    /// Trace: T_ii
    constexpr double trace() const noexcept { return data_.trace(); }

    /// Determinant: det(T)
    constexpr double determinant() const noexcept { return data_.determinant(); }

    /// Transpose: T^T
    constexpr Tensor2 transpose() const noexcept {
        return Tensor2{data_.transpose()};
    }

    /// Inverse: T^{-1}  (assumes non-singular)
    Tensor2 inverse() const noexcept {
        return Tensor2{data_.inverse()};
    }

    /// Inverse transpose: T^{-T} = (T^{-1})^T = (T^T)^{-1}
    Tensor2 inverse_transpose() const noexcept {
        return Tensor2{data_.inverse().transpose()};
    }

    /// Frobenius norm: ‖T‖_F = sqrt(T : T) = sqrt(Σ T_ij²)
    double norm() const noexcept { return data_.norm(); }

    /// Double contraction A : B = A_ij B_ij
    constexpr double double_contract(const Tensor2& other) const noexcept {
        // A:B = tr(A^T B) for general tensors
        double result = 0.0;
        for (std::size_t i = 0; i < dim; ++i)
            for (std::size_t j = 0; j < dim; ++j)
                result += data_(i, j) * other.data_(i, j);
        return result;
    }

    /// Symmetric part: sym(T) = ½(T + T^T)
    constexpr Tensor2 symmetric_part() const noexcept {
        return Tensor2{0.5 * (data_ + data_.transpose())};
    }

    /// Skew-symmetric part: skw(T) = ½(T - T^T)
    constexpr Tensor2 skew_part() const noexcept {
        return Tensor2{0.5 * (data_ - data_.transpose())};
    }

    /// Deviatoric part: dev(T) = T - (1/dim) tr(T) I
    constexpr Tensor2 deviatoric() const noexcept {
        return Tensor2{data_ - (trace() / static_cast<double>(dim)) * MatrixT::Identity()};
    }

    // ── Tensor product (composition) ─────────────────────────────────────────

    /// Matrix multiplication: (A · B)_ij = A_ik B_kj
    constexpr Tensor2 dot(const Tensor2& other) const noexcept {
        return Tensor2{data_ * other.data_};
    }

    /// Action on a vector: T · v
    constexpr VectorT dot(const VectorT& v) const noexcept {
        return data_ * v;
    }

    // ── Arithmetic operators ─────────────────────────────────────────────────

    constexpr Tensor2 operator+(const Tensor2& other) const noexcept {
        return Tensor2{(data_ + other.data_).eval()};
    }

    constexpr Tensor2 operator-(const Tensor2& other) const noexcept {
        return Tensor2{(data_ - other.data_).eval()};
    }

    constexpr Tensor2 operator*(double scalar) const noexcept {
        return Tensor2{(data_ * scalar).eval()};
    }

    constexpr Tensor2 operator/(double scalar) const noexcept {
        return Tensor2{(data_ / scalar).eval()};
    }

    constexpr Tensor2& operator+=(const Tensor2& other) noexcept {
        data_ += other.data_;
        return *this;
    }

    constexpr Tensor2& operator-=(const Tensor2& other) noexcept {
        data_ -= other.data_;
        return *this;
    }

    constexpr Tensor2& operator*=(double scalar) noexcept {
        data_ *= scalar;
        return *this;
    }

    constexpr Tensor2 operator-() const noexcept {
        return Tensor2{(-data_).eval()};
    }

    /// scalar * Tensor2
    friend constexpr Tensor2 operator*(double s, const Tensor2& t) noexcept {
        return t * s;
    }

    // ── Comparison ───────────────────────────────────────────────────────────

    bool approx_equal(const Tensor2& other, double tol = 1e-12) const noexcept {
        return (data_ - other.data_).norm() < tol;
    }

    // ── Invariants ───────────────────────────────────────────────────────────
    //
    // Principal invariants of a second-order tensor T:
    //   I₁ = tr(T)
    //   I₂ = ½[tr(T)² − tr(T²)]
    //   I₃ = det(T)
    //
    // These are the coefficients of the characteristic polynomial:
    //   λ³ − I₁ λ² + I₂ λ − I₃ = 0
    //
    // They are rotation-invariant (objective) and form the basis for
    // isotropic constitutive models.

    constexpr double I1() const noexcept { return trace(); }

    constexpr double I2() const noexcept {
        const double tr  = trace();
        const double trT2 = (data_ * data_).trace();
        return 0.5 * (tr * tr - trT2);
    }

    constexpr double I3() const noexcept { return determinant(); }

private:
    MatrixT data_;
};

// ── Free function aliases ────────────────────────────────────────────────────

template <std::size_t dim>
constexpr double trace(const Tensor2<dim>& T) noexcept { return T.trace(); }

template <std::size_t dim>
constexpr double det(const Tensor2<dim>& T) noexcept { return T.determinant(); }

template <std::size_t dim>
Tensor2<dim> inv(const Tensor2<dim>& T) noexcept { return T.inverse(); }

template <std::size_t dim>
constexpr Tensor2<dim> transpose(const Tensor2<dim>& T) noexcept { return T.transpose(); }

template <std::size_t dim>
constexpr Tensor2<dim> sym(const Tensor2<dim>& T) noexcept { return T.symmetric_part(); }

template <std::size_t dim>
constexpr Tensor2<dim> skw(const Tensor2<dim>& T) noexcept { return T.skew_part(); }

template <std::size_t dim>
constexpr Tensor2<dim> dev(const Tensor2<dim>& T) noexcept { return T.deviatoric(); }

/// Double contraction A : B = A_ij B_ij
template <std::size_t dim>
constexpr double double_contract(const Tensor2<dim>& A, const Tensor2<dim>& B) noexcept {
    return A.double_contract(B);
}

/// Dyadic product a ⊗ b
template <std::size_t dim>
constexpr Tensor2<dim> dyadic(const typename Tensor2<dim>::VectorT& a,
                               const typename Tensor2<dim>::VectorT& b) noexcept {
    return Tensor2<dim>::dyadic(a, b);
}

} // namespace continuum

#endif // FALL_N_TENSOR2_HH
