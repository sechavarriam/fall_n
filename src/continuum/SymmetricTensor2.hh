#ifndef FALL_N_SYMMETRIC_TENSOR2_HH
#define FALL_N_SYMMETRIC_TENSOR2_HH

// =============================================================================
//  SymmetricTensor2<dim> — Symmetric second-order tensor in ℝ^dim
// =============================================================================
//
//  Stores only the N independent components of a symmetric tensor T = Tᵀ
//  in Voigt notation, reusing the existing VoigtVector<N> convention:
//
//    dim=1  →  N=1:  { T₁₁ }
//    dim=2  →  N=3:  { T₁₁, T₂₂, T₁₂ }
//    dim=3  →  N=6:  { T₁₁, T₂₂, T₃₃, T₂₃, T₁₃, T₁₂ }
//
//  This matches the ordering of the existing VoigtVector<N> class, so
//  SymmetricTensor2 can interoperate directly with Strain<N>, Stress<N>.
//
//  IMPORTANT — Voigt factor convention:
//  ──────────────────────────────────────────────────────────────────────────
//  For STRESS-like tensors (σ, S, P), the Voigt vector stores the raw 
//  components:
//      {σ₁₁, σ₂₂, σ₃₃, σ₂₃, σ₁₃, σ₁₂}
//
//  For STRAIN-like tensors (ε, E, d), the engineering convention uses 
//  factor 2 on shear:
//      {ε₁₁, ε₂₂, ε₃₃, 2ε₂₃, 2ε₁₃, 2ε₁₂}   (= γ for shear)
//
//  This class stores TENSOR components (no factor 2). Conversions to/from
//  engineering Voigt are provided as explicit methods.
//  ──────────────────────────────────────────────────────────────────────────
//
//  Key operations:
//    • Conversion from/to full Tensor2<dim>
//    • Conversion from/to Voigt vector (with or without engineering factor)
//    • Invariants I₁, I₂, I₃
//    • Deviatoric / spherical decomposition
//    • Double contraction  S : E  (automatic handling of Voigt factor)
//    • Eigenvalue decomposition (spectral)
//
//  constexpr-friendly, zero-cost abstraction over Eigen.
//
// =============================================================================

#include <cstddef>
#include <cmath>
#include <array>

#include <Eigen/Dense>

#include "Tensor2.hh"

namespace continuum {

// ── Voigt dimension map ──────────────────────────────────────────────────────

template <std::size_t dim> requires ValidDim<dim>
consteval std::size_t voigt_size() {
    if constexpr (dim == 1) return 1;
    if constexpr (dim == 2) return 3;
    if constexpr (dim == 3) return 6;
}

// =============================================================================
//  SymmetricTensor2<dim>
// =============================================================================

template <std::size_t dim>
    requires ValidDim<dim>
class SymmetricTensor2 {
public:
    // ── Types and constants ──────────────────────────────────────────────────

    static constexpr std::size_t dimension     = dim;
    static constexpr std::size_t num_voigt     = voigt_size<dim>();
    static constexpr std::size_t num_full_comp = dim * dim;

    using VoigtVectorT = Eigen::Matrix<double, num_voigt, 1>;
    using MatrixT      = Eigen::Matrix<double, dim, dim>;
    using VectorT      = Eigen::Matrix<double, dim, 1>;

    // ── Voigt index maps ─────────────────────────────────────────────────────
    //
    // The Voigt convention maps the pair (i,j) with i≤j to a single index k.
    //
    // 3D:  (0,0)→0  (1,1)→1  (2,2)→2  (1,2)→3  (0,2)→4  (0,1)→5
    // 2D:  (0,0)→0  (1,1)→1  (0,1)→2
    // 1D:  (0,0)→0

    /// Map tensor indices (i,j) → Voigt index k.
    static constexpr std::size_t voigt_index(std::size_t i, std::size_t j) noexcept {
        // Ensure i ≤ j for the lookup
        if (i > j) { auto tmp = i; i = j; j = tmp; }

        if constexpr (dim == 1) {
            return 0;
        }
        else if constexpr (dim == 2) {
            // (0,0)→0  (1,1)→1  (0,1)→2
            if (i == j) return i;
            return 2;  // (0,1) → 2
        }
        else if constexpr (dim == 3) {
            // (0,0)→0  (1,1)→1  (2,2)→2  (1,2)→3  (0,2)→4  (0,1)→5
            if (i == j) return i;
            // Off-diagonal: (i,j) with i<j
            if (i == 1 && j == 2) return 3;
            if (i == 0 && j == 2) return 4;
            if (i == 0 && j == 1) return 5;
            return 0; // unreachable
        }
    }

    /// Inverse map: Voigt index k → tensor indices (i,j).
    /// Returns the pair with i ≤ j.
    static constexpr std::pair<std::size_t, std::size_t> tensor_indices(std::size_t k) noexcept {
        if constexpr (dim == 1) {
            return {0, 0};
        }
        else if constexpr (dim == 2) {
            constexpr std::pair<std::size_t, std::size_t> map[3] = {
                {0,0}, {1,1}, {0,1}
            };
            return map[k];
        }
        else if constexpr (dim == 3) {
            constexpr std::pair<std::size_t, std::size_t> map[6] = {
                {0,0}, {1,1}, {2,2}, {1,2}, {0,2}, {0,1}
            };
            return map[k];
        }
    }

    /// Is Voigt index k a diagonal (normal) component?
    static constexpr bool is_diagonal(std::size_t k) noexcept {
        auto [i, j] = tensor_indices(k);
        return i == j;
    }

    // ── Construction ─────────────────────────────────────────────────────────

    /// Default: zero tensor.
    constexpr SymmetricTensor2() noexcept : voigt_{VoigtVectorT::Zero()} {}

    /// From a Voigt vector (tensor components, NO factor 2 on shear).
    template <typename Derived>
    constexpr explicit SymmetricTensor2(const Eigen::MatrixBase<Derived>& v) noexcept
        : voigt_{v} {}

    /// From a full Tensor2 (only the symmetric part is kept).
    constexpr explicit SymmetricTensor2(const Tensor2<dim>& T) noexcept {
        auto S = T.symmetric_part();
        for (std::size_t k = 0; k < num_voigt; ++k) {
            auto [i, j] = tensor_indices(k);
            voigt_[k] = S(i, j);
        }
    }

    /// Static factory: identity tensor δ_ij.
    static constexpr SymmetricTensor2 identity() noexcept {
        SymmetricTensor2 t;
        for (std::size_t i = 0; i < dim; ++i)
            t.voigt_[i] = 1.0;
        return t;
    }

    /// Static factory: zero tensor.
    static constexpr SymmetricTensor2 zero() noexcept { return SymmetricTensor2{}; }

    /// Construct from individual components.
    /// 3D: (T11, T22, T33, T23, T13, T12)
    template <typename... Args>
        requires (sizeof...(Args) == num_voigt)
    constexpr SymmetricTensor2(Args... args) noexcept {
        double vals[] = {static_cast<double>(args)...};
        for (std::size_t k = 0; k < num_voigt; ++k)
            voigt_[k] = vals[k];
    }

    // ── Element access ───────────────────────────────────────────────────────

    /// Access by single Voigt index k (0-based).
    constexpr double  operator[](std::size_t k) const noexcept { return voigt_[k]; }
    constexpr double& operator[](std::size_t k)       noexcept { return voigt_[k]; }

    /// C++23 multidimensional subscript: T[i,j]
    /// Zero-cost "matrix view" — maps (i,j) → voigt_index without
    /// constructing the full dim×dim matrix.  Symmetric: T[i,j] == T[j,i].
    constexpr double  operator[](std::size_t i, std::size_t j) const noexcept {
        return voigt_[voigt_index(i, j)];
    }
    constexpr double& operator[](std::size_t i, std::size_t j)       noexcept {
        return voigt_[voigt_index(i, j)];
    }

    /// Access by tensor indices (i,j). Symmetric: T(i,j) == T(j,i).
    constexpr double operator()(std::size_t i, std::size_t j) const noexcept {
        return voigt_[voigt_index(i, j)];
    }

    /// Voigt vector (tensor components, NO factor of 2).
    constexpr const VoigtVectorT& voigt() const noexcept { return voigt_; }
    constexpr VoigtVectorT&       voigt()       noexcept { return voigt_; }

    /// Voigt vector with engineering shear strains (factor 2 on off-diag).
    /// Use this when converting a STRAIN tensor to Voigt for B·ε products.
    constexpr VoigtVectorT voigt_engineering() const noexcept {
        VoigtVectorT v = voigt_;
        for (std::size_t k = dim; k < num_voigt; ++k)
            v[k] *= 2.0;
        return v;
    }

    /// Set from engineering Voigt vector (divide shear components by 2).
    constexpr void set_from_engineering_voigt(const VoigtVectorT& v_eng) noexcept {
        voigt_ = v_eng;
        for (std::size_t k = dim; k < num_voigt; ++k)
            voigt_[k] /= 2.0;
    }

    // ── Conversion to full matrix ────────────────────────────────────────────
    //
    // DESIGN NOTE — Storage vs View
    // ─────────────────────────────────────────────────────────────────────────
    // The canonical storage is the Voigt vector (N components).  A true
    // zero-cost "view" onto a dim×dim matrix is impossible because the
    // Voigt layout  {T₁₁, T₂₂, T₃₃, T₂₃, T₁₃, T₁₂}  cannot be
    // reinterpreted as contiguous row-major or column-major 3×3 memory.
    //
    // Therefore matrix() is a *computed expansion*, called only when a full
    // dim×dim Eigen matrix is needed (eigendecomposition, Tensor2 interop,
    // F·S·Fᵀ products).  All performance-critical scalar operations —
    // trace, det, norm, I₁, I₂, I₃, double contraction, inverse — operate
    // directly on Voigt components with closed-form expressions.
    // ─────────────────────────────────────────────────────────────────────────

    /// Expand to full dim × dim symmetric Eigen matrix.
    /// This is a computed copy, not a view — use only when matrix form is
    /// actually needed (e.g. eigendecomposition, matrix products with F).
    constexpr MatrixT matrix() const noexcept {
        MatrixT m = MatrixT::Zero();
        for (std::size_t k = 0; k < num_voigt; ++k) {
            auto [i, j] = tensor_indices(k);
            m(i, j) = voigt_[k];
            m(j, i) = voigt_[k];
        }
        return m;
    }

    /// Convert to general Tensor2<dim>.
    constexpr Tensor2<dim> to_tensor2() const noexcept {
        return Tensor2<dim>{matrix()};
    }

    // ── Fundamental tensor operations ────────────────────────────────────────

    /// Trace: T_ii = T₁₁ + T₂₂ + T₃₃
    constexpr double trace() const noexcept {
        double tr = 0.0;
        for (std::size_t i = 0; i < dim; ++i)
            tr += voigt_[i];  // diagonal components are first
        return tr;
    }

    /// Determinant: det(T)
    /// Closed-form from Voigt components — no matrix construction.
    ///
    /// Voigt layout: {T₁₁, T₂₂, T₃₃, T₂₃, T₁₃, T₁₂}
    ///                 [0]  [1]  [2]  [3]  [4]  [5]
    constexpr double determinant() const noexcept {
        if constexpr (dim == 1) {
            return voigt_[0];
        }
        else if constexpr (dim == 2) {
            // det = T₁₁·T₂₂ − T₁₂²
            return voigt_[0] * voigt_[1] - voigt_[2] * voigt_[2];
        }
        else if constexpr (dim == 3) {
            // det = T₁₁(T₂₂·T₃₃ − T₂₃²)
            //     − T₁₂(T₁₂·T₃₃ − T₂₃·T₁₃)
            //     + T₁₃(T₁₂·T₂₃ − T₂₂·T₁₃)
            const double& a = voigt_[0];  // T₁₁
            const double& b = voigt_[1];  // T₂₂
            const double& c = voigt_[2];  // T₃₃
            const double& d = voigt_[3];  // T₂₃
            const double& e = voigt_[4];  // T₁₃
            const double& f = voigt_[5];  // T₁₂
            return a * (b * c - d * d)
                 - f * (f * c - d * e)
                 + e * (f * d - b * e);
        }
    }

    /// Frobenius norm: ‖T‖_F = sqrt(Σ T_ij²)
    /// For symmetric tensors: ‖T‖² = Σ_diag T_kk² + 2·Σ_offdiag T_ij²
    double norm() const noexcept {
        double n2 = 0.0;
        for (std::size_t k = 0; k < num_voigt; ++k) {
            double v = voigt_[k];
            n2 += is_diagonal(k) ? v * v : 2.0 * v * v;
        }
        return std::sqrt(n2);
    }

    /// Double contraction: A : B = A_ij B_ij
    /// For symmetric tensors: = Σ_diag A_k B_k + 2·Σ_offdiag A_k B_k
    constexpr double double_contract(const SymmetricTensor2& other) const noexcept {
        double result = 0.0;
        for (std::size_t k = 0; k < num_voigt; ++k) {
            double factor = is_diagonal(k) ? 1.0 : 2.0;
            result += factor * voigt_[k] * other.voigt_[k];
        }
        return result;
    }

    /// Spherical part: (1/dim) tr(T) I
    constexpr SymmetricTensor2 spherical() const noexcept {
        return identity() * (trace() / static_cast<double>(dim));
    }

    /// Deviatoric part: dev(T) = T − (1/dim) tr(T) I
    constexpr SymmetricTensor2 deviatoric() const noexcept {
        return *this - spherical();
    }

    /// Inverse: T⁻¹ (assumes non-singular).
    /// Closed-form cofactor / determinant — no matrix construction.
    constexpr SymmetricTensor2 inverse() const noexcept {
        if constexpr (dim == 1) {
            SymmetricTensor2 r;
            r.voigt_[0] = 1.0 / voigt_[0];
            return r;
        }
        else if constexpr (dim == 2) {
            // [T₂₂, −T₁₂; −T₁₂, T₁₁] / det
            const double inv_det = 1.0 / determinant();
            SymmetricTensor2 r;
            r.voigt_[0] =  voigt_[1] * inv_det;  // T₂₂ / det
            r.voigt_[1] =  voigt_[0] * inv_det;  // T₁₁ / det
            r.voigt_[2] = -voigt_[2] * inv_det;  // −T₁₂ / det
            return r;
        }
        else if constexpr (dim == 3) {
            // Closed-form adjugate / det for the 3×3 symmetric matrix:
            //
            //      M = [a  f  e]       Voigt: {a, b, c, d, e, f}
            //          [f  b  d]                T₁₁ T₂₂ T₃₃ T₂₃ T₁₃ T₁₂
            //          [e  d  c]                [0] [1]  [2] [3]  [4] [5]
            //
            //  M⁻¹ = (1/det) · adj(M), with adj(M) symmetric:
            //
            //      adj(0,0) = bc − d²       adj(0,1) = de − fc     adj(0,2) = fd − be
            //      adj(1,0) = de − fc       adj(1,1) = ac − e²     adj(1,2) = ef − ad
            //      adj(2,0) = fd − be       adj(2,1) = ef − ad     adj(2,2) = ab − f²
            //
            const double& a = voigt_[0];  // T₁₁
            const double& b = voigt_[1];  // T₂₂
            const double& c = voigt_[2];  // T₃₃
            const double& d = voigt_[3];  // T₂₃
            const double& e = voigt_[4];  // T₁₃
            const double& f = voigt_[5];  // T₁₂

            const double inv_det = 1.0 / determinant();

            SymmetricTensor2 r;
            r.voigt_[0] = (b * c - d * d) * inv_det;  // adj(0,0)
            r.voigt_[1] = (a * c - e * e) * inv_det;  // adj(1,1)
            r.voigt_[2] = (a * b - f * f) * inv_det;  // adj(2,2)
            r.voigt_[3] = (e * f - a * d) * inv_det;  // adj(1,2)
            r.voigt_[4] = (f * d - b * e) * inv_det;  // adj(0,2)
            r.voigt_[5] = (d * e - f * c) * inv_det;  // adj(0,1)
            return r;
        }
    }

    // ── Invariants ───────────────────────────────────────────────────────────
    //
    // Principal invariants of a symmetric tensor T:
    //   I₁ = tr(T) = T₁₁ + T₂₂ + T₃₃
    //   I₂ = ½[tr(T)² − tr(T²)] = T₁₁T₂₂ + T₂₂T₃₃ + T₁₁T₃₃ − T₁₂² − T₂₃² − T₁₃²
    //   I₃ = det(T)
    //
    // For the right Cauchy-Green tensor C = FᵀF:
    //   I₁ = tr(C),  I₂ = ½[tr(C)² − tr(C²)],  I₃ = det(C) = J²

    constexpr double I1() const noexcept { return trace(); }

    constexpr double I2() const noexcept {
        if constexpr (dim == 1) {
            return 0.0;
        }
        else if constexpr (dim == 2) {
            // I₂ = T₁₁·T₂₂ − T₁₂²
            return voigt_[0] * voigt_[1] - voigt_[2] * voigt_[2];
        }
        else if constexpr (dim == 3) {
            // I₂ = T₁₁T₂₂ + T₂₂T₃₃ + T₁₁T₃₃ − T₂₃² − T₁₃² − T₁₂²
            return voigt_[0] * voigt_[1]
                 + voigt_[1] * voigt_[2]
                 + voigt_[0] * voigt_[2]
                 - voigt_[3] * voigt_[3]
                 - voigt_[4] * voigt_[4]
                 - voigt_[5] * voigt_[5];
        }
    }

    constexpr double I3() const noexcept { return determinant(); }

    // ── Eigenvalue decomposition ─────────────────────────────────────────────
    //
    // Returns eigenvalues in ascending order.
    // For dim=3 this is a full spectral decomposition: T = Σ λᵢ nᵢ⊗nᵢ
    // Required for Hencky strain: ε_H = Σ ½ ln(λᵢ) nᵢ⊗nᵢ

    struct SpectralDecomposition {
        Eigen::Matrix<double, dim, 1>   eigenvalues;  // ascending order
        Eigen::Matrix<double, dim, dim> eigenvectors;  // columns = eigenvectors
    };

    SpectralDecomposition eigendecomposition() const {
        Eigen::SelfAdjointEigenSolver<MatrixT> solver(matrix());
        return {solver.eigenvalues(), solver.eigenvectors()};
    }

    /// Eigenvalues only (ascending order).
    Eigen::Matrix<double, dim, 1> eigenvalues() const {
        Eigen::SelfAdjointEigenSolver<MatrixT> solver(matrix(),
            Eigen::EigenvaluesOnly);
        return solver.eigenvalues();
    }

    // ── Tensor functions via spectral decomposition ──────────────────────────
    //
    // For a symmetric positive-definite tensor T with spectral decomposition
    //   T = Σ λᵢ nᵢ⊗nᵢ
    // we define:
    //   f(T) = Σ f(λᵢ) nᵢ⊗nᵢ
    //
    // This applies to: sqrt(T), log(T), exp(T), pow(T, α), etc.

    /// Square root: T^{1/2}.  Requires T positive definite.
    SymmetricTensor2 sqrt() const {
        auto [evals, evecs] = eigendecomposition();
        MatrixT result = MatrixT::Zero();
        for (std::size_t i = 0; i < dim; ++i) {
            VectorT ni = evecs.col(i);
            result += std::sqrt(evals[i]) * ni * ni.transpose();
        }
        return SymmetricTensor2{Tensor2<dim>{result}};
    }

    /// Natural logarithm: ln(T).  Requires T positive definite.
    /// This is the core of the Hencky (logarithmic) strain:
    ///   ε_H = ½ ln(C) = ½ ln(FᵀF)
    SymmetricTensor2 log() const {
        auto [evals, evecs] = eigendecomposition();
        MatrixT result = MatrixT::Zero();
        for (std::size_t i = 0; i < dim; ++i) {
            VectorT ni = evecs.col(i);
            result += std::log(evals[i]) * ni * ni.transpose();
        }
        return SymmetricTensor2{Tensor2<dim>{result}};
    }

    /// Exponential: exp(T).
    SymmetricTensor2 exp() const {
        auto [evals, evecs] = eigendecomposition();
        MatrixT result = MatrixT::Zero();
        for (std::size_t i = 0; i < dim; ++i) {
            VectorT ni = evecs.col(i);
            result += std::exp(evals[i]) * ni * ni.transpose();
        }
        return SymmetricTensor2{Tensor2<dim>{result}};
    }

    /// Power: T^α.  Requires T positive definite for non-integer α.
    SymmetricTensor2 pow(double alpha) const {
        auto [evals, evecs] = eigendecomposition();
        MatrixT result = MatrixT::Zero();
        for (std::size_t i = 0; i < dim; ++i) {
            VectorT ni = evecs.col(i);
            result += std::pow(evals[i], alpha) * ni * ni.transpose();
        }
        return SymmetricTensor2{Tensor2<dim>{result}};
    }

    // ── Arithmetic operators ─────────────────────────────────────────────────

    constexpr SymmetricTensor2 operator+(const SymmetricTensor2& other) const noexcept {
        return SymmetricTensor2{(voigt_ + other.voigt_).eval()};
    }

    constexpr SymmetricTensor2 operator-(const SymmetricTensor2& other) const noexcept {
        return SymmetricTensor2{(voigt_ - other.voigt_).eval()};
    }

    constexpr SymmetricTensor2 operator*(double scalar) const noexcept {
        return SymmetricTensor2{(voigt_ * scalar).eval()};
    }

    constexpr SymmetricTensor2 operator/(double scalar) const noexcept {
        return SymmetricTensor2{(voigt_ / scalar).eval()};
    }

    constexpr SymmetricTensor2& operator+=(const SymmetricTensor2& other) noexcept {
        voigt_ += other.voigt_;
        return *this;
    }

    constexpr SymmetricTensor2& operator-=(const SymmetricTensor2& other) noexcept {
        voigt_ -= other.voigt_;
        return *this;
    }

    constexpr SymmetricTensor2& operator*=(double scalar) noexcept {
        voigt_ *= scalar;
        return *this;
    }

    constexpr SymmetricTensor2 operator-() const noexcept {
        return SymmetricTensor2{(-voigt_).eval()};
    }

    friend constexpr SymmetricTensor2 operator*(double s, const SymmetricTensor2& t) noexcept {
        return t * s;
    }

    // ── Comparison ───────────────────────────────────────────────────────────

    bool approx_equal(const SymmetricTensor2& other, double tol = 1e-12) const noexcept {
        return (voigt_ - other.voigt_).norm() < tol;
    }

private:
    VoigtVectorT voigt_;
};

// ── Free function aliases ────────────────────────────────────────────────────

template <std::size_t dim>
constexpr double trace(const SymmetricTensor2<dim>& T) noexcept { return T.trace(); }

template <std::size_t dim>
constexpr double det(const SymmetricTensor2<dim>& T) noexcept { return T.determinant(); }

template <std::size_t dim>
SymmetricTensor2<dim> inv(const SymmetricTensor2<dim>& T) noexcept { return T.inverse(); }

template <std::size_t dim>
constexpr SymmetricTensor2<dim> dev(const SymmetricTensor2<dim>& T) noexcept { return T.deviatoric(); }

template <std::size_t dim>
constexpr double double_contract(const SymmetricTensor2<dim>& A,
                                  const SymmetricTensor2<dim>& B) noexcept {
    return A.double_contract(B);
}

} // namespace continuum

#endif // FALL_N_SYMMETRIC_TENSOR2_HH
