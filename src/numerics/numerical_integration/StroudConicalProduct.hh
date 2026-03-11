#ifndef FALL_N_STROUD_CONICAL_PRODUCT_HH
#define FALL_N_STROUD_CONICAL_PRODUCT_HH

// =============================================================================
//  StroudConicalProduct.hh — Arbitrary-order simplex quadrature via
//                            Duffy / collapsed-coordinate transformation
// =============================================================================
//
//  The Stroud conical product rule maps the D-simplex to [0,1]^D using:
//
//      ξ₁ = t₁
//      ξ_k = t_k ∏_{j=1}^{k-1} (1 − t_j)     for k = 2, …, D
//
//  Jacobian:  ∏_{k=1}^{D-1} (1 − t_k)^{D−k}
//
//  Each direction k uses Gauss–Jacobi quadrature with weight (1−t)^{D−1−k}
//  on [0,1], absorbing the Jacobian into the 1D rule.  Direction D (last)
//  uses plain Gauss–Legendre on [0,1].
//
//  ┌──────────────────────────────────────────────────────────────────────┐
//  │  ALL WEIGHTS ARE STRICTLY POSITIVE for any polynomial order.        │
//  │  This avoids the negative-weight artefacts of Grundmann–Möller,     │
//  │  Keast-5 and some Dunavant rules, which can corrupt lumped L2       │
//  │  projections on quadratic (or higher) simplex elements.             │
//  └──────────────────────────────────────────────────────────────────────┘
//
//  Number of points:   n^D     (n = Gauss points per direction)
//  Degree of exactness:  2n − 1
//
//  References:
//    A. H. Stroud, "Approximate Calculation of Multiple Integrals",
//      Prentice-Hall, 1971.
//    M. G. Duffy, "Quadrature over a Pyramid or Cube of Integrands
//      with a Singularity at a Vertex", SIAM J. Numer. Anal. 19(6), 1982.
//    J. Burkardt, "simplex_gm_rule" (MIT-licensed reference implementations).
//
// =============================================================================
//
//  Negative-weight problem (motivation for this module):
//
//  Several classical symmetric simplex rules (Keast degree-3 with 5 points,
//  Dunavant 2D degree-3 with 4 points, Grundmann–Möller for s ≥ 1) contain
//  one or more negative quadrature weights.
//
//  When these rules are used in a lumped (row-sum) L2 projection:
//
//      σ̃_I = ∑_e ∑_g N_I(ξ_g) w_g |J_g| σ_g   /   ∑_e ∑_g N_I(ξ_g) w_g |J_g|
//
//  the denominator can become negative or zero at vertex nodes of quadratic
//  (or higher) simplex elements, because the vertex basis functions
//  (e.g., N_i = λ_i(2λ_i − 1) for TET10) go negative at interior
//  quadrature points.  A negative denominator causes the projected value
//  to flip sign, producing a characteristic element-level checkerboard
//  ("diamond") pattern in Paraview.
//
//  This module provides the Stroud conical product as a drop-in replacement
//  that guarantees all-positive weights for any polynomial degree, completely
//  eliminating the checkerboard artefact.
//
// =============================================================================

#include <Eigen/Dense>
#include <array>
#include <vector>
#include <cmath>
#include <cassert>
#include <utility>

namespace simplex_quadrature {


// ─── Gauss–Jacobi quadrature on [−1, 1] ────────────────────────────────
//
//  Weight function: w(x) = (1 − x)^α (1 + x)^β,  α, β > −1
//
//  Golub–Welsch algorithm: the n quadrature nodes are eigenvalues of a
//  symmetric tridiagonal matrix built from the three-term recurrence
//  coefficients of the monic Jacobi polynomial.  Weights are computed from
//  the first component of each normalized eigenvector.  Uses Eigen's
//  SelfAdjointEigenSolver.  Robust for any α, β > −1.
//


struct GaussJacobiRule {
    std::vector<double> nodes;    // in [−1, 1]
    std::vector<double> weights;  // include the weight function
};


/// Compute the n-point Gauss–Jacobi rule on [−1, 1] with weight
/// (1−x)^α (1+x)^β via the Golub–Welsch algorithm.
/// Nodes are sorted in ascending order.
inline auto gauss_jacobi(int n, double alpha, double beta)
    -> GaussJacobiRule
{
    assert(n >= 1);
    assert(alpha > -1.0 && beta > -1.0);

    double ab = alpha + beta;

    // μ₀ = ∫_{-1}^{1} (1−x)^α (1+x)^β dx = 2^{α+β+1} B(α+1, β+1)
    double mu0 = std::pow(2.0, ab + 1.0)
               * std::tgamma(alpha + 1.0) * std::tgamma(beta + 1.0)
               / std::tgamma(ab + 2.0);

    if (n == 1) {
        // Single-point rule: node = (β−α)/(α+β+2), weight = μ₀
        double x0 = (beta - alpha) / (ab + 2.0);
        return {{x0}, {mu0}};
    }

    // ── Build the symmetric tridiagonal Jacobi matrix ────────────────
    //
    //  The monic Jacobi polynomial p̂_k satisfies:
    //     p̂_{k+1}(x) = (x − a_k) p̂_k(x) − b_k p̂_{k-1}(x)
    //
    //  The tridiagonal matrix T has:
    //     T(k,k)   = a_k          (diagonal)
    //     T(k,k+1) = T(k+1,k) = √b_{k+1}  (off-diagonal)
    //
    //  Recurrence coefficients for Jacobi (cf. Gautschi 2004, Table 1.1):
    //     a_k = (β² − α²) / ((2k+α+β)(2k+α+β+2))
    //     b_k = 4k(k+α)(k+β)(k+α+β) / ((2k+α+β)²(2k+α+β+1)(2k+α+β−1))
    //

    Eigen::VectorXd diag(n);
    Eigen::VectorXd offdiag(n - 1);

    for (int k = 0; k < n; ++k) {
        double kk = static_cast<double>(k);
        double denom1 = 2.0 * kk + ab;
        double denom2 = denom1 + 2.0;

        // Handle the special case 2k+α+β = 0 (only at k=0 when α+β=0)
        if (std::abs(denom1 * denom2) < 1.0e-30) {
            diag[k] = (beta - alpha) / (ab + 2.0);
        } else {
            diag[k] = (beta * beta - alpha * alpha) / (denom1 * denom2);
        }

        if (k > 0) {
            double num = 4.0 * kk * (kk + alpha) * (kk + beta) * (kk + ab);
            double d   = denom1 * denom1 * (denom1 + 1.0) * (denom1 - 1.0);
            offdiag[k - 1] = std::sqrt(num / d);
        }
    }

    // ── Solve the eigenvalue problem ─────────────────────────────────
    //  Build the full tridiagonal matrix and use Eigen.
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(n, n);
    for (int k = 0; k < n; ++k) {
        T(k, k) = diag[k];
    }
    for (int k = 0; k < n - 1; ++k) {
        T(k, k + 1) = offdiag[k];
        T(k + 1, k) = offdiag[k];
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(T);

    GaussJacobiRule rule;
    rule.nodes.resize(n);
    rule.weights.resize(n);

    for (int i = 0; i < n; ++i) {
        rule.nodes[i]   = solver.eigenvalues()(i);
        double v0       = solver.eigenvectors()(0, i);
        rule.weights[i] = mu0 * v0 * v0;
    }

    return rule;
}


/// Gauss–Jacobi on [0, 1] with weight (1 − t)^β_power.
///
/// Maps the standard Jacobi rule for (1−x)^α on [−1,1] to [0,1] via x = 2t−1:
///   ∫₀¹ f(t)(1−t)^β dt  =  1/2^{β+1} · ∫₋₁¹ f((1+x)/2)(1−x)^β dx
///
/// All weights are strictly positive.
inline auto gauss_jacobi_01(int n, double beta_power)
    -> std::pair<std::vector<double>, std::vector<double>>
{
    auto rule = gauss_jacobi(n, beta_power, 0.0);  // weight (1−x)^β on [−1,1]

    double scale = 1.0 / std::pow(2.0, beta_power + 1.0);

    std::vector<double> nodes_01(n), weights_01(n);
    for (int i = 0; i < n; ++i) {
        nodes_01[i]   = (1.0 + rule.nodes[i]) / 2.0;
        weights_01[i] = rule.weights[i] * scale;
    }

    return {nodes_01, weights_01};
}


// ─── Stroud conical product for D-simplex ───────────────────────────────
//
//  Input:  n_per_dir = number of Gauss points per collapsed direction
//  Output: n_per_dir^D quadrature points in the reference simplex
//          { ξ_i ≥ 0,  Σξ_i ≤ 1 }
//          with weights summing to 1/D!  (volume of the reference simplex)
//
//  Degree of exactness: 2·n_per_dir − 1
//

template <std::size_t Dim>
struct ConicalProductRule {
    std::vector<std::array<double, Dim>> points;
    std::vector<double>                  weights;
    std::size_t                          num_points;
};


template <std::size_t Dim>
inline auto stroud_conical_product(int n_per_dir) -> ConicalProductRule<Dim>
{
    static_assert(Dim >= 1 && Dim <= 3,
        "Stroud conical product implemented for Dim = 1, 2, 3.");
    assert(n_per_dir >= 1);

    // ── 1D Gauss–Jacobi rules for each collapsed direction ───────────
    //
    //  Direction k (0-indexed): weight (1−t)^{Dim−1−k} on [0,1]
    //    k = 0       : β = Dim−1
    //    k = 1       : β = Dim−2
    //    ...
    //    k = Dim−1   : β = 0  (Gauss–Legendre)
    //

    struct Rule1D { std::vector<double> nodes, weights; };

    std::array<Rule1D, Dim> rules;
    for (std::size_t k = 0; k < Dim; ++k) {
        double beta = static_cast<double>(Dim - 1 - k);
        auto [nodes, weights] = gauss_jacobi_01(n_per_dir, beta);
        rules[k] = {std::move(nodes), std::move(weights)};
    }

    // ── Tensor product in collapsed coordinates ──────────────────────
    std::size_t total = 1;
    for (std::size_t k = 0; k < Dim; ++k) total *= static_cast<std::size_t>(n_per_dir);

    ConicalProductRule<Dim> result;
    result.num_points = total;
    result.points.resize(total);
    result.weights.resize(total);

    for (std::size_t flat = 0; flat < total; ++flat) {
        // Decode flat index into multi-index [i₀, i₁, …, i_{D-1}]
        std::array<std::size_t, Dim> idx{};
        {
            std::size_t tmp = flat;
            for (std::size_t k = Dim; k > 0; --k) {
                idx[k - 1] = tmp % static_cast<std::size_t>(n_per_dir);
                tmp /= static_cast<std::size_t>(n_per_dir);
            }
        }

        // Build simplex coordinates via Duffy transform:
        //   ξ_0 = t_0
        //   ξ_k = t_k · ∏_{j<k} (1 − t_j)
        double weight = 1.0;
        double running_product = 1.0;   // = ∏_{j<k} (1 − t_j)

        for (std::size_t k = 0; k < Dim; ++k) {
            double tk = rules[k].nodes[idx[k]];
            result.points[flat][k] = running_product * tk;
            weight *= rules[k].weights[idx[k]];
            running_product *= (1.0 - tk);
        }

        result.weights[flat] = weight;
    }

    return result;
}


// ─── ConicalProductIntegrator<TopDim, NPerDir> ──────────────────────────
//
//  A compile-time-sized integrator that is compatible with the existing
//  ElementGeometry / OwningModel architecture (requires static constexpr
//  num_integration_points).
//
//  NPerDir = Gauss points per collapsed direction.
//    NPerDir = 1 → 1   point  (degree 1)  — same as centroid rule
//    NPerDir = 2 → 2^D points (degree 3)
//    NPerDir = 3 → 3^D points (degree 5)
//    NPerDir = 4 → 4^D points (degree 7)
//
//  All weights are strictly positive.
//

template <std::size_t TopDim, std::size_t NPerDir>
class ConicalProductIntegrator
{
    static constexpr std::size_t N = []{
        std::size_t val = 1;
        for (std::size_t k = 0; k < TopDim; ++k) val *= NPerDir;
        return val;
    }();

    using Point          = std::array<double, TopDim>;
    using LocalCoordView = std::span<const double>;

    // ── Static storage, populated once at program startup ────────────
    struct RuleData {
        std::array<Point,  N> points{};
        std::array<double, N> weights{};
    };

    static inline const RuleData rule_ = []{
        auto cpr = stroud_conical_product<TopDim>(static_cast<int>(NPerDir));
        RuleData r{};
        for (std::size_t i = 0; i < N; ++i) {
            r.points[i]  = cpr.points[i];
            r.weights[i] = cpr.weights[i];
        }
        return r;
    }();

public:

    static constexpr std::size_t num_integration_points = N;

    static auto reference_integration_point(std::size_t i) noexcept
        -> LocalCoordView
    {
        return LocalCoordView{rule_.points[i].data(), TopDim};
    }

    static double weight(std::size_t i) noexcept {
        return rule_.weights[i];
    }

    /// Pure quadrature: ∑ wᵢ · f(ξᵢ)
    /// Does NOT multiply by |J| — the element applies its own differential measure.
    decltype(auto) operator()(std::invocable<LocalCoordView> auto&& f) const noexcept {
        using ReturnType = std::invoke_result_t<decltype(f), LocalCoordView>;

        if constexpr (std::is_arithmetic_v<std::decay_t<ReturnType>>) {
            double result = 0.0;
            for (std::size_t i = 0; i < N; ++i) {
                result += rule_.weights[i]
                        * f(LocalCoordView{rule_.points[i].data(), TopDim});
            }
            return result;
        }
        else {
            // Eigen matrices or other algebraic types
            auto result = (f(LocalCoordView{rule_.points[0].data(), TopDim})
                          * rule_.weights[0]).eval();
            for (std::size_t i = 1; i < N; ++i) {
                result += f(LocalCoordView{rule_.points[i].data(), TopDim})
                        * rule_.weights[i];
            }
            return result;
        }
    }

    ConicalProductIntegrator() noexcept = default;
    ~ConicalProductIntegrator() noexcept = default;
};


} // namespace simplex_quadrature

#endif // FALL_N_STROUD_CONICAL_PRODUCT_HH
