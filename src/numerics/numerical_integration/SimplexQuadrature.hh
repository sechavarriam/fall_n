#ifndef FALL_N_SIMPLEX_QUADRATURE_HH
#define FALL_N_SIMPLEX_QUADRATURE_HH

// =============================================================================
//  SimplexQuadrature.hh — Gaussian quadrature rules on reference simplices
// =============================================================================
//
//  Reference simplex in D dimensions:
//    S_D = { (ξ₁, …, ξ_D) : ξ_i ≥ 0, ξ₁ + … + ξ_D ≤ 1 }
//
//  Volume:  |S_D| = 1/D!
//    D=1:  1      (segment [0,1])
//    D=2:  1/2    (triangle)
//    D=3:  1/6    (tetrahedron)
//
//  Weights sum to |S_D|.
//
// =============================================================================

#include <array>
#include <cstddef>

namespace simplex_quadrature {

template <std::size_t Dim, std::size_t NumPoints>
struct SimplexQuadratureRule {
    using Point = std::array<double, Dim>;

    std::array<Point , NumPoints> points{};
    std::array<double, NumPoints> weights{};

    static constexpr std::size_t num_points = NumPoints;
    static constexpr std::size_t dim        = Dim;

    constexpr const Point& get_point_coords(std::size_t i)  const noexcept { return points [i]; }
    constexpr double       get_point_weight (std::size_t i)  const noexcept { return weights[i]; }
};


// =============================================================================
//  Factory functions returning consteval SimplexQuadratureRule objects
// =============================================================================

// ── 1D rules (segment [0, 1], length = 1) ───────────────────────────────

/// 1 point, degree 1: midpoint rule
inline consteval auto make_simplex_rule_1d_1() {
    SimplexQuadratureRule<1, 1> r;
    r.points [0] = {0.5};
    r.weights[0] = 1.0;
    return r;
}

/// 2 points, degree 3: Gauss-Legendre mapped to [0,1]
inline consteval auto make_simplex_rule_1d_2() {
    SimplexQuadratureRule<1, 2> r;
    r.points [0] = {0.2113248654051871};
    r.points [1] = {0.7886751345948129};
    r.weights[0] = 0.5;
    r.weights[1] = 0.5;
    return r;
}

/// 3 points, degree 5: Gauss-Legendre mapped to [0,1]
inline consteval auto make_simplex_rule_1d_3() {
    SimplexQuadratureRule<1, 3> r;
    r.points [0] = {0.1127016653792583};
    r.points [1] = {0.5};
    r.points [2] = {0.8872983346207417};
    r.weights[0] = 0.2777777777777778;    // 5/18
    r.weights[1] = 0.4444444444444444;    // 8/18
    r.weights[2] = 0.2777777777777778;    // 5/18
    return r;
}


// ── 2D rules (triangle, area = 1/2) ────────────────────────────────────

/// 1 point, degree 1: centroid rule
inline consteval auto make_simplex_rule_2d_1() {
    SimplexQuadratureRule<2, 1> r;
    r.points [0] = {1.0/3.0, 1.0/3.0};
    r.weights[0] = 0.5;
    return r;
}

/// 3 points, degree 2: Hammer–Stroud
inline consteval auto make_simplex_rule_2d_3() {
    SimplexQuadratureRule<2, 3> r;
    r.points [0] = {1.0/6.0, 1.0/6.0};
    r.points [1] = {2.0/3.0, 1.0/6.0};
    r.points [2] = {1.0/6.0, 2.0/3.0};
    r.weights[0] = 1.0/6.0;
    r.weights[1] = 1.0/6.0;
    r.weights[2] = 1.0/6.0;
    return r;
}

/// 4 points, degree 3 (Dunavant)
inline consteval auto make_simplex_rule_2d_4() {
    SimplexQuadratureRule<2, 4> r;
    r.points [0] = {1.0/3.0, 1.0/3.0};
    r.points [1] = {0.6, 0.2};
    r.points [2] = {0.2, 0.6};
    r.points [3] = {0.2, 0.2};
    r.weights[0] = -0.28125;                // = -9/32
    r.weights[1] =  0.26041666666666669;     // = 25/96
    r.weights[2] =  0.26041666666666669;
    r.weights[3] =  0.26041666666666669;
    return r;
}

/// 7 points, degree 5 (Dunavant)
inline consteval auto make_simplex_rule_2d_7() {
    SimplexQuadratureRule<2, 7> r;
    r.points [0] = {0.33333333333333333, 0.33333333333333333};   // centroid
    r.points [1] = {0.47014206410511509, 0.47014206410511509};   // type 1
    r.points [2] = {0.05971587178976982, 0.47014206410511509};
    r.points [3] = {0.47014206410511509, 0.05971587178976982};
    r.points [4] = {0.10128650732345634, 0.10128650732345634};   // type 2
    r.points [5] = {0.79742698535308732, 0.10128650732345634};
    r.points [6] = {0.10128650732345634, 0.79742698535308732};
    r.weights[0] = 0.1125;                                      // = 0.225/2
    r.weights[1] = 0.066197076394253090;
    r.weights[2] = 0.066197076394253090;
    r.weights[3] = 0.066197076394253090;
    r.weights[4] = 0.062969590272413576;
    r.weights[5] = 0.062969590272413576;
    r.weights[6] = 0.062969590272413576;
    return r;
}


// ── 3D rules (tetrahedron, volume = 1/6) ──────────────────────────────

/// 1 point, degree 1: centroid rule
inline consteval auto make_simplex_rule_3d_1() {
    SimplexQuadratureRule<3, 1> r;
    r.points [0] = {0.25, 0.25, 0.25};
    r.weights[0] = 1.0/6.0;
    return r;
}

/// 4 points, degree 2: Hammer–Marlowe–Stroud
///   α = (5 − √5)/20 ≈ 0.13819660112501052
///   β = (5 + 3√5)/20 ≈ 0.58541019662496845
inline consteval auto make_simplex_rule_3d_4() {
    constexpr double a = 0.13819660112501052;
    constexpr double b = 0.58541019662496845;

    SimplexQuadratureRule<3, 4> r;
    r.points [0] = {a, a, a};
    r.points [1] = {b, a, a};
    r.points [2] = {a, b, a};
    r.points [3] = {a, a, b};
    r.weights[0] = 1.0/24.0;
    r.weights[1] = 1.0/24.0;
    r.weights[2] = 1.0/24.0;
    r.weights[3] = 1.0/24.0;
    return r;
}

/// 5 points, degree 3: Keast rule #0
///   Point 0: centroid (1/4, 1/4, 1/4),   w₀ = −2/15
///   Points 1–4: α = 1/6, β = 1/2,        w₁ = 3/40
inline consteval auto make_simplex_rule_3d_5() {
    SimplexQuadratureRule<3, 5> r;
    r.points [0] = {0.25,    0.25,    0.25   };
    r.points [1] = {1.0/6.0, 1.0/6.0, 1.0/6.0};
    r.points [2] = {0.5,     1.0/6.0, 1.0/6.0};
    r.points [3] = {1.0/6.0, 0.5,     1.0/6.0};
    r.points [4] = {1.0/6.0, 1.0/6.0, 0.5    };
    r.weights[0] = -2.0/15.0;     // = −0.1333…
    r.weights[1] =  3.0/40.0;     // =  0.075
    r.weights[2] =  3.0/40.0;
    r.weights[3] =  3.0/40.0;
    r.weights[4] =  3.0/40.0;
    return r;
}


// =============================================================================
//  Default rule selector — picks an appropriate rule for (Dim, Order)
// =============================================================================

template <std::size_t Dim, std::size_t Order>
consteval auto default_simplex_rule() {
    if constexpr (Dim == 1 && Order == 1) {
        return make_simplex_rule_1d_1();      // 1-point, degree 1
    }
    else if constexpr (Dim == 1 && Order == 2) {
        return make_simplex_rule_1d_2();      // 2-point, degree 3
    }
    else if constexpr (Dim == 2 && Order == 1) {
        return make_simplex_rule_2d_3();      // 3-point, degree 2
    }
    else if constexpr (Dim == 2 && Order == 2) {
        return make_simplex_rule_2d_7();      // 7-point, degree 5
    }
    else if constexpr (Dim == 3 && Order == 1) {
        return make_simplex_rule_3d_4();      // 4-point, degree 2
    }
    else if constexpr (Dim == 3 && Order == 2) {
        return make_simplex_rule_3d_5();      // 5-point, degree 3
    }
    else {
        static_assert(Dim <= 3 && Order <= 2,
            "No default simplex quadrature rule for this (Dim, Order) pair.");
    }
}


} // namespace simplex_quadrature

#endif // FALL_N_SIMPLEX_QUADRATURE_HH
