// =============================================================================
//  test_simplex.cpp — Unit tests for the simplex (tetrahedral) element family
// =============================================================================
//
//  Tests cover:
//    1. SimplexCell reference nodes & subentity topology
//    2. SimplexBasis shape functions (partition of unity, Kronecker delta)
//    3. SimplexQuadrature accuracy (constant, linear, quadratic integrands)
//    4. SimplexElement isoparametric mapping & Jacobian
//    5. SimplexIntegrator integration through ElementGeometry
//    6. VTK cell type dispatch
//    7. Face (sub-entity) geometry creation
//

#include <cassert>
#include <cmath>
#include <cstddef>
#include <array>
#include <iostream>
#include <span>
#include <numeric>

#include <vtkNew.h>
#include <vtkTriQuadraticHexahedron.h>

#include "header_files.hh"

namespace {

constexpr bool approx(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol;
}

int passed = 0, failed = 0;

void report(const char *name, bool ok) {
    if (ok) { ++passed; std::cout << "  PASS  " << name << "\n"; }
    else    { ++failed; std::cout << "  FAIL  " << name << "\n"; }
}

// =========================================================================
//  1. SimplexCell reference node layout
// =========================================================================

void test_simplex_cell_num_nodes() {
    using namespace geometry::simplex;
    bool ok = true;

    // Linear: num_nodes = D+1
    ok = ok && (simplex_num_nodes(1, 1) == 2);
    ok = ok && (simplex_num_nodes(2, 1) == 3);
    ok = ok && (simplex_num_nodes(3, 1) == 4);

    // Quadratic: num_nodes = (D+1)(D+2)/2
    ok = ok && (simplex_num_nodes(1, 2) == 3);
    ok = ok && (simplex_num_nodes(2, 2) == 6);
    ok = ok && (simplex_num_nodes(3, 2) == 10);

    report(__func__, ok);
}

void test_simplex_cell_reference_nodes_tet4() {
    using Cell = geometry::simplex::SimplexCell<3, 1>;
    bool ok = true;

    // 4 vertices of the standard tetrahedron
    constexpr auto nodes = Cell::reference_nodes;
    ok = ok && (nodes.size() == 4);

    // Vertex 0: origin
    ok = ok && approx(nodes[0].coord(0), 0.0);
    ok = ok && approx(nodes[0].coord(1), 0.0);
    ok = ok && approx(nodes[0].coord(2), 0.0);

    // Vertex 1: (1,0,0)
    ok = ok && approx(nodes[1].coord(0), 1.0);
    ok = ok && approx(nodes[1].coord(1), 0.0);
    ok = ok && approx(nodes[1].coord(2), 0.0);

    // Vertex 2: (0,1,0)
    ok = ok && approx(nodes[2].coord(0), 0.0);
    ok = ok && approx(nodes[2].coord(1), 1.0);
    ok = ok && approx(nodes[2].coord(2), 0.0);

    // Vertex 3: (0,0,1)
    ok = ok && approx(nodes[3].coord(0), 0.0);
    ok = ok && approx(nodes[3].coord(1), 0.0);
    ok = ok && approx(nodes[3].coord(2), 1.0);

    report(__func__, ok);
}

void test_simplex_cell_reference_nodes_tet10() {
    using Cell = geometry::simplex::SimplexCell<3, 2>;
    bool ok = true;

    constexpr auto nodes = Cell::reference_nodes;
    ok = ok && (nodes.size() == 10);

    // First 4 are vertices (same as tet4)
    ok = ok && approx(nodes[0].coord(0), 0.0);
    ok = ok && approx(nodes[1].coord(0), 1.0);
    ok = ok && approx(nodes[2].coord(1), 1.0);
    ok = ok && approx(nodes[3].coord(2), 1.0);

    // Midpoint (0,1) at index 4 = midpoint of (0,0,0) and (1,0,0)
    ok = ok && approx(nodes[4].coord(0), 0.5);
    ok = ok && approx(nodes[4].coord(1), 0.0);
    ok = ok && approx(nodes[4].coord(2), 0.0);

    // Midpoint (0,2) at index 5 = midpoint of (0,0,0) and (0,1,0)
    ok = ok && approx(nodes[5].coord(0), 0.0);
    ok = ok && approx(nodes[5].coord(1), 0.5);
    ok = ok && approx(nodes[5].coord(2), 0.0);

    // Midpoint (0,3) at index 6 = midpoint of (0,0,0) and (0,0,1)
    ok = ok && approx(nodes[6].coord(0), 0.0);
    ok = ok && approx(nodes[6].coord(1), 0.0);
    ok = ok && approx(nodes[6].coord(2), 0.5);

    // Midpoint (1,2) at index 7 = midpoint of (1,0,0) and (0,1,0)
    ok = ok && approx(nodes[7].coord(0), 0.5);
    ok = ok && approx(nodes[7].coord(1), 0.5);
    ok = ok && approx(nodes[7].coord(2), 0.0);

    // Midpoint (1,3) at index 8 = midpoint of (1,0,0) and (0,0,1)
    ok = ok && approx(nodes[8].coord(0), 0.5);
    ok = ok && approx(nodes[8].coord(1), 0.0);
    ok = ok && approx(nodes[8].coord(2), 0.5);

    // Midpoint (2,3) at index 9 = midpoint of (0,1,0) and (0,0,1)
    ok = ok && approx(nodes[9].coord(0), 0.0);
    ok = ok && approx(nodes[9].coord(1), 0.5);
    ok = ok && approx(nodes[9].coord(2), 0.5);

    report(__func__, ok);
}

void test_simplex_cell_face_topology() {
    using Cell = geometry::simplex::SimplexCell<3, 1>;
    bool ok = true;

    // A tetrahedron has 4 faces, each a triangle with 3 nodes
    ok = ok && (Cell::num_faces == 4);

    // Face f is opposite vertex f — check each face at compile time
    auto check_face = [&]<std::size_t f>() {
        constexpr auto fni = Cell::face_node_indices(f);
        ok = ok && (fni.size == 3);
        // Vertex f should NOT appear in the face
        for (std::size_t i = 0; i < fni.size; ++i) {
            ok = ok && (fni.indices[i] != f);
        }
    };
    check_face.template operator()<0>();
    check_face.template operator()<1>();
    check_face.template operator()<2>();
    check_face.template operator()<3>();

    report(__func__, ok);
}

void test_simplex_cell_face_topology_quadratic() {
    using Cell = geometry::simplex::SimplexCell<3, 2>;
    bool ok = true;

    ok = ok && (Cell::num_faces == 4);

    // Each face of a quadratic tet has 6 nodes (a quadratic triangle)
    auto check_face = [&]<std::size_t f>() {
        constexpr auto fni = Cell::face_node_indices(f);
        ok = ok && (fni.size == 6);
    };
    check_face.template operator()<0>();
    check_face.template operator()<1>();
    check_face.template operator()<2>();
    check_face.template operator()<3>();

    report(__func__, ok);
}

// =========================================================================
//  2. SimplexBasis — Partition of Unity & Kronecker Delta
// =========================================================================

void test_simplex_basis_partition_of_unity_tet4() {
    using Basis = geometry::simplex::SimplexBasis<3, 1>;
    Basis basis{};
    bool ok = true;

    // Test at several random-ish points inside the tet
    std::array<std::array<double, 3>, 5> test_points = {{
        {0.25, 0.25, 0.25},  // centroid
        {0.1,  0.1,  0.1},
        {0.5,  0.2,  0.1},
        {0.0,  0.0,  0.0},   // vertex 0
        {1.0,  0.0,  0.0}    // vertex 1
    }};

    for (const auto& pt : test_points) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            sum += basis.shape_function(i)(pt);
        }
        ok = ok && approx(sum, 1.0, 1e-12);
    }

    report(__func__, ok);
}

void test_simplex_basis_partition_of_unity_tet10() {
    using Basis = geometry::simplex::SimplexBasis<3, 2>;
    Basis basis{};
    bool ok = true;

    std::array<std::array<double, 3>, 4> test_points = {{
        {0.25, 0.25, 0.25},
        {0.1,  0.2,  0.3},
        {0.5,  0.0,  0.0},
        {0.0,  0.5,  0.0}
    }};

    for (const auto& pt : test_points) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 10; ++i) {
            sum += basis.shape_function(i)(pt);
        }
        ok = ok && approx(sum, 1.0, 1e-12);
    }

    report(__func__, ok);
}

void test_simplex_basis_kronecker_delta_tet4() {
    using Cell = geometry::simplex::SimplexCell<3, 1>;
    using Basis = geometry::simplex::SimplexBasis<3, 1>;
    Basis basis{};
    bool ok = true;

    constexpr auto nodes = Cell::reference_nodes;

    // N_i(x_j) = δ_ij  at reference node positions
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            double val = basis.shape_function(i)(nodes[j].coord());
            double expected = (i == j) ? 1.0 : 0.0;
            ok = ok && approx(val, expected, 1e-12);
        }
    }

    report(__func__, ok);
}

void test_simplex_basis_kronecker_delta_tet10() {
    using Cell = geometry::simplex::SimplexCell<3, 2>;
    using Basis = geometry::simplex::SimplexBasis<3, 2>;
    Basis basis{};
    bool ok = true;

    constexpr auto nodes = Cell::reference_nodes;

    for (std::size_t i = 0; i < 10; ++i) {
        for (std::size_t j = 0; j < 10; ++j) {
            double val = basis.shape_function(i)(nodes[j].coord());
            double expected = (i == j) ? 1.0 : 0.0;
            ok = ok && approx(val, expected, 1e-10);
        }
    }

    report(__func__, ok);
}

void test_simplex_basis_derivative_sum_zero() {
    // Sum of all shape function derivatives w.r.t. any direction = 0
    // (because ∂/∂ξ_j [∑ N_i] = 0)
    using Basis = geometry::simplex::SimplexBasis<3, 1>;
    Basis basis{};
    bool ok = true;

    std::array<double, 3> pt = {0.2, 0.3, 0.1};
    for (std::size_t j = 0; j < 3; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            sum += basis.shape_function_derivative(i, j)(pt);
        }
        ok = ok && approx(sum, 0.0, 1e-12);
    }

    report(__func__, ok);
}

void test_simplex_basis_derivative_sum_zero_quadratic() {
    using Basis = geometry::simplex::SimplexBasis<3, 2>;
    Basis basis{};
    bool ok = true;

    std::array<double, 3> pt = {0.15, 0.25, 0.35};
    for (std::size_t j = 0; j < 3; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 10; ++i) {
            sum += basis.shape_function_derivative(i, j)(pt);
        }
        ok = ok && approx(sum, 0.0, 1e-10);
    }

    report(__func__, ok);
}

// =========================================================================
//  2b. Triangle basis (2D simplex)
// =========================================================================

void test_simplex_basis_partition_unity_tri3() {
    using Basis = geometry::simplex::SimplexBasis<2, 1>;
    Basis basis{};
    bool ok = true;

    std::array<std::array<double, 2>, 4> test_points = {{
        {1.0/3.0, 1.0/3.0},
        {0.0, 0.0},
        {0.5, 0.0},
        {0.5, 0.5}
    }};

    for (const auto& pt : test_points) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 3; ++i)
            sum += basis.shape_function(i)(pt);
        ok = ok && approx(sum, 1.0, 1e-12);
    }

    report(__func__, ok);
}

void test_simplex_basis_kronecker_tri6() {
    using Cell = geometry::simplex::SimplexCell<2, 2>;
    using Basis = geometry::simplex::SimplexBasis<2, 2>;
    Basis basis{};
    bool ok = true;

    constexpr auto nodes = Cell::reference_nodes;

    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            double val = basis.shape_function(i)(nodes[j].coord());
            double expected = (i == j) ? 1.0 : 0.0;
            ok = ok && approx(val, expected, 1e-10);
        }
    }

    report(__func__, ok);
}

// =========================================================================
//  3. SimplexQuadrature — accuracy tests
// =========================================================================

void test_simplex_quadrature_tri_constant() {
    // ∫ 1 dA = 1/2  over reference triangle
    static constexpr auto rule = simplex_quadrature::make_simplex_rule_2d_3();
    bool ok = true;

    double integral = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        integral += rule.get_point_weight(i) * 1.0;
    }
    ok = ok && approx(integral, 0.5, 1e-14);

    report(__func__, ok);
}

void test_simplex_quadrature_tet_constant() {
    // ∫ 1 dV = 1/6  over reference tetrahedron
    static constexpr auto rule = simplex_quadrature::make_simplex_rule_3d_4();
    bool ok = true;

    double integral = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        integral += rule.get_point_weight(i) * 1.0;
    }
    ok = ok && approx(integral, 1.0 / 6.0, 1e-14);

    report(__func__, ok);
}

void test_simplex_quadrature_tri_linear() {
    // ∫ (ξ₁ + ξ₂) dA  over ref triangle = ∫₀¹ ∫₀^{1-x} (x+y) dy dx
    // = ∫₀¹ [x(1-x) + (1-x)²/2] dx = ∫₀¹ [x - x² + (1-2x+x²)/2] dx
    // = ∫₀¹ [x - x² + 1/2 - x + x²/2] dx = ∫₀¹ [1/2 - x²/2] dx
    // = 1/2 - 1/6 = 1/3
    static constexpr auto rule = simplex_quadrature::make_simplex_rule_2d_3();
    bool ok = true;

    double integral = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        const auto& pt = rule.get_point_coords(i);
        integral += rule.get_point_weight(i) * (pt[0] + pt[1]);
    }
    ok = ok && approx(integral, 1.0 / 3.0, 1e-14);

    report(__func__, ok);
}

void test_simplex_quadrature_tet_linear() {
    // ∫ (ξ₁ + ξ₂ + ξ₃) dV over ref tet
    // By symmetry, each ∫ ξ_i dV = 1/24, so total = 3/24 = 1/8
    static constexpr auto rule = simplex_quadrature::make_simplex_rule_3d_4();
    bool ok = true;

    double integral = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        const auto& pt = rule.get_point_coords(i);
        integral += rule.get_point_weight(i) * (pt[0] + pt[1] + pt[2]);
    }
    ok = ok && approx(integral, 1.0 / 8.0, 1e-14);

    report(__func__, ok);
}

void test_simplex_quadrature_tet_quadratic() {
    // ∫ ξ₁² dV  over ref tet = 1/60
    // (This tests degree ≥ 2 accuracy)
    static constexpr auto rule = simplex_quadrature::make_simplex_rule_3d_4();
    bool ok = true;

    double integral = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        const auto& pt = rule.get_point_coords(i);
        integral += rule.get_point_weight(i) * pt[0] * pt[0];
    }
    ok = ok && approx(integral, 1.0 / 60.0, 1e-14);

    report(__func__, ok);
}

void test_simplex_quadrature_1d() {
    // ∫₀¹ x² dx = 1/3  (1-point rule should NOT be exact, 2-point should)
    static constexpr auto rule2 = simplex_quadrature::make_simplex_rule_1d_2();
    bool ok = true;

    double integral = 0.0;
    for (std::size_t i = 0; i < rule2.num_points; ++i) {
        const auto& pt = rule2.get_point_coords(i);
        integral += rule2.get_point_weight(i) * pt[0] * pt[0];
    }
    ok = ok && approx(integral, 1.0 / 3.0, 1e-14);

    report(__func__, ok);
}

void test_simplex_quadrature_tri_degree5() {
    // ∫ ξ₁² ξ₂² dA  over ref triangle = 1/180
    // (requires degree ≥ 4 accuracy)
    static constexpr auto rule = simplex_quadrature::make_simplex_rule_2d_7();
    bool ok = true;

    double integral = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        const auto& pt = rule.get_point_coords(i);
        integral += rule.get_point_weight(i) * pt[0] * pt[0] * pt[1] * pt[1];
    }
    ok = ok && approx(integral, 1.0 / 180.0, 1e-12);

    report(__func__, ok);
}

// ─── 3b. Stroud conical product quadrature ──────────────────────────────

void test_conical_product_weight_sum_1d() {
    // Weights must sum to 1/1! = 1  (volume of [0,1] simplex)
    bool ok = true;
    for (int n = 1; n <= 6; ++n) {
        auto rule = simplex_quadrature::stroud_conical_product<1>(n);
        double sum = 0.0;
        for (std::size_t i = 0; i < rule.num_points; ++i) sum += rule.weights[i];
        ok = ok && approx(sum, 1.0, 1e-13);
    }
    report(__func__, ok);
}

void test_conical_product_weight_sum_2d() {
    // Weights must sum to 1/2! = 0.5  (area of reference triangle)
    bool ok = true;
    for (int n = 1; n <= 6; ++n) {
        auto rule = simplex_quadrature::stroud_conical_product<2>(n);
        double sum = 0.0;
        for (std::size_t i = 0; i < rule.num_points; ++i) sum += rule.weights[i];
        ok = ok && approx(sum, 0.5, 1e-13);
    }
    report(__func__, ok);
}

void test_conical_product_weight_sum_3d() {
    // Weights must sum to 1/3! = 1/6  (volume of reference tetrahedron)
    bool ok = true;
    for (int n = 1; n <= 6; ++n) {
        auto rule = simplex_quadrature::stroud_conical_product<3>(n);
        double sum = 0.0;
        for (std::size_t i = 0; i < rule.num_points; ++i) sum += rule.weights[i];
        ok = ok && approx(sum, 1.0 / 6.0, 1e-13);
    }
    report(__func__, ok);
}

void test_conical_product_all_weights_positive() {
    // ALL weights must be strictly positive for every n and every dimension
    bool ok = true;
    for (int n = 1; n <= 6; ++n) {
        { auto r = simplex_quadrature::stroud_conical_product<1>(n);
          for (auto w : r.weights) ok = ok && (w > 0.0); }
        { auto r = simplex_quadrature::stroud_conical_product<2>(n);
          for (auto w : r.weights) ok = ok && (w > 0.0); }
        { auto r = simplex_quadrature::stroud_conical_product<3>(n);
          for (auto w : r.weights) ok = ok && (w > 0.0); }
    }
    report(__func__, ok);
}

void test_conical_product_points_inside_simplex() {
    // All points must lie inside the reference simplex: ξ_i ≥ 0, Σξ_i ≤ 1
    bool ok = true;
    for (int n = 1; n <= 5; ++n) {
        auto rule = simplex_quadrature::stroud_conical_product<3>(n);
        for (std::size_t i = 0; i < rule.num_points; ++i) {
            double s = 0.0;
            for (std::size_t d = 0; d < 3; ++d) {
                ok = ok && (rule.points[i][d] >= -1e-15);
                s += rule.points[i][d];
            }
            ok = ok && (s <= 1.0 + 1e-14);
        }
    }
    report(__func__, ok);
}

void test_conical_product_degree3_tet() {
    // n=2 → degree 3 exactness.  Test all cubic monomials over tet:
    //   ∫ ξ₁³ dV = 1/120,  ∫ ξ₁²ξ₂ dV = 1/360,  ∫ ξ₁ξ₂ξ₃ dV = 1/720
    auto rule = simplex_quadrature::stroud_conical_product<3>(2);
    bool ok = true;

    // ∫ ξ₁³ dV on ref tet = B(4,1,1,1)·3! = 3!·0!·0!·0!/6! = 6/720 = 1/120
    double I1 = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        double x = rule.points[i][0];
        I1 += rule.weights[i] * x * x * x;
    }
    ok = ok && approx(I1, 1.0 / 120.0, 1e-13);

    // ∫ ξ₁²ξ₂ dV = 2!·1!·0!·0!/6! = 2/720 = 1/360
    double I2 = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        double x = rule.points[i][0], y = rule.points[i][1];
        I2 += rule.weights[i] * x * x * y;
    }
    ok = ok && approx(I2, 1.0 / 360.0, 1e-13);

    // ∫ ξ₁ξ₂ξ₃ dV = 1!·1!·1!·0!/6! = 1/720
    double I3 = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        auto& p = rule.points[i];
        I3 += rule.weights[i] * p[0] * p[1] * p[2];
    }
    ok = ok && approx(I3, 1.0 / 720.0, 1e-13);

    report(__func__, ok);
}

void test_conical_product_degree5_tet() {
    // n=3 → degree 5 exactness.  Test quintic monomial:
    //   ∫ ξ₁²ξ₂²ξ₃ dV = 2!·2!·1!·0!/8! = 4/40320 = 1/10080
    auto rule = simplex_quadrature::stroud_conical_product<3>(3);
    bool ok = true;

    double I = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        auto& p = rule.points[i];
        I += rule.weights[i] * p[0]*p[0] * p[1]*p[1] * p[2];
    }
    ok = ok && approx(I, 1.0 / 10080.0, 1e-12);

    report(__func__, ok);
}

void test_conical_product_degree7_tet() {
    // n=4 → degree 7.  Test ∫ ξ₁³ξ₂²ξ₃² dV = 3!·2!·2!·0!/10! = 24/3628800 = 1/151200
    auto rule = simplex_quadrature::stroud_conical_product<3>(4);
    bool ok = true;

    double I = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        auto& p = rule.points[i];
        I += rule.weights[i] * p[0]*p[0]*p[0] * p[1]*p[1] * p[2]*p[2];
    }
    ok = ok && approx(I, 1.0 / 151200.0, 1e-11);

    report(__func__, ok);
}

void test_conical_product_degree5_tri() {
    // n=3, D=2 → degree 5. Test ∫ ξ₁²ξ₂² dA = 2!·2!·0!/6! = 4/720 = 1/180
    auto rule = simplex_quadrature::stroud_conical_product<2>(3);
    bool ok = true;

    double I = 0.0;
    for (std::size_t i = 0; i < rule.num_points; ++i) {
        I += rule.weights[i] * rule.points[i][0]*rule.points[i][0]
                              * rule.points[i][1]*rule.points[i][1];
    }
    ok = ok && approx(I, 1.0 / 180.0, 1e-13);

    report(__func__, ok);
}

void test_conical_product_integrator_tet() {
    // Test the ConicalProductIntegrator template class.
    // NPerDir=2, TopDim=3 → 8 points, degree 3
    simplex_quadrature::ConicalProductIntegrator<3, 2> integrator;
    bool ok = true;

    // Check num_integration_points
    ok = ok && (integrator.num_integration_points == 8);

    // Weight sum = 1/6
    double wsum = 0.0;
    for (std::size_t i = 0; i < integrator.num_integration_points; ++i) {
        wsum += integrator.weight(i);
    }
    ok = ok && approx(wsum, 1.0 / 6.0, 1e-14);

    // Use operator() to integrate a constant function: ∑ w_i · 1 = 1/6
    double result = integrator([](std::span<const double>) { return 1.0; });
    ok = ok && approx(result, 1.0 / 6.0, 1e-14);

    // Integrate ξ₁² via operator(): should give 1/60
    double result2 = integrator([](std::span<const double> pt) {
        return pt[0] * pt[0];
    });
    ok = ok && approx(result2, 1.0 / 60.0, 1e-14);

    report(__func__, ok);
}

void test_conical_product_integrator_tri() {
    // NPerDir=3, TopDim=2 → 9 points, degree 5
    simplex_quadrature::ConicalProductIntegrator<2, 3> integrator;
    bool ok = true;

    ok = ok && (integrator.num_integration_points == 9);

    // ∫ ξ₁²ξ₂² dA = 1/180
    double result = integrator([](std::span<const double> pt) {
        return pt[0]*pt[0] * pt[1]*pt[1];
    });
    ok = ok && approx(result, 1.0 / 180.0, 1e-13);

    report(__func__, ok);
}


// =========================================================================
//  4. SimplexElement — isoparametric mapping & Jacobian
// =========================================================================

void test_simplex_element_tet4_unit_jacobian() {
    // Physical tetrahedron = reference tetrahedron
    // J should be the 3x3 identity, |J| = 1
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 1.0, 0.0};
    Node<3> n3{3, 0.0, 0.0, 1.0};

    std::optional<std::array<Node<3>*, 4>> nodes{
        std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}};

    SimplexElement<3, 3, 1> elem(nodes);
    bool ok = true;

    std::array<double, 3> center = {0.25, 0.25, 0.25};
    auto J = elem.evaluate_jacobian(center);

    // J should be identity for the reference tet
    ok = ok && approx(J(0, 0), 1.0);
    ok = ok && approx(J(1, 1), 1.0);
    ok = ok && approx(J(2, 2), 1.0);
    ok = ok && approx(J(0, 1), 0.0);
    ok = ok && approx(J(0, 2), 0.0);
    ok = ok && approx(J(1, 0), 0.0);
    ok = ok && approx(J(1, 2), 0.0);
    ok = ok && approx(J(2, 0), 0.0);
    ok = ok && approx(J(2, 1), 0.0);

    // det(J) = 1
    ok = ok && approx(J.determinant(), 1.0, 1e-12);

    report(__func__, ok);
}

void test_simplex_element_tet4_scaled_jacobian() {
    // Scale the reference tet by factor 2 in each direction
    // Vertices: (0,0,0), (2,0,0), (0,2,0), (0,0,2)
    // J = 2I, det(J) = 8
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 2.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 2.0, 0.0};
    Node<3> n3{3, 0.0, 0.0, 2.0};

    std::optional<std::array<Node<3>*, 4>> nodes{
        std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}};

    SimplexElement<3, 3, 1> elem(nodes);
    bool ok = true;

    std::array<double, 3> center = {0.25, 0.25, 0.25};
    auto J = elem.evaluate_jacobian(center);

    ok = ok && approx(J(0, 0), 2.0);
    ok = ok && approx(J(1, 1), 2.0);
    ok = ok && approx(J(2, 2), 2.0);
    ok = ok && approx(J.determinant(), 8.0, 1e-12);

    report(__func__, ok);
}

void test_simplex_element_tet4_map_local_point() {
    // Translated tet: (1,1,1), (2,1,1), (1,2,1), (1,1,2)
    Node<3> n0{0, 1.0, 1.0, 1.0};
    Node<3> n1{1, 2.0, 1.0, 1.0};
    Node<3> n2{2, 1.0, 2.0, 1.0};
    Node<3> n3{3, 1.0, 1.0, 2.0};

    std::optional<std::array<Node<3>*, 4>> nodes{
        std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}};

    SimplexElement<3, 3, 1> elem(nodes);
    bool ok = true;

    // Origin ξ=(0,0,0) → vertex 0 = (1,1,1)
    auto x0 = elem.map_local_point({0.0, 0.0, 0.0});
    ok = ok && approx(x0[0], 1.0) && approx(x0[1], 1.0) && approx(x0[2], 1.0);

    // ξ=(1,0,0) → vertex 1 = (2,1,1)
    auto x1 = elem.map_local_point({1.0, 0.0, 0.0});
    ok = ok && approx(x1[0], 2.0) && approx(x1[1], 1.0) && approx(x1[2], 1.0);

    // Centroid ξ=(1/4,1/4,1/4) → (1.25, 1.25, 1.25)
    auto xc = elem.map_local_point({0.25, 0.25, 0.25});
    ok = ok && approx(xc[0], 1.25) && approx(xc[1], 1.25) && approx(xc[2], 1.25);

    report(__func__, ok);
}

// =========================================================================
//  5. ElementGeometry with SimplexElement — type erasure & integration
// =========================================================================

void test_simplex_element_geometry_tet4_volume() {
    // Reference tet has volume 1/6
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 1.0, 0.0};
    Node<3> n3{3, 0.0, 0.0, 1.0};

    std::optional<std::array<Node<3>*, 4>> nodes{
        std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}};

    SimplexElement<3, 3, 1> elem(nodes);
    SimplexIntegrator<3, 1> integrator;

    ElementGeometry<3> geom(elem, integrator);

    bool ok = true;
    ok = ok && (geom.topological_dimension() == 3);
    ok = ok && (geom.num_nodes() == 4);
    ok = ok && (geom.num_integration_points() == 4); // 4-point rule

    // Integrate constant 1 → volume = 1/6
    double vol = geom.integrate([](std::span<const double>) { return 1.0; });
    ok = ok && approx(vol, 1.0 / 6.0, 1e-12);

    report(__func__, ok);
}

void test_simplex_element_geometry_scaled_tet4_volume() {
    // Tet with edge length 2, vol = 8/6 = 4/3
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 2.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 2.0, 0.0};
    Node<3> n3{3, 0.0, 0.0, 2.0};

    std::optional<std::array<Node<3>*, 4>> nodes{
        std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}};

    SimplexElement<3, 3, 1> elem(nodes);
    SimplexIntegrator<3, 1> integrator;

    ElementGeometry<3> geom(elem, integrator);

    double vol = geom.integrate([](std::span<const double>) { return 1.0; });
    report(__func__, approx(vol, 4.0 / 3.0, 1e-12));
}

void test_simplex_element_geometry_tri3_area() {
    // Triangle in 2D: (0,0), (1,0), (0,1) → area = 0.5
    Node<2> n0{0, 0.0, 0.0};
    Node<2> n1{1, 1.0, 0.0};
    Node<2> n2{2, 0.0, 1.0};

    std::optional<std::array<Node<2>*, 3>> nodes{
        std::array<Node<2>*, 3>{&n0, &n1, &n2}};

    SimplexElement<2, 2, 1> elem(nodes);
    SimplexIntegrator<2, 1> integrator;

    ElementGeometry<2> geom(elem, integrator);

    bool ok = true;
    ok = ok && (geom.topological_dimension() == 2);
    ok = ok && (geom.num_nodes() == 3);

    double area = geom.integrate([](std::span<const double>) { return 1.0; });
    ok = ok && approx(area, 0.5, 1e-12);

    report(__func__, ok);
}

void test_simplex_element_geometry_tri3_in_3d_area() {
    // Triangle embedded in 3D: (0,0,0), (1,0,0), (0,1,0) → area = 0.5
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 1.0, 0.0};

    std::optional<std::array<Node<3>*, 3>> nodes{
        std::array<Node<3>*, 3>{&n0, &n1, &n2}};

    SimplexElement<3, 2, 1> elem(nodes);
    SimplexIntegrator<2, 1> integrator;

    ElementGeometry<3> geom(elem, integrator);

    bool ok = true;
    ok = ok && (geom.topological_dimension() == 2);

    double area = geom.integrate([](std::span<const double>) { return 1.0; });
    ok = ok && approx(area, 0.5, 1e-12);

    report(__func__, ok);
}

void test_simplex_element_geometry_inclined_tri_area() {
    // Equilateral-ish triangle in 3D: area = ‖(v1-v0)×(v2-v0)‖/2
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 1.0, 1.0};

    std::optional<std::array<Node<3>*, 3>> nodes{
        std::array<Node<3>*, 3>{&n0, &n1, &n2}};

    SimplexElement<3, 2, 1> elem(nodes);
    SimplexIntegrator<2, 1> integrator;

    ElementGeometry<3> geom(elem, integrator);

    // (1,0,0)×(0,1,1) = (-0, -1, 1)  → ‖…‖ = √2  → area = √2/2
    double expected = std::sqrt(2.0) / 2.0;
    double area = geom.integrate([](std::span<const double>) { return 1.0; });
    report(__func__, approx(area, expected, 1e-12));
}

void test_simplex_element_geometry_tet10_volume() {
    // Quadratic tet with straight edges = same as linear tet
    // Volume of reference tet = 1/6
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 1.0, 0.0};
    Node<3> n3{3, 0.0, 0.0, 1.0};
    // Edge midpoints
    Node<3> n4{4, 0.5, 0.0, 0.0};  // (0,1)
    Node<3> n5{5, 0.0, 0.5, 0.0};  // (0,2)
    Node<3> n6{6, 0.0, 0.0, 0.5};  // (0,3)
    Node<3> n7{7, 0.5, 0.5, 0.0};  // (1,2)
    Node<3> n8{8, 0.5, 0.0, 0.5};  // (1,3)
    Node<3> n9{9, 0.0, 0.5, 0.5};  // (2,3)

    std::optional<std::array<Node<3>*, 10>> nodes{
        std::array<Node<3>*, 10>{&n0, &n1, &n2, &n3, &n4, &n5, &n6, &n7, &n8, &n9}};

    SimplexElement<3, 3, 2> elem(nodes);
    SimplexIntegrator<3, 2> integrator;

    ElementGeometry<3> geom(elem, integrator);

    bool ok = true;
    ok = ok && (geom.topological_dimension() == 3);
    ok = ok && (geom.num_nodes() == 10);

    double vol = geom.integrate([](std::span<const double>) { return 1.0; });
    ok = ok && approx(vol, 1.0 / 6.0, 1e-10);

    report(__func__, ok);
}

void test_simplex_element_geometry_linear_field_tet4() {
    // ∫ (x + y + z) dV over reference tet = 1/8
    // (since the tet is identity-mapped, x = ξ₁, y = ξ₂, z = ξ₃)
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 1.0, 0.0};
    Node<3> n3{3, 0.0, 0.0, 1.0};

    std::optional<std::array<Node<3>*, 4>> nodes{
        std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}};

    SimplexElement<3, 3, 1> elem(nodes);
    SimplexIntegrator<3, 1> integrator;

    auto geom_ptr = std::make_unique<ElementGeometry<3>>(elem, integrator);
    auto& geom = *geom_ptr;

    double integral = geom.integrate([&geom](std::span<const double> xi) {
        auto X = geom.map_local_point(xi);
        return X[0] + X[1] + X[2];
    });

    report(__func__, approx(integral, 1.0 / 8.0, 1e-12));
}

void test_simplex_element_geometry_matrix_integration() {
    // Integrate identity matrix over the reference tet → I · (vol = 1/6)
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 1.0, 0.0};
    Node<3> n3{3, 0.0, 0.0, 1.0};

    std::optional<std::array<Node<3>*, 4>> nodes{
        std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}};

    SimplexElement<3, 3, 1> elem(nodes);
    SimplexIntegrator<3, 1> integrator;

    ElementGeometry<3> geom(elem, integrator);

    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d M = geom.integrate(
        [&](std::span<const double>) -> Eigen::Matrix3d { return I; });

    bool ok = true;
    ok = ok && approx(M(0, 0), 1.0 / 6.0, 1e-12);
    ok = ok && approx(M(1, 1), 1.0 / 6.0, 1e-12);
    ok = ok && approx(M(2, 2), 1.0 / 6.0, 1e-12);
    ok = ok && approx(M(0, 1), 0.0, 1e-12);

    report(__func__, ok);
}

// =========================================================================
//  6. VTK cell type dispatch
// =========================================================================

void test_vtk_cell_type_simplex() {
    using namespace fall_n::vtk;
    bool ok = true;

    ok = ok && (cell_type_from(2, 3)  == VTK_TRIANGLE);
    ok = ok && (cell_type_from(2, 6)  == VTK_QUADRATIC_TRIANGLE);
    ok = ok && (cell_type_from(3, 4)  == VTK_TETRA);
    ok = ok && (cell_type_from(3, 10) == VTK_QUADRATIC_TETRA);

    // Existing hex types should still work
    ok = ok && (cell_type_from(3, 8)  == VTK_HEXAHEDRON);
    ok = ok && (cell_type_from(3, 27) == VTK_TRIQUADRATIC_HEXAHEDRON);

    report(__func__, ok);
}

void test_vtk_hex27_ordering_matches_triquadratic_hex() {
    using namespace fall_n::vtk;

    vtkIdType perm[27];
    node_ordering_into(3, 27, perm);

    vtkNew<vtkTriQuadraticHexahedron> vtk_hex27;
    const double* vtk_coords = vtk_hex27->GetParametricCoords();

    constexpr auto ref_nodes = geometry::cell::cell_nodes<3, 3, 3>();

    bool ok = true;
    for (std::size_t vtk_id = 0; vtk_id < 27; ++vtk_id) {
        const auto& local = ref_nodes[perm[vtk_id]];
        const double x = 0.5 * (local.coord(0) + 1.0);
        const double y = 0.5 * (local.coord(1) + 1.0);
        const double z = 0.5 * (local.coord(2) + 1.0);

        ok = ok && approx(x, vtk_coords[3 * vtk_id + 0], 1e-12);
        ok = ok && approx(y, vtk_coords[3 * vtk_id + 1], 1e-12);
        ok = ok && approx(z, vtk_coords[3 * vtk_id + 2], 1e-12);
    }

    report(__func__, ok);
}

void test_tet10_positive_quadrature_still_has_signed_lumped_weights() {
    using Tet10 = SimplexElement<3, 3, 2>;
    using HMS4 = SimplexIntegrator<3, 2>;
    using Stroud27 = simplex_quadrature::ConicalProductIntegrator<3, 3>;
    using namespace fall_n::vtk;

    constexpr auto ref_nodes = geometry::simplex::SimplexCell<3, 2>::reference_nodes;

    std::vector<Node<3>> nodes;
    nodes.reserve(10);
    std::array<Node<3>*, 10> node_ptrs{};
    for (std::size_t i = 0; i < node_ptrs.size(); ++i) {
        nodes.emplace_back(
            static_cast<PetscInt>(i),
            ref_nodes[i].coord(0),
            ref_nodes[i].coord(1),
            ref_nodes[i].coord(2));
        node_ptrs[i] = &nodes.back();
    }

    Tet10 elem(std::optional{node_ptrs});

    bool ok = true;

    {
        HMS4 integrator;
        ElementGeometry<3> geom(elem, integrator);

        for (std::size_t g = 0; g < geom.num_integration_points(); ++g) {
            ok = ok && (geom.weight(g) > 0.0);
        }

        std::array<double, 64> lump{};
        lumped_projection_weights_into(geom, lump.data());

        for (std::size_t i = 0; i < 4; ++i) {
            ok = ok && approx(lump[i], -1.0 / 120.0, 1e-12);
        }
        for (std::size_t i = 4; i < 10; ++i) {
            ok = ok && approx(lump[i], 1.0 / 30.0, 1e-12);
        }
        ok = ok && !has_strictly_positive_lumped_projection_weights(geom);
    }

    {
        Stroud27 integrator;
        ElementGeometry<3> geom(elem, integrator);

        for (std::size_t g = 0; g < geom.num_integration_points(); ++g) {
            ok = ok && (geom.weight(g) > 0.0);
        }

        std::array<double, 64> lump{};
        lumped_projection_weights_into(geom, lump.data());

        for (std::size_t i = 0; i < 4; ++i) {
            ok = ok && (lump[i] < 0.0);
        }
        for (std::size_t i = 4; i < 10; ++i) {
            ok = ok && (lump[i] > 0.0);
        }
        ok = ok && !has_strictly_positive_lumped_projection_weights(geom);
    }

    report(__func__, ok);
}

void test_hex27_lumped_projection_weights_are_positive() {
    using Hex27 = LagrangeElement<3, 3, 3, 3>;
    using Integrator = GaussLegendreCellIntegrator<3, 3, 3>;
    using namespace fall_n::vtk;

    constexpr auto ref_nodes = geometry::cell::cell_nodes<3, 3, 3>();

    std::vector<Node<3>> nodes;
    nodes.reserve(27);
    std::array<Node<3>*, 27> node_ptrs{};
    for (std::size_t i = 0; i < node_ptrs.size(); ++i) {
        nodes.emplace_back(
            static_cast<PetscInt>(i),
            ref_nodes[i].coord(0),
            ref_nodes[i].coord(1),
            ref_nodes[i].coord(2));
        node_ptrs[i] = &nodes.back();
    }

    Hex27 elem(std::optional{node_ptrs});
    Integrator integrator;
    ElementGeometry<3> geom(elem, integrator);

    std::array<double, 64> lump{};
    lumped_projection_weights_into(geom, lump.data());

    bool ok = has_strictly_positive_lumped_projection_weights(geom);
    for (std::size_t i = 0; i < geom.num_nodes(); ++i) {
        ok = ok && (lump[i] > 0.0);
    }

    report(__func__, ok);
}

// =========================================================================
//  7. Face geometry creation (sub-entity topology)
// =========================================================================

void test_simplex_face_geometry_tet4() {
    // Create a tet4 element and verify that faces are created correctly
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 1.0, 0.0};
    Node<3> n3{3, 0.0, 0.0, 1.0};

    std::optional<std::array<Node<3>*, 4>> nodes{
        std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}};

    SimplexElement<3, 3, 1> elem(nodes);
    SimplexIntegrator<3, 1> integrator;

    ElementGeometry<3> geom(elem, integrator);

    bool ok = true;

    // A tet has 4 faces
    ok = ok && (geom.num_faces() == 4);

    // Each face has 3 nodes
    for (std::size_t f = 0; f < 4; ++f) {
        ok = ok && (geom.face_num_nodes(f) == 3);
    }

    // Face 0 (opposite vertex 0) should contain nodes 1, 2, 3
    auto face0_nodes = geom.face_node_indices(0);
    ok = ok && (face0_nodes.size() == 3);
    ok = ok && (std::find(face0_nodes.begin(), face0_nodes.end(), 1) != face0_nodes.end());
    ok = ok && (std::find(face0_nodes.begin(), face0_nodes.end(), 2) != face0_nodes.end());
    ok = ok && (std::find(face0_nodes.begin(), face0_nodes.end(), 3) != face0_nodes.end());

    report(__func__, ok);
}

void test_simplex_make_face_geometry_tet4() {
    // Actually create a face geometry and integrate over it
    Node<3> n0{0, 0.0, 0.0, 0.0};
    Node<3> n1{1, 1.0, 0.0, 0.0};
    Node<3> n2{2, 0.0, 1.0, 0.0};
    Node<3> n3{3, 0.0, 0.0, 1.0};

    // Build global node ID array as PetscInt
    std::array<PetscInt, 4> plex_ids = {0, 1, 2, 3};

    SimplexElement<3, 3, 1> elem(std::size_t(0), std::span<PetscInt>(plex_ids));
    SimplexIntegrator<3, 1> integrator;

    ElementGeometry<3> geom(elem, integrator);

    // Get face 0 (opposite vertex 0): nodes 1, 2, 3
    auto face_indices = geom.face_node_indices(0);
    std::vector<PetscInt> face_ids;
    for (auto idx : face_indices) {
        face_ids.push_back(static_cast<PetscInt>(idx));
    }

    // Create face geometry
    auto face_geom = geom.make_face_geometry(0, 100, std::span<PetscInt>(face_ids));

    bool ok = true;
    ok = ok && (face_geom.topological_dimension() == 2);
    ok = ok && (face_geom.num_nodes() == 3);

    // Bind physical nodes to the face
    face_geom.bind_node(0, &n1);
    face_geom.bind_node(1, &n2);
    face_geom.bind_node(2, &n3);

    // Face (1,0,0)-(0,1,0)-(0,0,1) has area = √3/2
    double expected_area = std::sqrt(3.0) / 2.0;
    double area = face_geom.integrate([](std::span<const double>) { return 1.0; });
    ok = ok && approx(area, expected_area, 1e-10);

    report(__func__, ok);
}

// =========================================================================
//  8. General tet volume for arbitrary tetrahedra
// =========================================================================

void test_simplex_general_tet_volume() {
    // Arbitrary tetrahedron: v0=(1,0,0), v1=(0,2,0), v2=(0,0,3), v3=(0,0,0)
    // Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
    //        = |det([-1,2,0], [-1,0,3], [-1,0,0])| / 6
    //        = |(-1)(0·0 - 3·0) - 2((-1)·0 - 3·(-1)) + 0| / 6
    //        = |0 - 2(3) + 0| / 6 = 6/6 = 1
    Node<3> n0{0, 1.0, 0.0, 0.0};
    Node<3> n1{1, 0.0, 2.0, 0.0};
    Node<3> n2{2, 0.0, 0.0, 3.0};
    Node<3> n3{3, 0.0, 0.0, 0.0};

    std::optional<std::array<Node<3>*, 4>> nodes{
        std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}};

    SimplexElement<3, 3, 1> elem(nodes);
    SimplexIntegrator<3, 1> integrator;

    ElementGeometry<3> geom(elem, integrator);

    double vol = geom.integrate([](std::span<const double>) { return 1.0; });
    report(__func__, approx(vol, 1.0, 1e-12));
}

// =========================================================================
//  9. SimplexIntegrator operator() (raw quadrature, no |J|)
// =========================================================================

void test_simplex_integrator_operator() {
    // ∑ w_i · f(ξ_i) for f = constant 1 should give vol(ref tet) = 1/6
    SimplexIntegrator<3, 1> integ;
    bool ok = true;

    double result = integ([](std::span<const double>) { return 1.0; });
    ok = ok && approx(result, 1.0 / 6.0, 1e-12);

    report(__func__, ok);
}

void test_simplex_integrator_operator_tri() {
    // ∑ w_i · f(ξ_i) for f = 1 should give area(ref tri) = 1/2
    SimplexIntegrator<2, 1> integ;
    double result = integ([](std::span<const double>) { return 1.0; });
    report(__func__, approx(result, 0.5, 1e-12));
}

// =========================================================================
//  10. Concept tests
// =========================================================================

void test_simplex_concept_detection() {
    bool ok = true;

    ok = ok && is_SimplexElement<SimplexElement<3, 3, 1>>;
    ok = ok && is_SimplexElement<SimplexElement<3, 3, 2>>;
    ok = ok && is_SimplexElement<SimplexElement<3, 2, 1>>;
    ok = ok && is_SimplexElement<SimplexElement<2, 2, 1>>;

    // LagrangeElement should NOT satisfy is_SimplexElement
    ok = ok && !is_SimplexElement<LagrangeElement<3, 2, 2, 2>>;

    // SimplexElement should NOT satisfy is_LagrangeElement
    ok = ok && !is_LagrangeElement<SimplexElement<3, 3, 1>>;

    report(__func__, ok);
}

} // namespace

int main() {
    // 1. Cell & node layout
    test_simplex_cell_num_nodes();
    test_simplex_cell_reference_nodes_tet4();
    test_simplex_cell_reference_nodes_tet10();
    test_simplex_cell_face_topology();
    test_simplex_cell_face_topology_quadratic();

    // 2. Basis functions
    test_simplex_basis_partition_of_unity_tet4();
    test_simplex_basis_partition_of_unity_tet10();
    test_simplex_basis_kronecker_delta_tet4();
    test_simplex_basis_kronecker_delta_tet10();
    test_simplex_basis_derivative_sum_zero();
    test_simplex_basis_derivative_sum_zero_quadratic();
    test_simplex_basis_partition_unity_tri3();
    test_simplex_basis_kronecker_tri6();

    // 3. Quadrature accuracy
    test_simplex_quadrature_tri_constant();
    test_simplex_quadrature_tet_constant();
    test_simplex_quadrature_tri_linear();
    test_simplex_quadrature_tet_linear();
    test_simplex_quadrature_tet_quadratic();
    test_simplex_quadrature_1d();
    test_simplex_quadrature_tri_degree5();

    // 3b. Stroud conical product
    test_conical_product_weight_sum_1d();
    test_conical_product_weight_sum_2d();
    test_conical_product_weight_sum_3d();
    test_conical_product_all_weights_positive();
    test_conical_product_points_inside_simplex();
    test_conical_product_degree3_tet();
    test_conical_product_degree5_tet();
    test_conical_product_degree7_tet();
    test_conical_product_degree5_tri();
    test_conical_product_integrator_tet();
    test_conical_product_integrator_tri();

    // 4. Isoparametric mapping
    test_simplex_element_tet4_unit_jacobian();
    test_simplex_element_tet4_scaled_jacobian();
    test_simplex_element_tet4_map_local_point();

    // 5. ElementGeometry integration
    test_simplex_element_geometry_tet4_volume();
    test_simplex_element_geometry_scaled_tet4_volume();
    test_simplex_element_geometry_tri3_area();
    test_simplex_element_geometry_tri3_in_3d_area();
    test_simplex_element_geometry_inclined_tri_area();
    test_simplex_element_geometry_tet10_volume();
    test_simplex_element_geometry_linear_field_tet4();
    test_simplex_element_geometry_matrix_integration();

    // 6. VTK dispatch
    test_vtk_cell_type_simplex();
    test_vtk_hex27_ordering_matches_triquadratic_hex();
    test_tet10_positive_quadrature_still_has_signed_lumped_weights();
    test_hex27_lumped_projection_weights_are_positive();

    // 7. Face geometry
    test_simplex_face_geometry_tet4();
    test_simplex_make_face_geometry_tet4();

    // 8. General tet
    test_simplex_general_tet_volume();

    // 9. Raw integrator
    test_simplex_integrator_operator();
    test_simplex_integrator_operator_tri();

    // 10. Concepts
    test_simplex_concept_detection();

    std::cout << "\n=== Simplex Element Family Tests ===\n";
    std::cout << "=== " << passed << " PASSED, " << failed << " FAILED ===\n";

    return failed ? 1 : 0;
}
