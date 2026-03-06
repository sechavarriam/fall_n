// tests/test_point.cpp — Unit tests for geometry::Point<Dim>
//
// Compile: (added as fall_n_point_test in CMakeLists.txt)
// Run:     ./build/fall_n_point_test

#include <cassert>
#include <cmath>
#include <array>
#include <iostream>
#include <type_traits>

#include "../src/geometry/Point.hh"

// ---------- helpers ---------------------------------------------------------
static bool approx(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) < tol;
}

// ===========================================================================
// 1. Default construction — zero-initialized
// ===========================================================================
static void test_default_construction() {
    constexpr geometry::Point<3> p{};
    static_assert(p.coord(0) == 0.0);
    static_assert(p.coord(1) == 0.0);
    static_assert(p.coord(2) == 0.0);
    std::cout << "[PASS] test_default_construction\n";
}

// ===========================================================================
// 2. Variadic constructor
// ===========================================================================
static void test_variadic_constructor() {
    constexpr geometry::Point<3> p{1.0, 2.0, 3.0};
    static_assert(p.coord(0) == 1.0);
    static_assert(p.coord(1) == 2.0);
    static_assert(p.coord(2) == 3.0);
    std::cout << "[PASS] test_variadic_constructor\n";
}

// ===========================================================================
// 3. Array constructor
// ===========================================================================
static void test_array_constructor() {
    constexpr std::array<double, 2> arr{4.0, 5.0};
    constexpr geometry::Point<2> p{arr};
    static_assert(p.coord(0) == 4.0);
    static_assert(p.coord(1) == 5.0);
    std::cout << "[PASS] test_array_constructor\n";
}

// ===========================================================================
// 4. Coordinate mutation (set_coord)
// ===========================================================================
static void test_set_coord() {
    geometry::Point<3> p{};
    p.set_coord(0, 10.0);
    p.set_coord(1, 20.0);
    p.set_coord(2, 30.0);
    assert(approx(p.coord(0), 10.0));
    assert(approx(p.coord(1), 20.0));
    assert(approx(p.coord(2), 30.0));

    // set_coord with array
    p.set_coord(std::array<double, 3>{7.0, 8.0, 9.0});
    assert(approx(p.coord(0), 7.0));
    assert(approx(p.coord(1), 8.0));
    assert(approx(p.coord(2), 9.0));

    std::cout << "[PASS] test_set_coord\n";
}

// ===========================================================================
// 5. coord_ref returns const reference to internal array
// ===========================================================================
static void test_coord_ref() {
    geometry::Point<2> p{3.0, 4.0};
    const auto& ref = p.coord_ref();
    assert(&ref == &p.coord_ref());  // same object
    assert(approx(ref[0], 3.0));
    assert(approx(ref[1], 4.0));
    std::cout << "[PASS] test_coord_ref\n";
}

// ===========================================================================
// 6. Static dim member
// ===========================================================================
static void test_dim_member() {
    static_assert(geometry::Point<1>::dim == 1);
    static_assert(geometry::Point<2>::dim == 2);
    static_assert(geometry::Point<3>::dim == 3);
    std::cout << "[PASS] test_dim_member\n";
}

// ===========================================================================
// 7. PointT concept: Point satisfies it
// ===========================================================================
static void test_concept_point() {
    static_assert(geometry::PointT<geometry::Point<1>>);
    static_assert(geometry::PointT<geometry::Point<2>>);
    static_assert(geometry::PointT<geometry::Point<3>>);
    // Global alias also works
    static_assert(PointT<geometry::Point<3>>);
    std::cout << "[PASS] test_concept_point\n";
}

// ===========================================================================
// 8. Constexpr-ness: Point is fully usable at compile time
// ===========================================================================
static void test_constexpr() {
    constexpr geometry::Point<3> p{1.0, 2.0, 3.0};
    constexpr auto arr = p.coord();
    static_assert(arr[0] == 1.0);
    static_assert(arr[1] == 2.0);
    static_assert(arr[2] == 3.0);
    // coord_ref at runtime
    const auto& ref = p.coord_ref();
    assert(approx(ref[0], 1.0));
    std::cout << "[PASS] test_constexpr\n";
}

// ===========================================================================
// 9. 1D Point
// ===========================================================================
static void test_1d_point() {
    constexpr geometry::Point<1> p{42.0};
    static_assert(p.coord(0) == 42.0);
    static_assert(p.dim == 1);
    std::cout << "[PASS] test_1d_point\n";
}

// ===========================================================================
// 10. Copy semantics
// ===========================================================================
static void test_copy() {
    geometry::Point<3> a{1.0, 2.0, 3.0};
    geometry::Point<3> b = a;            // copy
    b.set_coord(0, 99.0);
    assert(approx(a.coord(0), 1.0));     // a unchanged
    assert(approx(b.coord(0), 99.0));
    std::cout << "[PASS] test_copy\n";
}

// ===========================================================================
// 11. Variadic constructor rejects wrong arity (SFINAE)
// ===========================================================================
static void test_wrong_arity_rejected() {
    // geometry::Point<3> p{1.0, 2.0};  // should NOT compile
    static_assert(!std::is_constructible_v<geometry::Point<3>, double, double>);
    // geometry::Point<2> q{1.0, 2.0, 3.0};  // should NOT compile
    static_assert(!std::is_constructible_v<geometry::Point<2>, double, double, double>);
    std::cout << "[PASS] test_wrong_arity_rejected\n";
}

// ===========================================================================

int main() {
    test_default_construction();
    test_variadic_constructor();
    test_array_constructor();
    test_set_coord();
    test_coord_ref();
    test_dim_member();
    test_concept_point();
    test_constexpr();
    test_1d_point();
    test_copy();
    test_wrong_arity_rejected();

    std::cout << "\n=== All 11 Point tests PASSED ===\n";
    return 0;
}
