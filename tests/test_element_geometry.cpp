#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <span>

#include "header_files.hh"

namespace {

constexpr bool approx(double a, double b, double tol = 1e-12) {
  return std::abs(a - b) <= tol;
}

// ── Reporting helpers ────────────────────────────────────────────────────
int passed = 0, failed = 0;

void report(const char *name, bool ok) {
  if (ok) { ++passed; std::cout << "  PASS  " << name << "\n"; }
  else    { ++failed; std::cout << "  FAIL  " << name << "\n"; }
}

void test_type_erasure_allows_embedded_topological_dim() {
  static constexpr std::size_t dim = 3;

  Node<dim> n1{0, 0.0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0, 0.0};

  // Line element (topological_dim = 1) embedded in 3D.
  std::optional<std::array<Node<dim> *, 2>> nodes{std::array<Node<dim> *, 2>{&n1, &n2}};
  LagrangeElement3D<2> element(nodes);
  GaussLegendreCellIntegrator<2> integrator;

  ElementGeometry<dim> geom(element, integrator);

  // The key regression: wrapper must accept topo_dim != dim.
  assert(geom.topological_dimension() == 1);
  assert(geom.num_nodes() == 2);

  // evaluate_jacobian returns a 3×1 matrix for embedded beams (non-square).
  auto J = geom.evaluate_jacobian(geom.reference_integration_point(0));
  assert(J.rows() == 3 && J.cols() == 1);

  // differential_measure for 1D-in-3D = ‖J‖ = L/2 = 0.5
  double dm = geom.differential_measure(geom.reference_integration_point(0));
  assert(approx(dm, 0.5, 1e-10));
}

void test_integrate_span_full_dim_2d_constant() {
  static constexpr std::size_t dim = 2;

  Node<dim> n1{0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0};

  // Q4 element in 2D (topological_dim = dim = 2)
  // Node ordering must follow tensor-product convention of LagrangianCell<2,2>:
  //   Node 0 = (-1,-1), Node 1 = (+1,-1), Node 2 = (-1,+1), Node 3 = (+1,+1)
  // i.e. bottom-left, bottom-right, top-left, top-right  (ξ varies fastest)
  std::optional<std::array<Node<dim> *, 4>> nodes{std::array<Node<dim> *, 4>{&n1, &n2, &n4, &n3}};
  LagrangeElement2D<2, 2> element(nodes);
  GaussLegendreCellIntegrator<2, 2> integrator;

  ElementGeometry<dim> geom(element, integrator);

  assert(geom.topological_dimension() == 2);
  assert(geom.num_integration_points() == 4);

  // ∫ I dΩ  over the unit square should yield I (area = 1).
  // Uses the non-virtual template integrate(F&&): no std::function, no heap.
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::MatrixXd M = geom.integrate([&](std::span<const double> /*X*/) { return I; });

  assert(M.rows() == 2 && M.cols() == 2);
  assert(approx(M(0, 0), 1.0, 1e-10));
  assert(approx(M(1, 1), 1.0, 1e-10));
  assert(approx(M(0, 1), 0.0, 1e-10));
  assert(approx(M(1, 0), 0.0, 1e-10));

  // Also test scalar integrate: area of unit square = 1
  double area = geom.integrate([](std::span<const double> /*X*/) { return 1.0; });
  assert(approx(area, 1.0, 1e-10));
}

void test_reference_integration_point_span_shape() {
  static constexpr std::size_t dim = 2;

  Node<dim> n1{0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0};

  // Tensor-product ordering: BL, BR, TL, TR → {n1, n2, n4, n3}
  std::optional<std::array<Node<dim> *, 4>> nodes{std::array<Node<dim> *, 4>{&n1, &n2, &n4, &n3}};
  LagrangeElement2D<2, 2> element(nodes);
  GaussLegendreCellIntegrator<2, 2> integrator;
  ElementGeometry<dim> geom(element, integrator);

  const auto p0 = geom.reference_integration_point(0);
  assert(p0.size() == 2);
  assert(p0.data() != nullptr);
  assert(p0[0] >= -1.0 && p0[0] <= 1.0);
  assert(p0[1] >= -1.0 && p0[1] <= 1.0);

  const auto p0_again = geom.reference_integration_point(0);
  assert(p0_again.size() == 2);
  // Should refer to stable storage (span view).
  assert(p0_again.data() == p0.data());
}

void test_integrate_embedded_1d_in_3d() {
  // A line element of length 1.0 embedded in 3D.
  // ∫ 1 ds = L = 1.0 (using differential_measure = ‖J‖ = L/2).
  static constexpr std::size_t dim = 3;

  Node<dim> n1{0, 0.0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0, 0.0};

  std::optional<std::array<Node<dim> *, 2>> nodes{std::array<Node<dim> *, 2>{&n1, &n2}};
  LagrangeElement3D<2> element(nodes);
  GaussLegendreCellIntegrator<2> integrator;
  ElementGeometry<dim> geom(element, integrator);

  double length = geom.integrate([](std::span<const double>) { return 1.0; });
  assert(approx(length, 1.0, 1e-10));

  // Diagonal line: length = sqrt(3)
  Node<dim> n3{2, 1.0, 1.0, 1.0};
  std::optional<std::array<Node<dim> *, 2>> diag{std::array<Node<dim> *, 2>{&n1, &n3}};
  LagrangeElement3D<2> elem_diag(diag);
  ElementGeometry<dim> geom_diag(elem_diag, integrator);

  double diag_len = geom_diag.integrate([](std::span<const double>) { return 1.0; });
  assert(approx(diag_len, std::sqrt(3.0), 1e-10));
}

// =====================================================================================
// New tests: inclined surfaces, scaled elements, quadrature order variations
// =====================================================================================

// -- Inclined plane in 3D (surface element: 2D topology in 3D space) -----------
void test_inclined_plane_surface_area() {
  // A Q4 surface element forming a tilted plane in 3D.
  // The plane is the unit square in XY rotated 45° about the X-axis.
  //
  //   n1 = (0, 0,       0)
  //   n2 = (1, 0,       0)
  //   n3 = (1, cos45°, sin45°)
  //   n4 = (0, cos45°, sin45°)
  //
  // The side length along Y maps to √(cos²45 + sin²45) = 1.0 in 3D.
  // So the area should remain 1.0.
  //
  // Actually, more interestingly: tilt the plane so area ≠ 1.
  // Let the plane be z = y, i.e. the tilted square:
  //
  //   n1 = (0, 0, 0), n2 = (1, 0, 0), n3 = (1, 1, 1), n4 = (0, 1, 1)
  //
  // J = [∂x/∂ξ | ∂x/∂η] = 3×2 matrix. For this affine element:
  //   ∂x/∂ξ = (n2-n1)/2 = (0.5, 0, 0)
  //   ∂x/∂η = (n4-n1)/2 = (0, 0.5, 0.5)  (tensor-product: n4 is top-left)
  //
  //   J₁ × J₂ = (0.5,0,0)×(0,0.5,0.5) = (0·0.5 - 0·0.5, 0·0 - 0.5·0.5, 0.5·0.5 - 0·0)
  //            = (0, -0.25, 0.25)
  //   ‖J₁ × J₂‖ = sqrt(0 + 0.0625 + 0.0625) = sqrt(0.125) = √2 / 4
  //
  //   Area = ∫∫ ‖J₁ × J₂‖ dξ dη (over [-1,1]²) = (√2/4) * 4 = √2
  //
  static constexpr std::size_t dim = 3;

  Node<dim> n1{0, 0.0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0, 1.0};

  // Tensor-product: BL=n1, BR=n2, TL=n4, TR=n3
  std::optional<std::array<Node<dim> *, 4>> nodes{
      std::array<Node<dim> *, 4>{&n1, &n2, &n4, &n3}};
  LagrangeElement<3, 2, 2> element(nodes); // 2D surface in 3D
  GaussLegendreCellIntegrator<2, 2> integrator;
  ElementGeometry<dim> geom(element, integrator);

  bool ok = true;

  // Check topology
  ok = ok && (geom.topological_dimension() == 2);
  ok = ok && (geom.num_nodes() == 4);
  ok = ok && (geom.num_integration_points() == 4);

  // Check Jacobian shape: 3×2
  auto J = geom.evaluate_jacobian(geom.reference_integration_point(0));
  ok = ok && (J.rows() == 3 && J.cols() == 2);

  // differential_measure at any GP (constant for affine element) = √2/4
  double dm = geom.differential_measure(geom.reference_integration_point(0));
  ok = ok && approx(dm, std::sqrt(2.0) / 4.0, 1e-10);

  // Integrate area: should be √2
  double area = geom.integrate([](std::span<const double>) { return 1.0; });
  ok = ok && approx(area, std::sqrt(2.0), 1e-10);

  report(__func__, ok);
}

// -- Scaled rectangle (non-unit square) ----------------------------------------
void test_scaled_rectangle_2d() {
  // A 2×3 rectangle: origin at (0,0), corners at (2,0), (2,3), (0,3).
  // Area = 6.
  static constexpr std::size_t dim = 2;

  Node<dim> n1{0, 0.0, 0.0};
  Node<dim> n2{1, 2.0, 0.0};
  Node<dim> n3{2, 2.0, 3.0};
  Node<dim> n4{3, 0.0, 3.0};

  // Tensor-product: BL=n1, BR=n2, TL=n4, TR=n3
  std::optional<std::array<Node<dim> *, 4>> nodes{
      std::array<Node<dim> *, 4>{&n1, &n2, &n4, &n3}};
  LagrangeElement2D<2, 2> element(nodes);
  GaussLegendreCellIntegrator<2, 2> integrator;
  ElementGeometry<dim> geom(element, integrator);

  bool ok = true;

  // differential_measure = |detJ| = (2/2)*(3/2) = 1.5 (constant for affine)
  double dm = geom.differential_measure(geom.reference_integration_point(0));
  ok = ok && approx(dm, 1.5, 1e-10);

  // Area = ∫ 1 dΩ = 6
  double area = geom.integrate([](std::span<const double>) { return 1.0; });
  ok = ok && approx(area, 6.0, 1e-10);

  // Integrate x²: ∫₀² ∫₀³ x² dy dx = 3 · [x³/3]₀² = 3 · 8/3 = 8
  // Need to map ξ → x: x(ξ,η) = 1 + ξ (for this element).
  // Use geom.map_local_point to get physical coords.
  double int_x2 = geom.integrate([&](std::span<const double> xi) {
    auto X = geom.map_local_point(xi);
    return X[0] * X[0];
  });
  ok = ok && approx(int_x2, 8.0, 1e-10);

  report(__func__, ok);
}

// -- Quadrature order convergence: 1×1, 2×2, 3×3 for a quadratic field --------
void test_quadrature_order_convergence_2d() {
  // Q4 unit square. Integrand: f(x,y) = x².
  // Exact: ∫₀¹ ∫₀¹ x² dy dx = 1/3.
  //
  // In reference coords: x = (1+ξ)/2 → x² = (1+2ξ+ξ²)/4 (degree 2 in ξ).
  // GL 1×1: exact for degree ≤ 1 per variable → NOT exact for ξ².
  // GL 2×2: exact for degree ≤ 3 per variable → EXACT for ξ².
  // GL 3×3: also exact.
  static constexpr std::size_t dim = 2;

  Node<dim> n1{0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0};

  std::optional<std::array<Node<dim> *, 4>> nodes{
      std::array<Node<dim> *, 4>{&n1, &n2, &n4, &n3}};
  LagrangeElement2D<2, 2> element(nodes);

  auto integrand = [](ElementGeometry<dim>& g, std::span<const double> xi) {
    auto X = g.map_local_point(xi);
    return X[0] * X[0]; // x²
  };

  const double exact = 1.0 / 3.0;
  bool ok = true;

  // 1×1 Gauss: 1 point — cannot integrate x² exactly
  {
    GaussLegendreCellIntegrator<1, 1> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) { return integrand(geom, xi); });
    ok = ok && !approx(val, exact, 1e-6); // Must NOT be exact
  }

  // 2×2 Gauss: exact for quadratics
  {
    GaussLegendreCellIntegrator<2, 2> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) { return integrand(geom, xi); });
    ok = ok && approx(val, exact, 1e-10);
  }

  // 3×3 Gauss: also exact
  {
    GaussLegendreCellIntegrator<3, 3> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) { return integrand(geom, xi); });
    ok = ok && approx(val, exact, 1e-10);
  }

  report(__func__, ok);
}

// -- Quadrature order convergence for polynomial: ∫₀¹ x⁴ dx ─────────────
void test_quadrature_1d_polynomial_convergence() {
  // ∫₀¹ x⁴ dx = 1/5 = 0.2
  // GL n-point rule is exact for polynomials up to degree 2n-1.
  //   n=2: exact up to degree 3 → NOT exact for x⁴
  //   n=3: exact up to degree 5 → EXACT for x⁴
  //   n=4: also exact
  static constexpr std::size_t dim = 2;

  // A 1D element embedded in 2D, from (0,0) to (1,0).
  Node<dim> a{0, 0.0, 0.0};
  Node<dim> b{1, 1.0, 0.0};

  std::optional<std::array<Node<dim>*, 2>> nodes{std::array<Node<dim>*, 2>{&a, &b}};
  LagrangeElement2D<2> element(nodes);

  bool ok = true;

  // n=2: not exact
  {
    GaussLegendreCellIntegrator<2> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) {
      auto X = geom.map_local_point(xi);
      return X[0] * X[0] * X[0] * X[0]; // x⁴
    });
    ok = ok && !approx(val, 0.2, 1e-6);
  }

  // n=3: exact
  {
    GaussLegendreCellIntegrator<3> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) {
      auto X = geom.map_local_point(xi);
      return X[0] * X[0] * X[0] * X[0]; // x⁴
    });
    ok = ok && approx(val, 0.2, 1e-10);
  }

  // n=5: exact
  {
    GaussLegendreCellIntegrator<5> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) {
      auto X = geom.map_local_point(xi);
      return X[0] * X[0] * X[0] * X[0]; // x⁴
    });
    ok = ok && approx(val, 0.2, 1e-10);
  }

  report(__func__, ok);
}

// -- Inclined plane: surface integral with scalar field
void test_surface_integral_with_field() {
  // Tilted plane z = y (same as test_inclined_plane_surface_area):
  //   n1=(0,0,0), n2=(1,0,0), n4=(0,1,1), n3=(1,1,1)
  //
  // Integrate f(x,y,z) = x over the surface.
  // In parametric coords: x = (1+ξ)/2, with surface measure √2/4 dξ dη.
  //
  // ∫∫ x dS = ∫₋₁¹ ∫₋₁¹ (1+ξ)/2 · (√2/4) dξ dη
  //         = (√2/4) · [∫₋₁¹ (1+ξ)/2 dξ] · [∫₋₁¹ dη]
  //         = (√2/4) · 1 · 2 = √2/2
  static constexpr std::size_t dim = 3;

  Node<dim> n1{0, 0.0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0, 1.0};

  std::optional<std::array<Node<dim> *, 4>> nodes{
      std::array<Node<dim> *, 4>{&n1, &n2, &n4, &n3}};
  LagrangeElement<3, 2, 2> element(nodes);
  GaussLegendreCellIntegrator<2, 2> integrator;
  ElementGeometry<dim> geom(element, integrator);

  double result = geom.integrate([&](std::span<const double> xi) {
    auto X = geom.map_local_point(xi);
    return X[0]; // f(x,y,z) = x
  });

  report(__func__, approx(result, std::sqrt(2.0) / 2.0, 1e-10));
}

// -- 3D full hex: volume of a cube ────────────────────────────────────────
void test_hex_volume() {
  // A 2×2×2 cube, nodes from (0,0,0) to (2,2,2). Volume = 8.
  static constexpr std::size_t dim = 3;

  // 8 nodes of the cube, tensor-product ordering:
  // (ξ fastest, then η, then ζ)
  //   0: (-1,-1,-1)→(0,0,0), 1: (+1,-1,-1)→(2,0,0)
  //   2: (-1,+1,-1)→(0,2,0), 3: (+1,+1,-1)→(2,2,0)
  //   4: (-1,-1,+1)→(0,0,2), 5: (+1,-1,+1)→(2,0,2)
  //   6: (-1,+1,+1)→(0,2,2), 7: (+1,+1,+1)→(2,2,2)
  Node<dim> n0{0, 0., 0., 0.};
  Node<dim> n1{1, 2., 0., 0.};
  Node<dim> n2{2, 0., 2., 0.};
  Node<dim> n3{3, 2., 2., 0.};
  Node<dim> n4{4, 0., 0., 2.};
  Node<dim> n5{5, 2., 0., 2.};
  Node<dim> n6{6, 0., 2., 2.};
  Node<dim> n7{7, 2., 2., 2.};

  std::optional<std::array<Node<dim>*, 8>> nodes{
      std::array<Node<dim>*, 8>{&n0, &n1, &n2, &n3, &n4, &n5, &n6, &n7}};
  LagrangeElement3D<2, 2, 2> element(nodes);
  GaussLegendreCellIntegrator<2, 2, 2> integrator;
  ElementGeometry<dim> geom(element, integrator);

  bool ok = true;

  ok = ok && (geom.topological_dimension() == 3);
  ok = ok && (geom.num_integration_points() == 8);

  // Volume = 8
  double vol = geom.integrate([](std::span<const double>) { return 1.0; });
  ok = ok && approx(vol, 8.0, 1e-10);

  // ∫ x dV over [0,2]³ = 2 · 2 · [x²/2]₀² = 4 · 2 = 8
  double int_x = geom.integrate([&](auto xi) {
    auto X = geom.map_local_point(xi);
    return X[0];
  });
  ok = ok && approx(int_x, 8.0, 1e-10);

  report(__func__, ok);
}

// -- Higher-order quadrature on 3D hex for quadratic field ────────────────
void test_hex_higher_order_quadrature() {
  // Unit cube [0,1]³. Integrate f(x,y,z) = x²·y²·z².
  // Exact: ∫₀¹ x² dx · ∫₀¹ y² dy · ∫₀¹ z² dz = (1/3)³ = 1/27
  //
  // 2×2×2 Gauss: exact for total degree ≤ 3 per variable → x²y²z² needs 2 per var → exact.
  // 1×1×1 Gauss: NOT exact.
  static constexpr std::size_t dim = 3;

  Node<dim> n0{0, 0., 0., 0.};
  Node<dim> n1{1, 1., 0., 0.};
  Node<dim> n2{2, 0., 1., 0.};
  Node<dim> n3{3, 1., 1., 0.};
  Node<dim> n4{4, 0., 0., 1.};
  Node<dim> n5{5, 1., 0., 1.};
  Node<dim> n6{6, 0., 1., 1.};
  Node<dim> n7{7, 1., 1., 1.};

  std::optional<std::array<Node<dim>*, 8>> nodes{
      std::array<Node<dim>*, 8>{&n0, &n1, &n2, &n3, &n4, &n5, &n6, &n7}};
  LagrangeElement3D<2, 2, 2> element(nodes);

  const double exact = 1.0 / 27.0;
  auto field = [](ElementGeometry<dim>& g, std::span<const double> xi) {
    auto X = g.map_local_point(xi);
    return X[0]*X[0] * X[1]*X[1] * X[2]*X[2];
  };

  bool ok = true;

  // 1×1×1: not exact
  {
    GaussLegendreCellIntegrator<1, 1, 1> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) { return field(geom, xi); });
    ok = ok && !approx(val, exact, 1e-6);
  }

  // 2×2×2: exact (each variable degree 2 → needs ≥ 2 points)
  {
    GaussLegendreCellIntegrator<2, 2, 2> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) { return field(geom, xi); });
    ok = ok && approx(val, exact, 1e-10);
  }

  // 3×3×3: also exact with extra precision
  {
    GaussLegendreCellIntegrator<3, 3, 3> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) { return field(geom, xi); });
    ok = ok && approx(val, exact, 1e-12);
  }

  report(__func__, ok);
}

// -- 1D line with different quadrature orders ──────────────────────────────
void test_line_quadrature_order_sweep() {
  // Line from (1,2,3) to (4,6,3): length = √(9+16+0) = 5.
  // Integrate f(s)=1 → length=5 for any order ≥ 1 (since linear element, dm is constant).
  static constexpr std::size_t dim = 3;

  Node<dim> a{0, 1.0, 2.0, 3.0};
  Node<dim> b{1, 4.0, 6.0, 3.0};

  std::optional<std::array<Node<dim>*, 2>> nodes{std::array<Node<dim>*, 2>{&a, &b}};
  LagrangeElement3D<2> element(nodes);

  bool ok = true;

  // n=1
  {
    GaussLegendreCellIntegrator<1> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && approx(geom.integrate([](auto) { return 1.0; }), 5.0, 1e-10);
  }
  // n=2
  {
    GaussLegendreCellIntegrator<2> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && approx(geom.integrate([](auto) { return 1.0; }), 5.0, 1e-10);
  }
  // n=5
  {
    GaussLegendreCellIntegrator<5> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && approx(geom.integrate([](auto) { return 1.0; }), 5.0, 1e-10);
  }
  // n=10
  {
    GaussLegendreCellIntegrator<10> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && approx(geom.integrate([](auto) { return 1.0; }), 5.0, 1e-10);
  }

  report(__func__, ok);
}

// -- Matrix integration with differential_measure on surface ──────────────
void test_surface_matrix_integration() {
  // Integrate the 3×3 identity over the tilted plane z=y (area = √2).
  // Result should be √2 · I₃.
  static constexpr std::size_t dim = 3;

  Node<dim> n1{0, 0.0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0, 1.0};

  std::optional<std::array<Node<dim> *, 4>> nodes{
      std::array<Node<dim> *, 4>{&n1, &n2, &n4, &n3}};
  LagrangeElement<3, 2, 2> element(nodes);
  GaussLegendreCellIntegrator<2, 2> integrator;
  ElementGeometry<dim> geom(element, integrator);

  const Eigen::MatrixXd I3 = Eigen::MatrixXd::Identity(3, 3);
  const Eigen::MatrixXd M = geom.integrate([&](std::span<const double>) { return I3; });

  bool ok = true;
  double expected = std::sqrt(2.0);
  ok = ok && (M.rows() == 3 && M.cols() == 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      ok = ok && approx(M(i, j), (i == j) ? expected : 0.0, 1e-10);

  report(__func__, ok);
}

// -- Embedded 1D in 2D: arc length ────────────────────────────────────────
void test_line_in_2d() {
  // Line from (0,0) to (3,4) in 2D. Length = 5.
  static constexpr std::size_t dim = 2;

  Node<dim> a{0, 0.0, 0.0};
  Node<dim> b{1, 3.0, 4.0};

  std::optional<std::array<Node<dim>*, 2>> nodes{std::array<Node<dim>*, 2>{&a, &b}};
  LagrangeElement2D<2> element(nodes);
  GaussLegendreCellIntegrator<2> integrator;
  ElementGeometry<dim> geom(element, integrator);

  bool ok = true;

  ok = ok && (geom.topological_dimension() == 1);
  ok = ok && approx(geom.differential_measure(geom.reference_integration_point(0)), 2.5, 1e-10);
  ok = ok && approx(geom.integrate([](auto) { return 1.0; }), 5.0, 1e-10);

  // Integrate x along the line: x(ξ) = 1.5(1+ξ), ∫₀⁵ (3t/5) dt where t=arc parameter
  // = ∫₋₁¹ 1.5(1+ξ) · 2.5 dξ = 2.5 · 1.5 · [ξ + ξ²/2]₋₁¹ = 2.5 · 1.5 · 2 = 7.5
  double int_x = geom.integrate([&](auto xi) {
    auto X = geom.map_local_point(xi);
    return X[0];
  });
  ok = ok && approx(int_x, 7.5, 1e-10);

  report(__func__, ok);
}

// -- Anisotropic quadrature: different orders per direction ────────────────
void test_anisotropic_quadrature() {
  // Unit square. Integrate f(x,y) = x³ (degree 3 in x, degree 0 in y).
  // Exact: ∫₀¹∫₀¹ x³ dy dx = 1/4.
  //
  // Need ≥ 2 points in x-direction (exact for degree 2n-1=3).
  // Can use 1 point in y-direction (constant in y).
  static constexpr std::size_t dim = 2;

  Node<dim> n1{0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0};

  std::optional<std::array<Node<dim> *, 4>> nodes{
      std::array<Node<dim> *, 4>{&n1, &n2, &n4, &n3}};
  LagrangeElement2D<2, 2> element(nodes);

  auto f = [](ElementGeometry<dim>& g, std::span<const double> xi) {
    auto X = g.map_local_point(xi);
    return X[0] * X[0] * X[0]; // x³
  };

  bool ok = true;

  // 2×1 Gauss (2 in ξ, 1 in η): exact
  {
    GaussLegendreCellIntegrator<2, 1> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && (geom.num_integration_points() == 2);
    double val = geom.integrate([&](auto xi) { return f(geom, xi); });
    ok = ok && approx(val, 0.25, 1e-10);
  }

  // 1×1 Gauss: NOT exact for x³
  {
    GaussLegendreCellIntegrator<1, 1> integ;
    ElementGeometry<dim> geom(element, integ);
    double val = geom.integrate([&](auto xi) { return f(geom, xi); });
    ok = ok && !approx(val, 0.25, 1e-6);
  }

  // 3×1 Gauss: also exact
  {
    GaussLegendreCellIntegrator<3, 1> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && (geom.num_integration_points() == 3);
    double val = geom.integrate([&](auto xi) { return f(geom, xi); });
    ok = ok && approx(val, 0.25, 1e-10);
  }

  report(__func__, ok);
}

// -- Tilted surface with higher-order quadrature ──────────────────────────
void test_inclined_plane_higher_order() {
  // Same tilted plane z=y, area = √2.
  // Integrate with 1×1, 2×2, 3×3 — should all be exact (constant integrand on affine element).
  static constexpr std::size_t dim = 3;

  Node<dim> n1{0, 0.0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0, 1.0};

  std::optional<std::array<Node<dim> *, 4>> nodes{
      std::array<Node<dim> *, 4>{&n1, &n2, &n4, &n3}};
  LagrangeElement<3, 2, 2> element(nodes);
  const double exact_area = std::sqrt(2.0);

  bool ok = true;

  // 1×1
  {
    GaussLegendreCellIntegrator<1, 1> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && approx(geom.integrate([](auto) { return 1.0; }), exact_area, 1e-10);
  }
  // 2×2
  {
    GaussLegendreCellIntegrator<2, 2> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && approx(geom.integrate([](auto) { return 1.0; }), exact_area, 1e-10);
  }
  // 3×3
  {
    GaussLegendreCellIntegrator<3, 3> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && approx(geom.integrate([](auto) { return 1.0; }), exact_area, 1e-10);
  }
  // 5×5
  {
    GaussLegendreCellIntegrator<5, 5> integ;
    ElementGeometry<dim> geom(element, integ);
    ok = ok && approx(geom.integrate([](auto) { return 1.0; }), exact_area, 1e-12);
  }

  report(__func__, ok);
}

} // namespace

int main() {
  // Original tests
  test_type_erasure_allows_embedded_topological_dim();
  test_integrate_span_full_dim_2d_constant();
  test_reference_integration_point_span_shape();
  test_integrate_embedded_1d_in_3d();

  // New integration tests
  test_inclined_plane_surface_area();
  test_scaled_rectangle_2d();
  test_quadrature_order_convergence_2d();
  test_quadrature_1d_polynomial_convergence();
  test_surface_integral_with_field();
  test_hex_volume();
  test_hex_higher_order_quadrature();
  test_line_quadrature_order_sweep();
  test_surface_matrix_integration();
  test_line_in_2d();
  test_anisotropic_quadrature();
  test_inclined_plane_higher_order();

  std::cout << "\n=== ElementGeometry Integration Tests ===\n";
  std::cout << "=== " << passed << " PASSED, " << failed << " FAILED ===\n";

  return failed ? 1 : 0;
}
