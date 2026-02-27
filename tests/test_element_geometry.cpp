#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>

#include "header_files.hh"

namespace {

constexpr bool approx(double a, double b, double tol = 1e-12) {
  return std::abs(a - b) <= tol;
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

  // Do not call detJ()/integrate() here: detJ for embedded elements is not
  // mathematically defined in the current implementation (non-square J).
}

void test_integrate_span_full_dim_2d_constant() {
  static constexpr std::size_t dim = 2;

  Node<dim> n1{0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0};

  // Q4 element in 2D (topological_dim = dim = 2)
  std::optional<std::array<Node<dim> *, 4>> nodes{std::array<Node<dim> *, 4>{&n1, &n2, &n3, &n4}};
  LagrangeElement2D<2, 2> element(nodes);
  GaussLegendreCellIntegrator<2, 2> integrator;

  ElementGeometry<dim> geom(element, integrator);

  assert(geom.topological_dimension() == 2);
  assert(geom.num_integration_points() == 4);

  // Integral of 1 over the unit square should be 1.
  const double area = geom.integrate([](std::span<const double> /*X*/) { return 1.0; });
  assert(approx(area, 1.0, 1e-10));

  // Same for a constant matrix.
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(2, 2);
  const Eigen::MatrixXd M = geom.integrate([&](std::span<const double> /*X*/) { return I; });
  assert(M.rows() == 2 && M.cols() == 2);
  assert(approx(M(0, 0), 1.0, 1e-10));
  assert(approx(M(1, 1), 1.0, 1e-10));
  assert(approx(M(0, 1), 0.0, 1e-10));
  assert(approx(M(1, 0), 0.0, 1e-10));
}

void test_reference_integration_point_span_shape() {
  static constexpr std::size_t dim = 2;

  Node<dim> n1{0, 0.0, 0.0};
  Node<dim> n2{1, 1.0, 0.0};
  Node<dim> n3{2, 1.0, 1.0};
  Node<dim> n4{3, 0.0, 1.0};

  std::optional<std::array<Node<dim> *, 4>> nodes{std::array<Node<dim> *, 4>{&n1, &n2, &n3, &n4}};
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

} // namespace

int main() {
  test_type_erasure_allows_embedded_topological_dim();
  test_integrate_span_full_dim_2d_constant();
  test_reference_integration_point_span_shape();
  return 0;
}
