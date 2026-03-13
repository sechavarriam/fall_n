#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

#include "src/domain/Domain.hh"

namespace {

void test_domain_stores_vertices_canonically() {
    Domain<3> domain;
    domain.add_node(2, 2.0, 0.0, 0.0);
    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1, 1.0, 0.0, 0.0);

    assert(domain.num_vertices() == 3);
    assert(domain.num_nodes() == 3);

    const auto vertices = domain.vertices();
    assert(vertices[0].id() == 2);
    assert(vertices[1].id() == 0);
    assert(vertices[2].id() == 1);

    const auto nodes = domain.nodes();
    assert(nodes[0].id() == 2);
    assert(nodes[1].id() == 0);
    assert(nodes[2].id() == 1);
    assert(nodes[0].coord(0) == 2.0);
    assert(nodes[1].coord(0) == 0.0);
    assert(nodes[2].coord(0) == 1.0);
}

void test_sort_vertices_updates_node_view() {
    Domain<3> domain;
    domain.add_node(2, 2.0, 0.0, 0.0);
    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1, 1.0, 0.0, 0.0);

    domain.sort_vertices_by_id();

    const auto vertices = domain.vertices();
    assert(vertices[0].id() == 0);
    assert(vertices[1].id() == 1);
    assert(vertices[2].id() == 2);

    const auto nodes = domain.nodes();
    assert(nodes[0].id() == 0);
    assert(nodes[1].id() == 1);
    assert(nodes[2].id() == 2);
    assert(nodes[0].coord(0) == 0.0);
    assert(nodes[1].coord(0) == 1.0);
    assert(nodes[2].coord(0) == 2.0);
}

void test_analysis_nodes_remain_distinct_from_vertices() {
    Domain<3> domain;
    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1, 1.0, 0.0, 0.0);

    auto nodes = domain.nodes();
    nodes[1].set_num_dof(6);
    nodes[1].set_dof_index(0, 42);

    assert(domain.node(1).num_dof() == 6);
    assert(domain.node(1).dof_index()[0] == 42);

    const auto& vertex = domain.vertex(1);
    assert(vertex.id() == 1);
    assert(vertex.coord(0) == 1.0);
    assert(vertex.coord(1) == 0.0);
    assert(vertex.coord(2) == 0.0);
}

void test_add_node_from_analysis_node_preserves_geometry() {
    Domain<3> domain;
    Node<3> analysis_node(7, 3.5, -1.0, 2.0);
    domain.add_node(std::move(analysis_node));

    assert(domain.num_vertices() == 1);
    assert(domain.vertex(7).coord(0) == 3.5);
    assert(domain.vertex(7).coord(1) == -1.0);
    assert(domain.vertex(7).coord(2) == 2.0);

    assert(domain.node(7).id() == 7);
    assert(domain.node(7).coord(0) == 3.5);
}

void test_geometry_link_does_not_materialize_analysis_nodes() {
    Domain<2> domain;
    domain.add_vertex(0, 0.0, 0.0);
    domain.add_vertex(1, 1.0, 0.0);
    domain.add_vertex(2, 0.0, 1.0);
    domain.add_vertex(3, 1.0, 1.0);

    PetscInt node_ids[] = {0, 1, 2, 3};
    domain.make_element<LagrangeElement<2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2>{}, 0, node_ids);

    assert(!domain.has_materialized_nodes());
    domain.link_geometry_to_elements();
    assert(!domain.has_materialized_nodes());
}

void test_element_geometry_binds_vertices_before_nodes() {
    Domain<2> domain;
    domain.add_vertex(0, 0.0, 0.0);
    domain.add_vertex(1, 1.0, 0.0);
    domain.add_vertex(2, 0.0, 1.0);
    domain.add_vertex(3, 1.0, 1.0);

    PetscInt node_ids[] = {0, 1, 2, 3};
    domain.make_element<LagrangeElement<2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2>{}, 0, node_ids);

    domain.link_geometry_to_elements();

    const auto& element = domain.element(0);
    const std::array<double, 2> center{0.0, 0.0};
    const auto mapped_center = element.map_local_point(center);

    assert(std::abs(mapped_center[0] - 0.5) < 1.0e-12);
    assert(std::abs(mapped_center[1] - 0.5) < 1.0e-12);
    assert(std::abs(element.point_p(0).coord(0) - 0.0) < 1.0e-12);
    assert(std::abs(element.point_p(3).coord(1) - 1.0) < 1.0e-12);

    const double area = element.integrate([](std::span<const double>) {
        return 1.0;
    });
    assert(std::abs(area - 1.0) < 1.0e-12);
}

void test_boundary_from_plane_uses_vertex_binding() {
    Domain<2> domain;
    domain.add_vertex(0, 0.0, 0.0);
    domain.add_vertex(1, 1.0, 0.0);
    domain.add_vertex(2, 0.0, 1.0);
    domain.add_vertex(3, 1.0, 1.0);

    PetscInt node_ids[] = {0, 1, 2, 3};
    domain.make_element<LagrangeElement<2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2>{}, 0, node_ids);

    domain.link_geometry_to_elements();
    assert(!domain.has_materialized_nodes());
    domain.create_boundary_from_plane("left", 0, 0.0);
    assert(!domain.has_materialized_nodes());

    assert(domain.has_boundary_group("left"));
    const auto edges = domain.boundary_elements("left");
    assert(edges.size() == 1);

    const auto& edge = edges[0];
    const std::array<double, 1> midpoint_ref{0.0};
    const auto midpoint = edge.map_local_point(midpoint_ref);
    const double length = edge.integrate([](std::span<const double>) {
        return 1.0;
    });

    assert(edge.num_nodes() == 2);
    assert(std::abs(midpoint[0] - 0.0) < 1.0e-12);
    assert(std::abs(midpoint[1] - 0.5) < 1.0e-12);
    assert(std::abs(length - 1.0) < 1.0e-12);
}

} // namespace

int main() {
    std::cout << "=== Domain/Vertex Refactor Tests ===\n";

    test_domain_stores_vertices_canonically();
    std::cout << "  PASS  test_domain_stores_vertices_canonically\n";

    test_sort_vertices_updates_node_view();
    std::cout << "  PASS  test_sort_vertices_updates_node_view\n";

    test_analysis_nodes_remain_distinct_from_vertices();
    std::cout << "  PASS  test_analysis_nodes_remain_distinct_from_vertices\n";

    test_add_node_from_analysis_node_preserves_geometry();
    std::cout << "  PASS  test_add_node_from_analysis_node_preserves_geometry\n";

    test_geometry_link_does_not_materialize_analysis_nodes();
    std::cout << "  PASS  test_geometry_link_does_not_materialize_analysis_nodes\n";

    test_element_geometry_binds_vertices_before_nodes();
    std::cout << "  PASS  test_element_geometry_binds_vertices_before_nodes\n";

    test_boundary_from_plane_uses_vertex_binding();
    std::cout << "  PASS  test_boundary_from_plane_uses_vertex_binding\n";

    return 0;
}
