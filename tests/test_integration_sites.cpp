#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "src/geometry/Point.hh"
#include "src/geometry/IntegrationPoint.hh"
#include "src/model/MaterialPoint.hh"
#include "src/elements/section/MaterialSection.hh"
#include "src/elements/section/NodeSection.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/update_strategy/IntegrationStrategy.hh"

namespace {

bool approx(double a, double b, double tol = 1.0e-12) {
    return std::abs(a - b) < tol;
}

void test_integration_point_satisfies_point_view() {
    static_assert(PointViewT<IntegrationPoint<1>>);
    static_assert(PointViewT<IntegrationPoint<2>>);
    static_assert(PointViewT<IntegrationPoint<3>>);

    IntegrationPoint<3> gp;
    gp.set_coord(std::array<double, 3>{1.0, 2.0, 3.0});
    gp.set_weight(0.5);

    assert(gp.coord_ref()[0] == 1.0);
    assert(gp.data() == gp.coord_ref().data());
    assert(approx(gp.weight(), 0.5));
}

void test_material_point_forwards_coordinate_view() {
    using Policy = ThreeDimensionalMaterial;

    Material<Policy> material{ContinuumIsotropicElasticMaterial{200.0e9, 0.3}, ElasticUpdate{}};
    MaterialPoint<Policy> mp(material);

    IntegrationPoint<3> gp;
    gp.set_coord(std::array<double, 3>{0.25, 0.5, 0.75});
    gp.set_weight(2.0);
    mp.bind_integration_point(gp);

    static_assert(PointViewT<MaterialPoint<Policy>>);
    assert(mp.integration_point() == &gp);
    assert(mp.data() == gp.data());
    assert(approx(mp.coord(1), 0.5));
    assert(approx(mp.weight(), 2.0));
}

void test_node_section_forwards_geometry_and_dofs() {
    Node<3> node(5, 1.0, 2.0, 3.0);
    node.set_num_dof(6);
    node.set_dof_index(0, 10);

    section::NodeSection<3> ns(node);

    static_assert(PointViewT<section::NodeSection<3>>);
    assert(ns.data() == node.data());
    assert(ns.dof_data() == node.dof_data());
    assert(approx(ns.coord(2), 3.0));
}

void test_material_section_forwards_integration_point_view() {
    TimoshenkoBeamMaterial3D mat_instance{200e9, 80e9, 0.01, 8.33e-6, 8.33e-6, 1.41e-5};
    Material<TimoshenkoBeam3D> material{mat_instance, ElasticUpdate{}};
    MaterialSection<TimoshenkoBeam3D> ms(std::move(material));

    IntegrationPoint<3> gp;
    gp.set_coord(std::array<double, 3>{0.75, 0.0, 0.0});
    gp.set_weight(1.5);
    ms.bind_integration_point(gp);

    static_assert(PointViewT<MaterialSection<TimoshenkoBeam3D>>);
    assert(ms.integration_point() == &gp);
    assert(ms.data() == gp.data());
    assert(approx(ms.coord(0), 0.75));
    assert(approx(ms.weight(), 1.5));
}

} // namespace

int main() {
    std::cout << "=== Integration Site Tests ===\n";

    test_integration_point_satisfies_point_view();
    std::cout << "  PASS  test_integration_point_satisfies_point_view\n";

    test_material_point_forwards_coordinate_view();
    std::cout << "  PASS  test_material_point_forwards_coordinate_view\n";

    test_node_section_forwards_geometry_and_dofs();
    std::cout << "  PASS  test_node_section_forwards_geometry_and_dofs\n";

    test_material_section_forwards_integration_point_view();
    std::cout << "  PASS  test_material_section_forwards_integration_point_view\n";

    return EXIT_SUCCESS;
}
