#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <span>
#include <string>

#include "header_files.hh"

namespace {

constexpr bool approx(double a, double b, double tol = 1e-12) {
    return std::abs(a - b) <= tol;
}

int passed = 0;
int failed = 0;

void report(const char* name, bool ok) {
    if (ok) {
        ++passed;
        std::cout << "  PASS  " << name << '\n';
    } else {
        ++failed;
        std::cout << "  FAIL  " << name << '\n';
    }
}

template <std::size_t NumNodes>
std::array<PetscInt, NumNodes> consecutive_ids() {
    std::array<PetscInt, NumNodes> ids{};
    for (std::size_t i = 0; i < NumNodes; ++i) {
        ids[i] = static_cast<PetscInt>(i);
    }
    return ids;
}

struct Hex20Fixture {
    static constexpr std::size_t dim = 3;
    using ElementT = SerendipityElement<3, 3, 2>;
    using IntegratorT = GaussLegendreCellIntegrator<3, 3, 3>;

    std::array<Node<dim>, ElementT::num_nodes> nodes;
    std::array<PetscInt, ElementT::num_nodes> ids;
    ElementT element;
    IntegratorT integrator;
    ElementGeometry<dim> geom;

    static auto make_nodes() {
        constexpr auto ref_nodes = geometry::cell::SerendipityCell<3, 2>::reference_nodes;
        std::array<Node<dim>, ElementT::num_nodes> result{{
            Node<dim>{0, 0.0, 0.0, 0.0}, Node<dim>{1, 0.0, 0.0, 0.0},
            Node<dim>{2, 0.0, 0.0, 0.0}, Node<dim>{3, 0.0, 0.0, 0.0},
            Node<dim>{4, 0.0, 0.0, 0.0}, Node<dim>{5, 0.0, 0.0, 0.0},
            Node<dim>{6, 0.0, 0.0, 0.0}, Node<dim>{7, 0.0, 0.0, 0.0},
            Node<dim>{8, 0.0, 0.0, 0.0}, Node<dim>{9, 0.0, 0.0, 0.0},
            Node<dim>{10, 0.0, 0.0, 0.0}, Node<dim>{11, 0.0, 0.0, 0.0},
            Node<dim>{12, 0.0, 0.0, 0.0}, Node<dim>{13, 0.0, 0.0, 0.0},
            Node<dim>{14, 0.0, 0.0, 0.0}, Node<dim>{15, 0.0, 0.0, 0.0},
            Node<dim>{16, 0.0, 0.0, 0.0}, Node<dim>{17, 0.0, 0.0, 0.0},
            Node<dim>{18, 0.0, 0.0, 0.0}, Node<dim>{19, 0.0, 0.0, 0.0}
        }};

        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto xi = ref_nodes[i].coord();
            result[i] = Node<dim>{
                i,
                0.5 * (xi[0] + 1.0),
                0.5 * (xi[1] + 1.0),
                0.5 * (xi[2] + 1.0)
            };
        }
        return result;
    }

    Hex20Fixture()
        : nodes(make_nodes())
        , ids(consecutive_ids<ElementT::num_nodes>())
        , element(std::size_t{0}, std::span<PetscInt>(ids))
        , geom(element, integrator)
    {
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            geom.bind_node(i, &nodes[i]);
        }
        geom.setup_integration_points(0);
    }
};

struct Tet10Fixture {
    static constexpr std::size_t dim = 3;
    using ElementT = SerendipitySimplexElement<3, 3, 2>;
    using IntegratorT = SimplexIntegrator<3, 2>;

    std::array<Node<dim>, 10> nodes;
    std::array<PetscInt, 10> ids;
    ElementT element;
    IntegratorT integrator;
    ElementGeometry<dim> geom;

    static auto make_nodes() {
        constexpr auto ref_nodes = geometry::simplex::SerendipitySimplexCell<3, 2>::reference_nodes;
        std::array<Node<dim>, 10> result{{
            Node<dim>{0, 0.0, 0.0, 0.0}, Node<dim>{1, 0.0, 0.0, 0.0},
            Node<dim>{2, 0.0, 0.0, 0.0}, Node<dim>{3, 0.0, 0.0, 0.0},
            Node<dim>{4, 0.0, 0.0, 0.0}, Node<dim>{5, 0.0, 0.0, 0.0},
            Node<dim>{6, 0.0, 0.0, 0.0}, Node<dim>{7, 0.0, 0.0, 0.0},
            Node<dim>{8, 0.0, 0.0, 0.0}, Node<dim>{9, 0.0, 0.0, 0.0}
        }};

        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto xi = ref_nodes[i].coord();
            result[i] = Node<dim>{i, xi[0], xi[1], xi[2]};
        }
        return result;
    }

    Tet10Fixture()
        : nodes(make_nodes())
        , ids(consecutive_ids<10>())
        , element(std::size_t{0}, std::span<PetscInt>(ids))
        , geom(element, integrator)
    {
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            geom.bind_node(i, &nodes[i]);
        }
        geom.setup_integration_points(0);
    }
};

struct Hex27Fixture {
    static constexpr std::size_t dim = 3;
    using ElementT = LagrangeElement<3, 3, 3, 3>;
    using IntegratorT = GaussLegendreCellIntegrator<3, 3, 3>;

    std::array<Node<dim>, ElementT::num_nodes> nodes;
    std::array<PetscInt, ElementT::num_nodes> ids;
    ElementT element;
    IntegratorT integrator;
    ElementGeometry<dim> geom;

    static auto make_nodes() {
        constexpr auto ref_nodes = geometry::cell::cell_nodes<3, 3, 3>();
        std::array<Node<dim>, ElementT::num_nodes> result{{
            Node<dim>{0, 0.0, 0.0, 0.0}, Node<dim>{1, 0.0, 0.0, 0.0},
            Node<dim>{2, 0.0, 0.0, 0.0}, Node<dim>{3, 0.0, 0.0, 0.0},
            Node<dim>{4, 0.0, 0.0, 0.0}, Node<dim>{5, 0.0, 0.0, 0.0},
            Node<dim>{6, 0.0, 0.0, 0.0}, Node<dim>{7, 0.0, 0.0, 0.0},
            Node<dim>{8, 0.0, 0.0, 0.0}, Node<dim>{9, 0.0, 0.0, 0.0},
            Node<dim>{10, 0.0, 0.0, 0.0}, Node<dim>{11, 0.0, 0.0, 0.0},
            Node<dim>{12, 0.0, 0.0, 0.0}, Node<dim>{13, 0.0, 0.0, 0.0},
            Node<dim>{14, 0.0, 0.0, 0.0}, Node<dim>{15, 0.0, 0.0, 0.0},
            Node<dim>{16, 0.0, 0.0, 0.0}, Node<dim>{17, 0.0, 0.0, 0.0},
            Node<dim>{18, 0.0, 0.0, 0.0}, Node<dim>{19, 0.0, 0.0, 0.0},
            Node<dim>{20, 0.0, 0.0, 0.0}, Node<dim>{21, 0.0, 0.0, 0.0},
            Node<dim>{22, 0.0, 0.0, 0.0}, Node<dim>{23, 0.0, 0.0, 0.0},
            Node<dim>{24, 0.0, 0.0, 0.0}, Node<dim>{25, 0.0, 0.0, 0.0},
            Node<dim>{26, 0.0, 0.0, 0.0}
        }};

        for (std::size_t i = 0; i < result.size(); ++i) {
            const auto xi = ref_nodes[i].coord();
            result[i] = Node<dim>{
                i,
                0.5 * (xi[0] + 1.0),
                0.5 * (xi[1] + 1.0),
                0.5 * (xi[2] + 1.0)
            };
        }
        return result;
    }

    Hex27Fixture()
        : nodes(make_nodes())
        , ids(consecutive_ids<ElementT::num_nodes>())
        , element(std::size_t{0}, std::span<PetscInt>(ids))
        , geom(element, integrator)
    {
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            geom.bind_node(i, &nodes[i]);
        }
    }
};

void test_serendipity_num_nodes() {
    bool ok = true;
    ok = ok && (geometry::cell::serendipity_num_nodes(1, 1) == 2);
    ok = ok && (geometry::cell::serendipity_num_nodes(1, 2) == 3);
    ok = ok && (geometry::cell::serendipity_num_nodes(2, 1) == 4);
    ok = ok && (geometry::cell::serendipity_num_nodes(2, 2) == 8);
    ok = ok && (geometry::cell::serendipity_num_nodes(3, 1) == 8);
    ok = ok && (geometry::cell::serendipity_num_nodes(3, 2) == 20);
    ok = ok && (geometry::simplex::simplex_num_nodes(3, 2) == 10);
    report(__func__, ok);
}

void test_serendipity_hex20_reference_nodes() {
    using Cell = geometry::cell::SerendipityCell<3, 2>;
    constexpr auto nodes = Cell::reference_nodes;

    bool ok = true;
    ok = ok && (nodes.size() == 20);
    ok = ok && approx(nodes[0].coord(0), -1.0);
    ok = ok && approx(nodes[0].coord(1), -1.0);
    ok = ok && approx(nodes[0].coord(2), -1.0);
    ok = ok && approx(nodes[8].coord(0), 0.0);
    ok = ok && approx(nodes[8].coord(1), -1.0);
    ok = ok && approx(nodes[8].coord(2), -1.0);
    ok = ok && approx(nodes[18].coord(0), 1.0);
    ok = ok && approx(nodes[18].coord(1), 1.0);
    ok = ok && approx(nodes[18].coord(2), 0.0);
    report(__func__, ok);
}

void test_serendipity_hex20_partition_of_unity() {
    using Basis = geometry::cell::SerendipityBasis<3, 2>;
    constexpr Basis basis{};

    bool ok = true;
    const std::array<std::array<double, 3>, 4> pts{{
        {0.0, 0.0, 0.0},
        {0.3, -0.4, 0.2},
        {-0.7, 0.1, -0.2},
        {0.5, 0.5, -0.5}
    }};

    for (const auto& pt : pts) {
        double sum = 0.0;
        for (std::size_t i = 0; i < Basis::num_nodes; ++i) {
            sum += basis.shape(i, pt);
        }
        ok = ok && approx(sum, 1.0, 1e-12);
    }

    report(__func__, ok);
}

void test_serendipity_hex20_kronecker_delta() {
    using Cell = geometry::cell::SerendipityCell<3, 2>;
    constexpr auto nodes = Cell::reference_nodes;
    constexpr auto basis = Cell::basis;

    bool ok = true;
    for (std::size_t a = 0; a < nodes.size(); ++a) {
        const auto xa = nodes[a].coord();
        for (std::size_t b = 0; b < nodes.size(); ++b) {
            const double expected = (a == b) ? 1.0 : 0.0;
            ok = ok && approx(basis.shape(b, xa), expected, 1e-12);
        }
    }
    report(__func__, ok);
}

void test_serendipity_hex20_quadratic_reproduction() {
    using Cell = geometry::cell::SerendipityCell<3, 2>;
    constexpr auto nodes = Cell::reference_nodes;
    constexpr auto basis = Cell::basis;

    std::array<double, Cell::num_nodes> nodal_values{};
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto x = nodes[i].coord();
        nodal_values[i] = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
                        + x[0] * x[1] - 0.5 * x[1] * x[2];
    }

    constexpr std::array<double, 3> xi{0.2, -0.3, 0.4};
    const double interpolated = basis.interpolate(nodal_values, xi);
    const double exact = xi[0] * xi[0] + xi[1] * xi[1] + xi[2] * xi[2]
                       + xi[0] * xi[1] - 0.5 * xi[1] * xi[2];

    report(__func__, approx(interpolated, exact, 1e-12));
}

void test_serendipity_hex20_geometry_volume() {
    Hex20Fixture fx;
    bool ok = true;

    ok = ok && (fx.geom.topological_dimension() == 3);
    ok = ok && (fx.geom.num_nodes() == 20);
    ok = ok && (fx.geom.num_integration_points() == 27);
    ok = ok && approx(fx.geom.integrate([](std::span<const double>) { return 1.0; }), 1.0, 1e-12);

    const double mixed = fx.geom.integrate([&](std::span<const double> xi) {
        const auto X = fx.geom.map_local_point(xi);
        return X[0] * X[1];
    });
    ok = ok && approx(mixed, 0.25, 1e-12);

    report(__func__, ok);
}

void test_serendipity_hex20_face_geometry() {
    Hex20Fixture fx;
    auto face_indices = fx.geom.face_node_indices(0);
    std::vector<PetscInt> face_ids;
    face_ids.reserve(face_indices.size());
    for (auto idx : face_indices) {
        face_ids.push_back(fx.geom.node(idx));
    }

    auto face = fx.geom.make_face_geometry(0, 100, std::span<PetscInt>(face_ids));
    for (std::size_t i = 0; i < face_indices.size(); ++i) {
        face.bind_node(i, &fx.nodes[face_indices[i]]);
    }

    bool ok = true;
    ok = ok && (face.topological_dimension() == 2);
    ok = ok && (face.num_nodes() == 8);
    ok = ok && approx(face.integrate([](std::span<const double>) { return 1.0; }), 1.0, 1e-12);

    report(__func__, ok);
}

void test_serendipity_simplex_partition_of_unity() {
    using Basis = geometry::simplex::SimplexBasis<3, 2>;
    constexpr Basis basis{};

    bool ok = true;
    const std::array<std::array<double, 3>, 4> pts{{
        {0.25, 0.25, 0.25},
        {0.10, 0.20, 0.30},
        {0.60, 0.10, 0.10},
        {0.00, 0.30, 0.20}
    }};

    for (const auto& pt : pts) {
        double sum = 0.0;
        for (std::size_t i = 0; i < 10; ++i) {
            sum += basis.shape_function(i)(pt);
        }
        ok = ok && approx(sum, 1.0, 1e-12);
    }

    report(__func__, ok);
}

void test_serendipity_simplex_geometry_volume() {
    Tet10Fixture fx;
    bool ok = true;

    ok = ok && (fx.geom.topological_dimension() == 3);
    ok = ok && (fx.geom.num_nodes() == 10);
    ok = ok && approx(fx.geom.integrate([](std::span<const double>) { return 1.0; }), 1.0 / 6.0, 1e-12);

    report(__func__, ok);
}

void test_vtk_traits_serendipity() {
    using namespace fall_n::vtk;
    bool ok = true;
    ok = ok && (cell_type_from(2, 8) == VTK_QUADRATIC_QUAD);
    ok = ok && (cell_type_from(3, 20) == VTK_QUADRATIC_HEXAHEDRON);
    ok = ok && (cell_type_from(3, 10) == VTK_QUADRATIC_TETRA);
    report(__func__, ok);
}

void attach_scalar_field(vtkUnstructuredGrid* grid, const std::string& name,
                         const std::vector<double>& values) {
    auto array = vtkSmartPointer<vtkDoubleArray>::New();
    array->SetNumberOfComponents(1);
    array->SetNumberOfTuples(static_cast<vtkIdType>(values.size()));
    array->SetName(name.c_str());
    for (vtkIdType i = 0; i < array->GetNumberOfTuples(); ++i) {
        array->SetTuple1(i, values[static_cast<std::size_t>(i)]);
    }
    grid->GetPointData()->AddArray(array);
}

template <typename NodesArray>
void write_element_vtk(const ElementGeometry<3>& geom,
                       const NodesArray& nodes,
                       const std::string& prefix) {
    namespace fs = std::filesystem;
#ifdef FALL_N_SOURCE_DIR
    const auto out_base = fs::path(FALL_N_SOURCE_DIR) / "data" / "output";
#else
    const auto out_base = fs::path("data") / "output";
#endif
    fs::create_directories(out_base);

    vtkNew<vtkPoints> points;
    points->SetNumberOfPoints(static_cast<vtkIdType>(nodes.size()));

    std::vector<double> nodal_scalar(nodes.size(), 0.0);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        points->SetPoint(static_cast<vtkIdType>(nodes[i].id()),
                         nodes[i].coord(0), nodes[i].coord(1), nodes[i].coord(2));
        nodal_scalar[i] = nodes[i].coord(0) + nodes[i].coord(1) + nodes[i].coord(2);
    }

    vtkNew<vtkUnstructuredGrid> mesh;
    mesh->SetPoints(points);

    vtkIdType ids[64];
    const auto nn = fall_n::vtk::ordered_node_ids(geom, ids);
    const auto ct = fall_n::vtk::cell_type_from(
        geom.topological_dimension(), geom.num_nodes());
    mesh->InsertNextCell(ct, static_cast<vtkIdType>(nn), ids);
    attach_scalar_field(mesh, "nodal_sum_coords", nodal_scalar);
    fall_n::vtk::write_vtu(
        mesh, (out_base / (prefix + "_mesh.vtu")).string());

    vtkNew<vtkPoints> gauss_points;
    gauss_points->SetNumberOfPoints(static_cast<vtkIdType>(geom.integration_points().size()));
    std::vector<double> gauss_scalar(geom.integration_points().size(), 0.0);

    for (const auto& gp : geom.integration_points()) {
        gauss_points->SetPoint(gp.id(), gp.coord(0), gp.coord(1), gp.coord(2));
        gauss_scalar[static_cast<std::size_t>(gp.id())] =
            gp.coord(0) + gp.coord(1) + gp.coord(2);
    }

    vtkNew<vtkUnstructuredGrid> gauss_grid;
    gauss_grid->SetPoints(gauss_points);
    for (const auto& gp : geom.integration_points()) {
        gauss_grid->InsertNextCell(VTK_VERTEX, 1, gp.id_p());
    }
    attach_scalar_field(gauss_grid, "gauss_sum_coords", gauss_scalar);
    fall_n::vtk::write_vtu(
        gauss_grid, (out_base / (prefix + "_gauss.vtu")).string());
}

void test_serendipity_vtk_output() {
    Hex20Fixture hex;
    Tet10Fixture tet;

    write_element_vtk(hex.geom, hex.nodes, "serendipity_hex20_sample");
    write_element_vtk(tet.geom, tet.nodes, "serendipity_tet10_sample");

    namespace fs = std::filesystem;
#ifdef FALL_N_SOURCE_DIR
    const auto out_base = fs::path(FALL_N_SOURCE_DIR) / "data" / "output";
#else
    const auto out_base = fs::path("data") / "output";
#endif
    bool ok = true;
    ok = ok && fs::exists(out_base / "serendipity_hex20_sample_mesh.vtu");
    ok = ok && fs::exists(out_base / "serendipity_hex20_sample_gauss.vtu");
    ok = ok && fs::exists(out_base / "serendipity_tet10_sample_mesh.vtu");
    ok = ok && fs::exists(out_base / "serendipity_tet10_sample_gauss.vtu");

    report(__func__, ok);
}

void benchmark_serendipity_family() {
    Hex20Fixture hex20;
    Hex27Fixture hex27;
    Tet10Fixture tet_ser;

    std::array<Node<3>, 10> tet_nodes = tet_ser.nodes;
    auto tet_ids = consecutive_ids<10>();
    SimplexElement<3, 3, 2> tet_lagrange(std::size_t{0}, std::span<PetscInt>(tet_ids));
    SimplexIntegrator<3, 2> tet_integrator;
    ElementGeometry<3> tet_geom(tet_lagrange, tet_integrator);
    for (std::size_t i = 0; i < tet_nodes.size(); ++i) {
        tet_geom.bind_node(i, &tet_nodes[i]);
    }

    constexpr int repeats = 4000;
    StopWatch sw_hex20;
    StopWatch sw_hex27;
    StopWatch sw_tet_ser;
    StopWatch sw_tet_lag;

    sw_hex20.start();
    double vol20 = 0.0;
    for (int i = 0; i < repeats; ++i) {
        vol20 += hex20.geom.integrate([](std::span<const double>) { return 1.0; });
    }
    sw_hex20.stop();

    sw_hex27.start();
    double vol27 = 0.0;
    for (int i = 0; i < repeats; ++i) {
        vol27 += hex27.geom.integrate([](std::span<const double>) { return 1.0; });
    }
    sw_hex27.stop();

    sw_tet_ser.start();
    double vol_tet_ser = 0.0;
    for (int i = 0; i < repeats; ++i) {
        vol_tet_ser += tet_ser.geom.integrate([](std::span<const double>) { return 1.0; });
    }
    sw_tet_ser.stop();

    sw_tet_lag.start();
    double vol_tet_lag = 0.0;
    for (int i = 0; i < repeats; ++i) {
        vol_tet_lag += tet_geom.integrate([](std::span<const double>) { return 1.0; });
    }
    sw_tet_lag.stop();

    std::cout << "\n=== Serendipity Benchmark ===\n";
    std::cout << "HEX20 integrate(constant): " << sw_hex20.elapsed()
              << " s  | accumulated volume = " << vol20 << '\n';
    std::cout << "HEX27 integrate(constant): " << sw_hex27.elapsed()
              << " s  | accumulated volume = " << vol27 << '\n';
    std::cout << "TET10 serendipity(simplex) : " << sw_tet_ser.elapsed()
              << " s  | accumulated volume = " << vol_tet_ser << '\n';
    std::cout << "TET10 simplex baseline     : " << sw_tet_lag.elapsed()
              << " s  | accumulated volume = " << vol_tet_lag << '\n';

    report(__func__,
           approx(vol20, static_cast<double>(repeats), 1e-8) &&
           approx(vol27, static_cast<double>(repeats), 1e-8) &&
           approx(vol_tet_ser, repeats / 6.0, 1e-8) &&
           approx(vol_tet_lag, repeats / 6.0, 1e-8));
}

} // namespace

int main() {
    test_serendipity_num_nodes();
    test_serendipity_hex20_reference_nodes();
    test_serendipity_hex20_partition_of_unity();
    test_serendipity_hex20_kronecker_delta();
    test_serendipity_hex20_quadratic_reproduction();
    test_serendipity_hex20_geometry_volume();
    test_serendipity_hex20_face_geometry();
    test_serendipity_simplex_partition_of_unity();
    test_serendipity_simplex_geometry_volume();
    test_vtk_traits_serendipity();
    test_serendipity_vtk_output();
    benchmark_serendipity_family();

    std::cout << "\n=== Serendipity Tests ===\n";
    std::cout << "=== " << passed << " PASSED, " << failed << " FAILED ===\n";

    return failed ? 1 : 0;
}
