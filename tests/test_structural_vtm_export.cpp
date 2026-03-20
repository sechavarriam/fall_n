#include <cassert>
#include <filesystem>
#include <iostream>

#include <vtkCompositeDataSet.h>
#include <vtkDataObject.h>
#include <vtkMultiBlockDataSet.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkXMLMultiBlockDataReader.h>

#include "header_files.hh"

namespace {

using BeamElem = BeamElement<TimoshenkoBeam3D, 3>;
using ShellElem = ShellElement<MindlinReissnerShell3D>;

struct BeamFixture {
    Node<3> n0, n1;
    LagrangeElement3D<2> element;
    GaussLegendreCellIntegrator<2> integrator;
    ElementGeometry<3> geom;
    TimoshenkoBeamMaterial3D relation{200.0, 80.0, 0.4 * 0.8,
                                      0.4 * 0.8 * 0.8 * 0.8 / 12.0,
                                      0.8 * 0.4 * 0.4 * 0.4 / 12.0,
                                      0.05, 5.0 / 6.0, 5.0 / 6.0};
    Material<TimoshenkoBeam3D> material{relation, ElasticUpdate{}};

    BeamFixture()
        : n0{0, 0.0, 0.0, 0.0}
        , n1{1, 2.0, 0.0, 0.0}
        , element{std::optional<std::array<Node<3>*, 2>>{
              std::array<Node<3>*, 2>{&n0, &n1}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    BeamElem make_element() {
        return BeamElem{&geom, material};
    }
};

struct MockBeamModel {
    using element_type = BeamElem;

    std::vector<BeamElem> elements_{};

    const auto& elements() const noexcept { return elements_; }
    Vec state_vector() const noexcept { return nullptr; }
};

struct ShellFixture {
    Node<3> n0, n1, n2, n3;
    LagrangeElement3D<2, 2> element;
    GaussLegendreCellIntegrator<2, 2> integrator;
    ElementGeometry<3> geom;
    MindlinShellMaterial relation{100.0, 0.25, 0.2};
    Material<MindlinReissnerShell3D> material{relation, ElasticUpdate{}};

    ShellFixture()
        : n0{0, 0.0, 0.0, 0.0}
        , n1{1, 1.0, 0.0, 0.0}
        , n2{2, 0.0, 1.0, 0.0}
        , n3{3, 1.0, 1.0, 0.0}
        , element{std::optional<std::array<Node<3>*, 4>>{
              std::array<Node<3>*, 4>{&n0, &n1, &n2, &n3}}}
        , geom{element, integrator}
    {
        geom.set_sieve_id(0);
    }

    ShellElem make_element() {
        return ShellElem{&geom, material};
    }
};

struct MockShellModel {
    using element_type = ShellElem;

    std::vector<ShellElem> elements_{};

    const auto& elements() const noexcept { return elements_; }
    Vec state_vector() const noexcept { return nullptr; }
};

void test_structural_multiblock_writer_contract() {
    BeamFixture fixture;
    MockBeamModel model;
    model.elements_.push_back(fixture.make_element());

    Eigen::Vector<double, 12> u_loc = Eigen::Vector<double, 12>::Zero();
    u_loc[6] = 0.02;
    u_loc[10] = 0.04;

    const auto output_path =
#ifdef FALL_N_SOURCE_DIR
        std::filesystem::path(FALL_N_SOURCE_DIR) / "build" / "test_structural_beam.vtm";
#else
        std::filesystem::path("build/test_structural_beam.vtm");
#endif

    fall_n::vtk::StructuralVTMExporter exporter(
        model,
        fall_n::reconstruction::RectangularSectionProfile<1>{0.4, 0.8});
    exporter.write_with_local_states(output_path.string(), [&](const auto&) { return u_loc; });

    vtkNew<vtkXMLMultiBlockDataReader> reader;
    reader->SetFileName(output_path.string().c_str());
    reader->Update();

    auto* multiblock =
        vtkMultiBlockDataSet::SafeDownCast(reader->GetOutputDataObject(0));
    assert(multiblock != nullptr);
    assert(multiblock->GetNumberOfBlocks() >= 4);

    auto* axis = vtkPolyData::SafeDownCast(multiblock->GetBlock(0));
    auto* surface = vtkPolyData::SafeDownCast(multiblock->GetBlock(1));
    auto* sections = vtkPolyData::SafeDownCast(multiblock->GetBlock(2));
    auto* material_sites = vtkPolyData::SafeDownCast(multiblock->GetBlock(3));
    auto* fibers = multiblock->GetNumberOfBlocks() > 4
        ? vtkPolyData::SafeDownCast(multiblock->GetBlock(4))
        : nullptr;

    assert(axis != nullptr);
    assert(surface != nullptr);
    assert(sections != nullptr);
    assert(material_sites != nullptr);
    assert(axis->GetNumberOfPoints() > 0);
    assert(surface->GetNumberOfPoints() > 0);
    assert(sections->GetNumberOfPoints() > 0);
    assert(material_sites->GetNumberOfPoints() > 0);
    if (fibers != nullptr) {
        assert(fibers->GetNumberOfPoints() == 0);
    }

    assert(axis->GetPointData()->GetArray("displacement") != nullptr);
    assert(surface->GetPointData()->GetArray("stress_xx") != nullptr);
    assert(material_sites->GetPointData()->GetArray("axial_force") != nullptr);

    const char* block_name =
        multiblock->GetMetaData(static_cast<unsigned int>(0))->Get(vtkCompositeDataSet::NAME());
    assert(block_name != nullptr);
    assert(std::string(block_name) == "axis");
}

void test_shell_structural_multiblock_writer_contract() {
    ShellFixture fixture;
    MockShellModel model;
    model.elements_.push_back(fixture.make_element());

    Eigen::Vector<double, 24> u_loc = Eigen::Vector<double, 24>::Zero();
    u_loc[6] = 0.01;
    u_loc[18] = 0.01;
    u_loc[10] = 0.1;
    u_loc[22] = 0.1;

    const auto output_path =
#ifdef FALL_N_SOURCE_DIR
        std::filesystem::path(FALL_N_SOURCE_DIR) / "build" / "test_structural_shell.vtm";
#else
        std::filesystem::path("build/test_structural_shell.vtm");
#endif

    fall_n::vtk::StructuralVTMExporter exporter(model);
    exporter.write_with_local_states(output_path.string(), [&](const auto&) { return u_loc; });

    vtkNew<vtkXMLMultiBlockDataReader> reader;
    reader->SetFileName(output_path.string().c_str());
    reader->Update();

    auto* multiblock =
        vtkMultiBlockDataSet::SafeDownCast(reader->GetOutputDataObject(0));
    assert(multiblock != nullptr);
    assert(multiblock->GetNumberOfBlocks() >= 4);

    auto* axis = vtkPolyData::SafeDownCast(multiblock->GetBlock(0));
    auto* surface = vtkPolyData::SafeDownCast(multiblock->GetBlock(1));
    auto* sections = vtkPolyData::SafeDownCast(multiblock->GetBlock(2));
    auto* material_sites = vtkPolyData::SafeDownCast(multiblock->GetBlock(3));

    assert(axis != nullptr);
    assert(surface != nullptr);
    assert(sections != nullptr);
    assert(material_sites != nullptr);
    assert(axis->GetNumberOfPoints() == 4);
    assert(surface->GetNumberOfPoints() == 8);
    assert(sections->GetNumberOfPoints() > 0);
    assert(material_sites->GetNumberOfPoints() > 0);

    assert(surface->GetPointData()->GetArray("displacement") != nullptr);
    assert(surface->GetPointData()->GetArray("strain_xx") != nullptr);
    assert(material_sites->GetPointData()->GetArray("membrane_11") != nullptr);
}

} // namespace

int main() {
    test_structural_multiblock_writer_contract();
    test_shell_structural_multiblock_writer_contract();
    std::cout << "structural_vtm_export: PASS\n";
    return 0;
}
