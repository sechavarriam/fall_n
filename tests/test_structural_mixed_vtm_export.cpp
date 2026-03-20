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

    BeamElem make_element() { return BeamElem{&geom, material}; }
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
        geom.set_sieve_id(1);
    }

    ShellElem make_element() { return ShellElem{&geom, material}; }
};

struct MockMixedModel {
    using element_type = StructuralElement;

    std::vector<StructuralElement> elements_{};

    const auto& elements() const noexcept { return elements_; }
    Vec state_vector() const noexcept { return nullptr; }
};

void test_structural_mixed_multiblock_writer_contract() {
    BeamFixture beam_fixture;
    ShellFixture shell_fixture;

    MockMixedModel model;
    model.elements_.emplace_back(beam_fixture.make_element());
    model.elements_.emplace_back(shell_fixture.make_element());

    Eigen::Vector<double, 12> beam_u = Eigen::Vector<double, 12>::Zero();
    beam_u[6] = 0.02;
    beam_u[10] = 0.04;

    Eigen::Vector<double, 24> shell_u = Eigen::Vector<double, 24>::Zero();
    shell_u[6] = 0.01;
    shell_u[18] = 0.01;
    shell_u[10] = 0.1;
    shell_u[22] = 0.1;

    const auto output_path = std::filesystem::path(
#ifdef FALL_N_SOURCE_DIR
        std::string(FALL_N_SOURCE_DIR) + "/build/test_structural_mixed.vtm");
#else
        "build/test_structural_mixed.vtm");
#endif

    fall_n::vtk::StructuralVTMExporter exporter(
        model,
        fall_n::reconstruction::RectangularSectionProfile<2>{0.4, 0.8},
        fall_n::reconstruction::ShellThicknessProfile<5>{});

    exporter.write_with_local_states(output_path.string(), [&](const auto& element) {
        using ElementT = std::remove_cvref_t<decltype(element)>;
        if constexpr (std::same_as<ElementT, BeamElem>) {
            return beam_u;
        } else if constexpr (std::same_as<ElementT, TimoshenkoBeamN<2>>) {
            return Eigen::Vector<double, 12>::Zero();
        } else if constexpr (std::same_as<ElementT, TimoshenkoBeamN<3>>) {
            return Eigen::Vector<double, 18>::Zero();
        } else if constexpr (std::same_as<ElementT, TimoshenkoBeamN<4>>) {
            return Eigen::Vector<double, 24>::Zero();
        } else if constexpr (fall_n::vtk::detail::is_beam_element<ElementT>::value) {
            return Eigen::Vector<double, 12>::Zero();
        } else {
            // Generic shell branch: use correct DOF count for each element type
            constexpr int N = ElementT::total_dofs;
            Eigen::Vector<double, N> u = Eigen::Vector<double, N>::Zero();
            if constexpr (N == 24) {
                u[6] = 0.01; u[18] = 0.01; u[10] = 0.1; u[22] = 0.1;
            }
            return u;
        }
    });

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
    if (fibers != nullptr) {
        assert(fibers->GetNumberOfPoints() == 0);
    }

    assert(axis->GetNumberOfPoints() > 0);
    assert(surface->GetNumberOfPoints() > 0);
    assert(sections->GetNumberOfPoints() > 0);
    assert(material_sites->GetNumberOfPoints() > 0);

    assert(axis->GetPointData()->GetArray("axial_strain") != nullptr);
    assert(axis->GetPointData()->GetArray("structural_family") != nullptr);
    assert(surface->GetPointData()->GetArray("stress_xx") != nullptr);
    assert(surface->GetPointData()->GetArray("thickness_offset") != nullptr);
    assert(material_sites->GetPointData()->GetArray("axial_force") != nullptr);
    assert(material_sites->GetPointData()->GetArray("membrane_11") != nullptr);

    const char* axis_name =
        multiblock->GetMetaData(static_cast<unsigned int>(0))->Get(vtkCompositeDataSet::NAME());
    assert(axis_name != nullptr);
    assert(std::string(axis_name) == "axis");

    auto* family = axis->GetPointData()->GetArray("structural_family");
    assert(family != nullptr);
    double range[2]{};
    family->GetRange(range);
    assert(range[0] == 0.0);
    assert(range[1] == 1.0);
}

} // namespace

int main() {
    test_structural_mixed_multiblock_writer_contract();
    std::cout << "structural_mixed_vtm_export: PASS\n";
    return 0;
}
