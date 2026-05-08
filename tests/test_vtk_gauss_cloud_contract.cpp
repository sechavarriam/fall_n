#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>
#include <vtkVertex.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>

namespace {

constexpr bool approx(double a, double b, double tol = 1.0e-12) {
    return std::abs(a - b) <= tol;
}

vtkSmartPointer<vtkUnstructuredGrid> make_gauss_cloud_grid() {
    vtkNew<vtkPoints> points;
    points->InsertNextPoint(0.25, 0.25, 0.0);
    points->InsertNextPoint(0.75, 0.25, 0.0);
    points->InsertNextPoint(0.25, 0.75, 0.0);
    points->InsertNextPoint(0.75, 0.75, 0.0);

    vtkNew<vtkUnstructuredGrid> grid;
    grid->SetPoints(points);
    grid->Allocate(4);

    for (vtkIdType i = 0; i < 4; ++i) {
        vtkNew<vtkVertex> vertex;
        vertex->GetPointIds()->SetId(0, i);
        grid->InsertNextCell(vertex->GetCellType(), vertex->GetPointIds());
    }

    vtkNew<vtkDoubleArray> displacement;
    displacement->SetName("displacement");
    displacement->SetNumberOfComponents(3);
    displacement->SetNumberOfTuples(4);
    for (vtkIdType i = 0; i < 4; ++i) {
        double x[3]{};
        grid->GetPoint(i, x);
        displacement->SetTuple3(i, x[0], 2.0 * x[1], 0.0);
    }
    grid->GetPointData()->AddArray(displacement);
    grid->GetPointData()->SetActiveVectors("displacement");

    vtkNew<vtkDoubleArray> qp_strain_xx;
    qp_strain_xx->SetName("qp_strain_xx");
    qp_strain_xx->SetNumberOfComponents(1);
    qp_strain_xx->SetNumberOfTuples(4);
    qp_strain_xx->SetValue(0, -0.1);
    qp_strain_xx->SetValue(1, -0.05);
    qp_strain_xx->SetValue(2, 0.05);
    qp_strain_xx->SetValue(3, 0.1);
    grid->GetPointData()->AddArray(qp_strain_xx);
    grid->GetPointData()->SetActiveScalars("qp_strain_xx");

    vtkNew<vtkDoubleArray> qp_stress_von_mises;
    qp_stress_von_mises->SetName("qp_stress_von_mises");
    qp_stress_von_mises->SetNumberOfComponents(1);
    qp_stress_von_mises->SetNumberOfTuples(4);
    qp_stress_von_mises->SetValue(0, 1.0);
    qp_stress_von_mises->SetValue(1, 2.0);
    qp_stress_von_mises->SetValue(2, 3.0);
    qp_stress_von_mises->SetValue(3, 4.0);
    grid->GetPointData()->AddArray(qp_stress_von_mises);

    const auto add_metadata = [&](const char* name, double base) {
        vtkNew<vtkDoubleArray> values;
        values->SetName(name);
        values->SetNumberOfComponents(1);
        values->SetNumberOfTuples(4);
        for (vtkIdType i = 0; i < 4; ++i) {
            values->SetValue(i, base + static_cast<double>(i));
        }
        grid->GetPointData()->AddArray(values);
    };
    add_metadata("gauss_id", 0.0);
    add_metadata("element_id", 10.0);
    add_metadata("material_id", 20.0);
    add_metadata("site_id", 30.0);
    add_metadata("parent_element_id", 40.0);
    add_metadata("qp_damage", 0.0);
    add_metadata("qp_num_cracks", 0.0);

    return grid;
}

void test_gauss_cloud_roundtrip_keeps_point_fields() {
    const auto output_path = std::filesystem::path(
#ifdef FALL_N_SOURCE_DIR
        std::string(FALL_N_SOURCE_DIR) + "/build/test_vtk_gauss_cloud.vtu");
#else
        "build/test_vtk_gauss_cloud.vtu");
#endif

    auto grid = make_gauss_cloud_grid();

    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetFileName(output_path.string().c_str());
    writer->SetInputData(grid);
    writer->SetDataModeToAscii();
    writer->Write();

    vtkNew<vtkXMLUnstructuredGridReader> reader;
    reader->SetFileName(output_path.string().c_str());
    reader->Update();

    auto* roundtripped =
        vtkUnstructuredGrid::SafeDownCast(reader->GetOutputDataObject(0));
    assert(roundtripped != nullptr);
    assert(roundtripped->GetNumberOfPoints() == 4);
    assert(roundtripped->GetNumberOfCells() == 4);

    auto* displacement = roundtripped->GetPointData()->GetArray("displacement");
    assert(displacement != nullptr);
    assert(displacement->GetNumberOfComponents() == 3);

    for (vtkIdType i = 0; i < 4; ++i) {
        double x[3]{};
        roundtripped->GetPoint(i, x);
        assert(approx(displacement->GetComponent(i, 0), x[0]));
        assert(approx(displacement->GetComponent(i, 1), 2.0 * x[1]));
        assert(approx(displacement->GetComponent(i, 2), 0.0));
    }

    auto* qp_strain_xx =
        roundtripped->GetPointData()->GetArray("qp_strain_xx");
    assert(qp_strain_xx != nullptr);
    assert(approx(qp_strain_xx->GetComponent(0, 0), -0.1));
    assert(approx(qp_strain_xx->GetComponent(3, 0), 0.1));

    auto* qp_stress_von_mises =
        roundtripped->GetPointData()->GetArray("qp_stress_von_mises");
    assert(qp_stress_von_mises != nullptr);
    assert(approx(qp_stress_von_mises->GetComponent(2, 0), 3.0));

    for (const auto* name : {"gauss_id",
                             "element_id",
                             "material_id",
                             "site_id",
                             "parent_element_id",
                             "qp_damage",
                             "qp_num_cracks"})
    {
        auto* metadata = roundtripped->GetPointData()->GetArray(name);
        assert(metadata != nullptr);
        assert(metadata->GetNumberOfComponents() == 1);
        assert(metadata->GetNumberOfTuples() == 4);
    }
    assert(approx(roundtripped->GetPointData()
                      ->GetArray("parent_element_id")
                      ->GetComponent(3, 0),
                  43.0));

    auto* active_vectors = roundtripped->GetPointData()->GetVectors();
    assert(active_vectors != nullptr);
    assert(std::string(active_vectors->GetName()) == "displacement");
}

} // namespace

int main() {
    std::cout << "=== VTK Gauss Cloud Contract Test ===\n";
    test_gauss_cloud_roundtrip_keeps_point_fields();
    std::cout << "  PASS  test_gauss_cloud_roundtrip_keeps_point_fields\n";
    return 0;
}
