#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkXMLPolyDataReader.h>

#include "header_files.hh"

namespace {

constexpr bool approx(double a, double b, double tol = 1.0e-9) {
    return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

std::filesystem::path repo_root() {
    return std::filesystem::path(__FILE__).parent_path().parent_path();
}

template <typename StrainT>
void append_linear_segment(std::vector<StrainT>& out,
                           double start,
                           double end,
                           int subdivisions)
{
    assert(subdivisions > 0);
    for (int i = 0; i <= subdivisions; ++i) {
        if (!out.empty() && i == 0) {
            continue;
        }

        const double t = static_cast<double>(i) / static_cast<double>(subdivisions);
        StrainT strain;
        strain.set_components(start + (end - start) * t);
        out.push_back(strain);
    }
}

std::vector<Strain<1>> make_uniaxial_protocol(const std::vector<double>& anchors,
                                              int subdivisions_per_segment)
{
    assert(anchors.size() >= 2);
    std::vector<Strain<1>> protocol;
    for (std::size_t i = 1; i < anchors.size(); ++i) {
        append_linear_segment(protocol,
                              anchors[i - 1],
                              anchors[i],
                              subdivisions_per_segment);
    }
    return protocol;
}

double strain_scalar(const Strain<1>& strain) {
    return strain.components();
}

double stress_scalar(const Stress<1>& stress) {
    return stress.components();
}

std::vector<ConstitutiveCurveSample> sample_menegotto_curve() {
    MaterialInstance<MenegottoPintoSteel, CommittedState> steel{
        200000.0, 420.0, 0.01
    };

    const auto protocol = make_uniaxial_protocol(
        {0.0, 0.012, -0.010, 0.018, -0.006, 0.0},
        20);

    return sample_constitutive_protocol(
        steel, protocol, strain_scalar, stress_scalar);
}

std::vector<ConstitutiveCurveSample> sample_kent_park_curve() {
    MaterialInstance<KentParkConcrete, CommittedState> concrete{30.0, 3.0};

    const auto protocol = make_uniaxial_protocol(
        {0.0, -0.0045, -0.0010, 0.0003, -0.0060, 0.0},
        20);

    return sample_constitutive_protocol(
        concrete, protocol, strain_scalar, stress_scalar);
}

std::vector<ConstitutiveCurveSample> sample_j2_curve() {
    MaterialInstance<J2PlasticityRelation<UniaxialMaterial>, CommittedState> j2{
        200000.0, 0.0, 420.0, 5000.0
    };

    const auto protocol = make_uniaxial_protocol(
        {0.0, 0.0008, 0.0018, 0.0035, 0.0050},
        20);

    return sample_constitutive_protocol(
        j2, protocol, strain_scalar, stress_scalar);
}

void test_material_protocol_sampling() {
    const auto steel_curve = sample_menegotto_curve();
    const auto concrete_curve = sample_kent_park_curve();
    const auto j2_curve = sample_j2_curve();

    assert(!steel_curve.empty());
    assert(!concrete_curve.empty());
    assert(!j2_curve.empty());

    double steel_min = steel_curve.front().ordinate;
    double steel_max = steel_curve.front().ordinate;
    for (const auto& sample : steel_curve) {
        steel_min = std::min(steel_min, sample.ordinate);
        steel_max = std::max(steel_max, sample.ordinate);
    }
    assert(steel_min < 0.0);
    assert(steel_max > 0.0);

    double concrete_min = concrete_curve.front().ordinate;
    double concrete_max = concrete_curve.front().ordinate;
    for (const auto& sample : concrete_curve) {
        concrete_min = std::min(concrete_min, sample.ordinate);
        concrete_max = std::max(concrete_max, sample.ordinate);
    }
    assert(concrete_min < 0.0);
    assert(concrete_max >= -1.0e-6);

    assert(j2_curve.back().ordinate > j2_curve.front().ordinate);
}

void test_vtk_curve_writer_roundtrip() {
    const auto output_dir = repo_root() / "data" / "output" / "material_curves";
    const auto vtm_path = output_dir / "material_hysteresis_curves.vtm";

    std::vector<fall_n::vtk::ConstitutiveCurveSeries> curves;
    curves.push_back({
        .name = "menegotto_cyclic",
        .abscissa_name = "strain",
        .ordinate_name = "stress",
        .samples = sample_menegotto_curve(),
    });
    curves.push_back({
        .name = "kent_park_cyclic",
        .abscissa_name = "strain",
        .ordinate_name = "stress",
        .samples = sample_kent_park_curve(),
    });
    curves.push_back({
        .name = "j2_backbone",
        .abscissa_name = "strain",
        .ordinate_name = "stress",
        .samples = sample_j2_curve(),
    });

    fall_n::vtk::VTKConstitutiveCurveWriter writer;
    writer.write_multiblock(curves, vtm_path.string());

    assert(std::filesystem::exists(vtm_path));
    const auto steel_path = output_dir / "material_hysteresis_curves_menegotto_cyclic.vtp";
    const auto concrete_path = output_dir / "material_hysteresis_curves_kent_park_cyclic.vtp";
    const auto j2_path = output_dir / "material_hysteresis_curves_j2_backbone.vtp";
    assert(std::filesystem::exists(steel_path));
    assert(std::filesystem::exists(concrete_path));
    assert(std::filesystem::exists(j2_path));

    vtkNew<vtkXMLPolyDataReader> reader;
    reader->SetFileName(steel_path.string().c_str());
    reader->Update();

    auto* poly = vtkPolyData::SafeDownCast(reader->GetOutputDataObject(0));
    assert(poly != nullptr);
    assert(poly->GetNumberOfPoints() == static_cast<vtkIdType>(curves[0].samples.size()));
    assert(poly->GetNumberOfLines() == 1);

    auto* strain = poly->GetPointData()->GetArray("strain");
    auto* stress = poly->GetPointData()->GetArray("stress");
    auto* step = poly->GetPointData()->GetArray("step");
    assert(strain != nullptr);
    assert(stress != nullptr);
    assert(step != nullptr);

    assert(approx(strain->GetComponent(0, 0), curves[0].samples.front().abscissa));
    assert(approx(stress->GetComponent(0, 0), curves[0].samples.front().ordinate));
    assert(approx(step->GetComponent(poly->GetNumberOfPoints() - 1, 0),
                  static_cast<double>(curves[0].samples.back().step)));

    auto* active_scalars = poly->GetPointData()->GetScalars();
    assert(active_scalars != nullptr);
    assert(std::string(active_scalars->GetName()) == "stress");
}

} // namespace

int main() {
    std::cout << "=== Material Curves VTK Test ===\n";
    test_material_protocol_sampling();
    std::cout << "  PASS  test_material_protocol_sampling\n";
    test_vtk_curve_writer_roundtrip();
    std::cout << "  PASS  test_vtk_curve_writer_roundtrip\n";
    return 0;
}
