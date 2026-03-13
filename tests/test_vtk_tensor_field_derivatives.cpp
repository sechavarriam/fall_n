#include <cassert>
#include <cmath>
#include <iostream>

#include "src/post-processing/VTK/VTKTensorFieldDerivatives.hh"

namespace {

constexpr bool approx(double a, double b, double tol = 1.0e-12) {
    return std::abs(a - b) <= tol;
}

void test_component_suffixes() {
    constexpr auto suffixes3 =
        fall_n::vtk::detail::voigt_component_suffixes<3>();
    static_assert(suffixes3.size() == 6);
    static_assert(suffixes3[0] == "xx");
    static_assert(suffixes3[5] == "xy");

    constexpr auto suffixes2 =
        fall_n::vtk::detail::voigt_component_suffixes<2>();
    static_assert(suffixes2.size() == 3);
    static_assert(suffixes2[2] == "xy");
}

void test_stress_scalars_uniaxial() {
    const double sigma[6] = {100.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto scalars =
        fall_n::vtk::detail::derive_stress_field_scalars<3>(sigma);

    assert(approx(scalars.trace, 100.0));
    assert(approx(scalars.mean_stress, 100.0 / 3.0));
    assert(approx(scalars.hydrostatic_stress, 100.0 / 3.0));
    assert(approx(scalars.pressure, -100.0 / 3.0));
    assert(approx(scalars.von_mises, 100.0));
    assert(approx(scalars.beltrami_haigh, 100.0));
    assert(approx(scalars.J2, 10000.0 / 3.0));
    assert(approx(scalars.octahedral_shear_stress,
                  std::sqrt(2.0 / 9.0) * 100.0));
}

void test_stress_scalars_hydrostatic() {
    const double sigma[6] = {80.0, 80.0, 80.0, 0.0, 0.0, 0.0};
    const auto scalars =
        fall_n::vtk::detail::derive_stress_field_scalars<3>(sigma);

    assert(approx(scalars.von_mises, 0.0));
    assert(approx(scalars.beltrami_haigh, 0.0));
    assert(approx(scalars.triaxiality, 0.0));
    assert(approx(scalars.hydrostatic_stress, 80.0));
}

void test_strain_scalars_engineering_voigt() {
    const double strain[6] = {0.06, 0.00, 0.00, 0.00, 0.00, 0.00};
    const auto scalars =
        fall_n::vtk::detail::derive_strain_field_scalars<3>(strain);

    assert(approx(scalars.trace, 0.06));
    assert(approx(scalars.volumetric_strain, 0.06));
    assert(approx(scalars.deviatoric_norm,
                  std::sqrt(2.0 / 3.0) * 0.06));
    assert(approx(scalars.equivalent_strain, 0.04));
}

void test_strain_scalars_pure_hydrostatic() {
    const double strain[6] = {0.02, 0.02, 0.02, 0.0, 0.0, 0.0};
    const auto scalars =
        fall_n::vtk::detail::derive_strain_field_scalars<3>(strain);

    assert(approx(scalars.trace, 0.06));
    assert(approx(scalars.deviatoric_norm, 0.0));
    assert(approx(scalars.equivalent_strain, 0.0));
}

} // namespace

int main() {
    std::cout << "=== VTK Tensor Field Derivatives Test ===\n";
    test_component_suffixes();
    std::cout << "  PASS  test_component_suffixes\n";
    test_stress_scalars_uniaxial();
    std::cout << "  PASS  test_stress_scalars_uniaxial\n";
    test_stress_scalars_hydrostatic();
    std::cout << "  PASS  test_stress_scalars_hydrostatic\n";
    test_strain_scalars_engineering_voigt();
    std::cout << "  PASS  test_strain_scalars_engineering_voigt\n";
    test_strain_scalars_pure_hydrostatic();
    std::cout << "  PASS  test_strain_scalars_pure_hydrostatic\n";
    return 0;
}
