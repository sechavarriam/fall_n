#include "src/validation/ReducedRCColumnTrussBaseline.hh"

#include <petsc.h>

#include <cmath>
#include <filesystem>
#include <iostream>

namespace {

using namespace fall_n::validation_reboot;

int passed = 0;
int failed = 0;

void check(bool condition, const char* message)
{
    if (condition) {
        ++passed;
        std::cout << "  [PASS] " << message << "\n";
    } else {
        ++failed;
        std::cout << "  [FAIL] " << message << "\n";
    }
}

void test_quadratic_truss_matches_menegotto_material_under_uniform_cycle()
{
    std::cout << "\nTest: Quadratic truss matches direct Menegotto material baseline\n";

    const auto out_dir = std::filesystem::path{
        "build/test_artifacts/reduced_rc_truss_baseline_regression"};
    std::filesystem::create_directories(out_dir);

    const auto result = run_reduced_rc_column_truss_baseline(
        ReducedRCColumnTrussBaselineRunSpec{
            .protocol_kind =
                ReducedRCColumnTrussProtocolKind::cyclic_compression_return,
            .element_length_m = 1.0,
            .custom_protocol = {
                {.step = 1, .strain = -5.0e-4},
                {.step = 2, .strain = -1.5e-3},
                {.step = 3, .strain = 0.0},
                {.step = 4, .strain = -2.5e-3},
                {.step = 5, .strain = 0.0},
            },
            .steps_per_branch = 8,
            .write_csv = false,
            .print_progress = false,
        },
        out_dir.string());

    check(result.completed_successfully, "truss baseline completes");
    check(
        result.material_records.size() == result.truss_records.size(),
        "material and truss paths produce the same number of records");
    check(
        !result.gauss_records.empty() &&
            result.gauss_records.size() ==
                result.truss_records.size() * 3,
        "quadratic truss exports three Gauss-point records per step");
    check(
        result.comparison.max_abs_stress_error_mpa < 1.0e-9,
        "quadratic truss stress matches direct Menegotto response");
    check(
        result.comparison.max_abs_tangent_error_mpa < 1.0e-8,
        "quadratic truss Gauss tangent matches direct Menegotto response");
    check(
        result.comparison.max_abs_element_tangent_error_mpa < 1.0e-8,
        "quadratic truss element tangent condenses to the same uniaxial tangent");
    check(
        result.comparison.max_abs_gp_strain_spread < 1.0e-12,
        "uniform affine displacement gives identical strain at all Gauss points");
    check(
        result.comparison.max_abs_gp_stress_spread_mpa < 1.0e-9,
        "uniform affine displacement gives identical stress at all Gauss points");
    check(
        result.comparison.max_abs_middle_node_force_mn < 1.0e-10,
        "quadratic truss mid-node axial force stays zero for constant strain");
    check(
        result.comparison.max_abs_axial_force_closure_mn < 1.0e-10,
        "quadratic truss end force closes with stress times area");
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    test_quadratic_truss_matches_menegotto_material_under_uniform_cycle();

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";

    PetscFinalize();
    return failed == 0 ? 0 : 1;
}
