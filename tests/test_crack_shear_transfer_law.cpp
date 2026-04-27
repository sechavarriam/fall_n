#include "../src/fracture/CrackShearTransferLaw.hh"

#include <cmath>
#include <iostream>

namespace {

int total_tests = 0;
int passed_tests = 0;

void check(bool condition, const char* message)
{
    ++total_tests;
    if (condition) {
        ++passed_tests;
        std::cout << "[PASS] " << message << "\n";
    } else {
        std::cout << "[FAIL] " << message << "\n";
    }
}

} // namespace

int main()
{
    using fall_n::fracture::CrackShearTransferInput;
    using fall_n::fracture::CrackShearTransferLawKind;
    using fall_n::fracture::CrackShearTransferLawParameters;
    using fall_n::fracture::retained_crack_shear_floor;
    using fall_n::fracture::retained_crack_shear_ratio;

    const CrackShearTransferLawParameters exponential{
        .kind = CrackShearTransferLawKind::opening_exponential,
        .residual_ratio = 0.20,
        .large_opening_ratio = 0.05,
        .opening_decay_strain = 0.010,
        .closure_shear_gain = 1.0,
        .max_closed_ratio = 1.0,
        .closure_reference_strain_multiplier = 1.0};

    const auto closed_onset = CrackShearTransferInput{
        .opening_strain = 0.0,
        .normal_strain = 0.0,
        .damage = 0.0,
        .crack_onset_strain = 1.0e-4};
    check(std::abs(retained_crack_shear_floor(exponential, closed_onset) -
                   0.20) < 1.0e-14,
          "opening-exponential law starts at the residual ratio");

    const auto wide_open = CrackShearTransferInput{
        .opening_strain = 0.10,
        .normal_strain = 0.10,
        .damage = 1.0,
        .crack_onset_strain = 1.0e-4};
    check(retained_crack_shear_floor(exponential, wide_open) < 0.051,
          "opening-exponential law tends to the large-opening floor");

    const CrackShearTransferLawParameters compression_gated{
        .kind = CrackShearTransferLawKind::compression_gated_opening,
        .residual_ratio = 0.20,
        .large_opening_ratio = 0.05,
        .opening_decay_strain = 0.010,
        .closure_shear_gain = 1.0,
        .max_closed_ratio = 1.0,
        .closure_reference_strain_multiplier = 1.0};

    const auto open_response =
        retained_crack_shear_floor(compression_gated, wide_open);
    const auto closing_response =
        retained_crack_shear_floor(
            compression_gated,
            {.opening_strain = 0.10,
             .normal_strain = -0.10,
             .damage = 1.0,
             .crack_onset_strain = 1.0e-4});
    check(closing_response > open_response,
          "compression-gated law restores shear transfer as the crack closes");
    check(std::abs(closing_response - 1.0) < 1.0e-12,
          "compression-gated law recovers the bounded closed-crack target");

    const auto damaged_open = CrackShearTransferInput{
        .opening_strain = 0.10,
        .normal_strain = 0.10,
        .damage = 0.30,
        .crack_onset_strain = 1.0e-4};
    check(std::abs(retained_crack_shear_ratio(exponential, damaged_open) -
                   0.70) < 1.0e-12,
          "retained shear ratio never drops below the intact damage fraction");

    std::cout << passed_tests << "/" << total_tests << " tests passed\n";
    return passed_tests == total_tests ? 0 : 1;
}
