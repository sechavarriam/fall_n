#ifndef FALL_N_FRACTURE_CRACK_SHEAR_TRANSFER_LAW_HH
#define FALL_N_FRACTURE_CRACK_SHEAR_TRANSFER_LAW_HH

#include <algorithm>
#include <cmath>
#include <string_view>

namespace fall_n::fracture {

enum class CrackShearTransferLawKind {
    constant_residual,
    opening_exponential,
    compression_gated_opening
};

[[nodiscard]] constexpr std::string_view
to_string(CrackShearTransferLawKind kind) noexcept
{
    switch (kind) {
        case CrackShearTransferLawKind::constant_residual:
            return "constant_residual";
        case CrackShearTransferLawKind::opening_exponential:
            return "opening_exponential";
        case CrackShearTransferLawKind::compression_gated_opening:
            return "compression_gated_opening";
    }
    return "unknown_crack_shear_transfer_law";
}

struct CrackShearTransferLawParameters {
    CrackShearTransferLawKind kind{
        CrackShearTransferLawKind::opening_exponential};
    double residual_ratio{0.20};
    double large_opening_ratio{0.20};
    double opening_decay_strain{1.0};
    double closure_shear_gain{1.0};
    double max_closed_ratio{1.0};
    double closure_reference_strain_multiplier{1.0};
};

struct CrackShearTransferInput {
    double opening_strain{0.0};
    double normal_strain{0.0};
    double damage{0.0};
    double crack_onset_strain{0.0};
};

[[nodiscard]] inline double smoothstep01(double value) noexcept
{
    const double t = std::clamp(value, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

[[nodiscard]] inline CrackShearTransferLawParameters
bounded_crack_shear_transfer_parameters(
    CrackShearTransferLawParameters parameters) noexcept
{
    parameters.residual_ratio =
        std::clamp(parameters.residual_ratio, 1.0e-8, 1.0);
    parameters.large_opening_ratio =
        std::clamp(parameters.large_opening_ratio, 1.0e-8, 1.0);
    parameters.opening_decay_strain =
        std::max(parameters.opening_decay_strain, 1.0e-12);
    parameters.closure_shear_gain =
        std::clamp(parameters.closure_shear_gain, 0.0, 1.0);
    parameters.max_closed_ratio =
        std::clamp(parameters.max_closed_ratio, 1.0e-8, 1.0);
    parameters.closure_reference_strain_multiplier =
        std::max(parameters.closure_reference_strain_multiplier, 1.0e-12);
    return parameters;
}

[[nodiscard]] inline double opening_exponential_shear_floor(
    const CrackShearTransferLawParameters& raw_parameters,
    const CrackShearTransferInput& input) noexcept
{
    const auto parameters =
        bounded_crack_shear_transfer_parameters(raw_parameters);
    const double opening = std::max(input.opening_strain, 0.0);
    if (opening <= 0.0 ||
        std::abs(parameters.large_opening_ratio -
                 parameters.residual_ratio) <= 1.0e-14) {
        return parameters.residual_ratio;
    }

    const double transition =
        std::exp(-opening / parameters.opening_decay_strain);
    return parameters.large_opening_ratio +
           (parameters.residual_ratio - parameters.large_opening_ratio) *
               transition;
}

[[nodiscard]] inline double retained_crack_shear_floor(
    const CrackShearTransferLawParameters& raw_parameters,
    const CrackShearTransferInput& input) noexcept
{
    const auto parameters =
        bounded_crack_shear_transfer_parameters(raw_parameters);
    if (parameters.kind ==
        CrackShearTransferLawKind::constant_residual) {
        return parameters.residual_ratio;
    }

    const double open_floor =
        opening_exponential_shear_floor(parameters, input);
    if (parameters.kind !=
        CrackShearTransferLawKind::compression_gated_opening) {
        return open_floor;
    }

    if (input.normal_strain >= 0.0) {
        return open_floor;
    }

    const double reference_strain = std::max(
        input.opening_strain,
        parameters.closure_reference_strain_multiplier *
            std::max(input.crack_onset_strain, 1.0e-14));
    if (reference_strain <= 1.0e-14) {
        return open_floor;
    }

    const double closure =
        smoothstep01((-input.normal_strain) / reference_strain);
    const double closed_target = std::min(
        parameters.max_closed_ratio,
        open_floor + parameters.closure_shear_gain * (1.0 - open_floor));
    return open_floor + (closed_target - open_floor) * closure;
}

[[nodiscard]] inline double retained_crack_shear_ratio(
    const CrackShearTransferLawParameters& parameters,
    const CrackShearTransferInput& input) noexcept
{
    const double intact_ratio =
        std::clamp(1.0 - input.damage, 1.0e-8, 1.0);
    return std::max(
        retained_crack_shear_floor(parameters, input),
        intact_ratio);
}

} // namespace fall_n::fracture

#endif // FALL_N_FRACTURE_CRACK_SHEAR_TRANSFER_LAW_HH
