#ifndef FALL_N_SRC_ANALYSIS_HOMOGENIZED_TANGENT_FINITE_DIFFERENCE_HH
#define FALL_N_SRC_ANALYSIS_HOMOGENIZED_TANGENT_FINITE_DIFFERENCE_HH

// =============================================================================
//  HomogenizedTangentFiniteDifference
// =============================================================================
//
//  Small, model-independent utilities for auditing section-level FE2 tangents.
//  The actual local problem remains owned by the local Model/adapter; this
//  header only defines the perturbation policy and the diagnostic metrics used
//  to decide whether a homogenized operator can be trusted by the macro solve.
//
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <Eigen/Dense>

#include "MultiscaleTypes.hh"

namespace fall_n {

enum class HomogenizedFiniteDifferenceScheme {
    Forward,
    Central
};

struct HomogenizedTangentFiniteDifferenceSettings {
    HomogenizedFiniteDifferenceScheme scheme{
        HomogenizedFiniteDifferenceScheme::Central};
    double relative_perturbation{1.0e-5};
    double absolute_perturbation_floor{1.0e-8};
    double max_perturbation{1.0e-3};
    double validation_relative_tolerance{5.0e-2};
    double validation_column_tolerance{1.0e-1};
    // A value of zero means that only exactly repeated controls reuse the last
    // audited tangent.  Positive values allow a controlled HPC shortcut in
    // long local batches; every reuse remains visible through the response
    // diagnostics because the returned tangent keeps the original perturbation
    // metadata.
    double reuse_control_relative_tolerance{0.0};
};

[[nodiscard]] inline double homogenized_tangent_perturbation_size(
    double value,
    const HomogenizedTangentFiniteDifferenceSettings& settings) noexcept
{
    const double scale = std::max(1.0, std::abs(value));
    const double h = std::max(settings.absolute_perturbation_floor,
                              settings.relative_perturbation * scale);
    return std::min(std::max(h, settings.absolute_perturbation_floor),
                    std::max(settings.absolute_perturbation_floor,
                             settings.max_perturbation));
}

[[nodiscard]] inline std::array<double, 6>
homogenized_tangent_perturbation_sizes(
    const Eigen::Vector<double, 6>& control,
    const HomogenizedTangentFiniteDifferenceSettings& settings) noexcept
{
    std::array<double, 6> out{};
    for (int i = 0; i < 6; ++i) {
        out[static_cast<std::size_t>(i)] =
            homogenized_tangent_perturbation_size(control[i], settings);
    }
    return out;
}

[[nodiscard]] inline double relative_matrix_gap(
    const Eigen::Matrix<double, 6, 6>& a,
    const Eigen::Matrix<double, 6, 6>& b,
    double floor = 1.0e-12) noexcept
{
    const double scale = std::max({a.norm(), b.norm(), floor});
    return (a - b).norm() / scale;
}

[[nodiscard]] inline double max_relative_column_gap(
    const Eigen::Matrix<double, 6, 6>& a,
    const Eigen::Matrix<double, 6, 6>& b,
    std::array<double, 6>* column_gaps = nullptr,
    double floor = 1.0e-12) noexcept
{
    double max_gap = 0.0;
    for (int j = 0; j < 6; ++j) {
        const double scale =
            std::max({a.col(j).norm(), b.col(j).norm(), floor});
        const double gap = (a.col(j) - b.col(j)).norm() / scale;
        if (column_gaps != nullptr) {
            (*column_gaps)[static_cast<std::size_t>(j)] = gap;
        }
        max_gap = std::max(max_gap, gap);
    }
    return max_gap;
}

inline void populate_tangent_validation_diagnostics(
    SectionHomogenizedResponse& response,
    const Eigen::Matrix<double, 6, 6>& reference,
    const HomogenizedTangentFiniteDifferenceSettings& settings,
    TangentValidationNormKind norm =
        TangentValidationNormKind::StateWeightedFrobenius)
{
    response.tangent_validation_status = TangentValidationStatus::Accepted;
    response.tangent_validation_norm = norm;
    response.tangent_validation_relative_tolerance =
        settings.validation_relative_tolerance;
    response.tangent_validation_max_column_tolerance =
        settings.validation_column_tolerance;
    response.tangent_validation_relative_gap =
        relative_matrix_gap(response.tangent, reference);
    response.tangent_validation_max_column_gap =
        max_relative_column_gap(response.tangent,
                                reference,
                                &response.tangent_validation_column_gaps);
    if (response.tangent_validation_relative_gap >
            settings.validation_relative_tolerance ||
        response.tangent_validation_max_column_gap >
            settings.validation_column_tolerance) {
        response.tangent_validation_status = TangentValidationStatus::Rejected;
    }
}

} // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_HOMOGENIZED_TANGENT_FINITE_DIFFERENCE_HH
