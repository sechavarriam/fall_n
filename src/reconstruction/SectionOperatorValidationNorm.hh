#ifndef FALL_N_SRC_RECONSTRUCTION_SECTION_OPERATOR_VALIDATION_NORM_HH
#define FALL_N_SRC_RECONSTRUCTION_SECTION_OPERATOR_VALIDATION_NORM_HH

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

#include "../analysis/MultiscaleTypes.hh"

namespace fall_n {

struct SectionOperatorValidationScales {
    TangentValidationNormKind norm{
        TangentValidationNormKind::StateWeightedFrobenius};
    double width{0.0};
    double height{0.0};
    double area{0.0};
    double radius_y{1.0};
    double radius_z{1.0};
    double radius_t{1.0};
    bool valid_geometry{false};
    std::array<double, 6> row{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
    std::array<double, 6> column{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
    std::array<double, 6> vector{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
};

struct SectionOperatorValidationMetrics {
    TangentValidationNormKind norm{
        TangentValidationNormKind::StateWeightedFrobenius};
    double relative_gap{0.0};
    double max_column_gap{0.0};
    std::array<double, 6> column_gaps{};
    std::array<double, 6> row_scales{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
    std::array<double, 6> column_scales{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
};

struct SectionVectorValidationMetrics {
    TangentValidationNormKind norm{
        TangentValidationNormKind::StateWeightedFrobenius};
    double relative_gap{0.0};
    double max_component_gap{0.0};
    std::array<double, 6> component_gaps{};
    std::array<double, 6> component_scales{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
    std::array<double, 6> row_scales{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
    std::array<double, 6> column_scales{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
};

[[nodiscard]] inline SectionOperatorValidationScales
make_section_operator_validation_scales(TangentValidationNormKind norm,
                                        double width,
                                        double height,
                                        const Eigen::Vector<double, 6>&
                                            reference_generalized_state =
                                                Eigen::Vector<double, 6>::Zero(),
                                        const Eigen::Vector<double, 6>&
                                            reference_generalized_force =
                                                Eigen::Vector<double, 6>::Zero())
{
    SectionOperatorValidationScales scales;
    scales.norm = norm;
    scales.width = width;
    scales.height = height;

    if (!(std::isfinite(width) && std::isfinite(height))
        || width <= 0.0
        || height <= 0.0)
    {
        return scales;
    }

    scales.area = width * height;
    scales.valid_geometry = true;

    if (norm == TangentValidationNormKind::RelativeFrobenius) {
        return scales;
    }

    scales.radius_y = std::max(0.5 * height,
                               std::numeric_limits<double>::epsilon());
    scales.radius_z = std::max(0.5 * width,
                               std::numeric_limits<double>::epsilon());
    scales.radius_t =
        std::max(0.5 * std::sqrt(width * width + height * height),
                 std::numeric_limits<double>::epsilon());

    scales.row = {{
        1.0,
        1.0 / scales.radius_y,
        1.0 / scales.radius_z,
        1.0,
        1.0,
        1.0 / scales.radius_t,
    }};

    const std::array<double, 6> scaled_state{{
        std::abs(reference_generalized_state[0]),
        scales.radius_y * std::abs(reference_generalized_state[1]),
        scales.radius_z * std::abs(reference_generalized_state[2]),
        std::abs(reference_generalized_state[3]),
        std::abs(reference_generalized_state[4]),
        scales.radius_t * std::abs(reference_generalized_state[5]),
    }};
    const double max_scaled_state = std::max(
        {1.0e-6,
         scaled_state[0],
         scaled_state[1],
         scaled_state[2],
         scaled_state[3],
         scaled_state[4],
         scaled_state[5]});
    const double state_floor = 1.0e-3;

    for (std::size_t j = 0; j < scales.column.size(); ++j) {
        scales.column[j] = std::max(
            scaled_state[j] / max_scaled_state,
            state_floor);
    }
    for (std::size_t i = 0; i < scales.vector.size(); ++i) {
        scales.vector[i] = scales.row[i] * scales.column[i];
    }

    if (norm == TangentValidationNormKind::StateWeightedFrobenius) {
        scales.row = {{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
        for (std::size_t i = 0; i < scales.vector.size(); ++i) {
            scales.vector[i] = scales.column[i];
        }
        return scales;
    }

    if (norm == TangentValidationNormKind::DualEnergyScaled) {
        const std::array<double, 6> force_state{{
            std::abs(reference_generalized_force[0]),
            std::abs(reference_generalized_force[1]),
            std::abs(reference_generalized_force[2]),
            std::abs(reference_generalized_force[3]),
            std::abs(reference_generalized_force[4]),
            std::abs(reference_generalized_force[5]),
        }};
        const double max_force_state = std::max(
            {1.0e-6,
             force_state[0],
             force_state[1],
             force_state[2],
             force_state[3],
             force_state[4],
             force_state[5]});
        const double force_floor = 1.0e-3 * max_force_state;
        double power_scale = 0.0;
        for (std::size_t i = 0; i < scales.column.size(); ++i) {
            power_scale += std::max(force_state[i], force_floor)
                         * std::max(scaled_state[i], state_floor);
        }
        power_scale = std::max(power_scale, max_force_state * state_floor);
        for (std::size_t i = 0; i < scales.row.size(); ++i) {
            const double activity =
                std::max(scaled_state[i], state_floor);
            scales.row[i] = (max_force_state * activity) / power_scale;
            scales.vector[i] = scales.row[i];
        }
        return scales;
    }
    return scales;
}

[[nodiscard]] inline Eigen::Matrix<double, 6, 6>
apply_section_operator_validation_scales(
    const Eigen::Matrix<double, 6, 6>& op,
    const SectionOperatorValidationScales& scales)
{
    Eigen::Matrix<double, 6, 6> scaled = op;
    for (int i = 0; i < 6; ++i) {
        const double row_scale =
            scales.row[static_cast<std::size_t>(i)];
        for (int j = 0; j < 6; ++j) {
            const double column_weight =
                scales.column[static_cast<std::size_t>(j)];
            scaled(i, j) *= row_scale * column_weight;
        }
    }
    return scaled;
}

[[nodiscard]] inline Eigen::Vector<double, 6>
apply_section_vector_validation_scales(
    const Eigen::Vector<double, 6>& vec,
    const SectionOperatorValidationScales& scales)
{
    Eigen::Vector<double, 6> scaled = vec;
    for (int i = 0; i < 6; ++i) {
        const auto index = static_cast<std::size_t>(i);
        scaled[i] *= scales.vector[index];
    }
    return scaled;
}

[[nodiscard]] inline SectionOperatorValidationMetrics
compute_section_operator_validation_metrics(
    const Eigen::Matrix<double, 6, 6>& lhs,
    const Eigen::Matrix<double, 6, 6>& rhs,
    const SectionOperatorValidationScales& scales)
{
    SectionOperatorValidationMetrics metrics;
    metrics.norm = scales.norm;
    metrics.row_scales = scales.row;
    metrics.column_scales = scales.column;

    const auto lhs_scaled =
        apply_section_operator_validation_scales(lhs, scales);
    const auto rhs_scaled =
        apply_section_operator_validation_scales(rhs, scales);

    const double denom = std::max(
        {1.0, lhs_scaled.norm(), rhs_scaled.norm()});
    metrics.relative_gap = (lhs_scaled - rhs_scaled).norm() / denom;

    for (int j = 0; j < 6; ++j) {
        const double column_denom = std::max(
            {1.0, lhs_scaled.col(j).norm(), rhs_scaled.col(j).norm()});
        metrics.column_gaps[static_cast<std::size_t>(j)] =
            (lhs_scaled.col(j) - rhs_scaled.col(j)).norm() / column_denom;
    }

    metrics.max_column_gap =
        *std::max_element(metrics.column_gaps.begin(),
                          metrics.column_gaps.end());
    return metrics;
}

[[nodiscard]] inline SectionVectorValidationMetrics
compute_section_vector_validation_metrics(
    const Eigen::Vector<double, 6>& lhs,
    const Eigen::Vector<double, 6>& rhs,
    const SectionOperatorValidationScales& scales)
{
    SectionVectorValidationMetrics metrics;
    metrics.norm = scales.norm;
    metrics.row_scales = scales.row;
    metrics.column_scales = scales.column;

    for (std::size_t i = 0; i < metrics.component_scales.size(); ++i) {
        metrics.component_scales[i] = scales.vector[i];
    }

    const auto lhs_scaled =
        apply_section_vector_validation_scales(lhs, scales);
    const auto rhs_scaled =
        apply_section_vector_validation_scales(rhs, scales);

    const double denom = std::max(
        {1.0, lhs_scaled.norm(), rhs_scaled.norm()});
    metrics.relative_gap = (lhs_scaled - rhs_scaled).norm() / denom;

    for (std::size_t i = 0; i < metrics.component_gaps.size(); ++i) {
        const double component_denom = std::max(
            {1.0,
             std::abs(lhs_scaled[static_cast<Eigen::Index>(i)]),
             std::abs(rhs_scaled[static_cast<Eigen::Index>(i)])});
        metrics.component_gaps[i] =
            std::abs(lhs_scaled[static_cast<Eigen::Index>(i)]
                     - rhs_scaled[static_cast<Eigen::Index>(i)])
            / component_denom;
    }
    metrics.max_component_gap =
        *std::max_element(metrics.component_gaps.begin(),
                          metrics.component_gaps.end());
    return metrics;
}

}  // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_SECTION_OPERATOR_VALIDATION_NORM_HH
