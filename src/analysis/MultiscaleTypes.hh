#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH

#include <cstddef>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

namespace fall_n {

enum class CouplingMode {
    OneWayDownscaling,
    LaggedFeedbackCoupling,
    IteratedTwoWayFE2
};

enum class HomogenizationOperator {
    BoundaryReaction,
    VolumeAverage
};

enum class ResponseStatus {
    Ok,
    NotReady,
    SolveFailed,
    InvalidOperator
};

enum class RegularizationPolicyKind {
    None,
    SPDProjection,
    DiagonalFloor
};

struct CouplingSite {
    std::size_t   macro_element_id{0};
    std::size_t   section_gp{0};
    double        xi{0.0};
    Eigen::Matrix3d local_frame{Eigen::Matrix3d::Identity()};
};

struct MacroSectionState {
    CouplingSite site{};
    Eigen::Vector<double, 6> strain{Eigen::Vector<double, 6>::Zero()};
    Eigen::Vector<double, 6> forces{Eigen::Vector<double, 6>::Zero()};
};

struct SectionHomogenizedResponse {
    CouplingSite site{};
    Eigen::Vector<double, 6> forces{Eigen::Vector<double, 6>::Zero()};
    Eigen::Matrix<double, 6, 6> tangent{
        Eigen::Matrix<double, 6, 6>::Zero()};
    Eigen::Vector<double, 6> strain_ref{Eigen::Vector<double, 6>::Zero()};

    ResponseStatus status{ResponseStatus::NotReady};
    HomogenizationOperator operator_used{
        HomogenizationOperator::BoundaryReaction};
    RegularizationPolicyKind regularization{
        RegularizationPolicyKind::None};

    double tangent_symmetry_error{0.0};
    double diagonal_floor{0.0};
    bool   tangent_regularized{false};
    bool   forces_consistent_with_tangent{true};
};

struct CouplingIterationReport {
    CouplingMode mode{CouplingMode::OneWayDownscaling};
    bool converged{true};
    int  iterations{0};
    int  failed_submodels{0};

    double max_force_residual_rel{0.0};
    double max_tangent_residual_rel{0.0};
    double macro_solve_seconds{0.0};
    double micro_solve_seconds{0.0};

    std::vector<double> force_residuals_rel{};
    std::vector<double> tangent_residuals_rel{};
};

[[nodiscard]] inline constexpr std::string_view to_string(CouplingMode mode)
{
    switch (mode) {
        case CouplingMode::OneWayDownscaling:
            return "OneWayDownscaling";
        case CouplingMode::LaggedFeedbackCoupling:
            return "LaggedFeedbackCoupling";
        case CouplingMode::IteratedTwoWayFE2:
            return "IteratedTwoWayFE2";
    }
    return "UnknownCouplingMode";
}

[[nodiscard]] inline constexpr std::string_view
to_string(HomogenizationOperator op)
{
    switch (op) {
        case HomogenizationOperator::BoundaryReaction:
            return "BoundaryReaction";
        case HomogenizationOperator::VolumeAverage:
            return "VolumeAverage";
    }
    return "UnknownHomogenizationOperator";
}

[[nodiscard]] inline constexpr std::string_view
to_string(ResponseStatus status)
{
    switch (status) {
        case ResponseStatus::Ok:
            return "Ok";
        case ResponseStatus::NotReady:
            return "NotReady";
        case ResponseStatus::SolveFailed:
            return "SolveFailed";
        case ResponseStatus::InvalidOperator:
            return "InvalidOperator";
    }
    return "UnknownResponseStatus";
}

[[nodiscard]] inline constexpr std::string_view
to_string(RegularizationPolicyKind kind)
{
    switch (kind) {
        case RegularizationPolicyKind::None:
            return "None";
        case RegularizationPolicyKind::SPDProjection:
            return "SPDProjection";
        case RegularizationPolicyKind::DiagonalFloor:
            return "DiagonalFloor";
    }
    return "UnknownRegularizationPolicy";
}

}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH
