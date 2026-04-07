#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH

#include <array>
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
    Degraded,
    NotReady,
    SolveFailed,
    InvalidOperator
};

enum class TangentLinearizationScheme {
    Unknown,
    AdaptiveFiniteDifference
};

enum class CouplingTerminationReason {
    NotRun,
    UncoupledMacroStep,
    OneWayStepCompleted,
    LaggedStepCompleted,
    Converged,
    MacroSolveFailed,
    MicroSolveFailed,
    MaxIterationsReached,
    InitializationFailed
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
    TangentLinearizationScheme tangent_scheme{
        TangentLinearizationScheme::Unknown};
    std::array<double, 6> perturbation_sizes{};
    std::array<bool, 6> tangent_column_valid{};
    std::array<bool, 6> tangent_column_central{};
    int failed_perturbations{0};
};

struct CouplingIterationReport {
    CouplingMode mode{CouplingMode::OneWayDownscaling};
    CouplingTerminationReason termination_reason{
        CouplingTerminationReason::NotRun};
    bool converged{true};
    int  iterations{0};
    int  failed_submodels{0};
    int  regularized_submodels{0};

    double max_force_residual_rel{0.0};
    double max_tangent_residual_rel{0.0};
    double macro_solve_seconds{0.0};
    double micro_solve_seconds{0.0};
    bool rollback_performed{false};
    bool relaxation_applied{false};
    bool regularization_detected{false};

    std::vector<double> force_residuals_rel{};
    std::vector<double> tangent_residuals_rel{};
    std::vector<CouplingSite> failed_sites{};
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
        case ResponseStatus::Degraded:
            return "Degraded";
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
to_string(TangentLinearizationScheme scheme)
{
    switch (scheme) {
        case TangentLinearizationScheme::Unknown:
            return "Unknown";
        case TangentLinearizationScheme::AdaptiveFiniteDifference:
            return "AdaptiveFiniteDifference";
    }
    return "UnknownTangentLinearizationScheme";
}

[[nodiscard]] inline constexpr std::string_view
to_string(CouplingTerminationReason reason)
{
    switch (reason) {
        case CouplingTerminationReason::NotRun:
            return "NotRun";
        case CouplingTerminationReason::UncoupledMacroStep:
            return "UncoupledMacroStep";
        case CouplingTerminationReason::OneWayStepCompleted:
            return "OneWayStepCompleted";
        case CouplingTerminationReason::LaggedStepCompleted:
            return "LaggedStepCompleted";
        case CouplingTerminationReason::Converged:
            return "Converged";
        case CouplingTerminationReason::MacroSolveFailed:
            return "MacroSolveFailed";
        case CouplingTerminationReason::MicroSolveFailed:
            return "MicroSolveFailed";
        case CouplingTerminationReason::MaxIterationsReached:
            return "MaxIterationsReached";
        case CouplingTerminationReason::InitializationFailed:
            return "InitializationFailed";
    }
    return "UnknownCouplingTerminationReason";
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
