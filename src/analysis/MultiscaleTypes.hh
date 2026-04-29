#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH

#include <array>
#include <cstddef>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

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
    LinearizedCondensation,
    AdaptiveFiniteDifference
};

enum class TangentComputationMode {
    PreferLinearizedCondensation,
    ValidateCondensationAgainstAdaptiveFiniteDifference,
    ForceAdaptiveFiniteDifference
};

enum class CondensedTangentStatus {
    NotAttempted,
    Success,
    MissingModel,
    AssemblyFailed,
    NoConstrainedDofs,
    ZeroTransfer,
    FactorizationFailed,
    MissingElementTangent,
    ElementTangentSizeMismatch,
    InvalidElementDofIndex,
    SolveFailed,
    ResidualTooLarge,
    ValidationRejected,
    ForcedAdaptiveFiniteDifference
};

enum class TangentValidationStatus {
    NotRequested,
    ReferenceUnavailable,
    Accepted,
    Rejected
};

enum class TangentValidationNormKind {
    RelativeFrobenius,
    StateWeightedFrobenius,
    SectionPowerScaledFrobenius,
    DualEnergyScaled
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
    TangentComputationMode tangent_mode_requested{
        TangentComputationMode::PreferLinearizedCondensation};
    CondensedTangentStatus condensed_tangent_status{
        CondensedTangentStatus::NotAttempted};
    TangentValidationStatus tangent_validation_status{
        TangentValidationStatus::NotRequested};
    TangentValidationNormKind tangent_validation_norm{
        TangentValidationNormKind::StateWeightedFrobenius};
    double condensed_solve_residual{0.0};
    bool condensed_pattern_reused{false};
    std::size_t condensed_symbolic_factorizations{0};
    double tangent_validation_relative_tolerance{0.0};
    double tangent_validation_max_column_tolerance{0.0};
    double tangent_validation_relative_gap{0.0};
    double tangent_validation_max_column_gap{0.0};
    std::array<double, 6> tangent_validation_column_gaps{};
    std::array<double, 6> tangent_validation_row_scales{
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
    std::array<double, 6> tangent_validation_column_scales{
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
    std::array<double, 6> perturbation_sizes{};
    std::array<bool, 6> tangent_column_valid{};
    std::array<bool, 6> tangent_column_central{};
    int failed_perturbations{0};
    double tangent_min_symmetric_eigenvalue{0.0};
    double tangent_max_symmetric_eigenvalue{0.0};
    double tangent_trace{0.0};
    int tangent_nonpositive_diagonal_entries{0};
};

struct CouplingIterationReport {
    CouplingMode mode{CouplingMode::OneWayDownscaling};
    CouplingTerminationReason termination_reason{
        CouplingTerminationReason::NotRun};
    bool converged{true};
    int  iterations{0};
    int  failed_submodels{0};
    int  regularized_submodels{0};
    TangentValidationNormKind force_residual_norm{
        TangentValidationNormKind::StateWeightedFrobenius};
    TangentValidationNormKind tangent_residual_norm{
        TangentValidationNormKind::StateWeightedFrobenius};

    double max_force_residual_rel{0.0};
    double max_force_component_residual_rel{0.0};
    double max_tangent_residual_rel{0.0};
    double max_tangent_column_residual_rel{0.0};
    double macro_solve_seconds{0.0};
    double micro_solve_seconds{0.0};
    bool local_runtime_adaptive_activation_enabled{false};
    bool local_runtime_seed_reuse_enabled{false};
    int local_runtime_active_sites{0};
    int local_runtime_inactive_sites{0};
    int local_runtime_solve_attempts{0};
    int local_runtime_failed_solve_attempts{0};
    int local_runtime_skipped_by_activation{0};
    int local_runtime_seed_restores{0};
    int local_runtime_checkpoint_saves{0};
    double local_runtime_total_solve_seconds{0.0};
    double local_runtime_mean_site_solve_seconds{0.0};
    double local_runtime_max_site_solve_seconds{0.0};
    int macro_solver_reason{0};
    int macro_solver_iterations{0};
    double macro_solver_function_norm{0.0};
    bool rollback_performed{false};
    bool relaxation_applied{false};
    bool predictor_admissibility_filter_applied{false};
    bool predictor_admissibility_satisfied{true};
    int predictor_admissibility_attempts{0};
    double predictor_admissibility_last_alpha{1.0};
    int macro_step_cutback_attempts{0};
    bool macro_step_cutback_succeeded{false};
    double macro_step_cutback_last_factor{1.0};
    double macro_step_cutback_initial_increment{0.0};
    double macro_step_cutback_last_increment{0.0};
    int macro_backtracking_attempts{0};
    bool macro_backtracking_succeeded{false};
    double macro_backtracking_last_alpha{1.0};
    bool regularization_detected{false};
    bool attempted_state_valid{false};
    int attempted_macro_step{0};
    double attempted_macro_time{0.0};

    std::vector<double> force_residuals_rel{};
    std::vector<double> force_component_residuals_rel{};
    std::vector<double> tangent_residuals_rel{};
    std::vector<double> tangent_column_residuals_rel{};
    std::vector<std::array<double, 6>> force_residual_component_scales{};
    std::vector<std::array<double, 6>> tangent_residual_row_scales{};
    std::vector<std::array<double, 6>> tangent_residual_column_scales{};
    std::vector<double> tangent_min_symmetric_eigenvalues{};
    std::vector<double> tangent_max_symmetric_eigenvalues{};
    std::vector<double> tangent_traces{};
    std::vector<int> tangent_nonpositive_diagonal_counts{};
    std::vector<CouplingSite> failed_sites{};
    std::vector<CouplingSite> predictor_inadmissible_sites{};
};

inline void refresh_section_operator_diagnostics(
    SectionHomogenizedResponse& response)
{
    const auto sym = 0.5 * (response.tangent + response.tangent.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig(sym);
    if (eig.info() == Eigen::Success) {
        response.tangent_min_symmetric_eigenvalue = eig.eigenvalues()[0];
        response.tangent_max_symmetric_eigenvalue = eig.eigenvalues()[5];
    } else {
        response.tangent_min_symmetric_eigenvalue = 0.0;
        response.tangent_max_symmetric_eigenvalue = 0.0;
    }
    response.tangent_trace = response.tangent.trace();
    int nonpositive = 0;
    for (int i = 0; i < 6; ++i) {
        if (response.tangent(i, i) <= 0.0) {
            ++nonpositive;
        }
    }
    response.tangent_nonpositive_diagonal_entries = nonpositive;
}

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
        case TangentLinearizationScheme::LinearizedCondensation:
            return "LinearizedCondensation";
        case TangentLinearizationScheme::AdaptiveFiniteDifference:
            return "AdaptiveFiniteDifference";
    }
    return "UnknownTangentLinearizationScheme";
}

[[nodiscard]] inline constexpr std::string_view
to_string(TangentComputationMode mode)
{
    switch (mode) {
        case TangentComputationMode::PreferLinearizedCondensation:
            return "PreferLinearizedCondensation";
        case TangentComputationMode::
            ValidateCondensationAgainstAdaptiveFiniteDifference:
            return "ValidateCondensationAgainstAdaptiveFiniteDifference";
        case TangentComputationMode::ForceAdaptiveFiniteDifference:
            return "ForceAdaptiveFiniteDifference";
    }
    return "UnknownTangentComputationMode";
}

[[nodiscard]] inline constexpr std::string_view
to_string(CondensedTangentStatus status)
{
    switch (status) {
        case CondensedTangentStatus::NotAttempted:
            return "NotAttempted";
        case CondensedTangentStatus::Success:
            return "Success";
        case CondensedTangentStatus::MissingModel:
            return "MissingModel";
        case CondensedTangentStatus::AssemblyFailed:
            return "AssemblyFailed";
        case CondensedTangentStatus::NoConstrainedDofs:
            return "NoConstrainedDofs";
        case CondensedTangentStatus::ZeroTransfer:
            return "ZeroTransfer";
        case CondensedTangentStatus::FactorizationFailed:
            return "FactorizationFailed";
        case CondensedTangentStatus::MissingElementTangent:
            return "MissingElementTangent";
        case CondensedTangentStatus::ElementTangentSizeMismatch:
            return "ElementTangentSizeMismatch";
        case CondensedTangentStatus::InvalidElementDofIndex:
            return "InvalidElementDofIndex";
        case CondensedTangentStatus::SolveFailed:
            return "SolveFailed";
        case CondensedTangentStatus::ResidualTooLarge:
            return "ResidualTooLarge";
        case CondensedTangentStatus::ValidationRejected:
            return "ValidationRejected";
        case CondensedTangentStatus::ForcedAdaptiveFiniteDifference:
            return "ForcedAdaptiveFiniteDifference";
    }
    return "UnknownCondensedTangentStatus";
}

[[nodiscard]] inline constexpr std::string_view
to_string(TangentValidationStatus status)
{
    switch (status) {
        case TangentValidationStatus::NotRequested:
            return "NotRequested";
        case TangentValidationStatus::ReferenceUnavailable:
            return "ReferenceUnavailable";
        case TangentValidationStatus::Accepted:
            return "Accepted";
        case TangentValidationStatus::Rejected:
            return "Rejected";
    }
    return "UnknownTangentValidationStatus";
}

[[nodiscard]] inline constexpr std::string_view
to_string(TangentValidationNormKind kind)
{
    switch (kind) {
        case TangentValidationNormKind::RelativeFrobenius:
            return "RelativeFrobenius";
        case TangentValidationNormKind::StateWeightedFrobenius:
            return "StateWeightedFrobenius";
        case TangentValidationNormKind::SectionPowerScaledFrobenius:
            return "SectionPowerScaledFrobenius";
        case TangentValidationNormKind::DualEnergyScaled:
            return "DualEnergyScaled";
    }
    return "UnknownTangentValidationNormKind";
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
