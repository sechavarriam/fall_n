#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH

#include <array>
#include <cstddef>
#include <string>
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
    HybridObservationStepCompleted,
    MacroSolveFailed,
    MicroSolveFailed,
    MaxIterationsReached,
    InitializationFailed
};

enum class TwoWayFailureRecoveryMode {
    StrictTwoWay,
    HybridObservationWindow,
    OneWayOnly
};

enum class CouplingRegime {
    StrictTwoWay,
    HybridObservationWindow,
    OneWayOnly
};

enum class CouplingFeedbackSource {
    CurrentHomogenized,
    LastConvergedHomogenized,
    ClearedOneWay,
    None
};

struct TwoWayFailureRecoveryPolicy {
    TwoWayFailureRecoveryMode mode{TwoWayFailureRecoveryMode::StrictTwoWay};
    int max_hybrid_steps{0};
    int return_success_steps{5};
    double work_gap_tolerance{0.05};
    double force_jump_tolerance{0.05};
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

struct CouplingSiteIterationRecord {
    int iteration{0};
    std::size_t local_site_index{0};
    CouplingSite site{};
    ResponseStatus status{ResponseStatus::NotReady};
    HomogenizationOperator operator_used{
        HomogenizationOperator::BoundaryReaction};
    TangentLinearizationScheme tangent_scheme{
        TangentLinearizationScheme::Unknown};
    CondensedTangentStatus condensed_tangent_status{
        CondensedTangentStatus::NotAttempted};
    RegularizationPolicyKind regularization{
        RegularizationPolicyKind::None};
    bool tangent_regularized{false};
    double force_residual_rel{0.0};
    double force_component_residual_rel{0.0};
    double tangent_residual_rel{0.0};
    double tangent_column_residual_rel{0.0};
    double tangent_min_symmetric_eigenvalue{0.0};
    double tangent_max_symmetric_eigenvalue{0.0};
    double tangent_trace{0.0};
    int tangent_nonpositive_diagonal_entries{0};
    bool adaptive_relaxation_applied{false};
    int adaptive_relaxation_attempts{0};
    double adaptive_relaxation_alpha{1.0};
    double previous_force_residual_rel{0.0};
    Eigen::Vector<double, 6> macro_strain{Eigen::Vector<double, 6>::Zero()};
    Eigen::Vector<double, 6> macro_forces{Eigen::Vector<double, 6>::Zero()};
    Eigen::Vector<double, 6> response_strain_ref{Eigen::Vector<double, 6>::Zero()};
    Eigen::Vector<double, 6> response_forces{Eigen::Vector<double, 6>::Zero()};
};

struct CouplingIterationReport {
    CouplingMode mode{CouplingMode::OneWayDownscaling};
    CouplingTerminationReason termination_reason{
        CouplingTerminationReason::NotRun};
    CouplingRegime coupling_regime{CouplingRegime::StrictTwoWay};
    CouplingFeedbackSource feedback_source{CouplingFeedbackSource::None};
    bool hybrid_active{false};
    std::string hybrid_reason{};
    std::string one_way_replay_status{};
    double work_gap{0.0};
    bool return_gate_passed{false};
    int hybrid_window_steps{0};
    int hybrid_success_steps{0};
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
    int local_runtime_seed_evictions{0};
    int local_runtime_checkpoint_saves{0};
    int local_runtime_cached_seed_states{0};
    int local_runtime_max_cached_seed_states{0};
    bool local_runtime_seed_cache_capacity_limited{false};
    double local_runtime_total_solve_seconds{0.0};
    double local_runtime_mean_site_solve_seconds{0.0};
    double local_runtime_max_site_solve_seconds{0.0};
    int macro_solver_reason{0};
    int macro_solver_iterations{0};
    double macro_solver_function_norm{0.0};
    bool rollback_performed{false};
    bool relaxation_applied{false};
    bool adaptive_relaxation_applied{false};
    int adaptive_relaxation_attempts{0};
    double adaptive_relaxation_min_alpha{1.0};
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
    std::vector<std::string> local_failure_messages{};
    std::vector<CouplingSite> predictor_inadmissible_sites{};
    std::vector<CouplingSiteIterationRecord> site_iteration_records{};
};

inline void append_site_iteration_record(
    CouplingIterationReport& report,
    int iteration,
    std::size_t local_site_index,
    const MacroSectionState& macro_state,
    const SectionHomogenizedResponse& response,
    double previous_force_residual_rel = 0.0,
    bool adaptive_relaxation_applied = false,
    int adaptive_relaxation_attempts = 0,
    double adaptive_relaxation_alpha = 1.0)
{
    auto force_residual = local_site_index < report.force_residuals_rel.size()
        ? report.force_residuals_rel[local_site_index]
        : 0.0;
    auto force_component_residual =
        local_site_index < report.force_component_residuals_rel.size()
            ? report.force_component_residuals_rel[local_site_index]
            : 0.0;
    auto tangent_residual =
        local_site_index < report.tangent_residuals_rel.size()
            ? report.tangent_residuals_rel[local_site_index]
            : 0.0;
    auto tangent_column_residual =
        local_site_index < report.tangent_column_residuals_rel.size()
            ? report.tangent_column_residuals_rel[local_site_index]
            : 0.0;

    report.site_iteration_records.push_back(CouplingSiteIterationRecord{
        .iteration = iteration,
        .local_site_index = local_site_index,
        .site = response.site,
        .status = response.status,
        .operator_used = response.operator_used,
        .tangent_scheme = response.tangent_scheme,
        .condensed_tangent_status = response.condensed_tangent_status,
        .regularization = response.regularization,
        .tangent_regularized = response.tangent_regularized,
        .force_residual_rel = force_residual,
        .force_component_residual_rel = force_component_residual,
        .tangent_residual_rel = tangent_residual,
        .tangent_column_residual_rel = tangent_column_residual,
        .tangent_min_symmetric_eigenvalue =
            response.tangent_min_symmetric_eigenvalue,
        .tangent_max_symmetric_eigenvalue =
            response.tangent_max_symmetric_eigenvalue,
        .tangent_trace = response.tangent_trace,
        .tangent_nonpositive_diagonal_entries =
            response.tangent_nonpositive_diagonal_entries,
        .adaptive_relaxation_applied = adaptive_relaxation_applied,
        .adaptive_relaxation_attempts = adaptive_relaxation_attempts,
        .adaptive_relaxation_alpha = adaptive_relaxation_alpha,
        .previous_force_residual_rel = previous_force_residual_rel,
        .macro_strain = macro_state.strain,
        .macro_forces = macro_state.forces,
        .response_strain_ref = response.strain_ref,
        .response_forces = response.forces});
}

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
to_string(TwoWayFailureRecoveryMode mode)
{
    switch (mode) {
        case TwoWayFailureRecoveryMode::StrictTwoWay:
            return "strict_two_way";
        case TwoWayFailureRecoveryMode::HybridObservationWindow:
            return "hybrid_observation_window";
        case TwoWayFailureRecoveryMode::OneWayOnly:
            return "one_way_only";
    }
    return "unknown_two_way_failure_recovery_mode";
}

[[nodiscard]] inline constexpr std::string_view
to_string(CouplingRegime regime)
{
    switch (regime) {
        case CouplingRegime::StrictTwoWay:
            return "strict_two_way";
        case CouplingRegime::HybridObservationWindow:
            return "hybrid_observation_window";
        case CouplingRegime::OneWayOnly:
            return "one_way_only";
    }
    return "unknown_coupling_regime";
}

[[nodiscard]] inline constexpr std::string_view
to_string(CouplingFeedbackSource source)
{
    switch (source) {
        case CouplingFeedbackSource::CurrentHomogenized:
            return "current_homogenized";
        case CouplingFeedbackSource::LastConvergedHomogenized:
            return "last_converged_homogenized";
        case CouplingFeedbackSource::ClearedOneWay:
            return "cleared_one_way";
        case CouplingFeedbackSource::None:
            return "none";
    }
    return "unknown_feedback_source";
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
        case CouplingTerminationReason::HybridObservationStepCompleted:
            return "HybridObservationStepCompleted";
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

// ─── Plan v2 §Fase 4C — UpscalingResult ──────────────────────────────────
//
// Bidirectional plumbing payload returned by an upscaling bridge after
// completing a sub-model homogenisation step. The macro model consumes
// this through `MaterialSection::set_homogenized_response(eps_ref,
// f_hom, D_hom)` (shim already in place per cyclic-validation-status).
//
// The `D_hom` block is computed primarily through bordered mixed-control
// (the principal mechanism per Plan v2 §Fase 4C); the optional Schur
// route remains a diagnostic comparator (§Fase 4-bis).
//
// Field semantics:
//   - eps_ref            : reference strain vector at which the
//                          homogenisation was sampled.
//   - f_hom              : homogenised generalised force vector.
//   - D_hom              : homogenised generalised tangent (square,
//                          ordered to match `f_hom`).
//   - frobenius_residual : ||K_macro_observed - D_hom||_F /
//                          ||D_hom||_F at the linearisation point;
//                          gate threshold for §Fase 4D guarded smoke
//                          (≤ 0.03 per Cap. 79).
//   - snes_iters         : SNES iterations consumed by the sub-step.
//   - converged          : true iff the sub-step converged within the
//                          guarded budget.
//   - status             : structured response classification.
//   - tangent_scheme     : how D_hom was obtained.
//   - condensed_status   : Schur condensation status (diagnostic).
//
// This is a plain-data struct — no virtual interfaces, no PETSc
// dependencies. Used as the return type of the upscaling bridge and
// (uncondensed) of the diagnostic Schur route.
struct UpscalingResult {
    Eigen::VectorXd eps_ref{};
    Eigen::VectorXd f_hom{};
    Eigen::MatrixXd D_hom{};

    double         frobenius_residual{0.0};
    std::size_t    snes_iters{0};
    bool           converged{false};

    ResponseStatus            status{ResponseStatus::NotReady};
    TangentLinearizationScheme tangent_scheme{TangentLinearizationScheme::Unknown};
    CondensedTangentStatus    condensed_status{CondensedTangentStatus::NotAttempted};

    [[nodiscard]] bool is_well_formed() const noexcept {
        const auto n = static_cast<Eigen::Index>(f_hom.size());
        if (n == 0) return false;
        if (eps_ref.size() != 0 && eps_ref.size() != n) return false;
        return D_hom.rows() == n && D_hom.cols() == n;
    }

    [[nodiscard]] bool passes_guarded_smoke_gate(
        double max_frobenius_residual = 0.03,
        std::size_t max_snes_iters = 6) const noexcept
    {
        return converged
            && status != ResponseStatus::SolveFailed
            && status != ResponseStatus::InvalidOperator
            && snes_iters <= max_snes_iters
            && frobenius_residual <= max_frobenius_residual
            && is_well_formed();
    }
};

}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MULTISCALE_TYPES_HH
