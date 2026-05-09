#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_ANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_ANALYSIS_HH

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "CouplingStrategy.hh"
#include "HomogenizedTangentFiniteDifference.hh"
#include "LocalSubproblemRuntime.hh"
#include "MicroSolveExecutor.hh"
#include "MultiscaleModel.hh"
#include "SteppableSolver.hh"
#include "../reconstruction/SectionOperatorValidationNorm.hh"

namespace fall_n {

template <CheckpointableSteppableSolver MacroSolverT,
          typename MacroBridgeT,
          LocalModelAdapter LocalModelT,
          typename ExecutorT = SerialExecutor>
class MultiscaleAnalysis {
    using ModelT = MultiscaleModel<MacroBridgeT, LocalModelT>;
    using MacroCheckpointT = typename std::remove_cvref_t<MacroSolverT>::checkpoint_type;
    using LocalCheckpointT = typename std::remove_cvref_t<LocalModelT>::checkpoint_type;

public:
    struct RestartBundle {
        MacroCheckpointT macro_checkpoint{};
        std::vector<LocalCheckpointT> local_checkpoints{};
        std::vector<SectionHomogenizedResponse> last_responses{};
        std::vector<SectionHomogenizedResponse> last_converged_responses{};
        std::size_t analysis_steps{0};
        bool valid{false};
    };

private:
    ModelT model_;
    MacroSolverT* macro_solver_{nullptr};
    ExecutorT executor_{};

    std::unique_ptr<CouplingAlgorithm>   algorithm_;
    std::unique_ptr<CouplingConvergence> convergence_;
    std::unique_ptr<RelaxationPolicy>    relaxation_;

    int    coupling_start_step_{0};
    double section_width_{0.30};
    double section_height_{0.30};
    double tangent_perturbation_{1.0e-6};
    TangentComputationMode local_tangent_computation_mode_{
        TangentComputationMode::PreferLinearizedCondensation};
    HomogenizedTangentFiniteDifferenceSettings
        local_fd_tangent_settings_{};
    TangentValidationNormKind force_residual_norm_{
        TangentValidationNormKind::StateWeightedFrobenius};
    TangentValidationNormKind tangent_residual_norm_{
        TangentValidationNormKind::StateWeightedFrobenius};
    bool predictor_admissibility_filter_enabled_{false};
    double predictor_min_symmetric_eigenvalue_{0.0};
    int predictor_admissibility_backtrack_attempts_{0};
    double predictor_admissibility_backtrack_factor_{0.5};
    int macro_step_cutback_attempts_{0};
    double macro_step_cutback_factor_{0.5};
    int macro_failure_backtrack_attempts_{0};
    double macro_failure_backtrack_factor_{0.5};
    SiteAdaptiveRelaxationSettings site_adaptive_relaxation_{};
    TwoWayFailureRecoveryPolicy two_way_failure_recovery_policy_{};
    int hybrid_window_steps_{0};
    int hybrid_success_steps_{0};
    std::size_t analysis_steps_{0};

    CouplingIterationReport last_report_{};
    std::vector<SectionHomogenizedResponse> last_responses_{};
    std::vector<SectionHomogenizedResponse> last_converged_responses_{};
    LocalSubproblemRuntimeManager<LocalModelT> local_runtime_{};

    [[nodiscard]] static double relative_norm_(
        const Eigen::Vector<double, 6>& a,
        const Eigen::Vector<double, 6>& b)
    {
        const double denom = std::max({1.0, a.norm(), b.norm()});
        return (a - b).norm() / denom;
    }

    [[nodiscard]] SectionOperatorValidationScales residual_scales_(
        TangentValidationNormKind norm,
        const MacroSectionState& macro_state) const
    {
        return make_section_operator_validation_scales(
            norm,
            section_width_,
            section_height_,
            macro_state.strain,
            macro_state.forces);
    }

    static void initialize_report_norms_(CouplingIterationReport& report,
                                         TangentValidationNormKind force_norm,
                                         TangentValidationNormKind tangent_norm)
    {
        report.force_residual_norm = force_norm;
        report.tangent_residual_norm = tangent_norm;
        report.attempted_state_valid = false;
        report.attempted_macro_step = 0;
        report.attempted_macro_time = 0.0;
        report.adaptive_relaxation_applied = false;
        report.adaptive_relaxation_attempts = 0;
        report.adaptive_relaxation_min_alpha = 1.0;
        report.predictor_admissibility_filter_applied = false;
        report.predictor_admissibility_satisfied = true;
        report.predictor_admissibility_attempts = 0;
        report.predictor_admissibility_last_alpha = 1.0;
        report.coupling_regime = CouplingRegime::StrictTwoWay;
        report.feedback_source = CouplingFeedbackSource::None;
        report.hybrid_active = false;
        report.hybrid_reason.clear();
        report.one_way_replay_status.clear();
        report.work_gap = 0.0;
        report.return_gate_passed = false;
        report.macro_step_cutback_attempts = 0;
        report.macro_step_cutback_succeeded = false;
        report.macro_step_cutback_last_factor = 1.0;
        report.macro_step_cutback_initial_increment = 0.0;
        report.macro_step_cutback_last_increment = 0.0;
        report.predictor_inadmissible_sites.clear();
        report.site_iteration_records.clear();
    }

    void capture_macro_solver_diagnostics_()
    {
        if constexpr (requires(const MacroSolverT& solver) {
                          { solver.converged_reason() };
                      }) {
            last_report_.macro_solver_reason =
                static_cast<int>(macro_solver_->converged_reason());
        }
        if constexpr (requires(const MacroSolverT& solver) {
                          { solver.num_iterations() };
                      }) {
            last_report_.macro_solver_iterations =
                static_cast<int>(macro_solver_->num_iterations());
        }
        if constexpr (requires(const MacroSolverT& solver) {
                          { solver.function_norm() } -> std::convertible_to<double>;
                      }) {
            last_report_.macro_solver_function_norm =
                macro_solver_->function_norm();
        }
    }

    void capture_attempted_macro_state_()
    {
        last_report_.attempted_state_valid = true;
        last_report_.attempted_macro_time = macro_solver_->current_time();
        if constexpr (requires(MacroSolverT& solver) {
                          { solver.current_step() } -> std::convertible_to<int>;
                      }) {
            last_report_.attempted_macro_step = macro_solver_->current_step();
        }
    }

    [[nodiscard]] double force_residual_metric_(
        const MacroSectionState& macro_state,
        const SectionHomogenizedResponse& response,
        std::array<double, 6>* component_scales = nullptr,
        double* max_component_gap = nullptr) const
    {
        const auto scales = residual_scales_(force_residual_norm_, macro_state);
        const auto metrics = compute_section_vector_validation_metrics(
            macro_state.forces, response.forces, scales);
        if (component_scales) {
            *component_scales = metrics.component_scales;
        }
        if (max_component_gap) {
            *max_component_gap = metrics.max_component_gap;
        }
        return metrics.relative_gap;
    }

    [[nodiscard]] double tangent_residual_metric_(
        const MacroSectionState& macro_state,
        const SectionHomogenizedResponse& current,
        const SectionHomogenizedResponse& previous,
        std::array<double, 6>* row_scales = nullptr,
        std::array<double, 6>* column_scales = nullptr,
        double* max_column_gap = nullptr) const
    {
        const auto scales =
            residual_scales_(tangent_residual_norm_, macro_state);
        const auto metrics = compute_section_operator_validation_metrics(
            current.tangent, previous.tangent, scales);
        if (row_scales) {
            *row_scales = metrics.row_scales;
        }
        if (column_scales) {
            *column_scales = metrics.column_scales;
        }
        if (max_column_gap) {
            *max_column_gap = metrics.max_column_gap;
        }
        return metrics.relative_gap;
    }

    [[nodiscard]] static double relative_norm_(
        const Eigen::Matrix<double, 6, 6>& a,
        const Eigen::Matrix<double, 6, 6>& b)
    {
        const double denom = std::max({1.0, a.norm(), b.norm()});
        return (a - b).norm() / denom;
    }

    [[nodiscard]] static bool is_hard_failure_(ResponseStatus status)
    {
        return status == ResponseStatus::NotReady
            || status == ResponseStatus::SolveFailed
            || status == ResponseStatus::InvalidOperator;
    }

    static void accumulate_response_diagnostics_(
        CouplingIterationReport& report,
        const SectionHomogenizedResponse& response)
    {
        if (is_hard_failure_(response.status)) {
            ++report.failed_submodels;
            report.failed_sites.push_back(response.site);
        }

        if (response.tangent_regularized) {
            ++report.regularized_submodels;
            report.regularization_detected = true;
        }
    }

    [[nodiscard]] static double max_force_gap_(
        const std::vector<SectionHomogenizedResponse>& a,
        const std::vector<SectionHomogenizedResponse>& b)
    {
        if (a.size() != b.size()) {
            return std::numeric_limits<double>::infinity();
        }

        double gap = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double denom =
                std::max({1.0, a[i].forces.norm(), b[i].forces.norm()});
            gap = std::max(gap, (a[i].forces - b[i].forces).norm() / denom);
        }
        return gap;
    }

    [[nodiscard]] bool is_predictor_response_admissible_(
        const SectionHomogenizedResponse& response) const
    {
        return response.tangent_min_symmetric_eigenvalue
            >= predictor_min_symmetric_eigenvalue_;
    }

    [[nodiscard]] bool predictor_is_admissible_(
        const std::vector<SectionHomogenizedResponse>& responses,
        std::vector<CouplingSite>* inadmissible_sites = nullptr) const
    {
        bool admissible = true;
        if (inadmissible_sites) {
            inadmissible_sites->clear();
        }

        for (const auto& response : responses) {
            if (!is_predictor_response_admissible_(response)) {
                admissible = false;
                if (inadmissible_sites) {
                    inadmissible_sites->push_back(response.site);
                }
            }
        }
        return admissible;
    }

    [[nodiscard]] std::vector<SectionHomogenizedResponse>
    predictor_admissibility_baseline_(
        const std::vector<SectionHomogenizedResponse>& predictor,
        const std::vector<SectionHomogenizedResponse>& baseline) const
    {
        if (baseline.size() == model_.num_local_models()) {
            return baseline;
        }

        if (predictor.size() != model_.num_local_models()) {
            return {};
        }

        auto zero_baseline = predictor;
        for (auto& response : zero_baseline) {
            response.forces.setZero();
            response.tangent.setZero();
            response.forces_consistent_with_tangent = true;
            refresh_section_operator_diagnostics(response);
        }
        return zero_baseline;
    }

    void apply_predictor_admissibility_filter_(
        std::vector<SectionHomogenizedResponse>& predictor,
        const std::vector<SectionHomogenizedResponse>& baseline)
    {
        last_report_.predictor_inadmissible_sites.clear();
        last_report_.predictor_admissibility_satisfied = true;

        if (!predictor_admissibility_filter_enabled_
            || predictor.size() != model_.num_local_models())
        {
            return;
        }

        if (predictor_is_admissible_(
                predictor, &last_report_.predictor_inadmissible_sites))
        {
            return;
        }

        const auto original_inadmissible_sites =
            last_report_.predictor_inadmissible_sites;

        last_report_.predictor_admissibility_filter_applied = true;
        last_report_.predictor_admissibility_satisfied = false;

        if (baseline.size() != model_.num_local_models()) {
            return;
        }

        auto filtered = predictor;
        for (int attempt = 1;
             attempt <= predictor_admissibility_backtrack_attempts_;
             ++attempt)
        {
            const double alpha = std::pow(
                predictor_admissibility_backtrack_factor_, attempt);
            filtered = predictor;
            for (std::size_t i = 0; i < filtered.size(); ++i) {
                blend_section_response(filtered[i], baseline[i], alpha);
            }
            last_report_.predictor_admissibility_attempts = attempt;
            last_report_.predictor_admissibility_last_alpha = alpha;

            if (predictor_is_admissible_(
                    filtered, &last_report_.predictor_inadmissible_sites))
            {
                predictor = std::move(filtered);
                last_report_.predictor_inadmissible_sites =
                    original_inadmissible_sites;
                last_report_.predictor_admissibility_satisfied = true;
                return;
            }
        }

        predictor = std::move(filtered);
        last_report_.predictor_inadmissible_sites = original_inadmissible_sites;
    }

    void set_macro_trial_mode_(bool enabled)
    {
        if constexpr (TrialControllableSolver<MacroSolverT>) {
            macro_solver_->set_auto_commit(!enabled);
        }

        if constexpr (requires(MacroSolverT& solver, bool flag) {
                          solver.set_observer_notifications(flag);
                      }) {
            macro_solver_->set_observer_notifications(!enabled);
        }
    }

    void restore_previous_injection_()
    {
        if (last_converged_responses_.size() == model_.num_local_models()) {
            for (const auto& response : last_converged_responses_) {
                model_.macro_bridge().inject_response(response);
            }
            return;
        }

        for (const auto& site : model_.sites()) {
            model_.macro_bridge().clear_response(site);
        }
    }

    [[nodiscard]] bool should_couple_this_step_() const
    {
        return model_.num_local_models() > 0
            && static_cast<int>(analysis_steps_) + 1 >= coupling_start_step_;
    }

    void sync_local_runtime_report_()
    {
        local_runtime_.populate_report(last_report_);
    }

    void finalize_local_models_(
        double time,
        const std::vector<SectionHomogenizedResponse>* accepted_responses =
            nullptr)
    {
        local_runtime_.resize(model_.num_local_models());
        for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
            auto& local_model = model_.local_models()[i];
            local_model.commit_trial_state();
            local_model.set_auto_commit(true);
            local_model.end_of_step(time);
            if (accepted_responses != nullptr &&
                i < accepted_responses->size())
            {
                auto macro_state =
                    model_.macro_bridge().extract_section_state(
                        model_.site(i));
                local_runtime_.save_accepted_state(
                    i,
                    local_model,
                    (*accepted_responses)[i],
                    macro_state);
            }
        }
    }

    void inject_or_clear_(
        const std::vector<SectionHomogenizedResponse>& responses)
    {
        if (responses.size() == model_.num_local_models()) {
            for (const auto& response : responses) {
                model_.macro_bridge().inject_response(response);
            }
            return;
        }

        for (const auto& site : model_.sites()) {
            model_.macro_bridge().clear_response(site);
        }
    }

    void configure_local_tangent_policy_(LocalModelT& local_model) const
    {
        if constexpr (requires(LocalModelT& lm, TangentComputationMode mode) {
                          lm.set_tangent_computation_mode(mode);
                      }) {
            local_model.set_tangent_computation_mode(
                local_tangent_computation_mode_);
        }
        if constexpr (requires(
                          LocalModelT& lm,
                          HomogenizedTangentFiniteDifferenceSettings settings) {
                          lm.set_finite_difference_tangent_settings(settings);
                      }) {
            local_model.set_finite_difference_tangent_settings(
                local_fd_tangent_settings_);
        }
    }

    [[nodiscard]] bool macro_solver_supports_increment_control_() const
    {
        if constexpr (requires(MacroSolverT& solver, double dp) {
                          { solver.get_increment_size() }
                              -> std::convertible_to<double>;
                          solver.set_increment_size(dp);
                      }) {
            return true;
        } else {
            return false;
        }
    }

    void solve_locals_once_(double time,
                            std::vector<SectionHomogenizedResponse>& responses,
                            int& failed_submodels)
    {
        local_runtime_.resize(model_.num_local_models());
        local_runtime_.reset_records();
        responses.resize(model_.num_local_models());
        failed_submodels = 0;
        last_report_.failed_submodels = 0;
        last_report_.local_failure_messages.assign(
            model_.num_local_models(), {});

        executor_.for_each(model_.num_local_models(), [&](std::size_t i) {
            auto response = SectionHomogenizedResponse{};
            response.site = model_.site(i);
            try {
                auto& local_model = model_.local_models()[i];
                const auto macro_state =
                    model_.macro_bridge().extract_section_state(
                        model_.site(i));
                if (!local_runtime_.should_solve(i, macro_state)) {
                    responses[i] = local_runtime_.inactive_response(
                        i,
                        model_.site(i),
                        macro_state);
                    return;
                }

                const bool restored_seed =
                    local_runtime_.restore_seed_before_solve(i, local_model);
                local_model.set_auto_commit(true);
                configure_local_tangent_policy_(local_model);

                const auto ek =
                    model_.macro_bridge().extract_element_kinematics(
                        model_.site(i).macro_element_id);
                local_model.update_kinematics(ek.kin_A, ek.kin_B);
                if constexpr (requires(LocalModelT& lm,
                                        const MacroSectionState& state) {
                                  lm.update_macro_section_state(state);
                              }) {
                    local_model.update_macro_section_state(macro_state);
                }

                const auto solve_t0 = std::chrono::steady_clock::now();
                auto result = local_model.solve_step(time);
                const auto solve_t1 = std::chrono::steady_clock::now();
                response = local_model.section_response(
                    section_width_, section_height_, tangent_perturbation_);
                response.site = model_.site(i);
                refresh_section_operator_diagnostics(response);
                bool converged = true;
                if constexpr (requires { result.converged; }) {
                    if (!result.converged) {
                        response.status = ResponseStatus::SolveFailed;
                        converged = false;
                    }
                }
                (void)restored_seed;
                local_runtime_.record_solve_attempt(
                    i,
                    std::chrono::duration<double>(
                        solve_t1 - solve_t0).count(),
                    converged);
            } catch (const std::exception& ex) {
                if (i < last_report_.local_failure_messages.size()) {
                    last_report_.local_failure_messages[i] = ex.what();
                }
                response.status = ResponseStatus::SolveFailed;
                local_runtime_.record_solve_attempt(i, 0.0, false);
            } catch (...) {
                if (i < last_report_.local_failure_messages.size()) {
                    last_report_.local_failure_messages[i] =
                        "unknown local-model exception";
                }
                response.status = ResponseStatus::SolveFailed;
                local_runtime_.record_solve_attempt(i, 0.0, false);
            }
            responses[i] = response;
        });

        last_report_.failed_sites.clear();
        last_report_.regularized_submodels = 0;
        last_report_.regularization_detected = false;
        for (const auto& response : responses) {
            accumulate_response_diagnostics_(last_report_, response);
        }
        sync_local_runtime_report_();
        failed_submodels = last_report_.failed_submodels;
    }

    bool perform_macro_only_step_()
    {
        set_macro_trial_mode_(false);

        last_report_ = CouplingIterationReport{};
        initialize_report_norms_(
            last_report_, force_residual_norm_, tangent_residual_norm_);
        last_responses_.clear();
        last_report_.mode = algorithm_->mode();
        last_report_.iterations = 1;
        last_report_.termination_reason =
            CouplingTerminationReason::UncoupledMacroStep;

        const bool ok = macro_solver_->step();
        capture_macro_solver_diagnostics_();
        if (ok) {
            capture_attempted_macro_state_();
        }
        last_report_.converged = ok;
        if (!ok) {
            last_report_.termination_reason =
                CouplingTerminationReason::MacroSolveFailed;
        }
        if (ok) {
            ++analysis_steps_;
        }
        return ok;
    }

    bool perform_one_way_downscaling_()
    {
        last_report_ = CouplingIterationReport{};
        initialize_report_norms_(
            last_report_, force_residual_norm_, tangent_residual_norm_);
        last_responses_.clear();
        last_report_.mode = CouplingMode::OneWayDownscaling;
        last_report_.coupling_regime = CouplingRegime::OneWayOnly;
        last_report_.feedback_source = CouplingFeedbackSource::ClearedOneWay;
        last_report_.one_way_replay_status = "active";
        last_report_.iterations = 1;
        last_report_.termination_reason =
            CouplingTerminationReason::OneWayStepCompleted;

        set_macro_trial_mode_(false);
        inject_or_clear_({});
        if (!macro_solver_->step()) {
            capture_macro_solver_diagnostics_();
            last_report_.converged = false;
            last_report_.termination_reason =
                CouplingTerminationReason::MacroSolveFailed;
            return false;
        }
        capture_macro_solver_diagnostics_();
        capture_attempted_macro_state_();

        std::vector<LocalCheckpointT> local_checkpoints;
        local_checkpoints.reserve(model_.num_local_models());
        for (auto& local_model : model_.local_models()) {
            local_checkpoints.push_back(local_model.capture_checkpoint());
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<SectionHomogenizedResponse> responses;
        solve_locals_once_(macro_solver_->current_time(),
                           responses,
                           last_report_.failed_submodels);
        last_responses_ = responses;
        if (last_report_.failed_submodels == 0) {
            finalize_local_models_(macro_solver_->current_time(), &responses);
        } else {
            for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
                model_.local_models()[i].restore_checkpoint(
                    local_checkpoints[i]);
                model_.local_models()[i].set_auto_commit(true);
            }
        }
        sync_local_runtime_report_();
        auto t1 = std::chrono::steady_clock::now();

        last_report_.micro_solve_seconds =
            std::chrono::duration<double>(t1 - t0).count();
        last_report_.converged = (last_report_.failed_submodels == 0);
        if (!last_report_.converged) {
            last_report_.termination_reason =
                CouplingTerminationReason::MicroSolveFailed;
        }
        if (last_report_.converged) {
            ++analysis_steps_;
        }
        return last_report_.converged;
    }

    bool perform_lagged_feedback_()
    {
        last_report_ = CouplingIterationReport{};
        initialize_report_norms_(
            last_report_, force_residual_norm_, tangent_residual_norm_);
        last_responses_.clear();
        last_report_.mode = CouplingMode::LaggedFeedbackCoupling;
        last_report_.iterations = 1;
        last_report_.termination_reason =
            CouplingTerminationReason::LaggedStepCompleted;

        set_macro_trial_mode_(false);
        if (!macro_solver_->step()) {
            capture_macro_solver_diagnostics_();
            last_report_.converged = false;
            last_report_.termination_reason =
                CouplingTerminationReason::MacroSolveFailed;
            return false;
        }
        capture_macro_solver_diagnostics_();
        capture_attempted_macro_state_();

        auto t0 = std::chrono::steady_clock::now();
        std::vector<SectionHomogenizedResponse> responses;
        solve_locals_once_(macro_solver_->current_time(),
                           responses,
                           last_report_.failed_submodels);
        last_responses_ = responses;

        last_report_.force_residuals_rel.assign(model_.num_local_models(), 0.0);
        last_report_.force_component_residuals_rel.assign(
            model_.num_local_models(), 0.0);
        last_report_.tangent_residuals_rel.assign(model_.num_local_models(), 0.0);
        last_report_.tangent_column_residuals_rel.assign(
            model_.num_local_models(), 0.0);
        last_report_.tangent_min_symmetric_eigenvalues.assign(
            model_.num_local_models(), 0.0);
        last_report_.tangent_max_symmetric_eigenvalues.assign(
            model_.num_local_models(), 0.0);
        last_report_.tangent_traces.assign(
            model_.num_local_models(), 0.0);
        last_report_.tangent_nonpositive_diagonal_counts.assign(
            model_.num_local_models(), 0);
        last_report_.force_residual_component_scales.assign(
            model_.num_local_models(),
            std::array<double, 6>{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}});
        last_report_.tangent_residual_row_scales.assign(
            model_.num_local_models(),
            std::array<double, 6>{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}});
        last_report_.tangent_residual_column_scales.assign(
            model_.num_local_models(),
            std::array<double, 6>{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}});

        for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
            auto macro_state =
                model_.macro_bridge().extract_section_state(model_.site(i));
            responses[i].strain_ref = macro_state.strain;
            model_.macro_bridge().inject_response(responses[i]);
            last_report_.force_residuals_rel[i] = force_residual_metric_(
                macro_state,
                responses[i],
                &last_report_.force_residual_component_scales[i],
                &last_report_.force_component_residuals_rel[i]);
            last_report_.tangent_min_symmetric_eigenvalues[i] =
                responses[i].tangent_min_symmetric_eigenvalue;
            last_report_.tangent_max_symmetric_eigenvalues[i] =
                responses[i].tangent_max_symmetric_eigenvalue;
            last_report_.tangent_traces[i] =
                responses[i].tangent_trace;
            last_report_.tangent_nonpositive_diagonal_counts[i] =
                responses[i].tangent_nonpositive_diagonal_entries;
            last_report_.max_force_residual_rel =
                std::max(last_report_.max_force_residual_rel,
                         last_report_.force_residuals_rel[i]);
            last_report_.max_force_component_residual_rel =
                std::max(last_report_.max_force_component_residual_rel,
                         last_report_.force_component_residuals_rel[i]);
        }

        finalize_local_models_(macro_solver_->current_time(), &responses);
        sync_local_runtime_report_();
        auto t1 = std::chrono::steady_clock::now();

        last_converged_responses_ = responses;
        last_report_.micro_solve_seconds =
            std::chrono::duration<double>(t1 - t0).count();
        last_report_.converged = (last_report_.failed_submodels == 0);
        if (!last_report_.converged) {
            last_report_.termination_reason =
                CouplingTerminationReason::MicroSolveFailed;
        }
        if (last_report_.converged) {
            ++analysis_steps_;
        }
        return last_report_.converged;
    }

    bool perform_iterated_two_way_fe2_()
    {
        static_assert(TrialControllableSolver<MacroSolverT>,
            "IteratedTwoWayFE2 requires a macro solver with trial-commit control");

        if (two_way_failure_recovery_policy_.mode ==
            TwoWayFailureRecoveryMode::OneWayOnly)
        {
            const bool ok = perform_one_way_downscaling_();
            last_report_.coupling_regime = CouplingRegime::OneWayOnly;
            last_report_.feedback_source = CouplingFeedbackSource::ClearedOneWay;
            last_report_.hybrid_active = false;
            last_report_.one_way_replay_status =
                ok ? "one_way_only_completed" : "one_way_only_failed";
            return ok;
        }

        last_report_ = CouplingIterationReport{};
        initialize_report_norms_(
            last_report_, force_residual_norm_, tangent_residual_norm_);
        last_responses_.clear();
        last_report_.mode = CouplingMode::IteratedTwoWayFE2;
        last_report_.coupling_regime = CouplingRegime::StrictTwoWay;
        last_report_.feedback_source = CouplingFeedbackSource::CurrentHomogenized;
        last_report_.hybrid_window_steps = hybrid_window_steps_;
        last_report_.hybrid_success_steps = hybrid_success_steps_;
        last_report_.iterations = 0;
        last_report_.termination_reason = CouplingTerminationReason::NotRun;

        const auto macro_checkpoint = macro_solver_->capture_checkpoint();
        const double macro_step_start_time = macro_solver_->current_time();
        double macro_trial_increment = 0.0;
        bool macro_trial_increment_available = false;
        if constexpr (requires(MacroSolverT& solver) {
                          { solver.get_increment_size() }
                              -> std::convertible_to<double>;
                      }) {
            macro_trial_increment = macro_solver_->get_increment_size();
            macro_trial_increment_available = macro_trial_increment > 0.0;
        }
        const double macro_nominal_increment = macro_trial_increment;
        const double macro_step_target_time =
            macro_step_start_time
            + (macro_trial_increment_available ? macro_trial_increment : 0.0);
        std::vector<LocalCheckpointT> local_checkpoints;
        local_runtime_.resize(model_.num_local_models());
        local_runtime_.reset_records();
        local_checkpoints.reserve(model_.num_local_models());
        for (auto& local_model : model_.local_models()) {
            local_model.set_auto_commit(false);
            local_checkpoints.push_back(local_model.capture_checkpoint());
        }

        std::vector<SectionHomogenizedResponse> predictor = last_converged_responses_;
        std::vector<SectionHomogenizedResponse> current(model_.num_local_models());
        std::vector<SectionHomogenizedResponse> macro_success_baseline =
            last_converged_responses_;

        set_macro_trial_mode_(true);

        bool accepted = false;
        const int max_iter = std::max(2, algorithm_->max_iterations());
        std::vector<double> previous_iteration_force_residuals(
            model_.num_local_models(),
            std::numeric_limits<double>::quiet_NaN());

        for (int iter = 0; iter < max_iter; ++iter) {
            last_report_.iterations = iter + 1;

            std::vector<SectionHomogenizedResponse> active_predictor = predictor;
            const auto predictor_baseline =
                predictor_admissibility_baseline_(
                    active_predictor, macro_success_baseline);
            apply_predictor_admissibility_filter_(
                active_predictor, predictor_baseline);
            auto try_macro_step = [&](
                                      const std::vector<SectionHomogenizedResponse>&
                                          candidate,
                                      double requested_increment) -> bool {
                macro_solver_->restore_checkpoint(macro_checkpoint);
                if constexpr (requires(MacroSolverT& solver, double dp) {
                                  solver.set_increment_size(dp);
                              }) {
                    if (requested_increment > 0.0) {
                        macro_solver_->set_increment_size(requested_increment);
                    }
                }
                inject_or_clear_(candidate);
                const auto macro_t0 = std::chrono::steady_clock::now();
                bool ok = false;
                if constexpr (requires(MacroSolverT& solver, double dp) {
                                  { solver.get_increment_size() }
                                      -> std::convertible_to<double>;
                                  solver.set_increment_size(dp);
                              }) {
                    if (requested_increment > 0.0
                        && macro_step_target_time > macro_step_start_time)
                    {
                        const auto verdict =
                            macro_solver_->step_to(macro_step_target_time);
                        ok = verdict != StepVerdict::Stop
                            && macro_solver_->current_time()
                                >= macro_step_target_time - 1.0e-12;
                    } else {
                        ok = macro_solver_->step();
                    }
                } else {
                    ok = macro_solver_->step();
                }
                const auto macro_t1 = std::chrono::steady_clock::now();
                last_report_.macro_solve_seconds +=
                    std::chrono::duration<double>(macro_t1 - macro_t0).count();
                capture_macro_solver_diagnostics_();
                return ok;
            };
            auto try_macro_with_cutback =
                [&](const std::vector<SectionHomogenizedResponse>& candidate)
                    -> bool {
                const double initial_increment = macro_trial_increment;
                if (macro_trial_increment_available) {
                    last_report_.macro_step_cutback_initial_increment =
                        macro_nominal_increment;
                    last_report_.macro_step_cutback_last_increment =
                        initial_increment;
                }

                bool ok = try_macro_step(candidate, macro_trial_increment);
                if (ok) {
                    return true;
                }

                if (!macro_trial_increment_available
                    || macro_step_cutback_attempts_ <= 0)
                {
                    return false;
                }

                for (int attempt = 1;
                     attempt <= macro_step_cutback_attempts_;
                     ++attempt)
                {
                    const double factor =
                        std::pow(macro_step_cutback_factor_, attempt);
                    const double reduced_increment =
                        initial_increment * factor;
                    last_report_.macro_step_cutback_attempts = attempt;
                    last_report_.macro_step_cutback_last_factor = factor;
                    last_report_.macro_step_cutback_last_increment =
                        reduced_increment;
                    ok = try_macro_step(candidate, reduced_increment);
                    if (ok) {
                        macro_trial_increment = reduced_increment;
                        last_report_.macro_step_cutback_succeeded = true;
                        return true;
                    }
                }
                return false;
            };

            bool macro_ok = try_macro_with_cutback(active_predictor);

            auto macro_backtrack_baseline = macro_success_baseline;
            if (macro_backtrack_baseline.size() != model_.num_local_models()
                && active_predictor.size() == model_.num_local_models())
            {
                macro_backtrack_baseline = active_predictor;
                for (auto& response : macro_backtrack_baseline) {
                    response.forces.setZero();
                    response.tangent.setZero();
                    response.forces_consistent_with_tangent = true;
                }
            }

            if (!macro_ok
                && active_predictor.size() == model_.num_local_models()
                && macro_backtrack_baseline.size() == model_.num_local_models()
                && macro_failure_backtrack_attempts_ > 0)
            {
                for (int attempt = 1;
                     attempt <= macro_failure_backtrack_attempts_;
                     ++attempt)
                {
                    last_report_.macro_backtracking_attempts = attempt;
                    last_report_.macro_backtracking_last_alpha =
                        std::pow(macro_failure_backtrack_factor_, attempt);
                    active_predictor = predictor;
                    for (std::size_t i = 0; i < active_predictor.size(); ++i) {
                        blend_section_response(
                            active_predictor[i],
                            macro_backtrack_baseline[i],
                            last_report_.macro_backtracking_last_alpha);
                    }
                    macro_ok = try_macro_with_cutback(active_predictor);
                    if (macro_ok) {
                        last_report_.macro_backtracking_succeeded = true;
                        break;
                    }
                }
            }

            if (!macro_ok) {
                macro_solver_->restore_checkpoint(macro_checkpoint);
                for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
                    model_.local_models()[i].restore_checkpoint(local_checkpoints[i]);
                    model_.local_models()[i].set_auto_commit(true);
                }
                last_report_.converged = false;
                last_report_.rollback_performed = true;
                last_report_.termination_reason =
                    CouplingTerminationReason::MacroSolveFailed;
                restore_previous_injection_();
                set_macro_trial_mode_(false);
                return false;
            }
            capture_attempted_macro_state_();
            macro_success_baseline = active_predictor;

            const auto micro_t0 = std::chrono::steady_clock::now();
            executor_.for_each(model_.num_local_models(), [&](std::size_t i) {
                current[i] = SectionHomogenizedResponse{};
                current[i].site = model_.site(i);
                try {
                    const auto macro_state =
                        model_.macro_bridge().extract_section_state(
                            model_.site(i));
                    if (!local_runtime_.should_solve(i, macro_state)) {
                        current[i] = local_runtime_.inactive_response(
                            i,
                            model_.site(i),
                            macro_state);
                        return;
                    }

                    auto& local_model = model_.local_models()[i];
                    local_model.restore_checkpoint(local_checkpoints[i]);
                    configure_local_tangent_policy_(local_model);

                    const auto ek =
                        model_.macro_bridge().extract_element_kinematics(
                            model_.site(i).macro_element_id);
                    local_model.update_kinematics(ek.kin_A, ek.kin_B);
                    if constexpr (requires(LocalModelT& lm,
                                            const MacroSectionState& state) {
                                      lm.update_macro_section_state(state);
                                  }) {
                        local_model.update_macro_section_state(macro_state);
                    }

                    const auto solve_t0 = std::chrono::steady_clock::now();
                    auto result =
                        local_model.solve_step(macro_solver_->current_time());
                    const auto solve_t1 = std::chrono::steady_clock::now();
                    current[i] = local_model.section_response(
                        section_width_, section_height_, tangent_perturbation_);
                    current[i].site = model_.site(i);
                    refresh_section_operator_diagnostics(current[i]);
                    bool converged = true;
                    if constexpr (requires { result.converged; }) {
                        if (!result.converged) {
                            current[i].status = ResponseStatus::SolveFailed;
                            converged = false;
                        }
                    }
                    local_runtime_.record_solve_attempt(
                        i,
                        std::chrono::duration<double>(
                            solve_t1 - solve_t0).count(),
                        converged);
                } catch (...) {
                    current[i].status = ResponseStatus::SolveFailed;
                    local_runtime_.record_solve_attempt(i, 0.0, false);
                }
            });
            last_responses_ = current;
            const auto micro_t1 = std::chrono::steady_clock::now();
            last_report_.micro_solve_seconds +=
                std::chrono::duration<double>(micro_t1 - micro_t0).count();
            sync_local_runtime_report_();

            last_report_.failed_submodels = 0;
            last_report_.regularized_submodels = 0;
            last_report_.regularization_detected = false;
            last_report_.failed_sites.clear();
            last_report_.force_residuals_rel.assign(model_.num_local_models(), 0.0);
            last_report_.force_component_residuals_rel.assign(
                model_.num_local_models(), 0.0);
            last_report_.tangent_residuals_rel.assign(model_.num_local_models(), 0.0);
            last_report_.tangent_column_residuals_rel.assign(
                model_.num_local_models(), 0.0);
            last_report_.tangent_min_symmetric_eigenvalues.assign(
                model_.num_local_models(), 0.0);
            last_report_.tangent_max_symmetric_eigenvalues.assign(
                model_.num_local_models(), 0.0);
            last_report_.tangent_traces.assign(
                model_.num_local_models(), 0.0);
            last_report_.tangent_nonpositive_diagonal_counts.assign(
                model_.num_local_models(), 0);
            last_report_.force_residual_component_scales.assign(
                model_.num_local_models(),
                std::array<double, 6>{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}});
            last_report_.tangent_residual_row_scales.assign(
                model_.num_local_models(),
                std::array<double, 6>{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}});
            last_report_.tangent_residual_column_scales.assign(
                model_.num_local_models(),
                std::array<double, 6>{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0}});
            last_report_.max_force_residual_rel = 0.0;
            last_report_.max_force_component_residual_rel = 0.0;
            last_report_.max_tangent_residual_rel = 0.0;
            last_report_.max_tangent_column_residual_rel = 0.0;
            std::vector<double> next_iteration_force_residuals(
                model_.num_local_models(),
                std::numeric_limits<double>::quiet_NaN());

            for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
                auto macro_state =
                    model_.macro_bridge().extract_section_state(model_.site(i));
                current[i].strain_ref = macro_state.strain;
                current[i].site.local_frame = macro_state.site.local_frame;
                int adaptive_attempts = 0;
                double adaptive_alpha = 1.0;
                bool adaptive_applied = false;

                auto update_tangent_metrics = [&]() {
                    if (active_predictor.size() == model_.num_local_models()) {
                        last_report_.tangent_residuals_rel[i] =
                            tangent_residual_metric_(
                                macro_state,
                                current[i],
                                active_predictor[i],
                                &last_report_.tangent_residual_row_scales[i],
                                &last_report_.tangent_residual_column_scales[i],
                                &last_report_.tangent_column_residuals_rel[i]);
                    }
                };
                auto update_force_metrics = [&]() {
                    last_report_.force_residuals_rel[i] =
                        force_residual_metric_(
                            macro_state,
                            current[i],
                            &last_report_.force_residual_component_scales[i],
                            &last_report_.force_component_residuals_rel[i]);
                };

                if (active_predictor.size() == model_.num_local_models()) {
                    const auto tangent_before = current[i].tangent;
                    const auto force_before = current[i].forces;
                    relaxation_->relax(current[i], active_predictor[i], iter);
                    if ((current[i].tangent - tangent_before).norm() > 0.0
                        || (current[i].forces - force_before).norm() > 0.0)
                    {
                        last_report_.relaxation_applied = true;
                    }
                    update_tangent_metrics();
                } else {
                    // Dampen the very first FE2 predictor as well. Otherwise
                    // the first macro re-solve sees the full micro feedback in
                    // one shot, and the relaxation policy has no chance to act
                    // before a macro divergence.
                    const auto tangent_before = current[i].tangent;
                    const auto force_before = current[i].forces;
                    auto zero_baseline = current[i];
                    zero_baseline.forces.setZero();
                    zero_baseline.tangent.setZero();
                    relaxation_->relax(current[i], zero_baseline, iter);
                    if ((current[i].tangent - tangent_before).norm() > 0.0
                        || (current[i].forces - force_before).norm() > 0.0)
                    {
                        last_report_.relaxation_applied = true;
                    }
                }

                update_force_metrics();

                const double previous_force_residual =
                    i < previous_iteration_force_residuals.size()
                        ? previous_iteration_force_residuals[i]
                        : std::numeric_limits<double>::quiet_NaN();
                if (site_adaptive_relaxation_.enabled
                    && active_predictor.size() == model_.num_local_models()
                    && iter > 0
                    && std::isfinite(previous_force_residual)
                    && previous_force_residual > 0.0
                    && last_report_.force_residuals_rel[i]
                        > site_adaptive_relaxation_.residual_growth_limit
                        * previous_force_residual)
                {
                    SectionHomogenizedResponse best = current[i];
                    double best_force_residual =
                        last_report_.force_residuals_rel[i];
                    const auto once_relaxed = current[i];
                    const auto growth_cap =
                        site_adaptive_relaxation_.residual_growth_limit
                        * previous_force_residual;

                    for (int attempt = 1;
                         attempt <= site_adaptive_relaxation_
                                        .max_backtracking_attempts;
                         ++attempt)
                    {
                        auto trial = once_relaxed;
                        double alpha = std::pow(
                            site_adaptive_relaxation_.backtracking_factor,
                            attempt);
                        alpha = std::max(
                            site_adaptive_relaxation_.min_alpha, alpha);
                        blend_section_response(
                            trial, active_predictor[i], alpha);
                        const double trial_force_residual =
                            force_residual_metric_(macro_state, trial);
                        if (trial_force_residual < best_force_residual) {
                            best = trial;
                            best_force_residual = trial_force_residual;
                            adaptive_attempts = attempt;
                            adaptive_alpha = alpha;
                            adaptive_applied = true;
                            if (trial_force_residual <= growth_cap) {
                                break;
                            }
                        }
                        if (alpha <= site_adaptive_relaxation_.min_alpha) {
                            break;
                        }
                    }

                    if (adaptive_applied) {
                        current[i] = best;
                        update_tangent_metrics();
                        update_force_metrics();
                        last_report_.adaptive_relaxation_applied = true;
                        last_report_.adaptive_relaxation_attempts +=
                            adaptive_attempts;
                        last_report_.adaptive_relaxation_min_alpha =
                            std::min(
                                last_report_.adaptive_relaxation_min_alpha,
                                adaptive_alpha);
                    }
                }

                last_report_.tangent_min_symmetric_eigenvalues[i] =
                    current[i].tangent_min_symmetric_eigenvalue;
                last_report_.tangent_max_symmetric_eigenvalues[i] =
                    current[i].tangent_max_symmetric_eigenvalue;
                last_report_.tangent_traces[i] =
                    current[i].tangent_trace;
                last_report_.tangent_nonpositive_diagonal_counts[i] =
                    current[i].tangent_nonpositive_diagonal_entries;

                accumulate_response_diagnostics_(last_report_, current[i]);

                last_report_.max_force_residual_rel =
                    std::max(last_report_.max_force_residual_rel,
                             last_report_.force_residuals_rel[i]);
                last_report_.max_force_component_residual_rel =
                    std::max(last_report_.max_force_component_residual_rel,
                             last_report_.force_component_residuals_rel[i]);
                last_report_.max_tangent_residual_rel =
                    std::max(last_report_.max_tangent_residual_rel,
                             last_report_.tangent_residuals_rel[i]);
                last_report_.max_tangent_column_residual_rel =
                    std::max(last_report_.max_tangent_column_residual_rel,
                             last_report_.tangent_column_residuals_rel[i]);

                append_site_iteration_record(
                    last_report_,
                    iter + 1,
                    i,
                    macro_state,
                    current[i],
                    previous_force_residual,
                    adaptive_applied,
                    adaptive_attempts,
                    adaptive_alpha);
                next_iteration_force_residuals[i] =
                    last_report_.force_residuals_rel[i];
            }

            previous_iteration_force_residuals =
                std::move(next_iteration_force_residuals);

            if (last_report_.failed_submodels > 0) {
                break;
            }

            predictor = current;

            if (iter == 0) {
                continue;
            }

            if (last_report_.failed_submodels == 0
                && convergence_->converged(last_report_))
            {
                accepted = true;
                break;
            }
        }

        auto try_hybrid_observation_window = [&]() -> bool {
            if (two_way_failure_recovery_policy_.mode !=
                TwoWayFailureRecoveryMode::HybridObservationWindow)
            {
                return false;
            }
            if (two_way_failure_recovery_policy_.max_hybrid_steps > 0 &&
                hybrid_window_steps_ >=
                    two_way_failure_recovery_policy_.max_hybrid_steps)
            {
                last_report_.hybrid_reason =
                    "hybrid_max_steps_exhausted";
                return false;
            }

            macro_solver_->restore_checkpoint(macro_checkpoint);
            for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
                model_.local_models()[i].restore_checkpoint(
                    local_checkpoints[i]);
                model_.local_models()[i].set_auto_commit(true);
            }

            const bool has_last_feedback =
                last_converged_responses_.size() == model_.num_local_models();
            if (has_last_feedback) {
                inject_or_clear_(last_converged_responses_);
                last_report_.feedback_source =
                    CouplingFeedbackSource::LastConvergedHomogenized;
            } else {
                inject_or_clear_({});
                last_report_.feedback_source =
                    CouplingFeedbackSource::ClearedOneWay;
            }

            const auto macro_t0 = std::chrono::steady_clock::now();
            bool macro_ok = false;
            if constexpr (requires(MacroSolverT& solver, double dp) {
                              { solver.get_increment_size() }
                                  -> std::convertible_to<double>;
                              solver.set_increment_size(dp);
                          }) {
                if (macro_trial_increment > 0.0) {
                    macro_solver_->set_increment_size(macro_trial_increment);
                }
                if (macro_step_target_time > macro_step_start_time) {
                    const auto verdict =
                        macro_solver_->step_to(macro_step_target_time);
                    macro_ok = verdict != StepVerdict::Stop &&
                        macro_solver_->current_time()
                            >= macro_step_target_time - 1.0e-12;
                } else {
                    macro_ok = macro_solver_->step();
                }
            } else {
                macro_ok = macro_solver_->step();
            }
            const auto macro_t1 = std::chrono::steady_clock::now();
            last_report_.macro_solve_seconds +=
                std::chrono::duration<double>(macro_t1 - macro_t0).count();
            capture_macro_solver_diagnostics_();

            if (!macro_ok) {
                macro_solver_->restore_checkpoint(macro_checkpoint);
                restore_previous_injection_();
                return false;
            }

            capture_attempted_macro_state_();
            macro_solver_->commit_trial_state();
            set_macro_trial_mode_(false);

            last_report_.converged = true;
            last_report_.rollback_performed = true;
            last_report_.hybrid_active = true;
            last_report_.coupling_regime =
                CouplingRegime::HybridObservationWindow;
            last_report_.termination_reason =
                CouplingTerminationReason::HybridObservationStepCompleted;
            last_report_.hybrid_reason =
                last_report_.failed_submodels > 0
                    ? "micro_solve_failed_advanced_with_last_feedback"
                    : "max_iterations_advanced_with_last_feedback";
            last_report_.one_way_replay_status =
                "deferred_after_two_way_failure";
            last_report_.work_gap = has_last_feedback
                ? max_force_gap_(current, last_converged_responses_)
                : std::numeric_limits<double>::infinity();
            ++hybrid_window_steps_;
            hybrid_success_steps_ = 0;
            last_report_.hybrid_window_steps = hybrid_window_steps_;
            last_report_.hybrid_success_steps = hybrid_success_steps_;
            last_report_.return_gate_passed = false;
            sync_local_runtime_report_();
            ++analysis_steps_;
            return true;
        };

        if (!accepted) {
            if (try_hybrid_observation_window()) {
                return true;
            }

            macro_solver_->restore_checkpoint(macro_checkpoint);
            for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
                model_.local_models()[i].restore_checkpoint(local_checkpoints[i]);
                model_.local_models()[i].set_auto_commit(true);
            }
            restore_previous_injection_();
            set_macro_trial_mode_(false);
            last_report_.converged = false;
            last_report_.rollback_performed = true;
            last_report_.termination_reason =
                (last_report_.failed_submodels > 0)
                    ? CouplingTerminationReason::MicroSolveFailed
                    : CouplingTerminationReason::MaxIterationsReached;
            return false;
        }

        for (const auto& response : current) {
            model_.macro_bridge().inject_response(response);
        }

        const double accepted_work_gap =
            last_converged_responses_.empty()
                ? 0.0
                : max_force_gap_(current, last_converged_responses_);

        macro_solver_->commit_trial_state();
        finalize_local_models_(macro_solver_->current_time(), &current);
        sync_local_runtime_report_();
        last_converged_responses_ = current;

        set_macro_trial_mode_(false);
        last_report_.converged = true;
        last_report_.termination_reason = CouplingTerminationReason::Converged;
        last_report_.coupling_regime = CouplingRegime::StrictTwoWay;
        last_report_.feedback_source = CouplingFeedbackSource::CurrentHomogenized;
        last_report_.hybrid_active = false;
        last_report_.one_way_replay_status = "not_needed";
        last_report_.work_gap = accepted_work_gap;
        if (hybrid_window_steps_ > 0) {
            const bool stable_force =
                last_report_.max_force_residual_rel <=
                two_way_failure_recovery_policy_.force_jump_tolerance;
            const bool stable_work =
                last_report_.work_gap <=
                two_way_failure_recovery_policy_.work_gap_tolerance;
            bool finite_tangent = true;
            for (const auto& response : current) {
                finite_tangent =
                    finite_tangent && response.tangent.allFinite();
            }
            if (stable_force && stable_work && finite_tangent) {
                ++hybrid_success_steps_;
            } else {
                hybrid_success_steps_ = 0;
            }
            last_report_.return_gate_passed =
                hybrid_success_steps_ >=
                two_way_failure_recovery_policy_.return_success_steps;
        } else {
            hybrid_success_steps_ = 0;
            last_report_.return_gate_passed = true;
        }
        last_report_.hybrid_window_steps = hybrid_window_steps_;
        last_report_.hybrid_success_steps = hybrid_success_steps_;
        ++analysis_steps_;
        return true;
    }

public:
    MultiscaleAnalysis(
        MacroSolverT& macro_solver,
        ModelT model,
        std::unique_ptr<CouplingAlgorithm> algorithm,
        std::unique_ptr<CouplingConvergence> convergence =
            std::make_unique<ForceAndTangentConvergence>(),
        std::unique_ptr<RelaxationPolicy> relaxation =
            std::make_unique<NoRelaxation>(),
        ExecutorT executor = {})
        : model_{std::move(model)}
        , macro_solver_{&macro_solver}
        , executor_{std::move(executor)}
        , algorithm_{std::move(algorithm)}
        , convergence_{std::move(convergence)}
        , relaxation_{std::move(relaxation)}
    {
        local_runtime_.resize(model_.num_local_models());
    }

    void set_coupling_start_step(int step) { coupling_start_step_ = step; }
    void set_section_dimensions(double w, double h) {
        section_width_ = w;
        section_height_ = h;
    }
    void set_tangent_perturbation(double h) { tangent_perturbation_ = h; }
    void set_local_tangent_computation_mode(TangentComputationMode mode)
    {
        local_tangent_computation_mode_ = mode;
    }
    void set_local_finite_difference_tangent_settings(
        HomogenizedTangentFiniteDifferenceSettings settings)
    {
        local_fd_tangent_settings_ = settings;
    }
    void set_macro_failure_backtracking(int attempts, double factor = 0.5)
    {
        macro_failure_backtrack_attempts_ = std::max(0, attempts);
        macro_failure_backtrack_factor_ = std::clamp(factor, 0.0, 1.0);
    }
    void set_macro_step_cutback(int attempts, double factor = 0.5)
    {
        macro_step_cutback_attempts_ = std::max(0, attempts);
        macro_step_cutback_factor_ = std::clamp(factor, 0.0, 1.0);
    }
    void set_site_adaptive_relaxation(
        SiteAdaptiveRelaxationSettings settings)
    {
        settings.residual_growth_limit =
            std::max(1.0, settings.residual_growth_limit);
        settings.max_backtracking_attempts =
            std::max(0, settings.max_backtracking_attempts);
        settings.backtracking_factor =
            std::clamp(settings.backtracking_factor, 0.0, 1.0);
        settings.min_alpha = std::clamp(settings.min_alpha, 0.0, 1.0);
        site_adaptive_relaxation_ = settings;
    }
    void set_two_way_failure_recovery_policy(
        TwoWayFailureRecoveryPolicy policy) noexcept
    {
        policy.max_hybrid_steps = std::max(0, policy.max_hybrid_steps);
        policy.return_success_steps =
            std::max(1, policy.return_success_steps);
        policy.work_gap_tolerance =
            std::max(0.0, policy.work_gap_tolerance);
        policy.force_jump_tolerance =
            std::max(0.0, policy.force_jump_tolerance);
        two_way_failure_recovery_policy_ = policy;
    }
    [[nodiscard]] const TwoWayFailureRecoveryPolicy&
    two_way_failure_recovery_policy() const noexcept
    {
        return two_way_failure_recovery_policy_;
    }
    void set_predictor_admissibility_filter(
        double min_symmetric_eigenvalue,
        int backtrack_attempts,
        double backtrack_factor = 0.5)
    {
        predictor_admissibility_filter_enabled_ = backtrack_attempts > 0;
        predictor_min_symmetric_eigenvalue_ = min_symmetric_eigenvalue;
        predictor_admissibility_backtrack_attempts_ =
            std::max(0, backtrack_attempts);
        predictor_admissibility_backtrack_factor_ =
            std::clamp(backtrack_factor, 0.0, 1.0);
    }
    void set_local_subproblem_runtime_settings(
        LocalSubproblemRuntimeSettings settings)
    {
        local_runtime_.set_settings(std::move(settings));
        local_runtime_.resize(model_.num_local_models());
    }
    void set_force_residual_norm(TangentValidationNormKind norm) noexcept {
        force_residual_norm_ = norm;
    }
    void set_tangent_residual_norm(TangentValidationNormKind norm) noexcept {
        tangent_residual_norm_ = norm;
    }

    [[nodiscard]] TangentValidationNormKind force_residual_norm() const noexcept
    {
        return force_residual_norm_;
    }

    [[nodiscard]] TangentValidationNormKind tangent_residual_norm()
        const noexcept
    {
        return tangent_residual_norm_;
    }

    [[nodiscard]] ModelT& model() noexcept { return model_; }
    [[nodiscard]] const ModelT& model() const noexcept { return model_; }

    [[nodiscard]] const CouplingIterationReport& last_report() const noexcept {
        return last_report_;
    }

    [[nodiscard]] const std::vector<SectionHomogenizedResponse>&
    last_responses() const noexcept {
        return last_responses_;
    }

    [[nodiscard]] int last_staggered_iterations() const noexcept {
        return last_report_.iterations;
    }

    [[nodiscard]] bool last_converged() const noexcept {
        return last_report_.converged;
    }

    [[nodiscard]] std::size_t analysis_step() const noexcept {
        return analysis_steps_;
    }

    [[nodiscard]] const LocalSubproblemRuntimeManager<LocalModelT>&
    local_subproblem_runtime() const noexcept
    {
        return local_runtime_;
    }

    [[nodiscard]] LocalSubproblemRuntimeManager<LocalModelT>&
    local_subproblem_runtime() noexcept
    {
        return local_runtime_;
    }

    [[nodiscard]] RestartBundle capture_restart_bundle() const
    {
        RestartBundle bundle;
        bundle.macro_checkpoint = macro_solver_->capture_checkpoint();
        bundle.local_checkpoints.reserve(model_.num_local_models());
        for (const auto& local_model : model_.local_models()) {
            bundle.local_checkpoints.push_back(
                local_model.capture_checkpoint());
        }
        bundle.last_responses = last_responses_;
        bundle.last_converged_responses = last_converged_responses_;
        bundle.analysis_steps = analysis_steps_;
        bundle.valid = true;
        return bundle;
    }

    void restore_restart_bundle(const RestartBundle& bundle)
    {
        if (!bundle.valid) {
            return;
        }

        macro_solver_->restore_checkpoint(bundle.macro_checkpoint);
        for (std::size_t i = 0;
             i < model_.num_local_models() && i < bundle.local_checkpoints.size();
             ++i)
        {
            model_.local_models()[i].restore_checkpoint(
                bundle.local_checkpoints[i]);
            model_.local_models()[i].set_auto_commit(true);
        }

        last_responses_ = bundle.last_responses;
        last_converged_responses_ = bundle.last_converged_responses;
        analysis_steps_ = bundle.analysis_steps;
        restore_previous_injection_();
        set_macro_trial_mode_(false);
        last_report_ = CouplingIterationReport{};
        initialize_report_norms_(
            last_report_, force_residual_norm_, tangent_residual_norm_);
    }

    bool initialize_local_models(bool seed_predictor = true)
    {
        last_report_ = CouplingIterationReport{};
        initialize_report_norms_(
            last_report_, force_residual_norm_, tangent_residual_norm_);
        last_responses_.clear();
        last_report_.mode = algorithm_->mode();
        last_report_.iterations = 0;
        last_report_.termination_reason = CouplingTerminationReason::NotRun;

        std::vector<SectionHomogenizedResponse> responses;
        solve_locals_once_(macro_solver_->current_time(),
                           responses,
                           last_report_.failed_submodels);

        for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
            auto macro_state =
                model_.macro_bridge().extract_section_state(model_.site(i));
            responses[i].strain_ref = macro_state.strain;
            responses[i].site.local_frame = macro_state.site.local_frame;

            auto& local_model = model_.local_models()[i];
            local_model.commit_trial_state();
            local_model.set_auto_commit(true);
            local_runtime_.save_accepted_state(
                i,
                local_model,
                responses[i],
                macro_state);
        }

        last_report_.converged = (last_report_.failed_submodels == 0);
        if (!last_report_.converged) {
            last_report_.termination_reason =
                CouplingTerminationReason::InitializationFailed;
        }
        if (seed_predictor && last_report_.converged) {
            last_converged_responses_ = responses;
        }
        last_responses_ = responses;
        sync_local_runtime_report_();
        return last_report_.converged;
    }

    bool step()
    {
        if (!should_couple_this_step_()) {
            return perform_macro_only_step_();
        }

        switch (algorithm_->mode()) {
            case CouplingMode::OneWayDownscaling:
                return perform_one_way_downscaling_();
            case CouplingMode::LaggedFeedbackCoupling:
                return perform_lagged_feedback_();
            case CouplingMode::IteratedTwoWayFE2:
                return perform_iterated_two_way_fe2_();
        }

        return false;
    }
};

}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MULTISCALE_ANALYSIS_HH
