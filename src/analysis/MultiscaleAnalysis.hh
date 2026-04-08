#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_ANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_ANALYSIS_HH

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "CouplingStrategy.hh"
#include "MicroSolveExecutor.hh"
#include "MultiscaleModel.hh"
#include "SteppableSolver.hh"

namespace fall_n {

template <CheckpointableSteppableSolver MacroSolverT,
          typename MacroBridgeT,
          LocalModelAdapter LocalModelT,
          typename ExecutorT = SerialExecutor>
class MultiscaleAnalysis {
    using ModelT = MultiscaleModel<MacroBridgeT, LocalModelT>;
    using MacroCheckpointT = typename std::remove_cvref_t<MacroSolverT>::checkpoint_type;
    using LocalCheckpointT = typename std::remove_cvref_t<LocalModelT>::checkpoint_type;

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
    std::size_t analysis_steps_{0};

    CouplingIterationReport last_report_{};
    std::vector<SectionHomogenizedResponse> last_responses_{};
    std::vector<SectionHomogenizedResponse> last_converged_responses_{};

    [[nodiscard]] static double relative_norm_(
        const Eigen::Vector<double, 6>& a,
        const Eigen::Vector<double, 6>& b)
    {
        const double denom = std::max({1.0, a.norm(), b.norm()});
        return (a - b).norm() / denom;
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

    void finalize_local_models_(double time)
    {
        for (auto& local_model : model_.local_models()) {
            local_model.commit_trial_state();
            local_model.set_auto_commit(true);
            local_model.end_of_step(time);
        }
    }

    void solve_locals_once_(double time,
                            std::vector<SectionHomogenizedResponse>& responses,
                            int& failed_submodels)
    {
        responses.resize(model_.num_local_models());
        failed_submodels = 0;
        last_report_.failed_submodels = 0;

        executor_.for_each(model_.num_local_models(), [&](std::size_t i) {
            auto& local_model = model_.local_models()[i];
            local_model.set_auto_commit(true);

            const auto ek =
                model_.macro_bridge().extract_element_kinematics(
                    model_.site(i).macro_element_id);
            local_model.update_kinematics(ek.kin_A, ek.kin_B);

            auto result = local_model.solve_step(time);
            auto response = local_model.section_response(
                section_width_, section_height_, tangent_perturbation_);
            response.site = model_.site(i);
            if constexpr (requires { result.converged; }) {
                if (!result.converged) {
                    response.status = ResponseStatus::SolveFailed;
                }
            }
            responses[i] = response;
        });

        last_report_.failed_sites.clear();
        last_report_.regularized_submodels = 0;
        last_report_.regularization_detected = false;
        for (const auto& response : responses) {
            accumulate_response_diagnostics_(last_report_, response);
        }
        failed_submodels = last_report_.failed_submodels;
    }

    bool perform_macro_only_step_()
    {
        set_macro_trial_mode_(false);

        last_report_ = CouplingIterationReport{};
        last_responses_.clear();
        last_report_.mode = algorithm_->mode();
        last_report_.iterations = 1;
        last_report_.termination_reason =
            CouplingTerminationReason::UncoupledMacroStep;

        const bool ok = macro_solver_->step();
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
        last_responses_.clear();
        last_report_.mode = CouplingMode::OneWayDownscaling;
        last_report_.iterations = 1;
        last_report_.termination_reason =
            CouplingTerminationReason::OneWayStepCompleted;

        set_macro_trial_mode_(false);
        if (!macro_solver_->step()) {
            last_report_.converged = false;
            last_report_.termination_reason =
                CouplingTerminationReason::MacroSolveFailed;
            return false;
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<SectionHomogenizedResponse> responses;
        solve_locals_once_(macro_solver_->current_time(),
                           responses,
                           last_report_.failed_submodels);
        last_responses_ = responses;
        finalize_local_models_(macro_solver_->current_time());
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
        last_responses_.clear();
        last_report_.mode = CouplingMode::LaggedFeedbackCoupling;
        last_report_.iterations = 1;
        last_report_.termination_reason =
            CouplingTerminationReason::LaggedStepCompleted;

        set_macro_trial_mode_(false);
        if (!macro_solver_->step()) {
            last_report_.converged = false;
            last_report_.termination_reason =
                CouplingTerminationReason::MacroSolveFailed;
            return false;
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<SectionHomogenizedResponse> responses;
        solve_locals_once_(macro_solver_->current_time(),
                           responses,
                           last_report_.failed_submodels);
        last_responses_ = responses;

        last_report_.force_residuals_rel.assign(model_.num_local_models(), 0.0);
        last_report_.tangent_residuals_rel.assign(model_.num_local_models(), 0.0);

        for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
            auto macro_state =
                model_.macro_bridge().extract_section_state(model_.site(i));
            responses[i].strain_ref = macro_state.strain;
            model_.macro_bridge().inject_response(responses[i]);
            last_report_.force_residuals_rel[i] =
                relative_norm_(macro_state.forces, responses[i].forces);
            last_report_.max_force_residual_rel =
                std::max(last_report_.max_force_residual_rel,
                         last_report_.force_residuals_rel[i]);
        }

        finalize_local_models_(macro_solver_->current_time());
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

        last_report_ = CouplingIterationReport{};
        last_responses_.clear();
        last_report_.mode = CouplingMode::IteratedTwoWayFE2;
        last_report_.iterations = 0;
        last_report_.termination_reason = CouplingTerminationReason::NotRun;

        const auto macro_checkpoint = macro_solver_->capture_checkpoint();
        std::vector<LocalCheckpointT> local_checkpoints;
        local_checkpoints.reserve(model_.num_local_models());
        for (auto& local_model : model_.local_models()) {
            local_model.set_auto_commit(false);
            local_checkpoints.push_back(local_model.capture_checkpoint());
        }

        std::vector<SectionHomogenizedResponse> predictor = last_converged_responses_;
        std::vector<SectionHomogenizedResponse> current(model_.num_local_models());

        set_macro_trial_mode_(true);

        bool accepted = false;
        const int max_iter = std::max(2, algorithm_->max_iterations());

        for (int iter = 0; iter < max_iter; ++iter) {
            last_report_.iterations = iter + 1;

            macro_solver_->restore_checkpoint(macro_checkpoint);

            if (predictor.size() == model_.num_local_models()) {
                for (const auto& response : predictor) {
                    model_.macro_bridge().inject_response(response);
                }
            } else {
                for (const auto& site : model_.sites()) {
                    model_.macro_bridge().clear_response(site);
                }
            }

            const auto macro_t0 = std::chrono::steady_clock::now();
            if (!macro_solver_->step()) {
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
            const auto macro_t1 = std::chrono::steady_clock::now();
            last_report_.macro_solve_seconds +=
                std::chrono::duration<double>(macro_t1 - macro_t0).count();

            const auto micro_t0 = std::chrono::steady_clock::now();
            executor_.for_each(model_.num_local_models(), [&](std::size_t i) {
                auto& local_model = model_.local_models()[i];
                local_model.restore_checkpoint(local_checkpoints[i]);

                const auto ek =
                    model_.macro_bridge().extract_element_kinematics(
                        model_.site(i).macro_element_id);
                local_model.update_kinematics(ek.kin_A, ek.kin_B);

                auto result = local_model.solve_step(macro_solver_->current_time());
                current[i] = local_model.section_response(
                    section_width_, section_height_, tangent_perturbation_);
                current[i].site = model_.site(i);
                if constexpr (requires { result.converged; }) {
                    if (!result.converged) {
                        current[i].status = ResponseStatus::SolveFailed;
                    }
                }
            });
            last_responses_ = current;
            const auto micro_t1 = std::chrono::steady_clock::now();
            last_report_.micro_solve_seconds +=
                std::chrono::duration<double>(micro_t1 - micro_t0).count();

            last_report_.failed_submodels = 0;
            last_report_.regularized_submodels = 0;
            last_report_.regularization_detected = false;
            last_report_.failed_sites.clear();
            last_report_.force_residuals_rel.assign(model_.num_local_models(), 0.0);
            last_report_.tangent_residuals_rel.assign(model_.num_local_models(), 0.0);
            last_report_.max_force_residual_rel = 0.0;
            last_report_.max_tangent_residual_rel = 0.0;

            for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
                auto macro_state =
                    model_.macro_bridge().extract_section_state(model_.site(i));
                current[i].strain_ref = macro_state.strain;
                current[i].site.local_frame = macro_state.site.local_frame;

                if (predictor.size() == model_.num_local_models()) {
                    const auto tangent_before = current[i].tangent;
                    const auto force_before = current[i].forces;
                    relaxation_->relax(current[i], predictor[i], iter);
                    if ((current[i].tangent - tangent_before).norm() > 0.0
                        || (current[i].forces - force_before).norm() > 0.0)
                    {
                        last_report_.relaxation_applied = true;
                    }
                    last_report_.tangent_residuals_rel[i] =
                        relative_norm_(current[i].tangent, predictor[i].tangent);
                }

                last_report_.force_residuals_rel[i] =
                    relative_norm_(macro_state.forces, current[i].forces);

                accumulate_response_diagnostics_(last_report_, current[i]);

                last_report_.max_force_residual_rel =
                    std::max(last_report_.max_force_residual_rel,
                             last_report_.force_residuals_rel[i]);
                last_report_.max_tangent_residual_rel =
                    std::max(last_report_.max_tangent_residual_rel,
                             last_report_.tangent_residuals_rel[i]);
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

        if (!accepted) {
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

        macro_solver_->commit_trial_state();
        finalize_local_models_(macro_solver_->current_time());
        last_converged_responses_ = current;

        set_macro_trial_mode_(false);
        last_report_.converged = true;
        last_report_.termination_reason = CouplingTerminationReason::Converged;
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
    {}

    void set_coupling_start_step(int step) { coupling_start_step_ = step; }
    void set_section_dimensions(double w, double h) {
        section_width_ = w;
        section_height_ = h;
    }
    void set_tangent_perturbation(double h) { tangent_perturbation_ = h; }

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

    bool initialize_local_models(bool seed_predictor = true)
    {
        last_report_ = CouplingIterationReport{};
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
