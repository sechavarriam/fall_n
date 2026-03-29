#ifndef FALL_N_SRC_ANALYSIS_MULTISCALE_ANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_MULTISCALE_ANALYSIS_HH

// =============================================================================
//  MultiscaleAnalysis — FE² staggered coupling orchestrator
// =============================================================================
//
//  Encapsulates the staggered iteration loop that couples a macro-scale
//  (global) solver with one or more meso-scale (local) sub-model solvers.
//  The class absorbs the ~80 lines of coupling logic currently spread
//  across main_lshaped_multiscale.cpp, replacing them with a single
//  call to step().
//
//  Design decisions:
//    - Template on LocalModel to avoid virtual dispatch in the inner loop.
//    - Strategy objects (CouplingStrategy.hh) are injected via constructor:
//        * ScaleBridgePolicy     — one-way vs two-way
//        * CouplingConvergence   — convergence criterion
//        * RelaxationPolicy      — tangent blending
//    - The class does NOT own the global solver — it only references the
//      MultiscaleModel<LocalModel> which provides kinematics extraction
//      and response injection callbacks.
//
//  Usage:
//    MultiscaleAnalysis<NonlinearSubModelEvolver> analysis(
//        std::move(ms_model),
//        std::make_unique<TwoWayStaggered>(4),
//        std::make_unique<FrobeniusConvergence>(0.05),
//        std::make_unique<ConstantRelaxation>(0.7)
//    );
//    analysis.set_coupling_start_step(10);
//    analysis.set_section_dimensions(0.30, 0.30);
//
//    for (int step = 0; step < n_steps; ++step)
//        analysis.step(t, step);
//
// =============================================================================

#include <cstddef>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "MultiscaleModel.hh"
#include "CouplingStrategy.hh"


namespace fall_n {


// =============================================================================
//  MultiscaleAnalysis
// =============================================================================

template <LocalModelAdapter LocalModel>
class MultiscaleAnalysis {

    // ── Members ──────────────────────────────────────────────────

    MultiscaleModel<LocalModel>             model_;

    std::unique_ptr<ScaleBridgePolicy>      bridge_;
    std::unique_ptr<CouplingConvergence>    convergence_;
    std::unique_ptr<RelaxationPolicy>       relaxation_;

    int    coupling_start_step_{0};
    double section_width_{0.30};
    double section_height_{0.30};
    double tangent_perturbation_{1.0e-6};

    /// Tangent history for convergence checking across staggered iterations.
    std::vector<Eigen::Matrix<double,6,6>> D_prev_;


public:

    // ── Construction ─────────────────────────────────────────────

    MultiscaleAnalysis(
        MultiscaleModel<LocalModel>          model,
        std::unique_ptr<ScaleBridgePolicy>   bridge,
        std::unique_ptr<CouplingConvergence> convergence,
        std::unique_ptr<RelaxationPolicy>    relaxation)
        : model_{std::move(model)}
        , bridge_{std::move(bridge)}
        , convergence_{std::move(convergence)}
        , relaxation_{std::move(relaxation)}
    {
        D_prev_.resize(model_.num_local_models(),
                       Eigen::Matrix<double,6,6>::Zero());
    }

    // ── Configuration ────────────────────────────────────────────

    void set_coupling_start_step(int step) { coupling_start_step_ = step; }
    void set_section_dimensions(double w, double h) {
        section_width_ = w;
        section_height_ = h;
    }
    void set_tangent_perturbation(double h) { tangent_perturbation_ = h; }

    // ── Access ───────────────────────────────────────────────────

    [[nodiscard]] MultiscaleModel<LocalModel>&       model()       { return model_; }
    [[nodiscard]] const MultiscaleModel<LocalModel>& model() const { return model_; }

    // ── Main step: staggered coupling for one global time step ───
    //
    //  1. For each sub-model: extract kinematics, update BCs, solve
    //  2. If two-way coupling is active and step ≥ coupling_start_step:
    //     a. Compute homogenised tangent D_hom per sub-model
    //     b. Apply relaxation
    //     c. Check convergence
    //     d. Inject D_hom and f_hom into global beam elements
    //     e. Repeat until converged or max iterations reached
    //
    //  Returns true if the staggered loop converged (or one-way coupling).

    bool step(double time, int global_step)
    {
        const bool do_coupling =
            bridge_->requires_feedback() &&
            global_step >= coupling_start_step_;

        const int max_iter = do_coupling
            ? bridge_->max_staggered_iterations()
            : 1;

        bool converged = true;

        for (int s_iter = 0; s_iter < max_iter; ++s_iter) {

            // ── (a) Downscale: extract kinematics, solve sub-models ──
            for (auto& lm : model_.local_models()) {
                auto ek = model_.extract_kinematics(lm.parent_element_id());
                lm.update_kinematics(ek.kin_A, ek.kin_B);
                lm.solve_step(time);
            }

            if (!do_coupling)
                break;

            // ── (b) Upscale: tangent + forces + inject ───────────────
            converged = true;
            std::vector<Eigen::Matrix<double,6,6>> D_curr(
                model_.num_local_models());

            for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
                auto& lm = model_.local_models()[i];
                D_curr[i] = lm.section_tangent(
                    section_width_, section_height_, tangent_perturbation_);

                // Relaxation
                if (s_iter > 0) {
                    relaxation_->relax(D_curr[i], D_prev_[i], s_iter);
                }
            }

            // Convergence check
            if (s_iter > 0) {
                converged = convergence_->converged(D_prev_, D_curr);
            }

            // Store for next iteration
            D_prev_ = D_curr;

            // Inject into global model
            for (std::size_t i = 0; i < model_.num_local_models(); ++i) {
                auto& lm = model_.local_models()[i];
                auto f_hom = lm.section_forces(
                    section_width_, section_height_);
                model_.inject_response(
                    lm.parent_element_id(), D_curr[i], f_hom);
            }

            if (s_iter == 0)
                continue;   // always do at least 2 iterations

            if (converged) {
                std::cout << "    [FE²] converged at staggered iter "
                          << s_iter << "\n";
                break;
            }
        }

        // ── Commit converged state ───────────────────────────────────
        for (auto& lm : model_.local_models())
            lm.commit_state();

        return converged;
    }
};


}  // namespace fall_n

#endif // FALL_N_SRC_ANALYSIS_MULTISCALE_ANALYSIS_HH
