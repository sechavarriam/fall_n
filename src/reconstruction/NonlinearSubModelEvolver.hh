#ifndef FALL_N_SRC_RECONSTRUCTION_NONLINEAR_SUB_MODEL_EVOLVER_HH
#define FALL_N_SRC_RECONSTRUCTION_NONLINEAR_SUB_MODEL_EVOLVER_HH

// =============================================================================
//  NonlinearSubModelEvolver  Persistent nonlinear continuum sub-model
// =============================================================================
//
//  Unlike SubModelEvolver (which creates a fresh solver each step, losing
//  material state), this class constructs the Model<> once and reuses it
//  across time steps.  The KoBatheConcrete3D material accumulates crack
//  history, plastic strain, and damage through the earthquake.
//
//  At each global step the caller:
//    1. Extracts updated SectionKinematics from the beam element.
//    2. Calls update_kinematics(kin_A, kin_B) to recompute face BCs.
//    3. Calls solve_step(time) which drives the model from its current
//       converged state to the new BC target via Newton iteration,
//       commits the material state, and optionally writes VTK output
//       (including crack plane glyphs).
//
//  The solver manages PETSc SNES directly rather than through the
//  NonlinearAnalysis wrapper, so that the displacement vector is NOT
//  reset to zero between steps  the converged state from step N serves
//  as the initial guess for step N+1.
//
// =============================================================================

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <petsc.h>

#include "../analysis/MultiscaleCoordinator.hh"
#include "../analysis/MultiscaleTypes.hh"
#include "../analysis/NLAnalysis.hh"
#include "../analysis/PenaltyCoupling.hh"

#include "../materials/MaterialPolicy.hh"
#include "../materials/Material.hh"
#include "../materials/InternalFieldSnapshot.hh"
#include "../materials/SubmodelMaterialFactory.hh"
#include "../materials/SubmodelMaterialFactoryDefaults.hh"

#include "BoundaryReactionHomogenizer.hh"
#include "HomogenizedSection.hh"
#include "LocalBoundaryConditionApplicator.hh"
#include "LocalCrackData.hh"
#include "LocalCrackDiagnostics.hh"
#include "LocalModelAdapter.hh"
#include "LocalVTKOutputWriter.hh"
#include "PersistentLocalStateOps.hh"
#include "../analysis/ArcLengthSolver.hh"

#include "../model/Model.hh"
#include "../model/PrismaticDomainBuilder.hh"
#include "../continuum/KinematicPolicy.hh"
#include "../elements/ContinuumElement.hh"
#include "../elements/TrussElement.hh"
#include "../elements/FEM_Element.hh"
#include "../elements/ElementPolicy.hh"

#include "../post-processing/VTK/VTKModelExporter.hh"
#include "../post-processing/VTK/PVDWriter.hh"


namespace fall_n {


// =============================================================================
//  PenaltyCouplingEntry  penalty spring for embedded rebar → hex coupling
// =============================================================================
//
//  For each interior rebar node (not on MinZ/MaxZ faces), a penalty spring
//  ties its displacement to the interpolated hex displacement at its
//  physical position:  F = α · (u_rebar − Σ Nᵢ · uᵢ_hex).
//
//  The hex_weights vector stores (sieve_point, Nᵢ) pairs for the nodes
//  of the host hex element, where Nᵢ is the trilinear/triquadratic
//  shape function evaluated at the rebar node's parent coordinates
//  (ξ, η, ζ) within the host element.
//
//  NOTE: PenaltyCouplingEntry is defined in analysis/PenaltyCoupling.hh
//        and shared between PenaltyCoupling (standalone) and
//        NonlinearSubModelEvolver (FE² sub-model).


// =============================================================================
//  NonlinearSubModelEvolver
// =============================================================================

class NonlinearSubModelEvolver {

    using Policy = ThreeDimensionalMaterial;
    static constexpr std::size_t NDOF = 3;
    using ContElem   = ContinuumElement<Policy, NDOF, continuum::SmallStrain>;
    using MixedModel = Model<Policy, continuum::SmallStrain, NDOF, MultiElementPolicy>;
    using StateOps = PersistentLocalStateOps<MixedModel>;
    using BoundaryConditionApplicator =
        LocalBoundaryConditionApplicator<MixedModel, MultiscaleSubModel>;
    using LocalHomogenizer =
        BoundaryReactionHomogenizer<MixedModel, MultiscaleSubModel>;
    using CondensationWorkspace =
        condensation::SparseSchurComplementWorkspace<Eigen::SparseMatrix<double>>;
    using ModelCheckpointT = typename MixedModel::checkpoint_type;

    //  Sub-model reference 
    MultiscaleSubModel* sub_;
    double              fc_;
    std::array<double,3> local_ex_{1,0,0}, local_ey_{0,1,0}, local_ez_{0,0,1};

    //  Rebar material parameters (used when sub_->has_rebar())
    double rebar_E_{200000.0};
    double rebar_fy_{420.0};
    double rebar_b_{0.01};

    //  Material factories (injectable — defaults built in constructor)
    std::unique_ptr<ConcreteMaterialFactory> concrete_factory_;
    std::unique_ptr<RebarMaterialFactory>    rebar_factory_;

    //  Persistent model 
    std::unique_ptr<MixedModel> model_;
    bool model_ready_{false};

    //  PETSc solver objects (persist across steps) 
    struct Context {
        MixedModel* model;
        Vec         f_ext;
        const std::vector<PenaltyCouplingEntry>* penalty_couplings{nullptr};
        double alpha_penalty{0.0};
    };
    Context ctx_{};

    SNES snes_{nullptr};
    Vec  U_{nullptr};            ///< global free-DOF displacement (NOT reset)
    Vec  R_{nullptr};            ///< residual work vector
    Vec  f_ext_{nullptr};        ///< external forces (zero for sub-model)
    Mat  J_{nullptr};            ///< tangent stiffness
    Vec  U_work_{nullptr};       ///< pre-allocated work vector (U-sized)
    Vec  imp_work_{nullptr};     ///< pre-allocated work vector (imposed-sized)
    /// Persistent bisection checkpoint storage for solve_ramp_adaptive_
    /// (Plan v2 §Fase 1 hot-path: avoid per-call VecDuplicate inside the
    /// adaptive bisection loop; previously two VecDuplicate + two VecDestroy
    /// per macro-step solve).
    Vec  U_checkpoint_{nullptr};
    Vec  imp_checkpoint_{nullptr};

    //  VTK output 
    LocalVTKOutputWriter output_writer_{};
    int          vtk_interval_;
    int          step_count_{0};
    bool         auto_commit_{true};

    //  Crack history 
    std::vector<CrackRecord> latest_cracks_;
    CrackSummary latest_crack_summary_{};
    CrackSummary last_attempted_crack_summary_{};
    SubModelSolverResult last_solve_result_{};

    //  NL solver parameters 
    int first_step_increments_{15};
    int first_step_bisect_{6};

    //  Arc-length control (Phase 2.3) 
    bool   use_arc_length_{false};
    int    consecutive_divergences_{0};
    double last_good_frac_{0.5};   // cache last converged sub-step fraction
    int    arc_length_threshold_{3}; // switch after this many consecutive divergences
    int    adaptive_max_substeps_{30};
    int    adaptive_max_bisections_{10};
    int    adaptive_tail_rescue_attempts_{0};
    double adaptive_tail_rescue_progress_threshold_{0.75};
    int    adaptive_tail_rescue_substep_bonus_{12};
    int    adaptive_tail_rescue_bisection_bonus_{4};
    double adaptive_tail_rescue_initial_fraction_{0.5};

    //  Crack VTK filter: minimum opening to include crack plane in output.
    //  Cracks below this threshold (in strain units) are omitted from VTK
    //  to reduce visual noise and file size.  Default 0.5e-3 (≈ 0.5 mm/m).
    double min_crack_opening_{0.5e-3};

    //  Penalty coupling for embedded rebar (Master-Slave interpolation).
    //  α should be ≫ max(E_truss·A/L, E_hex/h) for good coupling.
    double alpha_penalty_{1.0e6};
    int    snes_max_it_{50};
    double snes_atol_{1e-6};
    double snes_rtol_{1e-2};
    RegularizationPolicyKind regularization_policy_{
        RegularizationPolicyKind::DiagonalFloor};
    double diagonal_floor_{1.0};
    TangentComputationMode tangent_mode_{
        TangentComputationMode::PreferLinearizedCondensation};
    double tangent_validation_relative_tolerance_{1.0e-1};
    double tangent_validation_max_column_tolerance_{5.0e-1};
    TangentValidationNormKind tangent_validation_norm_{
        TangentValidationNormKind::StateWeightedFrobenius};
    std::vector<PenaltyCouplingEntry> penalty_couplings_;
    std::unique_ptr<CondensationWorkspace> condensed_workspace_{
        std::make_unique<CondensationWorkspace>()};

    //  Optional FE² state transfer callback (Phase A/B/C infrastructure)
    std::function<void(typename MixedModel::container_type&)>
        state_transfer_callback_;

    [[nodiscard]] StateOps make_state_ops_() noexcept
    {
        return StateOps{model_.get(), U_};
    }

    [[nodiscard]] BoundaryConditionApplicator make_bc_applicator_() noexcept
    {
        return BoundaryConditionApplicator{model_.get(), sub_};
    }

    [[nodiscard]] LocalHomogenizer make_local_homogenizer_(
        double section_width = 0.0,
        double section_height = 0.0) noexcept
    {
        return LocalHomogenizer{
            model_.get(),
            sub_,
            U_,
            U_work_,
            imp_work_,
            snes_,
            regularization_policy_,
            diagonal_floor_,
            tangent_mode_,
            tangent_validation_relative_tolerance_,
            tangent_validation_max_column_tolerance_,
            tangent_validation_norm_,
            section_width,
            section_height,
            &penalty_couplings_,
            alpha_penalty_,
            make_bc_applicator_(),
            make_state_ops_(),
            condensed_workspace_.get()};
    }

    struct MaterialSolveDiagnostics {
        int active_cracked_points{0};
        int max_num_cracks_at_point{0};
        int max_no_flow_stabilization_iterations{0};
        int total_no_flow_crack_state_switches{0};
        bool no_flow_unstabilized_detected{false};
        double max_no_flow_recovery_residual{0.0};
        double max_no_flow_coupling_update_norm{0.0};
    };

    struct AdaptiveRampSolveStats {
        bool converged{false};
        int last_reason{0};
        PetscInt last_snes_iterations{0};
        double last_function_norm{0.0};
        int total_substeps{0};
        int total_bisections{0};
        int total_tail_rescue_attempts{0};
        double achieved_fraction{0.0};
        double tail_rescue_trigger_fraction{0.0};
        double failed_target_fraction{0.0};
        double failed_step_fraction{0.0};
        double minimum_step_fraction{0.0};
        int failed_substep_index{0};
        SubModelFailureCause failure_cause{SubModelFailureCause::None};
    };

    [[nodiscard]] static SubModelFailureCause
    classify_snes_failure_(SNESConvergedReason reason) noexcept
    {
        switch (reason) {
        case SNES_DIVERGED_FUNCTION_DOMAIN:
            return SubModelFailureCause::FunctionDomain;
        case SNES_DIVERGED_LINEAR_SOLVE:
        case SNES_DIVERGED_INNER:
            return SubModelFailureCause::LinearSolveFailed;
        default:
            return SubModelFailureCause::NewtonDiverged;
        }
    }

    static void apply_material_diagnostics_(
        SubModelSolverResult& result,
        const MaterialSolveDiagnostics& diagnostics) noexcept
    {
        result.material_points_with_active_cracks =
            diagnostics.active_cracked_points;
        result.max_num_cracks_at_point =
            diagnostics.max_num_cracks_at_point;
        result.max_no_flow_stabilization_iterations =
            diagnostics.max_no_flow_stabilization_iterations;
        result.total_no_flow_crack_state_switches =
            diagnostics.total_no_flow_crack_state_switches;
        result.no_flow_unstabilized_detected =
            diagnostics.no_flow_unstabilized_detected;
        result.max_material_no_flow_recovery_residual =
            diagnostics.max_no_flow_recovery_residual;
        result.max_material_no_flow_coupling_update_norm =
            diagnostics.max_no_flow_coupling_update_norm;
    }

    [[nodiscard]] MaterialSolveDiagnostics collect_material_diagnostics_() const
    {
        MaterialSolveDiagnostics diagnostics;
        if (!model_) {
            return diagnostics;
        }

        for (const auto& elem : model_->elements()) {
            const auto snapshots = elem.material_point_snapshots();
            for (const auto& snap : snapshots) {
                if (snap.num_cracks.value_or(0) > 0) {
                    ++diagnostics.active_cracked_points;
                }
                diagnostics.max_num_cracks_at_point = std::max(
                    diagnostics.max_num_cracks_at_point,
                    snap.num_cracks.value_or(0));
                diagnostics.max_no_flow_stabilization_iterations = std::max(
                    diagnostics.max_no_flow_stabilization_iterations,
                    snap.no_flow_stabilization_iterations.value_or(0));
                diagnostics.total_no_flow_crack_state_switches +=
                    snap.no_flow_crack_state_switches.value_or(0);
                diagnostics.no_flow_unstabilized_detected =
                    diagnostics.no_flow_unstabilized_detected
                    || !snap.no_flow_stabilized.value_or(true);
                diagnostics.max_no_flow_recovery_residual = std::max(
                    diagnostics.max_no_flow_recovery_residual,
                    snap.no_flow_recovery_residual.value_or(0.0));
                diagnostics.max_no_flow_coupling_update_norm = std::max(
                    diagnostics.max_no_flow_coupling_update_norm,
                    snap.no_flow_coupling_update_norm.value_or(0.0));
            }
        }

        return diagnostics;
    }

    [[nodiscard]] bool should_attempt_tail_rescue_(
        const AdaptiveRampSolveStats& stats,
        double global_base_fraction,
        double global_span_fraction,
        int rescue_attempts_done) const noexcept
    {
        if (adaptive_tail_rescue_attempts_ <= 0
            || rescue_attempts_done >= adaptive_tail_rescue_attempts_
            || stats.converged)
        {
            return false;
        }

        if (stats.failure_cause != SubModelFailureCause::AdaptiveMinFractionReached
            && stats.failure_cause
                   != SubModelFailureCause::AdaptiveSubstepBudgetExceeded)
        {
            return false;
        }

        if (global_span_fraction <= 1.0e-12) {
            return false;
        }

        const double local_progress =
            (stats.achieved_fraction - global_base_fraction)
            / global_span_fraction;
        const double segment_end = global_base_fraction + global_span_fraction;
        const double remaining = segment_end - stats.achieved_fraction;
        return local_progress >= adaptive_tail_rescue_progress_threshold_
            && remaining > 1.0e-12;
    }

    [[nodiscard]] AdaptiveRampSolveStats solve_ramp_adaptive_(
        Vec imp_prev,
        Vec imp_target,
        double initial_step_fraction,
        int max_bisections,
        int max_substeps,
        bool track_last_good_fraction,
        double global_base_fraction = 0.0,
        double global_span_fraction = 1.0)
    {
        AdaptiveRampSolveStats stats;
        stats.achieved_fraction = global_base_fraction;
        stats.failed_target_fraction = global_base_fraction;

        const double clamped_initial_step =
            std::clamp(initial_step_fraction, 1.0e-12, 1.0);
        const int clamped_bisections = std::max(0, max_bisections);
        const int clamped_substeps = std::max(1, max_substeps);
        const double min_frac_local =
            std::ldexp(clamped_initial_step, -clamped_bisections);
        stats.minimum_step_fraction = global_span_fraction * min_frac_local;

        auto restore_imposed_at_progress = [&](double progress_fraction) {
            VecCopy(imp_prev, model_->imposed_solution());
            VecScale(model_->imposed_solution(), 1.0 - progress_fraction);
            VecAXPY(model_->imposed_solution(), progress_fraction, imp_target);
        };

        double progress = 0.0;
        double step_frac = clamped_initial_step;

        // Reuse persistent member checkpoints (Plan v2 §Fase 1 hot-path).
        // Aliases keep the rest of the body legible without renaming.
        Vec U_checkpoint = U_checkpoint_;
        Vec imp_checkpoint = imp_checkpoint_;
        VecCopy(U_, U_checkpoint);
        restore_imposed_at_progress(progress);
        VecCopy(model_->imposed_solution(), imp_checkpoint);

        while (progress < 1.0 - 1.0e-12) {
            const double target_p = std::min(progress + step_frac, 1.0);
            const double global_target_p =
                global_base_fraction + global_span_fraction * target_p;
            const double global_step_frac =
                global_span_fraction * step_frac;
            restore_imposed_at_progress(target_p);

            SNESSolve(snes_, nullptr, U_);

            SNESConvergedReason reason;
            SNESGetConvergedReason(snes_, &reason);
            PetscInt snes_iters = 0;
            SNESGetIterationNumber(snes_, &snes_iters);
            PetscReal fnorm = 0.0;
            SNESGetFunctionNorm(snes_, &fnorm);

            stats.last_reason = static_cast<int>(reason);
            stats.last_snes_iterations = snes_iters;
            stats.last_function_norm = static_cast<double>(fnorm);
            stats.failed_target_fraction = global_target_p;
            stats.failed_step_fraction = global_step_frac;
            stats.failed_substep_index = stats.total_substeps + 1;

            if (reason > 0) {
                commit_state();
                VecCopy(U_, U_checkpoint);
                VecCopy(model_->imposed_solution(), imp_checkpoint);
                progress = target_p;
                stats.achieved_fraction =
                    global_base_fraction + global_span_fraction * progress;
                ++stats.total_substeps;
                if (track_last_good_fraction) {
                    last_good_frac_ = global_step_frac;
                }

                if (stats.total_substeps >= clamped_substeps
                    && progress < 1.0 - 1.0e-12)
                {
                    stats.failure_cause =
                        SubModelFailureCause::AdaptiveSubstepBudgetExceeded;
                    break;
                }

                std::println(
                    "    [SubModel {:2d}] sub-step {:2d}: "
                    "p={:.1f}→{:.1f}%  SNES={:2d} iters  reason={}  frac={:.3e}",
                    sub_->parent_element_id,
                    stats.total_substeps,
                    (global_target_p - global_step_frac) * 100.0,
                    global_target_p * 100.0,
                    static_cast<int>(snes_iters),
                    static_cast<int>(reason),
                    global_step_frac);

                step_frac = std::min(step_frac * 2.0, 1.0 - progress);
                if (step_frac <= 0.0) {
                    break;
                }
                continue;
            }

            VecCopy(U_checkpoint, U_);
            VecCopy(imp_checkpoint, model_->imposed_solution());
            revert_state();
            ++stats.total_bisections;
            stats.failure_cause = classify_snes_failure_(reason);

            std::println(
                "    [SubModel {:2d}] BISECT at p={:.1f}%  "
                "SNES diverged (reason={})  frac {:.3e}→{:.3e}",
                sub_->parent_element_id,
                global_target_p * 100.0,
                static_cast<int>(reason),
                global_step_frac,
                global_span_fraction * step_frac * 0.5);

            step_frac *= 0.5;
            if (step_frac < min_frac_local) {
                stats.failure_cause =
                    SubModelFailureCause::AdaptiveMinFractionReached;
                break;
            }
        }

        stats.converged = (progress >= 1.0 - 1.0e-12);
        stats.achieved_fraction =
            global_base_fraction + global_span_fraction * progress;
        if (stats.converged) {
            stats.failure_cause = SubModelFailureCause::None;
        } else if (stats.total_substeps >= clamped_substeps
                   && progress < 1.0 - 1.0e-12)
        {
            stats.failure_cause =
                SubModelFailureCause::AdaptiveSubstepBudgetExceeded;
        }

        if (stats.converged) {
            std::println(
                "  [SubModel {}] Adaptive ramp: {} sub-steps, {} bisections "
                "to reach target",
                sub_->parent_element_id,
                stats.total_substeps,
                stats.total_bisections);
        } else if (stats.total_substeps > 0) {
            std::println(
                "  [SubModel {}] Adaptive ramp PARTIAL: {} sub-steps reached "
                "{:.1f}% ({} bisections)",
                sub_->parent_element_id,
                stats.total_substeps,
                progress * 100.0,
                stats.total_bisections);
        } else {
            std::println(
                "  [SubModel {}] Adaptive ramp FAILED after {} bisections "
                "(min frac {:.1e})",
                sub_->parent_element_id,
                stats.total_bisections,
                stats.minimum_step_fraction);
        }

        // Persistent checkpoints (U_checkpoint_, imp_checkpoint_) are owned
        // by the evolver; nothing to destroy here.
        return stats;
    }

    [[nodiscard]] AdaptiveRampSolveStats solve_ramp_with_tail_rescue_(
        Vec imp_prev,
        Vec imp_target,
        double initial_step_fraction,
        int max_bisections,
        int max_substeps,
        bool track_last_good_fraction,
        double global_base_fraction = 0.0,
        double global_span_fraction = 1.0)
    {
        auto stats = solve_ramp_adaptive_(
            imp_prev,
            imp_target,
            initial_step_fraction,
            max_bisections,
            max_substeps,
            track_last_good_fraction,
            global_base_fraction,
            global_span_fraction);

        const double segment_end = global_base_fraction + global_span_fraction;
        while (should_attempt_tail_rescue_(
            stats,
            global_base_fraction,
            global_span_fraction,
            stats.total_tail_rescue_attempts))
        {
            ++stats.total_tail_rescue_attempts;
            if (stats.tail_rescue_trigger_fraction <= 0.0) {
                stats.tail_rescue_trigger_fraction = stats.achieved_fraction;
            }

            const double rescue_base = stats.achieved_fraction;
            const double rescue_span = segment_end - rescue_base;
            const int rescue_substeps =
                std::max(1,
                         max_substeps
                             + stats.total_tail_rescue_attempts
                                   * adaptive_tail_rescue_substep_bonus_);
            const int rescue_bisections =
                std::max(0,
                         max_bisections
                             + stats.total_tail_rescue_attempts
                                   * adaptive_tail_rescue_bisection_bonus_);
            const int accepted_before = stats.total_substeps;

            Vec rescue_prev{nullptr};
            VecDuplicate(model_->imposed_solution(), &rescue_prev);
            VecCopy(model_->imposed_solution(), rescue_prev);

            std::println(
                "  [SubModel {}] Tail rescue {}/{} from {:.1f}% "
                "(remaining {:.3e}, substeps {}, bisections {})",
                sub_->parent_element_id,
                stats.total_tail_rescue_attempts,
                adaptive_tail_rescue_attempts_,
                rescue_base * 100.0,
                rescue_span,
                rescue_substeps,
                rescue_bisections);

            auto rescue_stats = solve_ramp_adaptive_(
                rescue_prev,
                imp_target,
                adaptive_tail_rescue_initial_fraction_,
                rescue_bisections,
                rescue_substeps,
                track_last_good_fraction,
                rescue_base,
                rescue_span);
            VecDestroy(&rescue_prev);

            stats.last_reason = rescue_stats.last_reason;
            stats.last_snes_iterations = rescue_stats.last_snes_iterations;
            stats.last_function_norm = rescue_stats.last_function_norm;
            stats.total_substeps += rescue_stats.total_substeps;
            stats.total_bisections += rescue_stats.total_bisections;
            stats.achieved_fraction = rescue_stats.achieved_fraction;
            stats.failed_target_fraction =
                rescue_stats.failed_target_fraction;
            stats.failed_step_fraction = rescue_stats.failed_step_fraction;
            stats.minimum_step_fraction = std::min(
                stats.minimum_step_fraction,
                rescue_stats.minimum_step_fraction);
            stats.failure_cause = rescue_stats.failure_cause;
            stats.converged = rescue_stats.converged;
            stats.failed_substep_index = rescue_stats.converged
                ? 0
                : accepted_before + rescue_stats.failed_substep_index;

            if (rescue_stats.converged) {
                stats.failure_cause = SubModelFailureCause::None;
                break;
            }

            if (rescue_stats.achieved_fraction <= rescue_base + 1.0e-12) {
                break;
            }
        }

        return stats;
    }

    [[nodiscard]] static double default_local_length_scale_mm_(
        const MultiscaleSubModel& sub) noexcept
    {
        // Ko-Bathe (2026), Sec. 2.1.5 / Fig. 7(b): when lb is not specified
        // globally, the local interpretation uses the maximum side length of
        // the element carrying the material point, not the cube-root volume.
        return 1.0e3 * std::max({sub.grid.dx, sub.grid.dy, sub.grid.dz});
    }

    void reset_transient_model_()
    {
        destroy_petsc_objects();
        model_.reset();
        model_ready_ = false;
        penalty_couplings_.clear();
        condensed_workspace_->reset();
        latest_cracks_.clear();
        latest_crack_summary_ = {};
        last_attempted_crack_summary_ = {};
    }


    //  SNES callbacks 

    static PetscErrorCode FormResidual(
        SNES /*snes*/, Vec u_global, Vec R_out, void* ctx_ptr)
    {
        PetscFunctionBeginUser;
        auto* ctx = static_cast<Context*>(ctx_ptr);
        auto* m   = ctx->model;
        DM    dm  = m->get_plex();

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, u_global, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, m->imposed_solution());

        Vec f_int_local;
        DMGetLocalVector(dm, &f_int_local);
        VecSet(f_int_local, 0.0);

        for (auto& elem : m->elements())
            elem.compute_internal_forces(u_local, f_int_local);

        // ── Penalty coupling for embedded rebar nodes ────────────
        if (ctx->penalty_couplings && !ctx->penalty_couplings->empty()) {
            PetscSection sec;
            DMGetLocalSection(dm, &sec);

            const PetscScalar* u_arr;
            VecGetArrayRead(u_local, &u_arr);
            PetscScalar* f_arr;
            VecGetArray(f_int_local, &f_arr);

            const double alpha = ctx->alpha_penalty;
            for (const auto& pc : *ctx->penalty_couplings) {
                PetscInt r_off;
                PetscSectionGetOffset(sec, pc.rebar_sieve_pt, &r_off);

                double u_interp[3] = {0.0, 0.0, 0.0};
                for (const auto& [sp, Ni] : pc.hex_weights) {
                    PetscInt h_off;
                    PetscSectionGetOffset(sec, sp, &h_off);
                    for (int d = 0; d < 3; ++d)
                        u_interp[d] += Ni * u_arr[h_off + d];
                }

                for (int d = 0; d < 3; ++d) {
                    const double gap = u_arr[r_off + d] - u_interp[d];
                    f_arr[r_off + d] += alpha * gap;
                    for (const auto& [sp, Ni] : pc.hex_weights) {
                        PetscInt h_off;
                        PetscSectionGetOffset(sec, sp, &h_off);
                        f_arr[h_off + d] -= alpha * Ni * gap;
                    }
                }
            }

            VecRestoreArrayRead(u_local, &u_arr);
            VecRestoreArray(f_int_local, &f_arr);
        }

        VecSet(R_out, 0.0);
        DMLocalToGlobal(dm, f_int_local, ADD_VALUES, R_out);
        VecAXPY(R_out, -1.0, ctx->f_ext);

        DMRestoreLocalVector(dm, &u_local);
        DMRestoreLocalVector(dm, &f_int_local);
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    static PetscErrorCode FormJacobian(
        SNES /*snes*/, Vec u_global, Mat J_mat, Mat /*P*/, void* ctx_ptr)
    {
        PetscFunctionBeginUser;
        auto* ctx = static_cast<Context*>(ctx_ptr);
        auto* m   = ctx->model;
        DM    dm  = m->get_plex();

        MatZeroEntries(J_mat);

        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, u_global, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, m->imposed_solution());

        for (auto& elem : m->elements())
            elem.inject_tangent_stiffness(u_local, J_mat);

        // ── Penalty stiffness for embedded rebar coupling ────────
        if (ctx->penalty_couplings && !ctx->penalty_couplings->empty()) {
            add_penalty_coupling_entries_to_jacobian(
                *ctx->penalty_couplings, ctx->alpha_penalty, J_mat, dm);
        }

        MatAssemblyBegin(J_mat, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(J_mat, MAT_FINAL_ASSEMBLY);

        DMRestoreLocalVector(dm, &u_local);
        PetscFunctionReturn(PETSC_SUCCESS);
    }


public:

    [[nodiscard]] static double default_local_length_scale_mm(
        const MultiscaleSubModel& sub) noexcept
    {
        return default_local_length_scale_mm_(sub);
    }

    NonlinearSubModelEvolver(MultiscaleSubModel& sub, double fc_MPa,
                             std::string output_dir, int vtk_interval = 1)
        : sub_{&sub}
        , fc_{fc_MPa}
        , concrete_factory_{make_default_submodel_concrete_factory(
              fc_MPa, default_local_length_scale_mm_(sub))}
        , rebar_factory_{make_default_submodel_rebar_factory()}
        , output_writer_{std::move(output_dir), sub.parent_element_id}
        , vtk_interval_{vtk_interval}
    {}

    NonlinearSubModelEvolver(MultiscaleSubModel& sub, double fc_MPa,
                             std::unique_ptr<ConcreteMaterialFactory> concrete_factory,
                             std::unique_ptr<RebarMaterialFactory> rebar_factory,
                             std::string output_dir, int vtk_interval = 1)
        : sub_{&sub}
        , fc_{fc_MPa}
        , concrete_factory_{std::move(concrete_factory)}
        , rebar_factory_{std::move(rebar_factory)}
        , output_writer_{std::move(output_dir), sub.parent_element_id}
        , vtk_interval_{vtk_interval}
    {}

    ~NonlinearSubModelEvolver() {
        destroy_petsc_objects();
    }

    /// Plan v2 §Fase 1.6 — runtime-adjustable VTK throttle.
    ///
    /// Allows the multiscale orchestrator to retune output frequency
    /// per-stage (e.g., dense for `enriched_fe2_guarded_smoke`, sparse
    /// for `local_site_batch`). Use `<= 0` to disable VTK output entirely
    /// while keeping the writer alive (matches existing logic at the
    /// emit site).
    void set_vtk_interval(int vtk_interval) noexcept {
        vtk_interval_ = vtk_interval;
    }

    [[nodiscard]] int vtk_interval() const noexcept { return vtk_interval_; }

    // Non-copyable
    NonlinearSubModelEvolver(const NonlinearSubModelEvolver&) = delete;
    NonlinearSubModelEvolver& operator=(const NonlinearSubModelEvolver&) = delete;

    // Movable
    NonlinearSubModelEvolver(NonlinearSubModelEvolver&& o) noexcept
        : sub_{o.sub_}, fc_{o.fc_}
        , local_ex_{o.local_ex_}, local_ey_{o.local_ey_}, local_ez_{o.local_ez_}
        , concrete_factory_{std::move(o.concrete_factory_)}
        , rebar_factory_{std::move(o.rebar_factory_)}
        , model_{std::move(o.model_)}, model_ready_{o.model_ready_}
        , snes_{std::exchange(o.snes_, nullptr)}
        , U_{std::exchange(o.U_, nullptr)}
        , R_{std::exchange(o.R_, nullptr)}
        , f_ext_{std::exchange(o.f_ext_, nullptr)}
        , J_{std::exchange(o.J_, nullptr)}
        , output_writer_{std::move(o.output_writer_)}
        , vtk_interval_{o.vtk_interval_}
        , step_count_{o.step_count_}
        , auto_commit_{o.auto_commit_}
        , latest_cracks_{std::move(o.latest_cracks_)}
        , latest_crack_summary_{o.latest_crack_summary_}
        , last_attempted_crack_summary_{o.last_attempted_crack_summary_}
        , last_solve_result_{o.last_solve_result_}
        , first_step_increments_{o.first_step_increments_}
        , first_step_bisect_{o.first_step_bisect_}
        , use_arc_length_{o.use_arc_length_}
        , consecutive_divergences_{o.consecutive_divergences_}
        , last_good_frac_{o.last_good_frac_}
        , arc_length_threshold_{o.arc_length_threshold_}
        , adaptive_max_substeps_{o.adaptive_max_substeps_}
        , adaptive_max_bisections_{o.adaptive_max_bisections_}
        , adaptive_tail_rescue_attempts_{o.adaptive_tail_rescue_attempts_}
        , adaptive_tail_rescue_progress_threshold_{
              o.adaptive_tail_rescue_progress_threshold_}
        , adaptive_tail_rescue_substep_bonus_{
              o.adaptive_tail_rescue_substep_bonus_}
        , adaptive_tail_rescue_bisection_bonus_{
              o.adaptive_tail_rescue_bisection_bonus_}
        , adaptive_tail_rescue_initial_fraction_{
              o.adaptive_tail_rescue_initial_fraction_}
        , min_crack_opening_{o.min_crack_opening_}
        , alpha_penalty_{o.alpha_penalty_}
        , snes_max_it_{o.snes_max_it_}
        , snes_atol_{o.snes_atol_}
        , snes_rtol_{o.snes_rtol_}
        , regularization_policy_{o.regularization_policy_}
        , diagonal_floor_{o.diagonal_floor_}
        , tangent_mode_{o.tangent_mode_}
        , tangent_validation_relative_tolerance_{
              o.tangent_validation_relative_tolerance_}
        , tangent_validation_max_column_tolerance_{
              o.tangent_validation_max_column_tolerance_}
        , tangent_validation_norm_{o.tangent_validation_norm_}
        , penalty_couplings_{std::move(o.penalty_couplings_)}
        , condensed_workspace_{std::move(o.condensed_workspace_)}
        , state_transfer_callback_{std::move(o.state_transfer_callback_)}
    {
        if (!condensed_workspace_) {
            condensed_workspace_ = std::make_unique<CondensationWorkspace>();
        }
        ctx_ = {model_.get(), f_ext_,
                &penalty_couplings_, alpha_penalty_};
    }

    NonlinearSubModelEvolver& operator=(NonlinearSubModelEvolver&& o) noexcept {
        if (this != &o) {
            destroy_petsc_objects();
            sub_ = o.sub_; fc_ = o.fc_;
            local_ex_ = o.local_ex_; local_ey_ = o.local_ey_; local_ez_ = o.local_ez_;
            concrete_factory_ = std::move(o.concrete_factory_);
            rebar_factory_    = std::move(o.rebar_factory_);
            model_      = std::move(o.model_);
            model_ready_ = o.model_ready_;
            snes_  = std::exchange(o.snes_, nullptr);
            U_     = std::exchange(o.U_, nullptr);
            R_     = std::exchange(o.R_, nullptr);
            f_ext_ = std::exchange(o.f_ext_, nullptr);
            J_     = std::exchange(o.J_, nullptr);
            output_writer_ = std::move(o.output_writer_);
            vtk_interval_ = o.vtk_interval_;
            step_count_    = o.step_count_;
            auto_commit_   = o.auto_commit_;
            latest_cracks_ = std::move(o.latest_cracks_);
            latest_crack_summary_ = o.latest_crack_summary_;
            last_attempted_crack_summary_ = o.last_attempted_crack_summary_;
            last_solve_result_ = o.last_solve_result_;
            first_step_increments_ = o.first_step_increments_;
            first_step_bisect_     = o.first_step_bisect_;
            use_arc_length_          = o.use_arc_length_;
            consecutive_divergences_ = o.consecutive_divergences_;
            last_good_frac_          = o.last_good_frac_;
            arc_length_threshold_    = o.arc_length_threshold_;
            adaptive_max_substeps_   = o.adaptive_max_substeps_;
            adaptive_max_bisections_ = o.adaptive_max_bisections_;
            adaptive_tail_rescue_attempts_ =
                o.adaptive_tail_rescue_attempts_;
            adaptive_tail_rescue_progress_threshold_ =
                o.adaptive_tail_rescue_progress_threshold_;
            adaptive_tail_rescue_substep_bonus_ =
                o.adaptive_tail_rescue_substep_bonus_;
            adaptive_tail_rescue_bisection_bonus_ =
                o.adaptive_tail_rescue_bisection_bonus_;
            adaptive_tail_rescue_initial_fraction_ =
                o.adaptive_tail_rescue_initial_fraction_;
            min_crack_opening_       = o.min_crack_opening_;
            alpha_penalty_           = o.alpha_penalty_;
            snes_max_it_             = o.snes_max_it_;
            snes_atol_               = o.snes_atol_;
            snes_rtol_               = o.snes_rtol_;
            regularization_policy_   = o.regularization_policy_;
            diagonal_floor_          = o.diagonal_floor_;
            tangent_mode_            = o.tangent_mode_;
            tangent_validation_relative_tolerance_ =
                o.tangent_validation_relative_tolerance_;
            tangent_validation_max_column_tolerance_ =
                o.tangent_validation_max_column_tolerance_;
            tangent_validation_norm_ = o.tangent_validation_norm_;
            penalty_couplings_       = std::move(o.penalty_couplings_);
            condensed_workspace_     = std::move(o.condensed_workspace_);
            if (!condensed_workspace_) {
                condensed_workspace_ =
                    std::make_unique<CondensationWorkspace>();
            }
            ctx_ = {model_.get(), f_ext_,
                    &penalty_couplings_, alpha_penalty_};
        }
        return *this;
    }

    struct SolverCheckpoint {
        petsc::OwnedVec displacement{};
        petsc::OwnedVec imposed_solution{};
        std::optional<ModelCheckpointT> model{};
        SectionKinematics kin_A{};
        SectionKinematics kin_B{};
        bool model_initialized{false};
        bool arc_length_active{false};
        int consecutive_divergences{0};
        double last_good_fraction{0.5};
    };

    using checkpoint_type = SolverCheckpoint;


    //  Configuration 

    void set_incremental_params(int num_steps, int max_bisect) {
        first_step_increments_ = num_steps;
        first_step_bisect_     = max_bisect;
    }

    void set_local_axes(const std::array<double,3>& ex,
                        const std::array<double,3>& ey,
                        const std::array<double,3>& ez) {
        local_ex_ = ex; local_ey_ = ey; local_ez_ = ez;
    }

    void set_rebar_material(double E, double fy, double b) {
        rebar_E_ = E; rebar_fy_ = fy; rebar_b_ = b;
    }

    void enable_arc_length(bool flag = true) { use_arc_length_ = flag; }
    void set_arc_length_threshold(int t) { arc_length_threshold_ = t; }
    void set_adaptive_substepping_limits(int max_substeps,
                                         int max_bisections) noexcept
    {
        adaptive_max_substeps_ = std::max(1, max_substeps);
        adaptive_max_bisections_ = std::max(1, max_bisections);
    }

    void set_adaptive_tail_rescue_policy(int max_attempts,
                                         double progress_threshold,
                                         int substep_bonus,
                                         int bisection_bonus,
                                         double initial_fraction = 0.5) noexcept
    {
        adaptive_tail_rescue_attempts_ = std::max(0, max_attempts);
        adaptive_tail_rescue_progress_threshold_ =
            std::clamp(progress_threshold, 0.0, 1.0);
        adaptive_tail_rescue_substep_bonus_ = std::max(0, substep_bonus);
        adaptive_tail_rescue_bisection_bonus_ = std::max(0, bisection_bonus);
        adaptive_tail_rescue_initial_fraction_ =
            std::clamp(initial_fraction, 1.0e-12, 1.0);
    }
    [[nodiscard]] bool arc_length_active() const noexcept { return use_arc_length_; }

    /// Set minimum crack opening (strain units) for VTK export filtering.
    /// Cracks with opening below this threshold are excluded from the
    /// crack plane VTU to reduce visual noise.  Default: 0.5e-3.
    void set_min_crack_opening(double thr) noexcept { min_crack_opening_ = thr; }

    void set_penalty_alpha(double alpha) noexcept { alpha_penalty_ = alpha; }

    void set_snes_params(int max_it, double atol, double rtol) noexcept {
        snes_max_it_ = max_it;
        snes_atol_   = atol;
        snes_rtol_   = rtol;
    }

    void set_auto_commit(bool enabled) noexcept { auto_commit_ = enabled; }
    [[nodiscard]] bool auto_commit() const noexcept { return auto_commit_; }

    void set_regularization_policy(
        RegularizationPolicyKind policy,
        double diagonal_floor = 1.0) noexcept
    {
        regularization_policy_ = policy;
        diagonal_floor_ = diagonal_floor;
    }

    void set_tangent_computation_mode(
        TangentComputationMode mode) noexcept
    {
        tangent_mode_ = mode;
    }

    void set_tangent_validation_relative_tolerance(
        double tolerance) noexcept
    {
        tangent_validation_relative_tolerance_ = std::max(0.0, tolerance);
    }

    void set_tangent_validation_max_column_tolerance(
        double tolerance) noexcept
    {
        tangent_validation_max_column_tolerance_ = std::max(0.0, tolerance);
    }

    void set_tangent_validation_norm(
        TangentValidationNormKind norm) noexcept
    {
        tangent_validation_norm_ = norm;
    }

    [[nodiscard]] TangentComputationMode tangent_computation_mode()
        const noexcept
    {
        return tangent_mode_;
    }

    [[nodiscard]] double tangent_validation_relative_tolerance()
        const noexcept
    {
        return tangent_validation_relative_tolerance_;
    }

    [[nodiscard]] double tangent_validation_max_column_tolerance()
        const noexcept
    {
        return tangent_validation_max_column_tolerance_;
    }

    [[nodiscard]] TangentValidationNormKind tangent_validation_norm()
        const noexcept
    {
        return tangent_validation_norm_;
    }

    // ── FE² State Transfer ─────────────────────────────────────────
    //
    //  When a state_transfer_callback is registered, first_solve() will
    //  invoke it after model setup but BEFORE the incremental ramp.
    //  The callback receives a mutable reference to the element container
    //  so it can call inject_material_state() on individual elements.
    //
    //  Usage:
    //    evolver.set_state_transfer([&](auto& elems) {
    //        for (auto& pkt : rebar_packets)
    //            elems[truss_offset + pkt.bar_index]
    //                .inject_material_state(pkt.internal_state);
    //    });

    using StateTransferCallback =
        std::function<void(typename MixedModel::container_type&)>;

    void set_state_transfer(StateTransferCallback cb) {
        state_transfer_callback_ = std::move(cb);
    }

    void commit_trial_state() {
        commit_state();
    }

    [[nodiscard]] checkpoint_type capture_checkpoint() const {
        checkpoint_type checkpoint;
        checkpoint.kin_A = sub_->kin_A;
        checkpoint.kin_B = sub_->kin_B;
        checkpoint.model_initialized = model_ready_;
        checkpoint.arc_length_active = use_arc_length_;
        checkpoint.consecutive_divergences = consecutive_divergences_;
        checkpoint.last_good_fraction = last_good_frac_;

        if (model_ready_ && U_ && model_) {
            checkpoint.model.emplace(model_->capture_checkpoint());
            FALL_N_PETSC_CHECK(VecDuplicate(U_,
                                            checkpoint.displacement.ptr()));
            FALL_N_PETSC_CHECK(VecCopy(U_,
                                       checkpoint.displacement.get()));
            FALL_N_PETSC_CHECK(VecDuplicate(model_->imposed_solution(),
                                            checkpoint.imposed_solution.ptr()));
            FALL_N_PETSC_CHECK(VecCopy(model_->imposed_solution(),
                                       checkpoint.imposed_solution.get()));
        }

        return checkpoint;
    }

    void restore_checkpoint(const checkpoint_type& checkpoint) {
        make_bc_applicator_().update_kinematics(
            checkpoint.kin_A, checkpoint.kin_B);
        use_arc_length_ = checkpoint.arc_length_active;
        consecutive_divergences_ = checkpoint.consecutive_divergences;
        last_good_frac_ = checkpoint.last_good_fraction;

        if (!checkpoint.model_initialized) {
            reset_transient_model_();
            return;
        }

        if (!model_ || !U_) {
            return;
        }

        if (checkpoint.model) {
            model_->restore_checkpoint(*checkpoint.model);
        } else {
            make_state_ops_().revert_state();
        }

        if (checkpoint.displacement) {
            FALL_N_PETSC_CHECK(VecCopy(checkpoint.displacement.get(), U_));
        } else {
            FALL_N_PETSC_CHECK(VecSet(U_, 0.0));
        }

        if (checkpoint.imposed_solution) {
            FALL_N_PETSC_CHECK(VecCopy(checkpoint.imposed_solution.get(),
                                       model_->imposed_solution()));
        } else {
            make_bc_applicator_().write_imposed_values();
        }

        make_state_ops_().sync_state_vector();
        model_ready_ = true;
    }


    //  BC update 

    void update_kinematics(const SectionKinematics& kin_A,
                           const SectionKinematics& kin_B)
    {
        make_bc_applicator_().update_kinematics(kin_A, kin_B);
    }


    //  Solve one time step 

    SubModelSolverResult solve_step(double /*time*/) {

        SubModelSolverResult result;

        if (!model_ready_)
            result = first_solve();
        else
            result = subsequent_solve();

        last_solve_result_ = result;
        if (model_ready_ && model_ && U_) {
            last_attempted_crack_summary_ =
                LocalCrackDiagnostics<MixedModel>::collect(
                    *model_, *sub_, U_, min_crack_opening_, false)
                    .summary;
        } else {
            last_attempted_crack_summary_ = {};
        }

        // Crack data & VTK output are NOT performed here during staggered
        // iterations — the caller should use end_of_step() once per global
        // time step after the staggered loop converges.  This avoids
        // redundant collect_crack_data() calls (6× per step) and prevents
        // step_count_ from inflating with staggered iterations.

        return result;
    }

    /// Call once per GLOBAL time step, after the staggered loop converges.
    /// Collects crack data, optionally writes VTK, and advances the counter.
    void end_of_step(double time) {
        collect_crack_data();

        if (vtk_interval_ > 0 && step_count_ % vtk_interval_ == 0)
            write_vtk_snapshot(time);

        ++step_count_;
    }

    //  Finalize 

    void finalize() {
        if (vtk_interval_ <= 0) {
            return;
        }
        output_writer_.finalize();
    }

    //  Accessors 

    [[nodiscard]] std::size_t parent_element_id() const noexcept {
        return sub_->parent_element_id;
    }
    [[nodiscard]] int step_count() const noexcept { return step_count_; }
    [[nodiscard]] const MultiscaleSubModel& sub_model() const noexcept {
        return *sub_;
    }
    [[nodiscard]] MultiscaleSubModel& sub_model() noexcept { return *sub_; }
    [[nodiscard]] const std::vector<CrackRecord>& latest_cracks() const noexcept {
        return latest_cracks_;
    }
    [[nodiscard]] const SubModelSolverResult& last_solve_result() const noexcept {
        return last_solve_result_;
    }

    [[nodiscard]] CrackSummary crack_summary() const noexcept {
        return latest_crack_summary_;
    }

    [[nodiscard]] CrackSummary last_attempted_crack_summary() const noexcept {
        return last_attempted_crack_summary_;
    }


    // ═══════════════════════════════════════════════════════════════════
    //  Homogenised section tangent via perturbation
    // ═══════════════════════════════════════════════════════════════════
    //
    // ═══════════════════════════════════════════════════════════════════
    //  Section forces from assembled internal-force reactions at Face B
    // ═══════════════════════════════════════════════════════════════════
    //
    //  Assembles f_int from the current displacement field and reads the
    //  internal-force contributions at Face B boundary nodes.  The sum of
    //  these nodal forces (and their moments about the centroid) gives the
    //  true 3-D section resultants — bypassing gauss_point_snapshots(),
    //  which reads stored material state that may be stale after a
    //  perturbation SNES solve.
    //
    //  Returns [N, My, Mz, Vy, Vz, Mt] in beam local frame.

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    compute_homogenized_tangent([[maybe_unused]] double width,
                                [[maybe_unused]] double height,
                                double h_pert = 1.0e-6)
    {
        return make_local_homogenizer_(width, height)
            .compute_homogenized_tangent(h_pert);
    }


    // ═══════════════════════════════════════════════════════════════════
    //  Homogenised section forces from current sub-model state
    // ═══════════════════════════════════════════════════════════════════
    //
    //  Returns a 6-component vector [N, My, Mz, Vy, Vz, Mt] from the
    //  boundary-reaction operator, keeping forces consistent with the
    //  tangent extracted from the same physical quantity.

    [[nodiscard]] Eigen::Vector<double, 6>
    compute_homogenized_forces([[maybe_unused]] double width,
                               [[maybe_unused]] double height) {
        if (!model_ready_) return Eigen::Vector<double, 6>::Zero();
        return make_local_homogenizer_(width, height)
            .compute_homogenized_forces();
    }

    [[nodiscard]] Eigen::Vector<double, 6>
    compute_volume_average_forces(double width, double height) {
        if (!model_ready_) return Eigen::Vector<double, 6>::Zero();

        SubModelSolverResult result = extract_results(true);
        HomogenizedBeamSection hs = homogenize(result, *sub_, width, height);

        Eigen::Vector<double, 6> f_hom;
        f_hom << hs.N, hs.My, hs.Mz, hs.Vy, hs.Vz, 0.0;
        return f_hom;
    }


    // ═══════════════════════════════════════════════════════════════════
    //  LocalModelAdapter concept conformance
    // ═══════════════════════════════════════════════════════════════════

    [[nodiscard]] Eigen::Matrix<double, 6, 6>
    section_tangent(double width, double height, double h_pert = 1.0e-6) {
        return compute_homogenized_tangent(width, height, h_pert);
    }

    [[nodiscard]] Eigen::Vector<double, 6>
    section_forces(double width, double height) {
        return compute_homogenized_forces(width, height);
    }

    [[nodiscard]] SectionHomogenizedResponse
    section_response(double width, double height, double h_pert = 1.0e-6) {
        return make_local_homogenizer_(width, height).section_response(h_pert);
    }

    void sync_state_vector_() {
        make_state_ops_().sync_state_vector();
    }

    void commit_state() {
        make_state_ops_().commit_state();
    }

    void revert_state() {
        make_state_ops_().revert_state();
    }


private:

    //  Destroy PETSc objects 

    void destroy_petsc_objects() {
        if (U_)             { VecDestroy(&U_);             U_             = nullptr; }
        if (R_)             { VecDestroy(&R_);             R_             = nullptr; }
        if (f_ext_)         { VecDestroy(&f_ext_);         f_ext_         = nullptr; }
        if (U_work_)        { VecDestroy(&U_work_);        U_work_        = nullptr; }
        if (imp_work_)      { VecDestroy(&imp_work_);      imp_work_      = nullptr; }
        if (U_checkpoint_)  { VecDestroy(&U_checkpoint_);  U_checkpoint_  = nullptr; }
        if (imp_checkpoint_){ VecDestroy(&imp_checkpoint_);imp_checkpoint_= nullptr; }
        if (J_)     { MatDestroy(&J_);     J_     = nullptr; }
        if (snes_)  { SNESDestroy(&snes_); snes_  = nullptr; }
    }


    //  Write BC values into imposed_solution 

    void write_imposed_values() {
        make_bc_applicator_().write_imposed_values();
    }


    //  1-D Lagrange shape function for linear (n=2) or quadratic (n=3) 

    static double shape_value_1d(int n, int i, double t) noexcept {
        if (n == 2)
            return (i == 0) ? (1.0 - t) * 0.5 : (1.0 + t) * 0.5;
        switch (i) {
            case 0: return t * (t - 1.0) * 0.5;
            case 1: return (1.0 - t) * (1.0 + t);
            case 2: return t * (t + 1.0) * 0.5;
            default: return 0.0;
        }
    }


    //  Pre-compute penalty coupling data for embedded rebar 
    //
    //  For each INTERIOR rebar node (not on MinZ/MaxZ faces),
    //  evaluates the host hex element's shape functions at the
    //  rebar node's parent coordinates (ξ, η, ζ) and stores the
    //  resulting (sieve_point, Nᵢ) pairs for use in FormResidual
    //  and FormJacobian.

    void compute_penalty_couplings() {
        penalty_couplings_.clear();

        if (!sub_->has_rebar() || sub_->rebar_embeddings.empty())
            return;

        const auto& grid = sub_->grid;
        const int step   = grid.step;
        const int nz     = grid.nz;
        const int n_per  = (step == 1) ? 2 : 3;  // nodes per element dir

        const std::size_t num_bars = sub_->rebar_diameters.size();
        const std::size_t rpb =
            static_cast<std::size_t>(step * nz + 1);

        auto& domain = sub_->domain;

        for (std::size_t b = 0; b < num_bars; ++b) {
            for (std::size_t iz_sub = 0; iz_sub < rpb; ++iz_sub) {
                // Skip face nodes — they have Dirichlet BCs
                if (iz_sub == 0 || iz_sub == rpb - 1)
                    continue;

                const std::size_t idx = b * rpb + iz_sub;
                const auto& emb = sub_->rebar_embeddings[idx];

                PetscInt rebar_sieve =
                    domain.node(static_cast<std::size_t>(
                        emb.rebar_node_id)).sieve_id.value();

                PenaltyCouplingEntry pc;
                pc.rebar_sieve_pt = rebar_sieve;

                for (int i2 = 0; i2 < n_per; ++i2) {
                    for (int i1 = 0; i1 < n_per; ++i1) {
                        for (int i0 = 0; i0 < n_per; ++i0) {
                            int gix = step * emb.host_elem_ix + i0;
                            int giy = step * emb.host_elem_iy + i1;
                            int giz = step * emb.host_elem_iz + i2;

                            PetscInt hnid = grid.node_id(gix, giy, giz);
                            PetscInt hsieve =
                                domain.node(static_cast<std::size_t>(hnid))
                                      .sieve_id.value();

                            double Ni =
                                shape_value_1d(n_per, i0, emb.xi)
                              * shape_value_1d(n_per, i1, emb.eta)
                              * shape_value_1d(n_per, i2, emb.zeta);

                            if (std::abs(Ni) > 1e-15)
                                pc.hex_weights.emplace_back(hsieve, Ni);
                        }
                    }
                }

                penalty_couplings_.push_back(std::move(pc));
            }
        }

        std::println("  [SubModel {}] Penalty coupling: {} interior rebar "
                     "nodes, α = {:.1e}",
                     sub_->parent_element_id,
                     penalty_couplings_.size(), alpha_penalty_);
    }


    //  Set up persistent SNES 

    void setup_snes() {
        DM dm = model_->get_plex();

        DMCreateGlobalVector(dm, &U_);
        VecDuplicate(U_, &R_);
        VecDuplicate(U_, &f_ext_);
        VecDuplicate(U_, &U_work_);
        VecDuplicate(U_, &U_checkpoint_);
        VecDuplicate(model_->imposed_solution(), &imp_work_);
        VecDuplicate(model_->imposed_solution(), &imp_checkpoint_);
        DMCreateMatrix(dm, &J_);
        MatSetOption(J_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

        VecSet(U_, 0.0);
        VecSet(f_ext_, 0.0);

        ctx_ = {model_.get(), f_ext_,
                &penalty_couplings_, alpha_penalty_};

        // Use l2 line search — minimizes ||F(x + αΔx)||² along the Newton
        // direction.  bt rejects too many steps with secant tangents (slow),
        // basic allows NaN-producing overshoots (DIVERGED_FUNCTION_DOMAIN).
        PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "l2");
        PetscOptionsSetValue(nullptr, "-snes_max_it",
                             std::to_string(snes_max_it_).c_str());
        PetscOptionsSetValue(nullptr, "-snes_atol",
                             std::to_string(snes_atol_).c_str());
        PetscOptionsSetValue(nullptr, "-snes_rtol",
                             std::to_string(snes_rtol_).c_str());
        PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
        PetscOptionsSetValue(nullptr, "-pc_type", "lu");

        SNESCreate(PETSC_COMM_SELF, &snes_);
        SNESSetFunction(snes_, R_, FormResidual, &ctx_);
        SNESSetJacobian(snes_, J_, J_, FormJacobian, &ctx_);
        SNESSetFromOptions(snes_);

        // ── RCM reordering on LU factorizer ──────────────────────────
        //  Reorders within the direct solver for reduced fill-in.
        //  ReuseOrdering caches symbolic factorization across steps.
        {
            KSP ksp;
            SNESGetKSP(snes_, &ksp);
            PC pc;
            KSPGetPC(ksp, &pc);
            PCFactorSetMatOrderingType(pc, MATORDERINGRCM);
            PCFactorSetReuseOrdering(pc, PETSC_TRUE);
        }
    }


    //  First solve: build model + incremental loading from zero 

    SubModelSolverResult first_solve() {
        // 1. Build material + model (via injectable factories)
        Material<Policy> mat = concrete_factory_->create();

        // Build element list (ContinuumElement for hex + TrussElement for rebar)
        std::vector<FEM_Element> elems;

        const std::size_t rebar_first = sub_->has_rebar()
            ? sub_->rebar_range.first : sub_->domain.num_elements();

        {
            std::size_t idx = 0;
            for (auto& geom : sub_->domain.elements()) {
                if (idx >= rebar_first) break;
                elems.emplace_back(ContElem{&geom, mat});
                ++idx;
            }
        }

        // Optional rebar truss elements
        if (sub_->has_rebar()) {
            Material<UniaxialMaterial> steel_mat = rebar_factory_->create();

            const int nz = sub_->grid.nz;
            std::size_t bar_idx = 0;
            for (std::size_t i = sub_->rebar_range.first;
                 i < sub_->rebar_range.last; ++i)
            {
                auto& geom = sub_->domain.elements()[i];
                double area = sub_->rebar_areas[bar_idx / static_cast<std::size_t>(nz)];
                elems.emplace_back(TrussElement<3>{&geom, steel_mat, area});
                ++bar_idx;
            }
        }

        model_ = std::make_unique<MixedModel>(sub_->domain, std::move(elems));

        // 2. Apply Dirichlet BCs (constrain_node must precede setup)
        for (const auto& [nid, u] : sub_->bc_min_z)
            model_->constrain_node(nid, {u[0], u[1], u[2]});
        for (const auto& [nid, u] : sub_->bc_max_z)
            model_->constrain_node(nid, {u[0], u[1], u[2]});

        model_->setup();

        // 2.5. Pre-compute penalty coupling for embedded rebar
        compute_penalty_couplings();
        condensed_workspace_->reset();

        // 3. Set up persistent SNES + allocate U, R, J
        setup_snes();

        // 3.5. FE² state transfer: inject material history from macro-beam
        //      into micro-model elements BEFORE the incremental ramp.
        //      This allows the sub-model's rebar truss elements to start
        //      with the correct cyclic history (reversal points, hardening)
        //      and continuum elements to receive initial stress/damage.
        if (state_transfer_callback_) {
            state_transfer_callback_(model_->elements());
        }

        // 4. Save target imposed values, then do incremental loading
        Vec target;
        VecDuplicate(model_->imposed_solution(), &target);
        VecCopy(model_->imposed_solution(), target);
        const int N = first_step_increments_;

        // Debug: check target BC norm
        {
            PetscReal tnorm;
            VecNorm(target, NORM_INFINITY, &tnorm);
            std::println("  [first_solve] target ||imposed||_inf = {:.6e}", tnorm);
            std::println("  [first_solve] N={} increments, snes_max_it={}, atol={:.2e}, rtol={:.2e}",
                         N, snes_max_it_, snes_atol_, snes_rtol_);
        }

        Vec imp_zero;
        VecDuplicate(model_->imposed_solution(), &imp_zero);
        VecSet(imp_zero, 0.0);
        VecCopy(imp_zero, model_->imposed_solution());

        const bool use_adaptive_first_ramp =
            use_arc_length_ || first_step_bisect_ > 0;
        AdaptiveRampSolveStats adaptive_stats{};
        bool converged = true;
        int completed_substeps = 0;
        int last_reason = 0;
        PetscInt last_nits = 0;
        double last_fnorm = 0.0;
        int total_bisections = 0;
        double minimum_step_fraction =
            (N > 0) ? 1.0 / static_cast<double>(N) : 0.0;
        double failed_target_fraction = 0.0;
        double failed_step_fraction = minimum_step_fraction;
        int failed_substep_index = 0;
        SubModelFailureCause failure_cause = SubModelFailureCause::None;

        if (use_adaptive_first_ramp) {
            const int max_substeps = std::max(
                adaptive_max_substeps_,
                std::max(1, N) * (std::max(0, first_step_bisect_) + 1));
            adaptive_stats = solve_ramp_with_tail_rescue_(
                imp_zero,
                target,
                (N > 0) ? 1.0 / static_cast<double>(N) : 1.0,
                first_step_bisect_,
                max_substeps,
                false);
            converged = adaptive_stats.converged;
            completed_substeps = adaptive_stats.total_substeps;
            last_reason = adaptive_stats.last_reason;
            last_nits = adaptive_stats.last_snes_iterations;
            last_fnorm = adaptive_stats.last_function_norm;
            total_bisections = adaptive_stats.total_bisections;
            minimum_step_fraction = adaptive_stats.minimum_step_fraction;
            failed_target_fraction = adaptive_stats.failed_target_fraction;
            failed_step_fraction = adaptive_stats.failed_step_fraction;
            failed_substep_index = adaptive_stats.failed_substep_index;
            failure_cause = adaptive_stats.failure_cause;
        } else {
            for (int k = 1; k <= N; ++k) {
                const double p =
                    static_cast<double>(k) / static_cast<double>(N);

                VecCopy(target, model_->imposed_solution());
                VecScale(model_->imposed_solution(), p);

                SNESSolve(snes_, nullptr, U_);

                SNESConvergedReason reason;
                SNESGetConvergedReason(snes_, &reason);

                PetscInt nits;
                SNESGetIterationNumber(snes_, &nits);

                PetscReal fnorm;
                SNESGetFunctionNorm(snes_, &fnorm);
                last_reason = static_cast<int>(reason);
                last_nits = nits;
                last_fnorm = static_cast<double>(fnorm);

                std::println(
                    "  [first_solve] k={}/{} p={:.4f} reason={} iters={} ||F||={:.4e}",
                    k,
                    N,
                    p,
                    static_cast<int>(reason),
                    nits,
                    fnorm);

                if (reason > 0) {
                    commit_state();
                    completed_substeps = k;
                } else {
                    revert_state();
                    converged = false;
                    failed_substep_index = completed_substeps + 1;
                    failed_target_fraction =
                        (N > 0)
                            ? static_cast<double>(failed_substep_index)
                                  / static_cast<double>(N)
                            : 0.0;
                    failure_cause = classify_snes_failure_(reason);
                    break;
                }
            }
        }

        VecDestroy(&target);
        VecDestroy(&imp_zero);
        model_ready_ = true;

        const auto material_diagnostics = collect_material_diagnostics_();
        auto result = extract_results(converged);
        result.stage = SubModelSolveStage::FirstSolveRamp;
        result.failure_cause = converged
            ? SubModelFailureCause::None
            : failure_cause;
        result.snes_reason = last_reason;
        result.snes_iterations = last_nits;
        result.function_norm = last_fnorm;
        result.adaptive_substeps = completed_substeps;
        result.adaptive_bisections = total_bisections;
        result.adaptive_tail_rescue_attempts =
            adaptive_stats.total_tail_rescue_attempts;
        result.achieved_fraction =
            use_adaptive_first_ramp
                ? adaptive_stats.achieved_fraction
                : ((N > 0)
                       ? static_cast<double>(completed_substeps)
                             / static_cast<double>(N)
                       : 0.0);
        result.adaptive_tail_rescue_trigger_fraction =
            adaptive_stats.tail_rescue_trigger_fraction;
        result.used_arc_length = use_arc_length_;
        result.failed_substep_index =
            converged ? 0 : failed_substep_index;
        result.failed_target_fraction =
            converged
                ? result.achieved_fraction
                : failed_target_fraction;
        result.failed_step_fraction = failed_step_fraction;
        result.minimum_step_fraction = minimum_step_fraction;
        apply_material_diagnostics_(result, material_diagnostics);
        return result;
    }


    //  Subsequent solve: update BCs, Newton from current state 
    //
    //  If SNES diverges repeatedly (≥ arc_length_threshold_ consecutive
    //  failures), the solver automatically enables arc-length control for
    //  subsequent steps.  Once arc-length is active, the solver uses adaptive
    //  displacement sub-stepping — the BC increment is subdivided and each
    //  sub-step is solved by Newton; on failure the step size is bisected.

    SubModelSolverResult subsequent_solve() {
        if (use_arc_length_)
            return subsequent_solve_adaptive();

        // ── Standard path: full-step SNES ────────────────────────────

        // Save imposed values and U (pre-allocated work vectors)
        VecCopy(model_->imposed_solution(), imp_work_);

        write_imposed_values();

        // Save U before SNES attempt so we can restore on failure
        VecCopy(U_, U_work_);

        // Newton from current U (NOT reset to zero)
        SNESSolve(snes_, nullptr, U_);

        SNESConvergedReason reason;
        SNESGetConvergedReason(snes_, &reason);
        PetscInt nits = 0;
        SNESGetIterationNumber(snes_, &nits);
        PetscReal fnorm = 0.0;
        SNESGetFunctionNorm(snes_, &fnorm);

        bool converged = (reason > 0);
        if (converged) {
            if (auto_commit_)
                commit_state();
            else
                sync_state_vector_();
            consecutive_divergences_ = 0;
        } else {
            VecCopy(U_work_, U_);                          // restore displacement vector
            VecCopy(imp_work_, model_->imposed_solution()); // restore imposed BCs
            revert_state();

            // Auto-switch to arc-length after repeated divergences
            ++consecutive_divergences_;
            if (consecutive_divergences_ >= arc_length_threshold_) {
                use_arc_length_ = true;
                std::println("  [SubModel {}] Auto-enabling adaptive sub-stepping "
                             "after {} consecutive SNES divergences",
                             sub_->parent_element_id,
                             consecutive_divergences_);
            }
        }

        const auto material_diagnostics = collect_material_diagnostics_();
        auto result = extract_results(converged);
        result.stage = SubModelSolveStage::SubsequentFullStep;
        result.failure_cause = converged
            ? SubModelFailureCause::None
            : classify_snes_failure_(reason);
        result.snes_reason = static_cast<int>(reason);
        result.snes_iterations = nits;
        result.function_norm = static_cast<double>(fnorm);
        result.achieved_fraction = converged ? 1.0 : 0.0;
        result.used_arc_length = false;
        result.failed_substep_index = converged ? 0 : 1;
        result.failed_target_fraction = 1.0;
        result.failed_step_fraction = 1.0;
        result.minimum_step_fraction = 1.0;
        apply_material_diagnostics_(result, material_diagnostics);
        return result;
    }


    //  Adaptive displacement sub-stepping (arc-length mode)
    //
    //  Interpolates boundary conditions from the previous converged state
    //  to the full target in sub-increments.  On SNES failure the step
    //  fraction is bisected; on success, the fraction doubles (up to 1).
    //  The method fails only when the minimum step fraction is reached.

    SubModelSolverResult subsequent_solve_adaptive() {
        // 1. Save previous imposed values (last converged step's BCs)
        Vec imp_prev;
        VecDuplicate(model_->imposed_solution(), &imp_prev);
        VecCopy(model_->imposed_solution(), imp_prev);

        // 2. Write full target imposed values
        write_imposed_values();

        Vec imp_target;
        VecDuplicate(model_->imposed_solution(), &imp_target);
        VecCopy(model_->imposed_solution(), imp_target);
        const auto stats = solve_ramp_with_tail_rescue_(
            imp_prev,
            imp_target,
            std::clamp(last_good_frac_, 1.0e-12, 1.0),
            adaptive_max_bisections_,
            adaptive_max_substeps_,
            true);

        // 4. Book-keeping  (last_good_frac_ is updated inside the loop)
        if (stats.converged) {
            consecutive_divergences_ = 0;
        } else {
            if (stats.total_substeps > 0) {
                // Partial progress: keep committed state, reset divergence counter
                consecutive_divergences_ = 0;
            } else {
                ++consecutive_divergences_;
            }
        }

        VecDestroy(&imp_prev);
        VecDestroy(&imp_target);

        const auto material_diagnostics = collect_material_diagnostics_();
        auto result = extract_results(stats.converged);
        result.stage = SubModelSolveStage::SubsequentAdaptiveStep;
        result.failure_cause = stats.converged
            ? SubModelFailureCause::None
            : stats.failure_cause;
        result.snes_reason = stats.last_reason;
        result.snes_iterations = stats.last_snes_iterations;
        result.function_norm = stats.last_function_norm;
        result.adaptive_substeps = stats.total_substeps;
        result.adaptive_bisections = stats.total_bisections;
        result.adaptive_tail_rescue_attempts =
            stats.total_tail_rescue_attempts;
        result.achieved_fraction = stats.achieved_fraction;
        result.adaptive_tail_rescue_trigger_fraction =
            stats.tail_rescue_trigger_fraction;
        result.used_arc_length = true;
        result.failed_target_fraction = stats.failed_target_fraction;
        result.failed_step_fraction = stats.failed_step_fraction;
        result.minimum_step_fraction = stats.minimum_step_fraction;
        result.failed_substep_index =
            stats.converged ? 0 : stats.failed_substep_index;
        apply_material_diagnostics_(result, material_diagnostics);
        return result;
    }


    //  Extract solver results 

    SubModelSolverResult extract_results(bool converged) {
        SubModelSolverResult result;
        result.converged = converged;

        // Max displacement from state_vector
        const PetscScalar* arr;
        PetscInt n;
        VecGetLocalSize(model_->state_vector(), &n);
        VecGetArrayRead(model_->state_vector(), &arr);
        for (PetscInt i = 0; i < n; ++i)
            result.max_displacement = std::max(result.max_displacement,
                                               std::abs(arr[i]));
        VecRestoreArrayRead(model_->state_vector(), &arr);

        // Volume-averaged stress/strain via gauss_point_snapshots
        Eigen::Vector<double, 6> sum_stress = Eigen::Vector<double, 6>::Zero();
        Eigen::Vector<double, 6> sum_strain = Eigen::Vector<double, 6>::Zero();
        double max_vm = 0.0;

        {
            DM dm = model_->get_plex();
            Vec u_local;
            DMGetLocalVector(dm, &u_local);
            VecSet(u_local, 0.0);
            DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
            VecAXPY(u_local, 1.0, model_->imposed_solution());

            for (auto& elem : model_->elements()) {
                for (const auto& snap : elem.gauss_point_snapshots(u_local)) {
                    sum_stress += snap.stress;
                    sum_strain += snap.strain;

                    const auto& sv = snap.stress;
                    const double vm = std::sqrt(std::max(0.0,
                        sv[0]*sv[0] + sv[1]*sv[1] + sv[2]*sv[2]
                      - sv[0]*sv[1] - sv[1]*sv[2] - sv[0]*sv[2]
                      + 3.0*(sv[3]*sv[3] + sv[4]*sv[4] + sv[5]*sv[5])));
                    max_vm = std::max(max_vm, vm);
                    ++result.num_gp;
                }
            }

            DMRestoreLocalVector(dm, &u_local);
        }

        if (result.num_gp > 0) {
            const double inv = 1.0 / static_cast<double>(result.num_gp);
            result.avg_stress = sum_stress * inv;
            result.avg_strain = sum_strain * inv;
        }
        result.max_stress_vm = max_vm;

        constexpr double eps_tol = 1e-15;
        if (std::abs(result.avg_strain[0]) > eps_tol)
            result.E_eff = result.avg_stress[0] / result.avg_strain[0];
        if (std::abs(result.avg_strain[5]) > eps_tol)
            result.G_eff = result.avg_stress[5] / result.avg_strain[5];

        return result;
    }


    //  Crack data collection 

    void collect_crack_data() {
        if (!model_) {
            latest_cracks_.clear();
            latest_crack_summary_ = {};
            last_attempted_crack_summary_ = {};
            return;
        }
        const bool retain_detail = vtk_interval_ > 0;
        const auto crack_state =
            LocalCrackDiagnostics<MixedModel>::collect(
                *model_, *sub_, U_, min_crack_opening_, retain_detail);
        latest_cracks_ = crack_state.cracks;
        latest_crack_summary_ = crack_state.summary;
        last_attempted_crack_summary_ = crack_state.summary;
    }


    //  VTK snapshot 

    void write_vtk_snapshot(double time) {
        output_writer_.write_snapshot(
            time,
            step_count_,
            *model_,
            U_,
            *sub_,
            local_ex_,
            local_ey_,
            local_ez_,
            latest_cracks_,
            min_crack_opening_);
    }

public:

    //  Extract per-bar average axial strain from sub-model rebar 

    [[nodiscard]] std::vector<double> extract_rebar_strains() const {
        if (!sub_->has_rebar() || !model_) return {};

        DM dm = model_->get_plex();
        Vec u_local;
        DMGetLocalVector(dm, &u_local);
        VecSet(u_local, 0.0);
        DMGlobalToLocal(dm, U_, INSERT_VALUES, u_local);
        VecAXPY(u_local, 1.0, model_->imposed_solution());

        const int nz = sub_->grid.nz;
        const std::size_t num_bars = sub_->rebar_diameters.size();
        std::vector<double> bar_strains(num_bars, 0.0);

        std::size_t bar_idx = 0;
        for (std::size_t i = sub_->rebar_range.first;
             i < sub_->rebar_range.last; ++i, ++bar_idx)
        {
            auto& elem = model_->elements()[i];
            auto gf = elem.collect_gauss_fields(u_local);
            if (!gf.empty() && !gf[0].strain.empty()) {
                std::size_t bar = bar_idx / static_cast<std::size_t>(nz);
                if (bar < num_bars)
                    bar_strains[bar] += gf[0].strain[0]
                                      / static_cast<double>(nz);
            }
        }

        DMRestoreLocalVector(dm, &u_local);
        return bar_strains;
    }
};

static_assert(LocalModelAdapter<NonlinearSubModelEvolver>,
    "NonlinearSubModelEvolver must satisfy the LocalModelAdapter concept");


} // namespace fall_n

#endif // FALL_N_SRC_RECONSTRUCTION_NONLINEAR_SUB_MODEL_EVOLVER_HH
