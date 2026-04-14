#ifndef FALL_N_SRC_ANALYSIS_LINEARANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_LINEARANALYSIS_HH

#include <cstddef>
#include <petscksp.h>

#include "../model/Model.hh"
#include "../petsc/KspCompatibility.hh"
#include "../petsc/PetscRaii.hh"
#include "../utils/Benchmark.hh"
#include "AnalysisRouteAudit.hh"

// =============================================================================
//  LinearAnalysis — PETSc KSP-driven linear static solver
// =============================================================================
//
//  Solves the linear system  K u = f  using PETSc's KSP interface.
//
//  Template parameters mirror NonlinearAnalysis for consistency:
//    - MaterialPolicy   : material model (dim, StrainT, StressT)
//    - KinematicPolicy  : strain measure (SmallStrain default)
//    - ndofs            : DOFs per node (default: dim)
//    - ElemPolicy       : element storage policy (Single/Multi)
//
// =============================================================================

template <typename MaterialPolicy,
          typename KinematicPolicy = continuum::SmallStrain,
          std::size_t ndofs = MaterialPolicy::dim,
          typename ElemPolicy = SingleElementPolicy<ContinuumElement<MaterialPolicy, ndofs, KinematicPolicy>>>
class LinearAnalysis
{
    using ModelT = Model<MaterialPolicy, KinematicPolicy, ndofs, ElemPolicy>;
    static constexpr auto dim = MaterialPolicy::dim;

    ModelT* model_{nullptr};
    petsc::OwnedKSP solver_{};

    petsc::OwnedMat K{};
    petsc::OwnedVec U{}, F{};

    AnalysisTimer timer_;

public:
    using model_type = ModelT;
    using element_type = typename ModelT::element_type;
    using analysis_route_tag =
        fall_n::AnalysisRouteTag<fall_n::AnalysisRouteKind::linear_static>;
    static constexpr fall_n::AnalysisRouteKind analysis_route_kind =
        fall_n::AnalysisRouteKind::linear_static;
    static constexpr fall_n::AnalysisRouteAuditScope analysis_route_audit_scope =
        fall_n::canonical_analysis_route_audit_scope(analysis_route_kind);

    /// Access the performance timer.
    const AnalysisTimer& timer() const { return timer_; }
          AnalysisTimer& timer()       { return timer_; }

    auto get_model() const { return model_; }

    void setup_vector_sizes(){
        FALL_N_PETSC_CHECK(DMCreateGlobalVector(model_->get_plex(), U.ptr()));
        FALL_N_PETSC_CHECK(VecDuplicate(U.get(), F.ptr()));
        FALL_N_PETSC_CHECK(VecSet(U.get(), 0.0));
        FALL_N_PETSC_CHECK(VecSet(F.get(), 0.0));
    }

    void setup_matrix_sizes(){
        FALL_N_PETSC_CHECK(DMCreateMatrix(model_->get_plex(), K.ptr()));
        FALL_N_PETSC_CHECK(MatSetOption(K.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
        FALL_N_PETSC_CHECK(MatZeroEntries(K.get()));
    }

    void set_RHS(){
        FALL_N_PETSC_CHECK(DMLocalToGlobal(model_->get_plex(), model_->force_vector(), ADD_VALUES, F.get()));
    }

    void setup_solver(){
        FALL_N_PETSC_CHECK(KSPSetDM(solver_.get(), model_->get_plex()));
        FALL_N_PETSC_CHECK(KSPSetFromOptions(solver_.get()));
        FALL_N_PETSC_CHECK(petsc::disable_dm_activity_for_manual_linear_solve(solver_.get()));
    }

    void commit_model_state(){
        VecSet(model_->state_vector(), 0.0);
        FALL_N_PETSC_CHECK(DMGlobalToLocal(model_->get_plex(), U.get(), INSERT_VALUES, model_->state_vector()));
        VecAXPY        (model_->state_vector(), 1.0, model_->imposed_solution());
    }

    void solve(){
        timer_.start("setup");
        setup_vector_sizes();
        setup_matrix_sizes();
        set_RHS();
        timer_.stop("setup");

        timer_.start("assembly");
        model_->inject_K(K.get());
        timer_.stop("assembly");

        MatViewFromOptions(K.get(), nullptr, "-mat_view");

        timer_.start("solve");
        FALL_N_PETSC_CHECK(KSPSetOperators(solver_.get(), K.get(), K.get()));
        FALL_N_PETSC_CHECK(KSPSolve(solver_.get(), F.get(), U.get()));
        timer_.stop("solve");

        timer_.start("commit");
        commit_model_state();
        model_->update_elements_state();
        timer_.stop("commit");
    }

    explicit LinearAnalysis(ModelT* model) : model_{model} {
        FALL_N_PETSC_CHECK(KSPCreate(PETSC_COMM_WORLD, solver_.ptr()));
        setup_solver();
    }

    LinearAnalysis() = delete;

    ~LinearAnalysis() = default;
};

#endif // FALL_N_SRC_ANALYSIS_LINEARANALYSIS_HH
