#ifndef FALL_N_SRC_ANALYSIS_LINEARANALYSIS_HH
#define FALL_N_SRC_ANALYSIS_LINEARANALYSIS_HH

#include <cstddef>
#include <petscksp.h>

#include "../model/Model.hh"
#include "../utils/Benchmark.hh"

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
    KSP     solver_{nullptr};

    Mat K{nullptr};
    Vec U{nullptr}, F{nullptr};

    AnalysisTimer timer_;

public:

    /// Access the performance timer.
    const AnalysisTimer& timer() const { return timer_; }
          AnalysisTimer& timer()       { return timer_; }

    auto get_model() const { return model_; }

    void setup_vector_sizes(){
        DMCreateGlobalVector(model_->get_plex(), &U);
        VecDuplicate(U, &F);
        VecSet(U, 0.0);
        VecSet(F, 0.0);
    }

    void setup_matrix_sizes(){
        DMCreateMatrix(model_->get_plex(), &K);
        DMSetMatType(model_->get_plex(), MATAIJ);
        DMSetUp(model_->get_plex());
        MatZeroEntries(K);
    }

    void set_RHS(){
        DMLocalToGlobal(model_->get_plex(), model_->force_vector(), ADD_VALUES, F);
    }

    void setup_solver(){
        KSPSetDM(solver_, model_->get_plex());
        KSPSetFromOptions(solver_);
        KSPSetDMActive(solver_, PETSC_FALSE);
    }

    void commit_model_state(){
        DMGlobalToLocal(model_->get_plex(), U, INSERT_VALUES, model_->state_vector());
        VecAXPY        (model_->state_vector(), 1.0, model_->imposed_solution());
    }

    void solve(){
        timer_.start("setup");
        setup_vector_sizes();
        setup_matrix_sizes();
        set_RHS();
        timer_.stop("setup");

        timer_.start("assembly");
        model_->inject_K(this->K);
        timer_.stop("assembly");

        // Honour PETSc runtime visualisation options:
        //   -mat_view draw          → X11 spy plot
        //   -mat_view ::ascii_info  → summary to stdout
        // No-op when none of these options are set.
        MatViewFromOptions(K, nullptr, "-mat_view");

        timer_.start("solve");
        KSPSetOperators(solver_, K, K);
        KSPSolve(solver_, F, U);
        timer_.stop("solve");

        timer_.start("commit");
        commit_model_state();
        model_->update_elements_state();
        timer_.stop("commit");
    }

    explicit LinearAnalysis(ModelT* model) : model_{model} {
        KSPCreate(PETSC_COMM_WORLD, &solver_);
        setup_solver();
    }

    LinearAnalysis() = delete;

    ~LinearAnalysis(){
        MatDestroy(&K);
        VecDestroy(&U);
        VecDestroy(&F);
        KSPDestroy(&solver_);
    }
};

#endif // FALL_N_SRC_ANALYSIS_LINEARANALYSIS_HH