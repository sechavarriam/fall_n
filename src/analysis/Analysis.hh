#ifndef FALL_N_ANALYSIS_INTERFACE
#define FALL_N_ANALYSIS_INTERFACE

#include <cstddef>
#include <petsc.h>

#include "../model/Model.hh"

// template<typename T> // Concept? CRTP? Policy?
class NLAnalysis
{
    Model<ThreeDimensionalMaterial, 3> *model_;

    Vec U, RHS, R; // Global solution vector and Residual.
    Mat J;         // Jacobian matrix.

    Vec u, f; // Local vectors.

    SNES solver_;

    void setup_solver()
    {
        SNESCreate(PETSC_COMM_WORLD, &solver_);
        SNESSetFromOptions(solver_);
        SNESSetDM(solver_, model_->get_plex());
    }

    bool is_setup{false};

public:
    void setup()
    {
        // if (! model_->is_bc_updated){model_->setup()}; // Setup DM sizes according to the model constraints.
        setup_vector_sizes();
        setup_matrix_sizes();
        is_setup = true;
    }

    void setup_vector_sizes()
    {

        DMCreateGlobalVector(model_->get_plex(), &U);
        VecDuplicate(U, &R);
        VecDuplicate(U, &RHS);

        DMCreateLocalVector(model_->get_plex(), &u);
        DMCreateLocalVector(model_->get_plex(), &f);

        VecSet(U, 0.0);
        VecSet(R, 0.0);
        VecSet(RHS, 0.0);
    }

    void setup_matrix_sizes()
    {
        DMCreateMatrix(model_->get_plex(), &J);
        DMSetMatType(model_->get_plex(), MATAIJ); // Set the matrix type for the mesh
                                                  // SetFromOptions in the future.
        DMSetUp(model_->get_plex());
        MatZeroEntries(J);
    }

    void set_RHS()
    {
        PetscSection global_section, local_section;
        DMGetGlobalSection(model_->get_plex(), &global_section);
        DMGetLocalSection(model_->get_plex(), &local_section);

        DMLocalToGlobal(model_->get_plex(), model_->nodal_forces, ADD_VALUES, RHS); // DMLocalToGlobal() is a short form of DMLocalToGlobalBegin() and DMLocalToGlobalEnd()
    }

    void compute_residual()
    { // Linear Ax-RHS
    }

    void compute_jacobian()
    {
        J = model_->Kt;
    }

    void solve()
    {
        if (!is_setup)
            setup();

        set_RHS();

        compute_residual();
        compute_jacobian();

        // SNESSetFunction(solver_, F, model_->compute_residual, model_);
        // SNESSetJacobian(solver_, J, J, model_->compute_jacobian, model_);
        // SNESSolve(solver_, F, U);
        // VecView(U, PETSC_VIEWER_STDOUT_WORLD);
    }

    NLAnalysis(Model<ThreeDimensionalMaterial, 3> *model) : model_{model}
    {
        SNESCreate(PETSC_COMM_WORLD, &solver_);
        setup_solver();
    }

    NLAnalysis() = default;

    ~NLAnalysis()
    {
        SNESDestroy(&solver_);
        VecDestroy(&U);
        VecDestroy(&u);
        VecDestroy(&f);
        MatDestroy(&J);
    }
};

class LinearAnalysis
{
    using ModelT = Model<ThreeDimensionalMaterial, 3>;
    
    ModelT *model_; // Pointer to a model base?

    KSP solver_;

public:

    Mat K;    // Global stiffness matrix
    Vec U, F; // Global solution vector and RHS.

    auto get_model() const { return model_; }

    void record_solution(VTKDataContainer &recorder){
        
        double *u;

        VecGetArray(model_->current_state, &u);
        recorder.load_vector_field<3>("displacement", u, model_->get_domain().num_nodes());
        VecRestoreArray(model_->global_imposed_solution, &u);
    }

    void setup_vector_sizes(){
        DMCreateGlobalVector(model_->get_plex(), &U);
        VecDuplicate(U, &F);
        VecSet(U, 0.0);
        VecSet(F, 0.0);
    }

    void setup_matrix_sizes(){
        DMCreateMatrix(model_->get_plex(), &K);
        DMSetMatType(model_->get_plex(), MATAIJ); // Set the matrix type for the mesh
                                                  // SetFromOptions in the future.
        DMSetUp(model_->get_plex());
        MatZeroEntries(K);
    }

    void set_RHS(){
        PetscSection global_section, local_section;
        DMGetGlobalSection(model_->get_plex(), &global_section);
        DMGetLocalSection(model_->get_plex(), &local_section);

        DMLocalToGlobal(model_->get_plex(), model_->nodal_forces, ADD_VALUES, F); // DMLocalToGlobal() is a short form of DMLocalToGlobalBegin() and DMLocalToGlobalEnd()
    }

    void setup_solver(){
        KSPSetDM(solver_, model_->get_plex());
        KSPSetFromOptions(solver_);
        KSPSetDMActive(solver_, PETSC_FALSE);
    }

    void commit_model_state(){
        DMGlobalToLocal(model_->get_plex(), U, INSERT_VALUES, model_->current_state);
        VecAXPY        (model_->current_state, 1.0, model_->global_imposed_solution);
    }

    void solve(){
        setup_vector_sizes();
        setup_matrix_sizes();
        set_RHS();

        model_->inject_K(this->K);

        MatView(this->K, PETSC_VIEWER_DRAW_WORLD);

        KSPSetOperators(solver_, K, K);
        KSPSolve(solver_, F, U);

        commit_model_state();
    }

    LinearAnalysis(Model<ThreeDimensionalMaterial, 3> *model) : model_{model}{
        KSPCreate(PETSC_COMM_WORLD, &solver_);
        setup_solver();
    }

    LinearAnalysis() = default;

    ~LinearAnalysis()
    {
        KSPDestroy(&solver_);
    }
};

#endif // FALL_N_ANALYSIS_INTERFACE