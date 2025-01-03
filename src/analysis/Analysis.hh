#ifndef FALL_N_ANALYSIS_INTERFACE
#define FALL_N_ANALYSIS_INTERFACE

#include <cstddef>
#include <petsc.h>

#include "../model/Model.hh"

//template<typename T> // Concept? CRTP? Policy?


class NLAnalysis
{
    Model<ThreeDimensionalMaterial, 3>* model_;

    Vec U, RHS, R; //Global solution vector and Residual.
    Mat J; //Jacobian matrix. 

    Vec u, f; //Local vectors.  
    
    SNES solver_;

    void setup_solver(){
        SNESCreate(PETSC_COMM_WORLD, &solver_);
        SNESSetFromOptions(solver_);
        SNESSetDM(solver_, model_->get_plex());
    }

    bool is_setup{false};
    
public:

    void setup(){
        //if (! model_->is_bc_updated){model_->setup()}; // Setup DM sizes according to the model constraints.
        setup_vector_sizes();
        setup_matrix_sizes();
        is_setup = true;
    }

    void setup_vector_sizes(){

        DMCreateGlobalVector(model_->get_plex(), &U);
        VecDuplicate(U, &R);
        VecDuplicate(U, &RHS);

        DMCreateLocalVector(model_->get_plex(), &u);
        DMCreateLocalVector(model_->get_plex(), &f);

        VecSet(U, 0.0);
        VecSet(R, 0.0);
        VecSet(RHS, 0.0);
    }

    void setup_matrix_sizes(){
        DMCreateMatrix(model_->get_plex(), &J);
        DMSetMatType  (model_->get_plex(), MATAIJ); // Set the matrix type for the mesh
                                                    // SetFromOptions in the future. 
        DMSetUp(model_->get_plex());
        MatZeroEntries(J);
    }

    void set_RHS(){
        PetscSection global_section, local_section;
        DMGetGlobalSection(model_->get_plex(), &global_section);
        DMGetLocalSection (model_->get_plex(), &local_section);

        DMLocalToGlobal(model_->get_plex(), model_->nodal_forces, ADD_VALUES, RHS); // DMLocalToGlobal() is a short form of DMLocalToGlobalBegin() and DMLocalToGlobalEnd()
    }
    
    void compute_residual(){
    }

    void compute_jacobian(){
        //model_->inject_K();       
        //MatView(model_->K, PETSC_VIEWER_STDOUT_WORLD); 
        //MatView(model_->K, PETSC_VIEWER_DRAW_WORLD);
    }


    void solve(){
        if (!is_setup) setup();

        set_RHS();

        compute_residual();
        compute_jacobian();

        //SNESSetFunction(solver_, F, model_->compute_residual, model_);
        //SNESSetJacobian(solver_, J, J, model_->compute_jacobian, model_);
        //SNESSolve(solver_, F, U);
        //VecView(U, PETSC_VIEWER_STDOUT_WORLD);
    }

    NLAnalysis(Model<ThreeDimensionalMaterial, 3>* model) : model_{model} {
        SNESCreate(PETSC_COMM_WORLD, &solver_);
        setup_solver();
    }

    NLAnalysis() = default;

    ~NLAnalysis(){
        SNESDestroy(&solver_);
        VecDestroy(&U);
        VecDestroy(&u);
        VecDestroy(&f);
        MatDestroy(&J);
    }

};
     
class LinearAnalysis {

    Model<ThreeDimensionalMaterial, 3>* model_;

    KSP solver_;

public:

    void setup_solver(){

        KSPSetDM(solver_, model_->get_plex());
        KSPSetFromOptions(solver_);

        KSPSetDMActive(solver_, PETSC_FALSE);
    }

    void solve(){
        //setup_model_vectors();
        //setup_model_matrices();
        //model_->inject_K();       
        //MatView(model_->K, PETSC_VIEWER_STDOUT_WORLD); 
        //MatView(model_->K, PETSC_VIEWER_DRAW_WORLD);
        //KSPSetOperators(solver_, model_->K, model_->K);
        //
        ////VecView(model_->U, PETSC_VIEWER_STDOUT_WORLD);
        //KSPSolve(solver_, model_->F, model_->U);
        //VecView(model_->U, PETSC_VIEWER_STDOUT_WORLD);
    }

    LinearAnalysis(Model<ThreeDimensionalMaterial, 3>* model) : model_{model} {
        KSPCreate(PETSC_COMM_WORLD, &solver_);
        setup_solver();
    }

    LinearAnalysis() = default;
    
    ~LinearAnalysis(){
        KSPDestroy(&solver_);
    }
};



#endif // FALL_N_ANALYSIS_INTERFACE