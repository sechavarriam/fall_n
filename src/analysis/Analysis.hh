#ifndef FALL_N_ANALYSIS_INTERFACE
#define FALL_N_ANALYSIS_INTERFACE


#include <cstddef>


#include <petsc.h>


#include "../model/Model.hh"


// https://petsc.org/release/manual/ksp/

//template<typename T> // Concept? CRTP? Policy?
class Analysis {

    Model<ThreeDimensionalMaterial, 3>* model_;

    KSP solver_;

public:

    /*
    [0]PETSC ERROR: Object is in wrong state
    [0]PETSC ERROR: You called KSPSetDM() but did not use DMKSPSetComputeOperators() or KSPSetDMActive(ksp,PETSC_FALSE);
    */

    void setup_solver(){
        KSPSetDM(solver_, model_->get_plex());
        KSPSetFromOptions(solver_);

        KSPSetDMActive(solver_, PETSC_FALSE);

        KSPSetOperators(solver_, model_->K, model_->K);


    }

    void solve(){
        KSPSolve(solver_, model_->F, model_->U);

        //View
        VecView(model_->U, PETSC_VIEWER_STDOUT_WORLD);
    }


    Analysis(Model<ThreeDimensionalMaterial, 3>* model) : model_{model} {
        KSPCreate(PETSC_COMM_WORLD, &solver_);

        setup_solver();

    }

    Analysis() = default;
    ~Analysis() = default;
    
};



#endif // FALL_N_ANALYSIS_INTERFACE