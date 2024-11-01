#ifndef FALL_N_ANALYSIS_INTERFACE
#define FALL_N_ANALYSIS_INTERFACE


#include <cstddef>


#include <petsc.h>


#include "../model/Model.hh"


// https://petsc.org/release/manual/ksp/

//template<typename T> // Concept? CRTP? Policy?
class Analysis {

using PetscMatrix = Mat;
using PetscVector = Vec;
using PetscSolver = KSP;

    Model<ThreeDimensionalMaterial, 3>* model_;

    KSP solver_;

public:

    void setup_solver(){
        KSPSetDM(solver_, model_->domain_->mesh.dm); // Attach the solver to the mesh  
        KSPSetFromOptions(solver_);

    }


    Analysis(Model<ThreeDimensionalMaterial, 3>* model) : model_{model} {
        KSPCreate(PETSC_COMM_WORLD, &solver_);

        setup_solver();
    }

    Analysis() = default;
    ~Analysis() = default;
    
};



#endif // FALL_N_ANALYSIS_INTERFACE