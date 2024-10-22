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

    //Model<ThreeDimensionalMaterial, 3> model_;

    //PetscSolver solver_;

public:

    //Analysis(Model<ThreeDimensionalMaterial, 3>& model) : model_{model} {
    //    //model_.assembly_K();
    //};

    Analysis() = default;
    ~Analysis() = default;
    
};



#endif // FALL_N_ANALYSIS_INTERFACE