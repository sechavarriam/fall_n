#ifndef FALL_N_ANALYSIS_INTERFACE
#define FALL_N_ANALYSIS_INTERFACE

#include <cstddef>
#include <petsc.h>

#include "../model/Model.hh"

//template<typename T> // Concept? CRTP? Policy?
class Analysis {

    Model<ThreeDimensionalMaterial, 3>* model_;

    KSP solver_;

public:

    void setup_model_vectors(){
        DMCreateGlobalVector(model_->get_plex(), &model_->U);
        DMCreateGlobalVector(model_->get_plex(), &model_->F);

        //DMCreateLocalVector(model_->get_plex(), &model_->F);

        VecSet(model_->U, 0.0);
        VecSet(model_->F, 0.0);

        std::cout << "UUUUUUUUUUUUUUUUUUUUUUUUUUU" << std::endl;
        VecView(model_->U, PETSC_VIEWER_STDOUT_WORLD);
        std::cout << "FFFFFFFFFFFFFFFFFFFFFFFFFFF" << std::endl;
        VecView(model_->F, PETSC_VIEWER_STDOUT_WORLD);

        DMSetUp(model_->get_plex());
    }

    void setup_model_matrices(){
        ////https://lists.mcs.anl.gov/mailman/htdig/petsc-users/2016-March/028797.htm

        DMCreateMatrix(model_->get_plex(), &model_->K);
        DMSetMatType  (model_->get_plex(), MATAIJ); // Set the matrix type for the mesh
                                                    // SetFromOptions in the future. 
        DMSetUp(model_->get_plex());
        //MatZeroEntries(model_->K);
    }

    void setup_solver(){

        KSPSetDM(solver_, model_->get_plex());
        KSPSetFromOptions(solver_);

        KSPSetDMActive(solver_, PETSC_FALSE);
    }

    void solve(){

        //setup_model_vectors();
        //setup_model_matrices();

        model_->inject_K();       
        MatView(model_->K, PETSC_VIEWER_STDOUT_WORLD); 
        MatView(model_->K, PETSC_VIEWER_DRAW_WORLD);

        KSPSetOperators(solver_, model_->K, model_->K);
        
        //VecView(model_->U, PETSC_VIEWER_STDOUT_WORLD);
        KSPSolve(solver_, model_->F, model_->U);
        //VecView(model_->U, PETSC_VIEWER_STDOUT_WORLD);
    }


    Analysis(Model<ThreeDimensionalMaterial, 3>* model) : model_{model} {
        KSPCreate(PETSC_COMM_WORLD, &solver_);
        setup_solver();
    }

    Analysis() = default;
    
    ~Analysis(){
        KSPDestroy(&solver_);
    }
};



#endif // FALL_N_ANALYSIS_INTERFACE