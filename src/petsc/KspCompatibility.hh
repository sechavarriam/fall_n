#ifndef FALL_N_SRC_PETSC_KSPCOMPATIBILITY_HH
#define FALL_N_SRC_PETSC_KSPCOMPATIBILITY_HH

#include <petscksp.h>
#include <petscversion.h>

namespace petsc {

inline PetscErrorCode disable_dm_activity_for_manual_linear_solve(KSP ksp)
{
#if defined(PETSC_VERSION_GE) && PETSC_VERSION_GE(3, 25, 0)
    PetscErrorCode ierr = PETSC_SUCCESS;
    ierr = KSPSetDMActive(ksp, KSP_DMACTIVE_OPERATOR, PETSC_FALSE);
    if (ierr != PETSC_SUCCESS) return ierr;
    ierr = KSPSetDMActive(ksp, KSP_DMACTIVE_RHS, PETSC_FALSE);
    if (ierr != PETSC_SUCCESS) return ierr;
    return KSPSetDMActive(ksp, KSP_DMACTIVE_INITIAL_GUESS, PETSC_FALSE);
#else
    return KSPSetDMActive(ksp, PETSC_FALSE);
#endif
}

} // namespace petsc

#endif // FALL_N_SRC_PETSC_KSPCOMPATIBILITY_HH
