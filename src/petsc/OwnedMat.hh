#ifndef FALL_N_PETSC_OWNED_MAT_HH
#define FALL_N_PETSC_OWNED_MAT_HH

// RAII wrapper for PETSc Mat. See OwnedHandle.hh for the ownership contract.

#include "OwnedHandle.hh"

namespace petsc {

using OwnedMat = OwnedHandle<Mat, MatDestroy>;

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_MAT_HH
