#ifndef FALL_N_PETSC_OWNED_VEC_HH
#define FALL_N_PETSC_OWNED_VEC_HH

// RAII wrapper for PETSc Vec. See OwnedHandle.hh for the ownership contract.

#include "OwnedHandle.hh"

namespace petsc {

using OwnedVec = OwnedHandle<Vec, VecDestroy>;

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_VEC_HH
