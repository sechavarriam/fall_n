#ifndef FALL_N_PETSC_OWNED_KSP_HH
#define FALL_N_PETSC_OWNED_KSP_HH

// RAII wrapper for PETSc KSP (linear solver). See OwnedHandle.hh for the
// ownership contract.

#include "OwnedHandle.hh"

namespace petsc {

using OwnedKSP = OwnedHandle<KSP, KSPDestroy>;

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_KSP_HH
