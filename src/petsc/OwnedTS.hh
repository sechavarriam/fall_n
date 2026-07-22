#ifndef FALL_N_PETSC_OWNED_TS_HH
#define FALL_N_PETSC_OWNED_TS_HH

// RAII wrapper for PETSc TS (time-stepping). See OwnedHandle.hh for the
// ownership contract.

#include "OwnedHandle.hh"

namespace petsc {

using OwnedTS = OwnedHandle<TS, TSDestroy>;

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_TS_HH
