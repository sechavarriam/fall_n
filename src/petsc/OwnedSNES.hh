#ifndef FALL_N_PETSC_OWNED_SNES_HH
#define FALL_N_PETSC_OWNED_SNES_HH

// RAII wrapper for PETSc SNES (nonlinear solver). See OwnedHandle.hh for the
// ownership contract.

#include "OwnedHandle.hh"

namespace petsc {

using OwnedSNES = OwnedHandle<SNES, SNESDestroy>;

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_SNES_HH
