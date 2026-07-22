#ifndef FALL_N_PETSC_OWNED_SECTION_HH
#define FALL_N_PETSC_OWNED_SECTION_HH

// RAII wrapper for PetscSection. See OwnedHandle.hh for the ownership contract.

#include "OwnedHandle.hh"

namespace petsc {

using OwnedSection = OwnedHandle<PetscSection, PetscSectionDestroy>;

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_SECTION_HH
