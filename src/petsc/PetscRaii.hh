#ifndef FALL_N_PETSC_RAII_HH
#define FALL_N_PETSC_RAII_HH

// ═══════════════════════════════════════════════════════════════════════
//  Convenience header: includes all PETSc RAII wrappers and check macro.
//
//  Usage in application code:
//      #include "petsc/PetscRaii.hh"
// ═══════════════════════════════════════════════════════════════════════

#include "check.hh"
#include "OwnedVec.hh"
#include "OwnedMat.hh"
#include "OwnedSection.hh"
#include "OwnedKSP.hh"
#include "OwnedSNES.hh"
#include "OwnedTS.hh"

#endif // FALL_N_PETSC_RAII_HH
