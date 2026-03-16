#ifndef FALL_N_PETSC_CHECK_HH
#define FALL_N_PETSC_CHECK_HH

// ═══════════════════════════════════════════════════════════════════════
//  PETSc error-checking utilities for fall_n.
//
//  Usage:
//      PetscErrorCode ierr;
//      ierr = VecCreate(PETSC_COMM_WORLD, &v); FALL_N_PETSC_CHECK(ierr);
//
//  Or, inline:
//      FALL_N_PETSC_CHECK(VecCreate(PETSC_COMM_WORLD, &v));
//
//  In debug builds (NDEBUG not defined), a nonzero PetscErrorCode
//  throws std::runtime_error with file/line/code context.
//
//  In release builds, behavior depends on FALL_N_PETSC_CHECK_RELEASE:
//    - If defined: checks are active even in release.
//    - Otherwise:  checks are no-ops (rely on PETSc's own handler).
// ═══════════════════════════════════════════════════════════════════════

#include <petsc.h>
#include <stdexcept>
#include <string>

namespace petsc {

/// Format a PETSc error into a human-readable message.
[[nodiscard]] inline std::string
format_petsc_error(PetscErrorCode code, const char* file, int line,
                   const char* expr) {
    std::string msg = "PETSc error in ";
    msg += file;
    msg += ':';
    msg += std::to_string(line);
    msg += "\n  Expression: ";
    msg += expr;
    msg += "\n  Error code: ";
    msg += std::to_string(static_cast<int>(code));

    const char* text = nullptr;
    PetscErrorMessage(code, &text, nullptr);
    if (text) {
        msg += "\n  Description: ";
        msg += text;
    }
    return msg;
}

/// Exception type for PETSc errors.  Carries the original error code.
class PetscError : public std::runtime_error {
    PetscErrorCode code_;
public:
    PetscError(PetscErrorCode code, const char* file, int line,
               const char* expr)
        : std::runtime_error(format_petsc_error(code, file, line, expr)),
          code_(code) {}

    [[nodiscard]] PetscErrorCode code() const noexcept { return code_; }
};

} // namespace petsc


// ── Macro ────────────────────────────────────────────────────────────

#if !defined(NDEBUG) || defined(FALL_N_PETSC_CHECK_RELEASE)

  #define FALL_N_PETSC_CHECK(expr)                                      \
    do {                                                                 \
        PetscErrorCode fall_n_petsc_ierr_ = (expr);                     \
        if (fall_n_petsc_ierr_ != 0) {                                  \
            throw ::petsc::PetscError(                                   \
                fall_n_petsc_ierr_, __FILE__, __LINE__, #expr);          \
        }                                                                \
    } while (0)

#else

  #define FALL_N_PETSC_CHECK(expr) ((void)(expr))

#endif


#endif // FALL_N_PETSC_CHECK_HH
