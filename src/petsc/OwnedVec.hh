#ifndef FALL_N_PETSC_OWNED_VEC_HH
#define FALL_N_PETSC_OWNED_VEC_HH

// ═══════════════════════════════════════════════════════════════════════
//  RAII wrapper for PETSc Vec.
//
//  Semantics:
//    - Owns the Vec handle.  Calls VecDestroy on destruction.
//    - Move-only (no copy).
//    - Default-constructs to a null (empty) state.
//    - Implicit conversion to Vec for seamless use with PETSc API.
//    - ptr() returns &handle for PETSc creation functions:
//        DMCreateLocalVector(dm, v.ptr());
//
//  Invariant:
//    After move-from, the source holds nullptr (valid empty state).
// ═══════════════════════════════════════════════════════════════════════

#include <petsc.h>
#include <utility> // std::exchange

namespace petsc {

class OwnedVec {
    Vec v_{nullptr};

public:
    // ── Construction / Destruction ────────────────────────────────────

    /// Default: empty (null) vector.
    OwnedVec() = default;

    /// Take ownership of an existing Vec handle.
    explicit OwnedVec(Vec v) noexcept : v_(v) {}

    /// Destroy the owned Vec (if any).
    ~OwnedVec() {
        if (v_) VecDestroy(&v_);
    }

    // ── Move semantics ───────────────────────────────────────────────

    OwnedVec(OwnedVec&& other) noexcept
        : v_(std::exchange(other.v_, nullptr)) {}

    OwnedVec& operator=(OwnedVec&& other) noexcept {
        if (this != &other) {
            if (v_) VecDestroy(&v_);
            v_ = std::exchange(other.v_, nullptr);
        }
        return *this;
    }

    // ── No copy ──────────────────────────────────────────────────────

    OwnedVec(const OwnedVec&)            = delete;
    OwnedVec& operator=(const OwnedVec&) = delete;

    // ── Access ───────────────────────────────────────────────────────

    /// Raw handle (non-owning view).
    [[nodiscard]] Vec get() const noexcept { return v_; }

    /// Pointer to handle for PETSc creation functions.
    [[nodiscard]] Vec* ptr() noexcept { return &v_; }

    /// Implicit conversion for direct use in PETSc API.
    operator Vec() const noexcept { return v_; }

    /// Boolean test: is the wrapper non-empty?
    explicit operator bool() const noexcept { return v_ != nullptr; }

    // ── Modifiers ────────────────────────────────────────────────────

    /// Release ownership and return the raw handle.
    [[nodiscard]] Vec release() noexcept {
        return std::exchange(v_, nullptr);
    }

    /// Destroy the current handle (if any) and reset to empty.
    void reset() noexcept {
        if (v_) VecDestroy(&v_);
        v_ = nullptr;
    }

    /// Destroy the current handle (if any) and take ownership of a new one.
    void reset(Vec v) noexcept {
        if (v_ && v_ != v) VecDestroy(&v_);
        v_ = v;
    }
};

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_VEC_HH
