#ifndef FALL_N_PETSC_OWNED_HANDLE_HH
#define FALL_N_PETSC_OWNED_HANDLE_HH

// ═══════════════════════════════════════════════════════════════════════
//  OwnedHandle<T, Destroy> — generic move-only RAII owner of a PETSc handle
// ═══════════════════════════════════════════════════════════════════════
//
//  PETSc objects are opaque handles (Vec, Mat, SNES, KSP, TS, PetscSection)
//  released through a matching `XxxDestroy(T*)` call. This one template
//  captures that ownership contract once; the per-type wrappers
//  (OwnedVec, OwnedMat, ...) are just aliases that bind T and its destroyer.
//
//  Semantics:
//    - Owns the handle; calls Destroy(&handle) on destruction / reset.
//    - Move-only (no copy); default-constructs to a null (empty) state.
//    - Implicit conversion to T for seamless use with the PETSc C API.
//    - ptr() returns &handle for PETSc creation functions, e.g.
//        DMCreateLocalVector(dm, v.ptr());
//
//  Invariant:
//    After a move-from, the source holds nullptr (a valid empty state).
// ═══════════════════════════════════════════════════════════════════════

#include <petsc.h>
#include <utility> // std::exchange

namespace petsc {

template <class T, PetscErrorCode (*Destroy)(T*)>
class OwnedHandle {
    T h_{nullptr};

public:
    // ── Construction / Destruction ────────────────────────────────────
    OwnedHandle() = default;
    explicit OwnedHandle(T h) noexcept : h_(h) {}
    ~OwnedHandle() {
        if (h_) Destroy(&h_);
    }

    // ── Move only ─────────────────────────────────────────────────────
    OwnedHandle(OwnedHandle&& other) noexcept
        : h_(std::exchange(other.h_, nullptr)) {}

    OwnedHandle& operator=(OwnedHandle&& other) noexcept {
        if (this != &other) {
            if (h_) Destroy(&h_);
            h_ = std::exchange(other.h_, nullptr);
        }
        return *this;
    }

    OwnedHandle(const OwnedHandle&)            = delete;
    OwnedHandle& operator=(const OwnedHandle&) = delete;

    // ── Access ────────────────────────────────────────────────────────
    /// Raw handle (non-owning view).
    [[nodiscard]] T  get() const noexcept { return h_; }
    /// Pointer to the handle, for PETSc creation functions.
    [[nodiscard]] T* ptr()       noexcept { return &h_; }
    /// Implicit conversion for direct use in the PETSc C API.
    operator T() const noexcept { return h_; }
    /// Non-empty test.
    explicit operator bool() const noexcept { return h_ != nullptr; }

    // ── Modifiers ─────────────────────────────────────────────────────
    /// Release ownership and return the raw handle.
    [[nodiscard]] T release() noexcept { return std::exchange(h_, nullptr); }
    /// Destroy the current handle (if any) and reset to empty.
    void reset() noexcept {
        if (h_) Destroy(&h_);
        h_ = nullptr;
    }
    /// Destroy the current handle (if any) and take ownership of a new one.
    void reset(T h) noexcept {
        if (h_ && h_ != h) Destroy(&h_);
        h_ = h;
    }
};

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_HANDLE_HH
