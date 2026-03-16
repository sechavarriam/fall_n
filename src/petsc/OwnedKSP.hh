#ifndef FALL_N_PETSC_OWNED_KSP_HH
#define FALL_N_PETSC_OWNED_KSP_HH

// ═══════════════════════════════════════════════════════════════════════
//  RAII wrapper for PETSc KSP (linear solver).
//
//  Same semantics as OwnedVec — see OwnedVec.hh for documentation.
// ═══════════════════════════════════════════════════════════════════════

#include <petscksp.h>
#include <utility>

namespace petsc {

class OwnedKSP {
    KSP k_{nullptr};

public:
    OwnedKSP() = default;
    explicit OwnedKSP(KSP k) noexcept : k_(k) {}

    ~OwnedKSP() {
        if (k_) KSPDestroy(&k_);
    }

    OwnedKSP(OwnedKSP&& other) noexcept
        : k_(std::exchange(other.k_, nullptr)) {}

    OwnedKSP& operator=(OwnedKSP&& other) noexcept {
        if (this != &other) {
            if (k_) KSPDestroy(&k_);
            k_ = std::exchange(other.k_, nullptr);
        }
        return *this;
    }

    OwnedKSP(const OwnedKSP&)            = delete;
    OwnedKSP& operator=(const OwnedKSP&) = delete;

    [[nodiscard]] KSP  get() const noexcept { return k_; }
    [[nodiscard]] KSP* ptr()       noexcept { return &k_; }

    operator KSP() const noexcept { return k_; }
    explicit operator bool() const noexcept { return k_ != nullptr; }

    [[nodiscard]] KSP release() noexcept {
        return std::exchange(k_, nullptr);
    }

    void reset() noexcept {
        if (k_) KSPDestroy(&k_);
        k_ = nullptr;
    }
};

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_KSP_HH
