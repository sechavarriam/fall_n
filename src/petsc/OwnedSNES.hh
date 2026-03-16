#ifndef FALL_N_PETSC_OWNED_SNES_HH
#define FALL_N_PETSC_OWNED_SNES_HH

// ═══════════════════════════════════════════════════════════════════════
//  RAII wrapper for PETSc SNES (nonlinear solver).
//
//  Same semantics as OwnedVec — see OwnedVec.hh for documentation.
// ═══════════════════════════════════════════════════════════════════════

#include <petscsnes.h>
#include <utility>

namespace petsc {

class OwnedSNES {
    SNES s_{nullptr};

public:
    OwnedSNES() = default;
    explicit OwnedSNES(SNES s) noexcept : s_(s) {}

    ~OwnedSNES() {
        if (s_) SNESDestroy(&s_);
    }

    OwnedSNES(OwnedSNES&& other) noexcept
        : s_(std::exchange(other.s_, nullptr)) {}

    OwnedSNES& operator=(OwnedSNES&& other) noexcept {
        if (this != &other) {
            if (s_) SNESDestroy(&s_);
            s_ = std::exchange(other.s_, nullptr);
        }
        return *this;
    }

    OwnedSNES(const OwnedSNES&)            = delete;
    OwnedSNES& operator=(const OwnedSNES&) = delete;

    [[nodiscard]] SNES  get() const noexcept { return s_; }
    [[nodiscard]] SNES* ptr()       noexcept { return &s_; }

    operator SNES() const noexcept { return s_; }
    explicit operator bool() const noexcept { return s_ != nullptr; }

    [[nodiscard]] SNES release() noexcept {
        return std::exchange(s_, nullptr);
    }

    void reset() noexcept {
        if (s_) SNESDestroy(&s_);
        s_ = nullptr;
    }
};

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_SNES_HH
