#ifndef FALL_N_PETSC_OWNED_TS_HH
#define FALL_N_PETSC_OWNED_TS_HH

// ═══════════════════════════════════════════════════════════════════════
//  RAII wrapper for PETSc TS (time-stepping).
//
//  Same semantics as OwnedVec — see OwnedVec.hh for documentation.
// ═══════════════════════════════════════════════════════════════════════

#include <petscts.h>
#include <utility>

namespace petsc {

class OwnedTS {
    TS t_{nullptr};

public:
    OwnedTS() = default;
    explicit OwnedTS(TS t) noexcept : t_(t) {}

    ~OwnedTS() {
        if (t_) TSDestroy(&t_);
    }

    OwnedTS(OwnedTS&& other) noexcept
        : t_(std::exchange(other.t_, nullptr)) {}

    OwnedTS& operator=(OwnedTS&& other) noexcept {
        if (this != &other) {
            if (t_) TSDestroy(&t_);
            t_ = std::exchange(other.t_, nullptr);
        }
        return *this;
    }

    OwnedTS(const OwnedTS&)            = delete;
    OwnedTS& operator=(const OwnedTS&) = delete;

    [[nodiscard]] TS  get() const noexcept { return t_; }
    [[nodiscard]] TS* ptr()       noexcept { return &t_; }

    operator TS() const noexcept { return t_; }
    explicit operator bool() const noexcept { return t_ != nullptr; }

    [[nodiscard]] TS release() noexcept {
        return std::exchange(t_, nullptr);
    }

    void reset() noexcept {
        if (t_) TSDestroy(&t_);
        t_ = nullptr;
    }
};

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_TS_HH
