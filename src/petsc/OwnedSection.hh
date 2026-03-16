#ifndef FALL_N_PETSC_OWNED_SECTION_HH
#define FALL_N_PETSC_OWNED_SECTION_HH

// ═══════════════════════════════════════════════════════════════════════
//  RAII wrapper for PETSc PetscSection.
//
//  Same semantics as OwnedVec — see OwnedVec.hh for documentation.
// ═══════════════════════════════════════════════════════════════════════

#include <petsc.h>
#include <utility>

namespace petsc {

class OwnedSection {
    PetscSection s_{nullptr};

public:
    OwnedSection() = default;
    explicit OwnedSection(PetscSection s) noexcept : s_(s) {}

    ~OwnedSection() {
        if (s_) PetscSectionDestroy(&s_);
    }

    // ── Move only ────────────────────────────────────────────────────

    OwnedSection(OwnedSection&& other) noexcept
        : s_(std::exchange(other.s_, nullptr)) {}

    OwnedSection& operator=(OwnedSection&& other) noexcept {
        if (this != &other) {
            if (s_) PetscSectionDestroy(&s_);
            s_ = std::exchange(other.s_, nullptr);
        }
        return *this;
    }

    OwnedSection(const OwnedSection&)            = delete;
    OwnedSection& operator=(const OwnedSection&) = delete;

    // ── Access ───────────────────────────────────────────────────────

    [[nodiscard]] PetscSection  get() const noexcept { return s_; }
    [[nodiscard]] PetscSection* ptr()       noexcept { return &s_; }

    operator PetscSection() const noexcept { return s_; }
    explicit operator bool() const noexcept { return s_ != nullptr; }

    // ── Modifiers ────────────────────────────────────────────────────

    [[nodiscard]] PetscSection release() noexcept {
        return std::exchange(s_, nullptr);
    }

    void reset() noexcept {
        if (s_) PetscSectionDestroy(&s_);
        s_ = nullptr;
    }

    void reset(PetscSection s) noexcept {
        if (s_ && s_ != s) PetscSectionDestroy(&s_);
        s_ = s;
    }
};

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_SECTION_HH
