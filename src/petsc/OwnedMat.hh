#ifndef FALL_N_PETSC_OWNED_MAT_HH
#define FALL_N_PETSC_OWNED_MAT_HH

// ═══════════════════════════════════════════════════════════════════════
//  RAII wrapper for PETSc Mat.
//
//  Same semantics as OwnedVec — see OwnedVec.hh for documentation.
// ═══════════════════════════════════════════════════════════════════════

#include <petsc.h>
#include <utility>

namespace petsc {

class OwnedMat {
    Mat m_{nullptr};

public:
    OwnedMat() = default;
    explicit OwnedMat(Mat m) noexcept : m_(m) {}

    ~OwnedMat() {
        if (m_) MatDestroy(&m_);
    }

    // ── Move only ────────────────────────────────────────────────────

    OwnedMat(OwnedMat&& other) noexcept
        : m_(std::exchange(other.m_, nullptr)) {}

    OwnedMat& operator=(OwnedMat&& other) noexcept {
        if (this != &other) {
            if (m_) MatDestroy(&m_);
            m_ = std::exchange(other.m_, nullptr);
        }
        return *this;
    }

    OwnedMat(const OwnedMat&)            = delete;
    OwnedMat& operator=(const OwnedMat&) = delete;

    // ── Access ───────────────────────────────────────────────────────

    [[nodiscard]] Mat  get() const noexcept { return m_; }
    [[nodiscard]] Mat* ptr()       noexcept { return &m_; }

    operator Mat() const noexcept { return m_; }
    explicit operator bool() const noexcept { return m_ != nullptr; }

    // ── Modifiers ────────────────────────────────────────────────────

    [[nodiscard]] Mat release() noexcept {
        return std::exchange(m_, nullptr);
    }

    void reset() noexcept {
        if (m_) MatDestroy(&m_);
        m_ = nullptr;
    }

    void reset(Mat m) noexcept {
        if (m_ && m_ != m) MatDestroy(&m_);
        m_ = m;
    }
};

} // namespace petsc

#endif // FALL_N_PETSC_OWNED_MAT_HH
