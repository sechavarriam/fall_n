#ifndef FALL_N_INTERNAL_FIELD_SNAPSHOT_HH
#define FALL_N_INTERNAL_FIELD_SNAPSHOT_HH

// ═══════════════════════════════════════════════════════════════════════════
//  InternalFieldSnapshot — lightweight, stack-allocated snapshot of
//  a material point's internal state for post-processing export.
//
//  ── Design rationale ────────────────────────────────────────────────────
//
//  The type-erased Material<Policy> class uses External Polymorphism
//  (MaterialConcept virtual base + OwningMaterialModel concrete impl).
//  A single virtual method  `internal_field_snapshot() const`  is added to
//  the virtual interface, returning this struct by value (~96 bytes, fully
//  stack-allocated).
//
//  WHY NOT migrate the OWNING path to Manual Virtual Dispatch (MVD) + SBO?
//
//  We evaluated a full MVD+SBO migration of the OWNING Material<Policy>.
//  The concrete material types are too large for SBO to be effective:
//
//    - Elastic 3D (ContinuumIsotropicRelation + ElasticUpdate): ~312 bytes
//    - Plastic 3D (PlasticityRelation<3D,VonMises,...> + InelasticUpdate): ~792 bytes
//
//  An SBO buffer ≤ 384 bytes covers elastic materials but plasticity (the
//  most interesting export case) still requires heap allocation, wasting
//  the buffer.  MVD without SBO saves only 8 bytes per instance (the
//  vtable pointer).  Both approaches have identical dispatch cost (~3 ns).
//
//  Therefore the owning wrapper remains heap-backed external polymorphism.
//  The non-owning borrowed views introduced later (MaterialRef /
//  MaterialConstRef) are different: they only store pointers to an already
//  owned concrete material and strategy, so their erased model fits naturally
//  in a fixed raw buffer of three pointers.
//
//  ── Extensibility ───────────────────────────────────────────────────────
//
//  New internal fields (backstress, damage variable, etc.) are added by
//  appending optional<> members here.  OwningMaterialModel fills them via
//  `if constexpr` — existing elastic models are unaffected (all nullopt).
//
// ═══════════════════════════════════════════════════════════════════════════

#include <cstddef>
#include <optional>
#include <span>

struct InternalFieldSnapshot {

    // ── Inelastic strain tensor (Voigt notation) ─────────────────────────
    //  Populated when the constitutive model has plastic_strain / eps_p().
    //  span points into the material's own storage — valid only during
    //  the snapshot's lifetime (do NOT store across iterations).
    std::optional<std::span<const double>> plastic_strain{};

    // ── Equivalent plastic strain scalar (ε̄ᵖ) ──────────────────────────
    //  Populated when the hardening state carries equivalent_plastic_strain.
    std::optional<double> equivalent_plastic_strain{};

    // ── (Future) backstress tensor β — for kinematic hardening ───────────
    //  std::optional<std::span<const double>> backstress{};

    // ── Damage scalar d — for continuum damage models ───────────────────
    //  Populated when the algorithmic state exposes either `.damage` or a
    //  convenience accessor `.d()`.  This keeps post-processing agnostic
    //  about the exact internal-state layout chosen by the constitutive law.
    std::optional<double> damage{};

    // ── Query helpers ────────────────────────────────────────────────────
    [[nodiscard]] bool has_plastic_strain()            const noexcept { return plastic_strain.has_value(); }
    [[nodiscard]] bool has_equivalent_plastic_strain()  const noexcept { return equivalent_plastic_strain.has_value(); }
    [[nodiscard]] bool has_damage()                     const noexcept { return damage.has_value(); }
};

#endif // FALL_N_INTERNAL_FIELD_SNAPSHOT_HH
