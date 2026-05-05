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

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

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

    // ── Smeared cracking state — for concrete / brittle models ──────────
    //  Populated when the internal state exposes `.num_cracks` and
    //  `.crack_normals[]`.  Crack normals are stored as 3D vectors
    //  (nz = 0 for plane stress) for direct ParaView Glyph visualization.

    std::optional<int> num_cracks{};

    // Crack normal vectors as 3D (x, y, z).  Up to 3 cracks.
    // crack_normal_N[2] = 0.0 for plane-stress models.
    std::optional<std::array<double, 3>> crack_normal_1{};
    std::optional<std::array<double, 3>> crack_normal_2{};
    std::optional<std::array<double, 3>> crack_normal_3{};

    // Per-crack opening strain (normal to crack face)
    std::optional<double> crack_strain_1{};
    std::optional<double> crack_strain_2{};
    std::optional<double> crack_strain_3{};

    // Per-crack maximum historical opening strain.  This is used only for
    // audit/visualization persistence; the current opening above remains the
    // state used for coloring and work-conjugate interpretation.
    std::optional<double> crack_strain_max_1{};
    std::optional<double> crack_strain_max_2{};
    std::optional<double> crack_strain_max_3{};

    // Per-crack closed status (0.0 = open, 1.0 = closed)
    std::optional<double> crack_closed_1{};
    std::optional<double> crack_closed_2{};
    std::optional<double> crack_closed_3{};

    // ── Fracturing history invariants — for concrete models ─────────────
    std::optional<double> sigma_o_max{};   // max octahedral normal stress (compression)
    std::optional<double> tau_o_max{};     // max octahedral shear stress
    std::optional<int> solution_mode{};
    std::optional<double> trial_sigma_o{};
    std::optional<double> trial_tau_o{};
    std::optional<double> no_flow_coupling_update_norm{};
    std::optional<double> no_flow_recovery_residual{};
    std::optional<int> no_flow_stabilization_iterations{};
    std::optional<int> no_flow_crack_state_switches{};
    std::optional<bool> no_flow_stabilized{};

    // -- Generic uniaxial concrete history surface ---------------------------------
    //  Used today by Kent-Park style concrete laws to expose closure/reopening
    //  state without forcing the validation layer to downcast the constitutive site.
    std::optional<int> history_state_code{};
    std::optional<double> history_min_strain{};
    std::optional<double> history_min_stress{};
    std::optional<double> history_closure_strain{};
    std::optional<double> history_max_tensile_strain{};
    std::optional<double> history_max_tensile_stress{};
    std::optional<double> history_committed_strain{};
    std::optional<double> history_committed_stress{};
    std::optional<bool> history_cracked{};

    // ── Query helpers ────────────────────────────────────────────────────
    [[nodiscard]] bool has_plastic_strain()            const noexcept { return plastic_strain.has_value(); }
    [[nodiscard]] bool has_equivalent_plastic_strain()  const noexcept { return equivalent_plastic_strain.has_value(); }
    [[nodiscard]] bool has_damage()                     const noexcept { return damage.has_value(); }
    [[nodiscard]] bool has_cracks()                     const noexcept { return num_cracks.has_value(); }
    [[nodiscard]] bool has_fracture_history()            const noexcept { return sigma_o_max.has_value(); }
    [[nodiscard]] bool has_no_flow_diagnostics()         const noexcept {
        return no_flow_stabilization_iterations.has_value()
            || no_flow_stabilized.has_value();
    }
    [[nodiscard]] bool has_uniaxial_concrete_history()    const noexcept {
        return history_state_code.has_value()
            || history_closure_strain.has_value()
            || history_max_tensile_strain.has_value();
    }
};


// ═══════════════════════════════════════════════════════════════════════════
//  GaussFieldRecord — per-integration-point field data for VTK export
//
//  Used by FEM_Element::collect_gauss_fields() to return material state
//  through the type-erased interface.  ContinuumElement populates full
//  Voigt stress/strain; elements without exportable material state
//  (e.g. TrussElement) return empty records.
//
//  The snapshot's span<const double> members (e.g. plastic_strain) point
//  into the record's own pstrain_storage, so the record is self-contained.
// ═══════════════════════════════════════════════════════════════════════════

struct GaussFieldRecord {
    std::vector<double>     stress;           // Voigt notation
    std::vector<double>     strain;           // Voigt notation
    std::vector<double>     pstrain_storage;  // owned copy of plastic strain
    InternalFieldSnapshot   snapshot;         // internal state (cracks, etc.)
};

#endif // FALL_N_INTERNAL_FIELD_SNAPSHOT_HH
