#ifndef FALL_N_SRC_RECONSTRUCTION_ENRICHMENT_ACTIVATION_POLICY_HH
#define FALL_N_SRC_RECONSTRUCTION_ENRICHMENT_ACTIVATION_POLICY_HH

// Plan v2 §Fase 3.5 — adaptive enrichment activation policy.
//
// Pure header, dependency-free predicate consumed by the multiscale
// orchestrator before promoting a continuum-smeared local site to an
// XFEM-enriched site. The predicate is a small piece of data + logic
// that both `LocalContinuumEvolver` and `LocalXFEMEvolver` can share
// without dragging in the macro / micro solver headers.
//
// Decision rule (canonical):
//   activate_enrichment ⇔
//       site_kind ∈ {xfem_shifted_heaviside}                            (gate)
//       AND damage_index               >= damage_threshold              (kinematic)
//       AND principal_strain_magnitude >= principal_strain_threshold    (kinematic)
//       AND macro_load_step            >= activation_macro_step         (guarded)
//
// Defaults match the value used in the 200 mm canonical run (NZ=4
// shifted-Heaviside, bounded dowel-x bridging) and the guarded coupling
// activation step (10) declared in Cap. 79 §Fase 6 hypothesis H1.

#include <string_view>

#include "src/reconstruction/LocalModelKind.hh"

namespace fall_n {

struct EnrichmentActivationThresholds {
    /// Scalar damage index in [0, 1]; site activates above this.
    double damage_threshold{0.20};
    /// Principal strain magnitude (absolute, dimensionless) above which
    /// strain localisation is considered onset.
    double principal_strain_threshold{2.5e-3};
    /// First macro step in which guarded activation is admissible.
    /// Mirrors `MultiscaleAnalysis::set_coupling_start_step()`.
    int activation_macro_step{10};
};

struct EnrichmentActivationProbe {
    LocalModelKind site_kind{LocalModelKind::unspecified};
    double damage_index{0.0};
    double principal_strain_magnitude{0.0};
    int macro_load_step{0};
};

[[nodiscard]] constexpr bool
should_activate_enrichment(const EnrichmentActivationProbe& probe,
                           const EnrichmentActivationThresholds& th =
                               EnrichmentActivationThresholds{}) noexcept
{
    if (!local_model_kind_supports_enrichment_activation(probe.site_kind)) {
        return false;
    }
    if (probe.macro_load_step < th.activation_macro_step) {
        return false;
    }
    if (probe.damage_index < th.damage_threshold) {
        return false;
    }
    if (probe.principal_strain_magnitude < th.principal_strain_threshold) {
        return false;
    }
    return true;
}

/// Reason code emitted by audit/diagnostics after a guarded check.
enum class EnrichmentActivationReason {
    activated,
    rejected_unsupported_site_kind,
    rejected_below_macro_step_gate,
    rejected_below_damage_threshold,
    rejected_below_principal_strain_threshold,
};

[[nodiscard]] constexpr EnrichmentActivationReason
classify_enrichment_activation(const EnrichmentActivationProbe& probe,
                               const EnrichmentActivationThresholds& th =
                                   EnrichmentActivationThresholds{}) noexcept
{
    if (!local_model_kind_supports_enrichment_activation(probe.site_kind)) {
        return EnrichmentActivationReason::rejected_unsupported_site_kind;
    }
    if (probe.macro_load_step < th.activation_macro_step) {
        return EnrichmentActivationReason::rejected_below_macro_step_gate;
    }
    if (probe.damage_index < th.damage_threshold) {
        return EnrichmentActivationReason::rejected_below_damage_threshold;
    }
    if (probe.principal_strain_magnitude < th.principal_strain_threshold) {
        return EnrichmentActivationReason::rejected_below_principal_strain_threshold;
    }
    return EnrichmentActivationReason::activated;
}

[[nodiscard]] constexpr std::string_view
enrichment_activation_reason_label(EnrichmentActivationReason r) noexcept
{
    switch (r) {
        case EnrichmentActivationReason::activated:
            return "activated";
        case EnrichmentActivationReason::rejected_unsupported_site_kind:
            return "rejected_unsupported_site_kind";
        case EnrichmentActivationReason::rejected_below_macro_step_gate:
            return "rejected_below_macro_step_gate";
        case EnrichmentActivationReason::rejected_below_damage_threshold:
            return "rejected_below_damage_threshold";
        case EnrichmentActivationReason::rejected_below_principal_strain_threshold:
            return "rejected_below_principal_strain_threshold";
    }
    return "unknown";
}

}  // namespace fall_n

#endif  // FALL_N_SRC_RECONSTRUCTION_ENRICHMENT_ACTIVATION_POLICY_HH
