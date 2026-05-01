// Plan v2 §Fase 3.5 — enrichment-activation policy unit gate.

#include <cassert>
#include <print>

#include "src/reconstruction/EnrichmentActivationPolicy.hh"

int main() {
    using namespace fall_n;

    EnrichmentActivationThresholds th{};
    assert(th.damage_threshold == 0.20);
    assert(th.activation_macro_step == 10);

    // 1. Continuum-smeared site is not a candidate even with high damage.
    {
        EnrichmentActivationProbe p{
            .site_kind = LocalModelKind::continuum_smeared,
            .damage_index = 0.95,
            .principal_strain_magnitude = 1.0e-2,
            .macro_load_step = 50};
        assert(!should_activate_enrichment(p, th));
        assert(classify_enrichment_activation(p, th) ==
               EnrichmentActivationReason::rejected_unsupported_site_kind);
    }

    // 2. XFEM site with sub-threshold step rejects.
    {
        EnrichmentActivationProbe p{
            .site_kind = LocalModelKind::xfem_shifted_heaviside,
            .damage_index = 0.5,
            .principal_strain_magnitude = 1.0e-2,
            .macro_load_step = 5};
        assert(!should_activate_enrichment(p, th));
        assert(classify_enrichment_activation(p, th) ==
               EnrichmentActivationReason::rejected_below_macro_step_gate);
    }

    // 3. XFEM site post-step but undamaged rejects.
    {
        EnrichmentActivationProbe p{
            .site_kind = LocalModelKind::xfem_shifted_heaviside,
            .damage_index = 0.05,
            .principal_strain_magnitude = 1.0e-2,
            .macro_load_step = 20};
        assert(!should_activate_enrichment(p, th));
        assert(classify_enrichment_activation(p, th) ==
               EnrichmentActivationReason::rejected_below_damage_threshold);
    }

    // 4. XFEM site damaged but small principal strain rejects.
    {
        EnrichmentActivationProbe p{
            .site_kind = LocalModelKind::xfem_shifted_heaviside,
            .damage_index = 0.5,
            .principal_strain_magnitude = 1.0e-4,
            .macro_load_step = 20};
        assert(!should_activate_enrichment(p, th));
        assert(classify_enrichment_activation(p, th) ==
               EnrichmentActivationReason::
                   rejected_below_principal_strain_threshold);
    }

    // 5. All gates pass → activated.
    {
        EnrichmentActivationProbe p{
            .site_kind = LocalModelKind::xfem_shifted_heaviside,
            .damage_index = 0.30,
            .principal_strain_magnitude = 5.0e-3,
            .macro_load_step = 12};
        assert(should_activate_enrichment(p, th));
        assert(classify_enrichment_activation(p, th) ==
               EnrichmentActivationReason::activated);
    }

    // 6. Boundary on damage index — equality passes.
    {
        EnrichmentActivationProbe p{
            .site_kind = LocalModelKind::xfem_shifted_heaviside,
            .damage_index = th.damage_threshold,
            .principal_strain_magnitude = th.principal_strain_threshold,
            .macro_load_step = th.activation_macro_step};
        assert(should_activate_enrichment(p, th));
    }

    // 7. Custom thresholds (more permissive) flip a previously-rejected probe.
    {
        EnrichmentActivationProbe p{
            .site_kind = LocalModelKind::xfem_shifted_heaviside,
            .damage_index = 0.10,
            .principal_strain_magnitude = 1.0e-3,
            .macro_load_step = 5};
        assert(!should_activate_enrichment(p));   // default
        EnrichmentActivationThresholds permissive{
            .damage_threshold = 0.05,
            .principal_strain_threshold = 5.0e-4,
            .activation_macro_step = 0};
        assert(should_activate_enrichment(p, permissive));
    }

    std::println("[enrichment_activation_policy] ALL PASS");
    return 0;
}
