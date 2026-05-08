#include <cassert>
#include <cmath>
#include <cstdio>

#include "src/validation/ReducedRCMacroInferredXfemSitePolicy.hh"

int main()
{
    using namespace fall_n;

    ReducedRCMultiscaleReplaySitePlan fixed_site{};
    fixed_site.site_index = 7;
    fixed_site.z_over_l = 0.08;
    fixed_site.activation_score = 3.0;
    fixed_site.selected_for_replay = true;

    ReducedRCManagedLocalPatchSpec base{};
    base.characteristic_length_m = 0.24;
    base.nx = 3;
    base.ny = 3;
    base.nz = 8;

    const auto fixed_patch =
        make_reduced_rc_macro_inferred_xfem_patch(fixed_site, base);
    assert(fixed_patch.site_index == fixed_site.site_index);
    assert(fixed_patch.crack_position_inferred_from_macro);
    assert(fixed_patch.longitudinal_bias_location ==
           ReducedRCLocalLongitudinalBiasLocation::fixed_end);
    assert(std::abs(fixed_patch.crack_z_over_l - fixed_site.z_over_l) <
           1.0e-14);
    assert(fixed_patch.longitudinal_bias_power > 1.0);

    ReducedRCMacroEndpointDemand double_hinge{};
    double_hinge.fixed_end_score = 2.0;
    double_hinge.loaded_end_score = 2.4;
    double_hinge.macro_section_z_over_l = 0.46;

    const auto both_patch =
        make_reduced_rc_macro_inferred_xfem_patch(
            fixed_site,
            base,
            double_hinge);
    assert(both_patch.longitudinal_bias_location ==
           ReducedRCLocalLongitudinalBiasLocation::both_ends);
    assert(both_patch.double_hinge_bias_inferred_from_macro);
    assert(both_patch.longitudinal_bias_power >
           fixed_patch.longitudinal_bias_power);
    assert(std::abs(both_patch.crack_z_over_l - 0.46) < 1.0e-14);

    ReducedRCMacroInferredXfemSitePolicy fallback_policy{};
    fallback_policy.preserve_macro_section_crack_location = false;
    const auto loaded_patch =
        make_reduced_rc_macro_inferred_xfem_patch(
            fixed_site,
            base,
            ReducedRCMacroEndpointDemand{
                .fixed_end_score = 0.1,
                .loaded_end_score = 1.5},
            fallback_policy);
    assert(loaded_patch.longitudinal_bias_location ==
           ReducedRCLocalLongitudinalBiasLocation::loaded_end);
    assert(std::abs(loaded_patch.crack_z_over_l -
                    fallback_policy.fallback_loaded_end_crack_z_over_l) <
           1.0e-14);

    const auto contains_z = [](const auto& candidates, double z_over_l) {
        for (const auto& candidate : candidates) {
            if (std::abs(candidate.z_over_l - z_over_l) < 1.0e-14) {
                return true;
            }
        }
        return false;
    };

    const auto fixed_candidates =
        infer_reduced_rc_macro_local_site_candidates(
            ReducedRCMacroEndpointDemand{
                .fixed_end_score = 0.65,
                .loaded_end_score = 0.0});
    assert(fixed_candidates.size() == 1);
    assert(contains_z(fixed_candidates, 0.05));
    assert(fixed_candidates.front().reason == "fixed_end_score");

    const auto double_hinge_candidates =
        infer_reduced_rc_macro_local_site_candidates(
            ReducedRCMacroEndpointDemand{
                .fixed_end_score = 1.20,
                .loaded_end_score = 1.10});
    assert(double_hinge_candidates.size() == 3);
    assert(contains_z(double_hinge_candidates, 0.05));
    assert(contains_z(double_hinge_candidates, 0.50));
    assert(contains_z(double_hinge_candidates, 0.95));

    const auto paired_end_candidates =
        infer_reduced_rc_macro_local_site_candidates(
            ReducedRCMacroEndpointDemand{
                .fixed_end_score = 2.0,
                .loaded_end_score = 1.6});
    assert(paired_end_candidates.size() == 3);
    assert(contains_z(paired_end_candidates, 0.05));
    assert(contains_z(paired_end_candidates, 0.50));
    assert(contains_z(paired_end_candidates, 0.95));

    ReducedRCMacroInferredLocalSiteSelectionPolicy control_policy{};
    control_policy.include_inactive_control_sites = true;
    const auto control_candidates =
        infer_reduced_rc_macro_local_site_candidates(
            ReducedRCMacroEndpointDemand{
                .fixed_end_score = 0.65,
                .loaded_end_score = 0.45},
            control_policy);
    assert(control_candidates.size() == 2);
    assert(contains_z(control_candidates, 0.05));
    assert(contains_z(control_candidates, 0.95));

    control_policy.include_center_control_site = true;
    const auto center_control_candidates =
        infer_reduced_rc_macro_local_site_candidates(
            ReducedRCMacroEndpointDemand{
                .fixed_end_score = 0.65,
                .loaded_end_score = 0.45},
            control_policy);
    assert(center_control_candidates.size() == 3);
    assert(contains_z(center_control_candidates, 0.05));
    assert(contains_z(center_control_candidates, 0.50));
    assert(contains_z(center_control_candidates, 0.95));

    std::printf("[macro-inferred-xfem-site-policy] fixed crack=%.3f "
                "both crack=%.3f loaded crack=%.3f\n",
                fixed_patch.crack_z_over_l,
                both_patch.crack_z_over_l,
                loaded_patch.crack_z_over_l);
    return 0;
}
