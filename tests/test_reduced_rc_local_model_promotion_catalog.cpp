#include <iostream>

#include "src/validation/ReducedRCLocalModelPromotionCatalog.hh"

namespace {

int passed = 0;
int failed = 0;

void report(const char* name, bool ok)
{
    if (ok) {
        ++passed;
        std::cout << "  PASS  " << name << "\n";
    } else {
        ++failed;
        std::cout << "  FAIL  " << name << "\n";
    }
}

constexpr auto promotion_table =
    fall_n::canonical_reduced_rc_local_model_promotion_table_v;

constexpr bool promotion_table_has_the_declared_roles()
{
    return promotion_table.size() == 5 &&
           fall_n::canonical_reduced_rc_closed_reference_count_v == 1 &&
           fall_n::canonical_reduced_rc_promoted_control_count_v == 1 &&
           fall_n::canonical_reduced_rc_promoted_physical_local_model_count_v == 1 &&
           fall_n::canonical_reduced_rc_primary_multiscale_candidate_count_v == 1;
}

constexpr bool structural_reference_is_not_a_multiscale_local_model()
{
    const auto row = fall_n::find_reduced_rc_local_model_promotion_row(
        promotion_table,
        "structural_n10_lobatto_fine_ultra_reference");
    return row.can_anchor_structural_reference &&
           !row.can_enter_multiscale_as_physical_local_model &&
           row.state_kind ==
               fall_n::ReducedRCLocalModelPromotionStateKind::closed_reference &&
           row.criteria.max_peak_normalized_rms_base_shear_error <= 0.005;
}

constexpr bool continuum_branch_is_a_control_not_the_final_local_model()
{
    const auto row = fall_n::find_reduced_rc_local_model_promotion_row(
        promotion_table,
        "continuum_dirichlet_composite_regression_control");
    return row.can_anchor_continuum_regression &&
           !row.can_enter_multiscale_as_physical_local_model &&
           row.blocking_issue_kind ==
               fall_n::ReducedRCLocalModelBlockingIssueKind::
                   distributed_crack_localization &&
           row.criteria.max_host_bar_rms_gap_m <= 1.0e-8 &&
           row.criteria.max_axial_balance_error_mn <= 1.0e-6;
}

constexpr bool xfem_is_the_only_primary_multiscale_candidate()
{
    const auto row = fall_n::find_reduced_rc_local_model_promotion_row(
        promotion_table,
        "xfem_global_secant_200mm_primary_candidate");
    return row.is_xfem_candidate() &&
           row.is_primary_multiscale_candidate &&
           row.can_enter_multiscale_as_physical_local_model &&
           row.requires_enriched_dofs &&
           row.requires_discrete_crack_geometry &&
           row.state_kind ==
               fall_n::ReducedRCLocalModelPromotionStateKind::
                   promoted_physical_local_model &&
           row.criteria.required_protocol_amplitude_mm == 200.0 &&
           row.criteria.max_peak_normalized_rms_base_shear_error <= 0.10 &&
           row.criteria.min_peak_base_shear_ratio <= 0.90 &&
           row.criteria.max_peak_base_shear_ratio >= 1.15 &&
           row.criteria.max_allowed_timeout_cases == 0;
}

constexpr bool guardrail_models_are_not_promoted_as_local_physics()
{
    const auto ko_bathe = fall_n::find_reduced_rc_local_model_promotion_row(
        promotion_table,
        "ko_bathe_hex20_hex27_heavy_reference");
    const auto opensees = fall_n::find_reduced_rc_local_model_promotion_row(
        promotion_table,
        "opensees_continuum_external_control");
    return !ko_bathe.can_enter_multiscale_as_physical_local_model &&
           !opensees.can_enter_multiscale_as_physical_local_model &&
           ko_bathe.blocking_issue_kind ==
               fall_n::ReducedRCLocalModelBlockingIssueKind::runtime_cost &&
           opensees.blocking_issue_kind ==
               fall_n::ReducedRCLocalModelBlockingIssueKind::
                   external_solver_frontier;
}

static_assert(promotion_table_has_the_declared_roles());
static_assert(structural_reference_is_not_a_multiscale_local_model());
static_assert(continuum_branch_is_a_control_not_the_final_local_model());
static_assert(xfem_is_the_only_primary_multiscale_candidate());
static_assert(guardrail_models_are_not_promoted_as_local_physics());

} // namespace

int main()
{
    std::cout << "=== Reduced RC Local Model Promotion Catalog Tests ===\n";

    report("promotion_table_has_the_declared_roles",
           promotion_table_has_the_declared_roles());
    report("structural_reference_is_not_a_multiscale_local_model",
           structural_reference_is_not_a_multiscale_local_model());
    report("continuum_branch_is_a_control_not_the_final_local_model",
           continuum_branch_is_a_control_not_the_final_local_model());
    report("xfem_is_the_only_primary_multiscale_candidate",
           xfem_is_the_only_primary_multiscale_candidate());
    report("guardrail_models_are_not_promoted_as_local_physics",
           guardrail_models_are_not_promoted_as_local_physics());

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
