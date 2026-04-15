#ifndef FALL_N_VALIDATION_CAMPAIGN_CATALOG_HH
#define FALL_N_VALIDATION_CAMPAIGN_CATALOG_HH

// =============================================================================
//  ValidationCampaignCatalog.hh -- canonical reboot plan for the library-wide
//                                  validation campaign
// =============================================================================
//
//  This header freezes, as compile-time metadata, the first validation program
//  that starts from zero instead of inheriting authority from legacy drivers.
//
//  The goal is intentionally methodological:
//
//    - no existing test/driver is assumed correct a priori;
//    - legacy validation surfaces may inform the new campaign, but only as
//      audited input;
//    - the first scientific target is not a full building, but a single
//      reinforced-concrete rectangular column under progressively amplified
//      cyclic displacement;
//    - escalation to continuum columns, enriched reinforcement models, and
//      larger structures is blocked until the preceding gates are closed.
//
//  The catalog is kept under src/validation rather than the common analysis
//  umbrella so that roadmap metadata remains reusable by tests, documentation,
//  and the thesis without polluting the numerical hot path.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

namespace fall_n {

enum class ValidationRebootPhaseKind {
    phase0_governance_and_legacy_quarantine,
    phase1_formulation_and_solver_audit,
    phase2_material_and_section_baseline,
    phase3_reduced_order_rc_column,
    phase4_continuum_rc_column,
    phase5_cross_model_equivalence,
    phase6_full_structure_escalation
};

enum class ValidationWorkstreamPriorityKind {
    mandatory_blocker,
    conditional_enabler,
    deferred_growth_path
};

enum class LegacyValidationSurfaceDispositionKind {
    keep_only_as_audited_input,
    quarantine_to_old_when_replacement_exists,
    retire_after_replacement_campaign,
    exploratory_only_do_not_anchor_claims
};

[[nodiscard]] constexpr std::string_view
to_string(ValidationRebootPhaseKind kind) noexcept
{
    switch (kind) {
        case ValidationRebootPhaseKind::phase0_governance_and_legacy_quarantine:
            return "phase0_governance_and_legacy_quarantine";
        case ValidationRebootPhaseKind::phase1_formulation_and_solver_audit:
            return "phase1_formulation_and_solver_audit";
        case ValidationRebootPhaseKind::phase2_material_and_section_baseline:
            return "phase2_material_and_section_baseline";
        case ValidationRebootPhaseKind::phase3_reduced_order_rc_column:
            return "phase3_reduced_order_rc_column";
        case ValidationRebootPhaseKind::phase4_continuum_rc_column:
            return "phase4_continuum_rc_column";
        case ValidationRebootPhaseKind::phase5_cross_model_equivalence:
            return "phase5_cross_model_equivalence";
        case ValidationRebootPhaseKind::phase6_full_structure_escalation:
            return "phase6_full_structure_escalation";
    }
    return "unknown_validation_reboot_phase_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(ValidationWorkstreamPriorityKind kind) noexcept
{
    switch (kind) {
        case ValidationWorkstreamPriorityKind::mandatory_blocker:
            return "mandatory_blocker";
        case ValidationWorkstreamPriorityKind::conditional_enabler:
            return "conditional_enabler";
        case ValidationWorkstreamPriorityKind::deferred_growth_path:
            return "deferred_growth_path";
    }
    return "unknown_validation_workstream_priority_kind";
}

[[nodiscard]] constexpr std::string_view
to_string(LegacyValidationSurfaceDispositionKind kind) noexcept
{
    switch (kind) {
        case LegacyValidationSurfaceDispositionKind::keep_only_as_audited_input:
            return "keep_only_as_audited_input";
        case LegacyValidationSurfaceDispositionKind::quarantine_to_old_when_replacement_exists:
            return "quarantine_to_old_when_replacement_exists";
        case LegacyValidationSurfaceDispositionKind::retire_after_replacement_campaign:
            return "retire_after_replacement_campaign";
        case LegacyValidationSurfaceDispositionKind::exploratory_only_do_not_anchor_claims:
            return "exploratory_only_do_not_anchor_claims";
    }
    return "unknown_legacy_validation_surface_disposition_kind";
}

struct ValidationCampaignWorkstreamRow {
    std::string_view row_label{};
    std::string_view module_label{};
    std::string_view objective_label{};
    std::string_view theory_anchor_label{};
    std::string_view planned_artifact_label{};
    std::string_view legacy_input_surface_label{};
    ValidationRebootPhaseKind phase_kind{
        ValidationRebootPhaseKind::phase0_governance_and_legacy_quarantine};
    ValidationWorkstreamPriorityKind priority_kind{
        ValidationWorkstreamPriorityKind::mandatory_blocker};
    LegacyValidationSurfaceDispositionKind legacy_surface_disposition{
        LegacyValidationSurfaceDispositionKind::keep_only_as_audited_input};
    bool required_for_reference_structural_column{false};
    bool required_for_reference_continuum_column{false};
    bool required_for_full_structure_escalation{false};
    bool requires_new_implementation{false};
    bool uses_legacy_surfaces_only_as_input{true};

    [[nodiscard]] constexpr bool blocks_any_stage() const noexcept
    {
        return required_for_reference_structural_column ||
               required_for_reference_continuum_column ||
               required_for_full_structure_escalation;
    }

    [[nodiscard]] constexpr bool is_growth_only_path() const noexcept
    {
        return priority_kind ==
               ValidationWorkstreamPriorityKind::deferred_growth_path;
    }
};

[[nodiscard]] constexpr ValidationCampaignWorkstreamRow
make_validation_campaign_workstream_row(
    std::string_view row_label,
    std::string_view module_label,
    std::string_view objective_label,
    std::string_view theory_anchor_label,
    std::string_view planned_artifact_label,
    std::string_view legacy_input_surface_label,
    ValidationRebootPhaseKind phase_kind,
    ValidationWorkstreamPriorityKind priority_kind,
    LegacyValidationSurfaceDispositionKind legacy_surface_disposition,
    bool required_for_reference_structural_column,
    bool required_for_reference_continuum_column,
    bool required_for_full_structure_escalation,
    bool requires_new_implementation,
    bool uses_legacy_surfaces_only_as_input) noexcept
{
    return {
        .row_label = row_label,
        .module_label = module_label,
        .objective_label = objective_label,
        .theory_anchor_label = theory_anchor_label,
        .planned_artifact_label = planned_artifact_label,
        .legacy_input_surface_label = legacy_input_surface_label,
        .phase_kind = phase_kind,
        .priority_kind = priority_kind,
        .legacy_surface_disposition = legacy_surface_disposition,
        .required_for_reference_structural_column =
            required_for_reference_structural_column,
        .required_for_reference_continuum_column =
            required_for_reference_continuum_column,
        .required_for_full_structure_escalation =
            required_for_full_structure_escalation,
        .requires_new_implementation = requires_new_implementation,
        .uses_legacy_surfaces_only_as_input =
            uses_legacy_surfaces_only_as_input
    };
}

[[nodiscard]] constexpr auto
canonical_validation_reboot_workstream_table() noexcept
{
    return std::to_array({
        make_validation_campaign_workstream_row(
            "governance_reset_and_evidence_protocol",
            "validation_governance",
            "Freeze the methodological rule that no legacy test, driver, or PDF"
            " may anchor a validation claim without re-audit.",
            "chapters4to6_claim_to_residual_to_history",
            "validation_reboot_master_plan + reproducibility protocol + legacy"
            " quarantine list",
            "ComprehensiveCyclicValidation.cpp ; TableCyclicValidationStructural.cpp ;"
            " ch82_cyclic_validation.tex ; ch85_comprehensive_cyclic_multiscale_validation.tex",
            ValidationRebootPhaseKind::phase0_governance_and_legacy_quarantine,
            ValidationWorkstreamPriorityKind::mandatory_blocker,
            LegacyValidationSurfaceDispositionKind::quarantine_to_old_when_replacement_exists,
            true,
            true,
            true,
            false,
            true),
        make_validation_campaign_workstream_row(
            "formulation_solver_scope_audit",
            "continuum_and_solver_core",
            "Audit TL, UL, corotational, incremental Newton, dynamics, and"
            " continuation before any physical-validation campaign is reopened.",
            "chapter4_virtual_work__chapter6_discrete_equilibrium",
            "scope matrix family x formulation x solver x validated slice",
            "existing continuum and solver audit catalogs + kinematic and steppable tests",
            ValidationRebootPhaseKind::phase1_formulation_and_solver_audit,
            ValidationWorkstreamPriorityKind::mandatory_blocker,
            LegacyValidationSurfaceDispositionKind::keep_only_as_audited_input,
            true,
            true,
            true,
            false,
            true),
        make_validation_campaign_workstream_row(
            "uniaxial_material_and_section_baseline",
            "materials_and_sections",
            "Re-audit steel, concrete, and fiber-section ingredients before using"
            " them in the reference column campaign.",
            "chapter5_constitutive_history_and_section_reduction",
            "material protocol suite + section moment-curvature baseline",
            "test_uniaxial_fiber.cpp ; test_rc_fiber_frame_nonlinear.cpp ;"
            " current RC section builders",
            ValidationRebootPhaseKind::phase2_material_and_section_baseline,
            ValidationWorkstreamPriorityKind::mandatory_blocker,
            LegacyValidationSurfaceDispositionKind::keep_only_as_audited_input,
            true,
            true,
            true,
            true,
            true),
        make_validation_campaign_workstream_row(
            "reduced_order_rc_column_matrix",
            "beam_structural_models",
            "Validate one rectangular RC column with Timoshenko beam models from"
            " 2 to 10 nodes/sections under progressively amplified cyclic"
            " displacement, with optional axial compression.",
            "chapter6_structural_reduction_and_section_resultants",
            "single-column beam benchmark suite + hysteresis + moment-curvature"
            " + convergence matrix",
            "TableCyclicValidationStructural.cpp ; test_timoshenko_cantilever_benchmark.cpp",
            ValidationRebootPhaseKind::phase3_reduced_order_rc_column,
            ValidationWorkstreamPriorityKind::mandatory_blocker,
            LegacyValidationSurfaceDispositionKind::quarantine_to_old_when_replacement_exists,
            true,
            true,
            true,
            true,
            true),
        make_validation_campaign_workstream_row(
            "beam_integration_family_extension",
            "beam_integration_rules",
            "Expose interchangeable Gauss, Lobatto, and Radau beam-integration"
            " strategies so the reference column can be validated against"
            " section-placement sensitivity instead of one hardcoded rule.",
            "distributed_plasticity_and_structural_quadrature",
            "integration-rule abstraction + regression matrix for TimoshenkoBeamN",
            "current hardcoded Gauss-Legendre path in TimoshenkoBeamN.hh",
            ValidationRebootPhaseKind::phase3_reduced_order_rc_column,
            ValidationWorkstreamPriorityKind::mandatory_blocker,
            LegacyValidationSurfaceDispositionKind::retire_after_replacement_campaign,
            true,
            true,
            true,
            true,
            true),
        make_validation_campaign_workstream_row(
            "continuum_rc_column_matrix",
            "continuum_structural_models",
            "Build the equivalent RC column with 3D continua (Hex8/20/27) and"
            " imposed top-face cyclic motion so that base reactions can be"
            " compared against the reduced-order column.",
            "chapter4_continuum_kinematics__chapter6_continuum_fem",
            "single-column continuum benchmark suite + mesh/order/material matrix",
            "ComprehensiveCyclicValidation.cpp ; KoBatheConcrete3D tests ; existing"
            " embedded rebar examples",
            ValidationRebootPhaseKind::phase4_continuum_rc_column,
            ValidationWorkstreamPriorityKind::mandatory_blocker,
            LegacyValidationSurfaceDispositionKind::quarantine_to_old_when_replacement_exists,
            false,
            true,
            true,
            true,
            true),
        make_validation_campaign_workstream_row(
            "reduced_vs_continuum_equivalence",
            "cross_model_validation",
            "Define and close the acceptance gate between the reduced-order"
            " column and the continuum column before any building-scale claim.",
            "claim_to_artifact_to_evidence_equivalence_gate",
            "equivalence tables for stiffness, strength, dissipation, crack onset,"
            " and loop shape",
            "existing structural and continuum cyclic drivers only as seed data",
            ValidationRebootPhaseKind::phase5_cross_model_equivalence,
            ValidationWorkstreamPriorityKind::mandatory_blocker,
            LegacyValidationSurfaceDispositionKind::keep_only_as_audited_input,
            false,
            true,
            true,
            true,
            true),
        make_validation_campaign_workstream_row(
            "reinforcement_discretization_extension",
            "embedded_rebar_and_compound_continua",
            "Investigate whether bonded embedded 2-node trusses are sufficient;"
            " if not, introduce TrussElement<Nnodes> or another reinforcement"
            " carrier with interior-node curvature/slip capability.",
            "compound_continuum_and_reinforcement_compatibility",
            "TrussElement<Nnodes> feasibility study + targeted benchmark if the"
            " current path proves insufficient",
            "current TrussElement.hh + reinforced submodel path",
            ValidationRebootPhaseKind::phase4_continuum_rc_column,
            ValidationWorkstreamPriorityKind::conditional_enabler,
            LegacyValidationSurfaceDispositionKind::exploratory_only_do_not_anchor_claims,
            false,
            false,
            true,
            true,
            true),
        make_validation_campaign_workstream_row(
            "alternative_concrete_model_extension",
            "constitutive_growth_paths",
            "Introduce additional concrete models only if the audited column"
            " campaign shows that current uniaxial or continuum concrete paths"
            " cannot reproduce the required mechanisms honestly.",
            "chapter5_constitutive_model_selection_by_evidence",
            "decision gate for Kent-Park extensions, confined concrete variants,"
            " damage-plasticity, or alternative continuum concrete",
            "KentParkConcrete.hh ; KoBatheConcrete.hh ; KoBatheConcrete3D.hh",
            ValidationRebootPhaseKind::phase2_material_and_section_baseline,
            ValidationWorkstreamPriorityKind::conditional_enabler,
            LegacyValidationSurfaceDispositionKind::exploratory_only_do_not_anchor_claims,
            false,
            false,
            true,
            true,
            true),
        make_validation_campaign_workstream_row(
            "force_based_structural_element_path",
            "structural_growth_path",
            "Keep force-based beam-column elements as a planned growth path, but"
            " do not let them block the first reference column validation unless"
            " the displacement-based route fails its evidence gate.",
            "distributed_plasticity_beyond_displacement_based_reference",
            "force-based element research track after baseline column closure",
            "no normative current path; literature-driven future module",
            ValidationRebootPhaseKind::phase6_full_structure_escalation,
            ValidationWorkstreamPriorityKind::deferred_growth_path,
            LegacyValidationSurfaceDispositionKind::exploratory_only_do_not_anchor_claims,
            false,
            false,
            false,
            true,
            false),
        make_validation_campaign_workstream_row(
            "full_structure_escalation_gate",
            "large_scale_models",
            "Do not reopen full buildings or FE2-heavy validation as normative"
            " targets until the single-column reduced/continuum campaign is"
            " closed and documented end to end.",
            "from_reference_column_to_full_structure_only_through_closed_gates",
            "escalation protocol toward walls, frames, subassemblages, and"
            " complete buildings",
            "current large cyclic/multiscale executables remain legacy or exploratory",
            ValidationRebootPhaseKind::phase6_full_structure_escalation,
            ValidationWorkstreamPriorityKind::mandatory_blocker,
            LegacyValidationSurfaceDispositionKind::exploratory_only_do_not_anchor_claims,
            false,
            false,
            true,
            false,
            true)
    });
}

inline constexpr auto canonical_validation_reboot_workstream_table_v =
    canonical_validation_reboot_workstream_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_validation_workstream_priority(
    const std::array<ValidationCampaignWorkstreamRow, N>& rows,
    ValidationWorkstreamPriorityKind priority) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.priority_kind == priority) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_validation_workstreams_required_for_reference_structural_column(
    const std::array<ValidationCampaignWorkstreamRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.required_for_reference_structural_column) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_validation_workstreams_required_for_reference_continuum_column(
    const std::array<ValidationCampaignWorkstreamRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.required_for_reference_continuum_column) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr std::size_t
count_validation_workstreams_required_for_full_structure_escalation(
    const std::array<ValidationCampaignWorkstreamRow, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.required_for_full_structure_escalation) {
            ++count;
        }
    }
    return count;
}

template <ValidationWorkstreamPriorityKind Priority>
inline constexpr std::size_t canonical_validation_workstream_priority_count_v =
    count_validation_workstream_priority(
        canonical_validation_reboot_workstream_table_v, Priority);

inline constexpr std::size_t
    canonical_validation_reference_structural_column_gate_count_v =
        count_validation_workstreams_required_for_reference_structural_column(
            canonical_validation_reboot_workstream_table_v);

inline constexpr std::size_t
    canonical_validation_reference_continuum_column_gate_count_v =
        count_validation_workstreams_required_for_reference_continuum_column(
            canonical_validation_reboot_workstream_table_v);

inline constexpr std::size_t
    canonical_validation_full_structure_escalation_gate_count_v =
        count_validation_workstreams_required_for_full_structure_escalation(
            canonical_validation_reboot_workstream_table_v);

} // namespace fall_n

#endif // FALL_N_VALIDATION_CAMPAIGN_CATALOG_HH
