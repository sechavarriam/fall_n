#ifndef FALL_N_REDUCED_RC_COLUMN_SECTION_BASELINE_HH
#define FALL_N_REDUCED_RC_COLUMN_SECTION_BASELINE_HH

// =============================================================================
//  ReducedRCColumnSectionBaseline.hh
// =============================================================================
//
//  Independent section-level baseline for the reduced reinforced-concrete
//  column reboot.
//
//  The structural reduced-column baseline exports a base-side
//  moment-curvature observable, but that observable should not be treated as
//  self-validating. This surface provides an independent section-level solve
//  over the same fiber section ingredients, using axial-force equilibrium
//  closure at each curvature step:
//
//      N(eps_0, kappa_y) = N_target
//
//  The result is a clean comparison artifact for later structural closure,
//  while keeping the implementation separate from the legacy table-cyclic
//  driver and from beam-element discretization effects.
//
// =============================================================================

#include "src/validation/ReducedRCColumnReferenceSpec.hh"
#include "src/analysis/LocalModelTaxonomy.hh"

#include <cstddef>
#include <string>
#include <vector>

namespace fall_n::validation_reboot {

enum class ReducedRCColumnSectionMaterialMode {
    nonlinear,
    elasticized,
};

enum class ReducedRCColumnSectionProtocolKind {
    monotonic,
    cyclic,
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnSectionMaterialMode mode) noexcept
{
    switch (mode) {
        case ReducedRCColumnSectionMaterialMode::nonlinear:
            return "nonlinear";
        case ReducedRCColumnSectionMaterialMode::elasticized:
            return "elasticized";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnSectionProtocolKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnSectionProtocolKind::monotonic:
            return "monotonic";
        case ReducedRCColumnSectionProtocolKind::cyclic:
            return "cyclic";
    }
    return "unknown";
}

struct ReducedRCColumnSectionBaselineRunSpec {
    ReducedRCColumnReferenceSpec reference_spec{};
    ReducedRCColumnSectionMaterialMode material_mode{
        ReducedRCColumnSectionMaterialMode::nonlinear};
    ReducedRCColumnSectionProtocolKind protocol_kind{
        ReducedRCColumnSectionProtocolKind::monotonic};
    double target_axial_compression_force_mn{0.0};
    double max_curvature_y{0.03};
    std::vector<double> cyclic_curvature_levels_y{};
    int steps{120};
    int steps_per_segment{4};
    int axial_force_newton_max_iterations{40};
    double axial_force_newton_tolerance_mn{1.0e-8};
    bool write_csv{true};
    bool print_progress{true};
};

[[nodiscard]] inline fall_n::LocalModelTaxonomy
describe_reduced_rc_column_section_local_model(
    const ReducedRCColumnSectionBaselineRunSpec&) noexcept
{
    return {
        .discretization_kind =
            fall_n::LocalModelDiscretizationKind::structural_section_surrogate,
        .fracture_representation_kind =
            fall_n::LocalFractureRepresentationKind::smeared_internal_state,
        .reinforcement_representation_kind =
            fall_n::LocalReinforcementRepresentationKind::
                constitutive_section_fibers,
        .maturity_kind =
            fall_n::LocalModelMaturityKind::comparison_control,
        .supports_discrete_crack_geometry = false,
        .requires_enriched_dofs = false,
        .requires_skeleton_trace_unknowns = false,
        .suitable_for_future_multiscale_local_model = false,
        .notes =
            "Independent section-level control problem over the reduced RC "
            "fiber section, used to audit moment-curvature and condensed "
            "tangents before structural closure."};
}

struct ReducedRCColumnSectionBaselineRecord {
    int step{0};
    double load_factor{0.0};
    double target_axial_force{0.0};
    double solved_axial_strain{0.0};
    double curvature_y{0.0};
    double curvature_z{0.0};
    double axial_force{0.0};
    double moment_y{0.0};
    double moment_z{0.0};
    // Axial tangent is exported as the raw constitutive entry. For bending we
    // keep both the raw constitutive entry and the axial-force-condensed
    // effective stiffness dM/dkappa|N=const. The latter is the quantity that
    // should agree with the numerical slope of the moment-curvature curve under
    // constant axial force.
    double tangent_ea{0.0};
    double tangent_eiy{0.0};
    double tangent_eiz{0.0};
    double tangent_eiy_direct_raw{0.0};
    double tangent_eiz_direct_raw{0.0};
    // Raw axial-flexural block entries kept for tangent diagnostics and
    // validation against external references. These do not change the primary
    // CSV contract; they support a separate diagnostic artifact.
    double raw_tangent_k00{0.0};
    double raw_tangent_k0y{0.0};
    double raw_tangent_ky0{0.0};
    double raw_tangent_kyy{0.0};
    int newton_iterations{0};
    double final_axial_force_residual{0.0};
};

struct ReducedRCColumnSectionFiberRecord {
    int step{0};
    double load_factor{0.0};
    double solved_axial_strain{0.0};
    double curvature_y{0.0};
    bool zero_curvature_anchor{false};
    std::size_t fiber_index{0};
    double y{0.0};
    double z{0.0};
    double area{0.0};
    RCSectionZoneKind zone{RCSectionZoneKind::cover_top};
    RCSectionMaterialRole material_role{
        RCSectionMaterialRole::unconfined_concrete};
    double strain_xx{0.0};
    double stress_xx{0.0};
    double tangent_xx{0.0};
    double axial_force_contribution{0.0};
    double moment_y_contribution{0.0};
    double raw_tangent_k00_contribution{0.0};
    double raw_tangent_k0y_contribution{0.0};
    double raw_tangent_kyy_contribution{0.0};
};

struct ReducedRCColumnSectionBaselineTimingSummary {
    double total_wall_seconds{0.0};
    double solve_wall_seconds{0.0};
    double output_write_wall_seconds{0.0};
};

struct ReducedRCColumnSectionBaselineResult {
    std::vector<ReducedRCColumnSectionBaselineRecord> records{};
    std::vector<ReducedRCColumnSectionFiberRecord> fiber_history_records{};
    ReducedRCColumnSectionBaselineTimingSummary timing{};
    bool completed_successfully{false};

    [[nodiscard]] bool empty() const noexcept { return records.empty(); }
};

[[nodiscard]] ReducedRCColumnSectionBaselineResult
run_reduced_rc_column_section_moment_curvature_baseline(
    const ReducedRCColumnSectionBaselineRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_SECTION_BASELINE_HH
