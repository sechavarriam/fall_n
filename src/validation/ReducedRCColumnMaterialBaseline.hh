#ifndef FALL_N_REDUCED_RC_COLUMN_MATERIAL_BASELINE_HH
#define FALL_N_REDUCED_RC_COLUMN_MATERIAL_BASELINE_HH

// =============================================================================
//  ReducedRCColumnMaterialBaseline.hh
// =============================================================================
//
//  Independent uniaxial-material baseline for the reduced RC-column reboot.
//
//  The section-level external bridge already showed that geometry, fiber layout,
//  and observable extraction close essentially exactly under elasticized parity.
//  The remaining external gap is therefore most likely constitutive. This
//  baseline isolates that question by driving the audited reduced-column
//  uniaxial ingredients directly:
//
//    - Menegotto-Pinto reinforcing steel
//    - Kent-Park unconfined concrete
//
//  The resulting stress-strain and tangent histories are later compared against
//  an OpenSeesPy material-testing bridge before reinterpreting the
//  section-level benchmark.
//
// =============================================================================

#include "src/validation/ReducedRCColumnReferenceSpec.hh"
#include "src/analysis/LocalModelTaxonomy.hh"

#include <string>
#include <vector>

namespace fall_n::validation_reboot {

enum class ReducedRCColumnMaterialReferenceKind {
    steel_rebar,
    concrete_unconfined,
};

enum class ReducedRCColumnMaterialProtocolKind {
    monotonic,
    cyclic,
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnMaterialReferenceKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnMaterialReferenceKind::steel_rebar:
            return "steel_rebar";
        case ReducedRCColumnMaterialReferenceKind::concrete_unconfined:
            return "concrete_unconfined";
    }
    return "unknown";
}

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnMaterialProtocolKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnMaterialProtocolKind::monotonic:
            return "monotonic";
        case ReducedRCColumnMaterialProtocolKind::cyclic:
            return "cyclic";
    }
    return "unknown";
}

struct ReducedRCColumnMaterialBaselineRunSpec {
    ReducedRCColumnReferenceSpec reference_spec{};
    ReducedRCColumnMaterialReferenceKind material_kind{
        ReducedRCColumnMaterialReferenceKind::steel_rebar};
    ReducedRCColumnMaterialProtocolKind protocol_kind{
        ReducedRCColumnMaterialProtocolKind::cyclic};
    double monotonic_target_strain{0.0};
    std::vector<double> amplitude_levels{};
    struct ProtocolPoint {
        int step{0};
        double strain{0.0};
    };
    std::vector<ProtocolPoint> custom_protocol{};
    int steps_per_branch{40};
    bool write_csv{true};
    bool print_progress{true};
};

[[nodiscard]] inline fall_n::LocalModelTaxonomy
describe_reduced_rc_column_material_local_model(
    const ReducedRCColumnMaterialBaselineRunSpec& spec) noexcept
{
    return {
        .discretization_kind =
            fall_n::LocalModelDiscretizationKind::uniaxial_constitutive_point,
        .fracture_representation_kind =
            spec.material_kind ==
                    ReducedRCColumnMaterialReferenceKind::concrete_unconfined
                ? fall_n::LocalFractureRepresentationKind::
                      smeared_internal_state
                : fall_n::LocalFractureRepresentationKind::none,
        .reinforcement_representation_kind =
            spec.material_kind ==
                    ReducedRCColumnMaterialReferenceKind::steel_rebar
                ? fall_n::LocalReinforcementRepresentationKind::
                      constitutive_section_fibers
                : fall_n::LocalReinforcementRepresentationKind::none,
        .maturity_kind =
            fall_n::LocalModelMaturityKind::comparison_control,
        .supports_discrete_crack_geometry = false,
        .requires_enriched_dofs = false,
        .requires_skeleton_trace_unknowns = false,
        .suitable_for_future_multiscale_local_model = false,
        .notes =
            "Independent uniaxial constitutive-point audit for the reduced RC "
            "column ingredients, used to isolate material behavior from "
            "section and continuum discretization effects."};
}

struct ReducedRCColumnMaterialBaselineRecord {
    int step{0};
    double strain{0.0};
    double stress_mpa{0.0};
    double tangent_mpa{0.0};
    double energy_density_mpa{0.0};
};

struct ReducedRCColumnMaterialBaselineTimingSummary {
    double total_wall_seconds{0.0};
    double solve_wall_seconds{0.0};
    double output_write_wall_seconds{0.0};
};

struct ReducedRCColumnMaterialBaselineResult {
    std::vector<ReducedRCColumnMaterialBaselineRecord> records{};
    ReducedRCColumnMaterialBaselineTimingSummary timing{};
    bool completed_successfully{false};

    [[nodiscard]] bool empty() const noexcept { return records.empty(); }
};

[[nodiscard]] ReducedRCColumnMaterialBaselineResult
run_reduced_rc_column_material_baseline(
    const ReducedRCColumnMaterialBaselineRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_MATERIAL_BASELINE_HH
