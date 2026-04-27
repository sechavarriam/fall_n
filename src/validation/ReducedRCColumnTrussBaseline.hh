#ifndef FALL_N_REDUCED_RC_COLUMN_TRUSS_BASELINE_HH
#define FALL_N_REDUCED_RC_COLUMN_TRUSS_BASELINE_HH

// =============================================================================
//  ReducedRCColumnTrussBaseline.hh
// =============================================================================
//
//  Standalone benchmark for the 3-node truss that later becomes the
//  reinforcement carrier inside the promoted continuum local model.
//
//  Why this benchmark exists:
//
//    1. The continuum-vs-structural gap is too large to keep assuming that the
//       embedded bar family is innocent.
//    2. Before embedding a quadratic truss inside cracked concrete, we need to
//       verify that the truss alone reproduces the same Menegotto-Pinto steel
//       hysteresis used in the audited RC section.
//    3. The benchmark therefore compares, under the exact same strain history:
//
//         direct material point  vs  TrussElement<3,3> Gauss points
//
//  The benchmark uses a prescribed affine displacement field on a single
//  quadratic bar so that the exact solution is uniform axial strain. Any gap is
//  therefore traceable to the truss formulation, quadrature/interpolation, or
//  material integration order—not to a global solve or geometric mismatch.
//
// =============================================================================

#include "src/analysis/LocalModelTaxonomy.hh"
#include "src/validation/ReducedRCColumnReferenceSpec.hh"
#include "src/validation/ReducedRCColumnMaterialBaseline.hh"

#include <string>
#include <vector>

namespace fall_n::validation_reboot {

enum class ReducedRCColumnTrussProtocolKind {
    monotonic_compression,
    cyclic_compression_return,
};

[[nodiscard]] constexpr const char*
to_string(ReducedRCColumnTrussProtocolKind kind) noexcept
{
    switch (kind) {
        case ReducedRCColumnTrussProtocolKind::monotonic_compression:
            return "monotonic_compression";
        case ReducedRCColumnTrussProtocolKind::cyclic_compression_return:
            return "cyclic_compression_return";
    }
    return "unknown";
}

struct ReducedRCColumnTrussBaselineRunSpec {
    ReducedRCColumnReferenceSpec reference_spec{};
    ReducedRCColumnTrussProtocolKind protocol_kind{
        ReducedRCColumnTrussProtocolKind::cyclic_compression_return};
    double element_length_m{0.0};
    double area_m2{0.0};
    double monotonic_target_strain{0.0};
    std::vector<double> compression_amplitude_levels{};
    std::vector<
        ReducedRCColumnMaterialBaselineRunSpec::ProtocolPoint> custom_protocol{};
    int steps_per_branch{40};
    bool write_csv{true};
    bool print_progress{true};
};

[[nodiscard]] inline double
resolve_reduced_rc_column_truss_length_m(
    const ReducedRCColumnTrussBaselineRunSpec& spec) noexcept
{
    return spec.element_length_m > 0.0 ? spec.element_length_m
                                       : spec.reference_spec.column_height_m;
}

[[nodiscard]] inline double
resolve_reduced_rc_column_truss_area_m2(
    const ReducedRCColumnTrussBaselineRunSpec& spec) noexcept
{
    if (spec.area_m2 > 0.0) {
        return spec.area_m2;
    }
    const double d = spec.reference_spec.longitudinal_bar_diameter_m;
    return 0.25 * 3.14159265358979323846 * d * d;
}

[[nodiscard]] inline fall_n::LocalModelTaxonomy
describe_reduced_rc_column_truss_local_model(
    const ReducedRCColumnTrussBaselineRunSpec&) noexcept
{
    return {
        .discretization_kind =
            fall_n::LocalModelDiscretizationKind::axial_line_member,
        .fracture_representation_kind =
            fall_n::LocalFractureRepresentationKind::none,
        .reinforcement_representation_kind =
            fall_n::LocalReinforcementRepresentationKind::standalone_truss_line,
        .maturity_kind = fall_n::LocalModelMaturityKind::comparison_control,
        .supports_discrete_crack_geometry = false,
        .requires_enriched_dofs = false,
        .requires_skeleton_trace_unknowns = false,
        .suitable_for_future_multiscale_local_model = false,
        .notes =
            "Standalone quadratic truss benchmark used to isolate the axial "
            "rebar carrier before embedding it in the promoted reduced-column "
            "continuum local model."};
}

struct ReducedRCColumnTrussBaselineRecord {
    int step{0};
    double axial_strain{0.0};
    double axial_stress_mpa{0.0};
    double tangent_mpa{0.0};
    double tangent_from_element_mpa{0.0};
    double end_displacement_m{0.0};
    double axial_force_mn{0.0};
    double middle_node_force_mn{0.0};
    double energy_density_mpa{0.0};
    double truss_work_mn_m{0.0};
    double gp_strain_spread{0.0};
    double gp_stress_spread_mpa{0.0};
    double gp_tangent_spread_mpa{0.0};
};

struct ReducedRCColumnTrussGaussRecord {
    int step{0};
    int gauss_point{0};
    double xi{0.0};
    double axial_strain{0.0};
    double axial_stress_mpa{0.0};
    double tangent_mpa{0.0};
};

struct ReducedRCColumnTrussBaselineComparisonSummary {
    double max_abs_stress_error_mpa{0.0};
    double rms_abs_stress_error_mpa{0.0};
    double max_abs_tangent_error_mpa{0.0};
    double rms_abs_tangent_error_mpa{0.0};
    double max_abs_element_tangent_error_mpa{0.0};
    double rms_abs_element_tangent_error_mpa{0.0};
    double max_abs_energy_density_error_mpa{0.0};
    double rms_abs_energy_density_error_mpa{0.0};
    double max_abs_axial_force_closure_mn{0.0};
    double max_abs_gp_strain_spread{0.0};
    double max_abs_gp_stress_spread_mpa{0.0};
    double max_abs_gp_tangent_spread_mpa{0.0};
    double max_abs_middle_node_force_mn{0.0};
};

struct ReducedRCColumnTrussBaselineTimingSummary {
    double total_wall_seconds{0.0};
    double analysis_wall_seconds{0.0};
    double output_write_wall_seconds{0.0};
};

struct ReducedRCColumnTrussBaselineResult {
    std::vector<ReducedRCColumnMaterialBaselineRecord> material_records{};
    std::vector<ReducedRCColumnTrussBaselineRecord> truss_records{};
    std::vector<ReducedRCColumnTrussGaussRecord> gauss_records{};
    ReducedRCColumnTrussBaselineComparisonSummary comparison{};
    ReducedRCColumnTrussBaselineTimingSummary timing{};
    bool completed_successfully{false};
};

[[nodiscard]] ReducedRCColumnTrussBaselineResult
run_reduced_rc_column_truss_baseline(
    const ReducedRCColumnTrussBaselineRunSpec& spec,
    const std::string& out_dir);

} // namespace fall_n::validation_reboot

#endif // FALL_N_REDUCED_RC_COLUMN_TRUSS_BASELINE_HH
