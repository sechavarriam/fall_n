#ifndef FALL_N_REDUCED_RC_MULTISCALE_VALIDATION_START_CATALOG_HH
#define FALL_N_REDUCED_RC_MULTISCALE_VALIDATION_START_CATALOG_HH

// =============================================================================
//  ReducedRCMultiscaleValidationStartCatalog.hh
// =============================================================================
//
//  Compile-time contract for the transition from the reduced RC validation
//  reboot to the first multiscale campaign.
//
//  This file deliberately contains no numerical hot path.  It is a stable seam
//  for documentation, Python/Julia wrappers, benchmark scripts, and VTK output
//  checks.  The scientific rule encoded here is simple: multiscale validation
//  starts with one-way replay of trusted structural histories into selected
//  local XFEM sites; two-way FE2 is enabled only after the local model exposes
//  the required kinematic, fracture, steel, solver, and visualization evidence.
//
// =============================================================================

#include <array>
#include <cstddef>
#include <string_view>

namespace fall_n {

enum class ReducedRCValidationModelScaleKind {
    structural_global,
    continuum_local,
    xfem_local,
    multiscale_coupled
};

enum class ReducedRCVTKFieldLocationKind {
    collection_metadata,
    structural_axis,
    structural_section,
    continuum_node,
    continuum_gauss_point,
    xfem_crack_surface,
    rebar_line
};

enum class ReducedRCMultiscaleStartStageKind {
    one_way_replay,
    local_site_batch,
    elastic_fe2_smoke,
    enriched_fe2_guarded_smoke
};

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCValidationModelScaleKind kind) noexcept
{
    switch (kind) {
        case ReducedRCValidationModelScaleKind::structural_global:
            return "structural_global";
        case ReducedRCValidationModelScaleKind::continuum_local:
            return "continuum_local";
        case ReducedRCValidationModelScaleKind::xfem_local:
            return "xfem_local";
        case ReducedRCValidationModelScaleKind::multiscale_coupled:
            return "multiscale_coupled";
    }
    return "unknown_validation_model_scale";
}

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCVTKFieldLocationKind kind) noexcept
{
    switch (kind) {
        case ReducedRCVTKFieldLocationKind::collection_metadata:
            return "collection_metadata";
        case ReducedRCVTKFieldLocationKind::structural_axis:
            return "structural_axis";
        case ReducedRCVTKFieldLocationKind::structural_section:
            return "structural_section";
        case ReducedRCVTKFieldLocationKind::continuum_node:
            return "continuum_node";
        case ReducedRCVTKFieldLocationKind::continuum_gauss_point:
            return "continuum_gauss_point";
        case ReducedRCVTKFieldLocationKind::xfem_crack_surface:
            return "xfem_crack_surface";
        case ReducedRCVTKFieldLocationKind::rebar_line:
            return "rebar_line";
    }
    return "unknown_vtk_field_location";
}

[[nodiscard]] constexpr std::string_view to_string(
    ReducedRCMultiscaleStartStageKind kind) noexcept
{
    switch (kind) {
        case ReducedRCMultiscaleStartStageKind::one_way_replay:
            return "one_way_replay";
        case ReducedRCMultiscaleStartStageKind::local_site_batch:
            return "local_site_batch";
        case ReducedRCMultiscaleStartStageKind::elastic_fe2_smoke:
            return "elastic_fe2_smoke";
        case ReducedRCMultiscaleStartStageKind::enriched_fe2_guarded_smoke:
            return "enriched_fe2_guarded_smoke";
    }
    return "unknown_multiscale_start_stage";
}

struct ReducedRCVTKFieldSpec {
    std::string_view name{};
    ReducedRCValidationModelScaleKind scale_kind{
        ReducedRCValidationModelScaleKind::xfem_local};
    ReducedRCVTKFieldLocationKind location_kind{
        ReducedRCVTKFieldLocationKind::continuum_node};
    int components{1};
    bool required_for_pseudo_time{true};
    bool required_for_physical_time{false};
    bool required_for_multiscale_replay{false};
    std::string_view interpretation{};
};

struct ReducedRCMultiscaleStartStageSpec {
    ReducedRCMultiscaleStartStageKind stage_kind{
        ReducedRCMultiscaleStartStageKind::one_way_replay};
    std::string_view key{};
    std::string_view driver_hint{};
    std::string_view prerequisite_gate{};
    std::string_view expected_artifact{};
    bool may_run_before_two_way_fe2{true};
    bool requires_xfem_enriched_dofs{false};
    bool writes_vtk_time_series{true};
};

[[nodiscard]] constexpr auto canonical_reduced_rc_vtk_field_table() noexcept
{
    using Location = ReducedRCVTKFieldLocationKind;
    using Scale = ReducedRCValidationModelScaleKind;

    return std::to_array({
        ReducedRCVTKFieldSpec{
            .name = "pseudo_time",
            .scale_kind = Scale::multiscale_coupled,
            .location_kind = Location::collection_metadata,
            .components = 1,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "monotone/load-path parameter for static cyclic replay and animation"},
        ReducedRCVTKFieldSpec{
            .name = "physical_time",
            .scale_kind = Scale::multiscale_coupled,
            .location_kind = Location::collection_metadata,
            .components = 1,
            .required_for_pseudo_time = false,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "dynamic-analysis time carried beside pseudo-time when seismic validation starts"},
        ReducedRCVTKFieldSpec{
            .name = "displacement",
            .scale_kind = Scale::continuum_local,
            .location_kind = Location::continuum_node,
            .components = 3,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "nodal displacement used for Warp By Vector and local deformation review"},
        ReducedRCVTKFieldSpec{
            .name = "section_curvature_y",
            .scale_kind = Scale::structural_global,
            .location_kind = Location::structural_section,
            .components = 1,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "structural curvature driving local replay site selection"},
        ReducedRCVTKFieldSpec{
            .name = "section_moment_y",
            .scale_kind = Scale::structural_global,
            .location_kind = Location::structural_section,
            .components = 1,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "structural generalized force paired with curvature"},
        ReducedRCVTKFieldSpec{
            .name = "crack_opening",
            .scale_kind = Scale::xfem_local,
            .location_kind = Location::xfem_crack_surface,
            .components = 1,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "normal opening of the explicit enriched crack surface"},
        ReducedRCVTKFieldSpec{
            .name = "cohesive_traction",
            .scale_kind = Scale::xfem_local,
            .location_kind = Location::xfem_crack_surface,
            .components = 3,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "interface traction in the declared cohesive traction measure"},
        ReducedRCVTKFieldSpec{
            .name = "cohesive_damage",
            .scale_kind = Scale::xfem_local,
            .location_kind = Location::xfem_crack_surface,
            .components = 1,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "scalar crack damage/localization indicator"},
        ReducedRCVTKFieldSpec{
            .name = "enriched_dof_norm",
            .scale_kind = Scale::xfem_local,
            .location_kind = Location::continuum_node,
            .components = 1,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "nodal norm of shifted-Heaviside enrichment amplitude"},
        ReducedRCVTKFieldSpec{
            .name = "steel_stress",
            .scale_kind = Scale::xfem_local,
            .location_kind = Location::rebar_line,
            .components = 1,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = true,
            .interpretation = "longitudinal steel stress for structural/local equivalence"},
        ReducedRCVTKFieldSpec{
            .name = "work_loop_density",
            .scale_kind = Scale::xfem_local,
            .location_kind = Location::xfem_crack_surface,
            .components = 1,
            .required_for_pseudo_time = true,
            .required_for_physical_time = true,
            .required_for_multiscale_replay = false,
            .interpretation = "accumulated local interface work used to compare hysteresis energy"}
    });
}

[[nodiscard]] constexpr auto
canonical_reduced_rc_multiscale_start_stage_table() noexcept
{
    using Stage = ReducedRCMultiscaleStartStageKind;

    return std::to_array({
        ReducedRCMultiscaleStartStageSpec{
            .stage_kind = Stage::one_way_replay,
            .key = "structural_history_to_xfem_site_replay",
            .driver_hint = "replay structural section histories into selected XFEM local sites",
            .prerequisite_gate = "structural reference closed and XFEM local 200 mm gate complete",
            .expected_artifact = "multiscale_replay_site_catalog.json",
            .may_run_before_two_way_fe2 = true,
            .requires_xfem_enriched_dofs = true,
            .writes_vtk_time_series = true},
        ReducedRCMultiscaleStartStageSpec{
            .stage_kind = Stage::local_site_batch,
            .key = "parallel_local_site_batch",
            .driver_hint = "run independent local XFEM sites with warm-started Newton and cached seeds",
            .prerequisite_gate = "one-way replay site histories generated",
            .expected_artifact = "local_site_batch_timing.csv",
            .may_run_before_two_way_fe2 = true,
            .requires_xfem_enriched_dofs = true,
            .writes_vtk_time_series = true},
        ReducedRCMultiscaleStartStageSpec{
            .stage_kind = Stage::elastic_fe2_smoke,
            .key = "elastic_fe2_smoke",
            .driver_hint = "exercise bidirectional FE2 plumbing with elastic local stiffness before nonlinear XFEM feedback",
            .prerequisite_gate = "local site batch establishes memory and timing envelope",
            .expected_artifact = "fe2_elastic_smoke_manifest.json",
            .may_run_before_two_way_fe2 = true,
            .requires_xfem_enriched_dofs = false,
            .writes_vtk_time_series = true},
        ReducedRCMultiscaleStartStageSpec{
            .stage_kind = Stage::enriched_fe2_guarded_smoke,
            .key = "xfem_fe2_guarded_one_step",
            .driver_hint = "activate one enriched local site in a guarded two-way FE2 step",
            .prerequisite_gate = "elastic FE2 smoke passes and XFEM site has reusable seed state",
            .expected_artifact = "xfem_fe2_guarded_one_step_manifest.json",
            .may_run_before_two_way_fe2 = false,
            .requires_xfem_enriched_dofs = true,
            .writes_vtk_time_series = true}
    });
}

inline constexpr auto canonical_reduced_rc_vtk_field_table_v =
    canonical_reduced_rc_vtk_field_table();

inline constexpr auto canonical_reduced_rc_multiscale_start_stage_table_v =
    canonical_reduced_rc_multiscale_start_stage_table();

template <std::size_t N>
[[nodiscard]] constexpr std::size_t count_required_replay_vtk_fields(
    const std::array<ReducedRCVTKFieldSpec, N>& rows) noexcept
{
    std::size_t count = 0;
    for (const auto& row : rows) {
        if (row.required_for_multiscale_replay) {
            ++count;
        }
    }
    return count;
}

template <std::size_t N>
[[nodiscard]] constexpr bool vtk_field_table_has_crack_visualization(
    const std::array<ReducedRCVTKFieldSpec, N>& rows) noexcept
{
    bool has_opening = false;
    bool has_traction = false;
    bool has_damage = false;
    for (const auto& row : rows) {
        if (row.location_kind !=
            ReducedRCVTKFieldLocationKind::xfem_crack_surface) {
            continue;
        }
        has_opening = has_opening || row.name == "crack_opening";
        has_traction = has_traction || row.name == "cohesive_traction";
        has_damage = has_damage || row.name == "cohesive_damage";
    }
    return has_opening && has_traction && has_damage;
}

template <std::size_t N>
[[nodiscard]] constexpr bool multiscale_start_table_is_ordered(
    const std::array<ReducedRCMultiscaleStartStageSpec, N>& rows) noexcept
{
    if constexpr (N < 4) {
        return false;
    }
    return rows[0].stage_kind == ReducedRCMultiscaleStartStageKind::one_way_replay &&
           rows[1].stage_kind == ReducedRCMultiscaleStartStageKind::local_site_batch &&
           rows[2].stage_kind == ReducedRCMultiscaleStartStageKind::elastic_fe2_smoke &&
           rows[3].stage_kind == ReducedRCMultiscaleStartStageKind::enriched_fe2_guarded_smoke;
}

inline constexpr std::size_t
canonical_reduced_rc_required_replay_vtk_field_count_v =
    count_required_replay_vtk_fields(canonical_reduced_rc_vtk_field_table_v);

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_MULTISCALE_VALIDATION_START_CATALOG_HH
