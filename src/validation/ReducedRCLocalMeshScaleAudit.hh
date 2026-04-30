#ifndef FALL_N_REDUCED_RC_LOCAL_MESH_SCALE_AUDIT_HH
#define FALL_N_REDUCED_RC_LOCAL_MESH_SCALE_AUDIT_HH

// =============================================================================
//  ReducedRCLocalMeshScaleAudit.hh
// =============================================================================
//
//  Cheap scaling audit for the reduced-RC local-model validation campaign.
//
//  The multiscale target is not just to match a structural hysteresis loop with
//  a tuned local response; it is to make richer local models affordable enough
//  to expose crack and steel states at many structural integration sites. This
//  header therefore keeps a deterministic pre-run estimate of:
//
//    - host continuum nodes/elements/material points,
//    - shifted-Heaviside XFEM enriched nodes and DOFs,
//    - explicit longitudinal-bar DOFs,
//    - sparse matrix/workspace/material-state memory, and
//    - solver/OpenMP scaling advice.
//
//  The estimates are intentionally conservative and are not used in the
//  numerical hot path. They form a validation/documentation seam that lets us
//  decide whether a candidate such as 7x7x25 should be run with direct LU,
//  iterative/domain-decomposition solvers, seed caches, or site-level OpenMP.
//
// =============================================================================

#include <algorithm>
#include <cstddef>
#include <string_view>

namespace fall_n {

enum class ReducedRCLocalCellTopologyKind {
    hex8_lagrange,
    hex20_serendipity,
    hex27_lagrange
};

enum class ReducedRCLocalConstitutiveCostKind {
    elastic_proxy,
    cyclic_crack_band_xfem,
    ko_bathe_heavy_reference
};

enum class ReducedRCLocalSolverScalingAdviceKind {
    direct_lu_smoke_ok,
    direct_lu_reference_only,
    iterative_preconditioner_required,
    domain_decomposition_or_multiscale_batch_required
};

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCLocalCellTopologyKind kind) noexcept
{
    switch (kind) {
        case ReducedRCLocalCellTopologyKind::hex8_lagrange:
            return "hex8_lagrange";
        case ReducedRCLocalCellTopologyKind::hex20_serendipity:
            return "hex20_serendipity";
        case ReducedRCLocalCellTopologyKind::hex27_lagrange:
            return "hex27_lagrange";
    }
    return "unknown_reduced_rc_local_cell_topology";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCLocalConstitutiveCostKind kind) noexcept
{
    switch (kind) {
        case ReducedRCLocalConstitutiveCostKind::elastic_proxy:
            return "elastic_proxy";
        case ReducedRCLocalConstitutiveCostKind::cyclic_crack_band_xfem:
            return "cyclic_crack_band_xfem";
        case ReducedRCLocalConstitutiveCostKind::ko_bathe_heavy_reference:
            return "ko_bathe_heavy_reference";
    }
    return "unknown_reduced_rc_local_constitutive_cost";
}

[[nodiscard]] constexpr std::string_view
to_string(ReducedRCLocalSolverScalingAdviceKind kind) noexcept
{
    switch (kind) {
        case ReducedRCLocalSolverScalingAdviceKind::direct_lu_smoke_ok:
            return "direct_lu_smoke_ok";
        case ReducedRCLocalSolverScalingAdviceKind::direct_lu_reference_only:
            return "direct_lu_reference_only";
        case ReducedRCLocalSolverScalingAdviceKind::iterative_preconditioner_required:
            return "iterative_preconditioner_required";
        case ReducedRCLocalSolverScalingAdviceKind::
            domain_decomposition_or_multiscale_batch_required:
            return "domain_decomposition_or_multiscale_batch_required";
    }
    return "unknown_reduced_rc_local_solver_scaling_advice";
}

struct ReducedRCLocalMeshScaleInput {
    std::size_t nx{1};
    std::size_t ny{1};
    std::size_t nz{4};
    ReducedRCLocalCellTopologyKind topology{
        ReducedRCLocalCellTopologyKind::hex8_lagrange};
    ReducedRCLocalConstitutiveCostKind constitutive_cost{
        ReducedRCLocalConstitutiveCostKind::cyclic_crack_band_xfem};
    bool shifted_heaviside_xfem{true};
    std::size_t planar_crack_count{1};
    std::size_t longitudinal_bar_count{8};
    bool bars_have_independent_nodes{true};
    std::size_t rebar_subsegments_per_host_element{1};
    bool crack_crossing_bridge_enabled{true};
    std::size_t solid_dofs_per_node{3};
    std::size_t enriched_dofs_per_node{3};
    std::size_t rebar_dofs_per_node{3};
    std::size_t workspace_vector_count{18};
    double sparse_matrix_safety_factor{1.35};
    double direct_factor_fill_multiplier{42.0};
};

struct ReducedRCLocalMeshScaleAudit {
    std::size_t host_element_count{0};
    std::size_t host_node_count{0};
    std::size_t host_nodes_per_element{0};
    std::size_t host_gauss_points_per_element{0};
    std::size_t host_material_point_count{0};
    std::size_t enriched_node_count{0};
    std::size_t host_displacement_dofs{0};
    std::size_t enrichment_dofs{0};
    std::size_t rebar_node_count{0};
    std::size_t rebar_dofs{0};
    std::size_t crack_crossing_bridge_element_count{0};
    std::size_t estimated_total_state_dofs{0};
    std::size_t estimated_sparse_nonzeros{0};
    double vector_mib{0.0};
    double newton_workspace_mib{0.0};
    double sparse_matrix_mib{0.0};
    double direct_factorization_risk_mib{0.0};
    double material_state_mib{0.0};
    double estimated_hot_state_mib{0.0};
    ReducedRCLocalSolverScalingAdviceKind solver_advice{
        ReducedRCLocalSolverScalingAdviceKind::direct_lu_smoke_ok};
    bool seed_state_cache_recommended{false};
    bool newton_warm_start_recommended{false};
    bool site_level_openmp_recommended{false};
    bool global_petsc_assembly_openmp_recommended{false};
    bool symmetric_matrix_storage_recommended{false};
    bool symmetric_matrix_storage_requires_tangent_audit{true};
    bool block_matrix_storage_candidate{false};
    bool field_split_or_asm_preconditioner_recommended{false};
    bool plain_gmres_ilu_rejected_for_enriched_branch{false};
};

[[nodiscard]] constexpr std::size_t positive_or_one(std::size_t value) noexcept
{
    return value == 0 ? std::size_t{1} : value;
}

[[nodiscard]] constexpr std::size_t
nodes_per_element(ReducedRCLocalCellTopologyKind topology) noexcept
{
    switch (topology) {
        case ReducedRCLocalCellTopologyKind::hex8_lagrange:
            return 8;
        case ReducedRCLocalCellTopologyKind::hex20_serendipity:
            return 20;
        case ReducedRCLocalCellTopologyKind::hex27_lagrange:
            return 27;
    }
    return 8;
}

[[nodiscard]] constexpr std::size_t
gauss_points_per_element(ReducedRCLocalCellTopologyKind topology) noexcept
{
    switch (topology) {
        case ReducedRCLocalCellTopologyKind::hex8_lagrange:
            return 8;
        case ReducedRCLocalCellTopologyKind::hex20_serendipity:
        case ReducedRCLocalCellTopologyKind::hex27_lagrange:
            return 27;
    }
    return 8;
}

[[nodiscard]] constexpr std::size_t
structured_hex_node_count(std::size_t nx,
                          std::size_t ny,
                          std::size_t nz,
                          ReducedRCLocalCellTopologyKind topology) noexcept
{
    nx = positive_or_one(nx);
    ny = positive_or_one(ny);
    nz = positive_or_one(nz);

    const auto corners = (nx + 1) * (ny + 1) * (nz + 1);
    switch (topology) {
        case ReducedRCLocalCellTopologyKind::hex8_lagrange:
            return corners;
        case ReducedRCLocalCellTopologyKind::hex20_serendipity: {
            const auto x_edge_mid = nx * (ny + 1) * (nz + 1);
            const auto y_edge_mid = (nx + 1) * ny * (nz + 1);
            const auto z_edge_mid = (nx + 1) * (ny + 1) * nz;
            return corners + x_edge_mid + y_edge_mid + z_edge_mid;
        }
        case ReducedRCLocalCellTopologyKind::hex27_lagrange:
            return (2 * nx + 1) * (2 * ny + 1) * (2 * nz + 1);
    }
    return corners;
}

[[nodiscard]] constexpr std::size_t
structured_cross_section_node_count(
    std::size_t nx,
    std::size_t ny,
    ReducedRCLocalCellTopologyKind topology) noexcept
{
    nx = positive_or_one(nx);
    ny = positive_or_one(ny);

    switch (topology) {
        case ReducedRCLocalCellTopologyKind::hex8_lagrange:
            return (nx + 1) * (ny + 1);
        case ReducedRCLocalCellTopologyKind::hex20_serendipity:
            return (nx + 1) * (ny + 1) + nx * (ny + 1) +
                   (nx + 1) * ny;
        case ReducedRCLocalCellTopologyKind::hex27_lagrange:
            return (2 * nx + 1) * (2 * ny + 1);
    }
    return (nx + 1) * (ny + 1);
}

[[nodiscard]] constexpr std::size_t
xfem_support_layers(ReducedRCLocalCellTopologyKind topology) noexcept
{
    return topology == ReducedRCLocalCellTopologyKind::hex8_lagrange
               ? std::size_t{2}
               : std::size_t{3};
}

[[nodiscard]] constexpr std::size_t material_state_bytes_per_point(
    ReducedRCLocalConstitutiveCostKind kind) noexcept
{
    switch (kind) {
        case ReducedRCLocalConstitutiveCostKind::elastic_proxy:
            return 128;
        case ReducedRCLocalConstitutiveCostKind::cyclic_crack_band_xfem:
            return 768;
        case ReducedRCLocalConstitutiveCostKind::ko_bathe_heavy_reference:
            return 2048;
    }
    return 768;
}

[[nodiscard]] constexpr double bytes_to_mib(std::size_t bytes) noexcept
{
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

[[nodiscard]] constexpr ReducedRCLocalSolverScalingAdviceKind
classify_reduced_rc_local_solver_scaling(
    std::size_t total_dofs,
    double factorization_risk_mib,
    std::size_t host_material_point_count) noexcept
{
    if (total_dofs <= 1500 && factorization_risk_mib <= 512.0) {
        return ReducedRCLocalSolverScalingAdviceKind::direct_lu_smoke_ok;
    }
    if (total_dofs <= 6500 && factorization_risk_mib <= 2048.0) {
        return ReducedRCLocalSolverScalingAdviceKind::direct_lu_reference_only;
    }
    if (total_dofs <= 20000 && host_material_point_count <= 120000) {
        return ReducedRCLocalSolverScalingAdviceKind::
            iterative_preconditioner_required;
    }
    return ReducedRCLocalSolverScalingAdviceKind::
        domain_decomposition_or_multiscale_batch_required;
}

[[nodiscard]] constexpr ReducedRCLocalMeshScaleAudit
make_reduced_rc_local_mesh_scale_audit(
    ReducedRCLocalMeshScaleInput input) noexcept
{
    input.nx = positive_or_one(input.nx);
    input.ny = positive_or_one(input.ny);
    input.nz = positive_or_one(input.nz);
    input.planar_crack_count = input.planar_crack_count == 0
                                   ? std::size_t{1}
                                   : input.planar_crack_count;
    input.longitudinal_bar_count =
        input.longitudinal_bar_count == 0 ? std::size_t{0}
                                          : input.longitudinal_bar_count;
    input.rebar_subsegments_per_host_element =
        input.rebar_subsegments_per_host_element == 0
            ? std::size_t{1}
            : input.rebar_subsegments_per_host_element;
    input.solid_dofs_per_node = positive_or_one(input.solid_dofs_per_node);
    input.enriched_dofs_per_node =
        positive_or_one(input.enriched_dofs_per_node);
    input.rebar_dofs_per_node = positive_or_one(input.rebar_dofs_per_node);
    input.workspace_vector_count =
        positive_or_one(input.workspace_vector_count);

    ReducedRCLocalMeshScaleAudit audit{};
    audit.host_element_count = input.nx * input.ny * input.nz;
    audit.host_node_count =
        structured_hex_node_count(input.nx, input.ny, input.nz,
                                  input.topology);
    audit.host_nodes_per_element = nodes_per_element(input.topology);
    audit.host_gauss_points_per_element = gauss_points_per_element(input.topology);
    audit.host_material_point_count =
        audit.host_element_count * audit.host_gauss_points_per_element;

    if (input.shifted_heaviside_xfem) {
        const auto section_nodes =
            structured_cross_section_node_count(input.nx, input.ny,
                                                input.topology);
        const auto support_nodes =
            section_nodes * xfem_support_layers(input.topology) *
            input.planar_crack_count;
        audit.enriched_node_count =
            std::min(support_nodes, audit.host_node_count);
    }

    audit.host_displacement_dofs =
        audit.host_node_count * input.solid_dofs_per_node;
    audit.enrichment_dofs =
        audit.enriched_node_count * input.enriched_dofs_per_node;

    if (input.bars_have_independent_nodes &&
        input.longitudinal_bar_count > 0) {
        audit.rebar_node_count =
            input.longitudinal_bar_count *
            (input.nz * input.rebar_subsegments_per_host_element + 1);
    }
    audit.rebar_dofs = audit.rebar_node_count * input.rebar_dofs_per_node;
    audit.crack_crossing_bridge_element_count =
        input.crack_crossing_bridge_enabled
            ? input.longitudinal_bar_count * input.planar_crack_count
            : std::size_t{0};
    audit.estimated_total_state_dofs =
        audit.host_displacement_dofs + audit.enrichment_dofs +
        audit.rebar_dofs;

    const auto host_element_dofs =
        audit.host_nodes_per_element * input.solid_dofs_per_node;
    const auto xfem_extra_width = input.shifted_heaviside_xfem
                                      ? host_element_dofs * std::size_t{3}
                                      : std::size_t{0};
    const auto rebar_extra_width =
        audit.rebar_dofs > 0 ? std::size_t{18} : std::size_t{0};
    const auto average_row_width =
        std::min(audit.estimated_total_state_dofs,
                 host_element_dofs * std::size_t{4} + xfem_extra_width +
                     rebar_extra_width);
    audit.estimated_sparse_nonzeros =
        audit.estimated_total_state_dofs * average_row_width;

    audit.vector_mib =
        bytes_to_mib(audit.estimated_total_state_dofs * sizeof(double));
    audit.newton_workspace_mib =
        audit.vector_mib * static_cast<double>(input.workspace_vector_count);
    audit.sparse_matrix_mib =
        bytes_to_mib(static_cast<std::size_t>(
            static_cast<double>(audit.estimated_sparse_nonzeros) *
            (sizeof(double) + sizeof(int)) * input.sparse_matrix_safety_factor));
    audit.direct_factorization_risk_mib =
        audit.sparse_matrix_mib * input.direct_factor_fill_multiplier;
    audit.material_state_mib = bytes_to_mib(
        audit.host_material_point_count *
        material_state_bytes_per_point(input.constitutive_cost));
    audit.estimated_hot_state_mib =
        audit.newton_workspace_mib + audit.sparse_matrix_mib +
        audit.material_state_mib;
    audit.solver_advice = classify_reduced_rc_local_solver_scaling(
        audit.estimated_total_state_dofs,
        audit.direct_factorization_risk_mib,
        audit.host_material_point_count);

    const bool mixed_active_set_layout =
        input.shifted_heaviside_xfem ||
        (input.bars_have_independent_nodes &&
         input.longitudinal_bar_count > std::size_t{0}) ||
        input.crack_crossing_bridge_enabled;
    const bool elastic_host_only_candidate =
        !mixed_active_set_layout &&
        input.constitutive_cost ==
            ReducedRCLocalConstitutiveCostKind::elastic_proxy;

    audit.seed_state_cache_recommended =
        audit.host_material_point_count >= 1000 ||
        input.constitutive_cost ==
            ReducedRCLocalConstitutiveCostKind::ko_bathe_heavy_reference;
    audit.newton_warm_start_recommended =
        audit.estimated_total_state_dofs >= 1500 ||
        audit.host_material_point_count >= 1000;
    audit.site_level_openmp_recommended =
        audit.estimated_total_state_dofs >= 1500 ||
        audit.host_material_point_count >= 1000;
    audit.global_petsc_assembly_openmp_recommended = false;
    audit.symmetric_matrix_storage_recommended = false;
    audit.symmetric_matrix_storage_requires_tangent_audit =
        audit.estimated_total_state_dofs > 0;
    audit.block_matrix_storage_candidate =
        elastic_host_only_candidate &&
        input.topology == ReducedRCLocalCellTopologyKind::hex8_lagrange;
    audit.field_split_or_asm_preconditioner_recommended =
        audit.solver_advice !=
        ReducedRCLocalSolverScalingAdviceKind::direct_lu_smoke_ok;
    audit.plain_gmres_ilu_rejected_for_enriched_branch =
        mixed_active_set_layout;

    return audit;
}

} // namespace fall_n

#endif // FALL_N_REDUCED_RC_LOCAL_MESH_SCALE_AUDIT_HH
