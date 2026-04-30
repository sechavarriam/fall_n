#include <cmath>
#include <iostream>

#include "src/validation/ReducedRCLocalMeshScaleAudit.hh"

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

constexpr bool baseline_xfem_counts_match_the_current_reference_mesh()
{
    constexpr auto audit =
        fall_n::make_reduced_rc_local_mesh_scale_audit({
            .nx = 1,
            .ny = 1,
            .nz = 4,
            .topology = fall_n::ReducedRCLocalCellTopologyKind::hex8_lagrange,
            .constitutive_cost =
                fall_n::ReducedRCLocalConstitutiveCostKind::
                    cyclic_crack_band_xfem,
            .shifted_heaviside_xfem = true,
            .planar_crack_count = 1,
            .longitudinal_bar_count = 8,
            .bars_have_independent_nodes = true,
            .rebar_subsegments_per_host_element = 1,
            .crack_crossing_bridge_enabled = true});

    return audit.host_element_count == 4 &&
           audit.host_node_count == 20 &&
           audit.host_material_point_count == 32 &&
           audit.enriched_node_count == 8 &&
           audit.host_displacement_dofs == 60 &&
           audit.enrichment_dofs == 24 &&
           audit.rebar_node_count == 40 &&
           audit.rebar_dofs == 120 &&
           audit.crack_crossing_bridge_element_count == 8 &&
           audit.estimated_total_state_dofs == 204 &&
           audit.solver_advice ==
               fall_n::ReducedRCLocalSolverScalingAdviceKind::
                   direct_lu_smoke_ok &&
           !audit.global_petsc_assembly_openmp_recommended;
}

constexpr bool seven_by_seven_by_twenty_five_xfem_is_a_reference_lu_case()
{
    constexpr auto audit =
        fall_n::make_reduced_rc_local_mesh_scale_audit({
            .nx = 7,
            .ny = 7,
            .nz = 25,
            .topology = fall_n::ReducedRCLocalCellTopologyKind::hex8_lagrange,
            .constitutive_cost =
                fall_n::ReducedRCLocalConstitutiveCostKind::
                    cyclic_crack_band_xfem,
            .shifted_heaviside_xfem = true,
            .planar_crack_count = 1,
            .longitudinal_bar_count = 8,
            .bars_have_independent_nodes = true,
            .rebar_subsegments_per_host_element = 1,
            .crack_crossing_bridge_enabled = true});

    return audit.host_element_count == 1225 &&
           audit.host_node_count == 1664 &&
           audit.host_material_point_count == 9800 &&
           audit.enriched_node_count == 128 &&
           audit.host_displacement_dofs == 4992 &&
           audit.enrichment_dofs == 384 &&
           audit.rebar_node_count == 208 &&
           audit.rebar_dofs == 624 &&
           audit.estimated_total_state_dofs == 6000 &&
           audit.estimated_sparse_nonzeros == 1116000 &&
           audit.seed_state_cache_recommended &&
           audit.newton_warm_start_recommended &&
           audit.site_level_openmp_recommended &&
           !audit.symmetric_matrix_storage_recommended &&
           audit.symmetric_matrix_storage_requires_tangent_audit &&
           !audit.block_matrix_storage_candidate &&
           audit.field_split_or_asm_preconditioner_recommended &&
           audit.plain_gmres_ilu_rejected_for_enriched_branch &&
           audit.solver_advice ==
               fall_n::ReducedRCLocalSolverScalingAdviceKind::
                   direct_lu_reference_only;
}

constexpr bool quadratic_xfem_pushes_us_to_domain_decomposition()
{
    constexpr auto audit =
        fall_n::make_reduced_rc_local_mesh_scale_audit({
            .nx = 7,
            .ny = 7,
            .nz = 25,
            .topology = fall_n::ReducedRCLocalCellTopologyKind::hex27_lagrange,
            .constitutive_cost =
                fall_n::ReducedRCLocalConstitutiveCostKind::
                    cyclic_crack_band_xfem,
            .shifted_heaviside_xfem = true,
            .planar_crack_count = 1,
            .longitudinal_bar_count = 8,
            .bars_have_independent_nodes = true,
            .rebar_subsegments_per_host_element = 1,
            .crack_crossing_bridge_enabled = true});

    return audit.host_node_count == 11475 &&
           audit.host_material_point_count == 33075 &&
           audit.enriched_node_count == 675 &&
           audit.estimated_total_state_dofs == 37074 &&
           audit.direct_factorization_risk_mib > 10000.0 &&
           audit.field_split_or_asm_preconditioner_recommended &&
           audit.solver_advice ==
               fall_n::ReducedRCLocalSolverScalingAdviceKind::
                   domain_decomposition_or_multiscale_batch_required;
}

bool ko_bathe_material_state_is_not_hidden_by_the_elastic_proxy()
{
    const auto elastic =
        fall_n::make_reduced_rc_local_mesh_scale_audit({
            .nx = 7,
            .ny = 7,
            .nz = 25,
            .topology = fall_n::ReducedRCLocalCellTopologyKind::hex8_lagrange,
            .constitutive_cost =
                fall_n::ReducedRCLocalConstitutiveCostKind::elastic_proxy,
            .shifted_heaviside_xfem = false,
            .longitudinal_bar_count = 8});
    const auto ko_bathe =
        fall_n::make_reduced_rc_local_mesh_scale_audit({
            .nx = 7,
            .ny = 7,
            .nz = 25,
            .topology = fall_n::ReducedRCLocalCellTopologyKind::hex8_lagrange,
            .constitutive_cost =
                fall_n::ReducedRCLocalConstitutiveCostKind::
                    ko_bathe_heavy_reference,
            .shifted_heaviside_xfem = false,
            .longitudinal_bar_count = 8});

    return ko_bathe.host_material_point_count ==
               elastic.host_material_point_count &&
           ko_bathe.material_state_mib > 15.0 * elastic.material_state_mib &&
           ko_bathe.seed_state_cache_recommended &&
           !ko_bathe.symmetric_matrix_storage_recommended;
}

static_assert(baseline_xfem_counts_match_the_current_reference_mesh());
static_assert(seven_by_seven_by_twenty_five_xfem_is_a_reference_lu_case());
static_assert(quadratic_xfem_pushes_us_to_domain_decomposition());

} // namespace

int main()
{
    std::cout << "=== Reduced RC Local Mesh Scale Audit Tests ===\n";

    report("baseline_xfem_counts_match_the_current_reference_mesh",
           baseline_xfem_counts_match_the_current_reference_mesh());
    report("seven_by_seven_by_twenty_five_xfem_is_a_reference_lu_case",
           seven_by_seven_by_twenty_five_xfem_is_a_reference_lu_case());
    report("quadratic_xfem_pushes_us_to_domain_decomposition",
           quadratic_xfem_pushes_us_to_domain_decomposition());
    report("ko_bathe_material_state_is_not_hidden_by_the_elastic_proxy",
           ko_bathe_material_state_is_not_hidden_by_the_elastic_proxy());

    const auto audit = fall_n::make_reduced_rc_local_mesh_scale_audit({
        .nx = 7,
        .ny = 7,
        .nz = 25,
        .topology = fall_n::ReducedRCLocalCellTopologyKind::hex8_lagrange,
        .constitutive_cost =
            fall_n::ReducedRCLocalConstitutiveCostKind::
                cyclic_crack_band_xfem,
        .shifted_heaviside_xfem = true,
        .longitudinal_bar_count = 8});
    std::cout << "  7x7x25 Hex8 XFEM estimate: dofs="
              << audit.estimated_total_state_dofs
              << ", material_points=" << audit.host_material_point_count
              << ", matrixMiB=" << audit.sparse_matrix_mib
              << ", directFactorRiskMiB="
              << audit.direct_factorization_risk_mib
              << ", advice=" << fall_n::to_string(audit.solver_advice)
              << ", fieldSplitOrAsm="
              << (audit.field_split_or_asm_preconditioner_recommended ? "yes"
                                                                       : "no")
              << "\n";

    std::cout << "\nSummary: " << passed << " passed, " << failed
              << " failed.\n";
    return failed == 0 ? 0 : 1;
}
