#include "src/validation/ReducedRCColumnContinuumBaseline.hh"
#include "src/analysis/PenaltyCoupling.hh"
#include "src/elements/TrussElement.hh"
#include "src/elements/ContinuumElement.hh"
#include "src/elements/FEM_Element.hh"
#include "src/materials/Material.hh"
#include "src/materials/MaterialPolicy.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/update_strategy/IntegrationStrategy.hh"
#include "src/materials/constitutive_models/non_lineal/MenegottoPintoSteel.hh"
#include "src/materials/RCSectionLayout.hh"
#include "src/model/PrismaticDomainBuilder.hh"
#include "src/model/Model.hh"
#include "src/petsc/PetscRaii.hh"

#include <petsc.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

using namespace fall_n;
using namespace fall_n::validation_reboot;

int passed = 0;
int total = 0;

void check(bool condition, const char* message)
{
    ++total;
    if (condition) {
        ++passed;
        std::cout << "  [PASS] " << message << '\n';
    } else {
        std::cout << "  [FAIL] " << message << '\n';
    }
}

Material<UniaxialMaterial> make_elastic_steel_for_embedding()
{
    constexpr double E_STEEL = 200000.0;
    constexpr double FY_HIGH = 1.0e12;
    constexpr double B_HARD = 0.01;
    InelasticMaterial<MenegottoPintoSteel> inst{E_STEEL, FY_HIGH, B_HARD};
    return Material<UniaxialMaterial>{std::move(inst), InelasticUpdate{}};
}

RebarSpec make_structural_matched_eight_bar_rebar(
    const ReducedRCColumnReferenceSpec& spec)
{
    const auto rc_spec = to_rc_column_section_spec(spec);
    const auto area = rc_column_longitudinal_bar_area(rc_spec);
    const auto positions = rc_column_longitudinal_bar_positions(rc_spec);

    RebarSpec rebar{};
    rebar.bars.reserve(positions.size());
    for (const auto& [y, z] : positions) {
        rebar.bars.push_back({
            .ly = y,
            .lz = z,
            .area = area,
            .diameter = spec.longitudinal_bar_diameter_m,
            .group = "Rebar",
        });
    }
    return rebar;
}

std::vector<std::size_t> active_face_nodes(
    const Domain<3>& domain,
    const std::vector<PetscInt>& raw_face)
{
    std::vector<std::size_t> nodes;
    nodes.reserve(raw_face.size());
    for (const auto raw_node_id : raw_face) {
        const auto node_id = static_cast<std::size_t>(raw_node_id);
        if (domain.node(node_id).num_dof() == 0) {
            continue;
        }
        nodes.push_back(node_id);
    }
    return nodes;
}

std::vector<std::size_t> top_rebar_node_ids(
    const ReinforcedDomainResult& reinforced)
{
    const auto rebar_nodes_per_bar =
        static_cast<std::size_t>(reinforced.grid.step * reinforced.grid.nz + 1);
    std::vector<std::size_t> nodes;
    nodes.reserve(reinforced.bar_diameters.size());
    for (std::size_t bar = 0; bar < reinforced.bar_diameters.size(); ++bar) {
        nodes.push_back(static_cast<std::size_t>(
            reinforced.embeddings[bar * rebar_nodes_per_bar +
                                  rebar_nodes_per_bar - 1]
                .rebar_node_id));
    }
    return nodes;
}

std::vector<std::size_t> base_rebar_node_ids(
    const ReinforcedDomainResult& reinforced)
{
    const auto rebar_nodes_per_bar =
        static_cast<std::size_t>(reinforced.grid.step * reinforced.grid.nz + 1);
    std::vector<std::size_t> nodes;
    nodes.reserve(reinforced.bar_diameters.size());
    for (std::size_t bar = 0; bar < reinforced.bar_diameters.size(); ++bar) {
        nodes.push_back(static_cast<std::size_t>(
            reinforced.embeddings[bar * rebar_nodes_per_bar].rebar_node_id));
    }
    return nodes;
}

template <typename ModelT>
void apply_axial_preload(
    ModelT& model,
    const ReducedRCColumnReferenceSpec& spec,
    const RebarSpec& rebar,
    const std::vector<std::size_t>& top_rebar_nodes,
    ReducedRCColumnContinuumAxialPreloadTransferMode transfer_mode)
{
    constexpr auto kTopFaceLoadGroup = "ReducedRCContinuumTopFaceLoad";
    const double gross_area = spec.section_b_m * spec.section_h_m;
    const double total_force_mn = 0.02;

    double host_force_mn = total_force_mn;
    if (transfer_mode ==
            ReducedRCColumnContinuumAxialPreloadTransferMode::
                composite_section_force_split &&
        !top_rebar_nodes.empty()) {
        double total_rebar_area = 0.0;
        for (const auto& bar : rebar.bars) {
            total_rebar_area += bar.area;
        }

        const double rebar_force_mn =
            total_force_mn * std::clamp(total_rebar_area / gross_area, 0.0, 1.0);
        host_force_mn = std::max(total_force_mn - rebar_force_mn, 0.0);

        if (total_rebar_area > 0.0) {
            for (std::size_t bar_index = 0;
                 bar_index < top_rebar_nodes.size() &&
                 bar_index < rebar.bars.size();
                 ++bar_index) {
                const double bar_share =
                    rebar_force_mn * (rebar.bars[bar_index].area / total_rebar_area);
                model.apply_node_force(top_rebar_nodes[bar_index], 0.0, 0.0, -bar_share);
            }
        }
    }

    model.apply_surface_traction(
        kTopFaceLoadGroup,
        0.0,
        0.0,
        -host_force_mn / gross_area);
}

[[maybe_unused]] void assemble_penalty_coupling_force_global_mirror(
    const Domain<3>& domain,
    const PrismaticGrid& grid,
    const std::vector<RebarNodeEmbedding>& embeddings,
    std::size_t num_bars,
    double alpha,
    bool skip_minz_maxz,
    HexOrder order,
    Vec solution,
    Vec force_global)
{
    PetscSection g_sec = nullptr;
    DM dm = domain.mesh.dm;
    FALL_N_PETSC_CHECK(DMGetGlobalSection(dm, &g_sec));

    const int step = grid.step;
    const int nz = grid.nz;
    const std::size_t rebar_nodes_per_bar =
        static_cast<std::size_t>(step * nz + 1);

    FALL_N_PETSC_CHECK(VecSet(force_global, 0.0));

    for (std::size_t b = 0; b < num_bars; ++b) {
        for (std::size_t iz = 0; iz < rebar_nodes_per_bar; ++iz) {
            if (skip_minz_maxz && (iz == 0 || iz + 1 == rebar_nodes_per_bar)) {
                continue;
            }

            const auto& emb = embeddings[b * rebar_nodes_per_bar + iz];
            const auto weights =
                penalty_coupling_host_weights(domain, grid, emb, order);

            PetscInt r_dof = 0;
            PetscSectionGetDof(g_sec, domain.node(
                                          static_cast<std::size_t>(
                                              emb.rebar_node_id))
                                          .sieve_id.value(),
                               &r_dof);
            if (r_dof <= 0) {
                continue;
            }

            PetscInt r_off = 0;
            PetscSectionGetOffset(
                g_sec,
                domain.node(static_cast<std::size_t>(emb.rebar_node_id))
                    .sieve_id.value(),
                &r_off);

            double gap[3] = {0.0, 0.0, 0.0};
            for (int d = 0; d < 3; ++d) {
                const PetscInt ridx = r_off + d;
                PetscScalar u_r = 0.0;
                FALL_N_PETSC_CHECK(VecGetValues(solution, 1, &ridx, &u_r));
                gap[d] = static_cast<double>(u_r);
            }

            for (const auto& [host_sieve, Ni] : weights) {
                PetscInt h_dof = 0;
                PetscSectionGetDof(g_sec, host_sieve, &h_dof);
                if (h_dof <= 0) {
                    continue;
                }

                PetscInt h_off = 0;
                PetscSectionGetOffset(g_sec, host_sieve, &h_off);
                for (int d = 0; d < 3; ++d) {
                    const PetscInt hidx = h_off + d;
                    PetscScalar u_h = 0.0;
                    FALL_N_PETSC_CHECK(VecGetValues(solution, 1, &hidx, &u_h));
                    gap[d] -= Ni * static_cast<double>(u_h);
                }
            }

            for (int d = 0; d < 3; ++d) {
                const PetscInt ridx = r_off + d;
                const PetscScalar v_r = alpha * gap[d];
                FALL_N_PETSC_CHECK(
                    VecSetValues(force_global, 1, &ridx, &v_r, ADD_VALUES));
            }

            for (const auto& [host_sieve, Ni] : weights) {
                PetscInt h_dof = 0;
                PetscSectionGetDof(g_sec, host_sieve, &h_dof);
                if (h_dof <= 0) {
                    continue;
                }

                PetscInt h_off = 0;
                PetscSectionGetOffset(g_sec, host_sieve, &h_off);
                for (int d = 0; d < 3; ++d) {
                    const PetscInt hidx = h_off + d;
                    const PetscScalar v_h = -alpha * Ni * gap[d];
                    FALL_N_PETSC_CHECK(
                        VecSetValues(force_global, 1, &hidx, &v_h, ADD_VALUES));
                }
            }
        }
    }

    FALL_N_PETSC_CHECK(VecAssemblyBegin(force_global));
    FALL_N_PETSC_CHECK(VecAssemblyEnd(force_global));
}

struct LinearizedPreloadSolveSummary {
    bool ksp_converged{false};
    double rhs_norm{0.0};
    double algebraic_residual_norm{0.0};
    double relative_algebraic_residual_norm{0.0};
    double element_consistency_norm{0.0};
    double relative_element_consistency_norm{0.0};
    double coupling_consistency_norm{0.0};
    double relative_coupling_consistency_norm{0.0};
    double coupling_mirror_consistency_norm{0.0};
    double relative_coupling_mirror_consistency_norm{0.0};
    double residual_norm{0.0};
    double relative_residual_norm{0.0};
    std::size_t coupling_count{0};
};

LinearizedPreloadSolveSummary solve_linearized_preload_slice(
    HexOrder hex_order,
    RebarLineInterpolation interpolation,
    ReducedRCColumnContinuumAxialPreloadTransferMode transfer_mode)
{
    using ContinuumElemT =
        ContinuumElement<ThreeDimensionalMaterial, 3, continuum::SmallStrain>;
    using ModelT =
        Model<ThreeDimensionalMaterial, continuum::SmallStrain, 3, MultiElementPolicy>;

    const ReducedRCColumnReferenceSpec spec{};
    const auto rebar = make_structural_matched_eight_bar_rebar(spec);

    PrismaticSpec prism{
        .width = spec.section_b_m,
        .height = spec.section_h_m,
        .length = spec.column_height_m,
        .nx = 2,
        .ny = 2,
        .nz = 2,
        .hex_order = hex_order,
        .longitudinal_bias_power = 3.0,
        .physical_group = "Concrete",
    };

    auto reinforced =
        make_reinforced_prismatic_domain(prism, rebar, interpolation);
    auto& domain = reinforced.domain;

    const double ec_mpa = 4700.0 * std::sqrt(spec.concrete_fpc_mpa);
    auto concrete_material = Material<ThreeDimensionalMaterial>{
        ContinuumIsotropicElasticMaterial{ec_mpa, spec.concrete_nu},
        ElasticUpdate{}};
    auto rebar_material = Material<UniaxialMaterial>{
        UniaxialIsotropicElasticMaterial{spec.steel_E_mpa},
        ElasticUpdate{}};

    std::vector<FEM_Element> elements;
    elements.reserve(domain.num_elements());
    for (std::size_t element_index = 0;
         element_index < reinforced.rebar_range.first;
         ++element_index) {
        elements.emplace_back(
            ContinuumElemT{&domain.element(element_index), concrete_material});
    }
    for (std::size_t element_index = reinforced.rebar_range.first;
         element_index < reinforced.rebar_range.last;
         ++element_index) {
        const auto bar_index =
            (element_index - reinforced.rebar_range.first) /
            static_cast<std::size_t>(reinforced.grid.nz);
        if (reinforced.rebar_line_num_nodes == 3) {
            elements.emplace_back(TrussElement<3, 3>{
                &domain.element(element_index),
                rebar_material,
                rebar.bars.at(bar_index).area});
        } else {
            elements.emplace_back(TrussElement<3, 2>{
                &domain.element(element_index),
                rebar_material,
                rebar.bars.at(bar_index).area});
        }
    }

    ModelT model{domain, std::move(elements)};

    const auto base_face_nodes =
        active_face_nodes(domain, reinforced.grid.nodes_on_face(PrismFace::MinZ));
    const auto top_face_nodes =
        active_face_nodes(domain, reinforced.grid.nodes_on_face(PrismFace::MaxZ));
    const auto base_rebars = base_rebar_node_ids(reinforced);
    const auto top_rebars = top_rebar_node_ids(reinforced);

    for (const auto node_id : base_face_nodes) {
        model.constrain_node(node_id, {0.0, 0.0, 0.0});
    }
    for (const auto node_id : top_face_nodes) {
        model.constrain_dof(node_id, 0, 0.0);
    }
    for (const auto node_id : base_rebars) {
        model.constrain_node(node_id, {0.0, 0.0, 0.0});
    }
    for (const auto node_id : top_rebars) {
        model.constrain_dof(node_id, 0, 0.0);
    }

    domain.create_boundary_from_plane(
        "ReducedRCContinuumTopFaceLoad",
        2,
        spec.column_height_m,
        1.0e-9,
        0,
        reinforced.rebar_range.first);

    model.setup();

    apply_axial_preload(model, spec, rebar, top_rebars, transfer_mode);

    PenaltyCoupling coupling;
    coupling.setup(
        domain,
        reinforced.grid,
        reinforced.embeddings,
        rebar.bars.size(),
        1.0e4 * ec_mpa,
        false,
        hex_order);

    DM dm = model.get_plex();
    petsc::OwnedVec U{};
    petsc::OwnedVec rhs{};
    petsc::OwnedVec residual{};
    petsc::OwnedVec algebraic_residual{};
    petsc::OwnedVec element_force{};
    petsc::OwnedVec coupling_force{};
    petsc::OwnedVec coupling_force_mirror{};
    petsc::OwnedVec element_consistency{};
    petsc::OwnedVec coupling_consistency{};
    petsc::OwnedVec coupling_mirror_consistency{};
    petsc::OwnedVec u_local{};
    petsc::OwnedVec f_int_local{};
    petsc::OwnedVec f_cpl_local{};
    petsc::OwnedVec solution{};
    petsc::OwnedMat K{};
    petsc::OwnedMat K_element{};
    petsc::OwnedMat K_coupling{};
    petsc::OwnedKSP ksp{};

    FALL_N_PETSC_CHECK(DMCreateGlobalVector(dm, U.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), rhs.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), residual.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), algebraic_residual.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), element_force.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), coupling_force.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), coupling_force_mirror.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), element_consistency.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), coupling_consistency.ptr()));
    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), coupling_mirror_consistency.ptr()));
    FALL_N_PETSC_CHECK(DMGetLocalVector(dm, u_local.ptr()));
    FALL_N_PETSC_CHECK(DMGetLocalVector(dm, f_int_local.ptr()));
    FALL_N_PETSC_CHECK(DMGetLocalVector(dm, f_cpl_local.ptr()));
    FALL_N_PETSC_CHECK(DMCreateMatrix(dm, K.ptr()));
    FALL_N_PETSC_CHECK(DMCreateMatrix(dm, K_element.ptr()));
    FALL_N_PETSC_CHECK(DMCreateMatrix(dm, K_coupling.ptr()));
    FALL_N_PETSC_CHECK(MatSetOption(
        K.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    FALL_N_PETSC_CHECK(MatSetOption(
        K_element.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    FALL_N_PETSC_CHECK(MatSetOption(
        K_coupling.get(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    FALL_N_PETSC_CHECK(KSPCreate(PETSC_COMM_WORLD, ksp.ptr()));

    FALL_N_PETSC_CHECK(VecSet(U.get(), 0.0));
    FALL_N_PETSC_CHECK(VecSet(rhs.get(), 0.0));
    FALL_N_PETSC_CHECK(
        DMLocalToGlobal(dm, model.force_vector(), ADD_VALUES, rhs.get()));

    FALL_N_PETSC_CHECK(VecSet(u_local.get(), 0.0));
    FALL_N_PETSC_CHECK(DMGlobalToLocal(dm, U.get(), INSERT_VALUES, u_local.get()));
    FALL_N_PETSC_CHECK(VecAXPY(u_local.get(), 1.0, model.imposed_solution()));

    FALL_N_PETSC_CHECK(MatZeroEntries(K.get()));
    FALL_N_PETSC_CHECK(MatZeroEntries(K_element.get()));
    FALL_N_PETSC_CHECK(MatZeroEntries(K_coupling.get()));
    for (auto& element : model.elements()) {
        element.inject_tangent_stiffness(u_local.get(), K.get());
        element.inject_tangent_stiffness(u_local.get(), K_element.get());
    }
    coupling.add_to_jacobian(u_local.get(), K.get(), dm);
    coupling.add_to_jacobian(u_local.get(), K_coupling.get(), dm);
    FALL_N_PETSC_CHECK(MatAssemblyBegin(K.get(), MAT_FINAL_ASSEMBLY));
    FALL_N_PETSC_CHECK(MatAssemblyEnd(K.get(), MAT_FINAL_ASSEMBLY));
    FALL_N_PETSC_CHECK(MatAssemblyBegin(K_element.get(), MAT_FINAL_ASSEMBLY));
    FALL_N_PETSC_CHECK(MatAssemblyEnd(K_element.get(), MAT_FINAL_ASSEMBLY));
    FALL_N_PETSC_CHECK(MatAssemblyBegin(K_coupling.get(), MAT_FINAL_ASSEMBLY));
    FALL_N_PETSC_CHECK(MatAssemblyEnd(K_coupling.get(), MAT_FINAL_ASSEMBLY));

    FALL_N_PETSC_CHECK(KSPSetOperators(ksp.get(), K.get(), K.get()));
    FALL_N_PETSC_CHECK(KSPSetType(ksp.get(), KSPPREONLY));
    PC pc = nullptr;
    FALL_N_PETSC_CHECK(KSPGetPC(ksp.get(), &pc));
    FALL_N_PETSC_CHECK(PCSetType(pc, PCLU));
    FALL_N_PETSC_CHECK(KSPSetUp(ksp.get()));

    FALL_N_PETSC_CHECK(VecDuplicate(U.get(), solution.ptr()));
    FALL_N_PETSC_CHECK(VecSet(solution.get(), 0.0));
    FALL_N_PETSC_CHECK(KSPSolve(ksp.get(), rhs.get(), solution.get()));

    FALL_N_PETSC_CHECK(VecSet(algebraic_residual.get(), 0.0));
    FALL_N_PETSC_CHECK(MatMult(K.get(), solution.get(), algebraic_residual.get()));
    FALL_N_PETSC_CHECK(VecAXPY(algebraic_residual.get(), -1.0, rhs.get()));
    FALL_N_PETSC_CHECK(VecSet(element_consistency.get(), 0.0));
    FALL_N_PETSC_CHECK(MatMult(
        K_element.get(), solution.get(), element_consistency.get()));
    FALL_N_PETSC_CHECK(VecSet(coupling_consistency.get(), 0.0));
    FALL_N_PETSC_CHECK(MatMult(
        K_coupling.get(), solution.get(), coupling_consistency.get()));
    FALL_N_PETSC_CHECK(VecSet(coupling_mirror_consistency.get(), 0.0));

    FALL_N_PETSC_CHECK(VecSet(u_local.get(), 0.0));
    FALL_N_PETSC_CHECK(
        DMGlobalToLocal(dm, solution.get(), INSERT_VALUES, u_local.get()));
    FALL_N_PETSC_CHECK(VecAXPY(u_local.get(), 1.0, model.imposed_solution()));

    FALL_N_PETSC_CHECK(VecSet(f_int_local.get(), 0.0));
    for (auto& element : model.elements()) {
        element.compute_internal_forces(u_local.get(), f_int_local.get());
    }
    FALL_N_PETSC_CHECK(VecSet(f_cpl_local.get(), 0.0));
    coupling.add_to_residual(u_local.get(), f_cpl_local.get(), dm);

    FALL_N_PETSC_CHECK(VecSet(element_force.get(), 0.0));
    FALL_N_PETSC_CHECK(
        DMLocalToGlobal(dm, f_int_local.get(), ADD_VALUES, element_force.get()));
    FALL_N_PETSC_CHECK(VecSet(coupling_force.get(), 0.0));
    FALL_N_PETSC_CHECK(
        DMLocalToGlobal(dm, f_cpl_local.get(), ADD_VALUES, coupling_force.get()));
    FALL_N_PETSC_CHECK(VecSet(coupling_force_mirror.get(), 0.0));
    coupling.add_to_global_residual(
        u_local.get(), coupling_force_mirror.get(), dm);
    FALL_N_PETSC_CHECK(VecAssemblyBegin(coupling_force_mirror.get()));
    FALL_N_PETSC_CHECK(VecAssemblyEnd(coupling_force_mirror.get()));

    FALL_N_PETSC_CHECK(VecAXPY(element_consistency.get(), -1.0, element_force.get()));
    FALL_N_PETSC_CHECK(
        VecCopy(coupling_consistency.get(), coupling_mirror_consistency.get()));
    FALL_N_PETSC_CHECK(VecAXPY(
        coupling_mirror_consistency.get(), -1.0, coupling_force_mirror.get()));
    FALL_N_PETSC_CHECK(
        VecAXPY(coupling_consistency.get(), -1.0, coupling_force.get()));

    FALL_N_PETSC_CHECK(VecSet(residual.get(), 0.0));
    FALL_N_PETSC_CHECK(VecAXPY(residual.get(), 1.0, element_force.get()));
    FALL_N_PETSC_CHECK(VecAXPY(residual.get(), 1.0, coupling_force_mirror.get()));
    FALL_N_PETSC_CHECK(
        VecAXPY(residual.get(), -1.0, rhs.get()));

    PetscReal rhs_norm = 0.0;
    PetscReal algebraic_residual_norm = 0.0;
    PetscReal element_force_norm = 0.0;
    PetscReal coupling_force_norm = 0.0;
    PetscReal element_consistency_norm = 0.0;
    PetscReal coupling_consistency_norm = 0.0;
    PetscReal coupling_force_mirror_norm = 0.0;
    PetscReal coupling_mirror_consistency_norm = 0.0;
    PetscReal residual_norm = 0.0;
    FALL_N_PETSC_CHECK(VecNorm(rhs.get(), NORM_2, &rhs_norm));
    FALL_N_PETSC_CHECK(
        VecNorm(algebraic_residual.get(), NORM_2, &algebraic_residual_norm));
    FALL_N_PETSC_CHECK(VecNorm(element_force.get(), NORM_2, &element_force_norm));
    FALL_N_PETSC_CHECK(VecNorm(coupling_force.get(), NORM_2, &coupling_force_norm));
    FALL_N_PETSC_CHECK(
        VecNorm(coupling_force_mirror.get(), NORM_2, &coupling_force_mirror_norm));
    FALL_N_PETSC_CHECK(
        VecNorm(element_consistency.get(), NORM_2, &element_consistency_norm));
    FALL_N_PETSC_CHECK(
        VecNorm(coupling_consistency.get(), NORM_2, &coupling_consistency_norm));
    FALL_N_PETSC_CHECK(VecNorm(
        coupling_mirror_consistency.get(),
        NORM_2,
        &coupling_mirror_consistency_norm));
    FALL_N_PETSC_CHECK(VecNorm(residual.get(), NORM_2, &residual_norm));
    KSPConvergedReason reason{};
    FALL_N_PETSC_CHECK(KSPGetConvergedReason(ksp.get(), &reason));

    FALL_N_PETSC_CHECK(DMRestoreLocalVector(dm, u_local.ptr()));
    FALL_N_PETSC_CHECK(DMRestoreLocalVector(dm, f_int_local.ptr()));
    FALL_N_PETSC_CHECK(DMRestoreLocalVector(dm, f_cpl_local.ptr()));

    return {
        .ksp_converged = reason > 0,
        .rhs_norm = static_cast<double>(rhs_norm),
        .algebraic_residual_norm = static_cast<double>(algebraic_residual_norm),
        .relative_algebraic_residual_norm =
            rhs_norm > 1.0e-16
                ? static_cast<double>(algebraic_residual_norm / rhs_norm)
                : static_cast<double>(algebraic_residual_norm),
        .element_consistency_norm = static_cast<double>(element_consistency_norm),
        .relative_element_consistency_norm =
            element_force_norm > 1.0e-16
                ? static_cast<double>(element_consistency_norm / element_force_norm)
                : static_cast<double>(element_consistency_norm),
        .coupling_consistency_norm = static_cast<double>(coupling_consistency_norm),
        .relative_coupling_consistency_norm =
            coupling_force_norm > 1.0e-16
                ? static_cast<double>(coupling_consistency_norm / coupling_force_norm)
                : static_cast<double>(coupling_consistency_norm),
        .coupling_mirror_consistency_norm =
            static_cast<double>(coupling_mirror_consistency_norm),
        .relative_coupling_mirror_consistency_norm =
            coupling_force_mirror_norm > 1.0e-16
                ? static_cast<double>(
                      coupling_mirror_consistency_norm / coupling_force_mirror_norm)
                : static_cast<double>(coupling_mirror_consistency_norm),
        .residual_norm = static_cast<double>(residual_norm),
        .relative_residual_norm =
            rhs_norm > 1.0e-16 ? static_cast<double>(residual_norm / rhs_norm)
                               : static_cast<double>(residual_norm),
        .coupling_count = coupling.num_couplings(),
    };
}

} // namespace

int main()
{
    int argc = 0;
    char** argv = nullptr;
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    using namespace fall_n;
    using namespace fall_n::validation_reboot;
    using namespace fall_n::table_cyclic_validation;

    std::cout << "Reduced RC continuum baseline smoke\n";

    {
        const auto structural_area =
            describe_reduced_rc_column_structural_steel_area();
        const ReducedRCColumnContinuumRunSpec structural_matched_spec{
            .reinforcement_mode =
                ReducedRCColumnContinuumReinforcementMode::
                    embedded_longitudinal_bars,
            .rebar_layout_mode =
                ReducedRCColumnContinuumRebarLayoutMode::
                    structural_matched_eight_bar,
        };
        const auto structural_matched_area =
            describe_reduced_rc_column_continuum_rebar_area(
                structural_matched_spec);

        check(
            structural_matched_area.bar_count ==
                structural_area.longitudinal_bar_count,
            "continuum structural-matched branch uses the structural bar count");
        check(
            std::abs(structural_matched_area.total_rebar_area_m2 -
                     structural_area.total_longitudinal_steel_area_m2) <
                1.0e-14,
            "continuum structural-matched branch preserves the structural steel area");
        check(
            structural_matched_area.area_equivalent_to_structural_baseline,
            "continuum structural-matched branch declares area equivalence");

        const ReducedRCColumnContinuumRunSpec promoted_default_spec{};
        check(
            promoted_default_spec.axial_preload_transfer_mode ==
                ReducedRCColumnContinuumAxialPreloadTransferMode::
                    composite_section_force_split,
            "continuum baseline defaults to composite axial preload split");

        const ReducedRCColumnContinuumRunSpec enriched_spec{
            .reinforcement_mode =
                ReducedRCColumnContinuumReinforcementMode::
                    embedded_longitudinal_bars,
            .rebar_layout_mode =
                ReducedRCColumnContinuumRebarLayoutMode::enriched_twelve_bar,
        };
        const auto enriched_area =
            describe_reduced_rc_column_continuum_rebar_area(enriched_spec);

        check(enriched_area.bar_count == 12,
              "continuum enriched branch declares twelve longitudinal bars");
        check(
            !enriched_area.area_equivalent_to_structural_baseline,
            "continuum enriched branch does not silently masquerade as the structural area");
    }

    {
        ReducedRCColumnContinuumRunSpec transverse_extension_spec{};
        transverse_extension_spec.material_mode =
            ReducedRCColumnContinuumMaterialMode::nonlinear;
        transverse_extension_spec.transverse_reinforcement_mode =
            ReducedRCColumnContinuumTransverseReinforcementMode::
                embedded_stirrup_loops;
        transverse_extension_spec.transverse_reinforcement_area_scale = 0.25;
        const auto taxonomy =
            describe_reduced_rc_column_continuum_local_model(
                transverse_extension_spec);
        const auto transverse_summary =
            describe_reduced_rc_column_continuum_transverse_reinforcement(
                transverse_extension_spec);

        check(taxonomy.maturity_kind ==
                  LocalModelMaturityKind::future_extension,
              "embedded stirrup loops are classified as a future-extension branch");
        check(!taxonomy.suitable_for_future_multiscale_local_model,
              "embedded stirrup loops are not promoted before bond-slip/localization validation");
        check(std::abs(transverse_summary.area_scale - 0.25) < 1.0e-14,
              "transverse reinforcement summary preserves the declared area scale");
        check(transverse_summary.volumetric_ratio <
                  transverse_extension_spec.reference_spec.rho_s,
              "transverse reinforcement area scale reduces the effective volumetric ratio");
    }

    const auto output_dir =
        std::filesystem::temp_directory_path() /
        "fall_n_reduced_rc_continuum_baseline_smoke";
    std::filesystem::create_directories(output_dir);

    CyclicValidationRunConfig cfg{};
    cfg.protocol_name = "monotonic_smoke";
    cfg.execution_profile_name = "reduced_rc_continuum_smoke";
    cfg.amplitudes_m = {0.0005};
    cfg.steps_per_segment = 2;
    cfg.max_bisections = 4;

    const ReducedRCColumnContinuumRunSpec spec{
        .material_mode = ReducedRCColumnContinuumMaterialMode::elasticized,
        .reinforcement_mode =
            ReducedRCColumnContinuumReinforcementMode::embedded_longitudinal_bars,
        .hex_order = HexOrder::Linear,
        .nx = 2,
        .ny = 2,
        .nz = 4,
        .penalty_alpha_scale_over_ec = 1.0e4,
        .continuation_kind =
            ReducedRCColumnContinuationKind::
                monolithic_incremental_displacement_control,
        .solver_policy_kind = ReducedRCColumnSolverPolicyKind::newton_l2_only,
        .continuation_segment_substep_factor = 1,
        .write_hysteresis_csv = true,
        .write_control_state_csv = true,
        .print_progress = false,
    };

    const auto result = run_reduced_rc_column_small_strain_continuum_case_result(
        spec, output_dir.string(), cfg);
    double peak_top_face_dx = 0.0;
    for (const auto& row : result.control_state_records) {
        peak_top_face_dx = std::max(
            peak_top_face_dx,
            row.average_top_face_total_lateral_displacement);
    }

    check(result.completed_successfully,
          "elasticized continuum baseline completes");
    check(result.hysteresis_records.size() >= 2,
          "continuum baseline records hysteresis points");
    check(result.control_state_records.size() >= 2,
          "continuum baseline records control states");
    check(!result.rebar_history_records.empty(),
          "continuum baseline records embedded rebar history");
    check(result.concrete_profile_details.tangent_mode ==
              ReducedRCColumnContinuumConcreteTangentMode::fracture_secant,
          "continuum baseline promotes the fracture-secant Ko-Bathe tangent");
    check(result.concrete_profile_details.characteristic_length_mode ==
              ReducedRCColumnContinuumCharacteristicLengthMode::
                  mean_longitudinal_host_edge_mm,
          "continuum baseline promotes the mesh-aware longitudinal characteristic length");
    check(std::abs(result.concrete_profile_details.characteristic_length_mm - 800.0) <
              1.0e-9,
          "continuum baseline reports the evaluated mean longitudinal host edge length");

    const auto& last = result.control_state_records.back();
    check(peak_top_face_dx > 0.0,
          "top face reaches a positive lateral displacement");
    check(std::abs(last.top_rebar_minus_face_lateral_gap) < 1.0e-4,
          "embedded top rebar follows the host top face closely");
    check(std::filesystem::exists(output_dir / "hysteresis.csv"),
          "continuum baseline writes hysteresis.csv");
    check(std::filesystem::exists(output_dir / "control_state.csv"),
          "continuum baseline writes control_state.csv");
    check(std::filesystem::exists(output_dir / "crack_state.csv"),
          "continuum baseline writes crack_state.csv");
    check(std::filesystem::exists(output_dir / "rebar_history.csv"),
          "continuum baseline writes rebar_history.csv");

    {
        const auto kinematics_output_dir =
            std::filesystem::current_path() / "build" / "tests_output" /
            "reduced_rc_continuum_kinematics_policy_smoke";
        std::filesystem::create_directories(kinematics_output_dir);

        const CyclicValidationRunConfig kinematics_cfg{
            .protocol_name = "monotonic",
            .execution_profile_name =
                "reduced_rc_continuum_kinematics_policy_smoke",
            .amplitudes_m = {0.5e-3},
            .steps_per_segment = 1,
            .max_bisections = 1,
        };

        double small_strain_base_shear = 0.0;
        for (const auto kind : {
                 ReducedRCColumnContinuumKinematicPolicyKind::small_strain,
                 ReducedRCColumnContinuumKinematicPolicyKind::total_lagrangian,
                 ReducedRCColumnContinuumKinematicPolicyKind::updated_lagrangian,
                 ReducedRCColumnContinuumKinematicPolicyKind::corotational}) {
            const ReducedRCColumnContinuumRunSpec kinematics_spec{
                .kinematic_policy_kind = kind,
                .material_mode =
                    ReducedRCColumnContinuumMaterialMode::elasticized,
                .reinforcement_mode =
                    ReducedRCColumnContinuumReinforcementMode::continuum_only,
                .hex_order = HexOrder::Linear,
                .nx = 1,
                .ny = 1,
                .nz = 1,
                .continuation_kind =
                    ReducedRCColumnContinuationKind::
                        monolithic_incremental_displacement_control,
                .solver_policy_kind =
                    ReducedRCColumnSolverPolicyKind::
                        newton_l2_lu_symbolic_reuse_only,
                .predictor_policy_kind =
                    ReducedRCColumnPredictorPolicyKind::current_state_only,
                .continuation_segment_substep_factor = 1,
                .write_hysteresis_csv = false,
                .write_control_state_csv = false,
                .write_crack_state_csv = false,
                .write_rebar_history_csv = false,
                .write_embedding_gap_csv = false,
                .write_host_probe_csv = false,
                .print_progress = false,
            };

            const auto kinematics_result =
                run_reduced_rc_column_continuum_case_result(
                    kinematics_spec,
                    (kinematics_output_dir / to_string(kind)).string(),
                    kinematics_cfg);

            const double base_shear =
                std::abs(kinematics_result.hysteresis_records.back().base_shear);
            if (kind == ReducedRCColumnContinuumKinematicPolicyKind::small_strain) {
                small_strain_base_shear = base_shear;
            }
            const std::string policy_label =
                kind == ReducedRCColumnContinuumKinematicPolicyKind::small_strain
                    ? "small-strain"
                    : kind == ReducedRCColumnContinuumKinematicPolicyKind::total_lagrangian
                          ? "Total Lagrangian"
                          : kind == ReducedRCColumnContinuumKinematicPolicyKind::updated_lagrangian
                                ? "Updated Lagrangian"
                                : "corotational";
            check(
                kinematics_result.completed_successfully,
                (policy_label +
                 " continuum kinematics dispatch completes")
                    .c_str());
            check(
                std::isfinite(base_shear) && base_shear > 0.0,
                (policy_label +
                 " continuum kinematics records a finite reaction")
                    .c_str());
            if (kind != ReducedRCColumnContinuumKinematicPolicyKind::small_strain) {
                check(
                    std::abs(base_shear - small_strain_base_shear) /
                            std::max(small_strain_base_shear, 1.0e-12) <
                        1.0e-5,
                    (policy_label +
                     " continuum kinematics matches the small-displacement limit")
                        .c_str());
            }
        }

    }

    {
        const auto affine_cap_output_dir =
            std::filesystem::current_path() / "build" / "tests_output" /
            "reduced_rc_continuum_affine_top_cap_smoke";
        std::filesystem::create_directories(affine_cap_output_dir);

        const CyclicValidationRunConfig affine_cap_cfg{
            .protocol_name = "monotonic",
            .execution_profile_name =
                "reduced_rc_continuum_affine_top_cap_smoke",
            .amplitudes_m = {0.5e-3},
            .steps_per_segment = 1,
            .max_bisections = 1,
        };

        const double rotation_ratio = 0.5;
        const ReducedRCColumnContinuumRunSpec affine_cap_spec{
            .material_mode =
                ReducedRCColumnContinuumMaterialMode::elasticized,
            .reinforcement_mode =
                ReducedRCColumnContinuumReinforcementMode::continuum_only,
            .hex_order = HexOrder::Linear,
            .nx = 2,
            .ny = 2,
            .nz = 1,
            .top_cap_penalty_alpha_scale_over_ec = 1.0e4,
            .top_cap_bending_rotation_drift_ratio = rotation_ratio,
            .top_cap_mode =
                ReducedRCColumnContinuumTopCapMode::
                    affine_bending_rotation_penalty_cap,
            .continuation_kind =
                ReducedRCColumnContinuationKind::
                    monolithic_incremental_displacement_control,
            .solver_policy_kind =
                ReducedRCColumnSolverPolicyKind::newton_l2_only,
            .continuation_segment_substep_factor = 1,
            .write_hysteresis_csv = false,
            .write_control_state_csv = false,
            .write_crack_state_csv = false,
            .write_rebar_history_csv = false,
            .write_embedding_gap_csv = false,
            .write_host_probe_csv = false,
            .print_progress = false,
        };

        const auto affine_cap_result =
            run_reduced_rc_column_small_strain_continuum_case_result(
                affine_cap_spec,
                affine_cap_output_dir.string(),
                affine_cap_cfg);
        const auto& affine_last =
            affine_cap_result.control_state_records.back();
        const double expected_rotation_y =
            rotation_ratio * affine_last.target_drift /
            affine_cap_spec.reference_spec.column_height_m;

        check(
            affine_cap_result.completed_successfully,
            "affine bending-rotation top cap smoke completes");
        check(
            std::abs(
                affine_last.top_face_estimated_rotation_y -
                expected_rotation_y) < 1.0e-6,
            "affine bending-rotation top cap enforces the requested top rotation");
        check(
            affine_last.top_face_axial_plane_rms_residual < 1.0e-8,
            "affine bending-rotation top cap keeps the top axial field planar");
        check(
            affine_last.top_face_axial_displacement_range > 0.0,
            "affine bending-rotation top cap permits the expected axial wedge");
    }

    std::set<std::size_t> distinct_bar_indices;
    double max_rebar_kinematic_consistency_gap = 0.0;
    double max_host_bar_axial_strain_gap = 0.0;
    double max_nearest_host_gp_distance = 0.0;
    for (const auto& row : result.rebar_history_records) {
        distinct_bar_indices.insert(row.bar_index);
        max_rebar_kinematic_consistency_gap = std::max(
            max_rebar_kinematic_consistency_gap,
            std::abs(row.axial_strain - row.rebar_projected_axial_strain));
        max_host_bar_axial_strain_gap = std::max(
            max_host_bar_axial_strain_gap,
            std::abs(row.projected_axial_strain_gap));
        max_nearest_host_gp_distance = std::max(
            max_nearest_host_gp_distance,
            row.nearest_host_gp_distance);
    }
    check(distinct_bar_indices.size() == 8,
          "continuum baseline defaults to the structural eight-bar rebar layout");
    check(max_rebar_kinematic_consistency_gap < 1.0e-12,
          "continuum baseline keeps the exported truss strain consistent with the projected bar kinematics");
    check(max_host_bar_axial_strain_gap < 2.0e-4,
          "continuum baseline keeps the embedded steel close to the projected host axial strain in the elasticized smoke case");
    check(max_nearest_host_gp_distance > 0.0,
          "continuum baseline records the nearest host Gauss point around the embedded steel path");

    {
        const auto reaction_output_dir =
            std::filesystem::current_path() / "build" / "tests_output" /
            "reduced_rc_continuum_support_resultant_smoke";
        std::filesystem::create_directories(reaction_output_dir);

        const CyclicValidationRunConfig reaction_cfg{
            .protocol_name = "monotonic",
            .execution_profile_name = "reduced_rc_continuum_support_resultant_smoke",
            .amplitudes_m = {0.0},
            .steps_per_segment = 1,
            .max_bisections = 0,
        };

        for (const auto hex_order : {HexOrder::Serendipity, HexOrder::Quadratic}) {
            ReducedRCColumnContinuumRunSpec reaction_spec{
                .material_mode =
                    ReducedRCColumnContinuumMaterialMode::elasticized,
                .reinforcement_mode =
                    ReducedRCColumnContinuumReinforcementMode::continuum_only,
                .hex_order = hex_order,
                .nx = 2,
                .ny = 2,
                .nz = 2,
                .axial_compression_force_mn = 0.02,
                .use_equilibrated_axial_preload_stage = true,
                .axial_preload_steps = 2,
                .continuation_kind =
                    ReducedRCColumnContinuationKind::
                        monolithic_incremental_displacement_control,
                .solver_policy_kind =
                    ReducedRCColumnSolverPolicyKind::newton_l2_only,
                .continuation_segment_substep_factor = 1,
                .write_hysteresis_csv = false,
                .write_control_state_csv = false,
                .write_crack_state_csv = false,
                .write_rebar_history_csv = false,
                .write_embedding_gap_csv = false,
                .write_host_probe_csv = false,
                .print_progress = false,
            };

            const auto reaction_result =
                run_reduced_rc_column_small_strain_continuum_case_result(
                    reaction_spec,
                    (reaction_output_dir / to_string(hex_order)).string(),
                    reaction_cfg);

            check(
                reaction_result.completed_successfully,
                hex_order == HexOrder::Serendipity
                    ? "Hex20 support resultant smoke completes"
                    : "Hex27 support resultant smoke completes");

            const auto support_reaction =
                reaction_result.control_state_records.back().base_axial_reaction;
            check(
                std::abs(std::abs(support_reaction) - 0.02) < 1.0e-6,
                hex_order == HexOrder::Serendipity
                    ? "Hex20 support resultant recovers the applied surface traction magnitude"
                    : "Hex27 support resultant recovers the applied surface traction magnitude");
        }
    }

    {
        const auto embedded_reaction_output_dir =
            std::filesystem::current_path() / "build" / "tests_output" /
            "reduced_rc_continuum_embedded_support_resultant_smoke";
        std::filesystem::create_directories(embedded_reaction_output_dir);

        const CyclicValidationRunConfig embedded_reaction_cfg{
            .protocol_name = "monotonic",
            .execution_profile_name =
                "reduced_rc_continuum_embedded_support_resultant_smoke",
            .amplitudes_m = {0.0},
            .steps_per_segment = 1,
            .max_bisections = 0,
        };

        const ReducedRCColumnContinuumRunSpec embedded_reaction_spec{
            .material_mode = ReducedRCColumnContinuumMaterialMode::elasticized,
            .reinforcement_mode =
                ReducedRCColumnContinuumReinforcementMode::
                    embedded_longitudinal_bars,
            .hex_order = HexOrder::Serendipity,
            .nx = 2,
            .ny = 2,
            .nz = 2,
            .axial_compression_force_mn = 0.02,
            .axial_preload_transfer_mode =
                ReducedRCColumnContinuumAxialPreloadTransferMode::
                    composite_section_force_split,
            .use_equilibrated_axial_preload_stage = true,
            .axial_preload_steps = 2,
            .continuation_kind =
                ReducedRCColumnContinuationKind::
                    monolithic_incremental_displacement_control,
            .solver_policy_kind =
                ReducedRCColumnSolverPolicyKind::newton_l2_only,
            .continuation_segment_substep_factor = 1,
            .write_hysteresis_csv = false,
            .write_control_state_csv = false,
            .write_crack_state_csv = false,
            .write_rebar_history_csv = false,
            .write_embedding_gap_csv = false,
            .write_host_probe_csv = false,
            .print_progress = false,
        };

        const auto embedded_reaction_result =
            run_reduced_rc_column_small_strain_continuum_case_result(
                embedded_reaction_spec,
                embedded_reaction_output_dir.string(),
                embedded_reaction_cfg);

        check(
            embedded_reaction_result.completed_successfully,
            "embedded support resultant smoke completes");
        check(
            std::abs(
                std::abs(
                    embedded_reaction_result.control_state_records.back()
                        .base_axial_reaction) -
                0.02) < 1.0e-6,
            "embedded support resultant includes the rebar-endcap reaction contribution");
    }

    {
        const auto cover_core_output_dir =
            std::filesystem::current_path() /
            "build" / "tests_output" / "reduced_rc_continuum_cover_core_smoke";
        std::filesystem::create_directories(cover_core_output_dir);

        const CyclicValidationRunConfig cover_core_cfg{
            .protocol_name = "monotonic",
            .execution_profile_name = "reduced_rc_continuum_cover_core_smoke",
            .amplitudes_m = {0.5e-3},
            .steps_per_segment = 3,
            .max_bisections = 0,
        };

        const ReducedRCColumnContinuumRunSpec cover_core_spec{
            .material_mode =
                ReducedRCColumnContinuumMaterialMode::elasticized,
            .reinforcement_mode =
                ReducedRCColumnContinuumReinforcementMode::continuum_only,
            .host_concrete_zoning_mode =
                ReducedRCColumnContinuumHostConcreteZoningMode::cover_core_split,
            .transverse_mesh_mode =
                ReducedRCColumnContinuumTransverseMeshMode::cover_aligned,
            .hex_order = HexOrder::Serendipity,
            .nx = 4,
            .ny = 4,
            .nz = 2,
            .continuation_kind =
                ReducedRCColumnContinuationKind::
                    monolithic_incremental_displacement_control,
            .solver_policy_kind =
                ReducedRCColumnSolverPolicyKind::newton_l2_only,
            .continuation_segment_substep_factor = 1,
            .write_hysteresis_csv = false,
            .write_control_state_csv = false,
            .write_crack_state_csv = false,
            .write_rebar_history_csv = false,
            .write_embedding_gap_csv = false,
            .write_host_probe_csv = true,
            .print_progress = false,
            .host_probe_specs = {{
                .label = "cover_probe",
                .x = 0.095,
                .y = 0.095,
                .z = 1.6,
            }},
        };

        const auto cover_core_result =
            run_reduced_rc_column_small_strain_continuum_case_result(
                cover_core_spec,
                cover_core_output_dir.string(),
                cover_core_cfg);

        check(cover_core_result.completed_successfully,
              "cover-core aligned elasticized continuum baseline completes");
        check(cover_core_result.rebar_history_records.empty(),
              "continuum-only cover-core baseline omits embedded steel history");
        check(!cover_core_result.host_probe_records.empty(),
              "continuum baseline records host probe history without embedded steel");
        check(std::filesystem::exists(cover_core_output_dir / "host_probe_history.csv"),
              "continuum baseline writes host_probe_history.csv");
    }

    {
        const auto proxy_output_dir =
            std::filesystem::current_path() / "build" / "tests_output" /
            "reduced_rc_continuum_bimodular_proxy_smoke";
        std::filesystem::create_directories(proxy_output_dir);

        const CyclicValidationRunConfig proxy_cfg{
            .protocol_name = "cyclic",
            .execution_profile_name = "reduced_rc_continuum_bimodular_proxy_smoke",
            .amplitudes_m = {5.0e-3, 10.0e-3},
            .steps_per_segment = 2,
            .max_bisections = 2,
        };

        const ReducedRCColumnContinuumRunSpec proxy_spec{
            .material_mode =
                ReducedRCColumnContinuumMaterialMode::
                    orthotropic_bimodular_proxy,
            .reinforcement_mode =
                ReducedRCColumnContinuumReinforcementMode::
                    embedded_longitudinal_bars,
            .hex_order = HexOrder::Linear,
            .nx = 2,
            .ny = 2,
            .nz = 2,
            .concrete_tension_stiffness_ratio = 0.10,
            .axial_compression_force_mn = 0.02,
            .continuation_kind =
                ReducedRCColumnContinuationKind::
                    reversal_guarded_incremental_displacement_control,
            .solver_policy_kind =
                ReducedRCColumnSolverPolicyKind::newton_l2_only,
            .continuation_segment_substep_factor = 1,
            .write_hysteresis_csv = false,
            .write_control_state_csv = false,
            .write_crack_state_csv = false,
            .write_rebar_history_csv = false,
            .write_embedding_gap_csv = false,
            .write_host_probe_csv = false,
            .print_progress = false,
        };

        const auto proxy_result =
            run_reduced_rc_column_small_strain_continuum_case_result(
                proxy_spec,
                proxy_output_dir.string(),
                proxy_cfg);

        double max_abs_proxy_rebar_stress = 0.0;
        for (const auto& row : proxy_result.rebar_history_records) {
            max_abs_proxy_rebar_stress = std::max(
                max_abs_proxy_rebar_stress,
                std::abs(row.axial_stress));
        }

        check(
            proxy_result.completed_successfully,
            "bimodular proxy continuum baseline completes");
        check(
            !proxy_result.rebar_history_records.empty(),
            "bimodular proxy continuum baseline records nonlinear steel history");
        check(
            max_abs_proxy_rebar_stress > 25.0,
            "bimodular proxy continuum baseline develops a nontrivial steel stress response");
    }

    {
        const auto transverse_output_dir =
            std::filesystem::current_path() / "build" / "tests_output" /
            "reduced_rc_continuum_transverse_history_smoke";
        std::filesystem::create_directories(transverse_output_dir);

        const CyclicValidationRunConfig transverse_cfg{
            .protocol_name = "monotonic",
            .execution_profile_name =
                "reduced_rc_continuum_transverse_history_smoke",
            .amplitudes_m = {5.0e-4},
            .steps_per_segment = 1,
            .max_bisections = 1,
        };

        const ReducedRCColumnContinuumRunSpec transverse_spec{
            .material_mode = ReducedRCColumnContinuumMaterialMode::elasticized,
            .reinforcement_mode =
                ReducedRCColumnContinuumReinforcementMode::
                    embedded_longitudinal_bars,
            .transverse_reinforcement_mode =
                ReducedRCColumnContinuumTransverseReinforcementMode::
                    embedded_stirrup_loops,
            .hex_order = HexOrder::Linear,
            .nx = 2,
            .ny = 2,
            .nz = 1,
            .transverse_reinforcement_area_scale = 0.25,
            .continuation_kind =
                ReducedRCColumnContinuationKind::
                    monolithic_incremental_displacement_control,
            .solver_policy_kind =
                ReducedRCColumnSolverPolicyKind::newton_l2_only,
            .continuation_segment_substep_factor = 1,
            .write_hysteresis_csv = false,
            .write_control_state_csv = false,
            .write_crack_state_csv = false,
            .write_rebar_history_csv = true,
            .write_embedding_gap_csv = false,
            .write_host_probe_csv = false,
            .print_progress = false,
        };

        const auto transverse_result =
            run_reduced_rc_column_small_strain_continuum_case_result(
                transverse_spec,
                transverse_output_dir.string(),
                transverse_cfg);

        check(
            transverse_result.completed_successfully,
            "embedded stirrup smoke completes");
        check(
            !transverse_result.transverse_rebar_history_records.empty(),
            "embedded stirrup smoke records transverse steel history");
        check(
            std::filesystem::exists(
                transverse_output_dir / "transverse_rebar_history.csv"),
            "embedded stirrup smoke writes transverse_rebar_history.csv");
    }

    {
        PrismaticSpec prism{
            .width = 0.3,
            .height = 0.3,
            .length = 1.0,
            .nx = 1,
            .ny = 1,
            .nz = 1,
            .hex_order = HexOrder::Serendipity,
            .physical_group = "Host",
        };
        RebarSpec rebar{
            .bars = {{
                .ly = 0.0,
                .lz = 0.0,
                .area = 1.0e-4,
                .diameter = 0.012,
                .group = "Rebar",
            }},
        };

        auto linear_rd = make_reinforced_prismatic_domain(
            prism,
            rebar,
            RebarLineInterpolation::two_node_linear);
        auto quadratic_rd = make_reinforced_prismatic_domain(
            prism,
            rebar,
            RebarLineInterpolation::three_node_quadratic);

        {
            auto& geom = linear_rd.domain.element(linear_rd.rebar_range.first);
            TrussElement<3, 2> truss{
                &geom, make_elastic_steel_for_embedding(), rebar.bars.front().area};
            truss.set_num_dof_in_nodes();
            PenaltyCoupling coupling;
            coupling.setup(
                linear_rd.domain,
                linear_rd.grid,
                linear_rd.embeddings,
                1,
                1.0e4,
                true,
                HexOrder::Serendipity);
            check(coupling.num_couplings() == 0,
                  "two-node embedded truss on a single Hex20 element leaves no interior bar node to couple");
        }

        {
            auto& geom =
                quadratic_rd.domain.element(quadratic_rd.rebar_range.first);
            TrussElement<3, 3> truss{
                &geom, make_elastic_steel_for_embedding(), rebar.bars.front().area};
            truss.set_num_dof_in_nodes();
            PenaltyCoupling coupling;
            coupling.setup(
                quadratic_rd.domain,
                quadratic_rd.grid,
                quadratic_rd.embeddings,
                1,
                1.0e4,
                true,
                HexOrder::Serendipity);
            check(coupling.num_couplings() == 1,
                  "three-node embedded truss couples its interior midpoint to the Hex20 host");
        }
    }

    {
        using ContinuumElemT =
            ContinuumElement<ThreeDimensionalMaterial, 3, continuum::SmallStrain>;
        using ModelT =
            Model<ThreeDimensionalMaterial, continuum::SmallStrain, 3, MultiElementPolicy>;

        PrismaticSpec prism{
            .width = 0.3,
            .height = 0.3,
            .length = 1.0,
            .nx = 2,
            .ny = 2,
            .nz = 1,
            .hex_order = HexOrder::Linear,
            .physical_group = "Host",
        };
        auto [domain, grid] = make_prismatic_domain(prism);
        auto concrete_material = Material<ThreeDimensionalMaterial>{
            ContinuumIsotropicElasticMaterial{25000.0, 0.2},
            ElasticUpdate{}};

        std::vector<FEM_Element> elements;
        elements.reserve(domain.num_elements());
        for (std::size_t element_index = 0;
             element_index < domain.num_elements();
             ++element_index) {
            elements.emplace_back(
                ContinuumElemT{&domain.element(element_index), concrete_material});
        }
        ModelT model{domain, std::move(elements)};
        const auto top_nodes =
            active_face_nodes(domain, grid.nodes_on_face(PrismFace::MaxZ));

        PenaltyDofTie top_cap_tie;
        top_cap_tie.setup(domain, top_nodes, top_nodes.front(), 2, 1.0e6);
        check(top_cap_tie.num_ties() + 1 == top_nodes.size(),
              "uniform axial top-cap tie connects every non-anchor top-face node");
        check(top_cap_tie.component() == 2,
              "uniform axial top-cap tie acts on the axial displacement component");
        check(top_cap_tie.anchor_node_id() == top_nodes.front(),
              "uniform axial top-cap tie keeps the selected master node explicit");
        (void)model;
    }

    {
        PrismaticSpec prism{
            .width = 0.25,
            .height = 0.25,
            .length = 1.0,
            .nx = 4,
            .ny = 4,
            .nz = 1,
            .hex_order = HexOrder::Serendipity,
            .x_corner_levels_local = {-0.125, -0.095, 0.0, 0.095, 0.125},
            .y_corner_levels_local = {-0.125, -0.095, 0.0, 0.095, 0.125},
            .physical_group = "Host",
        };
        RebarSpec boundary_rebar{
            .bars = {{
                .ly = -0.125,
                .lz = 0.0,
                .area = 1.0e-4,
                .diameter = 0.012,
                .group = "Rebar",
            }},
        };

        auto boundary_rd = make_reinforced_prismatic_domain(
            prism,
            boundary_rebar,
            RebarLineInterpolation::three_node_quadratic);
        const auto& midpoint_embedding = boundary_rd.embeddings.at(1);

        check(std::abs(boundary_rd.grid.x_coordinate(2) - (-0.095)) < 1.0e-12,
              "cover-aligned prism stores the requested transverse x levels");
        check(midpoint_embedding.host_elem_ix == 0,
              "boundary rebar stays attached to the outer host strip");
        check(std::abs(midpoint_embedding.xi + 1.0) < 1.0e-12,
              "boundary rebar maps to the host boundary with xi = -1");
    }

    {
        PrismaticSpec prism{
            .width = 0.25,
            .height = 0.25,
            .length = 1.0,
            .nx = 4,
            .ny = 4,
            .nz = 1,
            .hex_order = HexOrder::Serendipity,
            .x_corner_levels_local = {-0.125, -0.095, 0.0, 0.095, 0.125},
            .y_corner_levels_local = {-0.125, -0.095, 0.0, 0.095, 0.125},
            .physical_group = "Host",
        };
        RebarSpec interface_rebar{
            .bars = {{
                .ly = -0.095,
                .lz = 0.0,
                .area = 1.0e-4,
                .diameter = 0.012,
                .group = "Rebar",
            }},
        };

        auto interface_rd = make_reinforced_prismatic_domain(
            prism,
            interface_rebar,
            RebarLineInterpolation::three_node_quadratic);
        const auto& midpoint_embedding = interface_rd.embeddings.at(1);

        check(midpoint_embedding.host_elem_ix == 1,
              "interface rebar is attached to the first interior strip on a cover-aligned prism");
        check(std::abs(midpoint_embedding.xi + 1.0) < 1.0e-12,
              "interface rebar maps to a shared internal face with xi = -1");
    }

    {
        PrismaticSpec prism{
            .width = 0.30,
            .height = 0.30,
            .length = 1.0,
            .nx = 2,
            .ny = 2,
            .nz = 2,
            .hex_order = HexOrder::Linear,
            .physical_group = "Host",
        };
        EmbeddedRebarSpec stirrup_spec{
            .polylines = {{
                .local_points = {
                    {-0.10, -0.10, 0.25},
                    { 0.10, -0.10, 0.25},
                    { 0.10,  0.10, 0.25},
                    {-0.10,  0.10, 0.25},
                },
                .closed = true,
                .area = 5.0e-5,
                .diameter = 8.0e-3,
                .group = "TransverseStirrup",
            }},
        };

        auto stirrup_rd = make_reinforced_prismatic_domain(
            prism,
            RebarSpec{},
            RebarLineInterpolation::two_node_linear,
            stirrup_spec);

        check(stirrup_rd.rebar_range.first == stirrup_rd.rebar_range.last,
              "empty longitudinal rebar spec leaves the longitudinal range empty");
        check(stirrup_rd.embedded_rebar_range.last -
                    stirrup_rd.embedded_rebar_range.first ==
                  4,
              "one closed embedded stirrup polyline creates four truss segments");
        check(stirrup_rd.embedded_rebar_embeddings.size() == 4,
              "one closed embedded stirrup polyline creates four embedded nodes");
        check(stirrup_rd.embedded_rebar_elements.size() == 4,
              "embedded stirrup segment metadata follows the truss elements");

        const auto& emb = stirrup_rd.embedded_rebar_embeddings.front();
        check(emb.host_elem_ix == 0 && emb.host_elem_iy == 0 &&
                  emb.host_elem_iz == 0,
              "embedded stirrup node is located in the expected host element");
        check(std::abs(emb.zeta) < 1.0e-12,
              "embedded stirrup node stores the exact parent zeta coordinate");

        {
            auto& geom =
                stirrup_rd.domain.element(stirrup_rd.embedded_rebar_range.first);
            TrussElement<3, 2> truss{
                &geom, make_elastic_steel_for_embedding(), 5.0e-5};
            truss.set_num_dof_in_nodes();
            PenaltyCoupling coupling;
            coupling.setup_embedded_nodes(
                stirrup_rd.domain,
                stirrup_rd.grid,
                stirrup_rd.embedded_rebar_embeddings,
                1.0e4,
                HexOrder::Linear);
            check(coupling.num_couplings() == 2,
                  "arbitrary embedded-node coupling includes active stirrup nodes");
        }
    }

    {
        const auto hex20_truss2 = solve_linearized_preload_slice(
            HexOrder::Serendipity,
            RebarLineInterpolation::two_node_linear,
            ReducedRCColumnContinuumAxialPreloadTransferMode::host_surface_only);
        const auto hex20_truss3 = solve_linearized_preload_slice(
            HexOrder::Serendipity,
            RebarLineInterpolation::three_node_quadratic,
            ReducedRCColumnContinuumAxialPreloadTransferMode::host_surface_only);
        const auto hex27_truss3 = solve_linearized_preload_slice(
            HexOrder::Quadratic,
            RebarLineInterpolation::automatic,
            ReducedRCColumnContinuumAxialPreloadTransferMode::host_surface_only);

        check(hex20_truss2.ksp_converged,
              "Hex20 + Truss<2> preload slice admits a direct linear solve");
        check(hex20_truss2.relative_residual_norm < 1.0e-8,
              "Hex20 + Truss<2> preload linear solve leaves a small relative residual");

        check(hex20_truss3.ksp_converged,
              "Hex20 + Truss<3> preload slice admits a direct linear solve");
        check(hex20_truss3.relative_residual_norm < 1.0e-8,
              "Hex20 + Truss<3> preload linear solve leaves a small relative residual");

        check(hex27_truss3.ksp_converged,
              "Hex27 + Truss<3> preload slice admits a direct linear solve");
        check(hex27_truss3.relative_residual_norm < 1.0e-8,
              "Hex27 + Truss<3> preload linear solve leaves a small relative residual");
    }

    std::cout << "\nPassed: " << passed << "  Failed: " << (total - passed)
              << '\n';
    PetscFinalize();
    return (passed == total) ? 0 : 1;
}
