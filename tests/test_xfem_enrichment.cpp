#include "src/xfem/CohesiveCrackLaw.hh"
#include "src/xfem/CrackKinematics.hh"
#include "src/xfem/ShiftedHeavisideCrackCrossingRebarElement.hh"
#include "src/xfem/ShiftedHeavisideSolidElement.hh"
#include "src/xfem/XFEMDofManager.hh"
#include "src/xfem/XFEMEnrichedApproximation.hh"
#include "src/xfem/XFEMEnrichment.hh"
#include "src/domain/Domain.hh"
#include "src/elements/ContinuumElement.hh"
#include "src/elements/ElementPolicy.hh"
#include "src/elements/element_geometry/LagrangeElement.hh"
#include "src/materials/LinealElasticMaterial.hh"
#include "src/materials/Material.hh"
#include "src/materials/MaterialPolicy.hh"
#include "src/analysis/IncrementalControl.hh"
#include "src/analysis/NLAnalysis.hh"
#include "src/model/Model.hh"
#include "src/numerics/numerical_integration/GaussLegendreCellIntegrator.hh"

#include <Eigen/Dense>
#include <petsc.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numbers>
#include <vector>

namespace {

int total_tests = 0;
int passed_tests = 0;

void check(bool condition, const char* message)
{
    ++total_tests;
    if (condition) {
        ++passed_tests;
        std::cout << "[PASS] " << message << "\n";
    } else {
        std::cout << "[FAIL] " << message << "\n";
    }
}

void make_unit_hex8_domain(Domain<3>& domain)
{
    domain.preallocate_node_capacity(8);
    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1, 1.0, 0.0, 0.0);
    domain.add_node(2, 0.0, 1.0, 0.0);
    domain.add_node(3, 1.0, 1.0, 0.0);
    domain.add_node(4, 0.0, 0.0, 1.0);
    domain.add_node(5, 1.0, 0.0, 1.0);
    domain.add_node(6, 0.0, 1.0, 1.0);
    domain.add_node(7, 1.0, 1.0, 1.0);

    std::array<PetscInt, 8> conn{0, 1, 2, 3, 4, 5, 6, 7};
    domain.make_element<LagrangeElement3D<2, 2, 2>>(
        GaussLegendreCellIntegrator<2, 2, 2>{},
        0,
        conn.data());
    domain.assemble_sieve();
}

} // namespace

int main(int argc, char** argv)
{
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    using fall_n::xfem::HeavisideSide;
    using fall_n::xfem::PlaneCrackLevelSet;
    using fall_n::xfem::XFEMCrackPlane;
    using fall_n::xfem::XFEMCrackPlaneSource;
    using fall_n::xfem::ShiftedHeavisideEnrichment;
    using fall_n::xfem::BilinearCohesiveLawParameters;
    using fall_n::xfem::CohesiveCrackState;
    using fall_n::xfem::evaluate_shifted_heaviside_kinematics;
    using fall_n::xfem::evaluate_bilinear_cohesive_law;
    using fall_n::xfem::element_is_cut_by_crack;
    using fall_n::xfem::crack_band_opening_mm;
    using fall_n::xfem::make_crack_band_consistent_cohesive_law;
    using fall_n::xfem::
        make_metre_jump_crack_band_consistent_cohesive_law_from_mpa_n_per_mm;
    using fall_n::xfem::mark_heaviside_enriched_nodes;
    using fall_n::xfem::apply_shifted_heaviside_dof_layout;
    using fall_n::xfem::node_has_shifted_heaviside_enrichment;
    using fall_n::xfem::petsc_global_dof_index;
    using fall_n::xfem::petsc_local_dof_index;
    using fall_n::xfem::shifted_heaviside_enriched_component;
    using fall_n::xfem::split_crack_jump;
    using fall_n::xfem::ShiftedHeavisideCrackCrossingRebarElement;
    using fall_n::xfem::CrackCrossingRebarAxisFrameKind;
    using fall_n::xfem::BoundedSlipBridgeParameters;
    using fall_n::xfem::BoundedSlipBridgeState;
    using fall_n::xfem::evaluate_bounded_slip_bridge_law;

    const PlaneCrackLevelSet crack{
        Eigen::Vector3d::Zero(),
        Eigen::Vector3d{2.0, 0.0, 0.0}};

    check(std::abs(crack.normal.norm() - 1.0) < 1.0e-14,
          "plane crack level-set normal is normalized");
    check(crack.side(Eigen::Vector3d{-1.0, 0.0, 0.0}) ==
              HeavisideSide::negative,
          "level set classifies the negative side");
    check(crack.side(Eigen::Vector3d{1.0, 0.0, 0.0}) ==
              HeavisideSide::positive,
          "level set classifies the positive side");
    check(crack.side(Eigen::Vector3d::Zero()) ==
              HeavisideSide::on_interface,
          "level set classifies points on the interface");

    const ShiftedHeavisideEnrichment enrichment{crack};
    const Eigen::Vector3d negative_node{-1.0, 0.0, 0.0};
    check(enrichment(negative_node, negative_node) == 0.0,
          "shifted Heaviside enrichment vanishes at its own node");
    check(enrichment(Eigen::Vector3d{1.0, 0.0, 0.0}, negative_node) == 2.0,
          "shifted Heaviside enrichment captures the crack jump");

    {
        const auto jump = split_crack_jump(
            Eigen::Vector3d{2.0, 0.0, 0.0},
            Eigen::Vector3d{0.10, 0.03, 0.0});
        check(std::abs(jump.normal_opening - 0.10) < 1.0e-14,
              "shared crack kinematics extracts normal opening");
        check((jump.tangential_jump - Eigen::Vector3d{0.0, 0.03, 0.0}).norm() <
                  1.0e-14,
              "shared crack kinematics extracts tangential slip");
        check(std::abs(crack_band_opening_mm(2.0e-4, 400.0) - 0.08) <
                  1.0e-14,
              "crack-band strain converts to an objective opening length");
    }

    const std::array<Eigen::Vector3d, 5> nodes{
        Eigen::Vector3d{-1.0, -1.0, 0.0},
        Eigen::Vector3d{1.0, -1.0, 0.0},
        Eigen::Vector3d{-1.0, 1.0, 0.0},
        Eigen::Vector3d{1.0, 1.0, 0.0},
        Eigen::Vector3d{2.0, 2.0, 0.0}};
    const std::array<std::size_t, 4> cut_element{0, 1, 2, 3};
    const std::array<std::size_t, 3> uncut_element{1, 3, 4};

    check(element_is_cut_by_crack(nodes, cut_element, crack),
          "sign-changing element is classified as cut");
    check(!element_is_cut_by_crack(nodes, uncut_element, crack),
          "same-side element is not classified as cut");

    const std::vector<std::vector<std::size_t>> elements{
        {0, 1, 2, 3},
        {1, 3, 4}};
    const auto enriched = mark_heaviside_enriched_nodes(nodes, elements, crack);

    check(enriched.size() == nodes.size(),
          "XFEM enrichment mask has one flag per node");
    check(enriched[0] && enriched[1] && enriched[2] && enriched[3],
          "nodes belonging to a cut element are enriched");
    check(!enriched[4],
          "node used only by an uncut element is not enriched");

    {
        const PlaneCrackLevelSet base_crack{
            Eigen::Vector3d{0.0, 0.0, 0.25},
            Eigen::Vector3d::UnitZ()};
        const std::array<Eigen::Vector3d, 12> column_nodes{
            Eigen::Vector3d{-1.0, -1.0, 0.0},
            Eigen::Vector3d{1.0, -1.0, 0.0},
            Eigen::Vector3d{-1.0, 1.0, 0.0},
            Eigen::Vector3d{1.0, 1.0, 0.0},
            Eigen::Vector3d{-1.0, -1.0, 1.0},
            Eigen::Vector3d{1.0, -1.0, 1.0},
            Eigen::Vector3d{-1.0, 1.0, 1.0},
            Eigen::Vector3d{1.0, 1.0, 1.0},
            Eigen::Vector3d{-1.0, -1.0, 2.0},
            Eigen::Vector3d{1.0, -1.0, 2.0},
            Eigen::Vector3d{-1.0, 1.0, 2.0},
            Eigen::Vector3d{1.0, 1.0, 2.0}};
        const std::vector<std::vector<std::size_t>> column_hexes{
            {0, 1, 3, 2, 4, 5, 7, 6},
            {4, 5, 7, 6, 8, 9, 11, 10}};
        const auto column_enriched = mark_heaviside_enriched_nodes(
            column_nodes,
            column_hexes,
            base_crack);

        bool base_element_enriched = true;
        for (std::size_t node_id = 0; node_id < 8; ++node_id) {
            base_element_enriched =
                base_element_enriched && column_enriched[node_id];
        }
        bool healthy_top_unenriched = true;
        for (std::size_t node_id = 8; node_id < column_enriched.size();
             ++node_id) {
            healthy_top_unenriched =
                healthy_top_unenriched && !column_enriched[node_id];
        }
        check(base_element_enriched,
              "3D column crack enriches only nodes attached to the cut base element");
        check(healthy_top_unenriched,
              "3D column crack does not enrich nodes used only by uncut elements");
    }

    {
        const std::array<double, 4> N{0.25, 0.25, 0.25, 0.25};
        const std::array<Eigen::Vector3d, 4> grad_N{
            Eigen::Vector3d{-0.25, -0.25, 0.0},
            Eigen::Vector3d{0.25, -0.25, 0.0},
            Eigen::Vector3d{-0.25, 0.25, 0.0},
            Eigen::Vector3d{0.25, 0.25, 0.0}};
        const std::array<Eigen::Vector3d, 4> local_nodes{
            Eigen::Vector3d{-1.0, -1.0, 0.0},
            Eigen::Vector3d{1.0, -1.0, 0.0},
            Eigen::Vector3d{-1.0, 1.0, 0.0},
            Eigen::Vector3d{1.0, 1.0, 0.0}};
        const std::array<Eigen::Vector3d, 4> standard_dofs = local_nodes;
        const std::array<Eigen::Vector3d, 4> enriched_dofs{
            Eigen::Vector3d{0.10, 0.02, 0.0},
            Eigen::Vector3d{0.10, 0.02, 0.0},
            Eigen::Vector3d{0.10, 0.02, 0.0},
            Eigen::Vector3d{0.10, 0.02, 0.0}};
        const std::array<std::uint8_t, 4> enriched_flags{1, 1, 1, 1};

        const auto positive_side = evaluate_shifted_heaviside_kinematics(
            N,
            grad_N,
            local_nodes,
            standard_dofs,
            enriched_dofs,
            enriched_flags,
            crack,
            Eigen::Vector3d{0.25, 0.0, 0.0});
        const auto negative_side = evaluate_shifted_heaviside_kinematics(
            N,
            grad_N,
            local_nodes,
            standard_dofs,
            enriched_dofs,
            enriched_flags,
            crack,
            Eigen::Vector3d{-0.25, 0.0, 0.0});

        check((positive_side.displacement -
               Eigen::Vector3d{0.10, 0.02, 0.0}).norm() < 1.0e-12,
              "shifted Heaviside bulk displacement evaluates the positive-side branch");
        check((negative_side.displacement +
               Eigen::Vector3d{0.10, 0.02, 0.0}).norm() < 1.0e-12,
              "shifted Heaviside bulk displacement evaluates the negative-side branch");
        check(std::abs(positive_side.displacement_gradient(0, 0) - 0.9) <
                  1.0e-12,
              "bulk enriched gradient adds the shifted Heaviside contribution");
        check(std::abs(positive_side.engineering_strain[0] - 0.9) <
                  1.0e-12,
              "XFEM utility exports engineering strain from the enriched gradient");
        check((positive_side.crack_jump -
               Eigen::Vector3d{0.20, 0.04, 0.0}).norm() < 1.0e-12,
              "shifted Heaviside trace jump equals two times the enriched trace field");
        check(std::abs(positive_side.normal_opening - 0.20) < 1.0e-12,
              "normal opening projects the crack jump onto the level-set normal");
        check((positive_side.tangential_jump -
               Eigen::Vector3d{0.0, 0.04, 0.0}).norm() < 1.0e-12,
              "tangential jump removes the normal opening component");
    }

    {
        const BilinearCohesiveLawParameters cohesive{
            .normal_stiffness = 1000.0,
            .shear_stiffness = 500.0,
            .tensile_strength = 10.0,
            .fracture_energy = 0.10,
            .mode_mixity_weight = 1.0,
            .compression_stiffness = 2000.0,
            .residual_shear_fraction = 0.20};
        const Eigen::Vector3d n = Eigen::Vector3d::UnitX();

        const auto elastic = evaluate_bilinear_cohesive_law(
            cohesive,
            CohesiveCrackState{},
            n,
            0.005,
            Eigen::Vector3d::Zero());
        check(std::abs(elastic.damage) < 1.0e-14,
              "cohesive law stays undamaged before the onset separation");
        check(std::abs(elastic.traction.x() - 5.0) < 1.0e-12,
              "cohesive law returns the elastic normal traction");

        const auto softening = evaluate_bilinear_cohesive_law(
            cohesive,
            elastic.updated_state,
            n,
            0.015,
            Eigen::Vector3d::Zero());
        check(std::abs(softening.damage - (2.0 / 3.0)) < 1.0e-12,
              "cohesive law follows the bilinear softening damage envelope");
        check(std::abs(softening.traction.x() - 5.0) < 1.0e-12,
              "cohesive softening traction lies on the descending branch");

        auto central_tangent_cohesive = cohesive;
        central_tangent_cohesive.tangent_mode =
            fall_n::xfem::CohesiveCrackTangentMode::
                adaptive_central_difference_with_secant_fallback;
        const auto softening_with_tangent = evaluate_bilinear_cohesive_law(
            central_tangent_cohesive,
            elastic.updated_state,
            n,
            0.015,
            Eigen::Vector3d::Zero());
        check(softening_with_tangent.tangent_stiffness(0, 0) < 0.0,
              "cohesive central tangent captures negative tensile softening");
        check(std::abs(softening_with_tangent.secant_stiffness(0, 0) -
                       softening.secant_stiffness(0, 0)) < 1.0e-10,
              "cohesive central tangent preserves the secant response field");

        auto active_set_cohesive = cohesive;
        active_set_cohesive.tangent_mode =
            fall_n::xfem::CohesiveCrackTangentMode::
                active_set_consistent;
        const auto active_set_softening = evaluate_bilinear_cohesive_law(
            active_set_cohesive,
            elastic.updated_state,
            n,
            0.015,
            Eigen::Vector3d::Zero());
        check(active_set_softening.tangent_stiffness(0, 0) < 0.0,
              "cohesive active-set tangent captures normal tensile softening");
        check(std::abs(active_set_softening.tangent_stiffness(0, 0) -
                       softening_with_tangent.tangent_stiffness(0, 0)) <
                  1.0e-3,
              "cohesive active-set tangent matches central tangent in pure normal loading");

        const auto closed = evaluate_bilinear_cohesive_law(
            cohesive,
            softening.updated_state,
            n,
            -0.001,
            Eigen::Vector3d::Zero());
        check(std::abs(closed.traction.x() + 2.0) < 1.0e-12,
              "cohesive law uses contact stiffness under crack closure");

        const auto failed_shear = evaluate_bilinear_cohesive_law(
            cohesive,
            CohesiveCrackState{.max_effective_separation = 0.020},
            n,
            0.0,
            Eigen::Vector3d{0.0, 0.010, 0.0});
        check(std::abs(failed_shear.damage - 1.0) < 1.0e-12,
              "cohesive history reaches full damage at critical separation");
        check(std::abs(failed_shear.traction.y() - 1.0) < 1.0e-12,
              "cohesive law keeps only the residual shear transfer after failure");

        auto capped_shear_cohesive = cohesive;
        capped_shear_cohesive.shear_traction_cap = 0.25;
        const auto capped_shear = evaluate_bilinear_cohesive_law(
            capped_shear_cohesive,
            CohesiveCrackState{},
            n,
            0.0,
            Eigen::Vector3d{0.0, 0.010, 0.0});
        check(std::abs(capped_shear.traction.y() - 0.25) < 1.0e-12,
              "cohesive law caps tangential crack transfer consistently");

        capped_shear_cohesive.tangent_mode =
            fall_n::xfem::CohesiveCrackTangentMode::
                active_set_consistent;
        const auto capped_shear_active_tangent = evaluate_bilinear_cohesive_law(
            capped_shear_cohesive,
            CohesiveCrackState{},
            n,
            0.0,
            Eigen::Vector3d{0.0, 0.010, 0.0});
        check(std::abs(
                  capped_shear_active_tangent.tangent_stiffness(1, 1)) <
                  1.0e-10,
              "cohesive active-set tangent freezes capped slip direction");
        check(std::abs(
                  capped_shear_active_tangent.tangent_stiffness(2, 2) -
                  25.0) < 1.0e-10,
              "cohesive active-set tangent keeps transverse capped-slip stiffness");

        const auto crack_band_cohesive =
            make_crack_band_consistent_cohesive_law(
                30000.0,
                12500.0,
                2.0,
                0.06,
                400.0,
                1.0,
                1.0,
                1.0,
                0.15);
        check(std::abs(crack_band_cohesive.normal_stiffness - 75.0) <
                  1.0e-12,
              "cohesive factory maps continuum Ec/lc to interface stiffness");
        check(std::abs(crack_band_cohesive.shear_stiffness - 31.25) <
                  1.0e-12,
              "cohesive factory maps continuum G/lc to interface shear stiffness");
        check(std::abs(crack_band_cohesive.residual_shear_fraction - 0.15) <
                  1.0e-14,
              "cohesive factory preserves residual shear transfer fraction");

        const auto metre_jump_cohesive =
            make_metre_jump_crack_band_consistent_cohesive_law_from_mpa_n_per_mm(
                30000.0,
                12500.0,
                2.0,
                0.06,
                0.40,
                1.0,
                1.0,
                1.0,
                0.15);
        check(std::abs(metre_jump_cohesive.normal_stiffness - 75000.0) <
                  1.0e-10,
              "metre-jump cohesive factory maps Ec/lc to MPa per metre");
        check(std::abs(metre_jump_cohesive.fracture_energy - 6.0e-5) <
                  1.0e-14,
              "metre-jump cohesive factory converts N/mm to MN/m");
        check(std::abs(
                  2.0 * metre_jump_cohesive.fracture_energy /
                      metre_jump_cohesive.tensile_strength -
                  6.0e-5) < 1.0e-14,
              "metre-jump cohesive factory returns delta_f in metres");

        auto contact_cohesive = cohesive;
        contact_cohesive.shear_transfer_law = {
            .kind = fall_n::fracture::CrackShearTransferLawKind::
                compression_gated_opening,
            .residual_ratio = 0.05,
            .large_opening_ratio = 0.05,
            .opening_decay_strain = 0.01,
            .closure_shear_gain = 1.0,
            .max_closed_ratio = 1.0,
            .closure_reference_strain_multiplier = 1.0};
        contact_cohesive.residual_shear_fraction = 0.05;
        const auto open_sliding = evaluate_bilinear_cohesive_law(
            contact_cohesive,
            CohesiveCrackState{.max_effective_separation = 0.020},
            n,
            0.010,
            Eigen::Vector3d{0.0, 0.010, 0.0});
        const auto closed_sliding = evaluate_bilinear_cohesive_law(
            contact_cohesive,
            CohesiveCrackState{.max_effective_separation = 0.020},
            n,
            -0.010,
            Eigen::Vector3d{0.0, 0.010, 0.0});
        check(closed_sliding.traction.y() > open_sliding.traction.y(),
              "XFEM cohesive law can reuse compression-gated shear transfer");
    }

    {
        const BoundedSlipBridgeParameters bounded{
            .initial_stiffness_mn_per_m = 10.0,
            .yield_force_mn = 1.0e-3,
            .hardening_ratio = 0.0,
            .force_cap_mn = 1.0e-3};
        const auto elastic = evaluate_bounded_slip_bridge_law(
            bounded,
            BoundedSlipBridgeState{},
            5.0e-5);
        check(std::abs(elastic.force_mn - 5.0e-4) < 1.0e-14,
              "bounded-slip crack bridge is elastic before yield");
        check(std::abs(elastic.tangent_mn_per_m - 10.0) < 1.0e-14,
              "bounded-slip crack bridge exposes its elastic tangent");

        const auto plastic = evaluate_bounded_slip_bridge_law(
            bounded,
            BoundedSlipBridgeState{},
            2.0e-4);
        check(std::abs(plastic.force_mn - 1.0e-3) < 1.0e-14,
              "bounded-slip crack bridge caps the localized transfer force");
        check(std::abs(plastic.tangent_mn_per_m) < 1.0e-14,
              "bounded-slip crack bridge returns a plastic tangent at the cap");
    }

    {
        Domain<3> domain;
        make_unit_hex8_domain(domain);

        ContinuumIsotropicElasticMaterial mat_site{200.0, 0.25};
        Material<ThreeDimensionalMaterial> material{
            mat_site,
            ElasticUpdate{}};
        std::vector<bool> enriched_mask(domain.num_nodes(), false);
        enriched_mask[0] = true;
        enriched_mask[1] = true;

        Model<ThreeDimensionalMaterial> model{
            domain,
            material,
            [&](Domain<3>& dof_domain) {
                const auto count =
                    apply_shifted_heaviside_dof_layout(
                        dof_domain,
                        enriched_mask);
                check(count == 2,
                      "XFEM DOF hook marks exactly the requested enriched nodes");
            }};
        model.setup();

        check(domain.node(0).num_dof() == 6,
              "enriched host node carries standard plus Heaviside DOFs");
        check(domain.node(2).num_dof() == 3,
              "unenriched host node keeps the standard displacement block");
        check(node_has_shifted_heaviside_enrichment(domain.node(1)),
              "XFEM layout query detects enriched nodes");
        check(!node_has_shifted_heaviside_enrichment(domain.node(7)),
              "XFEM layout query rejects standard nodes");

        PetscSection section = nullptr;
        ISLocalToGlobalMapping local_to_global = nullptr;
        DMGetLocalSection(model.get_plex(), &section);
        DMGetLocalToGlobalMapping(model.get_plex(), &local_to_global);

        PetscInt node0_dof = 0;
        PetscInt node2_dof = 0;
        PetscSectionGetDof(section, domain.node(0).sieve_id.value(), &node0_dof);
        PetscSectionGetDof(section, domain.node(2).sieve_id.value(), &node2_dof);
        check(node0_dof == 6,
              "PETSc section stores six DOFs for an enriched node");
        check(node2_dof == 3,
              "PETSc section stores three DOFs for a standard node");

        const auto enriched_x = static_cast<PetscInt>(
            shifted_heaviside_enriched_component<3>(0));
        const PetscInt local_index = petsc_local_dof_index(
            section,
            domain.node(0).sieve_id.value(),
            enriched_x);
        const PetscInt global_index = petsc_global_dof_index(
            section,
            local_to_global,
            domain.node(0).sieve_id.value(),
            enriched_x);
        check(local_index >= 0,
              "XFEM helper maps enriched component to a PETSc local index");
        check(global_index >= 0,
              "XFEM helper maps enriched component to a PETSc global index");

        auto& element = model.elements().front();
        const auto& standard_dofs = element.get_dof_indices();
        check(standard_dofs.size() == 24,
              "standard continuum kernel gathers only the leading displacement block");
    }

    {
        Domain<3> domain;
        make_unit_hex8_domain(domain);

        ContinuumIsotropicElasticMaterial mat_site{200.0, 0.25};
        Material<ThreeDimensionalMaterial> material{
            mat_site,
            ElasticUpdate{}};
        const PlaneCrackLevelSet mid_height_crack{
            Eigen::Vector3d{0.0, 0.0, 0.5},
            Eigen::Vector3d::UnitZ()};
        const BilinearCohesiveLawParameters cohesive{
            .normal_stiffness = 1000.0,
            .shear_stiffness = 500.0,
            .tensile_strength = 10.0,
            .fracture_energy = 0.10,
            .mode_mixity_weight = 1.0,
            .compression_stiffness = 2000.0,
            .residual_shear_fraction = 0.10};

        using XFEMElement =
            fall_n::xfem::ShiftedHeavisideSolidElement<
                ThreeDimensionalMaterial>;
        std::vector<XFEMElement> elements;
        for (auto& geometry : domain.elements()) {
            elements.emplace_back(
                &geometry,
                material,
                mid_height_crack,
                cohesive);
        }

        using XFEMModel = Model<
            ThreeDimensionalMaterial,
            continuum::SmallStrain,
            3,
            SingleElementPolicy<XFEMElement>>;
        XFEMModel model{domain, std::move(elements)};
        model.setup();

        auto& element = model.elements().front();
        check(element.is_cut_by_crack(),
              "global XFEM solid element detects the cut cell");
        check(element.get_dof_indices().size() == 48,
              "global XFEM solid element gathers standard plus enriched DOFs");

        Eigen::VectorXd u = Eigen::VectorXd::Zero(48);
        for (Eigen::Index local = 5; local < 48; local += 6) {
            u[local] = 1.0e-3;
        }
        const auto f = element.compute_internal_force_vector(u);
        const auto K = element.compute_tangent_stiffness_matrix(u);
        check(f.size() == 48,
              "global XFEM solid residual has one entry per gathered DOF");
        check(K.rows() == 48 && K.cols() == 48,
              "global XFEM solid tangent has the enriched algebraic size");
        check(f.tail(24).norm() > 0.0,
              "global XFEM cohesive/volumetric residual acts on enriched DOFs");
        check(K.bottomRightCorner(24, 24).norm() > 0.0,
              "global XFEM tangent couples enriched DOFs");
    }

    {
        Domain<3> domain;
        make_unit_hex8_domain(domain);

        ContinuumIsotropicElasticMaterial mat_site{200.0, 0.25};
        Material<ThreeDimensionalMaterial> material{
            mat_site,
            ElasticUpdate{}};
        std::vector<XFEMCrackPlane> cracks{
            XFEMCrackPlane{
                PlaneCrackLevelSet{
                    Eigen::Vector3d{0.0, 0.0, 0.35},
                    Eigen::Vector3d{0.20, 0.30, 1.0}},
                10,
                2,
                XFEMCrackPlaneSource::prescribed,
                7,
                0.25,
                true},
            XFEMCrackPlane{
                PlaneCrackLevelSet{
                    Eigen::Vector3d{0.50, 0.50, 0.50},
                    Eigen::Vector3d{1.0, 0.0, 1.0}},
                25,
                5,
                XFEMCrackPlaneSource::prescribed,
                9,
                0.50,
                true}};
        const BilinearCohesiveLawParameters cohesive{
            .normal_stiffness = 1000.0,
            .shear_stiffness = 500.0,
            .tensile_strength = 10.0,
            .fracture_energy = 0.10,
            .mode_mixity_weight = 1.0,
            .compression_stiffness = 2000.0,
            .residual_shear_fraction = 0.10};

        using XFEMElement =
            fall_n::xfem::ShiftedHeavisideSolidElement<
                ThreeDimensionalMaterial>;
        std::vector<XFEMElement> elements;
        for (auto& geometry : domain.elements()) {
            elements.emplace_back(
                &geometry,
                material,
                cracks,
                cohesive);
        }

        using XFEMModel = Model<
            ThreeDimensionalMaterial,
            continuum::SmallStrain,
            3,
            SingleElementPolicy<XFEMElement>>;
        XFEMModel model{domain, std::move(elements)};
        model.setup();

        auto& element = model.elements().front();
        check(element.get_dof_indices().size() == 72,
              "multi-plane XFEM solid gathers one enriched vector block per plane");
        check(element.crack_planes().size() == 2 &&
                  element.crack_planes()[0].plane_id == 10 &&
                  element.crack_planes()[1].sequence_id == 5,
              "multi-plane XFEM solid preserves explicit descriptor ids");

        Eigen::VectorXd u = Eigen::VectorXd::Zero(72);
        for (Eigen::Index node = 0; node < 8; ++node) {
            u[9 * node + 5] = 1.0e-4; // plane 1, z jump
            u[9 * node + 6] = 2.0e-4; // plane 2, x jump
        }
        const auto f = element.compute_internal_force_vector(u);
        const auto K = element.compute_tangent_stiffness_matrix(u);
        check(f.size() == 72 && K.rows() == 72 && K.cols() == 72,
              "multi-plane XFEM residual and tangent use the expanded algebraic size");
        check(f.tail(48).norm() > 0.0,
              "multi-plane XFEM cohesive/volumetric residual acts on both enriched blocks");

        const auto records = element.collect_crack_records(model.state_vector());
        bool saw_first_plane = false;
        bool saw_second_plane = false;
        bool saw_oblique_normal = false;
        bool saw_activation_metadata = false;
        for (const auto& record : records) {
            saw_first_plane =
                saw_first_plane ||
                (record.plane_id == 10 && record.sequence_id == 2);
            saw_second_plane =
                saw_second_plane ||
                (record.plane_id == 25 && record.sequence_id == 5);
            saw_oblique_normal =
                saw_oblique_normal ||
                (record.plane_id == 25 &&
                 std::abs(std::abs(record.normal_1.z()) - 1.0) > 1.0e-6);
            saw_activation_metadata =
                saw_activation_metadata ||
                (record.plane_id == 10 &&
                 record.activation_step == 7 &&
                 std::abs(record.activation_time - 0.25) < 1.0e-14 &&
                 record.source_id ==
                     fall_n::xfem::source_id(
                         XFEMCrackPlaneSource::prescribed));
        }
        check(saw_first_plane && saw_second_plane,
              "multi-plane XFEM crack records preserve explicit plane ids and sequence ids");
        check(saw_oblique_normal,
              "multi-plane XFEM crack records preserve arbitrary plane orientation");
        check(saw_activation_metadata,
              "multi-plane XFEM crack records preserve activation metadata");
    }

    {
        Domain<3> domain;
        make_unit_hex8_domain(domain);

        ContinuumIsotropicElasticMaterial mat_site{1.0e-6, 0.25};
        Material<ThreeDimensionalMaterial> material{
            mat_site,
            ElasticUpdate{}};
        const PlaneCrackLevelSet mid_height_crack{
            Eigen::Vector3d{0.0, 0.0, 0.5},
            Eigen::Vector3d::UnitZ()};
        const BilinearCohesiveLawParameters cohesive{
            .normal_stiffness = 1000.0,
            .shear_stiffness = 500.0,
            .tensile_strength = 10.0,
            .fracture_energy = 0.10,
            .mode_mixity_weight = 1.0,
            .compression_stiffness = 2000.0,
            .residual_shear_fraction = 0.10};
        fall_n::xfem::ShiftedHeavisideSolidOptions options;
        options.cohesive_surface_tangent_mode =
            fall_n::xfem::ShiftedHeavisideSolidOptions::
                CohesiveSurfaceTangentMode::
                    finite_difference_surface_frame;
        options.cohesive_traction_measure_kind =
            fall_n::xfem::ShiftedHeavisideCohesiveTractionMeasureKind::
                current_spatial;
        options.cohesive_surface_tangent_relative_step = 0.0;
        options.cohesive_surface_tangent_absolute_step = 1.0e-7;

        using XFEMElement =
            fall_n::xfem::ShiftedHeavisideSolidElement<
                ThreeDimensionalMaterial,
                continuum::TotalLagrangian>;
        std::vector<XFEMElement> elements;
        for (auto& geometry : domain.elements()) {
            elements.emplace_back(
                &geometry,
                material,
                mid_height_crack,
                cohesive,
                options);
        }

        using XFEMModel = Model<
            ThreeDimensionalMaterial,
            continuum::TotalLagrangian,
            3,
            SingleElementPolicy<XFEMElement>>;
        XFEMModel model{domain, std::move(elements)};
        model.setup();

        auto& element = model.elements().front();
        Eigen::VectorXd u = Eigen::VectorXd::Zero(48);
        for (Eigen::Index local = 5; local < 48; local += 6) {
            u[local] = 1.0e-4;
        }
        const auto K = element.compute_cohesive_tangent_stiffness_matrix(u);
        Eigen::VectorXd direction = Eigen::VectorXd::Zero(48);
        direction[6] = 1.0;
        constexpr double h = 1.0e-7;
        const auto f_plus =
            element.compute_cohesive_internal_force_vector(u + h * direction);
        const auto f_minus =
            element.compute_cohesive_internal_force_vector(u - h * direction);
        const Eigen::VectorXd fd_column = (f_plus - f_minus) / (2.0 * h);
        const double mismatch =
            (K.col(6) - fd_column).norm() /
            std::max(1.0e-14, fd_column.norm());
        check(K.col(6).norm() > 1.0e-10,
              "TL XFEM finite-difference surface tangent sees standard-DOF area sensitivity");
        check(mismatch < 1.0e-8,
              "TL XFEM finite-difference surface tangent matches cohesive residual derivative");

        Domain<3> nanson_domain;
        make_unit_hex8_domain(nanson_domain);
        auto nanson_options = options;
        nanson_options.cohesive_surface_tangent_mode =
            fall_n::xfem::ShiftedHeavisideSolidOptions::
                CohesiveSurfaceTangentMode::
                    nanson_geometric_surface_frame;
        std::vector<XFEMElement> nanson_elements;
        for (auto& geometry : nanson_domain.elements()) {
            nanson_elements.emplace_back(
                &geometry,
                material,
                mid_height_crack,
                cohesive,
                nanson_options);
        }
        XFEMModel nanson_model{nanson_domain, std::move(nanson_elements)};
        nanson_model.setup();
        auto& nanson_element = nanson_model.elements().front();
        const auto K_nanson =
            nanson_element.compute_cohesive_tangent_stiffness_matrix(u);
        const double nanson_mismatch =
            (K_nanson.col(6) - fd_column).norm() /
            std::max(1.0e-14, fd_column.norm());
        check(nanson_mismatch < 1.0e-5,
              "TL XFEM Nanson-geometric surface tangent matches the residual derivative for area-sensitive columns");

        Domain<3> nominal_domain;
        make_unit_hex8_domain(nominal_domain);
        auto nominal_options = options;
        nominal_options.cohesive_traction_measure_kind =
            fall_n::xfem::ShiftedHeavisideCohesiveTractionMeasureKind::
                reference_nominal;
        std::vector<XFEMElement> nominal_elements;
        for (auto& geometry : nominal_domain.elements()) {
            nominal_elements.emplace_back(
                &geometry,
                material,
                mid_height_crack,
                cohesive,
                nominal_options);
        }
        XFEMModel nominal_model{nominal_domain, std::move(nominal_elements)};
        nominal_model.setup();
        auto& nominal_element = nominal_model.elements().front();
        const auto K_nominal =
            nominal_element.compute_cohesive_tangent_stiffness_matrix(u);
        const auto f_nominal_plus =
            nominal_element.compute_cohesive_internal_force_vector(
                u + h * direction);
        const auto f_nominal_minus =
            nominal_element.compute_cohesive_internal_force_vector(
                u - h * direction);
        const Eigen::VectorXd fd_nominal_column =
            (f_nominal_plus - f_nominal_minus) / (2.0 * h);
        check(K_nominal.col(6).norm() < 1.0e-10 &&
                  fd_nominal_column.norm() < 1.0e-10,
              "TL XFEM reference-nominal traction removes current-area sensitivity from standard DOFs");
    }

    {
        Domain<3> domain;
        make_unit_hex8_domain(domain);

        MaterialInstance<ElasticRelation<UniaxialMaterial>> elastic_steel{
            200000.0};
        Material<UniaxialMaterial> steel{
            std::move(elastic_steel),
            ElasticUpdate{}};
        ShiftedHeavisideCrackCrossingRebarElement bridge{
            &domain.element(0),
            steel,
            std::array<double, 3>{0.0, 0.0, 0.0},
            Eigen::Vector3d::UnitZ(),
            1.0e-4,
            0.10};
        bridge.set_num_dof_in_nodes();

        int next_dof = 0;
        for (auto& node : domain.nodes()) {
            std::vector<PetscInt> dofs(node.num_dof());
            for (auto& dof : dofs) {
                dof = next_dof++;
            }
            node.set_dof_index(dofs);
        }

        check(domain.node(0).num_dof() == 6,
              "crack-crossing rebar bridge declares enriched host DOFs");
        check(bridge.get_dof_indices().size() == 8,
              "crack-crossing rebar bridge gathers only axial enriched components");

        Eigen::VectorXd u =
            Eigen::VectorXd::Constant(
                static_cast<Eigen::Index>(bridge.get_dof_indices().size()),
                5.0e-5);
        const auto f = bridge.compute_internal_force_vector(u);
        const auto K = bridge.compute_tangent_stiffness_matrix(u);
        check(f.norm() > 0.0,
              "crack-crossing rebar bridge contributes internal force");
        check(K.rows() == f.size() && K.cols() == f.size(),
              "crack-crossing rebar bridge tangent matches local residual size");
        check(K.norm() > 0.0,
              "crack-crossing rebar bridge contributes a material tangent");

        const auto gauss = bridge.collect_gauss_fields(u);
        check(gauss.size() == 1 &&
                  std::abs(gauss.front().stress[0]) > 0.0,
              "crack-crossing rebar bridge exports uniaxial steel stress");
    }

    {
        Domain<3> domain;
        make_unit_hex8_domain(domain);

        ShiftedHeavisideCrackCrossingRebarElement bridge{
            &domain.element(0),
            std::array<double, 3>{0.0, 0.0, 0.0},
            Eigen::Vector3d::UnitZ(),
            1.0e-4,
            0.10,
            BoundedSlipBridgeParameters{
                .initial_stiffness_mn_per_m = 2.0,
                .yield_force_mn = 1.0e-4,
                .hardening_ratio = 0.0,
                .force_cap_mn = 1.0e-4}};
        bridge.set_num_dof_in_nodes();

        int next_dof = 0;
        for (auto& node : domain.nodes()) {
            std::vector<PetscInt> dofs(node.num_dof());
            for (auto& dof : dofs) {
                dof = next_dof++;
            }
            node.set_dof_index(dofs);
        }

        Eigen::VectorXd small_u =
            Eigen::VectorXd::Constant(
                static_cast<Eigen::Index>(bridge.get_dof_indices().size()),
                1.0e-5);
        const auto K = bridge.compute_tangent_stiffness_matrix(small_u);
        check(K.norm() > 0.0,
              "bounded-slip crack bridge contributes an elastic tangent before yield");

        Eigen::VectorXd large_u =
            Eigen::VectorXd::Constant(
                static_cast<Eigen::Index>(bridge.get_dof_indices().size()),
                1.0e-3);
        const auto f = bridge.compute_internal_force_vector(large_u);
        const auto gauss = bridge.collect_gauss_fields(large_u);
        check(f.norm() > 0.0,
              "bounded-slip crack bridge contributes capped internal force");
        check(gauss.size() == 1 &&
                  std::abs(gauss.front().stress[0] - 1.0) < 1.0e-12,
              "bounded-slip crack bridge reports force-over-area equivalent stress");
    }

    {
        Domain<3> domain;
        make_unit_hex8_domain(domain);

        const BoundedSlipBridgeParameters bridge_law{
            .initial_stiffness_mn_per_m = 1.0,
            .yield_force_mn = 1.0,
            .hardening_ratio = 0.0,
            .force_cap_mn = 1.0};
        ShiftedHeavisideCrackCrossingRebarElement bridge{
            &domain.element(0),
            std::array<double, 3>{0.0, 0.0, 0.0},
            Eigen::Vector3d::UnitX(),
            1.0e-4,
            0.10,
            bridge_law,
            ShiftedHeavisideCrackCrossingRebarElement::Options{
                .axis_frame_kind =
                    CrackCrossingRebarAxisFrameKind::corotational_host,
                .include_corotational_host_axis_tangent = true}};
        bridge.set_num_dof_in_nodes();

        int next_dof = 0;
        for (auto& node : domain.nodes()) {
            std::vector<PetscInt> dofs(node.num_dof());
            for (auto& dof : dofs) {
                dof = next_dof++;
            }
            node.set_dof_index(dofs);
        }

        check(bridge.get_dof_indices().size() == 48,
              "corotational crack bridge gathers host-frame plus enriched DOFs");

        const double angle = 0.50 * std::numbers::pi;
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        R(0, 0) = std::cos(angle);
        R(0, 1) = -std::sin(angle);
        R(1, 0) = std::sin(angle);
        R(1, 1) = std::cos(angle);

        Eigen::VectorXd u = Eigen::VectorXd::Zero(
            static_cast<Eigen::Index>(bridge.get_dof_indices().size()));
        for (std::size_t node = 0; node < domain.num_nodes(); ++node) {
            const auto& p = domain.node(node);
            const Eigen::Vector3d X{p.coord(0), p.coord(1), p.coord(2)};
            const Eigen::Vector3d ui = R * X - X;
            for (std::size_t c = 0; c < 3; ++c) {
                u[static_cast<Eigen::Index>(6 * node + c)] =
                    ui[static_cast<Eigen::Index>(c)];
            }
        }
        for (std::size_t node = 0; node < domain.num_nodes(); ++node) {
            u[static_cast<Eigen::Index>(6 * node + 4)] = 1.0e-4;
        }

        const auto f = bridge.compute_internal_force_vector(u);
        const auto K = bridge.compute_tangent_stiffness_matrix(u);
        double host_residual_norm = 0.0;
        double enriched_residual_norm = 0.0;
        double enriched_by_host_tangent_norm = 0.0;
        for (std::size_t node = 0; node < domain.num_nodes(); ++node) {
            host_residual_norm +=
                f.segment(static_cast<Eigen::Index>(6 * node), 3)
                    .squaredNorm();
            enriched_residual_norm +=
                f.segment(static_cast<Eigen::Index>(6 * node + 3), 3)
                    .squaredNorm();
            for (std::size_t other = 0; other < domain.num_nodes();
                 ++other) {
                enriched_by_host_tangent_norm +=
                    K.block(static_cast<Eigen::Index>(6 * node + 3),
                            static_cast<Eigen::Index>(6 * other),
                            3,
                            3)
                        .squaredNorm();
            }
        }
        check(enriched_residual_norm > 0.0,
              "corotational crack bridge projects slip on the rotated host axis");
        check(enriched_by_host_tangent_norm > 0.0,
              "corotational crack bridge tangent sees host-frame axis changes");
        check(host_residual_norm == 0.0,
              "corotational crack bridge does not create artificial host-frame residual");
    }

    {
        Domain<3> domain;
        make_unit_hex8_domain(domain);

        ContinuumIsotropicElasticMaterial mat_site{200.0, 0.25};
        Material<ThreeDimensionalMaterial> material{
            mat_site,
            ElasticUpdate{}};
        const PlaneCrackLevelSet mid_height_crack{
            Eigen::Vector3d{0.0, 0.0, 0.5},
            Eigen::Vector3d::UnitZ()};
        const BilinearCohesiveLawParameters cohesive{
            .normal_stiffness = 1000.0,
            .shear_stiffness = 500.0,
            .tensile_strength = 10.0,
            .fracture_energy = 0.10,
            .mode_mixity_weight = 1.0,
            .compression_stiffness = 2000.0,
            .residual_shear_fraction = 0.10};

        using XFEMElement =
            fall_n::xfem::ShiftedHeavisideSolidElement<
                ThreeDimensionalMaterial>;
        std::vector<XFEMElement> elements;
        for (auto& geometry : domain.elements()) {
            elements.emplace_back(
                &geometry,
                material,
                mid_height_crack,
                cohesive);
        }

        using XFEMPolicy = SingleElementPolicy<XFEMElement>;
        using XFEMModel = Model<
            ThreeDimensionalMaterial,
            continuum::SmallStrain,
            3,
            XFEMPolicy>;
        XFEMModel model{domain, std::move(elements)};

        for (std::size_t node : {0ul, 1ul, 2ul, 3ul}) {
            model.fix_node(node);
        }
        std::vector<DisplacementControl::PrescribedDOF> controls;
        for (std::size_t node : {4ul, 5ul, 6ul, 7ul}) {
            model.constrain_dof(node, 0, 0.0);
            model.constrain_dof(node, 1, 0.0);
            model.constrain_dof(node, 2, 0.0);
            controls.push_back({node, 2, 1.0e-4});
        }
        model.setup();

        NonlinearAnalysis<
            ThreeDimensionalMaterial,
            continuum::SmallStrain,
            3,
            XFEMPolicy> analysis{&model};
        const bool ok = analysis.solve_incremental(
            2,
            2,
            DisplacementControl{std::move(controls)});
        check(ok,
              "global XFEM solid participates in a PETSc/SNES displacement-control solve");
    }

    std::cout << passed_tests << "/" << total_tests << " tests passed\n";
    const int exit_code = passed_tests == total_tests ? 0 : 1;
    PetscFinalize();
    return exit_code;
}
