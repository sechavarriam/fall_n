#include "src/validation/TableCyclicValidationFE2Setup.hh"

#include "src/validation/TableCyclicValidationSubmodelFactories.hh"

#include <filesystem>

namespace fall_n::table_cyclic_validation {

FE2CaseContext build_fe2_case_context(
    bool two_way,
    const std::string& out_dir,
    const CyclicValidationRunConfig& cfg)
{
    FE2CaseContext ctx;
    ctx.domain = std::make_unique<Domain<3>>();
    ctx.coordinator = std::make_unique<MultiscaleCoordinator>();
    auto& domain = *ctx.domain;
    auto& coordinator = *ctx.coordinator;

    PetscInt tag = 0;

    domain.add_node(0, 0.0, 0.0, 0.0);
    domain.add_node(1,  LX, 0.0, 0.0);
    domain.add_node(2, 0.0,  LY, 0.0);
    domain.add_node(3,  LX,  LY, 0.0);

    {
        PetscInt nid = 4;
        for (int jy = 0; jy < 4; ++jy) {
            for (int jx = 0; jx < 4; ++jx) {
                domain.add_node(
                    nid++, LX * jx / 3.0, LY * jy / 3.0, H);
            }
        }
    }

    const std::array<std::pair<PetscInt, PetscInt>, 4> col_pairs = {{
        {0, 4}, {1, 7}, {2, 16}, {3, 19}
    }};
    for (auto [base, top] : col_pairs) {
        PetscInt conn[2] = {base, top};
        auto& g = domain.make_element<LagrangeElement3D<2>>(
            GaussLegendreCellIntegrator<2>{}, tag++, conn);
        g.set_physical_group("Columns");
    }

    {
        PetscInt slab_conn[16];
        for (int i = 0; i < 16; ++i) {
            slab_conn[i] = 4 + i;
        }
        auto& g = domain.make_element<LagrangeElement3D<4, 4>>(
            GaussLegendreCellIntegrator<4, 4>{}, tag++, slab_conn);
        g.set_physical_group("Slabs");
    }

    domain.assemble_sieve();

    const auto col_mat = make_rc_column_section({
        .b = COL_B, .h = COL_H, .cover = COL_CVR,
        .bar_diameter = COL_BAR, .tie_spacing = COL_TIE,
        .fpc = COL_FPC, .nu = NU_RC,
        .steel_E = STEEL_E, .steel_fy = STEEL_FY, .steel_b = STEEL_B,
        .tie_fy = TIE_FY,
    });

    MindlinShellMaterial slab_relation{SLAB_E, NU_RC, SLAB_T};
    const auto slab_mat = Material<MindlinReissnerShell3D>{
        slab_relation, ElasticUpdate{}};

    auto builder = StructuralModelBuilder<
        BeamElemT2, ShellElemT,
        TimoshenkoBeam3D, MindlinReissnerShell3D>{};
    builder.set_frame_material("Columns", col_mat);
    builder.set_shell_material("Slabs", slab_mat);

    auto elements = builder.build_elements(domain);
    ctx.model = std::make_unique<StructModel>(domain, std::move(elements));
    auto& model = *ctx.model;

    model.fix_z(0.0);

    const std::array<std::size_t, 4> slab_corners = {4, 7, 16, 19};
    for (auto nid : slab_corners) {
        model.constrain_dof(nid, 0, 0.0);
    }

    model.setup();

    std::println("  Nodes: {}  Elements: {}", domain.num_nodes(),
                 model.elements().size());

    auto extract_beam_kinematics = [&model](std::size_t e_idx) -> ElementKinematics {
        const auto& se = model.elements()[e_idx];
        const auto* beam = se.as<BeamElemT2>();
        const auto u_e = se.extract_element_dofs(model.state_vector());

        auto kin_A = extract_section_kinematics(*beam, u_e, -1.0);
        auto kin_B = extract_section_kinematics(*beam, u_e, +1.0);
        kin_A.E = EC_COL;
        kin_A.G = GC_COL;
        kin_A.nu = NU_RC;
        kin_B.E = EC_COL;
        kin_B.G = GC_COL;
        kin_B.nu = NU_RC;

        ElementKinematics ek;
        ek.element_id = e_idx;
        ek.kin_A = kin_A;
        ek.kin_B = kin_B;
        ek.endpoint_A = beam->geometry().map_local_point(std::array{-1.0});
        ek.endpoint_B = beam->geometry().map_local_point(std::array{+1.0});
        ek.up_direction = std::array<double, 3>{1.0, 0.0, 0.0};
        return ek;
    };

    for (std::size_t eid = 0; eid < 4; ++eid) {
        coordinator.add_critical_element(extract_beam_kinematics(eid));
    }

    const double cvr = COL_CVR;
    const double bar_d = COL_BAR;
    const double bar_a = std::numbers::pi / 4.0 * bar_d * bar_d;
    const double y0 = -COL_B / 2.0 + cvr + bar_d / 2.0;
    const double y1 =  COL_B / 2.0 - cvr - bar_d / 2.0;
    const double z0 = -COL_H / 2.0 + cvr + bar_d / 2.0;
    const double z1 =  COL_H / 2.0 - cvr - bar_d / 2.0;

    std::vector<SubModelSpec::RebarBar> bars = {
        {y0, z0, bar_a, bar_d}, {y1, z0, bar_a, bar_d},
        {y0, z1, bar_a, bar_d}, {y1, z1, bar_a, bar_d},
        {0.0, z0, bar_a, bar_d}, {0.0, z1, bar_a, bar_d},
        {y0, 0.0, bar_a, bar_d}, {y1, 0.0, bar_a, bar_d},
    };

    coordinator.build_sub_models(SubModelSpec{
        .section_width = COL_B,
        .section_height = COL_H,
        .nx = SUB_NX,
        .ny = SUB_NY,
        .nz = SUB_NZ,
        .hex_order = HexOrder::Quadratic,
        .rebar_bars = std::move(bars),
        .rebar_E = STEEL_E,
        .rebar_fy = STEEL_FY,
        .rebar_b = STEEL_B,
    });

    {
        const auto rpt = coordinator.report();
        std::println("  Sub-models: {}  ({} hex/sub, {} nodes/sub)",
                     rpt.num_elements, rpt.total_elements, rpt.total_nodes);
    }

    const std::string evol_dir = out_dir + "/sub_models";
    std::filesystem::create_directories(evol_dir);

    std::vector<NonlinearSubModelEvolver> nl_evolvers;
    for (auto& sub : coordinator.sub_models()) {
        nl_evolvers.emplace_back(
            sub,
            COL_FPC,
            make_table_submodel_concrete_factory(sub, cfg),
            make_table_submodel_rebar_factory(),
            evol_dir,
            cfg.submodel_output_interval);
        nl_evolvers.back().set_incremental_params(
            cfg.submodel_increment_steps, cfg.submodel_max_bisections);
        nl_evolvers.back().set_penalty_alpha(EC_COL * 10.0);
        nl_evolvers.back().set_arc_length_threshold(
            cfg.submodel_arc_length_threshold);
        nl_evolvers.back().enable_arc_length(
            cfg.submodel_enable_arc_length_from_start);
        nl_evolvers.back().set_adaptive_substepping_limits(
            cfg.submodel_adaptive_max_substeps,
            cfg.submodel_adaptive_max_bisections);
        nl_evolvers.back().set_adaptive_tail_rescue_policy(
            cfg.submodel_tail_rescue_attempts,
            cfg.submodel_tail_rescue_progress_threshold,
            cfg.submodel_tail_rescue_substep_bonus,
            cfg.submodel_tail_rescue_bisection_bonus,
            cfg.submodel_tail_rescue_initial_fraction);
        nl_evolvers.back().set_snes_params(
            cfg.submodel_snes_max_it,
            cfg.submodel_snes_atol,
            cfg.submodel_snes_rtol);
        nl_evolvers.back().set_min_crack_opening(cfg.min_crack_opening);
    }

    MultiscaleModel<ValidationMacroBridge, NonlinearSubModelEvolver> ms_model{
        ValidationMacroBridge{model}};

    for (auto& ev : nl_evolvers) {
        const auto eid = ev.parent_element_id();
        ms_model.register_local_model(
            ms_model.macro_bridge().default_site(eid),
            std::move(ev));
    }

    std::unique_ptr<CouplingAlgorithm> algorithm = two_way
        ? std::unique_ptr<CouplingAlgorithm>(
            std::make_unique<IteratedTwoWayFE2>(
                cfg.max_staggered_iterations))
        : std::unique_ptr<CouplingAlgorithm>(
            std::make_unique<OneWayDownscaling>());

    std::println("  Sub-models will ramp at first coupling step...");

    PetscOptionsSetValue(nullptr, "-snes_linesearch_type", "bt");
    PetscOptionsSetValue(nullptr, "-snes_max_it", "100");
    PetscOptionsSetValue(nullptr, "-ksp_type", "preonly");
    PetscOptionsSetValue(nullptr, "-pc_type", "lu");

    ctx.nl = std::make_unique<ValidationNlAnalysis>(&model);
    ctx.analysis = std::make_unique<ValidationAnalysis>(
        *ctx.nl,
        std::move(ms_model),
        std::move(algorithm),
        std::make_unique<ForceAndTangentConvergence>(
            cfg.staggered_tol, cfg.staggered_tol),
        std::make_unique<ConstantRelaxation>(cfg.staggered_relaxation),
        ValidationMicroExecutor{});
    ctx.analysis->set_coupling_start_step(COUPLING_START_STEP);
    ctx.analysis->set_section_dimensions(COL_B, COL_H);
    ctx.analysis->set_predictor_admissibility_filter(
        cfg.predictor_admissibility_min_symmetric_eigenvalue,
        cfg.predictor_admissibility_backtrack_attempts,
        cfg.predictor_admissibility_backtrack_factor);
    ctx.analysis->set_macro_step_cutback(
        cfg.macro_step_cutback_attempts,
        cfg.macro_step_cutback_factor);
    ctx.analysis->set_macro_failure_backtracking(
        cfg.macro_failure_backtrack_attempts,
        cfg.macro_failure_backtrack_factor);

    ctx.base_nodes = {0, 1, 2, 3};
    return ctx;
}

} // namespace fall_n::table_cyclic_validation
