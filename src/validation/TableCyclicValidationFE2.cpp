#include "src/validation/TableCyclicValidationSupport.hh"

#include <chrono>
#include <limits>

namespace fall_n::table_cyclic_validation {

namespace {

[[nodiscard]] double csv_scalar_or_nan(bool available, double value)
{
    return available ? value : std::numeric_limits<double>::quiet_NaN();
}

} // namespace

std::vector<StepRecord>
run_case_fe2(bool two_way, const std::string& out_dir,
             const CyclicValidationRunConfig& cfg)
{
    const char* label = two_way
        ? "5 (Iterated two-way FE²)"
        : "4 (One-way downscaling)";
    std::println("\n  Case {}: Full table + FE² sub-models", label);

    Domain<3> domain;
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
    StructModel model{domain, std::move(elements)};

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

    MultiscaleCoordinator coordinator;
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
            sub, COL_FPC, evol_dir, cfg.submodel_output_interval);
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
        nl_evolvers.back().set_snes_params(
            cfg.submodel_snes_max_it,
            cfg.submodel_snes_atol,
            cfg.submodel_snes_rtol);
        nl_evolvers.back().set_min_crack_opening(cfg.min_crack_opening);
    }

    using MacroBridge = BeamMacroBridge<StructModel, BeamElemT2>;
    MultiscaleModel<MacroBridge, NonlinearSubModelEvolver> ms_model{
        MacroBridge{model}};

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

    NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF,
                      StructPolicy> nl{&model};
    using ValidationMicroExecutor = SerialExecutor;
    using ValidationAnalysisT = MultiscaleAnalysis<
        NonlinearAnalysis<TimoshenkoBeam3D, continuum::SmallStrain, NDOF,
                          StructPolicy>,
        MacroBridge,
        NonlinearSubModelEvolver,
        ValidationMicroExecutor>;
    ValidationAnalysisT analysis(
        nl,
        std::move(ms_model),
        std::move(algorithm),
        std::make_unique<ForceAndTangentConvergence>(
            cfg.staggered_tol, cfg.staggered_tol),
        std::make_unique<ConstantRelaxation>(cfg.staggered_relaxation),
        ValidationMicroExecutor{});
    analysis.set_coupling_start_step(COUPLING_START_STEP);
    analysis.set_section_dimensions(COL_B, COL_H);
    const int protocol_steps = cfg.total_steps();
    const int executed_steps =
        (cfg.max_steps > 0) ? std::min(protocol_steps, cfg.max_steps)
                            : protocol_steps;
    const bool truncated_protocol = executed_steps < protocol_steps;

    std::vector<StepRecord> records;
    records.reserve(static_cast<std::size_t>(executed_steps) + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    const std::vector<std::size_t> base_nodes = {0, 1, 2, 3};
    const auto t0 = std::chrono::steady_clock::now();

    const bool write_global_vtk = cfg.global_output_interval > 0;
    if (write_global_vtk) {
        std::filesystem::create_directories(out_dir + "/vtk");
    }
    std::filesystem::create_directories(out_dir + "/recorders");
    MaxStrainDamageCriterion damage_crit{EPS_YIELD};
    FiberHysteresisRecorder<StructModel> hysteresis_rec{
        damage_crit, classify_table_fiber, {}, 5, 1};
    const std::string global_history_path =
        out_dir + "/recorders/global_history.csv";
    const std::string hysteresis_path = out_dir + "/hysteresis.csv";
    const std::string crack_path =
        out_dir + "/recorders/crack_evolution.csv";
    const std::string solver_path =
        out_dir + "/recorders/solver_diagnostics.csv";

    const std::string global_header =
        "step,p,drift_m,base_shear_MN,peak_damage,"
        "submodel_damage_scalar_available,peak_submodel_damage_scalar,"
        "most_compressive_submodel_sigma_o_max,max_submodel_tau_o_max,"
        "total_cracked_gps,total_cracks,max_opening,fe2_iterations,"
        "max_force_residual_rel,max_force_component_residual_rel,"
        "max_tangent_residual_rel,max_tangent_column_residual_rel\n";
    const std::string hysteresis_header = "step,p,drift_m,base_shear_MN\n";
    std::ostringstream crack_header;
    crack_header
        << "step,p,drift_m,total_cracked_gps,total_cracks,"
           "damage_scalar_available,peak_damage_scalar,"
           "most_compressive_sigma_o_max,max_tau_o_max,max_opening";
    for (std::size_t i = 0; i < analysis.model().num_local_models(); ++i) {
        crack_header << ",sub" << i << "_cracks";
    }
    crack_header << "\n";
    std::ostringstream solver_header;
    write_fe2_solver_diagnostics_header(solver_header, analysis);

    auto rewrite_buffer = [](
        const std::string& path,
        const std::string& header,
        const std::vector<std::string>& rows)
    {
        std::ofstream os(path, std::ios::trunc);
        os << header;
        for (const auto& row : rows) {
            os << row;
        }
    };

    std::vector<std::string> global_rows;
    std::vector<std::string> hysteresis_rows;
    std::vector<std::string> crack_rows;
    std::vector<std::string> solver_rows;
    {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(8)
            << 0 << "," << 0.0 << "," << 0.0 << "," << 0.0 << "\n";
        hysteresis_rows.push_back(oss.str());
    }

    rewrite_buffer(global_history_path, global_header, global_rows);
    rewrite_buffer(hysteresis_path, hysteresis_header, hysteresis_rows);
    rewrite_buffer(crack_path, crack_header.str(), crack_rows);
    rewrite_buffer(solver_path, solver_header.str(), solver_rows);
    PVDWriter pvd_fe2{out_dir + "/vtk/table_fe2"};

    auto beam_profile =
        fall_n::reconstruction::RectangularSectionProfile<2>{COL_B, COL_H};
    auto shell_profile =
        fall_n::reconstruction::ShellThicknessProfile<3>{};

    auto scheme = make_control(
        [&slab_corners, &cfg]
        (double p, Vec /*f_full*/, Vec f_ext, StructModel* m) {
            VecSet(f_ext, 0.0);
            const double d = cfg.displacement(p);
            for (auto nid : slab_corners) {
                m->update_imposed_value(nid, 0, d);
            }
        });

    nl.set_step_callback({});
    hysteresis_rec.on_analysis_start(model);
    nl.begin_incremental(protocol_steps, cfg.max_bisections, scheme);

    auto retune_local_models = [&](int restart_attempt) {
        const int increment_steps =
            cfg.submodel_increment_steps
            + restart_attempt * cfg.restart_submodel_increment_step_bonus;
        const int max_bisections =
            cfg.submodel_max_bisections
            + restart_attempt * cfg.restart_submodel_bisection_bonus;
        const int adaptive_substeps =
            cfg.submodel_adaptive_max_substeps
            + restart_attempt * cfg.restart_adaptive_substep_bonus;
        const int adaptive_bisections =
            cfg.submodel_adaptive_max_bisections
            + restart_attempt * cfg.restart_adaptive_bisection_bonus;
        const int snes_max_it =
            cfg.submodel_snes_max_it
            + restart_attempt * cfg.restart_snes_max_it_bonus;

        for (auto& ev : analysis.model().local_models()) {
            ev.set_incremental_params(increment_steps, max_bisections);
            ev.set_adaptive_substepping_limits(
                adaptive_substeps, adaptive_bisections);
            ev.set_snes_params(
                snes_max_it,
                cfg.submodel_snes_atol,
                cfg.submodel_snes_rtol);
        }
    };

    struct TurningPointFrame {
        typename ValidationAnalysisT::RestartBundle analysis{};
        int step{0};
        std::size_t record_count{0};
        std::size_t global_rows_count{0};
        std::size_t hysteresis_rows_count{0};
        std::size_t crack_rows_count{0};
        std::size_t solver_rows_count{0};
        int restart_attempts{0};
        bool valid{false};
    } last_turning_point;

    const bool restart_capable =
        cfg.enable_turning_point_checkpoints && !write_global_vtk;

    bool ok = true;
    int step = 1;
    while (step <= executed_steps) {
        if (!analysis.step()) {
            const auto& report = analysis.last_report();
            const double p = nl.current_time();
            const double d = cfg.displacement(p);
            if (restart_capable
                && last_turning_point.valid
                && last_turning_point.step < step
                && last_turning_point.restart_attempts
                       < cfg.max_turning_point_restarts)
            {
                ++last_turning_point.restart_attempts;
                std::println(
                    "    FE2 step {} failed after turning point {}. "
                    "Restarting from checkpoint (attempt {}/{}) with tighter "
                    "local budgets.",
                    step,
                    last_turning_point.step,
                    last_turning_point.restart_attempts,
                    cfg.max_turning_point_restarts);

                analysis.restore_restart_bundle(last_turning_point.analysis);
                retune_local_models(last_turning_point.restart_attempts);

                records.resize(last_turning_point.record_count);
                global_rows.resize(last_turning_point.global_rows_count);
                hysteresis_rows.resize(last_turning_point.hysteresis_rows_count);
                crack_rows.resize(last_turning_point.crack_rows_count);
                solver_rows.resize(last_turning_point.solver_rows_count);
                rewrite_buffer(
                    global_history_path, global_header, global_rows);
                rewrite_buffer(
                    hysteresis_path, hysteresis_header, hysteresis_rows);
                rewrite_buffer(
                    crack_path, crack_header.str(), crack_rows);
                rewrite_buffer(
                    solver_path, solver_header.str(), solver_rows);

                step = last_turning_point.step + 1;
                continue;
            }

            {
                std::ostringstream solver_row;
                write_fe2_solver_diagnostics_row(
                    solver_row, step, p, d, analysis);
                solver_rows.push_back(solver_row.str());
                rewrite_buffer(solver_path, solver_header.str(), solver_rows);
            }
            std::println(
                "    FE2 step {} aborted: reason={} failed_submodels={} "
                "rollback={} failed_sites={}",
                step,
                to_string(report.termination_reason),
                report.failed_submodels,
                report.rollback_performed ? "yes" : "no",
                summarize_failed_sites(report.failed_sites));
            ok = false;
            break;
        }

        const double p = nl.current_time();
        const double d = cfg.displacement(p);
        const double shear = extract_base_shear_x(model, base_nodes);
        records.push_back({step, p, d, shear});
        hysteresis_rec.on_step(
            fall_n::StepEvent{step, p, model.state_vector(), nullptr}, model);

        int total_cracked_gps = 0;
        int total_cracks = 0;
        bool damage_scalar_available = false;
        double peak_submodel_damage_scalar =
            std::numeric_limits<double>::quiet_NaN();
        bool fracture_history_available = false;
        double most_compressive_submodel_sigma_o_max = 0.0;
        double max_submodel_tau_o_max = 0.0;
        double max_opening = 0.0;
        std::vector<int> submodel_cracks;
        submodel_cracks.reserve(analysis.model().num_local_models());
        for (auto& ev : analysis.model().local_models()) {
            const auto cs = ev.crack_summary();
            total_cracked_gps += cs.num_cracked_gps;
            total_cracks += cs.total_cracks;
            if (cs.damage_scalar_available) {
                if (!damage_scalar_available) {
                    peak_submodel_damage_scalar = cs.max_damage_scalar;
                } else {
                    peak_submodel_damage_scalar = std::max(
                        peak_submodel_damage_scalar, cs.max_damage_scalar);
                }
                damage_scalar_available = true;
            }
            if (cs.fracture_history_available) {
                if (!fracture_history_available) {
                    most_compressive_submodel_sigma_o_max =
                        cs.most_compressive_sigma_o_max;
                } else {
                    most_compressive_submodel_sigma_o_max = std::min(
                        most_compressive_submodel_sigma_o_max,
                        cs.most_compressive_sigma_o_max);
                }
                max_submodel_tau_o_max = std::max(
                    max_submodel_tau_o_max, cs.max_tau_o_max);
                fracture_history_available = true;
            }
            max_opening = std::max(max_opening, cs.max_opening);
            submodel_cracks.push_back(cs.total_cracks);
        }
        const double peak_damage = peak_structural_damage(model, damage_crit);
        const auto& report = analysis.last_report();
        {
            std::ostringstream row;
            row << step << "," << p << "," << d << "," << shear << ","
                << peak_damage << ","
                << (damage_scalar_available ? 1 : 0) << ","
                << csv_scalar_or_nan(
                       damage_scalar_available, peak_submodel_damage_scalar)
                << ","
                << csv_scalar_or_nan(fracture_history_available,
                                     most_compressive_submodel_sigma_o_max)
                << ","
                << csv_scalar_or_nan(fracture_history_available,
                                     max_submodel_tau_o_max)
                << "," << total_cracked_gps << "," << total_cracks << ","
                << max_opening << ","
                << analysis.last_staggered_iterations() << ","
                << report.max_force_residual_rel << ","
                << report.max_force_component_residual_rel << ","
                << report.max_tangent_residual_rel << ","
                << report.max_tangent_column_residual_rel << "\n";
            global_rows.push_back(row.str());
        }
        {
            std::ostringstream row;
            row << std::scientific << std::setprecision(8)
                << step << "," << p << "," << d << "," << shear << "\n";
            hysteresis_rows.push_back(row.str());
        }
        {
            std::ostringstream row;
            row << step << "," << p << "," << d << ","
                << total_cracked_gps << "," << total_cracks << ","
                << (damage_scalar_available ? 1 : 0) << ","
                << csv_scalar_or_nan(
                       damage_scalar_available, peak_submodel_damage_scalar)
                << ","
                << csv_scalar_or_nan(fracture_history_available,
                                     most_compressive_submodel_sigma_o_max)
                << ","
                << csv_scalar_or_nan(fracture_history_available,
                                     max_submodel_tau_o_max)
                << "," << max_opening;
            for (int count : submodel_cracks) {
                row << "," << count;
            }
            row << "\n";
            crack_rows.push_back(row.str());
        }
        {
            std::ostringstream row;
            write_fe2_solver_diagnostics_row(row, step, p, d, analysis);
            solver_rows.push_back(row.str());
        }

        rewrite_buffer(global_history_path, global_header, global_rows);
        rewrite_buffer(hysteresis_path, hysteresis_header, hysteresis_rows);
        rewrite_buffer(crack_path, crack_header.str(), crack_rows);
        rewrite_buffer(solver_path, solver_header.str(), solver_rows);

        if (write_global_vtk
            && ((step - 1) % cfg.global_output_interval == 0))
        {
            fall_n::vtk::StructuralVTMExporter vtm{
                model, beam_profile, shell_profile};
            vtm.set_displacement(model.state_vector());
            vtm.set_yield_strain(EPS_YIELD);
            const auto vtm_file = std::format(
                "{}/vtk/table_fe2_{:04d}.vtm", out_dir, step);
            vtm.write(vtm_file);
            pvd_fe2.add_timestep(p, vtm_file);
        }

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (step % 5 == 0 || step == executed_steps) {
            std::println(
                "    step={:3d}/{:3d}  p={:.4f}  d={:+.4e} m  "
                "V={:+.4e} MN  cracks={}  max_open={:.4e}  stag={}  t={}s",
                step, executed_steps, p, d, shear,
                total_cracks, max_opening,
                analysis.last_staggered_iterations(), elapsed);
            std::fflush(stdout);
        }

        if (restart_capable && cfg.is_turning_point_step(step)) {
            last_turning_point.analysis = analysis.capture_restart_bundle();
            last_turning_point.step = step;
            last_turning_point.record_count = records.size();
            last_turning_point.global_rows_count = global_rows.size();
            last_turning_point.hysteresis_rows_count = hysteresis_rows.size();
            last_turning_point.crack_rows_count = crack_rows.size();
            last_turning_point.solver_rows_count = solver_rows.size();
            last_turning_point.restart_attempts = 0;
            last_turning_point.valid = true;
        }

        ++step;
    }

    std::println("  Result: {} ({} records)",
                 ok ? (truncated_protocol ? "TRUNCATED" : "COMPLETED")
                    : "ABORTED",
                 records.size());
    hysteresis_rec.on_analysis_end(model);

    for (auto& ev : analysis.model().local_models()) {
        ev.finalize();
    }

    if (write_global_vtk) {
        pvd_fe2.write();
    }
    hysteresis_rec.write_hysteresis_csv(
        out_dir + "/recorders/fiber_hysteresis");

    return records;
}

} // namespace fall_n::table_cyclic_validation
