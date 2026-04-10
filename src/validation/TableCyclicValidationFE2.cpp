#include "src/validation/TableCyclicValidationSupport.hh"
#include "src/validation/TableCyclicValidationFE2Recorders.hh"
#include "src/validation/TableCyclicValidationFE2Setup.hh"
#include "src/validation/TableCyclicValidationFE2Restart.hh"
#include "src/validation/TableCyclicValidationFE2StepPostprocess.hh"
#include "src/validation/TableCyclicValidationRuntimeIO.hh"

#include <chrono>
#include <limits>

namespace fall_n::table_cyclic_validation {

std::vector<StepRecord>
run_case_fe2(bool two_way, const std::string& out_dir,
             const CyclicValidationRunConfig& cfg)
{
    const char* label = two_way
        ? "5 (Iterated two-way FE²)"
        : "4 (One-way downscaling)";
    std::println("\n  Case {}: Full table + FE² sub-models", label);

    auto ctx = build_fe2_case_context(two_way, out_dir, cfg);
    auto& model = *ctx.model;
    auto& nl = *ctx.nl;
    auto& analysis = *ctx.analysis;
    const std::array<std::size_t, 4> slab_corners = {4, 7, 16, 19};
    const int protocol_steps = cfg.total_steps();
    const int executed_steps =
        (cfg.max_steps > 0) ? std::min(protocol_steps, cfg.max_steps)
                            : protocol_steps;
    const bool truncated_protocol = executed_steps < protocol_steps;

    std::vector<StepRecord> records;
    records.reserve(static_cast<std::size_t>(executed_steps) + 1);
    records.push_back({0, 0.0, 0.0, 0.0});

    const auto& base_nodes = ctx.base_nodes;
    const auto t0 = std::chrono::steady_clock::now();

    const bool write_global_vtk = cfg.global_output_interval > 0;
    if (write_global_vtk) {
        std::filesystem::create_directories(out_dir + "/vtk");
    }
    std::filesystem::create_directories(out_dir + "/recorders");
    MaxStrainDamageCriterion damage_crit{EPS_YIELD};
    FiberHysteresisRecorder<StructModel> hysteresis_rec{
        damage_crit, classify_table_fiber, {}, 5, 1};
    FE2RecorderBuffers recorder_buffers =
        initialize_fe2_recorders(out_dir, analysis);
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

    FE2TurningPointFrame<typename ValidationAnalysis::RestartBundle>
        last_turning_point;

    const bool restart_capable =
        cfg.enable_turning_point_checkpoints && !write_global_vtk;

    bool ok = true;
    int step = 1;
    while (step <= executed_steps) {
        if (!analysis.step()) {
            const auto& report = analysis.last_report();
            const double p = report.attempted_state_valid
                ? report.attempted_macro_time
                : nl.current_time();
            const double d = cfg.displacement(p);
            if (restart_capable
                && try_restart_from_turning_point(
                    step,
                    analysis,
                    cfg,
                    last_turning_point,
                    records,
                    recorder_buffers))
            {
                continue;
            }

            {
                const auto diagnostics =
                    collect_fe2_failed_step_diagnostics(analysis);
                append_fe2_step_records(
                    recorder_buffers,
                    step,
                    p,
                    d,
                    std::numeric_limits<double>::quiet_NaN(),
                    diagnostics,
                    analysis);
                recorder_buffers.rewrite_all();
            }
            std::println(
                "    FE2 step {} aborted at p={:.6f} drift={:.6e}: "
                "reason={} failed_submodels={} rollback={} failed_sites={}",
                step,
                p,
                d,
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
        const auto diagnostics = collect_fe2_step_diagnostics(
            model, analysis, damage_crit);
        append_fe2_step_records(
            recorder_buffers, step, p, d, shear, diagnostics, analysis);

        recorder_buffers.rewrite_all();

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

        if (step % 5 == 0 || step == executed_steps) {
            print_fe2_step_progress(
                step,
                executed_steps,
                p,
                d,
                shear,
                diagnostics,
                analysis.last_staggered_iterations(),
                t0);
        }

        if (restart_capable && cfg.is_turning_point_step(step)) {
            last_turning_point.analysis = analysis.capture_restart_bundle();
            last_turning_point.step = step;
            last_turning_point.record_count = records.size();
            last_turning_point.recorder_counts = recorder_buffers.counts();
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
