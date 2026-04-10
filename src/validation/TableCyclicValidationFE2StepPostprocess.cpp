#include "src/validation/TableCyclicValidationFE2StepPostprocess.hh"

#include "src/validation/TableCyclicValidationSupport.hh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

namespace fall_n::table_cyclic_validation {

namespace {

[[nodiscard]] double csv_scalar_or_nan(bool available, double value)
{
    return available ? value : std::numeric_limits<double>::quiet_NaN();
}

[[nodiscard]] double csv_finite_or_nan(double value)
{
    return std::isfinite(value)
         ? value
         : std::numeric_limits<double>::quiet_NaN();
}

[[nodiscard]] bool has_attempted_local_state(const SubModelSolverResult& solve)
{
    return solve.converged || solve.achieved_fraction > 0.0;
}

} // namespace

FE2StepDiagnostics collect_fe2_step_diagnostics(
    StructModel& model,
    ValidationAnalysis& analysis,
    const DamageCriterion& damage_crit)
{
    FE2StepDiagnostics diagnostics;
    diagnostics.peak_submodel_damage_scalar =
        std::numeric_limits<double>::quiet_NaN();
    diagnostics.submodel_cracks.reserve(analysis.model().num_local_models());

    for (auto& ev : analysis.model().local_models()) {
        const auto& solve = ev.last_solve_result();
        diagnostics.total_active_crack_history_points +=
            solve.material_points_with_active_cracks;
        diagnostics.max_num_cracks_at_point = std::max(
            diagnostics.max_num_cracks_at_point,
            solve.max_num_cracks_at_point);
        const auto cs = ev.crack_summary();
        diagnostics.total_cracked_gps += cs.num_cracked_gps;
        diagnostics.total_cracks += cs.total_cracks;
        if (cs.damage_scalar_available) {
            if (!diagnostics.damage_scalar_available) {
                diagnostics.peak_submodel_damage_scalar = cs.max_damage_scalar;
            } else {
                diagnostics.peak_submodel_damage_scalar = std::max(
                    diagnostics.peak_submodel_damage_scalar,
                    cs.max_damage_scalar);
            }
            diagnostics.damage_scalar_available = true;
        }
        if (cs.fracture_history_available) {
            if (!diagnostics.fracture_history_available) {
                diagnostics.most_compressive_submodel_sigma_o_max =
                    cs.most_compressive_sigma_o_max;
            } else {
                diagnostics.most_compressive_submodel_sigma_o_max = std::min(
                    diagnostics.most_compressive_submodel_sigma_o_max,
                    cs.most_compressive_sigma_o_max);
            }
            diagnostics.max_submodel_tau_o_max = std::max(
                diagnostics.max_submodel_tau_o_max,
                cs.max_tau_o_max);
            diagnostics.fracture_history_available = true;
        }
        diagnostics.max_opening = std::max(
            diagnostics.max_opening, cs.max_opening);
        diagnostics.submodel_cracks.push_back(cs.total_cracks);
    }

    diagnostics.peak_damage =
        peak_structural_damage(model, damage_crit);
    return diagnostics;
}

FE2StepDiagnostics collect_fe2_failed_step_diagnostics(
    ValidationAnalysis& analysis)
{
    FE2StepDiagnostics diagnostics;
    diagnostics.accepted = false;
    diagnostics.peak_submodel_damage_scalar =
        std::numeric_limits<double>::quiet_NaN();
    diagnostics.submodel_cracks.reserve(analysis.model().num_local_models());

    for (auto& ev : analysis.model().local_models()) {
        const auto& solve = ev.last_solve_result();
        diagnostics.total_active_crack_history_points +=
            solve.material_points_with_active_cracks;
        diagnostics.max_num_cracks_at_point = std::max(
            diagnostics.max_num_cracks_at_point,
            solve.max_num_cracks_at_point);
        if (!has_attempted_local_state(solve)) {
            diagnostics.submodel_cracks.push_back(0);
            continue;
        }

        const auto cs = ev.last_attempted_crack_summary();
        diagnostics.total_cracked_gps += cs.num_cracked_gps;
        diagnostics.total_cracks += cs.total_cracks;
        if (cs.damage_scalar_available) {
            if (!diagnostics.damage_scalar_available) {
                diagnostics.peak_submodel_damage_scalar = cs.max_damage_scalar;
            } else {
                diagnostics.peak_submodel_damage_scalar = std::max(
                    diagnostics.peak_submodel_damage_scalar,
                    cs.max_damage_scalar);
            }
            diagnostics.damage_scalar_available = true;
        }
        if (cs.fracture_history_available) {
            if (!diagnostics.fracture_history_available) {
                diagnostics.most_compressive_submodel_sigma_o_max =
                    cs.most_compressive_sigma_o_max;
            } else {
                diagnostics.most_compressive_submodel_sigma_o_max = std::min(
                    diagnostics.most_compressive_submodel_sigma_o_max,
                    cs.most_compressive_sigma_o_max);
            }
            diagnostics.max_submodel_tau_o_max = std::max(
                diagnostics.max_submodel_tau_o_max,
                cs.max_tau_o_max);
            diagnostics.fracture_history_available = true;
        }
        diagnostics.max_opening = std::max(
            diagnostics.max_opening, cs.max_opening);
        diagnostics.submodel_cracks.push_back(cs.total_cracks);
    }

    return diagnostics;
}

void append_fe2_step_records(
    FE2RecorderBuffers& recorder_buffers,
    int step,
    double p,
    double d,
    double shear,
    const FE2StepDiagnostics& diagnostics,
    const ValidationAnalysis& analysis)
{
    const auto& report = analysis.last_report();
    {
        std::ostringstream row;
        row << step << "," << p << "," << d << ","
            << csv_finite_or_nan(shear) << ","
            << (diagnostics.accepted ? 1 : 0) << ","
            << to_string(report.termination_reason) << ","
            << (report.rollback_performed ? 1 : 0) << ","
            << csv_finite_or_nan(diagnostics.peak_damage) << ","
            << (diagnostics.damage_scalar_available ? 1 : 0) << ","
            << csv_scalar_or_nan(
                   diagnostics.damage_scalar_available,
                   diagnostics.peak_submodel_damage_scalar)
            << ","
            << csv_scalar_or_nan(
                   diagnostics.fracture_history_available,
                   diagnostics.most_compressive_submodel_sigma_o_max)
            << ","
            << csv_scalar_or_nan(
                   diagnostics.fracture_history_available,
                   diagnostics.max_submodel_tau_o_max)
            << "," << diagnostics.total_cracked_gps << ","
            << diagnostics.total_cracks << "," << diagnostics.max_opening
            << "," << diagnostics.total_active_crack_history_points
            << "," << diagnostics.max_num_cracks_at_point
            << "," << analysis.last_staggered_iterations() << ","
            << report.max_force_residual_rel << ","
            << report.max_force_component_residual_rel << ","
            << report.max_tangent_residual_rel << ","
            << report.max_tangent_column_residual_rel << "\n";
        recorder_buffers.global_rows.push_back(row.str());
    }
    {
        if (diagnostics.accepted && std::isfinite(shear)) {
            std::ostringstream row;
            row << std::scientific << std::setprecision(8)
                << step << "," << p << "," << d << "," << shear << "\n";
            recorder_buffers.hysteresis_rows.push_back(row.str());
        }
    }
    {
        std::ostringstream row;
        row << step << "," << p << "," << d << ","
            << (diagnostics.accepted ? 1 : 0) << ","
            << to_string(report.termination_reason) << ","
            << (report.rollback_performed ? 1 : 0) << ","
            << diagnostics.total_cracked_gps << ","
            << diagnostics.total_cracks << ","
            << diagnostics.total_active_crack_history_points << ","
            << diagnostics.max_num_cracks_at_point << ","
            << (diagnostics.damage_scalar_available ? 1 : 0) << ","
            << csv_scalar_or_nan(
                   diagnostics.damage_scalar_available,
                   diagnostics.peak_submodel_damage_scalar)
            << ","
            << csv_scalar_or_nan(
                   diagnostics.fracture_history_available,
                   diagnostics.most_compressive_submodel_sigma_o_max)
            << ","
            << csv_scalar_or_nan(
                   diagnostics.fracture_history_available,
                   diagnostics.max_submodel_tau_o_max)
            << "," << diagnostics.max_opening;
        for (int count : diagnostics.submodel_cracks) {
            row << "," << count;
        }
        row << "\n";
        recorder_buffers.crack_rows.push_back(row.str());
    }
    {
        std::ostringstream row;
        write_fe2_solver_diagnostics_row(row, step, p, d, analysis);
        recorder_buffers.solver_rows.push_back(row.str());
    }
}

void print_fe2_step_progress(
    int step,
    int executed_steps,
    double p,
    double d,
    double shear,
    const FE2StepDiagnostics& diagnostics,
    int staggered_iterations,
    std::chrono::steady_clock::time_point start_time)
{
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start_time).count();
    std::println(
        "    step={:3d}/{:3d}  p={:.4f}  d={:+.4e} m  "
        "V={:+.4e} MN  cracks={}  max_open={:.4e}  stag={}  t={}s",
        step, executed_steps, p, d, shear,
        diagnostics.total_cracks, diagnostics.max_opening,
        staggered_iterations, elapsed);
    std::fflush(stdout);
}

} // namespace fall_n::table_cyclic_validation
