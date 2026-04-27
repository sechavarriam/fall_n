#include "src/validation/ReducedRCColumnSectionBaseline.hh"

#include "src/materials/beam/BeamGeneralizedStrain.hh"
#include "src/utils/Benchmark.hh"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <print>
#include <ranges>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

using BeamSectionStrain = BeamGeneralizedStrain<6, 3>;

struct SectionProtocolPoint {
    int step{0};
    double load_factor{0.0};
    double curvature_y{0.0};
};

[[nodiscard]] Material<TimoshenkoBeam3D> make_section_material(
    const ReducedRCColumnSectionBaselineRunSpec& spec)
{
    const auto section_spec = to_rc_column_section_spec(spec.reference_spec);
    switch (spec.material_mode) {
        case ReducedRCColumnSectionMaterialMode::nonlinear:
            return make_rc_column_section(section_spec);
        case ReducedRCColumnSectionMaterialMode::elasticized:
            return make_rc_column_section_elasticized(section_spec);
    }
    throw std::invalid_argument(
        "Reduced RC section baseline received an unknown section material mode.");
}

[[nodiscard]] double signed_target_axial_force_mn(
    double target_axial_compression_force_mn) noexcept
{
    return -std::abs(target_axial_compression_force_mn);
}

[[nodiscard]] std::vector<double>
section_cyclic_levels(const ReducedRCColumnSectionBaselineRunSpec& spec)
{
    if (!spec.cyclic_curvature_levels_y.empty()) {
        std::vector<double> levels = spec.cyclic_curvature_levels_y;
        std::ranges::transform(levels, levels.begin(), [](double value) {
            return std::abs(value);
        });
        return levels;
    }
    return {std::abs(spec.max_curvature_y)};
}

[[nodiscard]] std::vector<SectionProtocolPoint>
build_section_protocol(const ReducedRCColumnSectionBaselineRunSpec& spec)
{
    std::vector<SectionProtocolPoint> protocol;
    protocol.push_back({.step = 0, .load_factor = 0.0, .curvature_y = 0.0});

    if (spec.protocol_kind == ReducedRCColumnSectionProtocolKind::monotonic) {
        for (int step = 1; step <= spec.steps; ++step) {
            const auto load_factor =
                static_cast<double>(step) / static_cast<double>(spec.steps);
            protocol.push_back(
                {.step = step,
                 .load_factor = load_factor,
                 .curvature_y = load_factor * spec.max_curvature_y});
        }
        return protocol;
    }

    const auto levels = section_cyclic_levels(spec);
    const auto steps_per_segment = std::max(spec.steps_per_segment, 1);
    const auto add_branch = [&](double start, double end, int branch_steps) {
        for (int i = 1; i <= branch_steps; ++i) {
            const auto t =
                static_cast<double>(i) / static_cast<double>(branch_steps);
            protocol.push_back(
                {.step = static_cast<int>(protocol.size()),
                 .load_factor = 0.0,
                 .curvature_y = start + t * (end - start)});
        }
    };

    double current = 0.0;
    for (const double level : levels) {
        add_branch(current, level, steps_per_segment);
        add_branch(level, -level, 2 * steps_per_segment);
        add_branch(-level, 0.0, steps_per_segment);
        current = 0.0;
    }

    if (protocol.size() > 1u) {
        const auto denom = static_cast<double>(protocol.size() - 1u);
        for (std::size_t i = 1; i < protocol.size(); ++i) {
            protocol[i].load_factor = static_cast<double>(i) / denom;
        }
    }
    return protocol;
}

[[nodiscard]] double condensed_bending_tangent(
    const Eigen::Matrix<double, 6, 6>& tangent,
    int bending_index) noexcept
{
    const double axial_tangent = tangent(0, 0);
    if (!std::isfinite(axial_tangent) || std::abs(axial_tangent) < 1.0e-12) {
        return tangent(bending_index, bending_index);
    }

    return tangent(bending_index, bending_index) -
           tangent(bending_index, 0) * tangent(0, bending_index) /
               axial_tangent;
}

[[nodiscard]] double relative_error(double lhs, double rhs) noexcept
{
    const double scale = std::max(std::abs(rhs), 1.0e-12);
    return std::abs(lhs - rhs) / scale;
}

[[nodiscard]] std::vector<RCSectionFiberLayoutRecord>
canonical_section_layout(const ReducedRCColumnSectionBaselineRunSpec& spec)
{
    return build_rc_column_fiber_layout(
        to_rc_column_section_spec(spec.reference_spec));
}

template <typename RowT, typename XFn, typename YFn>
[[nodiscard]] double numerical_slope(
    const std::vector<RowT>& rows,
    std::size_t index,
    XFn&& x_of,
    YFn&& y_of,
    int direction) noexcept
{
    if (rows.size() < 2u || index >= rows.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto slope = [&](std::size_t i0, std::size_t i1) {
        const double dx = x_of(rows[i1]) - x_of(rows[i0]);
        if (std::abs(dx) <= 1.0e-14) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return (y_of(rows[i1]) - y_of(rows[i0])) / dx;
    };

    if (direction < 0) {
        return index == 0u
                   ? std::numeric_limits<double>::quiet_NaN()
                   : slope(index - 1u, index);
    }
    if (direction > 0) {
        return index + 1u >= rows.size()
                   ? std::numeric_limits<double>::quiet_NaN()
                   : slope(index, index + 1u);
    }

    const double left = numerical_slope(
        rows, index, std::forward<XFn>(x_of), std::forward<YFn>(y_of), -1);
    const double right = numerical_slope(
        rows, index, std::forward<XFn>(x_of), std::forward<YFn>(y_of), +1);
    if (std::isfinite(left) && std::isfinite(right)) {
        return 0.5 * (left + right);
    }
    return std::isfinite(left) ? left : right;
}

void write_section_baseline_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnSectionBaselineRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,load_factor,target_axial_force_MN,solved_axial_strain,"
           "curvature_y,curvature_z,axial_force_MN,moment_y_MNm,moment_z_MNm,"
           "tangent_ea,tangent_eiy,tangent_eiz,tangent_eiy_direct_raw,"
           "tangent_eiz_direct_raw,newton_iterations,final_axial_force_residual_MN\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& r : records) {
        ofs << r.step << ","
            << r.load_factor << ","
            << r.target_axial_force << ","
            << r.solved_axial_strain << ","
            << r.curvature_y << ","
            << r.curvature_z << ","
            << r.axial_force << ","
            << r.moment_y << ","
            << r.moment_z << ","
            << r.tangent_ea << ","
            << r.tangent_eiy << ","
            << r.tangent_eiz << ","
            << r.tangent_eiy_direct_raw << ","
            << r.tangent_eiz_direct_raw << ","
            << r.newton_iterations << ","
            << r.final_axial_force_residual << "\n";
    }

    std::println("  CSV: {} ({} records)", path, records.size());
}

void write_section_tangent_diagnostics_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnSectionBaselineRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,load_factor,curvature_y,moment_y_MNm,zero_curvature_anchor,"
           "tangent_eiy_condensed,tangent_eiy_direct_raw,tangent_eiy_numerical,"
           "tangent_eiy_left,tangent_eiy_right,tangent_consistency_rel_error,"
           "raw_k00,raw_k0y,raw_ky0,raw_kyy\n";
    ofs << std::scientific << std::setprecision(8);

    const auto curvature_of = [](const auto& row) { return row.curvature_y; };
    const auto moment_of = [](const auto& row) { return row.moment_y; };

    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& row = records[i];
        const double tangent_left =
            numerical_slope(records, i, curvature_of, moment_of, -1);
        const double tangent_right =
            numerical_slope(records, i, curvature_of, moment_of, +1);
        const double tangent_numerical =
            numerical_slope(records, i, curvature_of, moment_of, 0);
        const double consistency_error =
            std::isfinite(tangent_numerical)
                ? relative_error(row.tangent_eiy, tangent_numerical)
                : std::numeric_limits<double>::quiet_NaN();

        ofs << row.step << ","
            << row.load_factor << ","
            << row.curvature_y << ","
            << row.moment_y << ","
            << (std::abs(row.curvature_y) <= 1.0e-12 ? 1 : 0) << ","
            << row.tangent_eiy << ","
            << row.tangent_eiy_direct_raw << ","
            << tangent_numerical << ","
            << tangent_left << ","
            << tangent_right << ","
            << consistency_error << ","
            << row.raw_tangent_k00 << ","
            << row.raw_tangent_k0y << ","
            << row.raw_tangent_ky0 << ","
            << row.raw_tangent_kyy << "\n";
    }

    std::println("  CSV: {} ({} records)", path, records.size());
}

void append_section_fiber_history_records(
    const Material<TimoshenkoBeam3D>& section_material,
    const std::vector<RCSectionFiberLayoutRecord>& layout,
    const ReducedRCColumnSectionBaselineRecord& step_record,
    std::vector<ReducedRCColumnSectionFiberRecord>& out_records)
{
    const auto snapshot = section_material.section_snapshot();
    if (!snapshot.has_fibers()) {
        throw std::runtime_error(
            "Reduced RC section baseline expected a fiber-resolved section "
            "snapshot, but the constitutive handle did not expose fibers.");
    }
    if (snapshot.fibers.size() != layout.size()) {
        throw std::runtime_error(
            "Reduced RC section baseline found a mismatch between the canonical "
            "RC fiber layout and the section snapshot fiber count.");
    }

    for (std::size_t i = 0; i < layout.size(); ++i) {
        const auto& fiber_layout = layout[i];
        const auto& fiber_state = snapshot.fibers[i];
        const double axial_force_contribution = fiber_state.stress_xx * fiber_state.area;
        const double moment_y_contribution =
            -fiber_state.stress_xx * fiber_state.z * fiber_state.area;
        const double raw_tangent_k00_contribution =
            fiber_state.tangent_xx * fiber_state.area;
        const double raw_tangent_k0y_contribution =
            -fiber_state.tangent_xx * fiber_state.z * fiber_state.area;
        const double raw_tangent_kyy_contribution =
            fiber_state.tangent_xx * fiber_state.z * fiber_state.z *
            fiber_state.area;

        out_records.push_back({
            .step = step_record.step,
            .load_factor = step_record.load_factor,
            .solved_axial_strain = step_record.solved_axial_strain,
            .curvature_y = step_record.curvature_y,
            .zero_curvature_anchor =
                std::abs(step_record.curvature_y) <= 1.0e-12,
            .fiber_index = fiber_state.fiber_index,
            .y = fiber_state.y,
            .z = fiber_state.z,
            .area = fiber_state.area,
            .zone = fiber_layout.zone,
            .material_role = fiber_layout.material_role,
            .strain_xx = fiber_state.strain_xx,
            .stress_xx = fiber_state.stress_xx,
            .tangent_xx = fiber_state.tangent_xx,
            .axial_force_contribution = axial_force_contribution,
            .moment_y_contribution = moment_y_contribution,
            .raw_tangent_k00_contribution = raw_tangent_k00_contribution,
            .raw_tangent_k0y_contribution = raw_tangent_k0y_contribution,
            .raw_tangent_kyy_contribution = raw_tangent_kyy_contribution,
        });
    }
}

void write_section_fiber_history_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnSectionFiberRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,load_factor,solved_axial_strain,curvature_y,zero_curvature_anchor,"
           "fiber_index,y,z,area,zone,material_role,strain_xx,stress_xx_MPa,"
           "tangent_xx_MPa,axial_force_contribution_MN,moment_y_contribution_MNm,"
           "raw_k00_contribution,raw_k0y_contribution,raw_kyy_contribution\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : records) {
        ofs << row.step << ","
            << row.load_factor << ","
            << row.solved_axial_strain << ","
            << row.curvature_y << ","
            << (row.zero_curvature_anchor ? 1 : 0) << ","
            << row.fiber_index << ","
            << row.y << ","
            << row.z << ","
            << row.area << ","
            << std::string{to_string(row.zone)} << ","
            << std::string{to_string(row.material_role)} << ","
            << row.strain_xx << ","
            << row.stress_xx << ","
            << row.tangent_xx << ","
            << row.axial_force_contribution << ","
            << row.moment_y_contribution << ","
            << row.raw_tangent_k00_contribution << ","
            << row.raw_tangent_k0y_contribution << ","
            << row.raw_tangent_kyy_contribution << "\n";
    }

    std::println("  CSV: {} ({} fiber records)", path, records.size());
}

void write_section_control_trace_csv(
    const std::string& path,
    const std::vector<SectionProtocolPoint>& protocol,
    const std::vector<ReducedRCColumnSectionBaselineRecord>& records)
{
    if (protocol.size() != records.size()) {
        throw std::runtime_error(
            "Reduced RC section baseline expected protocol and record counts "
            "to match when writing the control-trace CSV.");
    }

    std::ofstream ofs(path);
    ofs << "step,load_factor,stage,target_curvature_y,actual_curvature_y,"
           "delta_target_curvature_y,delta_actual_curvature_y,pseudo_time_before,"
           "pseudo_time_after,pseudo_time_increment,domain_time_before,"
           "domain_time_after,domain_time_increment,control_dof_before,"
           "control_dof_after,target_increment_direction,"
           "actual_increment_direction,protocol_branch_id,reversal_index,"
           "branch_step_index,accepted_substep_count,max_bisection_level,"
           "newton_iterations,newton_iterations_per_substep,"
           "target_axial_force_MN,actual_axial_force_MN,"
           "axial_force_residual_MN\n";
    ofs << std::scientific << std::setprecision(8);

    const auto signum = [](double value) noexcept {
        return (value > 1.0e-14) ? 1 : ((value < -1.0e-14) ? -1 : 0);
    };

    double previous_target_curvature = 0.0;
    double previous_actual_curvature = 0.0;
    double previous_pseudo_time = 0.0;
    int previous_target_direction = 0;
    int protocol_branch_id = 0;
    int reversal_index = 0;
    int branch_step_index = 0;

    for (std::size_t i = 0; i < records.size(); ++i) {
        const auto& point = protocol[i];
        const auto& row = records[i];
        const bool is_initial = i == 0u;
        const double delta_target_curvature =
            is_initial ? 0.0 : point.curvature_y - previous_target_curvature;
        const double delta_actual_curvature =
            is_initial ? 0.0 : row.curvature_y - previous_actual_curvature;
        const double pseudo_time_before = previous_pseudo_time;
        const double pseudo_time_after = row.load_factor;
        const double pseudo_time_increment =
            is_initial ? 0.0 : pseudo_time_after - pseudo_time_before;
        const int accepted_substep_count =
            (is_initial || std::abs(delta_target_curvature) <= 1.0e-14) ? 0 : 1;
        const int target_increment_direction = signum(delta_target_curvature);
        const int actual_increment_direction = signum(delta_actual_curvature);
        if (target_increment_direction != 0) {
            if (previous_target_direction == 0) {
                protocol_branch_id = 1;
                branch_step_index = 1;
            } else if (target_increment_direction != previous_target_direction) {
                ++protocol_branch_id;
                ++reversal_index;
                branch_step_index = 1;
            } else {
                ++branch_step_index;
            }
            previous_target_direction = target_increment_direction;
        } else if (is_initial) {
            protocol_branch_id = 0;
            branch_step_index = 0;
        }
        const double newton_iterations_per_substep =
            accepted_substep_count > 0
                ? static_cast<double>(row.newton_iterations) /
                      static_cast<double>(accepted_substep_count)
                : 0.0;

        ofs << row.step << ","
            << row.load_factor << ","
            << (is_initial ? "preload_equilibrated" : "curvature_branch") << ","
            << point.curvature_y << ","
            << row.curvature_y << ","
            << delta_target_curvature << ","
            << delta_actual_curvature << ","
            << pseudo_time_before << ","
            << pseudo_time_after << ","
            << pseudo_time_increment << ","
            << pseudo_time_before << ","
            << pseudo_time_after << ","
            << pseudo_time_increment << ","
            << previous_actual_curvature << ","
            << row.curvature_y << ","
            << target_increment_direction << ","
            << actual_increment_direction << ","
            << protocol_branch_id << ","
            << reversal_index << ","
            << branch_step_index << ","
            << accepted_substep_count << ","
            << 0 << ","
            << row.newton_iterations << ","
            << newton_iterations_per_substep << ","
            << row.target_axial_force << ","
            << row.axial_force << ","
            << row.final_axial_force_residual << "\n";

        previous_target_curvature = point.curvature_y;
        previous_actual_curvature = row.curvature_y;
        previous_pseudo_time = row.load_factor;
    }

    std::println("  CSV: {} ({} control steps)", path, records.size());
}

[[nodiscard]] ReducedRCColumnSectionBaselineRecord solve_section_step(
    Material<TimoshenkoBeam3D>& section_material,
    const ReducedRCColumnSectionBaselineRunSpec& spec,
    const SectionProtocolPoint& point,
    double starting_axial_strain)
{
    const double target_axial_force =
        signed_target_axial_force_mn(spec.target_axial_compression_force_mn);

    BeamSectionStrain strain{};
    strain[1] = point.curvature_y;
    strain[2] = 0.0;

    double axial_strain = starting_axial_strain;
    int iterations = 0;
    double residual = 0.0;
    BeamSectionForces<6> response{};
    Eigen::Matrix<double, 6, 6> tangent = Eigen::Matrix<double, 6, 6>::Zero();

    for (iterations = 1; iterations <= spec.axial_force_newton_max_iterations;
         ++iterations) {
        strain[0] = axial_strain;
        section_material.update_state(strain);
        response = section_material.compute_response(strain);
        tangent = section_material.tangent(strain);

        residual = response.axial_force() - target_axial_force;
        if (std::abs(residual) <= spec.axial_force_newton_tolerance_mn) {
            break;
        }

        const double tangent_ea = tangent(0, 0);
        if (!std::isfinite(tangent_ea) || std::abs(tangent_ea) < 1.0e-12) {
            throw std::runtime_error(
                "Reduced RC section baseline encountered a singular axial tangent "
                "while solving section-level axial equilibrium.");
        }

        axial_strain -= residual / tangent_ea;
    }

    if (std::abs(residual) > spec.axial_force_newton_tolerance_mn) {
        throw std::runtime_error(
            "Reduced RC section baseline did not converge the axial-force "
            "equilibrium solve within the configured Newton budget.");
    }

    section_material.commit(strain);

    return ReducedRCColumnSectionBaselineRecord{
        .step = point.step,
        .load_factor = point.load_factor,
        .target_axial_force = target_axial_force,
        .solved_axial_strain = axial_strain,
        .curvature_y = strain[1],
        .curvature_z = strain[2],
        .axial_force = response.axial_force(),
        .moment_y = response.moment_y(),
        .moment_z = response.moment_z(),
        .tangent_ea = tangent(0, 0),
        .tangent_eiy = condensed_bending_tangent(tangent, 1),
        .tangent_eiz = condensed_bending_tangent(tangent, 2),
        .tangent_eiy_direct_raw = tangent(1, 1),
        .tangent_eiz_direct_raw = tangent(2, 2),
        .raw_tangent_k00 = tangent(0, 0),
        .raw_tangent_k0y = tangent(0, 1),
        .raw_tangent_ky0 = tangent(1, 0),
        .raw_tangent_kyy = tangent(1, 1),
        .newton_iterations = iterations,
        .final_axial_force_residual = residual,
    };
}

} // namespace

ReducedRCColumnSectionBaselineResult
run_reduced_rc_column_section_moment_curvature_baseline(
    const ReducedRCColumnSectionBaselineRunSpec& spec,
    const std::string& out_dir)
{
    if (spec.protocol_kind == ReducedRCColumnSectionProtocolKind::monotonic &&
        spec.steps <= 0) {
        throw std::invalid_argument(
            "Reduced RC section baseline requires a strictly positive number of "
            "steps.");
    }
    if (spec.protocol_kind == ReducedRCColumnSectionProtocolKind::cyclic &&
        std::max(spec.steps_per_segment, 0) <= 0) {
        throw std::invalid_argument(
            "Reduced RC section baseline requires a strictly positive "
            "steps_per_segment for cyclic curvature control.");
    }

    StopWatch total_timer;
    total_timer.start();
    StopWatch solve_timer;
    solve_timer.start();

    auto section_material = make_section_material(spec);
    const auto protocol = build_section_protocol(spec);
    const auto section_layout = canonical_section_layout(spec);

    ReducedRCColumnSectionBaselineResult result;
    result.records.reserve(protocol.size());
    result.fiber_history_records.reserve(protocol.size() * section_layout.size());

    const auto initial_record = solve_section_step(
        section_material,
        spec,
        protocol.front(),
        0.0);
    result.records.push_back(initial_record);
    append_section_fiber_history_records(
        section_material,
        section_layout,
        initial_record,
        result.fiber_history_records);

    double starting_axial_strain = initial_record.solved_axial_strain;
    for (std::size_t i = 1; i < protocol.size(); ++i) {
        const auto record = solve_section_step(
            section_material,
            spec,
            protocol[i],
            starting_axial_strain);

        starting_axial_strain = record.solved_axial_strain;
        result.records.push_back(record);
        append_section_fiber_history_records(
            section_material,
            section_layout,
            record,
            result.fiber_history_records);

        if (spec.print_progress &&
            (record.step % 20 == 0 || i + 1 == protocol.size())) {
            std::println(
                "    reduced-section step={:3d}  kappa_y={:+.4e}  M_y={:+.4e} MNm"
                "  N={:+.4e} MN  eps0={:+.4e}  newton={}  residual={:+.4e}",
                record.step,
                record.curvature_y,
                record.moment_y,
                record.axial_force,
                record.solved_axial_strain,
                record.newton_iterations,
                record.final_axial_force_residual);
        }
    }

    const double solve_wall_seconds = solve_timer.stop();
    StopWatch output_timer;
    output_timer.start();

    if (spec.write_csv) {
        std::filesystem::create_directories(out_dir);
        write_section_baseline_csv(
            out_dir + "/section_moment_curvature_baseline.csv",
            result.records);
        write_section_tangent_diagnostics_csv(
            out_dir + "/section_tangent_diagnostics.csv",
            result.records);
        write_section_fiber_history_csv(
            out_dir + "/section_fiber_state_history.csv",
            result.fiber_history_records);
        write_section_control_trace_csv(
            out_dir + "/section_control_trace.csv",
            protocol,
            result.records);
    }

    result.timing = ReducedRCColumnSectionBaselineTimingSummary{
        .total_wall_seconds = total_timer.stop(),
        .solve_wall_seconds = solve_wall_seconds,
        .output_write_wall_seconds = output_timer.stop(),
    };
    result.completed_successfully = true;

    return result;
}

} // namespace fall_n::validation_reboot
