#include "src/validation/ReducedRCColumnStructuralBaseline.hh"

#include "src/analysis/IncrementalControl.hh"
#include "src/analysis/NLAnalysis.hh"
#include "src/domain/Domain.hh"
#include "src/elements/ElementPolicy.hh"
#include "src/elements/TimoshenkoBeamN.hh"
#include "src/elements/element_geometry/LagrangeElement.hh"
#include "src/model/Model.hh"
#include "src/post-processing/StateQuery.hh"
#include "src/utils/Benchmark.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <print>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

using table_cyclic_validation::StepRecord;

inline constexpr std::size_t kReducedRCColumnNDoF = 6;

struct ReducedRCColumnControlPath {
    int lateral_steps{0};
    int axial_preload_steps{0};

    [[nodiscard]] int total_runtime_steps() const noexcept
    {
        return lateral_steps + axial_preload_steps;
    }

    [[nodiscard]] bool has_preload_stage() const noexcept
    {
        return axial_preload_steps > 0;
    }

    [[nodiscard]] double preload_completion_runtime_p() const noexcept
    {
        if (!has_preload_stage()) {
            return 0.0;
        }
        return static_cast<double>(axial_preload_steps) /
               static_cast<double>(total_runtime_steps());
    }

    [[nodiscard]] bool is_preload_runtime_step(int runtime_step) const noexcept
    {
        return has_preload_stage() && runtime_step <= axial_preload_steps;
    }

    [[nodiscard]] bool is_preload_completion_runtime_step(
        int runtime_step) const noexcept
    {
        return has_preload_stage() && runtime_step == axial_preload_steps;
    }

    [[nodiscard]] int logical_lateral_step(int runtime_step) const noexcept
    {
        if (!has_preload_stage()) {
            return runtime_step;
        }
        return std::max(runtime_step - axial_preload_steps, 0);
    }

    [[nodiscard]] double preload_progress(double runtime_p) const noexcept
    {
        if (!has_preload_stage()) {
            return 1.0;
        }
        return std::clamp(
            runtime_p / preload_completion_runtime_p(),
            0.0,
            1.0);
    }

    [[nodiscard]] double lateral_progress(double runtime_p) const noexcept
    {
        if (!has_preload_stage()) {
            return std::clamp(runtime_p, 0.0, 1.0);
        }

        if (runtime_p <= preload_completion_runtime_p()) {
            return 0.0;
        }

        const double denom = 1.0 - preload_completion_runtime_p();
        return std::clamp(
            (runtime_p - preload_completion_runtime_p()) / denom,
            0.0,
            1.0);
    }
};

[[nodiscard]] int signed_direction(double value,
                                   double tol = 1.0e-14) noexcept
{
    if (value > tol) {
        return 1;
    }
    if (value < -tol) {
        return -1;
    }
    return 0;
}

struct ReducedRCColumnControlTraceState {
    double previous_target_drift{0.0};
    double previous_actual_drift{0.0};
    int current_target_direction{0};
    int protocol_branch_id{0};
    int reversal_index{0};
    int branch_step_index{0};
    bool has_lateral_branch{false};
};

struct ReducedRCColumnControlTraceSnapshot {
    int target_increment_direction{0};
    int actual_increment_direction{0};
    int protocol_branch_id{0};
    int reversal_index{0};
    int branch_step_index{0};
};

struct ReducedRCColumnFailedTrialCapture {
    bool valid{false};
    double runtime_p{0.0};
    double logical_p{0.0};
    double target_drift{0.0};
    double actual_tip_lateral_displacement{0.0};
    double actual_tip_lateral_total_state_displacement{0.0};
    double prescribed_top_bending_rotation{0.0};
    double actual_top_bending_rotation{0.0};
    double top_axial_displacement{0.0};
    double base_shear{0.0};
    double base_axial_reaction{0.0};
    ReducedRCColumnControlTraceSnapshot trace_snapshot{};
    ReducedRCColumnStructuralElementTangentAuditRecord tangent_audit{};
    bool has_tangent_audit{false};
    std::vector<ReducedRCColumnStructuralSectionTangentAuditRecord>
        section_tangent_audit_records{};
    bool has_section_tangent_audit{false};
    std::vector<ReducedRCColumnSectionResponseRecord> section_response_records{};
    std::vector<ReducedRCColumnStructuralSectionFiberRecord> fiber_history_records{};
};

[[nodiscard]] ReducedRCColumnControlTraceSnapshot
advance_control_trace(
    ReducedRCColumnControlTraceState& state,
    double target_drift,
    double actual_drift,
    bool lateral_branch_active) noexcept
{
    if (!lateral_branch_active) {
        return {};
    }

    auto target_increment_direction =
        signed_direction(target_drift - state.previous_target_drift);
    const auto actual_increment_direction =
        signed_direction(actual_drift - state.previous_actual_drift);

    if (target_increment_direction == 0 && state.has_lateral_branch) {
        target_increment_direction = state.current_target_direction;
    }

    if (!state.has_lateral_branch && target_increment_direction != 0) {
        state.has_lateral_branch = true;
        state.current_target_direction = target_increment_direction;
        state.protocol_branch_id = 1;
        state.reversal_index = 0;
        state.branch_step_index = 1;
    } else if (state.has_lateral_branch &&
               target_increment_direction != 0 &&
               target_increment_direction != state.current_target_direction) {
        state.current_target_direction = target_increment_direction;
        state.protocol_branch_id += 1;
        state.reversal_index += 1;
        state.branch_step_index = 1;
    } else if (state.has_lateral_branch) {
        state.branch_step_index += 1;
    }

    state.previous_target_drift = target_drift;
    state.previous_actual_drift = actual_drift;

    return {
        .target_increment_direction = target_increment_direction,
        .actual_increment_direction = actual_increment_direction,
        .protocol_branch_id = state.protocol_branch_id,
        .reversal_index = state.reversal_index,
        .branch_step_index = state.branch_step_index,
    };
}

struct ReducedRCColumnStructuralSolveOutcome {
    bool success{false};
    int failed_runtime_step{0};
    double failed_runtime_target{0.0};
};

[[nodiscard]] int effective_segment_substep_factor(
    const ReducedRCColumnStructuralRunSpec& spec) noexcept
{
    return std::max(spec.continuation_segment_substep_factor, 1);
}

[[nodiscard]] bool is_monotonic_protocol(
    const table_cyclic_validation::CyclicValidationRunConfig& cfg) noexcept
{
    return cfg.protocol_name == "monotonic" ||
           cfg.protocol_name.starts_with("monotonic_");
}

[[nodiscard]] int runtime_lateral_step_count(
    const ReducedRCColumnStructuralRunSpec& spec,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg) noexcept
{
    if (is_monotonic_protocol(cfg)) {
        return std::max(cfg.steps_per_segment, 1);
    }

    int runtime_steps = cfg.total_steps();
    if (spec.continuation_kind ==
        ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control) {
        runtime_steps *= effective_segment_substep_factor(spec);
    }

    return std::max(runtime_steps, 0);
}

[[nodiscard]] double target_drift_at(
    const table_cyclic_validation::CyclicValidationRunConfig& cfg,
    double p) noexcept
{
    if (is_monotonic_protocol(cfg)) {
        return p * cfg.max_amplitude_m();
    }
    return cfg.displacement(p);
}

[[nodiscard]] double target_top_bending_rotation(
    const ReducedRCColumnStructuralRunSpec& spec,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg,
    double lateral_p) noexcept
{
    if (!spec.prescribes_top_bending_rotation()) {
        return 0.0;
    }
    if (spec.clamp_top_bending_rotation) {
        return 0.0;
    }
    const double height =
        std::max(spec.reference_spec.column_height_m,
                 std::numeric_limits<double>::epsilon());
    return spec.top_bending_rotation_drift_ratio *
           target_drift_at(cfg, lateral_p) / height;
}

[[nodiscard]] Material<TimoshenkoBeam3D>
make_structural_section_material(const ReducedRCColumnStructuralRunSpec& spec)
{
    const auto section_spec = to_rc_column_section_spec(spec.reference_spec);
    switch (spec.material_mode) {
        case ReducedRCColumnStructuralMaterialMode::nonlinear:
            return make_rc_column_section(section_spec);
        case ReducedRCColumnStructuralMaterialMode::elasticized:
            return make_rc_column_section_elasticized(section_spec);
    }

    throw std::invalid_argument(
        "Reduced structural column baseline received an unknown section "
        "material mode.");
}

[[nodiscard]] std::vector<double> build_runtime_targets(
    const ReducedRCColumnStructuralRunSpec& spec,
    const ReducedRCColumnControlPath& control_path,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    std::vector<double> targets;

    if (control_path.total_runtime_steps() <= 0) {
        return targets;
    }

    const auto append_uniform_targets =
        [&](int count, int offset, int total) {
            for (int i = 1; i <= count; ++i) {
                targets.push_back(
                    static_cast<double>(offset + i) / static_cast<double>(total));
            }
        };

    if (spec.continuation_kind ==
        ReducedRCColumnContinuationKind::
            monolithic_incremental_displacement_control) {
        append_uniform_targets(
            control_path.total_runtime_steps(), 0, control_path.total_runtime_steps());
        return targets;
    }

    if (spec.continuation_kind ==
        ReducedRCColumnContinuationKind::arc_length_continuation_candidate) {
        throw std::invalid_argument(
            "Reduced RC column structural baseline does not yet expose an "
            "arc-length continuation path for the displacement-driven "
            "TimoshenkoBeamN validation slice. Keep the benchmark on the "
            "audited displacement-control path and only escalate to "
            "arc-length after introducing a dedicated reduced-column "
            "continuation wrapper.");
    }

    if (is_monotonic_protocol(cfg)) {
        const int total_runtime_steps = control_path.total_runtime_steps();
        targets.reserve(static_cast<std::size_t>(total_runtime_steps));
        append_uniform_targets(
            control_path.axial_preload_steps, 0, total_runtime_steps);
        append_uniform_targets(
            control_path.lateral_steps,
            control_path.axial_preload_steps,
            total_runtime_steps);
        return targets;
    }

    const int cyclic_segments =
        fall_n::cyclic_segment_count(cfg.amplitudes_m.size());
    const int baseline_segment_steps = std::max(cfg.steps_per_segment, 1);

    int continuation_segment_steps = baseline_segment_steps;
    if (spec.continuation_kind ==
        ReducedRCColumnContinuationKind::
            reversal_guarded_incremental_displacement_control) {
        continuation_segment_steps *= effective_segment_substep_factor(spec);
    }

    const int lateral_runtime_steps =
        cyclic_segments * continuation_segment_steps;
    const int total_runtime_steps =
        control_path.axial_preload_steps + lateral_runtime_steps;

    targets.reserve(static_cast<std::size_t>(total_runtime_steps));
    append_uniform_targets(
        control_path.axial_preload_steps, 0, total_runtime_steps);

    const double preload_fraction =
        control_path.axial_preload_steps > 0
            ? static_cast<double>(control_path.axial_preload_steps) /
                  static_cast<double>(total_runtime_steps)
            : 0.0;

    for (int seg = 0; seg < cyclic_segments; ++seg) {
        for (int local = 1; local <= continuation_segment_steps; ++local) {
            const double lateral_p =
                (static_cast<double>(seg) +
                 static_cast<double>(local) /
                     static_cast<double>(continuation_segment_steps)) /
                static_cast<double>(cyclic_segments);
            targets.push_back(
                preload_fraction + (1.0 - preload_fraction) * lateral_p);
        }
    }

    return targets;
}

template <typename AnalysisT, typename SchemeT>
[[nodiscard]] ReducedRCColumnStructuralSolveOutcome solve_structural_runtime_targets(
    AnalysisT& nl,
    SchemeT&& scheme,
    const std::vector<double>& runtime_targets,
    int max_bisections)
{
    if (runtime_targets.empty()) {
        return {.success = true};
    }

    nl.begin_incremental(1, max_bisections, std::forward<SchemeT>(scheme));

    int runtime_step = 0;
    for (const double target : runtime_targets) {
        ++runtime_step;
        const double delta = target - nl.current_time();
        if (delta <= 1.0e-14) {
            continue;
        }

        nl.set_increment_size(delta);
        const auto verdict = nl.step_to(target);
        if (verdict != fall_n::StepVerdict::Continue) {
            return {
                .success = false,
                .failed_runtime_step = runtime_step,
                .failed_runtime_target = target,
            };
        }
    }

    return {.success = true};
}

void write_hysteresis_csv(
    const std::string& path,
    const std::vector<StepRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,p,drift_m,base_shear_MN\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& r : records) {
        ofs << r.step << ","
            << r.p << ","
            << r.drift << ","
            << r.base_shear << "\n";
    }
    std::println("  CSV: {} ({} records)", path, records.size());
}

template <typename ModelT>
[[nodiscard]] double extract_support_resultant_component(
    const ModelT& model,
    const std::vector<std::size_t>& support_nodes,
    std::size_t component)
{
    Vec f_int;
    VecDuplicate(model.state_vector(), &f_int);
    VecSet(f_int, 0.0);

    auto& mut_model = const_cast<ModelT&>(model);
    for (auto& elem : mut_model.elements()) {
        elem.compute_internal_forces(model.state_vector(), f_int);
    }
    VecAssemblyBegin(f_int);
    VecAssemblyEnd(f_int);

    double shear = 0.0;
    for (auto nid : support_nodes) {
        PetscScalar val{};
        PetscInt idx = static_cast<PetscInt>(
            model.get_domain().node(nid).dof_index()[component]);
        VecGetValues(f_int, 1, &idx, &val);
        shear += val;
    }

    VecDestroy(&f_int);
    return shear;
}

void write_section_response_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnSectionResponseRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,p,drift_m,section_gp,xi,axial_strain,curvature_y,curvature_z,"
           "axial_force_MN,moment_y_MNm,moment_z_MNm,tangent_ea,"
           "tangent_eiy,tangent_eiz,tangent_eiy_direct_raw,"
           "tangent_eiz_direct_raw,raw_k00,raw_k0y,raw_ky0,raw_kyy\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& r : records) {
        ofs << r.step << ","
            << r.p << ","
            << r.drift << ","
            << r.section_gp << ","
            << r.xi << ","
            << r.axial_strain << ","
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
            << r.raw_tangent_k00 << ","
            << r.raw_tangent_k0y << ","
            << r.raw_tangent_ky0 << ","
            << r.raw_tangent_kyy << "\n";
    }
    std::println("  CSV: {} ({} records)", path, records.size());
}

void write_base_side_moment_curvature_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnSectionResponseRecord>& records)
{
    if (records.empty()) {
        return;
    }

    std::size_t controlling_gp = records.front().section_gp;
    double min_xi = records.front().xi;
    for (const auto& r : records) {
        if (r.xi < min_xi || (r.xi == min_xi && r.section_gp < controlling_gp)) {
            min_xi = r.xi;
            controlling_gp = r.section_gp;
        }
    }

    std::ofstream ofs(path);
    ofs << "step,p,drift_m,section_gp,xi,curvature_y,moment_y_MNm,"
           "axial_force_MN,tangent_eiy\n";
    ofs << std::scientific << std::setprecision(8);
    std::size_t written = 0;
    for (const auto& r : records) {
        if (r.section_gp != controlling_gp) {
            continue;
        }
        ofs << r.step << ","
            << r.p << ","
            << r.drift << ","
            << r.section_gp << ","
            << r.xi << ","
            << r.curvature_y << ","
            << r.moment_y << ","
            << r.axial_force << ","
            << r.tangent_eiy << "\n";
        ++written;
    }
    std::println(
        "  CSV: {} ({} records, base-side section_gp={}, xi={:+.6f})",
        path,
        written,
        controlling_gp,
        min_xi);
}

void write_section_fiber_history_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnStructuralSectionFiberRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "step,p,drift_m,section_gp,xi,axial_strain,curvature_y,"
           "zero_curvature_anchor,fiber_index,y,z,area,zone,material_role,"
           "strain_xx,stress_xx_MPa,tangent_xx_MPa,axial_force_contribution_MN,"
           "moment_y_contribution_MNm,raw_k00_contribution,"
           "raw_k0y_contribution,raw_kyy_contribution,"
           "history_state_code,history_min_strain,history_min_stress_MPa,"
           "history_closure_strain,history_max_tensile_strain,"
           "history_max_tensile_stress_MPa,history_committed_strain,"
           "history_committed_stress_MPa,history_cracked\n";
    ofs << std::scientific << std::setprecision(8);
    for (const auto& row : records) {
        ofs << row.step << ","
            << row.p << ","
            << row.drift << ","
            << row.section_gp << ","
            << row.xi << ","
            << row.axial_strain << ","
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
            << row.raw_tangent_kyy_contribution << ","
            << row.history_state_code << ","
            << row.history_min_strain << ","
            << row.history_min_stress << ","
            << row.history_closure_strain << ","
            << row.history_max_tensile_strain << ","
            << row.history_max_tensile_stress << ","
            << row.history_committed_strain << ","
            << row.history_committed_stress << ","
            << (row.history_cracked ? 1 : 0) << "\n";
    }
    std::println("  CSV: {} ({} fiber records)", path, records.size());
}

void write_element_tangent_audit_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnStructuralElementTangentAuditRecord>& records)
{
    std::ofstream out(path);
    out << "runtime_step,step,p,runtime_p,drift_m,failed_attempt,"
           "fd_step_reference,displacement_inf_norm,internal_force_norm,"
           "tangent_frobenius_norm,fd_tangent_frobenius_norm,"
           "tangent_fd_rel_error,max_column_rel_error,worst_column_index,"
           "worst_column_node,worst_column_local_dof,worst_column_tangent_norm,"
           "worst_column_fd_norm,top_control_column_rel_error,"
           "top_bending_rotation_column_rel_error\n";
    out << std::setprecision(16);

    for (const auto& row : records) {
        out << row.runtime_step << ","
            << row.step << ","
            << row.p << ","
            << row.runtime_p << ","
            << row.drift << ","
            << (row.failed_attempt ? 1 : 0) << ","
            << row.fd_step_reference << ","
            << row.displacement_inf_norm << ","
            << row.internal_force_norm << ","
            << row.tangent_frobenius_norm << ","
            << row.fd_tangent_frobenius_norm << ","
            << row.tangent_fd_rel_error << ","
            << row.max_column_rel_error << ","
            << row.worst_column_index << ","
            << row.worst_column_node << ","
            << row.worst_column_local_dof << ","
            << row.worst_column_tangent_norm << ","
            << row.worst_column_fd_norm << ","
            << row.top_control_column_rel_error << ","
            << row.top_bending_rotation_column_rel_error << "\n";
    }

    std::println("  CSV: {} ({} tangent-audit records)", path, records.size());
}

void write_section_tangent_audit_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnStructuralSectionTangentAuditRecord>& records)
{
    std::ofstream ofs(path);
    ofs << "runtime_step,step,p,runtime_p,drift_m,failed_attempt,section_gp,xi,"
           "axial_strain,curvature_y,curvature_z,fd_step_reference,"
           "tangent_frobenius_norm,fd_tangent_frobenius_norm,tangent_fd_rel_error,"
           "max_column_rel_error,worst_column_index,axial_column_rel_error,"
           "curvature_y_column_rel_error,shear_z_column_rel_error,"
           "raw_tangent_k00,raw_tangent_k0y,raw_tangent_ky0,raw_tangent_kyy,"
           "fd_tangent_k00,fd_tangent_k0y,fd_tangent_ky0,fd_tangent_kyy,"
           "rel_error_k00,rel_error_k0y,rel_error_ky0,rel_error_kyy\n";
    ofs << std::scientific << std::setprecision(12);

    for (const auto& row : records) {
        ofs << row.runtime_step << ","
            << row.step << ","
            << row.p << ","
            << row.runtime_p << ","
            << row.drift << ","
            << (row.failed_attempt ? 1 : 0) << ","
            << row.section_gp << ","
            << row.xi << ","
            << row.axial_strain << ","
            << row.curvature_y << ","
            << row.curvature_z << ","
            << row.fd_step_reference << ","
            << row.tangent_frobenius_norm << ","
            << row.fd_tangent_frobenius_norm << ","
            << row.tangent_fd_rel_error << ","
            << row.max_column_rel_error << ","
            << row.worst_column_index << ","
            << row.axial_column_rel_error << ","
            << row.curvature_y_column_rel_error << ","
            << row.shear_z_column_rel_error << ","
            << row.raw_tangent_k00 << ","
            << row.raw_tangent_k0y << ","
            << row.raw_tangent_ky0 << ","
            << row.raw_tangent_kyy << ","
            << row.fd_tangent_k00 << ","
            << row.fd_tangent_k0y << ","
            << row.fd_tangent_ky0 << ","
            << row.fd_tangent_kyy << ","
            << row.rel_error_k00 << ","
            << row.rel_error_k0y << ","
            << row.rel_error_ky0 << ","
            << row.rel_error_kyy << "\n";
    }

    std::println("  CSV: {} ({} section-tangent records)", path, records.size());
}

template <typename BeamModelT>
[[nodiscard]] std::size_t total_structural_section_station_count(
    const BeamModelT& model) noexcept
{
    std::size_t count = 0;
    for (const auto& beam : model.elements()) {
        count += beam.num_integration_points();
    }
    return count;
}

template <typename BeamT>
[[nodiscard]] double global_beam_axis_xi_at_gp(
    const BeamT& beam,
    std::size_t gp,
    double column_height_m) noexcept
{
    const auto xi_view = beam.geometry().reference_integration_point(gp);
    const auto point = beam.geometry().map_local_point(xi_view);
    if (!std::isfinite(column_height_m) || column_height_m <= 0.0) {
        return xi_view[0];
    }
    return 2.0 * point[2] / column_height_m - 1.0;
}

template <typename BeamModelT>
std::vector<ReducedRCColumnSectionResponseRecord>
extract_section_response_records(
    const BeamModelT& model,
    double column_height_m,
    int step,
    double p,
    double drift)
{
    std::vector<ReducedRCColumnSectionResponseRecord> records;
    records.reserve(total_structural_section_station_count(model));

    const auto condensed_bending_tangent =
        [](const Eigen::Matrix<double, 6, 6>& tangent,
           int bending_index) noexcept {
            const double axial_tangent = tangent(0, 0);
            if (!std::isfinite(axial_tangent) ||
                std::abs(axial_tangent) < 1.0e-12) {
                return tangent(bending_index, bending_index);
            }

            return tangent(bending_index, bending_index) -
                   tangent(bending_index, 0) * tangent(0, bending_index) /
                       axial_tangent;
        };

    std::size_t section_gp = 0;
    for (const auto& beam : model.elements()) {
        const auto u_loc = beam.local_state_vector(model.state_vector());
        for (std::size_t gp = 0; gp < beam.num_integration_points(); ++gp) {
            const auto strain = beam.sample_generalized_strain_at_gp(gp, u_loc);
            const auto resultant = beam.sample_resultants_at_gp(gp, u_loc);
            const auto tangent = beam.sections()[gp].tangent(strain);

            records.push_back({
                .step = step,
                .p = p,
                .drift = drift,
                .section_gp = section_gp++,
                .xi = global_beam_axis_xi_at_gp(beam, gp, column_height_m),
                .axial_strain = strain.axial_strain(),
                .curvature_y = strain.curvature_y(),
                .curvature_z = strain.curvature_z(),
                .axial_force = resultant.axial_force(),
                .moment_y = resultant.moment_y(),
                .moment_z = resultant.moment_z(),
                .tangent_ea = tangent(0, 0),
                .tangent_eiy = condensed_bending_tangent(tangent, 1),
                .tangent_eiz = condensed_bending_tangent(tangent, 2),
                .tangent_eiy_direct_raw = tangent(1, 1),
                .tangent_eiz_direct_raw = tangent(2, 2),
                .raw_tangent_k00 = tangent(0, 0),
                .raw_tangent_k0y = tangent(0, 1),
                .raw_tangent_ky0 = tangent(1, 0),
                .raw_tangent_kyy = tangent(1, 1),
            });
        }
    }

    return records;
}

template <typename BeamModelT>
std::vector<ReducedRCColumnStructuralSectionFiberRecord>
extract_section_fiber_history_records(
    const BeamModelT& model,
    const std::vector<RCSectionFiberLayoutRecord>& layout,
    double column_height_m,
    int step,
    double p,
    double drift)
{
    std::vector<ReducedRCColumnStructuralSectionFiberRecord> records;
    records.reserve(total_structural_section_station_count(model) * layout.size());

    std::size_t section_gp = 0;
    for (const auto& beam : model.elements()) {
        const auto u_loc = beam.local_state_vector(model.state_vector());
        for (std::size_t gp = 0; gp < beam.num_integration_points(); ++gp) {
            const auto strain = beam.sample_generalized_strain_at_gp(gp, u_loc);
            const auto snapshot = beam.sections()[gp].section_snapshot();
            if (!snapshot.has_fibers()) {
                throw std::runtime_error(
                    "Reduced RC structural baseline expected fiber-resolved "
                    "section snapshots for the structural beam slice.");
            }
            if (snapshot.fibers.size() != layout.size()) {
                throw std::runtime_error(
                    "Reduced RC structural baseline found a mismatch between the "
                    "canonical RC fiber layout and the structural section "
                    "snapshot fiber count.");
            }
            const auto global_section_gp = section_gp++;
            const auto global_xi =
                global_beam_axis_xi_at_gp(beam, gp, column_height_m);

            for (std::size_t i = 0; i < layout.size(); ++i) {
                const auto& fiber_layout = layout[i];
                const auto& fiber_state = snapshot.fibers[i];
                const double axial_force_contribution =
                    fiber_state.stress_xx * fiber_state.area;
                const double moment_y_contribution =
                    -fiber_state.stress_xx * fiber_state.z * fiber_state.area;
                const double raw_tangent_k00_contribution =
                    fiber_state.tangent_xx * fiber_state.area;
                const double raw_tangent_k0y_contribution =
                    -fiber_state.tangent_xx * fiber_state.z * fiber_state.area;
                const double raw_tangent_kyy_contribution =
                    fiber_state.tangent_xx * fiber_state.z * fiber_state.z *
                    fiber_state.area;
                const auto& internal = fiber_state.internal_fields;

                records.push_back({
                    .step = step,
                    .p = p,
                    .drift = drift,
                    .section_gp = global_section_gp,
                    .xi = global_xi,
                    .axial_strain = strain.axial_strain(),
                    .curvature_y = strain.curvature_y(),
                    .zero_curvature_anchor =
                        std::abs(strain.curvature_y()) <= 1.0e-12,
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
                    .history_state_code = internal.history_state_code.value_or(0),
                    .history_min_strain = internal.history_min_strain.value_or(0.0),
                    .history_min_stress = internal.history_min_stress.value_or(0.0),
                    .history_closure_strain =
                        internal.history_closure_strain.value_or(0.0),
                    .history_max_tensile_strain =
                        internal.history_max_tensile_strain.value_or(0.0),
                    .history_max_tensile_stress =
                        internal.history_max_tensile_stress.value_or(0.0),
                    .history_committed_strain =
                        internal.history_committed_strain.value_or(0.0),
                    .history_committed_stress =
                        internal.history_committed_stress.value_or(0.0),
                    .history_cracked = internal.history_cracked.value_or(false),
                });
            }
        }
    }

    return records;
}

template <typename T>
void append_records(std::vector<T>& into, const std::vector<T>& extra)
{
    into.insert(into.end(), extra.begin(), extra.end());
}

template <typename SectionT>
[[nodiscard]] auto evaluate_section_response_from_frozen_state(
    const SectionT& section,
    const typename SectionT::ConstitutiveSpace::StateVariableT& strain)
{
    auto frozen_section = section;
    return frozen_section.compute_response(strain);
}

template <typename SectionT>
[[nodiscard]] auto evaluate_section_tangent_from_frozen_state(
    const SectionT& section,
    const typename SectionT::ConstitutiveSpace::StateVariableT& strain)
{
    auto frozen_section = section;
    return frozen_section.tangent(strain);
}

template <typename BeamT>
[[nodiscard]] Eigen::VectorXd evaluate_element_internal_force_from_frozen_state(
    const BeamT& beam,
    const Eigen::VectorXd& u_e)
{
    auto frozen_beam = beam;
    return frozen_beam.compute_internal_force_vector(u_e);
}

template <typename BeamT>
[[nodiscard]] Eigen::MatrixXd evaluate_element_tangent_from_frozen_state(
    const BeamT& beam,
    const Eigen::VectorXd& u_e)
{
    auto frozen_beam = beam;
    return frozen_beam.compute_tangent_stiffness_matrix(u_e);
}

template <typename BeamModelT>
[[nodiscard]] std::vector<ReducedRCColumnStructuralSectionTangentAuditRecord>
make_section_tangent_audit_records(
    const BeamModelT& model,
    double column_height_m,
    int runtime_step,
    int logical_step,
    double runtime_p,
    double logical_p,
    double drift,
    bool failed_attempt) noexcept
{
    std::vector<ReducedRCColumnStructuralSectionTangentAuditRecord> records;
    records.reserve(total_structural_section_station_count(model));

    std::size_t section_gp = 0;
    for (const auto& beam : model.elements()) {
        const auto u_loc = beam.local_state_vector(model.state_vector());
        for (std::size_t gp = 0; gp < beam.num_integration_points(); ++gp) {
            const auto strain = beam.sample_generalized_strain_at_gp(gp, u_loc);
            const auto tangent =
                evaluate_section_tangent_from_frozen_state(beam.sections()[gp], strain);

            Eigen::Matrix<double, 6, 6> fd_tangent =
                Eigen::Matrix<double, 6, 6>::Zero();
            const double strain_inf = strain.components().cwiseAbs().maxCoeff();
            const double reference_step = 1.0e-7 * std::max(1.0, strain_inf);

            double max_column_rel_error = 0.0;
            int worst_column_index = 0;

            for (int j = 0; j < tangent.cols(); ++j) {
                const double h =
                    1.0e-7 * std::max(1.0, std::abs(strain.components()[j]));
                auto strain_plus = strain;
                auto strain_minus = strain;
                Eigen::Vector<double, 6> plus_components = strain.components();
                Eigen::Vector<double, 6> minus_components = strain.components();
                plus_components[j] += h;
                minus_components[j] -= h;
                strain_plus.set_components(plus_components);
                strain_minus.set_components(minus_components);

                const auto response_plus =
                    evaluate_section_response_from_frozen_state(
                        beam.sections()[gp], strain_plus);
                const auto response_minus =
                    evaluate_section_response_from_frozen_state(
                        beam.sections()[gp], strain_minus);
                const auto fd_column =
                    (response_plus.components() - response_minus.components()) /
                    (2.0 * h);
                fd_tangent.col(j) = fd_column;

                const double analytic_norm = tangent.col(j).norm();
                const double fd_norm = fd_column.norm();
                const double rel_error =
                    (tangent.col(j) - fd_column).norm() /
                    std::max({1.0, analytic_norm, fd_norm});

                if (rel_error > max_column_rel_error) {
                    max_column_rel_error = rel_error;
                    worst_column_index = j;
                }
            }

            const auto column_rel_error = [&](int j) noexcept {
                return (tangent.col(j) - fd_tangent.col(j)).norm() /
                       std::max(
                           {1.0, tangent.col(j).norm(), fd_tangent.col(j).norm()});
            };

            const double matrix_rel_error =
                (tangent - fd_tangent).norm() /
                std::max({1.0, tangent.norm(), fd_tangent.norm()});

            records.push_back({
                .runtime_step = runtime_step,
                .step = logical_step,
                .p = logical_p,
                .runtime_p = runtime_p,
                .drift = drift,
                .failed_attempt = failed_attempt,
                .section_gp = section_gp++,
                .xi = global_beam_axis_xi_at_gp(beam, gp, column_height_m),
                .axial_strain = strain.axial_strain(),
                .curvature_y = strain.curvature_y(),
                .curvature_z = strain.curvature_z(),
                .fd_step_reference = reference_step,
                .tangent_frobenius_norm = tangent.norm(),
                .fd_tangent_frobenius_norm = fd_tangent.norm(),
                .tangent_fd_rel_error = matrix_rel_error,
                .max_column_rel_error = max_column_rel_error,
                .worst_column_index = worst_column_index,
                .axial_column_rel_error = column_rel_error(0),
                .curvature_y_column_rel_error = column_rel_error(1),
                .shear_z_column_rel_error = column_rel_error(4),
                .raw_tangent_k00 = tangent(0, 0),
                .raw_tangent_k0y = tangent(0, 1),
                .raw_tangent_ky0 = tangent(1, 0),
                .raw_tangent_kyy = tangent(1, 1),
                .fd_tangent_k00 = fd_tangent(0, 0),
                .fd_tangent_k0y = fd_tangent(0, 1),
                .fd_tangent_ky0 = fd_tangent(1, 0),
                .fd_tangent_kyy = fd_tangent(1, 1),
                .rel_error_k00 =
                    std::abs(tangent(0, 0) - fd_tangent(0, 0)) /
                    std::max({1.0, std::abs(tangent(0, 0)), std::abs(fd_tangent(0, 0))}),
                .rel_error_k0y =
                    std::abs(tangent(0, 1) - fd_tangent(0, 1)) /
                    std::max({1.0, std::abs(tangent(0, 1)), std::abs(fd_tangent(0, 1))}),
                .rel_error_ky0 =
                    std::abs(tangent(1, 0) - fd_tangent(1, 0)) /
                    std::max({1.0, std::abs(tangent(1, 0)), std::abs(fd_tangent(1, 0))}),
                .rel_error_kyy =
                    std::abs(tangent(1, 1) - fd_tangent(1, 1)) /
                    std::max({1.0, std::abs(tangent(1, 1)), std::abs(fd_tangent(1, 1))}),
            });
        }
    }

    return records;
}

template <typename BeamModelT>
[[nodiscard]] ReducedRCColumnStructuralElementTangentAuditRecord
make_element_tangent_audit_record(
    const BeamModelT& model,
    int runtime_step,
    int logical_step,
    double runtime_p,
    double logical_p,
    double drift,
    bool failed_attempt) noexcept
{
    constexpr int dofs_per_node = 6;

    const auto& beam = model.elements().front();
    const Eigen::VectorXd u_e = beam.extract_element_dofs(model.state_vector());
    const Eigen::VectorXd f_ref =
        evaluate_element_internal_force_from_frozen_state(beam, u_e);
    const Eigen::MatrixXd K_ref =
        evaluate_element_tangent_from_frozen_state(beam, u_e);

    const int n = static_cast<int>(u_e.size());
    Eigen::MatrixXd K_fd(n, n);
    K_fd.setZero();

    const double u_inf = u_e.cwiseAbs().maxCoeff();
    const double reference_step = 1.0e-7 * std::max(1.0, u_inf);

    double max_column_rel_error = 0.0;
    int worst_column_index = 0;
    double worst_column_tangent_norm = 0.0;
    double worst_column_fd_norm = 0.0;

    for (int j = 0; j < n; ++j) {
        const double h = 1.0e-7 * std::max(1.0, std::abs(u_e[j]));
        Eigen::VectorXd u_plus = u_e;
        Eigen::VectorXd u_minus = u_e;
        u_plus[j] += h;
        u_minus[j] -= h;

        const Eigen::VectorXd f_plus =
            evaluate_element_internal_force_from_frozen_state(beam, u_plus);
        const Eigen::VectorXd f_minus =
            evaluate_element_internal_force_from_frozen_state(beam, u_minus);
        const Eigen::VectorXd fd_col = (f_plus - f_minus) / (2.0 * h);
        K_fd.col(j) = fd_col;

        const double analytic_norm = K_ref.col(j).norm();
        const double fd_norm = fd_col.norm();
        const double rel_error =
            (K_ref.col(j) - fd_col).norm() /
            std::max({1.0, analytic_norm, fd_norm});

        if (rel_error > max_column_rel_error) {
            max_column_rel_error = rel_error;
            worst_column_index = j;
            worst_column_tangent_norm = analytic_norm;
            worst_column_fd_norm = fd_norm;
        }
    }

    const double matrix_rel_error =
        (K_ref - K_fd).norm() /
        std::max({1.0, K_ref.norm(), K_fd.norm()});

    const int top_control_column = std::max(n - dofs_per_node, 0);
    const int top_bending_rotation_column =
        std::min(std::max(n - 2, 0), n - 1);

    const auto column_rel_error = [&](int j) noexcept {
        if (j < 0 || j >= n) {
            return 0.0;
        }
        return (K_ref.col(j) - K_fd.col(j)).norm() /
               std::max({1.0, K_ref.col(j).norm(), K_fd.col(j).norm()});
    };

    return {
        .runtime_step = runtime_step,
        .step = logical_step,
        .p = logical_p,
        .runtime_p = runtime_p,
        .drift = drift,
        .failed_attempt = failed_attempt,
        .fd_step_reference = reference_step,
        .displacement_inf_norm = u_inf,
        .internal_force_norm = f_ref.norm(),
        .tangent_frobenius_norm = K_ref.norm(),
        .fd_tangent_frobenius_norm = K_fd.norm(),
        .tangent_fd_rel_error = matrix_rel_error,
        .max_column_rel_error = max_column_rel_error,
        .worst_column_index = worst_column_index,
        .worst_column_node = worst_column_index / dofs_per_node,
        .worst_column_local_dof = worst_column_index % dofs_per_node,
        .worst_column_tangent_norm = worst_column_tangent_norm,
        .worst_column_fd_norm = worst_column_fd_norm,
        .top_control_column_rel_error = column_rel_error(top_control_column),
        .top_bending_rotation_column_rel_error =
            column_rel_error(top_bending_rotation_column),
    };
}

template <typename IncrementDiagT>
[[nodiscard]] ReducedRCColumnStructuralControlStateRecord make_control_state_record(
    int runtime_step,
    int logical_step,
    double runtime_p,
    double logical_p,
    double target_drift,
    double actual_tip_lateral_displacement,
    double actual_tip_lateral_total_state_displacement,
    double prescribed_top_bending_rotation,
    double actual_top_bending_rotation,
    double top_axial_displacement,
    double base_shear,
    double base_axial_reaction,
    bool preload_equilibrated,
    const ReducedRCColumnControlTraceSnapshot& trace_snapshot,
    const IncrementDiagT& solver_diag,
    bool converged) noexcept
{
    const double newton_iterations_per_substep =
        solver_diag.accepted_substep_count > 0
            ? static_cast<double>(solver_diag.total_newton_iterations) /
                  static_cast<double>(solver_diag.accepted_substep_count)
            : 0.0;

    return {
        .runtime_step = runtime_step,
        .step = logical_step,
        .p = logical_p,
        .runtime_p = runtime_p,
        .target_drift = target_drift,
        .actual_tip_lateral_displacement = actual_tip_lateral_displacement,
        .actual_tip_lateral_total_state_displacement =
            actual_tip_lateral_total_state_displacement,
        .imposed_vs_total_state_tip_displacement_error =
            std::abs(actual_tip_lateral_displacement -
                     actual_tip_lateral_total_state_displacement),
        .prescribed_top_bending_rotation = prescribed_top_bending_rotation,
        .actual_top_bending_rotation = actual_top_bending_rotation,
        .imposed_vs_total_state_top_bending_rotation_error =
            std::abs(prescribed_top_bending_rotation -
                     actual_top_bending_rotation),
        .top_axial_displacement = top_axial_displacement,
        .base_shear = base_shear,
        .base_axial_reaction = base_axial_reaction,
        .preload_equilibrated = preload_equilibrated,
        .target_increment_direction = trace_snapshot.target_increment_direction,
        .actual_increment_direction = trace_snapshot.actual_increment_direction,
        .protocol_branch_id = trace_snapshot.protocol_branch_id,
        .reversal_index = trace_snapshot.reversal_index,
        .branch_step_index = trace_snapshot.branch_step_index,
        .accepted_substep_count = solver_diag.accepted_substep_count,
        .max_bisection_level = solver_diag.max_bisection_level,
        .newton_iterations =
            static_cast<double>(solver_diag.total_newton_iterations),
        .newton_iterations_per_substep = newton_iterations_per_substep,
        .solver_profile_attempt_count = solver_diag.solver_profile_attempt_count,
        .solver_profile_label = solver_diag.last_solver_profile_label,
        .solver_snes_type = solver_diag.last_solver_snes_type,
        .solver_linesearch_type = solver_diag.last_solver_linesearch_type,
        .solver_ksp_type = solver_diag.last_solver_ksp_type,
        .solver_pc_type = solver_diag.last_solver_pc_type,
        .last_snes_reason = solver_diag.last_snes_reason,
        .last_function_norm = solver_diag.last_function_norm,
        .accepted_by_small_residual_policy =
            solver_diag.accepted_by_small_residual_policy,
        .accepted_function_norm_threshold =
            solver_diag.accepted_function_norm_threshold,
        .converged = converged,
    };
}

template <typename BeamModelT, typename IncrementDiagT>
void append_runtime_observables(
    ReducedRCColumnStructuralRunResult& result,
    const BeamModelT& model,
    const std::vector<RCSectionFiberLayoutRecord>& section_layout,
    double column_height_m,
    const std::vector<std::size_t>& base_nodes,
    int runtime_step,
    int logical_step,
    double runtime_p,
    double logical_p,
    double target_drift,
    std::size_t top_node,
    bool preload_equilibrated,
    ReducedRCColumnControlTraceState& control_trace_state,
    const IncrementDiagT& solver_diag)
{
    const double actual_tip_lateral_displacement =
        model.prescribed_value(top_node, 0);
    const double actual_tip_lateral_total_state_displacement =
        query::nodal_dof_value(model, model.state_vector(), top_node, 0);
    const double prescribed_top_bending_rotation =
        model.prescribed_value(top_node, 4);
    const double actual_top_bending_rotation =
        query::nodal_dof_value(model, model.state_vector(), top_node, 4);
    const double top_axial_displacement =
        query::nodal_dof_value(model, model.state_vector(), top_node, 2);
    const double base_shear =
        extract_support_resultant_component(model, base_nodes, 0);
    const double base_axial_reaction =
        extract_support_resultant_component(model, base_nodes, 2);
    const auto trace_snapshot = advance_control_trace(
        control_trace_state,
        target_drift,
        actual_tip_lateral_displacement,
        logical_step > 0);

    result.hysteresis_records.push_back({
        logical_step,
        logical_p,
        actual_tip_lateral_displacement,
        base_shear});
    result.control_state_records.push_back(make_control_state_record(
        runtime_step,
        logical_step,
        runtime_p,
        logical_p,
        target_drift,
        actual_tip_lateral_displacement,
        actual_tip_lateral_total_state_displacement,
        prescribed_top_bending_rotation,
        actual_top_bending_rotation,
        top_axial_displacement,
        base_shear,
        base_axial_reaction,
        preload_equilibrated,
        trace_snapshot,
        solver_diag,
        solver_diag.converged));
    append_records(
        result.section_response_records,
        extract_section_response_records(
            model,
            column_height_m,
            logical_step,
            logical_p,
            actual_tip_lateral_displacement));
    append_records(
        result.fiber_history_records,
        extract_section_fiber_history_records(
            model,
            section_layout,
            column_height_m,
            logical_step,
            logical_p,
            actual_tip_lateral_displacement));
}

template <std::size_t N, BeamAxisQuadratureFamily QuadratureFamily>
[[nodiscard]] ReducedRCColumnStructuralRunResult
run_reduced_rc_column_small_strain_beam_case_impl(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    StopWatch total_timer;
    total_timer.start();
    StopWatch analysis_timer;
    analysis_timer.start();

    using QuadratureT = BeamAxisQuadratureT<QuadratureFamily, N - 1>;
    using BeamElemT = TimoshenkoBeamN<N>;
    using BeamPolicy = SingleElementPolicy<BeamElemT>;
    using BeamModel =
        Model<TimoshenkoBeam3D, continuum::SmallStrain, kReducedRCColumnNDoF, BeamPolicy>;

    const auto& reference_spec = spec.reference_spec;

    Domain<3> domain;
    PetscInt tag = 0;
    const auto structural_element_count =
        std::max<std::size_t>(spec.structural_element_count, 1);
    const auto total_centerline_intervals =
        structural_element_count * (N - 1);
    const auto total_centerline_nodes = total_centerline_intervals + 1;

    for (std::size_t i = 0; i < total_centerline_nodes; ++i) {
        const double z =
            reference_spec.column_height_m *
            static_cast<double>(i) /
            static_cast<double>(total_centerline_intervals);
        domain.add_node(static_cast<PetscInt>(i), 0.0, 0.0, z);
    }

    for (std::size_t e = 0; e < structural_element_count; ++e) {
        PetscInt conn[N];
        const auto first_node = e * (N - 1);
        for (std::size_t i = 0; i < N; ++i) {
            conn[i] = static_cast<PetscInt>(first_node + i);
        }

        auto& geom = domain.template make_element<LagrangeElement3D<N>>(
            QuadratureT{}, tag++, conn);
        geom.set_physical_group("ReducedRCColumn");
    }

    domain.assemble_sieve();

    const auto col_mat = make_structural_section_material(spec);
    const auto section_layout =
        build_rc_column_fiber_layout(to_rc_column_section_spec(reference_spec));

    std::vector<BeamElemT> elements;
    elements.reserve(structural_element_count);
    for (auto& geom : domain.elements()) {
        elements.emplace_back(&geom, col_mat);
    }

    BeamModel model{domain, std::move(elements)};
    model.constrain_node(0, {0, 0, 0, 0, 0, 0});

    const std::size_t top_node = total_centerline_nodes - 1;
    model.constrain_dof(top_node, 0, 0.0);
    if (spec.prescribes_top_bending_rotation()) {
        // On the current reduced-column beam slice, the column axis is z and
        // the imposed lateral drift is x, so the corresponding bending
        // rotation is theta_y (DOF 4).
        model.constrain_dof(top_node, 4, 0.0);
    }
    model.setup();

    if (spec.axial_compression_force_mn != 0.0) {
        model.apply_node_force(
            top_node, 0.0, 0.0, -spec.axial_compression_force_mn, 0.0, 0.0, 0.0);
    }

    using AnalysisT =
        NonlinearAnalysis<TimoshenkoBeam3D,
                          continuum::SmallStrain,
                          kReducedRCColumnNDoF,
                          BeamPolicy>;

    AnalysisT nl{&model};
    nl.set_incremental_logging(spec.print_progress);
    nl.set_solve_profiles(
        make_reduced_rc_validation_solve_profiles(spec.solver_policy_kind));

    const int runtime_lateral_steps = runtime_lateral_step_count(spec, cfg);

    const ReducedRCColumnControlPath control_path{
        .lateral_steps = runtime_lateral_steps,
        .axial_preload_steps =
            spec.uses_equilibrated_axial_preload_stage() ? spec.axial_preload_steps : 0};

    if (control_path.total_runtime_steps() <= 0) {
        throw std::invalid_argument(
            "Reduced structural column baseline requires a strictly positive "
            "number of lateral or preload runtime steps.");
    }

    {
        using Adapt = typename decltype(nl)::IncrementAdaptationSettings;
        const auto nominal_increment =
            1.0 / static_cast<double>(control_path.total_runtime_steps());
        nl.set_increment_adaptation(Adapt{
            .enabled = true,
            .min_increment_size =
                std::ldexp(nominal_increment,
                           -(std::max(cfg.max_bisections, 0) + 3)),
            .max_increment_size = nominal_increment,
            .cutback_factor = 0.5,
            .growth_factor = 1.15,
            .max_cutbacks_per_step = std::max(8, cfg.max_bisections * 2),
            .easy_newton_iterations = 6,
            .difficult_newton_iterations = 12,
            .easy_steps_before_growth = 2,
        });
    }

    ReducedRCColumnStructuralRunResult result;
    ReducedRCColumnControlTraceState control_trace_state{};
    ReducedRCColumnFailedTrialCapture failed_trial_capture{};
    result.hysteresis_records.reserve(
        static_cast<std::size_t>(control_path.lateral_steps) + 1u);
    const auto section_station_count =
        structural_element_count * std::max<std::size_t>(1, N - 1);
    if (spec.write_element_tangent_audit_csv) {
        result.element_tangent_audit_records.reserve(
            static_cast<std::size_t>(control_path.total_runtime_steps()) + 1u);
        result.section_tangent_audit_records.reserve(
            (static_cast<std::size_t>(control_path.total_runtime_steps()) + 1u) *
            section_station_count);
    }
    result.fiber_history_records.reserve(
        static_cast<std::size_t>(control_path.total_runtime_steps() + 1) *
        section_station_count * section_layout.size());
    if (!control_path.has_preload_stage()) {
        result.hysteresis_records.push_back({0, 0.0, 0.0, 0.0});
        result.control_state_records.push_back({
            .runtime_step = 0,
            .step = 0,
            .p = 0.0,
            .runtime_p = 0.0,
            .target_drift = 0.0,
            .actual_tip_lateral_displacement = 0.0,
            .top_axial_displacement = 0.0,
            .base_shear = 0.0,
            .base_axial_reaction = 0.0,
            .preload_equilibrated = false,
            .converged = true,
        });
        append_records(
            result.section_response_records,
            extract_section_response_records(
                model,
                reference_spec.column_height_m,
                0,
                0.0,
                0.0));
        append_records(
            result.fiber_history_records,
            extract_section_fiber_history_records(
                model,
                section_layout,
                reference_spec.column_height_m,
                0,
                0.0,
                0.0));
        if (spec.write_element_tangent_audit_csv) {
            result.element_tangent_audit_records.push_back(
                make_element_tangent_audit_record(
                    model, 0, 0, 0.0, 0.0, 0.0, false));
            append_records(
                result.section_tangent_audit_records,
                make_section_tangent_audit_records(
                    model,
                    reference_spec.column_height_m,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    false));
        }
    }

    const std::vector<std::size_t> base_nodes = {0};

    nl.set_step_callback([&](
                             int runtime_step,
                             double runtime_p,
                             const BeamModel& m) {
        if (control_path.is_preload_runtime_step(runtime_step) &&
            !control_path.is_preload_completion_runtime_step(runtime_step)) {
            return;
        }

        const double logical_p = control_path.lateral_progress(runtime_p);
        const double drift = target_drift_at(cfg, logical_p);
        const int logical_step = control_path.logical_lateral_step(runtime_step);

        append_runtime_observables(
            result,
            m,
            section_layout,
            reference_spec.column_height_m,
            base_nodes,
            runtime_step,
            logical_step,
            runtime_p,
            logical_p,
            drift,
            top_node,
            control_path.is_preload_completion_runtime_step(runtime_step),
            control_trace_state,
            nl.last_increment_step_diagnostics());
        if (spec.write_element_tangent_audit_csv) {
            result.element_tangent_audit_records.push_back(
                make_element_tangent_audit_record(
                    m,
                    runtime_step,
                    logical_step,
                    runtime_p,
                    logical_p,
                    drift,
                    false));
            append_records(
                result.section_tangent_audit_records,
                make_section_tangent_audit_records(
                    m,
                    reference_spec.column_height_m,
                    runtime_step,
                    logical_step,
                    runtime_p,
                    logical_p,
                    drift,
                    false));
        }

        const bool report_step =
            logical_step == 0 ||
            (logical_step % 20 == 0) ||
            (logical_step == control_path.lateral_steps);

        if (spec.print_progress && report_step) {
            const auto step_section_records =
                extract_section_response_records(
                    m,
                    reference_spec.column_height_m,
                    logical_step,
                    logical_p,
                    drift);
            const auto controlling = std::min_element(
                step_section_records.begin(),
                step_section_records.end(),
                [](const auto& a, const auto& b) {
                    if (a.xi == b.xi) {
                        return a.section_gp < b.section_gp;
                    }
                    return a.xi < b.xi;
                });
            std::println(
                "    reduced-column step={:3d}  p={:.4f}  d={:+.4e} m"
                "  V={:+.4e} MN  M_y={:+.4e} MNm  kappa_y={:+.4e}"
                "  quad={}  stage={}",
                logical_step,
                logical_p,
                drift,
                result.hysteresis_records.back().base_shear,
                controlling != step_section_records.end() ? controlling->moment_y : 0.0,
                controlling != step_section_records.end() ? controlling->curvature_y : 0.0,
                beam_axis_quadrature_family_name<QuadratureFamily>(),
                control_path.is_preload_completion_runtime_step(runtime_step)
                    ? "preload_equilibrated"
                    : "lateral_branch");
        }
    });
    nl.set_failed_attempt_callback([&](const BeamModel& m, const auto& solver_diag) {
        const double runtime_p = solver_diag.last_attempt_p_target;
        const double logical_p = control_path.lateral_progress(runtime_p);
        const int logical_step =
            result.hysteresis_records.empty()
                ? 0
                : result.hysteresis_records.back().step + 1;

        const double target_drift = target_drift_at(cfg, logical_p);
        const double actual_tip_lateral_displacement = m.prescribed_value(top_node, 0);
        const double actual_tip_lateral_total_state_displacement =
            query::nodal_dof_value(m, m.state_vector(), top_node, 0);
        const double prescribed_top_bending_rotation =
            m.prescribed_value(top_node, 4);
        const double actual_top_bending_rotation =
            query::nodal_dof_value(m, m.state_vector(), top_node, 4);
        const double top_axial_displacement =
            query::nodal_dof_value(m, m.state_vector(), top_node, 2);
        const double base_shear =
            extract_support_resultant_component(m, base_nodes, 0);
        const double base_axial_reaction =
            extract_support_resultant_component(m, base_nodes, 2);
        auto trace_preview_state = control_trace_state;
        const auto trace_snapshot = advance_control_trace(
            trace_preview_state,
            target_drift,
            actual_tip_lateral_displacement,
            logical_p > 0.0);

        failed_trial_capture.valid = true;
        failed_trial_capture.runtime_p = runtime_p;
        failed_trial_capture.logical_p = logical_p;
        failed_trial_capture.target_drift = target_drift;
        failed_trial_capture.actual_tip_lateral_displacement =
            actual_tip_lateral_displacement;
        failed_trial_capture.actual_tip_lateral_total_state_displacement =
            actual_tip_lateral_total_state_displacement;
        failed_trial_capture.prescribed_top_bending_rotation =
            prescribed_top_bending_rotation;
        failed_trial_capture.actual_top_bending_rotation =
            actual_top_bending_rotation;
        failed_trial_capture.top_axial_displacement = top_axial_displacement;
        failed_trial_capture.base_shear = base_shear;
        failed_trial_capture.base_axial_reaction = base_axial_reaction;
        failed_trial_capture.trace_snapshot = trace_snapshot;
        if (spec.write_element_tangent_audit_csv) {
            failed_trial_capture.tangent_audit =
                make_element_tangent_audit_record(
                    m,
                    result.hysteresis_records.empty()
                        ? 0
                        : result.hysteresis_records.back().step + 1,
                    logical_step,
                    runtime_p,
                    logical_p,
                    actual_tip_lateral_displacement,
                    true);
            failed_trial_capture.has_tangent_audit = true;
            failed_trial_capture.section_tangent_audit_records =
                make_section_tangent_audit_records(
                    m,
                    reference_spec.column_height_m,
                    result.hysteresis_records.empty()
                        ? 0
                        : result.hysteresis_records.back().step + 1,
                    logical_step,
                    runtime_p,
                    logical_p,
                    actual_tip_lateral_displacement,
                    true);
            failed_trial_capture.has_section_tangent_audit = true;
        }
        failed_trial_capture.section_response_records =
            extract_section_response_records(
                m,
                reference_spec.column_height_m,
                logical_step,
                logical_p,
                actual_tip_lateral_displacement);
        failed_trial_capture.fiber_history_records =
            extract_section_fiber_history_records(
                m,
                section_layout,
                reference_spec.column_height_m,
                logical_step,
                logical_p,
                actual_tip_lateral_displacement);
    });

    auto scheme = make_control(
        [top_node, &cfg, &spec, control_path](
            double runtime_p, Vec f_full, Vec f_ext, BeamModel* m) {
            const double lateral_p = control_path.lateral_progress(runtime_p);
            const double target_rotation =
                target_top_bending_rotation(spec, cfg, lateral_p);

            if (control_path.has_preload_stage() &&
                runtime_p <= control_path.preload_completion_runtime_p()) {
                VecCopy(f_full, f_ext);
                VecScale(f_ext, control_path.preload_progress(runtime_p));
                m->update_imposed_value(top_node, 0, 0.0);
                if (spec.prescribes_top_bending_rotation()) {
                    m->update_imposed_value(top_node, 4, 0.0);
                }
                return;
            }

            VecCopy(f_full, f_ext);
            m->update_imposed_value(top_node, 0, target_drift_at(cfg, lateral_p));
            if (spec.prescribes_top_bending_rotation()) {
                m->update_imposed_value(top_node, 4, target_rotation);
            }
        });

    const auto solve_outcome = solve_structural_runtime_targets(
        nl,
        scheme,
        build_runtime_targets(spec, control_path, cfg),
        cfg.max_bisections);

    result.completed_successfully = solve_outcome.success;
    if (!solve_outcome.success) {
        result.has_failed_attempt_control_state = true;
        const auto runtime_p = failed_trial_capture.valid
                                   ? failed_trial_capture.runtime_p
                                   : nl.last_increment_step_diagnostics()
                                         .last_attempt_p_target;
        const auto logical_p = failed_trial_capture.valid
                                   ? failed_trial_capture.logical_p
                                   : control_path.lateral_progress(runtime_p);
        const auto trace_snapshot = failed_trial_capture.valid
                                        ? failed_trial_capture.trace_snapshot
                                        : ReducedRCColumnControlTraceSnapshot{};
        result.failed_attempt_control_state = make_control_state_record(
            solve_outcome.failed_runtime_step,
            control_path.logical_lateral_step(solve_outcome.failed_runtime_step),
            runtime_p,
            logical_p,
            failed_trial_capture.valid
                ? failed_trial_capture.target_drift
                : target_drift_at(cfg, logical_p),
            failed_trial_capture.valid
                ? failed_trial_capture.actual_tip_lateral_displacement
                : std::numeric_limits<double>::quiet_NaN(),
            failed_trial_capture.valid
                ? failed_trial_capture.actual_tip_lateral_total_state_displacement
                : std::numeric_limits<double>::quiet_NaN(),
            failed_trial_capture.valid
                ? failed_trial_capture.prescribed_top_bending_rotation
                : std::numeric_limits<double>::quiet_NaN(),
            failed_trial_capture.valid
                ? failed_trial_capture.actual_top_bending_rotation
                : std::numeric_limits<double>::quiet_NaN(),
            failed_trial_capture.valid
                ? failed_trial_capture.top_axial_displacement
                : std::numeric_limits<double>::quiet_NaN(),
            failed_trial_capture.valid
                ? failed_trial_capture.base_shear
                : std::numeric_limits<double>::quiet_NaN(),
            failed_trial_capture.valid
                ? failed_trial_capture.base_axial_reaction
                : std::numeric_limits<double>::quiet_NaN(),
            control_path.is_preload_completion_runtime_step(
                solve_outcome.failed_runtime_step),
            trace_snapshot,
            nl.last_increment_step_diagnostics(),
            false);
        if (failed_trial_capture.has_tangent_audit) {
            result.has_failed_attempt_element_tangent_audit = true;
            result.failed_attempt_element_tangent_audit =
                failed_trial_capture.tangent_audit;
            result.failed_attempt_element_tangent_audit.runtime_step =
                solve_outcome.failed_runtime_step;
            result.failed_attempt_element_tangent_audit.step =
                control_path.logical_lateral_step(solve_outcome.failed_runtime_step);
        }
        if (failed_trial_capture.has_section_tangent_audit) {
            result.has_failed_attempt_section_tangent_audit = true;
            result.failed_attempt_section_tangent_audit_records =
                std::move(failed_trial_capture.section_tangent_audit_records);
            for (auto& row : result.failed_attempt_section_tangent_audit_records) {
                row.runtime_step = solve_outcome.failed_runtime_step;
                row.step =
                    control_path.logical_lateral_step(solve_outcome.failed_runtime_step);
            }
        }
        if (failed_trial_capture.valid) {
            result.failed_attempt_section_response_records =
                std::move(failed_trial_capture.section_response_records);
            result.failed_attempt_fiber_history_records =
                std::move(failed_trial_capture.fiber_history_records);
        }
    }
    result.timing.solve_wall_seconds = analysis_timer.stop();

    if (spec.print_progress) {
        std::println(
            "  Reduced-column baseline result: {} ({} logical records,"
            " N={}, quadrature={}, preload_stage={}, continuation={})",
            result.completed_successfully ? "COMPLETED" : "ABORTED",
            result.hysteresis_records.size(),
            N,
            beam_axis_quadrature_family_name<QuadratureFamily>(),
            control_path.has_preload_stage() ? "enabled" : "disabled",
            to_string(spec.continuation_kind));
    }

    StopWatch output_timer;
    if (spec.write_hysteresis_csv) {
        output_timer.start();
        std::filesystem::create_directories(out_dir);
        write_hysteresis_csv(
            out_dir + "/hysteresis.csv", result.hysteresis_records);
        output_timer.stop();
    }

    if (spec.write_section_response_csv) {
        output_timer.start();
        std::filesystem::create_directories(out_dir);
        write_section_response_csv(
            out_dir + "/section_response.csv",
            result.section_response_records);
        write_base_side_moment_curvature_csv(
            out_dir + "/moment_curvature_base.csv",
            result.section_response_records);
        output_timer.stop();
    }

    if (spec.write_section_fiber_history_csv) {
        output_timer.start();
        std::filesystem::create_directories(out_dir);
        write_section_fiber_history_csv(
            out_dir + "/section_fiber_state_history.csv",
            result.fiber_history_records);
        output_timer.stop();
    }

    if (spec.write_element_tangent_audit_csv) {
        output_timer.start();
        std::filesystem::create_directories(out_dir);
        write_element_tangent_audit_csv(
            out_dir + "/element_tangent_audit.csv",
            result.element_tangent_audit_records);
        write_section_tangent_audit_csv(
            out_dir + "/section_tangent_audit.csv",
            result.section_tangent_audit_records);
        if (result.has_failed_attempt_element_tangent_audit) {
            write_element_tangent_audit_csv(
                out_dir + "/failed_attempt_element_tangent_audit.csv",
                {result.failed_attempt_element_tangent_audit});
        }
        if (result.has_failed_attempt_section_tangent_audit) {
            write_section_tangent_audit_csv(
                out_dir + "/failed_attempt_section_tangent_audit.csv",
                result.failed_attempt_section_tangent_audit_records);
        }
        output_timer.stop();
    }

    result.timing.output_write_wall_seconds = output_timer.elapsed();
    result.timing.total_wall_seconds = total_timer.stop();

    return result;
}

template <std::size_t N>
[[nodiscard]] ReducedRCColumnStructuralRunResult
dispatch_small_strain_quadrature(
    BeamAxisQuadratureFamily quadrature_family,
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    switch (quadrature_family) {
        case BeamAxisQuadratureFamily::GaussLegendre:
            return run_reduced_rc_column_small_strain_beam_case_impl<
                N, BeamAxisQuadratureFamily::GaussLegendre>(spec, out_dir, cfg);
        case BeamAxisQuadratureFamily::GaussLobatto:
            return run_reduced_rc_column_small_strain_beam_case_impl<
                N, BeamAxisQuadratureFamily::GaussLobatto>(spec, out_dir, cfg);
        case BeamAxisQuadratureFamily::GaussRadauLeft:
            return run_reduced_rc_column_small_strain_beam_case_impl<
                N, BeamAxisQuadratureFamily::GaussRadauLeft>(spec, out_dir, cfg);
        case BeamAxisQuadratureFamily::GaussRadauRight:
            return run_reduced_rc_column_small_strain_beam_case_impl<
                N, BeamAxisQuadratureFamily::GaussRadauRight>(spec, out_dir, cfg);
    }

    throw std::invalid_argument("Unsupported beam-axis quadrature family.");
}

} // namespace

ReducedRCColumnStructuralRunResult
run_reduced_rc_column_small_strain_beam_case_result(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    switch (spec.beam_nodes) {
        case 2:
            return dispatch_small_strain_quadrature<2>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 3:
            return dispatch_small_strain_quadrature<3>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 4:
            return dispatch_small_strain_quadrature<4>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 5:
            return dispatch_small_strain_quadrature<5>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 6:
            return dispatch_small_strain_quadrature<6>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 7:
            return dispatch_small_strain_quadrature<7>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 8:
            return dispatch_small_strain_quadrature<8>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 9:
            return dispatch_small_strain_quadrature<9>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        case 10:
            return dispatch_small_strain_quadrature<10>(
                spec.beam_axis_quadrature_family, spec, out_dir, cfg);
        default:
            throw std::invalid_argument(
                "ReducedRCColumnStructuralRunSpec supports TimoshenkoBeamN with"
                " N in [2, 10].");
    }
}

std::vector<table_cyclic_validation::StepRecord>
run_reduced_rc_column_small_strain_beam_case(
    const ReducedRCColumnStructuralRunSpec& spec,
    const std::string& out_dir,
    const table_cyclic_validation::CyclicValidationRunConfig& cfg)
{
    return run_reduced_rc_column_small_strain_beam_case_result(
               spec,
               out_dir,
               cfg)
        .hysteresis_records;
}

} // namespace fall_n::validation_reboot
