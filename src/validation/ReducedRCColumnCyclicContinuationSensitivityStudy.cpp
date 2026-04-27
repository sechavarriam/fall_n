#include "src/validation/ReducedRCColumnCyclicContinuationSensitivityStudy.hh"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <print>
#include <string_view>
#include <stdexcept>

namespace fall_n::validation_reboot {

namespace {

[[nodiscard]] constexpr std::string_view
beam_axis_quadrature_family_key(BeamAxisQuadratureFamily family) noexcept
{
    switch (family) {
        case BeamAxisQuadratureFamily::GaussLegendre:
            return "gauss_legendre";
        case BeamAxisQuadratureFamily::GaussLobatto:
            return "gauss_lobatto";
        case BeamAxisQuadratureFamily::GaussRadauLeft:
            return "gauss_radau_left";
        case BeamAxisQuadratureFamily::GaussRadauRight:
            return "gauss_radau_right";
    }
    return "unknown_quadrature";
}

template <typename T>
[[nodiscard]] std::vector<T> sorted_copy(std::vector<T> values)
{
    std::ranges::sort(values);
    return values;
}

[[nodiscard]] bool equivalent_protocols(
    const table_cyclic_validation::CyclicValidationRunConfig& a,
    const table_cyclic_validation::CyclicValidationRunConfig& b) noexcept
{
    return a.amplitudes_m == b.amplitudes_m &&
           a.steps_per_segment == b.steps_per_segment &&
           a.max_steps == b.max_steps &&
           a.max_bisections == b.max_bisections;
}

[[nodiscard]] bool equivalent_case_filters(
    const ReducedRCColumnCyclicNodeRefinementRunSpec& a,
    const ReducedRCColumnCyclicNodeRefinementRunSpec& b)
{
    return a.include_only_phase3_runtime_baseline ==
               b.include_only_phase3_runtime_baseline &&
           sorted_copy(a.beam_nodes_filter) == sorted_copy(b.beam_nodes_filter) &&
           sorted_copy(a.quadrature_filter) == sorted_copy(b.quadrature_filter);
}

void validate_comparable_specs(
    const ReducedRCColumnCyclicContinuationSensitivityRunSpec& spec)
{
    if (!equivalent_protocols(
            spec.baseline_spec.structural_protocol,
            spec.candidate_spec.structural_protocol)) {
        throw std::invalid_argument(
            "Reduced RC cyclic continuation sensitivity study requires the "
            "baseline and candidate policies to share the same cyclic protocol.");
    }

    if (!equivalent_case_filters(spec.baseline_spec, spec.candidate_spec)) {
        throw std::invalid_argument(
            "Reduced RC cyclic continuation sensitivity study requires the "
            "baseline and candidate policies to span the same case filters.");
    }
}

[[nodiscard]] const ReducedRCColumnCyclicNodeRefinementCaseRow*
find_case_row(
    const std::vector<ReducedRCColumnCyclicNodeRefinementCaseRow>& rows,
    const std::string& case_id)
{
    const auto it = std::ranges::find_if(
        rows,
        [&case_id](const auto& row) { return row.case_id == case_id; });
    return it == rows.end() ? nullptr : &(*it);
}

template <typename MetricT>
[[nodiscard]] constexpr double delta_metric(
    MetricT baseline,
    MetricT candidate) noexcept
{
    return static_cast<double>(candidate) - static_cast<double>(baseline);
}

struct SummaryAccumulator {
    std::size_t case_count{0};
    std::size_t baseline_completed_case_count{0};
    std::size_t candidate_completed_case_count{0};
    std::size_t baseline_representative_pass_count{0};
    std::size_t candidate_representative_pass_count{0};
    std::size_t candidate_additional_representative_pass_count{0};
    std::size_t candidate_lost_representative_pass_count{0};
    std::size_t candidate_improves_terminal_return_count{0};
    std::size_t candidate_improves_moment_history_count{0};
    std::size_t candidate_improves_tangent_history_count{0};
    std::size_t candidate_improves_secant_history_count{0};
    std::size_t candidate_improves_turning_point_count{0};
    std::size_t candidate_improves_axial_force_count{0};
    std::size_t candidate_improves_station_shift_count{0};
    double max_abs_delta_rel_terminal_return_moment_drift{0.0};
    std::string max_abs_delta_terminal_return_case_id{};
    double max_abs_delta_max_rel_moment_history_drift{0.0};
    std::string max_abs_delta_moment_history_case_id{};
    double max_abs_delta_max_rel_tangent_history_drift{0.0};
    std::string max_abs_delta_tangent_history_case_id{};
    double max_abs_delta_max_rel_secant_history_drift{0.0};
    std::string max_abs_delta_secant_history_case_id{};
    double max_abs_delta_max_rel_turning_point_moment_drift{0.0};
    std::string max_abs_delta_turning_point_case_id{};
    double max_abs_delta_max_rel_axial_force_history_drift{0.0};
    std::string max_abs_delta_axial_force_case_id{};
    double max_abs_delta_abs_station_xi_shift{0.0};
    std::string max_abs_delta_station_shift_case_id{};

    void observe(
        const ReducedRCColumnCyclicContinuationSensitivityCaseRow& row)
    {
        ++case_count;
        baseline_completed_case_count += row.baseline_execution_ok ? 1u : 0u;
        candidate_completed_case_count += row.candidate_execution_ok ? 1u : 0u;
        baseline_representative_pass_count +=
            row.baseline_representative_internal_cyclic_refinement_passes ? 1u : 0u;
        candidate_representative_pass_count +=
            row.candidate_representative_internal_cyclic_refinement_passes ? 1u : 0u;
        candidate_additional_representative_pass_count +=
            (!row.baseline_representative_internal_cyclic_refinement_passes &&
             row.candidate_representative_internal_cyclic_refinement_passes)
                ? 1u
                : 0u;
        candidate_lost_representative_pass_count +=
            (row.baseline_representative_internal_cyclic_refinement_passes &&
             !row.candidate_representative_internal_cyclic_refinement_passes)
                ? 1u
                : 0u;

        const auto count_if = [](bool predicate, std::size_t& counter) {
            counter += predicate ? 1u : 0u;
        };
        count_if(
            row.candidate_improves_terminal_return_drift,
            candidate_improves_terminal_return_count);
        count_if(
            row.candidate_improves_moment_history_drift,
            candidate_improves_moment_history_count);
        count_if(
            row.candidate_improves_tangent_history_drift,
            candidate_improves_tangent_history_count);
        count_if(
            row.candidate_improves_secant_history_drift,
            candidate_improves_secant_history_count);
        count_if(
            row.candidate_improves_turning_point_drift,
            candidate_improves_turning_point_count);
        count_if(
            row.candidate_improves_axial_force_drift,
            candidate_improves_axial_force_count);
        count_if(
            row.candidate_improves_station_shift,
            candidate_improves_station_shift_count);

        const auto update_max_abs_delta =
            [&](double value, double& current, std::string& case_id) {
                const double abs_value = std::abs(value);
                if (abs_value >= current) {
                    current = abs_value;
                    case_id = row.case_id;
                }
            };

        update_max_abs_delta(
            row.delta_rel_terminal_return_moment_drift,
            max_abs_delta_rel_terminal_return_moment_drift,
            max_abs_delta_terminal_return_case_id);
        update_max_abs_delta(
            row.delta_max_rel_moment_history_drift,
            max_abs_delta_max_rel_moment_history_drift,
            max_abs_delta_moment_history_case_id);
        update_max_abs_delta(
            row.delta_max_rel_tangent_history_drift,
            max_abs_delta_max_rel_tangent_history_drift,
            max_abs_delta_tangent_history_case_id);
        update_max_abs_delta(
            row.delta_max_rel_secant_history_drift,
            max_abs_delta_max_rel_secant_history_drift,
            max_abs_delta_secant_history_case_id);
        update_max_abs_delta(
            row.delta_max_rel_turning_point_moment_drift,
            max_abs_delta_max_rel_turning_point_moment_drift,
            max_abs_delta_turning_point_case_id);
        update_max_abs_delta(
            row.delta_max_rel_axial_force_history_drift,
            max_abs_delta_max_rel_axial_force_history_drift,
            max_abs_delta_axial_force_case_id);
        update_max_abs_delta(
            row.delta_abs_station_xi_shift,
            max_abs_delta_abs_station_xi_shift,
            max_abs_delta_station_shift_case_id);
    }

    [[nodiscard]] ReducedRCColumnCyclicContinuationSensitivitySummaryRow
    make_summary_row(std::size_t beam_nodes) const
    {
        ReducedRCColumnCyclicContinuationSensitivitySummaryRow out{};
        out.beam_nodes = beam_nodes;
        out.case_count = case_count;
        out.baseline_completed_case_count = baseline_completed_case_count;
        out.candidate_completed_case_count = candidate_completed_case_count;
        out.baseline_representative_pass_count = baseline_representative_pass_count;
        out.candidate_representative_pass_count = candidate_representative_pass_count;
        out.candidate_additional_representative_pass_count =
            candidate_additional_representative_pass_count;
        out.candidate_lost_representative_pass_count =
            candidate_lost_representative_pass_count;
        out.candidate_improves_terminal_return_count =
            candidate_improves_terminal_return_count;
        out.candidate_improves_moment_history_count =
            candidate_improves_moment_history_count;
        out.candidate_improves_tangent_history_count =
            candidate_improves_tangent_history_count;
        out.candidate_improves_secant_history_count =
            candidate_improves_secant_history_count;
        out.candidate_improves_turning_point_count =
            candidate_improves_turning_point_count;
        out.candidate_improves_axial_force_count =
            candidate_improves_axial_force_count;
        out.candidate_improves_station_shift_count =
            candidate_improves_station_shift_count;
        out.max_abs_delta_rel_terminal_return_moment_drift =
            max_abs_delta_rel_terminal_return_moment_drift;
        out.max_abs_delta_max_rel_moment_history_drift =
            max_abs_delta_max_rel_moment_history_drift;
        out.max_abs_delta_max_rel_tangent_history_drift =
            max_abs_delta_max_rel_tangent_history_drift;
        out.max_abs_delta_max_rel_secant_history_drift =
            max_abs_delta_max_rel_secant_history_drift;
        out.max_abs_delta_max_rel_turning_point_moment_drift =
            max_abs_delta_max_rel_turning_point_moment_drift;
        out.max_abs_delta_max_rel_axial_force_history_drift =
            max_abs_delta_max_rel_axial_force_history_drift;
        out.max_abs_delta_abs_station_xi_shift = max_abs_delta_abs_station_xi_shift;
        return out;
    }

    [[nodiscard]] ReducedRCColumnCyclicContinuationSensitivitySummary
    make_summary(std::size_t total_case_count) const
    {
        ReducedRCColumnCyclicContinuationSensitivitySummary out{};
        out.total_case_count = total_case_count;
        out.compared_case_count = case_count;
        out.baseline_completed_case_count = baseline_completed_case_count;
        out.candidate_completed_case_count = candidate_completed_case_count;
        out.baseline_representative_pass_count = baseline_representative_pass_count;
        out.candidate_representative_pass_count = candidate_representative_pass_count;
        out.candidate_additional_representative_pass_count =
            candidate_additional_representative_pass_count;
        out.candidate_lost_representative_pass_count =
            candidate_lost_representative_pass_count;
        out.candidate_improves_terminal_return_count =
            candidate_improves_terminal_return_count;
        out.candidate_improves_moment_history_count =
            candidate_improves_moment_history_count;
        out.candidate_improves_tangent_history_count =
            candidate_improves_tangent_history_count;
        out.candidate_improves_secant_history_count =
            candidate_improves_secant_history_count;
        out.candidate_improves_turning_point_count =
            candidate_improves_turning_point_count;
        out.candidate_improves_axial_force_count =
            candidate_improves_axial_force_count;
        out.candidate_improves_station_shift_count =
            candidate_improves_station_shift_count;
        out.max_abs_delta_rel_terminal_return_moment_drift =
            max_abs_delta_rel_terminal_return_moment_drift;
        out.max_abs_delta_terminal_return_case_id =
            max_abs_delta_terminal_return_case_id;
        out.max_abs_delta_max_rel_moment_history_drift =
            max_abs_delta_max_rel_moment_history_drift;
        out.max_abs_delta_moment_history_case_id =
            max_abs_delta_moment_history_case_id;
        out.max_abs_delta_max_rel_tangent_history_drift =
            max_abs_delta_max_rel_tangent_history_drift;
        out.max_abs_delta_tangent_history_case_id =
            max_abs_delta_tangent_history_case_id;
        out.max_abs_delta_max_rel_secant_history_drift =
            max_abs_delta_max_rel_secant_history_drift;
        out.max_abs_delta_secant_history_case_id =
            max_abs_delta_secant_history_case_id;
        out.max_abs_delta_max_rel_turning_point_moment_drift =
            max_abs_delta_max_rel_turning_point_moment_drift;
        out.max_abs_delta_turning_point_case_id =
            max_abs_delta_turning_point_case_id;
        out.max_abs_delta_max_rel_axial_force_history_drift =
            max_abs_delta_max_rel_axial_force_history_drift;
        out.max_abs_delta_axial_force_case_id =
            max_abs_delta_axial_force_case_id;
        out.max_abs_delta_abs_station_xi_shift =
            max_abs_delta_abs_station_xi_shift;
        out.max_abs_delta_station_shift_case_id =
            max_abs_delta_station_shift_case_id;
        return out;
    }
};

[[nodiscard]] ReducedRCColumnCyclicContinuationSensitivityCaseRow
make_case_row(
    const ReducedRCColumnCyclicNodeRefinementCaseRow& baseline,
    const ReducedRCColumnCyclicNodeRefinementCaseRow& candidate)
{
    ReducedRCColumnCyclicContinuationSensitivityCaseRow row{};
    row.beam_nodes = baseline.beam_nodes;
    row.beam_axis_quadrature_family = baseline.beam_axis_quadrature_family;
    row.formulation_kind = baseline.formulation_kind;
    row.case_id = baseline.case_id;
    row.scope_label = baseline.scope_label;
    row.rationale_label = baseline.rationale_label;

    const auto assign_baseline =
        [&](const ReducedRCColumnCyclicNodeRefinementCaseRow& source) {
            row.baseline_continuation_kind = source.continuation_kind;
            row.baseline_continuation_segment_substep_factor =
                source.continuation_segment_substep_factor;
            row.baseline_execution_ok = source.execution_ok;
            row.baseline_representative_internal_cyclic_refinement_passes =
                source.representative_internal_cyclic_refinement_passes();
            row.baseline_history_point_count = source.history_point_count;
            row.baseline_turning_point_count = source.turning_point_count;
            row.baseline_rel_terminal_return_moment_drift =
                source.rel_terminal_return_moment_drift;
            row.baseline_max_rel_moment_history_drift =
                source.max_rel_moment_history_drift;
            row.baseline_max_rel_tangent_history_drift =
                source.max_rel_tangent_history_drift;
            row.baseline_max_rel_secant_history_drift =
                source.max_rel_secant_history_drift;
            row.baseline_max_rel_turning_point_moment_drift =
                source.max_rel_turning_point_moment_drift;
            row.baseline_max_rel_axial_force_history_drift =
                source.max_rel_axial_force_history_drift;
            row.baseline_abs_station_xi_shift = source.abs_station_xi_shift;
        };
    const auto assign_candidate =
        [&](const ReducedRCColumnCyclicNodeRefinementCaseRow& source) {
            row.candidate_continuation_kind = source.continuation_kind;
            row.candidate_continuation_segment_substep_factor =
                source.continuation_segment_substep_factor;
            row.candidate_execution_ok = source.execution_ok;
            row.candidate_representative_internal_cyclic_refinement_passes =
                source.representative_internal_cyclic_refinement_passes();
            row.candidate_history_point_count = source.history_point_count;
            row.candidate_turning_point_count = source.turning_point_count;
            row.candidate_rel_terminal_return_moment_drift =
                source.rel_terminal_return_moment_drift;
            row.candidate_max_rel_moment_history_drift =
                source.max_rel_moment_history_drift;
            row.candidate_max_rel_tangent_history_drift =
                source.max_rel_tangent_history_drift;
            row.candidate_max_rel_secant_history_drift =
                source.max_rel_secant_history_drift;
            row.candidate_max_rel_turning_point_moment_drift =
                source.max_rel_turning_point_moment_drift;
            row.candidate_max_rel_axial_force_history_drift =
                source.max_rel_axial_force_history_drift;
            row.candidate_abs_station_xi_shift = source.abs_station_xi_shift;
        };
    const auto assign_delta = [](double baseline_value,
                                 double candidate_value,
                                 double& delta,
                                 bool& improves) {
        delta = delta_metric(baseline_value, candidate_value);
        improves = delta < 0.0;
    };

    assign_baseline(baseline);
    assign_candidate(candidate);
    assign_delta(
        baseline.rel_terminal_return_moment_drift,
        candidate.rel_terminal_return_moment_drift,
        row.delta_rel_terminal_return_moment_drift,
        row.candidate_improves_terminal_return_drift);
    assign_delta(
        baseline.max_rel_moment_history_drift,
        candidate.max_rel_moment_history_drift,
        row.delta_max_rel_moment_history_drift,
        row.candidate_improves_moment_history_drift);
    assign_delta(
        baseline.max_rel_tangent_history_drift,
        candidate.max_rel_tangent_history_drift,
        row.delta_max_rel_tangent_history_drift,
        row.candidate_improves_tangent_history_drift);
    assign_delta(
        baseline.max_rel_secant_history_drift,
        candidate.max_rel_secant_history_drift,
        row.delta_max_rel_secant_history_drift,
        row.candidate_improves_secant_history_drift);
    assign_delta(
        baseline.max_rel_turning_point_moment_drift,
        candidate.max_rel_turning_point_moment_drift,
        row.delta_max_rel_turning_point_moment_drift,
        row.candidate_improves_turning_point_drift);
    assign_delta(
        baseline.max_rel_axial_force_history_drift,
        candidate.max_rel_axial_force_history_drift,
        row.delta_max_rel_axial_force_history_drift,
        row.candidate_improves_axial_force_drift);
    assign_delta(
        baseline.abs_station_xi_shift,
        candidate.abs_station_xi_shift,
        row.delta_abs_station_xi_shift,
        row.candidate_improves_station_shift);

    return row;
}

void write_case_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnCyclicContinuationSensitivityCaseRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "case_id,beam_nodes,beam_axis_quadrature_family,formulation_kind,"
           "baseline_continuation_kind,baseline_continuation_segment_substep_factor,"
           "baseline_execution_ok,baseline_representative_internal_cyclic_refinement_passes,"
           "baseline_history_point_count,baseline_turning_point_count,"
           "baseline_rel_terminal_return_moment_drift,"
           "baseline_max_rel_moment_history_drift,"
           "baseline_max_rel_tangent_history_drift,"
           "baseline_max_rel_secant_history_drift,"
           "baseline_max_rel_turning_point_moment_drift,"
           "baseline_max_rel_axial_force_history_drift,"
           "baseline_abs_station_xi_shift,"
           "candidate_continuation_kind,candidate_continuation_segment_substep_factor,"
           "candidate_execution_ok,candidate_representative_internal_cyclic_refinement_passes,"
           "candidate_history_point_count,candidate_turning_point_count,"
           "candidate_rel_terminal_return_moment_drift,"
           "candidate_max_rel_moment_history_drift,"
           "candidate_max_rel_tangent_history_drift,"
           "candidate_max_rel_secant_history_drift,"
           "candidate_max_rel_turning_point_moment_drift,"
           "candidate_max_rel_axial_force_history_drift,"
           "candidate_abs_station_xi_shift,"
           "delta_rel_terminal_return_moment_drift,"
           "delta_max_rel_moment_history_drift,"
           "delta_max_rel_tangent_history_drift,"
           "delta_max_rel_secant_history_drift,"
           "delta_max_rel_turning_point_moment_drift,"
           "delta_max_rel_axial_force_history_drift,"
           "delta_abs_station_xi_shift,"
           "candidate_improves_terminal_return_drift,"
           "candidate_improves_moment_history_drift,"
           "candidate_improves_tangent_history_drift,"
           "candidate_improves_secant_history_drift,"
           "candidate_improves_turning_point_drift,"
           "candidate_improves_axial_force_drift,"
           "candidate_improves_station_shift,scope_label,rationale_label\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.case_id << ","
            << row.beam_nodes << ","
            << beam_axis_quadrature_family_key(row.beam_axis_quadrature_family) << ","
            << continuum::to_string(row.formulation_kind) << ","
            << to_string(row.baseline_continuation_kind) << ","
            << row.baseline_continuation_segment_substep_factor << ","
            << (row.baseline_execution_ok ? 1 : 0) << ","
            << (row.baseline_representative_internal_cyclic_refinement_passes ? 1 : 0)
            << ","
            << row.baseline_history_point_count << ","
            << row.baseline_turning_point_count << ","
            << row.baseline_rel_terminal_return_moment_drift << ","
            << row.baseline_max_rel_moment_history_drift << ","
            << row.baseline_max_rel_tangent_history_drift << ","
            << row.baseline_max_rel_secant_history_drift << ","
            << row.baseline_max_rel_turning_point_moment_drift << ","
            << row.baseline_max_rel_axial_force_history_drift << ","
            << row.baseline_abs_station_xi_shift << ","
            << to_string(row.candidate_continuation_kind) << ","
            << row.candidate_continuation_segment_substep_factor << ","
            << (row.candidate_execution_ok ? 1 : 0) << ","
            << (row.candidate_representative_internal_cyclic_refinement_passes ? 1 : 0)
            << ","
            << row.candidate_history_point_count << ","
            << row.candidate_turning_point_count << ","
            << row.candidate_rel_terminal_return_moment_drift << ","
            << row.candidate_max_rel_moment_history_drift << ","
            << row.candidate_max_rel_tangent_history_drift << ","
            << row.candidate_max_rel_secant_history_drift << ","
            << row.candidate_max_rel_turning_point_moment_drift << ","
            << row.candidate_max_rel_axial_force_history_drift << ","
            << row.candidate_abs_station_xi_shift << ","
            << row.delta_rel_terminal_return_moment_drift << ","
            << row.delta_max_rel_moment_history_drift << ","
            << row.delta_max_rel_tangent_history_drift << ","
            << row.delta_max_rel_secant_history_drift << ","
            << row.delta_max_rel_turning_point_moment_drift << ","
            << row.delta_max_rel_axial_force_history_drift << ","
            << row.delta_abs_station_xi_shift << ","
            << (row.candidate_improves_terminal_return_drift ? 1 : 0) << ","
            << (row.candidate_improves_moment_history_drift ? 1 : 0) << ","
            << (row.candidate_improves_tangent_history_drift ? 1 : 0) << ","
            << (row.candidate_improves_secant_history_drift ? 1 : 0) << ","
            << (row.candidate_improves_turning_point_drift ? 1 : 0) << ","
            << (row.candidate_improves_axial_force_drift ? 1 : 0) << ","
            << (row.candidate_improves_station_shift ? 1 : 0) << ","
            << row.scope_label << ","
            << row.rationale_label << "\n";
    }

    std::println(
        "  CSV: {} ({} continuation-sensitivity cases)",
        path,
        rows.size());
}

void write_summary_rows_csv(
    const std::string& path,
    const std::vector<ReducedRCColumnCyclicContinuationSensitivitySummaryRow>& rows)
{
    std::ofstream ofs(path);
    ofs << "beam_nodes,case_count,baseline_completed_case_count,"
           "candidate_completed_case_count,baseline_representative_pass_count,"
           "candidate_representative_pass_count,"
           "candidate_additional_representative_pass_count,"
           "candidate_lost_representative_pass_count,"
           "candidate_improves_terminal_return_count,"
           "candidate_improves_moment_history_count,"
           "candidate_improves_tangent_history_count,"
           "candidate_improves_secant_history_count,"
           "candidate_improves_turning_point_count,"
           "candidate_improves_axial_force_count,"
           "candidate_improves_station_shift_count,"
           "max_abs_delta_rel_terminal_return_moment_drift,"
           "max_abs_delta_max_rel_moment_history_drift,"
           "max_abs_delta_max_rel_tangent_history_drift,"
           "max_abs_delta_max_rel_secant_history_drift,"
           "max_abs_delta_max_rel_turning_point_moment_drift,"
           "max_abs_delta_max_rel_axial_force_history_drift,"
           "max_abs_delta_abs_station_xi_shift\n";
    ofs << std::scientific << std::setprecision(8);

    for (const auto& row : rows) {
        ofs << row.beam_nodes << ","
            << row.case_count << ","
            << row.baseline_completed_case_count << ","
            << row.candidate_completed_case_count << ","
            << row.baseline_representative_pass_count << ","
            << row.candidate_representative_pass_count << ","
            << row.candidate_additional_representative_pass_count << ","
            << row.candidate_lost_representative_pass_count << ","
            << row.candidate_improves_terminal_return_count << ","
            << row.candidate_improves_moment_history_count << ","
            << row.candidate_improves_tangent_history_count << ","
            << row.candidate_improves_secant_history_count << ","
            << row.candidate_improves_turning_point_count << ","
            << row.candidate_improves_axial_force_count << ","
            << row.candidate_improves_station_shift_count << ","
            << row.max_abs_delta_rel_terminal_return_moment_drift << ","
            << row.max_abs_delta_max_rel_moment_history_drift << ","
            << row.max_abs_delta_max_rel_tangent_history_drift << ","
            << row.max_abs_delta_max_rel_secant_history_drift << ","
            << row.max_abs_delta_max_rel_turning_point_moment_drift << ","
            << row.max_abs_delta_max_rel_axial_force_history_drift << ","
            << row.max_abs_delta_abs_station_xi_shift << "\n";
    }

    std::println(
        "  CSV: {} ({} continuation-sensitivity summary rows)",
        path,
        rows.size());
}

void write_summary_csv(
    const std::string& path,
    const ReducedRCColumnCyclicContinuationSensitivitySummary& summary)
{
    std::ofstream ofs(path);
    ofs << "total_case_count,compared_case_count,baseline_completed_case_count,"
           "candidate_completed_case_count,baseline_representative_pass_count,"
           "candidate_representative_pass_count,"
           "candidate_additional_representative_pass_count,"
           "candidate_lost_representative_pass_count,"
           "candidate_improves_terminal_return_count,"
           "candidate_improves_moment_history_count,"
           "candidate_improves_tangent_history_count,"
           "candidate_improves_secant_history_count,"
           "candidate_improves_turning_point_count,"
           "candidate_improves_axial_force_count,"
           "candidate_improves_station_shift_count,"
           "max_abs_delta_rel_terminal_return_moment_drift,"
           "max_abs_delta_terminal_return_case_id,"
           "max_abs_delta_max_rel_moment_history_drift,"
           "max_abs_delta_moment_history_case_id,"
           "max_abs_delta_max_rel_tangent_history_drift,"
           "max_abs_delta_tangent_history_case_id,"
           "max_abs_delta_max_rel_secant_history_drift,"
           "max_abs_delta_secant_history_case_id,"
           "max_abs_delta_max_rel_turning_point_moment_drift,"
           "max_abs_delta_turning_point_case_id,"
           "max_abs_delta_max_rel_axial_force_history_drift,"
           "max_abs_delta_axial_force_case_id,"
           "max_abs_delta_abs_station_xi_shift,"
           "max_abs_delta_station_shift_case_id,"
           "baseline_and_candidate_all_cases_completed\n";
    ofs << std::scientific << std::setprecision(8);
    ofs << summary.total_case_count << ","
        << summary.compared_case_count << ","
        << summary.baseline_completed_case_count << ","
        << summary.candidate_completed_case_count << ","
        << summary.baseline_representative_pass_count << ","
        << summary.candidate_representative_pass_count << ","
        << summary.candidate_additional_representative_pass_count << ","
        << summary.candidate_lost_representative_pass_count << ","
        << summary.candidate_improves_terminal_return_count << ","
        << summary.candidate_improves_moment_history_count << ","
        << summary.candidate_improves_tangent_history_count << ","
        << summary.candidate_improves_secant_history_count << ","
        << summary.candidate_improves_turning_point_count << ","
        << summary.candidate_improves_axial_force_count << ","
        << summary.candidate_improves_station_shift_count << ","
        << summary.max_abs_delta_rel_terminal_return_moment_drift << ","
        << summary.max_abs_delta_terminal_return_case_id << ","
        << summary.max_abs_delta_max_rel_moment_history_drift << ","
        << summary.max_abs_delta_moment_history_case_id << ","
        << summary.max_abs_delta_max_rel_tangent_history_drift << ","
        << summary.max_abs_delta_tangent_history_case_id << ","
        << summary.max_abs_delta_max_rel_secant_history_drift << ","
        << summary.max_abs_delta_secant_history_case_id << ","
        << summary.max_abs_delta_max_rel_turning_point_moment_drift << ","
        << summary.max_abs_delta_turning_point_case_id << ","
        << summary.max_abs_delta_max_rel_axial_force_history_drift << ","
        << summary.max_abs_delta_axial_force_case_id << ","
        << summary.max_abs_delta_abs_station_xi_shift << ","
        << summary.max_abs_delta_station_shift_case_id << ","
        << (summary.baseline_and_candidate_all_cases_completed() ? 1 : 0)
        << "\n";

    std::println(
        "  CSV: {} (continuation-sensitivity summary row written)",
        path);
}

template <typename PredicateT>
[[nodiscard]] ReducedRCColumnCyclicContinuationSensitivitySummaryRow
make_summary_row(
    std::size_t beam_nodes,
    const std::vector<ReducedRCColumnCyclicContinuationSensitivityCaseRow>& rows,
    PredicateT&& predicate)
{
    SummaryAccumulator accumulator{};
    for (const auto& row : rows) {
        if (!predicate(row)) {
            continue;
        }
        accumulator.observe(row);
    }

    return accumulator.make_summary_row(beam_nodes);
}

[[nodiscard]] std::vector<ReducedRCColumnCyclicContinuationSensitivitySummaryRow>
summarize_by_beam_nodes(
    const std::vector<ReducedRCColumnCyclicContinuationSensitivityCaseRow>& rows)
{
    std::vector<ReducedRCColumnCyclicContinuationSensitivitySummaryRow> out;
    for (const auto beam_nodes : canonical_reduced_rc_column_beam_node_counts_v) {
        out.push_back(make_summary_row(
            beam_nodes,
            rows,
            [beam_nodes](const auto& row) { return row.beam_nodes == beam_nodes; }));
    }
    return out;
}

[[nodiscard]] ReducedRCColumnCyclicContinuationSensitivitySummary summarize_cases(
    const std::vector<ReducedRCColumnCyclicContinuationSensitivityCaseRow>& rows)
{
    SummaryAccumulator accumulator{};
    for (const auto& row : rows) {
        accumulator.observe(row);
    }

    return accumulator.make_summary(rows.size());
}

} // namespace

ReducedRCColumnCyclicContinuationSensitivityResult
run_reduced_rc_column_cyclic_continuation_sensitivity_study(
    const ReducedRCColumnCyclicContinuationSensitivityRunSpec& spec,
    const std::string& out_dir)
{
    validate_comparable_specs(spec);
    std::filesystem::create_directories(out_dir);

    const auto baseline_out_dir =
        (std::filesystem::path{out_dir} / "baseline").string();
    const auto candidate_out_dir =
        (std::filesystem::path{out_dir} / "candidate").string();

    ReducedRCColumnCyclicContinuationSensitivityResult result{};
    result.baseline_result = run_reduced_rc_column_cyclic_node_refinement_study(
        spec.baseline_spec,
        baseline_out_dir);
    result.candidate_result = run_reduced_rc_column_cyclic_node_refinement_study(
        spec.candidate_spec,
        candidate_out_dir);

    result.case_rows.reserve(result.baseline_result.case_rows.size());
    for (const auto& baseline_case : result.baseline_result.case_rows) {
        const auto* candidate_case =
            find_case_row(result.candidate_result.case_rows, baseline_case.case_id);
        if (!candidate_case) {
            throw std::runtime_error(
                "Reduced RC cyclic continuation sensitivity study requires "
                "matching case identifiers between baseline and candidate runs.");
        }

        result.case_rows.push_back(make_case_row(baseline_case, *candidate_case));
    }

    if (result.case_rows.size() != result.candidate_result.case_rows.size()) {
        throw std::runtime_error(
            "Reduced RC cyclic continuation sensitivity study requires the "
            "baseline and candidate runs to span the same case count.");
    }

    result.summary_rows = summarize_by_beam_nodes(result.case_rows);
    result.summary = summarize_cases(result.case_rows);

    if (spec.print_progress) {
        std::println(
            "  Reduced RC cyclic continuation sensitivity: cases={} baseline_pass={} "
            "candidate_pass={} candidate_extra_pass={} candidate_lost_pass={} "
            "improve_history={} improve_axial={}",
            result.summary.total_case_count,
            result.summary.baseline_representative_pass_count,
            result.summary.candidate_representative_pass_count,
            result.summary.candidate_additional_representative_pass_count,
            result.summary.candidate_lost_representative_pass_count,
            result.summary.candidate_improves_moment_history_count,
            result.summary.candidate_improves_axial_force_count);
    }

    if (spec.write_csv) {
        write_case_rows_csv(
            (std::filesystem::path{out_dir} /
             "cyclic_continuation_sensitivity_case_comparisons.csv")
                .string(),
            result.case_rows);
        write_summary_rows_csv(
            (std::filesystem::path{out_dir} /
             "cyclic_continuation_sensitivity_summary.csv")
                .string(),
            result.summary_rows);
        write_summary_csv(
            (std::filesystem::path{out_dir} /
             "cyclic_continuation_sensitivity_overall_summary.csv")
                .string(),
            result.summary);
    }

    return result;
}

} // namespace fall_n::validation_reboot
